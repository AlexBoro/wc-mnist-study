#!/usr/bin/env python3
"""Apply human-adjudication overrides onto a master classification CSV.

The classifier (classify_runs.py) is PURE: it rebuilds every label from the
figure pair, so any hand-edit to the master is transient. This module keeps
human verdicts in their own ledger (human_overrides.csv) and stamps them onto
the master as a final, idempotent pass:  auto-labels first, human overrides on
top. Run it as many times as you like — the result is identical.

Ledger schema (human_overrides.csv), one row per adjudicated disagreement:
    run_id                    matches master_classification.csv run_id
    human_code                the verified taxonomy code (0-5)
    human_label_name          display label, e.g. "FPFA (2)"  (optional; derived if blank)
    reason                    free-text justification (goes into the clause field)
    auto_code_when_recorded   the classifier's auto call at adjudication time
                              -> used for the STALENESS check

Staleness: if the master's CURRENT auto label differs from
auto_code_when_recorded, the detector has changed its mind about this run since
you adjudicated it. The override is APPLIED but FLAGGED (confidence='human?'),
and a warning is emitted so you re-review — never silently dropped, never
silently trusted.

CLI:
    python apply_overrides.py --csv master_classification.csv \
                              --overrides human_overrides.csv
Seed a ledger from an adjudication export:
    python apply_overrides.py --seed-from-export EXPORT.csv \
                              --overrides human_overrides.csv
"""
import argparse
import re
import csv
import os
from collections import Counter

# Canonical taxonomy names (single source of truth for derived label_name).
LABEL_NAMES = {
    "0": "none (0)", "1": "FIS (1)", "2": "FPFA (2)",
    "3": "FSW (3)", "4": "GSW (4)", "5": "seizure (5)",
}
LEDGER_COLS = ["run_id", "human_code", "human_label_name",
               "reason", "auto_code_when_recorded", "corpus"]

# `run_id` is seed<NNN>__<config> -- it is NOT unique across corpora. Two
# different parameter-exploration sweeps produce the SAME 1500 run_ids over
# entirely different simulations, so a ledger adjudicated against one sweep
# will join cleanly and silently to another. The `corpus` column carries the
# sweep's identity (the params fingerprint, and optionally the code tag) and
# apply() REFUSES to stamp a row whose corpus does not match the master's.
# A blank corpus is treated as unknown-provenance and also refused unless
# --allow-unstamped is passed explicitly.
CORPUS_COL = "corpus"


def _norm_code(v):
    """Normalise a taxonomy code to a bare string integer, or None if blank.

    Preserves a NEGATIVE sign. The review sentinel is -1, and a first-digit
    scan silently turned it into "1" (= FIS), which made every override
    recorded against a review run read as permanently stale and mislabelled
    the auto-call in the audit clause. Only a minus immediately preceding the
    digits counts, so "FPFA (2)" and "run-2 FPFA" still normalise to "2".
    """
    v = ("" if v is None else str(v)).strip()
    if v == "":
        return None
    # tolerate "2 FPFA" / "FPFA (2)" / "-1" / "(-1) review" style values.
    # A minus counts as a SIGN only when it is not an intra-word hyphen, i.e.
    # what precedes it is nothing or a non-alphanumeric -- so "run-2 FPFA"
    # normalises to "2", while "-1" and "review (-1)" give "-1".
    for i, ch in enumerate(v):
        if ch.isdigit():
            neg = (i > 0 and v[i - 1] == "-"
                   and (i == 1 or not v[i - 2].isalnum()))
            return ("-" + ch) if neg else ch
    return v


def read_ledger(overrides_path):
    """Read human_overrides.csv -> list of normalised dict rows. Missing file
    is not an error (returns [])."""
    if not os.path.exists(overrides_path):
        return []
    out = []
    with open(overrides_path, newline="") as f:
        for r in csv.DictReader(f):
            rid = (r.get("run_id") or "").strip()
            if not rid:
                continue
            out.append(dict(
                run_id=rid,
                human_code=_norm_code(r.get("human_code")),
                human_label_name=(r.get("human_label_name") or "").strip(),
                reason=(r.get("reason") or "").strip(),
                auto_code_when_recorded=_norm_code(r.get("auto_code_when_recorded")),
                corpus=(r.get(CORPUS_COL) or "").strip(),
            ))
    return out


# --- corpus identity -------------------------------------------------------
#
# WHY THIS IS NOT JUST A FINGERPRINT COMPARISON.
#
# `model_fingerprint` (param_explore.py) deliberately hashes only the constants
# that define the MODEL -- E->E drive terms, both scale ranges, connectivity
# degrees, rewiring/inhibitory ranges, dt. It deliberately EXCLUDES per-run
# sampled values and their sampling RANGES.
#
# But the earlier sweep and the current one differ precisely in a sampling
# range: tau_H was capped at (0.5, 3.0) and is now 0.037-9.98. That change does
# NOT alter the fingerprint. Both corpora carry fingerprint a3dee49abc12.
#
# So the fingerprint CANNOT distinguish these two corpora, and fingerprint
# agreement is NOT evidence of a corpus match. Only `code_tag` -- an explicit
# label naming the sweep -- is authoritative. The fingerprint is retained
# because a fingerprint DISAGREEMENT is still conclusive evidence of a
# mismatch; it just cannot establish a match on its own.
CORPUS_UNKNOWN, CORPUS_MATCH, CORPUS_CONFLICT = "unknown", "match", "conflict"
_HEX12 = re.compile(r"\A[0-9a-f]{12}\Z")


def corpus_id(code_tag="", fingerprint=""):
    """Canonical corpus identity string: 'tag=<code_tag>;fp=<fingerprint>'.

    Explicit keys, not '/'-joined tokens: a bare token is ambiguous between a
    sweep label and a model hash, and the two are NOT interchangeable (see the
    note above). Either part may be absent; "" means wholly unknown.
    """
    parts = []
    if str(code_tag or "").strip():
        parts.append(f"tag={str(code_tag).strip()}")
    if str(fingerprint or "").strip():
        parts.append(f"fp={str(fingerprint).strip()}")
    return ";".join(parts)


def parse_corpus(s):
    """Parse a canonical identity string -> (code_tag, fingerprint).

    Tolerates legacy/hand-written stamps: a bare 12-char hex value is read as a
    fingerprint, any other bare value as a code_tag.
    """
    tag = fp = ""
    s = str(s or "").strip()
    if not s:
        return "", ""
    if "=" in s:
        for piece in re.split(r"[;,]", s):
            piece = piece.strip()
            if piece.startswith("tag="):
                tag = piece[4:].strip()
            elif piece.startswith("fp="):
                fp = piece[3:].strip()
        return tag, fp
    for tok in s.split("/"):
        tok = tok.strip()
        if not tok:
            continue
        if _HEX12.match(tok.lower()):
            fp = tok
        else:
            tag = tok
    return tag, fp


def corpus_verdict(a, b):
    """Compare two corpus identities -> CORPUS_MATCH / CONFLICT / UNKNOWN.

    - CONFLICT when code_tags are both present and differ, or fingerprints are
      both present and differ. Either is conclusive evidence of different sweeps.
    - MATCH only when both code_tags are present and equal. Fingerprint
      agreement alone is NOT a match: the fingerprint excludes sampling ranges
      (tau_H), so two genuinely different corpora share one. This is the whole
      reason for code_tag.
    - UNKNOWN otherwise -- including "fingerprints agree, no code_tag". Callers
      must refuse UNKNOWN by default and never treat it as a match.
    """
    ta, fa = parse_corpus(a)
    tb, fb = parse_corpus(b)
    if ta and tb and ta != tb:
        return CORPUS_CONFLICT
    if fa and fb and fa != fb:
        return CORPUS_CONFLICT
    if ta and tb and ta == tb:
        return CORPUS_MATCH
    return CORPUS_UNKNOWN


def corpus_conflict(a, b):
    """Back-compat shim: True only for a positive CONFLICT verdict.

    Prefer corpus_verdict -- this collapses CONFLICT and UNKNOWN's distinction
    and callers that gate on `not corpus_conflict(...)` will accept UNKNOWN.
    """
    return corpus_verdict(a, b) == CORPUS_CONFLICT


def corpus_of(csv_path, params_path=None):
    """Identify the corpus a labels CSV describes, as a canonical identity string.

    Prefers explicit `code_tag`/`fingerprint` columns on the CSV itself; falls
    back to a params table (all_runs_params.csv) joined on run_id. Returns ""
    when provenance cannot be established -- callers must treat "" as unknown,
    never as a match. Raises when an identity column is not constant within the
    file, which means the file already mixes corpora.
    """
    def _ident(rows, where):
        vals = {}
        for key in ("code_tag", "fingerprint"):
            v = {(r.get(key) or "").strip() for r in rows}
            v.discard("")
            if len(v) > 1:
                raise ValueError(
                    f"{key} is not constant in {where}: {sorted(v)[:4]} "
                    "-- this file mixes corpora and must not be used")
            vals[key] = v.pop() if v else ""
        return corpus_id(vals["code_tag"], vals["fingerprint"])

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    ident = _ident(rows, csv_path)
    if ident:
        return ident
    if params_path and os.path.exists(params_path):
        with open(params_path, newline="") as f:
            prows = list(csv.DictReader(f))
        ids = {(r.get("run_id") or "").strip() for r in rows}
        prows = [r for r in prows if (r.get("run_id") or "").strip() in ids]
        if prows:
            return _ident(prows, params_path)
    return ""


def seed_from_export(export_path, overrides_path, verbose=True, corpus=None):
    """Build/refresh a ledger from an adjudication HTML export. Keeps only the
    rows the human DISAGREED with (agreed == 'no'). Merges with any existing
    ledger (export rows win on run_id collision).

    `corpus` stamps each new row with the sweep identity it was adjudicated
    against; apply() refuses rows whose stamp does not match the master. An
    export carrying its own `corpus` column wins over this argument. Seeding
    with no corpus at all is allowed but WARNS -- such rows are refused by
    apply() unless --allow-unstamped is passed.
    """
    existing = {r["run_id"]: r for r in read_ledger(overrides_path)}
    added, no_stamp = 0, 0
    with open(export_path, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("agreed") or "").strip().lower() != "no":
                continue
            rid = (r.get("run_id") or "").strip()
            hc = _norm_code(r.get("your_code"))
            ac = _norm_code(r.get("model_code"))
            if not rid or hc is None:
                continue
            oc = (r.get(CORPUS_COL) or "").strip() or (corpus or "")
            if not oc:
                no_stamp += 1
            existing[rid] = dict(
                run_id=rid,
                human_code=hc,
                human_label_name=LABEL_NAMES.get(hc, hc),
                reason=(r.get("reason") or f"human adjudication: auto={ac} -> {hc}").strip(),
                auto_code_when_recorded=ac if ac is not None else "",
                corpus=oc,
            )
            added += 1
    ordered = sorted(existing.values(), key=lambda x: x["run_id"])
    tmp = overrides_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLS, extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(ordered)
    os.replace(tmp, overrides_path)
    if verbose:
        print(f"seeded ledger: {len(ordered)} rows total "
              f"({added} from export) -> {overrides_path}")
        stamps = {r.get("corpus") or "<none>" for r in ordered}
        print(f"  corpus stamps present: {sorted(stamps)}")
        if no_stamp:
            print(f"  WARNING {no_stamp} new row(s) carry NO corpus stamp and "
                  "will be REFUSED by apply(); pass --corpus (or --csv, to read "
                  "it from the master) when seeding")
    return ordered


def apply(csv_path, overrides_path, verbose=True, params_path=None,
          allow_unstamped=False):
    """Stamp ledger overrides onto master CSV in place (atomic). Idempotent.

    Returns a summary dict: applied / stale / missing_run counts + the flagged
    run lists. Preserves the master's exact column schema and row order.

    CORPUS GATE: `run_id` is not unique across sweeps, so a ledger from another
    corpus joins silently. Any ledger row whose `corpus` does not match this
    master's identity is REFUSED (counted in wrong_corpus, never stamped).
    Rows with a blank corpus are refused too unless allow_unstamped=True.
    """
    ledger = read_ledger(overrides_path)
    if not ledger:
        if verbose:
            print(f"no overrides to apply ({overrides_path} absent or empty)")
        return dict(applied=0, stale=[], missing_run=[], total_ledger=0,
                    wrong_corpus=[], unstamped=[], corpus="")

    master_corpus = corpus_of(csv_path, params_path)
    if verbose:
        print(f"master corpus: {master_corpus or '<unknown>'}")
        if not master_corpus:
            print("  WARNING master carries no fingerprint/code_tag and no "
                  "--params was given: corpus cannot be verified")

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames)
        rows = list(reader)
    by_id = {r["run_id"]: r for r in rows}

    applied, stale, missing, preserved = 0, [], [], []
    wrong_corpus, unstamped = [], []
    for ov in ledger:
        rid = ov["run_id"]
        row = by_id.get(rid)
        if row is None:
            missing.append(rid)
            continue
        hc = ov["human_code"]
        if hc is None:
            continue
        # Corpus gate -- BEFORE any comparison against this master's labels.
        #
        # Requires a POSITIVE match, so UNKNOWN is refused alongside CONFLICT.
        # This matters because fingerprint agreement is not a match: the
        # fingerprint excludes tau_H's sampling range, which is exactly how the
        # two corpora differ, so both carry a3dee49abc12. A gate that accepted
        # "not a conflict" would apply the earlier sweep's ledger to this one.
        oc = ov.get("corpus") or ""
        verdict = corpus_verdict(oc, master_corpus)
        if verdict == CORPUS_CONFLICT:
            wrong_corpus.append(dict(run_id=rid, ledger_corpus=oc,
                                     master_corpus=master_corpus,
                                     verdict=verdict))
            continue
        if verdict != CORPUS_MATCH and not allow_unstamped:
            unstamped.append(dict(run_id=rid, ledger_corpus=oc,
                                  master_corpus=master_corpus,
                                  verdict=verdict))
            continue
        # Idempotency guard: a row already stamped as human_override has had its
        # `label` field OVERWRITTEN by a prior pass, so it no longer holds the
        # auto-call — the staleness comparison below would be meaningless (it
        # would compare the human code against itself). Such a row is already
        # final; leave it exactly as-is. This makes repeated standalone applies
        # a true no-op (byte-identical). To pick up an EDITED ledger, re-run the
        # classifier with --force first: that resets every row to its fresh auto
        # call (rule='...'), releasing this guard so the row re-stamps cleanly.
        if (row.get("rule") or "") == "human_override":
            preserved.append(rid)
            continue
        current_auto = _norm_code(row.get("label"))
        recorded_auto = ov["auto_code_when_recorded"]
        is_stale = (recorded_auto is not None and recorded_auto != ""
                    and current_auto != recorded_auto)
        # stamp verified verdict onto the label fields
        row["label"] = hc
        row["label_name"] = ov["human_label_name"] or LABEL_NAMES.get(hc, hc)
        row["confidence"] = "human?" if is_stale else "human"
        row["rule"] = "human_override"
        note = ov["reason"] or "human adjudication"
        row["clause"] = (
            f"HUMAN OVERRIDE of auto-call {recorded_auto or '?'}"
            f"{' (STALE: auto is now ' + str(current_auto) + ' — re-review)' if is_stale else ''}"
            f". {note}"
        )
        if "review" in row:
            row["review"] = "1" if is_stale else "0"
        applied += 1
        if is_stale:
            stale.append((rid, recorded_auto, current_auto))

    # Nothing was stamped -> do not touch the file at all. A refused ledger must
    # leave the master byte-identical (and its mtime untouched), so that "the
    # master was not modified" is verifiable by cmp, not just by reading a count.
    if applied == 0:
        if verbose:
            print(f"nothing applied -> {csv_path} left untouched")
            print(f"overrides applied: 0/{len(ledger)} "
                  f"(stale=0, missing_run={len(missing)}, "
                  f"preserved_already_stamped={len(preserved)}, "
                  f"wrong_corpus={len(wrong_corpus)}, unstamped={len(unstamped)})")
            for w in wrong_corpus:
                print(f"  REFUSED {w['run_id']}: ledger corpus "
                      f"{w['ledger_corpus'] or '<none>'} CONFLICTS with master "
                      f"corpus {w['master_corpus'] or '<none>'} -> NOT stamped")
            if unstamped:
                print(f"  REFUSED {len(unstamped)} row(s): provenance UNKNOWN "
                      f"(no code_tag match could be established)")
                for u in unstamped[:6]:
                    print(f"    {u['run_id']}: ledger {u['ledger_corpus'] or '<none>'} "
                          f"vs master {u['master_corpus'] or '<none>'}")
                if len(unstamped) > 6:
                    print(f"    ... and {len(unstamped)-6} more")
                print("    -> a matching FINGERPRINT is not sufficient: it excludes "
                      "sampling ranges (tau_H), so different corpora share one. "
                      "A code_tag on both sides is required.")
                print("    -> pass --allow-unstamped only if you have verified "
                      "by hand that this ledger belongs to this corpus")
        return dict(applied=0, stale=[], missing_run=missing,
                    preserved=preserved, total_ledger=len(ledger),
                    wrong_corpus=wrong_corpus, unstamped=unstamped,
                    corpus=master_corpus)

    # atomic write, schema + order preserved
    tmp = str(csv_path) + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, csv_path)

    if verbose:
        print(f"overrides applied: {applied}/{len(ledger)} "
              f"(stale={len(stale)}, missing_run={len(missing)}, "
              f"preserved_already_stamped={len(preserved)}, "
              f"wrong_corpus={len(wrong_corpus)}, unstamped={len(unstamped)})")
        for w in wrong_corpus:
            print(f"  REFUSED {w['run_id']}: ledger corpus {w['ledger_corpus']} "
                  f"!= master corpus {w['master_corpus']} -> NOT stamped")
        if unstamped:
            print(f"  REFUSED {len(unstamped)} row(s) with no corpus stamp "
                  f"(unknown provenance): {unstamped[:6]}"
                  f"{' ...' if len(unstamped) > 6 else ''}")
            print("    -> pass --allow-unstamped only if you have verified by "
                  "hand that this ledger belongs to this corpus")
        for rid, rec, cur in stale:
            print(f"  STALE  {rid}: recorded auto={rec}, current auto={cur} "
                  f"-> applied + flagged confidence='human?', review=1")
        for rid in missing:
            print(f"  MISSING {rid}: in ledger but not in master CSV -> skipped")
        for rid in preserved:
            print(f"  PRESERVED {rid}: already human_override -> left as-is "
                  f"(re-classify with --force to re-stamp from an edited ledger)")
        hist = Counter(r["label"] for r in rows)
        print("post-override histogram:", dict(sorted(hist.items())))
    return dict(applied=applied, stale=stale, missing_run=missing,
                preserved=preserved, total_ledger=len(ledger),
                wrong_corpus=wrong_corpus, unstamped=unstamped,
                corpus=master_corpus)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="master_classification.csv to stamp in place")
    ap.add_argument("--overrides", required=True,
                    help="ledger of human corrections (REQUIRED -- there is no "
                         "default, because a ledger belongs to exactly one "
                         "corpus and picking one up implicitly is a data leak)")
    ap.add_argument("--params", default=None,
                    help="all_runs_params.csv, used to identify the master's "
                         "corpus when the labels CSV carries no fingerprint")
    ap.add_argument("--corpus", default=None,
                    help="corpus stamp for rows seeded from an export; "
                         "defaults to the identity read off --csv/--params")
    ap.add_argument("--allow-unstamped", action="store_true",
                    help="apply ledger rows that carry no corpus stamp "
                         "(unsafe: only after verifying provenance by hand)")
    ap.add_argument("--seed-from-export",
                    help="build/refresh the ledger from an adjudication HTML export CSV")
    args = ap.parse_args()
    if args.seed_from_export:
        stamp = args.corpus
        if stamp is None and args.csv:
            stamp = corpus_of(args.csv, args.params) or None
        seed_from_export(args.seed_from_export, args.overrides, corpus=stamp)
    if args.csv:
        apply(args.csv, args.overrides, params_path=args.params,
              allow_unstamped=args.allow_unstamped)
