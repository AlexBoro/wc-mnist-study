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
import csv
import os
from collections import Counter

# Canonical taxonomy names (single source of truth for derived label_name).
LABEL_NAMES = {
    "0": "none (0)", "1": "FIS (1)", "2": "FPFA (2)",
    "3": "FSW (3)", "4": "GSW (4)", "5": "seizure (5)",
}
LEDGER_COLS = ["run_id", "human_code", "human_label_name",
               "reason", "auto_code_when_recorded"]


def _norm_code(v):
    """Normalise a taxonomy code to a bare string digit, or None if blank."""
    v = ("" if v is None else str(v)).strip()
    if v == "":
        return None
    # tolerate "2 FPFA" / "FPFA (2)" style values -> pull the first digit
    for ch in v:
        if ch.isdigit():
            return ch
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
            ))
    return out


def seed_from_export(export_path, overrides_path, verbose=True):
    """Build/refresh a ledger from an adjudication HTML export. Keeps only the
    rows the human DISAGREED with (agreed == 'no'). Merges with any existing
    ledger (export rows win on run_id collision)."""
    existing = {r["run_id"]: r for r in read_ledger(overrides_path)}
    added = 0
    with open(export_path, newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("agreed") or "").strip().lower() != "no":
                continue
            rid = (r.get("run_id") or "").strip()
            hc = _norm_code(r.get("your_code"))
            ac = _norm_code(r.get("model_code"))
            if not rid or hc is None:
                continue
            existing[rid] = dict(
                run_id=rid,
                human_code=hc,
                human_label_name=LABEL_NAMES.get(hc, hc),
                reason=(r.get("reason") or f"human adjudication: auto={ac} -> {hc}").strip(),
                auto_code_when_recorded=ac if ac is not None else "",
            )
            added += 1
    ordered = sorted(existing.values(), key=lambda x: x["run_id"])
    tmp = overrides_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(ordered)
    os.replace(tmp, overrides_path)
    if verbose:
        print(f"seeded ledger: {len(ordered)} rows total "
              f"({added} from export) -> {overrides_path}")
    return ordered


def apply(csv_path, overrides_path, verbose=True):
    """Stamp ledger overrides onto master CSV in place (atomic). Idempotent.

    Returns a summary dict: applied / stale / missing_run counts + the flagged
    run lists. Preserves the master's exact column schema and row order.
    """
    ledger = read_ledger(overrides_path)
    if not ledger:
        if verbose:
            print(f"no overrides to apply ({overrides_path} absent or empty)")
        return dict(applied=0, stale=[], missing_run=[], total_ledger=0)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames)
        rows = list(reader)
    by_id = {r["run_id"]: r for r in rows}

    applied, stale, missing, preserved = 0, [], [], []
    for ov in ledger:
        rid = ov["run_id"]
        row = by_id.get(rid)
        if row is None:
            missing.append(rid)
            continue
        hc = ov["human_code"]
        if hc is None:
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

    # atomic write, schema + order preserved
    tmp = str(csv_path) + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, csv_path)

    if verbose:
        print(f"overrides applied: {applied}/{len(ledger)} "
              f"(stale={len(stale)}, missing_run={len(missing)}, "
              f"preserved_already_stamped={len(preserved)})")
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
                preserved=preserved, total_ledger=len(ledger))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="master_classification.csv to stamp in place")
    ap.add_argument("--overrides", default="human_overrides.csv",
                    help="ledger of human corrections")
    ap.add_argument("--seed-from-export",
                    help="build/refresh the ledger from an adjudication HTML export CSV")
    args = ap.parse_args()
    if args.seed_from_export:
        seed_from_export(args.seed_from_export, args.overrides)
    if args.csv:
        apply(args.csv, args.overrides)
