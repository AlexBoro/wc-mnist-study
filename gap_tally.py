#!/usr/bin/env python
"""Tally the runs the automatic classification infrastructure could not handle.

Produces the hand-scoring numbers for the results/methods section. Two
categories are counted SEPARATELY because they mean different things:

  gap          the firing rule is the catch-all -- every taxonomy arm returned
               False, so the classifier made no call at all. These are gaps in
               the taxonomy or its implementation.
  override     the classifier DID make a call and a human disagreed with it
               (a row in the override ledger). Not a gap; a correction.

"hand-scored" for reporting purposes is the union: runs whose final label came
from a human rather than from a rule.

A ledger row is reported as STALE when the auto-call recorded at adjudication
time no longer matches the current master. Stale rows are counted but flagged:
they were scored against a different classifier or corpus and should be
re-reviewed before they are applied.

Usage
-----
    python gap_tally.py --csv master_classification.csv \
        [--overrides human_overrides.csv] [--gap-rule review_marginal] \
        [--scores wcmnist_gap_scores_YYYY-MM-DD.csv] [--out gap_tally.csv]

`--scores` is an export from build_gap_adjudication_html.py; passing it adds
the resolved / unresolved split over the gap set, including the two
non-taxonomy verdicts (needs_new_type, unscorable) which have no code and are
therefore hand-scored-but-unresolved.
"""
from __future__ import annotations

import os
import sys
import csv
import json
import argparse

GAP_RULE = "review_marginal"
NON_TAXONOMY = ("needs_new_type", "unscorable")


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _norm(v):
    v = ("" if v is None else str(v)).strip()
    return v


def tally(csv_path, overrides_path=None, scores_path=None, gap_rule=GAP_RULE,
          corpus=None, allow_unstamped=False):
    """Tally the hand-scoring effort for ONE corpus.

    `corpus` is the sweep identity this master belongs to. Ledger rows stamped
    for a different corpus (or unstamped, unless allow_unstamped) are EXCLUDED
    from every count and returned under `refused_ledger_rows` -- annotating them
    is not enough, because the reported hand-scored percentage is a claim about
    this corpus only.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from apply_overrides import (corpus_verdict as _verdict,
                                 CORPUS_MATCH, CORPUS_CONFLICT)
    rows = _read(csv_path)
    total = len(rows)
    by_id = {_norm(r.get("run_id")): r for r in rows}

    gap = [r for r in rows if _norm(r.get("rule")) == gap_rule]
    stamped = [r for r in rows if _norm(r.get("rule")) == "human_override"]

    # ledger
    ledger, stale, orphan, refused = [], [], [], []
    if overrides_path and os.path.exists(overrides_path):
        for ov in _read(overrides_path):
            rid = _norm(ov.get("run_id"))
            oc = _norm(ov.get("corpus"))
            # corpus gate -- before this row can influence any count
            if not oc:
                if not allow_unstamped:
                    refused.append(dict(run_id=rid, ledger_corpus="<none>",
                                        why="unstamped"))
                    continue
            else:
                # Require a POSITIVE match: a shared fingerprint is not one
                # (it excludes tau_H's range, which is how the corpora differ).
                v = _verdict(oc, corpus)
                if v != CORPUS_MATCH and not allow_unstamped:
                    refused.append(dict(
                        run_id=rid, ledger_corpus=oc,
                        why=("conflicting corpus" if v == CORPUS_CONFLICT
                             else "provenance unknown (no code_tag match)")))
                    continue
            ledger.append(ov)
            cur = by_id.get(rid)
            if cur is None:
                orphan.append(rid)
                continue
            rec = _norm(ov.get("auto_code_when_recorded"))
            # a row already stamped no longer carries its auto call
            if _norm(cur.get("rule")) == "human_override":
                continue
            if rec and rec != _norm(cur.get("label")):
                stale.append(dict(run_id=rid, recorded_auto=rec,
                                  current_auto=_norm(cur.get("label")),
                                  current_rule=_norm(cur.get("rule"))))

    ledger_ids = {_norm(o.get("run_id")) for o in ledger}
    gap_ids = {_norm(r.get("run_id")) for r in gap}
    hand_ids = ledger_ids | gap_ids | {_norm(r.get("run_id")) for r in stamped}
    hand_ids &= set(by_id)  # ignore ledger rows for runs not in this corpus

    out = {
        "total_runs": total,
        "gap_runs": len(gap),
        "gap_pct": 100.0 * len(gap) / total if total else 0.0,
        "override_runs_in_corpus": len(ledger_ids & set(by_id)),
        "override_runs_orphaned": len(orphan),
        "already_stamped": len(stamped),
        "hand_scored_runs": len(hand_ids),
        "hand_scored_pct": 100.0 * len(hand_ids) / total if total else 0.0,
        "stale_ledger_rows": len(stale),
        "refused_ledger_rows": len(refused),
        "corpus": corpus or "",
        "gap_by_config": {},
    }
    for r in gap:
        k = _norm(r.get("config_name")) or _norm(r.get("config_dir"))
        out["gap_by_config"][k] = out["gap_by_config"].get(k, 0) + 1

    # optional: progress over the gap set from a page export
    if scores_path and os.path.exists(scores_path):
        sc = _read(scores_path)
        resolved = [s for s in sc if _norm(s.get("your_code")) != ""]
        unres = [s for s in sc if _norm(s.get("gap_verdict")) in NON_TAXONOMY]
        # Compare IDENTITIES, not counts: once scores are applied the runs leave
        # the gap set (their rule becomes human_override), so len(gap) - len(sc)
        # goes negative. Unscored = gap runs with no row in the export.
        scored_ids = {_norm(s.get("run_id")) for s in sc}
        out["gap_scored"] = len(gap_ids & scored_ids)
        out["gap_scored_rows_in_export"] = len(sc)
        out["gap_resolved_to_type"] = len(resolved)
        out["gap_unresolved"] = len(unres)
        out["gap_unscored"] = len(gap_ids - scored_ids)
        out["export_rows_already_applied"] = len(scored_ids - gap_ids)
        out["gap_verdict_counts"] = {}
        for s in sc:
            v = _norm(s.get("gap_verdict"))
            out["gap_verdict_counts"][v] = out["gap_verdict_counts"].get(v, 0) + 1

    return out, gap, stale, orphan, refused


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--overrides", default=None,
                    help="ledger of human corrections for THIS corpus. No "
                         "default: a ledger belongs to one sweep, and picking "
                         "one up implicitly would mix corpora.")
    ap.add_argument("--params", default=None,
                    help="all_runs_params.csv, used to report the corpus "
                         "identity alongside the tally")
    ap.add_argument("--corpus", default=None,
                    help="override the corpus identity (normally read from "
                         "--csv/--params)")
    ap.add_argument("--allow-unstamped", action="store_true",
                    help="count ledger rows that carry no corpus stamp "
                         "(unsafe: only after verifying provenance by hand)")
    ap.add_argument("--scores", default=None)
    ap.add_argument("--gap-rule", default=GAP_RULE)
    ap.add_argument("--out", default=None, help="write the tally as a one-row CSV")
    a = ap.parse_args()

    # Provenance is resolved BEFORE tallying: the corpus gate decides which
    # ledger rows may contribute to the reported hand-scored percentage.
    # This file lives beside apply_overrides.py, but PYTHONSAFEPATH (set in some
    # envs) keeps the script directory off sys.path, so add it explicitly rather
    # than relying on an implicit cwd import.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from apply_overrides import corpus_of
    corpus = a.corpus or corpus_of(a.csv, a.params)
    if not corpus:
        print("WARNING corpus of this master could not be identified "
              "(no fingerprint/code_tag column and no usable --params). "
              "Ledger rows cannot be provenance-checked.")

    out, gap, stale, orphan, refused = tally(
        a.csv, a.overrides, a.scores, a.gap_rule,
        corpus=corpus, allow_unstamped=a.allow_unstamped)
    print(json.dumps(out, indent=2))
    if refused:
        print(f"\nREFUSED {len(refused)} ledger row(s) -- NOT counted above "
              f"(master corpus: {corpus or '<unknown>'}):")
        for r in refused:
            print(f"  {r['run_id']}: {r['why']} (ledger corpus "
                  f"{r['ledger_corpus']})")
    if stale:
        print("\nSTALE ledger rows (scored against a different auto-call):")
        for s in stale:
            print(f"  {s['run_id']}: recorded auto={s['recorded_auto']} "
                  f"but master now says {s['current_auto']} "
                  f"via rule={s['current_rule']}")
    if orphan:
        print(f"\nledger rows for runs absent from this corpus: {orphan}")
    if a.out:
        flat = {k: (json.dumps(v) if isinstance(v, dict) else v)
                for k, v in out.items()}
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat), lineterminator="\n")
            w.writeheader()
            w.writerow(flat)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
