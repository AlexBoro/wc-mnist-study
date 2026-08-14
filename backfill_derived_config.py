#!/usr/bin/env python
"""Backfill ``derived_config`` into manifests written before it existed.

WHY THIS IS NOT JUST "read the config and write it in"
-----------------------------------------------------
``derived_config`` records the render/measurement knobs that decide a run's
LABEL without touching its dynamics (see ``param_persist.derived_config``).
Bundles from the v3 sweep predate that block, so ``classify_runs`` reports them
as ``unknown`` and -- correctly -- refuses them, because absence of a record is
not evidence of agreement.

The values can be reconstructed, but ONLY because two facts were established
independently, and the second one is why a naive reconstruction is wrong:

1. ``LEAD_IN_S`` is 1.0 at every commit in the corpus (6ebe0b5, bc58019,
   971435f), and was introduced in fe68fed, which is ancestral to all three.
   The corpus is genuinely uniform on it, so 1.0 is not a guess.

2. ``LFP_SPAN`` is NOT recoverable from ``git_commit`` alone. The 60 pilot runs
   record 6ebe0b5, whose ``wc_config.py`` says ``LFP_SPAN = 150.0`` -- but their
   figures were RE-RENDERED after the 150 -> 240 recalibration (bc58019), so
   the stored ``lfp.png`` those runs are classified from is a 240 render. Three
   independent checks agreed: the whole corpus was re-rendered and re-classified
   at 240 and reproduced the stored classification table; the pilot bundles'
   figure mtimes postdate their array mtimes by ~30 min; and the electrode-
   ladder pitch measured directly in pixels is 13 px in both eras. Taking the
   committed value would therefore write a FALSE 150 onto exactly the runs whose
   provenance was hardest to establish.

So the reconstruction is: lead_in_s from the (uniform) config, lfp_span from the
RENDER era rather than the commit, and ``source="reconstructed"`` on every row
so no consumer can mistake an inference for an observation.

Idempotent. Refuses to overwrite a block whose ``source`` is ``"recorded"``.
"""
import argparse
import json
import sys
from pathlib import Path

# Values in force for the v3 corpus. Deliberately a literal table, not a
# ``git show`` at runtime: the whole point of fact 2 above is that the committed
# value and the rendered value DISAGREE for part of this corpus, so deriving
# lfp_span from git would silently reintroduce the error this script exists to
# avoid.
V3 = dict(lfp_span=240.0, lead_in_s=1.0,
          output_average_window=0.01, image_duration=0.25)

# Commits whose bundles were re-rendered at the recalibrated span after the
# fact. Their committed LFP_SPAN (150.0) is NOT what their figures show.
RERENDERED = {"6ebe0b5"}

EVIDENCE = (
    "lead_in_s: uniform 1.0 across all corpus commits (introduced fe68fed, "
    "ancestral to 6ebe0b5/bc58019/971435f). "
    "lfp_span: 240.0 by RENDER era, not by git_commit -- 6ebe0b5 bundles commit "
    "150.0 but were re-rendered post-bc58019 (corpus-wide re-render reproduced "
    "the stored labels; figure mtimes postdate arrays; ladder pitch 13 px in "
    "both eras)."
)


def backfill_one(run_dir, force=False, dry_run=False):
    """Returns one of: written / skipped_recorded / skipped_present / no_manifest."""
    man = Path(run_dir) / "manifest.json"
    if not man.exists():
        return "no_manifest"
    with open(man) as f:
        m = json.load(f)
    existing = m.get("derived_config")
    if isinstance(existing, dict) and existing:
        # Never clobber a block that was actually OBSERVED at render time.
        if existing.get("source") == "recorded" and not force:
            return "skipped_recorded"
        if not force:
            return "skipped_present"
    commit = str(m.get("git_commit", ""))[:7]
    dc = dict(V3)
    dc["source"] = "reconstructed"
    dc["reconstructed_from"] = commit or "unknown"
    dc["reconstruction_evidence"] = EVIDENCE
    if commit in RERENDERED:
        dc["note"] = (f"committed LFP_SPAN at {commit} was 150.0; figures were "
                      f"re-rendered at 240.0 after bc58019, and the stored "
                      f"lfp.png (what the classifier reads) is the 240 render.")
    if dry_run:
        return "written"
    m["derived_config"] = dc
    tmp = str(man) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=2)
    Path(tmp).replace(man)
    return "written"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True, help="bundle tree root")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing derived_config, INCLUDING a "
                         "'recorded' one (which is normally protected)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change, write nothing")
    args = ap.parse_args(argv)

    root = Path(args.out_root)
    if not root.exists():
        raise SystemExit(f"no such tree: {root}")
    tally = {}
    for man in sorted(root.glob("*/run_*/manifest.json")):
        r = backfill_one(man.parent, force=args.force, dry_run=args.dry_run)
        tally[r] = tally.get(r, 0) + 1
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb}: {tally.get('written', 0)}")
    for k in sorted(tally):
        if k != "written":
            print(f"  {k}: {tally[k]}")
    if not tally:
        print("  (no manifests matched */run_*/manifest.json under this root)")
    return 0


if __name__ == "__main__":
    sys.exit(main())