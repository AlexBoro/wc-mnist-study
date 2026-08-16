#!/usr/bin/env python
"""grade_epochs.py -- score every CAPTURED epoch of a training tree against the
WC-MNIST epileptiform/seizure taxonomy.

This is the training-path counterpart to classify_runs.py.  That script screens a
no-training sweep: one figure pair per RUN, no val_acc, mean_E passed as None.
Here every run has a longitudinal series of capture epochs, so the unit of
classification is the (run, epoch) pair and both mean_E and val_acc are real
per-epoch quantities read from metrics.csv.

WHY mean_E / val_acc MATTER HERE AND NOT THERE
    classify_runs.py notes that passing them as None means "the energy-collapse
    ictal corroboration arm simply does not fire, which is correct for this
    study".  In a TRAINING tree that arm can and should fire: mean_E is the
    epoch's own mean excitatory activity and val_acc its validation accuracy, so
    an epoch whose activity collapses while accuracy falls is exactly the case
    the arm exists to corroborate.  Passing them is therefore not an
    embellishment; withholding them would silently disable a rule.

WHAT IS GRADED, AND WHAT IS NOT
    The taxonomy reads the rendered heatmap_E.png + lfp.png for an epoch.  Those
    exist only at CAPTURE epochs -- epoch 0 then every topo_track_interval, i.e.
    26 of 100 epochs per run (0, 3, 7, 11, ... 99).  This script grades exactly
    those and NOTHING ELSE.  It does not interpolate, forward-fill, or substitute
    a scalar proxy for the 74 uncaptured epochs per run: a label is emitted only
    where the activity trace it was computed from actually exists.  Downstream
    plots must therefore draw uncaptured epochs as connecting line only.

OUTPUT
    epoch_grades.csv -- one row per (run, epoch): identity, every extracted
    feature, the taxonomy label + name + confidence + firing rule + clause, and
    the metrics.csv context (mean_E, val_acc, signed distance, on_seizure_side).
    Rows with label == -1 are the taxonomy's own "route to human adjudication"
    verdict and are flagged, never silently coerced to 0.

Usage:
    python grade_epochs.py --root /vol/wc_mnist_study100 --toolkit ./labeling_toolkit \
                           --csv /work/out/epoch_grades.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# With PYTHONSAFEPATH / -P the script's own directory is not on sys.path (this
# bit run_topology_pass.py and run_study.py before; see their comments).
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_metrics(run_dir: Path) -> dict:
    """epoch -> metrics.csv row, for the per-epoch context the classifier uses."""
    m = run_dir / "metrics.csv"
    if not m.exists():
        return {}
    with open(m) as f:
        return {int(r["epoch"]): r for r in csv.DictReader(f)}


def _num(row, key):
    """metrics.csv value as float, or None when absent/blank/nan."""
    v = (row or {}).get(key)
    if v in (None, "", "nan", "NaN"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="study tree with seed*__*/ run dirs")
    ap.add_argument("--toolkit", default="labeling_toolkit",
                    help="path to the labeling_toolkit package")
    ap.add_argument("--csv", required=True, help="output CSV path")
    args = ap.parse_args()

    tk = Path(args.toolkit).resolve()
    sys.path.insert(0, str(tk.parent))
    from labeling_toolkit.features import extract_features
    from labeling_toolkit.classifier import classify, LABEL_NAMES, SPIKE_AMP_FEATURE
    import labeling_toolkit.analytic_features as AF
    import wc_config as _cfg
    import wc_figures as _WF
    import wc_experiment as _X

    root = Path(args.root)
    runs = sorted(d for d in root.glob("seed*__*") if d.is_dir())
    if not runs:
        raise SystemExit(f"no seed*__* run dirs under {root}")

    rows, skipped = [], []
    for rd in runs:
        met = load_metrics(rd)
        fig_epochs = sorted(rd.glob("figures/epoch_*"))
        for ep_dir in fig_epochs:
            epoch = int(ep_dir.name.split("_")[-1])
            heat, lfp = ep_dir / "heatmap_E.png", ep_dir / "lfp.png"
            if not (heat.exists() and lfp.exists()):
                # Do not guess a label from a missing trace -- record the gap.
                skipped.append(dict(run=rd.name, epoch=epoch,
                                    reason=f"heatmap={heat.exists()} lfp={lfp.exists()}"))
                continue
            mrow = met.get(epoch, {})
            feats = extract_features(str(heat), str(lfp),
                                     mean_E=_num(mrow, "mean_E_activity"),
                                     val_acc=_num(mrow, "val_acc"))
            # ANALYTIC features from arrays.npz.  Load-bearing, not optional:
            # classifier.SPIKE_AMP_FEATURE is "amp_prominence", and _spike_amp
            # FALLS BACK to the pixel measure "spike_amp_max" when it is absent.
            # Without this merge the Type 0/1 boundary is decided by the legacy
            # pixel measure the user explicitly moved away from on 2026-08-12
            # (slow drift can accumulate into an apparent spike there), and the
            # fallback is silent -- the run looks fine and grades the wrong way.
            #
            # epoch_reconstructor (NOT param_render._reconstruct_connectivity)
            # supplies the TRAINED CSR stored in the bundle; the seed-based
            # reconstructor would rebuild the connectivity the run STARTED with,
            # which is wrong for a trained epoch.
            ab = ep_dir.parent.parent / "arrays" / ep_dir.name
            analytic_ok, analytic_err = False, ""
            if (ab / "arrays.npz").exists() and (ab / "manifest.json").exists():
                try:
                    feats.update(AF.extract(ab, _X.epoch_reconstructor(ab), _cfg, _WF))
                    analytic_ok = True
                except Exception as e:                      # recorded, never silent
                    analytic_err = f"{type(e).__name__}: {e}"
            else:
                analytic_err = "no arrays bundle"
            v = classify(feats)
            out = dict(run=rd.name, config=rd.name.split("__", 1)[1],
                       seed=int(rd.name[4:7]), epoch=epoch,
                       label=v["label"], label_name=v["label_name"],
                       confidence=v["confidence"], rule=v["rule"], clause=v["clause"],
                       review=int(v["label"] == -1),
                       # provenance of the amplitude that decided Type 0/1
                       analytic_ok=int(analytic_ok), analytic_err=analytic_err,
                       spike_amp_source=(SPIKE_AMP_FEATURE if feats.get(SPIKE_AMP_FEATURE)
                                         is not None else "spike_amp_max[FALLBACK]"),
                       mean_E_activity=_num(mrow, "mean_E_activity"),
                       val_acc=_num(mrow, "val_acc"),
                       bias_e=_num(mrow, "bias_e"), bias_i=_num(mrow, "bias_i"),
                       signed_dist_to_boundary=_num(mrow, "signed_dist_to_boundary"),
                       boundary_slope=_num(mrow, "boundary_slope"),
                       boundary_intercept=_num(mrow, "boundary_intercept"),
                       on_seizure_side=(mrow or {}).get("on_seizure_side"))
            out.update({f"feat_{k}": val for k, val in feats.items()})
            rows.append(out)
        print(f"  {rd.name:<26} graded {len([r for r in rows if r['run']==rd.name]):>3} "
              f"of {len(fig_epochs)} capture epochs", flush=True)

    if not rows:
        raise SystemExit("no epochs graded -- check that figures/epoch_*/ exist")

    cols = [c for c in rows[0] if not c.startswith("feat_")]
    featcols = sorted({k for r in rows for k in r if k.startswith("feat_")})
    outp = Path(args.csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + featcols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Distribution, and the review count -- reported, never hidden.
    from collections import Counter
    dist = Counter(r["label"] for r in rows)
    print(f"\ngraded {len(rows)} (run, epoch) pairs across {len(runs)} runs")
    for lab in sorted(dist, key=lambda x: (x == -1, x)):
        print(f"  {LABEL_NAMES[lab]:<24} {dist[lab]:>4}")
    # Amplitude provenance: a silent fallback to the pixel measure would change
    # the Type 0/1 boundary, so it is reported rather than left to inspection.
    n_fb = sum(1 for r in rows if r["spike_amp_source"] != SPIKE_AMP_FEATURE)
    print(f"\nanalytic features attached: {sum(r['analytic_ok'] for r in rows)}/{len(rows)}")
    if n_fb:
        print(f"[warn] {n_fb} row(s) fell back to the pixel amplitude measure "
              f"instead of {SPIKE_AMP_FEATURE}:")
        for r in [r for r in rows if r["spike_amp_source"] != SPIKE_AMP_FEATURE][:5]:
            print(f"    {r['run']} epoch {r['epoch']}: {r['analytic_err'] or 'feature absent'}")
    if skipped:
        print(f"\n[warn] {len(skipped)} capture epoch(s) lacked a figure pair:")
        for s in skipped[:10]:
            print("   ", s)
    json.dump(dict(n_rows=len(rows), dist={str(k): v for k, v in dist.items()},
                   skipped=skipped),
              open(str(outp).replace(".csv", "_summary.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
