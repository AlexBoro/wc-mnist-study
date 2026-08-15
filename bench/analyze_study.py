#!/usr/bin/env python
"""bench/analyze_study.py -- analysis for the 3-seed x 3-config x 100-epoch study.

Consumes a harvested run tree (runs/<run>/metrics.csv, as the launch script
copies into /work/out) and answers the two questions the study was launched for:

  Q1  Does the boundary drift reproduce at 100 epochs, and does the sequential
      vs simultaneous gap survive 5x the pilot's training length?
  Q2  Is the drift explained by initial ACTIVITY or by initial DISTANCE to the
      boundary?  The pilot could not separate these: the two predictors are
      collinear (r = 0.51 over 6 runs) and with 2 seeds per config a
      within-config partial correlation is not arithmetically possible.

      CALIBRATION OF THAT HOPE (measured 2026-08-15, correcting an earlier
      overstatement): three seeds does NOT make a per-config partial correlation
      interpretable.  partial_corr returns nan at n <= 3 by construction, and at
      n = 4 a 200-draw check on pure random data gives |r| > 0.9 in ~28% of draws
      (median |r| = 0.71) -- one residual degree of freedom cannot distinguish
      signal from noise.  So the per-config entries below are reported for
      completeness only.  The analysis that IS valid here is
      ``pooled_within_config``: center predictors and drift within each config
      (removing the confounded between-config effect) and correlate residuals
      across all 9 runs.  Even that is 9 points with 6 residual df -- a
      direction, not an inference.

DESIGN NOTES THAT MATTER FOR CORRECTNESS
  * Every run has its OWN boundary.  It is per-config, and for sequential_H it is
    per-SEED because it depends on that run's drawn alpha_H.  Distances are read
    from each run's own recorded columns and any boundary line drawn in bias
    space is that run's own -- sharing one line across runs was a real error made
    once in the pilot figure and corrected.
  * Drift is measured as (signed distance at the LAST epoch present) minus (at
    epoch 0), read per run rather than assuming 100 rows: a shard cut short by
    the container ceiling leaves fewer, and this must degrade gracefully rather
    than crash or silently compare unequal spans.  The per-run epoch count is
    reported so a truncated run is visible instead of averaged in.
  * The accuracy dip found in the pilot (peak ~epoch 5, collapse at 8-12,
    recovery) is quantified per run rather than eyeballed, because at 100 epochs
    it may recur or resolve and either is informative.

Usage:
    python bench/analyze_study.py --root <harvested out dir> --out-prefix study100
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_tree(root: str) -> pd.DataFrame:
    """Load every run's metrics.csv under ``root``, tagging config and seed."""
    paths = sorted(glob.glob(os.path.join(root, "runs", "*", "metrics.csv")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(root, "*", "metrics.csv")))
    if not paths:
        raise SystemExit(f"no runs/*/metrics.csv under {root}")
    frames = []
    for p in paths:
        run = os.path.basename(os.path.dirname(p))
        d = pd.read_csv(p)
        d["run"] = run
        d["config"] = run.split("__", 1)[1] if "__" in run else "unknown"
        d["seed"] = int(run[4:7]) if run.startswith("seed") else -1
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def per_run_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, r in df.groupby("run"):
        r = r.sort_values("epoch")
        first, last = r.iloc[0], r.iloc[-1]
        # Peak-then-dip structure, measured rather than assumed to exist.
        va = r.val_acc.values
        pk = int(np.argmax(va[:min(len(va), 15)]))          # early peak
        post = va[pk:]
        dip = pk + int(np.argmin(post))
        rows.append(dict(
            run=run, config=first.config, seed=int(first.seed),
            epochs=len(r), last_epoch=int(last.epoch),
            complete=bool(len(r) >= 100),
            mean_E_0=first.mean_E_activity,
            sd_0=first.signed_dist_to_boundary,
            sd_end=last.signed_dist_to_boundary,
            drift=last.signed_dist_to_boundary - first.signed_dist_to_boundary,
            crossed=int(r.crossed_this_epoch.sum()) if "crossed_this_epoch" in r else np.nan,
            epochs_on_seizure_side=int(r.on_seizure_side.sum()) if "on_seizure_side" in r else np.nan,
            val_acc_peak=100 * float(np.max(va)),
            val_acc_end=100 * float(va[-1]),
            early_peak_epoch=int(r.epoch.iloc[pk]),
            dip_epoch=int(r.epoch.iloc[dip]),
            val_acc_at_dip=100 * float(va[dip]),
            dip_depth_pts=100 * float(va[pk] - va[dip]),
            alpha_H=first.alpha_H if "alpha_H" in r else np.nan,
            boundary_slope=first.boundary_slope if "boundary_slope" in r else np.nan,
        ))
    return pd.DataFrame(rows).sort_values(["config", "seed"]).reset_index(drop=True)


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Correlation of x and y with z linearly removed from both.  Returns nan when
    there are too few points to estimate it (n <= 3), rather than a spurious 1.0."""
    x, y, z = map(np.asarray, (x, y, z))
    n = len(x)
    if n <= 3 or np.std(z) == 0:
        return float("nan")
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def q2_table(s: pd.DataFrame) -> dict:
    """Activity-vs-distance, overall and within each config."""
    out = {"n_runs": int(len(s))}
    out["overall"] = {
        "r_meanE0_drift": float(s.mean_E_0.corr(s.drift)),
        "r_sd0_drift": float(s.sd_0.corr(s.drift)),
        "collinearity_r_meanE0_sd0": float(s.mean_E_0.corr(s.sd_0)),
        "partial_meanE0_drift_given_sd0": partial_corr(s.mean_E_0, s.drift, s.sd_0),
        "partial_sd0_drift_given_meanE0": partial_corr(s.sd_0, s.drift, s.mean_E_0),
    }
    # POOLED within-config test.  This is the analysis that is actually usable at
    # 3 seeds x 3 configs.  Per-config partial correlations are NOT: partial_corr
    # returns nan at n<=3, and at n=4 random data already yields |r|>0.9 in ~28%
    # of draws (200-draw check), so a per-config partial at this seed count would
    # be noise dressed as a result.  Instead, center each run's predictors and
    # drift WITHIN its config -- which removes the config main effect, i.e. the
    # sequential-vs-simultaneous difference that is confounded by construction --
    # and correlate the residuals across all runs.  That asks the separable
    # question: among runs sharing an input structure and a boundary, does
    # starting quieter predict drifting further, over and above starting farther?
    if len(s) >= 6:
        w = s.copy()
        for col in ("mean_E_0", "sd_0", "drift"):
            w[col + "_c"] = w[col] - w.groupby("config")[col].transform("mean")
        out["pooled_within_config"] = {
            "n": int(len(w)),
            "note": ("predictors and drift centered within config, so the "
                     "confounded between-config effect is removed; df is "
                     "n - n_configs = "
                     f"{len(w) - w.config.nunique()}"),
            "r_meanE0_drift": float(w.mean_E_0_c.corr(w.drift_c)),
            "r_sd0_drift": float(w.sd_0_c.corr(w.drift_c)),
            "collinearity_r": float(w.mean_E_0_c.corr(w.sd_0_c)),
            "partial_meanE0_drift_given_sd0": partial_corr(
                w.mean_E_0_c.values, w.drift_c.values, w.sd_0_c.values),
            "partial_sd0_drift_given_meanE0": partial_corr(
                w.sd_0_c.values, w.drift_c.values, w.mean_E_0_c.values),
        }

    out["within_config"] = {}
    for cn, g in s.groupby("config"):
        g = g.sort_values("mean_E_0")
        out["within_config"][cn] = {
            "n": int(len(g)),
            "r_meanE0_drift": float(g.mean_E_0.corr(g.drift)) if len(g) > 2 else None,
            "r_sd0_drift": float(g.sd_0.corr(g.drift)) if len(g) > 2 else None,
            "partial_meanE0_drift_given_sd0": partial_corr(g.mean_E_0, g.drift, g.sd_0),
            "quietest_drift": float(g.drift.iloc[0]),
            "loudest_drift": float(g.drift.iloc[-1]),
            "quieter_drifts_further": bool(g.drift.iloc[0] > g.drift.iloc[-1]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="harvested output dir (contains runs/)")
    ap.add_argument("--out-prefix", default="study100")
    args = ap.parse_args()

    df = load_tree(args.root)
    s = per_run_summary(df)

    print(f"loaded {df.run.nunique()} runs, {len(df)} epoch-rows")
    incomplete = s[~s.complete]
    if len(incomplete):
        print(f"[warn] {len(incomplete)} run(s) short of 100 epochs -- reported, not dropped:")
        print(incomplete[["run", "epochs", "last_epoch"]].to_string(index=False))

    cols = ["config", "seed", "epochs", "mean_E_0", "sd_0", "sd_end", "drift",
            "crossed", "val_acc_peak", "val_acc_end", "dip_epoch", "dip_depth_pts"]
    print("\n=== per-run summary ===")
    print(s[cols].round(4).to_string(index=False))

    print("\n=== drift by config (mean +- sd over seeds) ===")
    g = s.groupby("config").drift.agg(["mean", "std", "min", "max", "count"])
    print(g.round(4).to_string())

    q2 = q2_table(s)
    print("\n=== Q2: activity vs distance ===")
    print(json.dumps(q2, indent=2))

    s.to_csv(f"{args.out_prefix}_run_summary.csv", index=False)
    df.to_csv(f"{args.out_prefix}_metrics_all_runs.csv", index=False)
    Path(f"{args.out_prefix}_q2.json").write_text(json.dumps(q2, indent=2))
    print(f"\nwrote {args.out_prefix}_run_summary.csv, "
          f"{args.out_prefix}_metrics_all_runs.csv, {args.out_prefix}_q2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
