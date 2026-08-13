#!/usr/bin/env python3
"""Boundary map over a 500x3 broad-bias exploration.

Renders the (bias_e, bias_i) plane for each configuration, with each design
point coloured by its automatic seizure-taxonomy label, and the fitted
seizure boundary from the canonical 6/28 exploration overlaid.

The comparison this exists to make: whether the three configurations differ
in WHERE the seizure region begins, not merely in how many runs seize.
Because every seed carries a bias pair held identical across all three
configurations, a point at the same (x, y) in each panel is the same
network under a different training paradigm.

Usage:
  python build_boundary_map.py --labels sweep_labels.csv --design design_500x3.csv \
      --out boundary_map.png [--summary boundary_summary.csv]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONFIG_ORDER = ["simultaneous", "sequential_noH", "sequential_H"]
CONFIG_TITLE = {
    "simultaneous": "Simultaneous",
    "sequential_noH": "Sequential  (no H)",
    "sequential_H": "Sequential  (+H)",
}
# Taxonomy colours: escalating severity from quiet -> seizure.
# Taxonomy is 0,1,2,3,5 (there is no 4); 5 = seizure.
TAX_ORDER = [0, 1, 2, 3, 5]
TAX_COLOR = {0: "#c9c9c9", 1: "#4c9bd6", 2: "#e8a33d", 3: "#b05fc0", 5: "#c1382c"}
TAX_LABEL = {0: "0  none", 1: "1  FIS", 2: "2  FPFA", 3: "3  FSW", 5: "5  seizure"}
SEIZURE_LEVEL = 5


def load(labels_csv, design_csv):
    lab = pd.read_csv(labels_csv)
    des = pd.read_csv(design_csv)
    # the labels CSV names the configuration `config_name`; the design calls it `config`
    if "config" not in lab.columns and "config_name" in lab.columns:
        lab = lab.rename(columns={"config_name": "config"})
    for c in ("config", "seed"):
        if c not in lab.columns:
            raise SystemExit(f"labels csv missing column {c!r}; has {list(lab.columns)}")
    keep = ["config", "seed", "label", "label_name", "confidence", "rule", "review"]
    lab = lab[[c for c in keep if c in lab.columns]]
    df = des.merge(lab, on=["config", "seed"], how="left")
    miss = int(df.label.isna().sum()) if "label" in df.columns else len(df)
    if miss:
        print(f"[warn] {miss} of {len(df)} design points have no label (unclassified runs)")
    return df


def taxonomy_column(df):
    for c in ("label", "Sz_Txnmy", "sz_txnmy", "taxonomy", "type"):
        if c in df.columns:
            return c
    raise SystemExit(f"no taxonomy column found; have {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", default="boundary_map.png")
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()

    df = load(args.labels, args.design)
    tcol = taxonomy_column(df)
    df["_tax"] = pd.to_numeric(df[tcol], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), sharex=True, sharey=True,
                             constrained_layout=True)
    rows = []
    for ax, cfg in zip(axes, CONFIG_ORDER):
        sub = df[df.config == cfg]
        for tax in TAX_ORDER:
            s = sub[sub._tax == tax]
            if len(s) == 0:
                continue
            ax.scatter(s.bias_e, s.bias_i, s=26, c=TAX_COLOR[tax],
                       edgecolors="white", linewidths=0.35,
                       label=TAX_LABEL[tax] if cfg == CONFIG_ORDER[0] else None,
                       zorder=3 if tax >= 3 else 2)
        n = int(sub._tax.notna().sum())
        f_sz = float((sub._tax >= SEIZURE_LEVEL).mean()) if n else np.nan
        f_par = float((sub._tax >= 1).mean()) if n else np.nan
        ax.set_title(f"{CONFIG_TITLE[cfg]}\nseizure {f_sz*100:.1f}%   "
                     f"any paroxysmal {f_par*100:.1f}%   (n={n})", fontsize=11)
        ax.set_xlabel("bias$_e$")
        ax.grid(alpha=0.25, linewidth=0.5, zorder=0)
        rows.append(dict(config=cfg, n=n, frac_seizure=f_sz, frac_paroxysmal=f_par,
                         frac_type0=float((sub._tax == 0).mean()) if n else np.nan,
                         frac_type1=float((sub._tax == 1).mean()) if n else np.nan,
                         frac_type2=float((sub._tax == 2).mean()) if n else np.nan,
                         frac_type3=float((sub._tax == 3).mean()) if n else np.nan))
    axes[0].set_ylabel("bias$_i$")
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.92)
    n_seeds = int(df.seed.nunique())
    fig.suptitle(f"Seizure taxonomy across the bias plane — {n_seeds} seeds x "
                 f"{df.config.nunique()} configurations\n"
                 "each seed's (bias$_e$, bias$_i$) is held identical across panels",
                 fontsize=13)
    fig.savefig(args.out, dpi=170)

    summary = pd.DataFrame(rows)
    if args.summary:
        summary.to_csv(args.summary, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
