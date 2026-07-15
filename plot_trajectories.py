"""plot_trajectories.py — (bias_e, bias_i) trajectory panels (§15.6).

Figure-1-style (bE, bI) panels with figure-2-style epoch-by-epoch paths
overlaid, colored by epoch (or accuracy), with the §10 seizure boundaries drawn.
Reads the ``_combined`` Parquet built by ``analysis_loader``.

One panel per configuration; each run's trajectory is a path colored by epoch,
with start (o) and end (*) marked, and a single fixed reference seizure boundary
per config drawn from the §10 constants (``sequential_H`` uses the aggregate fit,
not the αH-stratified strata, so the line is independent of per-run αH).

CLI:
    python plot_trajectories.py --out-root /content/drive/MyDrive/WC_MNIST_Longitudinal
    python plot_trajectories.py --color-by val_acc --save trajectories.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_combined(out_root: str) -> pd.DataFrame:
    combined = Path(out_root) / "_combined" / "all_runs_metrics.parquet"
    if combined.exists():
        return pd.read_parquet(combined)
    # Fall back to building it on the fly
    import analysis_loader
    df, _ = analysis_loader.build_combined(out_root, verbose=False)
    return df


def _fixed_boundary(config_name: str) -> tuple[float, float]:
    """One fixed reference boundary per config (§10), independent of per-run αH.

    ``sequential_H`` uses the aggregate logistic fit rather than the αH-stratified
    strata, so each panel shows a single seizure boundary.
    """
    import wc_config as cfg
    fixed = {
        "simultaneous": cfg.BOUNDARY_SIMULTANEOUS,
        "sequential_noH": cfg.BOUNDARY_SEQUENTIAL_NOH,
        "sequential_H": cfg.BOUNDARY_SEQUENTIAL_H_AGGREGATE,
    }
    return fixed.get(config_name, cfg.get_boundary(config_name))


def plot(df: pd.DataFrame, color_by: str = "epoch", save_path: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    configs = list(dict.fromkeys(df["config_name"]))   # preserve first-seen order
    n = len(configs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False)
    axes = axes[0]

    # Shared axis limits across all panels (from the global bias range).
    be_g = df["bias_e"].to_numpy(dtype=float)
    bi_g = df["bias_i"].to_numpy(dtype=float)
    xpad = 0.05 * (np.nanmax(be_g) - np.nanmin(be_g))
    ypad = 0.05 * (np.nanmax(bi_g) - np.nanmin(bi_g))
    xlim = (np.nanmin(be_g) - xpad, np.nanmax(be_g) + xpad)
    ylim = (np.nanmin(bi_g) - ypad, np.nanmax(bi_g) + ypad)

    cmap = plt.get_cmap("viridis")
    for ax, cname in zip(axes, configs):
        sub = df[df["config_name"] == cname]
        cvals = sub[color_by].to_numpy(dtype=float)
        vmin, vmax = (np.nanmin(cvals), np.nanmax(cvals)) if len(cvals) else (0, 1)
        if vmin == vmax:
            vmax = vmin + 1.0

        be_all = sub["bias_e"].to_numpy(dtype=float)
        for run_id, run in sub.groupby("run_id"):
            run = run.sort_values("epoch")
            be = run["bias_e"].to_numpy(dtype=float)
            bi = run["bias_i"].to_numpy(dtype=float)
            c = run[color_by].to_numpy(dtype=float)

            # Colored path (figure-2 style)
            pts = np.array([be, bi]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(vmin, vmax), alpha=0.8)
            lc.set_array(c[:-1])
            lc.set_linewidth(1.5)
            ax.add_collection(lc)
            ax.plot(be[0], bi[0], "o", color="green", ms=5, zorder=3)   # start
            ax.plot(be[-1], bi[-1], "*", color="red", ms=10, zorder=3)  # end

        # One fixed reference seizure boundary for the whole config (§10):
        # bI = m*bE + c. Drawn once, spanning the config's bias_e range.
        m, c0 = _fixed_boundary(cname)
        xline = np.linspace(be_all.min() - 0.5, be_all.max() + 0.5, 50)
        ax.plot(xline, m * xline + c0, color="gray", lw=1.0, ls="--", alpha=0.6, zorder=1)

        ax.set_title(f"{cname}\n(— boundary; o start, * end)")
        ax.set_xlabel("bias_e")
        ax.set_ylabel("bias_i")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes), fraction=0.025, pad=0.02)
    cbar.set_label(f"{color_by} (per-panel normalized)")
    fig.suptitle("Bias drift trajectories toward the seizure boundary", fontsize=13)

    if save_path is None:
        save_path = "trajectories.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main() -> int:
    import wc_config as cfg
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=cfg.OUT_ROOT)
    ap.add_argument("--color-by", default="epoch",
                    help="metric column to color paths by (e.g. epoch, val_acc)")
    ap.add_argument("--save", default=None, help="output PNG (default: <out_root>/_combined/trajectories.png)")
    args = ap.parse_args()

    df = _load_combined(args.out_root)
    if df is None or df.empty:
        print(f"No metrics found under {args.out_root}")
        return 1
    save_path = args.save or str(Path(args.out_root) / "_combined" / "trajectories.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    out = plot(df, color_by=args.color_by, save_path=save_path)
    print(f"Wrote {out}  ({df['run_id'].nunique()} runs, {len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
