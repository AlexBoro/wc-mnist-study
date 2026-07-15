"""analysis_loader.py — Concatenate per-run outputs into ``_combined`` (§7/§9).

Walks ``out_root`` for ``seed*__*`` run directories and concatenates their
``metrics.parquet`` into ``_combined/all_runs_metrics.parquet`` and their
``figure_manifest.csv`` into ``_combined/all_figures_manifest.csv``.

This is the analysis-side loader (the driver never writes ``_combined``).  It is
read-only w.r.t. the per-run directories.

CLI:
    python analysis_loader.py --out-root /content/drive/MyDrive/WC_MNIST_Longitudinal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_all_metrics(out_root: str) -> pd.DataFrame:
    """Concatenate every run's metrics.parquet (falling back to metrics.csv)."""
    root = Path(out_root)
    frames = []
    for run_dir in sorted(root.glob("seed*__*")):
        if not run_dir.is_dir():
            continue
        pq = run_dir / "metrics.parquet"
        csv = run_dir / "metrics.csv"
        if pq.exists():
            frames.append(pd.read_parquet(pq))
        elif csv.exists():
            frames.append(pd.read_csv(csv))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_all_manifests(out_root: str) -> pd.DataFrame:
    root = Path(out_root)
    frames = []
    for run_dir in sorted(root.glob("seed*__*")):
        fm = run_dir / "figure_manifest.csv"
        if fm.exists():
            frames.append(pd.read_csv(fm))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_combined(out_root: str, verbose: bool = True):
    """Write ``_combined/all_runs_metrics.parquet`` and
    ``_combined/all_figures_manifest.csv``.  Returns (metrics_df, manifest_df)."""
    root = Path(out_root)
    combined = root / "_combined"
    combined.mkdir(parents=True, exist_ok=True)

    metrics = load_all_metrics(out_root)
    manifest = load_all_manifests(out_root)

    if not metrics.empty:
        try:
            metrics.to_parquet(combined / "all_runs_metrics.parquet", index=False)
        except Exception as e:
            print(f"  [warn] parquet write failed ({e}); writing CSV only")
        metrics.to_csv(combined / "all_runs_metrics.csv", index=False)
    if not manifest.empty:
        manifest.to_csv(combined / "all_figures_manifest.csv", index=False)

    if verbose:
        n_runs = metrics["run_id"].nunique() if not metrics.empty else 0
        print(f"_combined built: {len(metrics)} metric rows from {n_runs} runs, "
              f"{len(manifest)} figure rows -> {combined}")
    return metrics, manifest


def main() -> int:
    import wc_config as cfg
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=cfg.OUT_ROOT)
    args = ap.parse_args()
    build_combined(args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
