"""smoke_test.py — Acceptance Tests A3 (smoke) and A4 (resume).

A3: a 2-epoch x 3-config run at seed 0 completes and produces the full §7 Parquet
    schema, the per-epoch CSV mirror, epoch-0 figures + manifest (§12), a
    checkpoint, and the DONE sentinel.

A4: interrupt-and-resume reproduces an uninterrupted run bit-for-bit — a run
    taken to 1 epoch then resumed to 2 epochs yields the same loss/acc/bias
    curves as a single uninterrupted 2-epoch run.

Topology measurement is disabled here (it needs gudhi/pyhessian and is timed
separately by ``timing_calibration.py``); the trigger logic, figures, schema,
checkpoint and resume are all exercised.  Uses a shortened ``image_duration`` so
it runs off-Colab in well under a minute.

Usage:  python smoke_test.py
Exit code 0 on success, 1 on failure.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

import wc_config as cfg
import wc_experiment as E
from wc_figures import FIG_TYPES


def _tiny(config: cfg.ExperimentConfig) -> cfg.ExperimentConfig:
    return dataclasses.replace(config, images_per_epoch=8, eval_images=4, batch_size=8,
                               grad_probe_images=2, image_duration=0.02,
                               output_average_window=0.01)


def check_smoke(device: str) -> list[str]:
    problems: list[str] = []
    out_root = tempfile.mkdtemp(prefix="wc_smoke_")
    for config in cfg.make_configs():
        tcfg = _tiny(config)
        res = E.run_experiment(0, tcfg, n_epochs=2, resume=True, out_root=out_root,
                               device=device, measure_topology_flag=False,
                               save_figures=True, verbose=False)
        run_dir = Path(res.out_dir)

        # metrics.parquet present, 2 rows, full schema
        mp = run_dir / "metrics.parquet"
        if not mp.exists():
            problems.append(f"{config.name}: metrics.parquet missing")
            continue
        df = pd.read_parquet(mp)
        if len(df) != 2:
            problems.append(f"{config.name}: expected 2 rows, got {len(df)}")
        if list(df.columns) != E.METRIC_COLUMNS:
            missing = set(E.METRIC_COLUMNS) - set(df.columns)
            extra = set(df.columns) - set(E.METRIC_COLUMNS)
            problems.append(f"{config.name}: schema mismatch (missing={missing}, extra={extra})")

        # CSV mirror
        if not (run_dir / "metrics.csv").exists():
            problems.append(f"{config.name}: metrics.csv mirror missing")

        # epoch-0 figures + manifest (epoch0 is a measurement epoch)
        fig_dir = run_dir / "figures" / "epoch_0000"
        for ft in FIG_TYPES:
            if not (fig_dir / f"{ft}.png").exists():
                problems.append(f"{config.name}: figure {ft}.png missing")
        fm = run_dir / "figure_manifest.csv"
        if not fm.exists():
            problems.append(f"{config.name}: figure_manifest.csv missing")
        else:
            man = pd.read_csv(fm)
            if list(man.columns) != E.FIGURE_MANIFEST_COLUMNS:
                problems.append(f"{config.name}: manifest schema mismatch")
            if len(man[man["epoch"] == 0]) != len(FIG_TYPES):
                problems.append(f"{config.name}: expected {len(FIG_TYPES)} epoch-0 figures, "
                                f"got {len(man[man['epoch'] == 0])}")

        # checkpoint + DONE
        ckpts = list((run_dir / "checkpoints").glob("checkpoint_epoch_*.pkl"))
        if not ckpts:
            problems.append(f"{config.name}: no checkpoint written")
        if len(ckpts) > 2:
            problems.append(f"{config.name}: keep_last=2 violated ({len(ckpts)} checkpoints)")
        if not (run_dir / "DONE").exists():
            problems.append(f"{config.name}: DONE sentinel missing")

    if not problems:
        print(f"  [OK]  A3 smoke: 3 configs x 2 epochs — schema, figures, manifest, "
              f"checkpoint, DONE all present ({device})")
    return problems


def check_resume(device: str) -> list[str]:
    """A4: a run interrupted at 1 epoch then resumed to 2 epochs matches an
    uninterrupted 2-epoch run bit-for-bit."""
    problems: list[str] = []
    tcfg = _tiny(cfg.make_config("sequential_noH"))
    curve_cols = ["train_loss", "train_acc", "val_loss", "val_acc", "bias_e", "bias_i"]

    # Interrupted: 1 epoch, then resume to 2.
    dir_a = tempfile.mkdtemp(prefix="wc_resume_a_")
    E.run_experiment(0, tcfg, n_epochs=1, resume=True, out_root=dir_a, device=device,
                     measure_topology_flag=False, save_figures=False, verbose=False)
    res_a = E.run_experiment(0, tcfg, n_epochs=2, resume=True, out_root=dir_a, device=device,
                             measure_topology_flag=False, save_figures=False, verbose=False)
    df_a = pd.read_parquet(res_a.metrics_path)

    # Reference: uninterrupted 2 epochs.
    dir_b = tempfile.mkdtemp(prefix="wc_resume_b_")
    res_b = E.run_experiment(0, tcfg, n_epochs=2, resume=False, out_root=dir_b, device=device,
                             measure_topology_flag=False, save_figures=False, verbose=False)
    df_b = pd.read_parquet(res_b.metrics_path)

    if len(df_a) != 2 or len(df_b) != 2:
        problems.append(f"resume: row counts {len(df_a)}/{len(df_b)} (expected 2/2)")
    for col in curve_cols:
        a = df_a[col].to_numpy(dtype=np.float64)
        b = df_b[col].to_numpy(dtype=np.float64)
        if len(a) == len(b) and not np.array_equal(a, b):
            idx = int(np.argmax(a != b))
            problems.append(f"resume: {col} differs from uninterrupted at epoch {idx} "
                            f"({a[idx]!r} vs {b[idx]!r})")
    if not problems:
        print(f"  [OK]  A4 resume: interrupted(1)->resume(2) matches uninterrupted(2) "
              f"bit-for-bit on {curve_cols}")
    return problems


def main() -> int:
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    print(f"A3/A4 smoke + resume test (device={device})\n")
    problems: list[str] = []
    print("[A3] Smoke — schema / files / figures / manifest:")
    problems += check_smoke(device)
    print("\n[A4] Resume — interrupted run matches uninterrupted:")
    problems += check_resume(device)

    print("\n" + "=" * 60)
    if problems:
        print("SMOKE/RESUME FAILED:")
        for p in problems:
            print("  - " + p)
        return 1
    print("A3 + A4 PASSED — harness produces the full schema and resumes identically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
