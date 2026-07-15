"""repro_test.py — Acceptance Test A1: determinism & cross-config init identity.

Two checks (spec §5 / §16 A1):

  A1a  Determinism — running the same (seed, config) twice produces *bit-exact*
       train/val loss and accuracy curves on CPU.

  A1b  Identical init across a set — the three configs of one set
       (simultaneous / sequential_noH / sequential_H) share identical epoch-0
       model parameters, because all initialization is drawn from the
       set-seed-derived ``init_rng`` and never from config knobs.

On CPU the comparison is exact.  On GPU, sparse/scatter kernels may be
nondeterministic; pass ``--device cuda --tol 1e-5`` to allow a documented
tolerance (the run logs ``device`` and ``deterministic_ok`` either way).

Usage:
    python repro_test.py                 # CPU, bit-exact
    python repro_test.py --device cuda --tol 1e-5
Exit code 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

import wc_config as cfg
import wc_experiment as E

# Columns whose reproducibility A1 certifies (§16).
CURVE_COLUMNS = ["train_loss", "train_acc", "val_loss", "val_acc"]


def _tiny(config: cfg.ExperimentConfig, n_epochs: int) -> cfg.ExperimentConfig:
    """A fast variant of a config for the determinism test.

    Determinism does not depend on the *amount* of dynamics, so we shrink
    ``image_duration`` (fewer Euler steps/image — the per-step sparse-CSR rebuild
    is the CPU bottleneck) and the image counts.  The full study uses the frozen
    defaults from ``wc_config``; only this off-Colab acceptance check is reduced.
    """
    return dataclasses.replace(config, num_epochs=n_epochs, images_per_epoch=8,
                               eval_images=4, batch_size=8, grad_probe_images=2,
                               image_duration=0.02, output_average_window=0.01)


def check_determinism(seed: int, config_name: str, n_epochs: int, device: str,
                      tol: float) -> list[str]:
    problems: list[str] = []
    config = _tiny(cfg.make_config(config_name), n_epochs)

    dfs = []
    for run_idx in range(2):
        out_root = tempfile.mkdtemp(prefix=f"wc_repro_{run_idx}_")
        res = E.run_experiment(seed, config, n_epochs=n_epochs, resume=False,
                               out_root=out_root, device=device,
                               measure_topology_flag=False, save_figures=False,
                               verbose=False)
        dfs.append(pd.read_parquet(res.metrics_path))

    d0, d1 = dfs
    for col in CURVE_COLUMNS:
        a = d0[col].to_numpy(dtype=np.float64)
        b = d1[col].to_numpy(dtype=np.float64)
        if len(a) != len(b):
            problems.append(f"{col}: length mismatch {len(a)} vs {len(b)}")
            continue
        if tol == 0.0:
            if not np.array_equal(a, b):
                idx = int(np.argmax(a != b))
                problems.append(f"{col}: not bit-exact (first diff at epoch {idx}: "
                                f"{a[idx]!r} vs {b[idx]!r})")
        else:
            if not np.allclose(a, b, rtol=0.0, atol=tol):
                idx = int(np.argmax(np.abs(a - b)))
                problems.append(f"{col}: exceeds tol={tol} (max|Δ|={np.abs(a-b).max():.3e} "
                                f"at epoch {idx})")
    if not problems:
        mode = "bit-exact" if tol == 0.0 else f"within tol={tol}"
        print(f"  [OK]  A1a determinism: {CURVE_COLUMNS} {mode} across two runs "
              f"(seed={seed}, {config_name}, {n_epochs} epochs, {device})")
    return problems


def _init_param_snapshot(seed: int, config: cfg.ExperimentConfig, device: str) -> dict:
    """Build the model for (seed, config) and snapshot epoch-0 parameters."""
    E.seed_everything(seed)
    init_rng, config_rng = E.make_rngs(seed)
    alpha_H, tau_H = E.resolve_alpha_tau(config, config_rng)
    model = E.build_model(seed, config, torch.device(device), init_rng, alpha_H, tau_H)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, alpha_H, tau_H


def check_init_identity(seed: int, device: str) -> list[str]:
    problems: list[str] = []
    configs = cfg.make_configs()
    snaps = {}
    alphas = {}
    for c in configs:
        snap, alpha_H, tau_H = _init_param_snapshot(seed, c, device)
        snaps[c.name] = snap
        alphas[c.name] = alpha_H

    ref_name = configs[0].name
    ref = snaps[ref_name]
    for c in configs[1:]:
        s = snaps[c.name]
        if set(s.keys()) != set(ref.keys()):
            problems.append(f"{c.name}: parameter set differs from {ref_name}")
            continue
        for k in ref:
            if not torch.equal(ref[k], s[k]):
                md = float((ref[k].float() - s[k].float()).abs().max())
                problems.append(f"{c.name}: param {k!r} differs from {ref_name} (max|Δ|={md:.3e})")
    if not problems:
        print(f"  [OK]  A1b init identity: all 3 configs share identical epoch-0 "
              f"parameters (seed={seed})")
        print(f"        (alpha_H: simultaneous={alphas['simultaneous']:.3f}, "
              f"sequential_noH={alphas['sequential_noH']:.3f}, "
              f"sequential_H={alphas['sequential_H']:.3f} — config knobs differ as expected)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="sequential_noH",
                    help="config for the determinism check")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="0.0 = bit-exact (CPU); set e.g. 1e-5 for GPU tolerance")
    args = ap.parse_args()

    print(f"A1 reproducibility test (device={args.device}, tol={args.tol})\n")
    problems: list[str] = []
    print("[A1a] Determinism within a (seed, config):")
    problems += check_determinism(args.seed, args.config, args.epochs, args.device, args.tol)
    print("\n[A1b] Identical initialization across the three configs of a set:")
    problems += check_init_identity(args.seed, args.device)

    print("\n" + "=" * 60)
    if problems:
        print("A1 FAILED:")
        for p in problems:
            print("  - " + p)
        return 1
    print("A1 PASSED — runs are reproducible and a set's configs share one init.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
