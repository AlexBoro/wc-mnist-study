"""regenerate_figures.py — Re-render per-epoch figures for completed runs.

The raw E/I activity windows the figures are drawn from are **not persisted**
(only the PNGs, scalar metrics, and checkpoints are).  So for each run this
reloads the archived measurement-epoch checkpoints (``checkpoints/topology/``),
rebuilds the frozen model exactly as ``run_experiment`` does on resume, restores
the carry-over network state, runs a short **deterministic frozen simulation**
driven by the fixed eval subset to synthesize an E/I window, and re-renders the
figures with the current ``wc_figures`` code (fixed-scale heatmaps + LFP proxy;
no dot raster).  The run's ``figure_manifest.csv`` is rewritten to match.

Because the window is a fresh frozen simulation (not the original *training*
window, which was never saved), the figures are NOT bit-identical to the live
run — they are a reproducible frozen readout of each epoch's model, matching the
notebook's original post-training eval-trajectory style.  Metrics are untouched.

After running, rebuild the combined manifest:
    python analysis_loader.py --out-root DIR

CLI:
    python regenerate_figures.py --out-root DIR
    python regenerate_figures.py --out-root DIR --runs seed000__sequential_H
    python regenerate_figures.py --out-root DIR --device cuda --window-seconds 10
"""

from __future__ import annotations

import argparse
import csv
import functools
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

import wc_config as cfg
import wc_figures as figs
import wc_model as M
from wc_experiment import (seed_everything, make_rngs, resolve_alpha_tau,
                           build_model, build_data)

MANIFEST_COLS = ("run_id", "seed", "config_name", "epoch", "fig_type", "path")


def _load_ckpt(path, device):
    """Unpickle a checkpoint, mapping tensors to ``device``.  Checkpoints saved on
    a CUDA host must be remapped to load on a CPU-only machine; the nested
    ``torch.load`` inside the pickle doesn't get map_location on its own, so patch
    the default for the duration of the load."""
    orig = torch.load
    torch.load = functools.partial(orig, map_location=device)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    finally:
        torch.load = orig


def _model_csr(crow, col, vals, shape):
    """Current (trained) excitatory connectivity as a SciPy CSR for the LFP proxy."""
    return sp.csr_matrix(
        (vals.detach().cpu().numpy(),
         col.detach().cpu().numpy(),
         crow.detach().cpu().numpy()),
        shape=tuple(shape))


@torch.no_grad()
def _simulate_window(model, images, net_state, config, n_images):
    """Frozen forward simulation from the carry-over state, capturing the E/I window.

    Mirrors the per-step dynamics of ``train_one_image`` (``model.forward`` each
    step) but with no loss/backward/optimizer.  ``images`` is a list of flat MNIST
    tensors on-device; they are cycled to reach ``n_images``.  Returns
    ``(E_window, I_window, t_window)`` as NumPy arrays [T, N] / [T].
    """
    dev = model.device
    E = net_state["E"].to(dev).clone()
    I = net_state["I"].to(dev).clone()
    O = net_state["O"].to(dev).clone()
    H = net_state["H"].to(dev).clone()
    t = float(net_state.get("global_train_time_s", 0.0))

    dt = config.dt
    num_steps = int(config.image_duration / dt)
    scale = config.mnist_input_scale
    seq = config.sequential_row_input
    if seq:
        rd = config.row_duration if config.row_duration is not None else config.image_duration / 28

    E_buf, I_buf, t_buf = [], [], []
    for k in range(n_images):
        img = images[k % len(images)]
        if not seq:
            I_ext = model.compute_mnist_input(img * scale)
        for step in range(num_steps):
            if seq:
                cur = M.active_row_at(step, dt, config.image_duration, rd)
                I_ext = model.compute_mnist_input(img * scale, active_row=cur)
            E, I, O, H = model.forward(E, I, O, I_ext, dt, H=H)
            E_buf.append(E.detach())
            I_buf.append(I.detach())
            t += dt
            t_buf.append(t)

    E_window = torch.stack(E_buf).cpu().numpy()
    I_window = torch.stack(I_buf).cpu().numpy()
    return E_window, I_window, np.asarray(t_buf, dtype=float)


def _figure_epochs(run_dir: Path) -> list[int]:
    """Epochs that had figures originally (from figure_manifest.csv), else fall back
    to every archived topology-checkpoint epoch."""
    man = run_dir / "figure_manifest.csv"
    if man.exists():
        epochs = set()
        with open(man, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    epochs.add(int(row["epoch"]))
                except (KeyError, ValueError):
                    pass
        if epochs:
            return sorted(epochs)
    topo = run_dir / "checkpoints" / "topology"
    return sorted(int(p.stem.split("_")[-1]) for p in topo.glob("topo_checkpoint_epoch_*.pkl"))


def regenerate_figures_for_run(run_dir, device=None, data_root: str | None = None,
                               window_seconds: float = 10.0, verbose: bool = True) -> dict:
    """Re-render figures for one run from its archived measurement-epoch checkpoints."""
    run_dir = Path(run_dir)
    topo_dir = run_dir / "checkpoints" / "topology"
    ckpts = {int(p.stem.split("_")[-1]): p
             for p in topo_dir.glob("topo_checkpoint_epoch_*.pkl")} if topo_dir.exists() else {}
    if not ckpts:
        if verbose:
            print(f"[{run_dir.name}] no archived checkpoints — skipping")
        return {"run_dir": str(run_dir), "regenerated_epochs": [], "n_regenerated": 0}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Identity + deterministic rebuild (same path as run_experiment / topology pass).
    head = _load_ckpt(next(iter(ckpts.values())), device)
    seed = int(head["seed"])
    config = cfg.make_config(head["config_name"])

    seed_everything(seed)
    init_rng, config_rng = make_rngs(seed)
    alpha_H, tau_H = resolve_alpha_tau(config, config_rng)
    model = build_model(seed, config, device, init_rng, alpha_H, tau_H)
    model.eval()
    _, _, _, fixed_eval_batch = build_data(seed, config, device, data_root)
    images = [img for img, _label in fixed_eval_batch]
    if not images:
        raise RuntimeError(f"[{run_dir.name}] fixed eval subset is empty; cannot simulate.")

    n_images = max(1, round(window_seconds / config.image_duration))

    manifest_rows: list[dict] = []
    done_epochs: list[int] = []
    for epoch in _figure_epochs(run_dir):
        cp = ckpts.get(epoch)
        if cp is None:
            if verbose:
                print(f"[{run_dir.name}] epoch {epoch}: no checkpoint — skipping")
            continue
        ck = _load_ckpt(cp, device)
        model.load_state_dict(ck["model_state_dict"])

        E_window, I_window, t_window = _simulate_window(
            model, images, ck["network_state"], config, n_images)

        CeLoc = _model_csr(model.CeLoc_crow_indices, model.CeLoc_col_indices,
                           model.CeLoc_values, model.CeLoc_shape)
        CeRem = _model_csr(model.CeRem_crow_indices, model.CeRem_col_indices,
                           model.CeRem_values, model.CeRem_shape)

        epoch_dir = run_dir / "figures" / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        # Remove the retired dot-raster if a prior run left one behind.
        stale = epoch_dir / "raster.png"
        if stale.exists():
            stale.unlink()

        produced = figs.save_all_figures(
            str(epoch_dir), E_window, I_window, t_window, config.dt,
            CeLoc=CeLoc, CeRem=CeRem,
            image_duration=config.image_duration, pulsed=config.pulsed_stimuli)
        for fig_type, path in produced:
            manifest_rows.append(dict(run_id=run_dir.name, seed=seed,
                                      config_name=config.name, epoch=epoch,
                                      fig_type=fig_type, path=str(path)))
        done_epochs.append(epoch)
        if verbose:
            print(f"[{run_dir.name}] regenerated epoch {epoch} ({len(produced)} figs)")

    # Rewrite the per-run manifest with the regenerated rows.
    with open(run_dir / "figure_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(MANIFEST_COLS))
        w.writeheader()
        w.writerows(manifest_rows)

    return {"run_dir": str(run_dir), "seed": seed, "config_name": config.name,
            "regenerated_epochs": done_epochs, "n_regenerated": len(done_epochs)}


def regenerate_figures(out_root: str, runs=None, device=None, data_root: str | None = None,
                       window_seconds: float = 10.0, verbose: bool = True) -> list[dict]:
    """Regenerate figures for the given runs (or every run under ``out_root`` that
    has archived checkpoints)."""
    out_root = Path(out_root)
    if runs:
        run_dirs = [out_root / r for r in runs]
    else:
        run_dirs = [d for d in sorted(out_root.iterdir())
                    if d.is_dir() and (d / "checkpoints" / "topology").exists()]

    results = []
    for rd in run_dirs:
        results.append(regenerate_figures_for_run(
            rd, device=device, data_root=data_root,
            window_seconds=window_seconds, verbose=verbose))
    total = sum(r["n_regenerated"] for r in results)
    if verbose:
        print(f"\nRegeneration complete: {total} epoch(s) across {len(results)} run(s).")
        print("Now rebuild the combined manifest:  python analysis_loader.py --out-root "
              f"{out_root}")
    return results


def _parse_runs(spec: str | None):
    return [s.strip() for s in spec.split(",") if s.strip()] if spec else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-render per-epoch figures from archived checkpoints.")
    ap.add_argument("--out-root", default=cfg.OUT_ROOT)
    ap.add_argument("--runs", default=None,
                    help="comma-separated run_ids (default: all runs with archived checkpoints)")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--data-root", default=None, help="MNIST root (default: $WC_DATA_ROOT or ./mnist_data)")
    ap.add_argument("--window-seconds", type=float, default=10.0,
                    help="simulated window length per epoch (default: 10.0)")
    args = ap.parse_args()

    regenerate_figures(out_root=args.out_root, runs=_parse_runs(args.runs),
                       device=args.device, data_root=args.data_root,
                       window_seconds=args.window_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
