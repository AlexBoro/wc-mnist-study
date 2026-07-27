"""param_explore.py — clean no-training parameter-exploration driver.

One run = a single free-running 10-second simulation of a freshly-initialised
random Wilson-Cowan network, driven by MNIST as external input, with NO
training (no loss, no backprop, no epoch loop).  See ``param_explore_spec.md``.

The network initialisation is *identical* to the longitudinal harness
(``wc_experiment.build_model``), so a param-explore run and a longitudinal
epoch-0 network built from the same ``seed`` + ``config`` share parameters.
Only the training loop is dropped.

Reproducibility contract (§5 of the spec): a run is fully described by
``(seed, config_name)`` plus the pinned code version.  ``run_one_exploration``
is deterministic on a fixed library/GPU stack.
"""
from __future__ import annotations

import os
import sys
import time
import platform
from dataclasses import dataclass

import numpy as np
import torch

# Local engine (frozen dynamics + canonical init) --------------------------
import wc_config as cfg
import wc_model as M
import wc_experiment as X

RUN_SECONDS = 10.0                       # single-run simulation length (§3)


# ---------------------------------------------------------------------------
# MNIST image stream (seed-123 split, seed-seeded shuffle) — reproducible
# ---------------------------------------------------------------------------
def get_image_stream(seed: int, config: cfg.ExperimentConfig, n_images: int,
                     device, data_root=None):
    """Return (images [n_images, 784] float32 on device, dataset_indices list).

    Uses the identical seed-123 train/val split and the seed-seeded shuffle
    order that ``wc_experiment.build_data`` uses, then takes the first
    ``n_images`` images of that shuffled train stream.  The dataset indices are
    recorded so the exact stimulus sequence is reconstructable.
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset

    if data_root is None:
        data_root = X._resolve_data_root()

    full_train = datasets.MNIST(root=data_root, transform=transforms.ToTensor(),
                                download=True, train=True)

    indices = np.arange(len(full_train))
    split_rng = np.random.default_rng(config.split_seed)
    split_rng.shuffle(indices)
    train_indices = indices[:cfg.TRAIN_SIZE]
    mnist_train = Subset(full_train, train_indices)

    # Seed-seeded shuffle over the seed-123 train split.  There is no training,
    # so the stimulus only needs to be reproducible (seeded randperm) and
    # recorded (true underlying MNIST dataset indices) — it does NOT need to
    # match the longitudinal DataLoader's RandomSampler order.
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(mnist_train), generator=g).tolist()
    take = perm[:n_images]

    imgs, ds_idx = [], []
    for sub_i in take:
        img, _label = mnist_train[sub_i]                     # Subset.__getitem__
        imgs.append(img.view(-1))
        ds_idx.append(int(train_indices[sub_i]))             # underlying dataset index
    images = torch.stack(imgs).to(device)                    # [n_images, 784]
    return images, ds_idx


# ---------------------------------------------------------------------------
# One exploration run
# ---------------------------------------------------------------------------
@dataclass
class ExplorationResult:
    run_id: str
    seed: int
    config_name: str
    input_mode: str
    alpha_H: float
    tau_H: float
    E_traj: np.ndarray          # [T, N] float32
    I_traj: np.ndarray          # [T, N] float32
    times: np.ndarray           # [T] float32
    stim_onsets: np.ndarray     # [n_images] float32
    mnist_indices: list
    params: dict                # all sampled hyperparameters (+ summary stats)
    runtime_s: float


def _summarise(name, arr):
    a = np.asarray(arr, dtype=np.float64).ravel()
    return {f"{name}_mean": float(a.mean()), f"{name}_std": float(a.std()),
            f"{name}_min": float(a.min()), f"{name}_max": float(a.max())}


def _collect_params(model, seed, config, alpha_H, tau_H):
    """Every sampled hyperparameter + per-unit array summary stats (§5)."""
    p = dict(
        seed=int(seed), config_name=config.name, input_mode=config.input_mode,
        alpha_h_mode=config.alpha_h_mode, alpha_H=float(alpha_H), tau_H=float(tau_H),
        bias_e=float(model.bias_e.detach().cpu()),
        bias_i=float(model.bias_i.detach().cpu()),
        bias_out=float(model.bias_out),
        N=int(model.N), num_output=int(model.num_output),
        dt=float(cfg.DT), run_seconds=float(RUN_SECONDS),
        image_duration=float(cfg.IMAGE_DURATION),
        mnist_input_scale=float(cfg.MNIST_INPUT_SCALE),
        split_seed=int(config.split_seed),
    )
    # per-unit / per-edge summaries
    p.update(_summarise("tauPy_eff", model.tauPy_eff.detach().cpu().numpy()))
    p.update(_summarise("tauInh_eff", model.tauInh_eff.detach().cpu().numpy()))
    p.update(_summarise("mnist_weights", model.mnist_weights.detach().cpu().numpy()))
    p.update(_summarise("inh_to_exc_gain", model.inh_to_exc_gain.detach().cpu().numpy()))
    p.update(_summarise("CeLoc_values", model.CeLoc_values.detach().cpu().numpy()))
    p.update(_summarise("CeRem_values", model.CeRem_values.detach().cpu().numpy()))
    p.update(_summarise("CeLocI_values", model.CeLocI_values.detach().cpu().numpy()))
    # connectivity edge counts (proxy for realised rewiring)
    p["n_CeLoc_edges"] = int(model.CeLoc_values.numel())
    p["n_CeRem_edges"] = int(model.CeRem_values.numel())
    p["n_CeLocI_edges"] = int(model.CeLocI_values.numel())
    return p


def run_one_exploration(seed: int, config: cfg.ExperimentConfig, device=None,
                        run_seconds: float = RUN_SECONDS, data_root=None,
                        record_stride: int = 1, dtype_out=np.float32,
                        verbose: bool = False) -> ExplorationResult:
    """Single 10s free-running simulation of a random network (§3).

    Deterministic given ``(seed, config)`` on a fixed library/GPU stack.  The
    init draw order is IDENTICAL to ``wc_experiment.run_experiment`` (so a
    param-explore run shares parameters with the longitudinal epoch-0 network
    of the same seed+config).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    dt = cfg.DT
    # ---- canonical init (mirror run_experiment lines 540-556) -------------
    X.seed_everything(seed)
    init_rng, config_rng = X.make_rngs(seed)
    alpha_H, tau_H = X.resolve_alpha_tau(config, config_rng)
    model = X.build_model(seed, config, device, init_rng, alpha_H, tau_H)
    model.eval()

    # Initial state: E=I=O=0, H drawn from init_rng (matches the harness §4).
    H0 = init_rng.uniform(cfg.H_INIT_MIN, cfg.H_INIT_MAX, size=model.N).astype(np.float32)
    E = torch.zeros(model.N, device=device)
    I = torch.zeros(model.N, device=device)
    O = torch.zeros(model.num_output, device=device)
    H = torch.tensor(H0, device=device)

    # ---- stimulus stream --------------------------------------------------
    num_steps = int(round(run_seconds / dt))
    steps_per_image = int(round(cfg.IMAGE_DURATION / dt))
    n_images = int(np.ceil(num_steps / steps_per_image))
    images, mnist_indices = get_image_stream(seed, config, n_images, device, data_root)

    seq = config.sequential_row_input
    row_dur = config.row_duration if seq else None
    steps_per_row = max(1, int(round((row_dur if row_dur else cfg.IMAGE_DURATION / 28) / dt)))
    scale = cfg.MNIST_INPUT_SCALE

    # pre-scale images once
    scaled_images = images * scale

    # record buffers
    rec_idx = list(range(0, num_steps, record_stride))
    T = len(rec_idx)
    E_traj = np.empty((T, model.N), dtype=dtype_out)
    I_traj = np.empty((T, model.N), dtype=dtype_out)
    times = np.empty(T, dtype=np.float32)
    stim_onsets = np.array([i * cfg.IMAGE_DURATION for i in range(n_images)
                            if i * cfg.IMAGE_DURATION < run_seconds], dtype=np.float32)

    t0 = time.time()
    rec_ptr = 0
    with torch.no_grad():
        for step in range(num_steps):
            img_idx = min(step // steps_per_image, n_images - 1)
            mnist_image = scaled_images[img_idx]
            if seq:
                current_row = min((step % steps_per_image) // steps_per_row, 27)
                I_ext = model.compute_mnist_input(mnist_image, active_row=current_row)
            else:
                I_ext = model.compute_mnist_input(mnist_image)
            E, I, O, H = model.step(E, I, O, I_ext, dt, H=H)
            if rec_ptr < T and step == rec_idx[rec_ptr]:
                E_traj[rec_ptr] = E.detach().cpu().numpy().astype(dtype_out)
                I_traj[rec_ptr] = I.detach().cpu().numpy().astype(dtype_out)
                times[rec_ptr] = step * dt
                rec_ptr += 1
    runtime_s = time.time() - t0

    params = _collect_params(model, seed, config, alpha_H, tau_H)
    params["runtime_s"] = runtime_s
    run_id = f"seed{seed:03d}__{config.name}"
    if verbose:
        print(f"[{run_id}] alpha_H={alpha_H:.3f} tau_H={tau_H:.3f} "
              f"E[max]={E_traj.max():.3f} t={runtime_s:.1f}s", flush=True)

    return ExplorationResult(
        run_id=run_id, seed=int(seed), config_name=config.name,
        input_mode=config.input_mode, alpha_H=float(alpha_H), tau_H=float(tau_H),
        E_traj=E_traj, I_traj=I_traj, times=times, stim_onsets=stim_onsets,
        mnist_indices=mnist_indices, params=params, runtime_s=runtime_s,
    )
