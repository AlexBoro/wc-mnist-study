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
import json
import time
import hashlib
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


def model_fingerprint(config: cfg.ExperimentConfig) -> dict:
    """Every constant that changes what a run MEANS, plus a short hash of them.

    A resumable sweep skips a run whose bundle already exists.  That is only
    sound if the existing bundle was produced by the same model definition -- and
    nothing in the bundle used to record the model definition, so a mid-sweep
    change to a coefficient or a scale range would be silently mixed into the
    same output tree with no way to tell the halves apart afterwards.

    The fingerprint is written to every manifest and the hash is compared on
    resume.  It covers the E->E drive terms (coefficients and both scale ranges),
    the connectivity degrees and rewiring ranges, the inhibitory ranges, and the
    integration step -- i.e. everything that would make two runs incomparable.
    Deliberately EXCLUDES per-run sampled values (those are in params.json) and
    anything cosmetic.
    """
    fp = dict(
        w_py2py_self=float(cfg.W_PY2PY_SELF),
        w_celoc=float(getattr(config, "w_celoc", cfg.W_CELOC)),
        w_cerem=float(getattr(config, "w_cerem", cfg.W_CEREM)),
        celoc_scale=[float(cfg.CELOC_INIT_SCALE_MIN), float(cfg.CELOC_INIT_SCALE_MAX)],
        cerem_scale=[float(cfg.CEREM_INIT_SCALE_MIN), float(cfg.CEREM_INIT_SCALE_MAX)],
        celoci_scale=[float(cfg.CELOCI_INIT_SCALE_MIN), float(cfg.CELOCI_INIT_SCALE_MAX)],
        inh2py_gain=[float(cfg.INH2PY_GAIN_MIN), float(cfg.INH2PY_GAIN_MAX)],
        bias_e_init=[float(cfg.BIAS_E_INIT_MIN), float(cfg.BIAS_E_INIT_MAX)],
        bias_i_init=[float(cfg.BIAS_I_INIT_MIN), float(cfg.BIAS_I_INIT_MAX)],
        mnist_weight=[float(cfg.MNIST_WEIGHT_MIN), float(cfg.MNIST_WEIGHT_MAX)],
        outdeg_pypy=int(cfg.TARGET_OUTDEG_PyPy),
        outdeg_pyinh=int(cfg.TARGET_OUTDEG_PyInh),
        p_rewire_pypy=[float(cfg.P_REWIRE_PyPy_MIN), float(cfg.P_REWIRE_PyPy_MAX)],
        p_rewire_pyinh=[float(cfg.P_REWIRE_PyInh_MIN), float(cfg.P_REWIRE_PyInh_MAX)],
        dt=float(cfg.DT),
        mnist_input_scale=float(cfg.MNIST_INPUT_SCALE),
    )
    # Designed biases are part of what a run MEANS: two runs at the same
    # (seed, config) with different targeted biases are different runs, and a
    # resume must not inherit one for the other.  Absent when not overridden,
    # so every pre-existing bundle keeps its original hash.
    for key in ("bias_e_override", "bias_i_override"):
        val = getattr(config, key, None)
        if val is not None:
            fp[key] = float(val)
    blob = json.dumps(fp, sort_keys=True, separators=(",", ":"))
    fp["hash"] = hashlib.sha256(blob.encode()).hexdigest()[:12]
    return fp


def _collect_params(model, seed, config, alpha_H, tau_H, ic_e_max=None, ic_i_max=None):
    """Every sampled hyperparameter + per-unit array summary stats (§5)."""
    p = dict(
        seed=int(seed), config_name=config.name, input_mode=config.input_mode,
        alpha_h_mode=config.alpha_h_mode, alpha_H=float(alpha_H), tau_H=float(tau_H),
        ic_e_max=(None if ic_e_max is None else float(ic_e_max)),
        ic_i_max=(None if ic_i_max is None else float(ic_i_max)),
        bias_e=float(model.bias_e.detach().cpu()),
        bias_i=float(model.bias_i.detach().cpu()),
        bias_out=float(model.bias_out),
        N=int(model.N), num_output=int(model.num_output),
        dt=float(cfg.DT), run_seconds=float(RUN_SECONDS),
        image_duration=float(cfg.IMAGE_DURATION),
        mnist_input_scale=float(cfg.MNIST_INPUT_SCALE),
        split_seed=int(config.split_seed),
        # E->E drive coefficients: per-run, because a pilot arm varies them.
        w_celoc=float(getattr(config, "w_celoc", cfg.W_CELOC)),
        w_cerem=float(getattr(config, "w_cerem", cfg.W_CEREM)),
        arm=str(getattr(config, "arm", "")),
        fingerprint=model_fingerprint(config)["hash"],
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
                        ic_e_max: float = None, ic_i_max: float = None,
                        h_init_max: float = None,
                        h_init_mode: str = None,
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

    # Initial state.  H is drawn from init_rng (matches the harness §4).
    #
    # E0/I0: the 6/28/26 notebook's run_exploration_simulation draws
    #     E0 = rand(N) * IC_E_MAX,  I0 = rand(N) * IC_I_MAX   (both 1.0)
    # while this module previously started at E=I=0.  For a bistable network
    # that is a DIFFERENT BASIN, not a transient: it suppressed seizures in the
    # sequential configurations specifically (past-boundary rate 0.11 vs the
    # notebook's 0.94) while barely affecting `simultaneous`, whose 784-unit
    # drive supplies its own kick.  Defaults now follow the notebook.
    #
    # The draw uses a SEPARATE generator seeded from `seed`, deliberately NOT
    # init_rng, so changing ic_e_max/ic_i_max cannot shift the init_rng stream
    # position -- network parameters stay bit-identical to a run at any other
    # IC, isolating the initial condition as the only difference.
    ic_e_max = cfg.IC_E_MAX if ic_e_max is None else float(ic_e_max)
    ic_i_max = cfg.IC_I_MAX if ic_i_max is None else float(ic_i_max)
    # h_init_max override: probes H0's range WITHOUT moving the init_rng stream.
    # uniform(lo, hi, size=N) consumes exactly N draws whatever the bounds are, so
    # network parameters stay bit-identical to a run at any other h_init_max --
    # same discipline as ic_e_max/ic_i_max above.  H0 is the STATE of the
    # adaptation variable; alpha_H (its strength) is untouched.
    h_init_max = cfg.H_INIT_MAX if h_init_max is None else float(h_init_max)
    H0 = init_rng.uniform(cfg.H_INIT_MIN, h_init_max, size=model.N).astype(np.float32)
    ic_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xE0]))
    E0 = (ic_rng.random(model.N) * ic_e_max).astype(np.float32)
    I0 = (ic_rng.random(model.N) * ic_i_max).astype(np.float32)
    # h_init_mode="match_e0": start every unit at its own adaptation fixed
    # point.  The model integrates tau_H*dH/dt = -H + E, so H_i(0) = E_i(0)
    # gives dH/dt = 0 per unit at t=0 -- no charging transient, no free
    # parameter.
    #
    # DO NOT USE THIS FOR PRODUCTION RUNS.  Retained as a documented negative
    # result.  Tested 2026-08-13 on 22 designed seeds (sequential_H): it does
    # NOT preserve the epileptiform dynamics.  Late-window (final 2 s of the
    # 5 s probe) population E is statistically indistinguishable from an
    # INDEPENDENT H0 ~ U(0, 1) draw -- r(log) = 1.000 across seeds, median
    # ratio 1.00, and the same two seeds (069, 075) survive under both.
    # Seizure-class late E collapses 0.319 (control) -> 0.006.
    #
    # The hypothesis this mode was built to test -- that per-unit CORRELATION
    # with E0 would give graded suppression that spares the seizure runs --
    # is refuted.  What matters is the MEAN of H0 (~0.5 here, vs 0.005 for
    # the U(0, 0.01) default); the correlation structure is irrelevant.
    # See H0_matched_result.png / H0_matched_late.csv.  Use H_INIT_MAX = 0.15
    # instead: it removes the charging transient and preserves seizure-class
    # late activity (0.318 vs 0.319 control).
    #
    # The init_rng.uniform draw above is still executed and then discarded, so
    # the stream position -- and hence all 11 weight tensors -- stays
    # bit-identical to any other mode (verified by sha256 over
    # named_parameters + named_buffers).
    _hmode = "uniform" if h_init_mode is None else str(h_init_mode)
    if _hmode == "match_e0":
        H0 = E0.copy()
    elif _hmode != "uniform":
        raise ValueError(f"unknown h_init_mode {_hmode!r} "
                         "(expected 'uniform' or 'match_e0')")
    E = torch.tensor(E0, device=device)
    I = torch.tensor(I0, device=device)
    O = torch.zeros(model.num_output, device=device)
    H = torch.tensor(H0, device=device)

    # ---- stimulus stream --------------------------------------------------
    num_steps = int(round(run_seconds / dt))
    steps_per_image = int(round(cfg.IMAGE_DURATION / dt))
    n_images = int(np.ceil(num_steps / steps_per_image))
    images, mnist_indices = get_image_stream(seed, config, n_images, device, data_root)

    seq = config.sequential_row_input
    row_dur = config.row_duration if seq else None
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
                current_row = M.active_row_at(step, dt, cfg.IMAGE_DURATION, row_dur)
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

    params = _collect_params(model, seed, config, alpha_H, tau_H, ic_e_max, ic_i_max)
    params["h_init_max"] = float(h_init_max)
    params["h_init_mode"] = _hmode
    params["runtime_s"] = runtime_s
    arm = str(getattr(config, "arm", "") or "")
    run_id = f"seed{seed:03d}__{config.name}" + (f"__{arm}" if arm else "")
    # A targeted-bias run gets the designed biases in its id, so two design
    # points at one seed cannot resolve to the same output directory.
    if getattr(config, "bias_e_override", None) is not None or \
       getattr(config, "bias_i_override", None) is not None:
        run_id += f"__be{params['bias_e']:+.2f}_bi{params['bias_i']:+.2f}"
    if verbose:
        print(f"[{run_id}] alpha_H={alpha_H:.3f} tau_H={tau_H:.3f} "
              f"E[max]={E_traj.max():.3f} t={runtime_s:.1f}s", flush=True)

    return ExplorationResult(
        run_id=run_id, seed=int(seed), config_name=config.name,
        input_mode=config.input_mode, alpha_H=float(alpha_H), tau_H=float(tau_H),
        E_traj=E_traj, I_traj=I_traj, times=times, stim_onsets=stim_onsets,
        mnist_indices=mnist_indices, params=params, runtime_s=runtime_s,
    )
