"""wc_experiment.py — One (seed, config) longitudinal run.

``run_experiment`` is the unit of work: it builds the seed-derived RNG streams
(§5), instantiates the frozen model (``wc_model``), runs the 100-epoch loop, and
on every epoch writes a Parquet/CSV row (§7), checkpoints (§11), and — on
topology-trigger epochs (§8) — measures loss-surface topology (``wc_topology``)
and saves activity figures + manifest rows (``wc_figures``, §12).

Idempotent & resumable: re-invoking resumes from the latest checkpoint and
continues; a completed run is marked by a ``DONE`` sentinel (§9/§13).

Determinism (A1): all global RNGs are seeded from the set seed and the two
numpy streams are spawned from a single ``SeedSequence(seed)``, so two runs of
the same (seed, config) produce identical loss/acc curves, and the three configs
of a set share identical epoch-0 model parameters.
"""

from __future__ import annotations

# CUBLAS workspace must be configured *before* torch is imported for
# deterministic cuBLAS GEMM on GPU (spec §5).  No-op on CPU.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import pickle
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import wc_config as cfg
import wc_model as M

CHECKPOINT_VERSION = "v82_longitudinal"
TRAIN_HEATMAP_WINDOW = 10.0   # seconds of E/I activity retained per epoch (figures)

# Canonical Parquet column order (§7).  Kept explicit so the schema is stable.
METRIC_COLUMNS = [
    # identity / config
    "run_id", "seed", "config_name", "input_mode", "alpha_H", "tau_H",
    "device", "deterministic_ok", "epoch",
    # performance
    "train_loss", "train_acc", "val_loss", "val_acc",
    "d_train_loss", "d_train_acc", "d_val_loss", "d_val_acc",
    # bias parameters (the drift)
    "bias_e", "bias_i",
    # clean gradients
    "grad_bias_e", "grad_bias_i",
    "grad_norm_CeLoc", "grad_norm_CeRem", "grad_norm_CeLocI",
    "grad_angle_to_boundary_deg",
    # connectivity / heterogeneity summaries
    "CeLoc_mean", "CeLoc_std", "CeLoc_l2",
    "CeRem_mean", "CeRem_std", "CeRem_l2",
    "CeLocI_mean", "CeLocI_std", "CeLocI_l2",
    "inh_to_exc_gain_mean", "inh_to_exc_gain_std",
    "mnist_weights_mean", "mnist_weights_std",
    "tauPy_eff_mean", "tauPy_eff_std", "tauInh_eff_mean", "tauInh_eff_std",
    "d_CeLoc_l2", "d_CeRem_l2",
    # seizure-boundary bookkeeping
    "boundary_slope", "boundary_intercept", "signed_dist_to_boundary",
    "on_seizure_side", "crossed_this_epoch",
    # topology
    "topology_measured", "trigger_reason",
    "domain_n_saddle", "domain_avg_persistence", "domain_max_persistence",
    "domain_loss_min", "domain_loss_max", "domain_loss_range", "domain_loss_std",
    "hessian_n_saddle", "hessian_avg_persistence", "hessian_max_persistence",
    "hessian_eig1", "hessian_eig2",
    # activity / seizure markers
    "H_final_mean", "H_final_std", "mean_E_activity",
]

FIGURE_MANIFEST_COLUMNS = ["run_id", "seed", "config_name", "epoch", "fig_type", "path"]


@dataclass
class RunResult:
    run_id: str
    seed: int
    config_name: str
    out_dir: str
    device: str
    deterministic_ok: bool
    alpha_H: float
    tau_H: float
    n_epochs: int
    epochs_completed: int
    done: bool
    metrics_path: str


# =============================================================================
# RNG (spec §5)
# =============================================================================
def seed_everything(seed: int) -> bool:
    """Seed all global RNGs and enable determinism flags.  Returns whether
    ``torch.use_deterministic_algorithms`` could be enabled."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        return True
    except Exception:
        return False


def make_rngs(seed: int):
    """Two independent numpy Generators spawned from one SeedSequence(seed):
    ``init_rng`` seeds all network initialization; ``config_rng`` draws alpha_H /
    tau_H.  Spawning guarantees the streams are independent yet both derive only
    from the set seed, so toggling a config never shifts the init stream."""
    init_ss, config_ss = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(init_ss), np.random.default_rng(config_ss)


def resolve_alpha_tau(config: cfg.ExperimentConfig, config_rng) -> tuple[float, float]:
    """Return (alpha_H, tau_H).  OFF configs return (0.0, default tau_H=1.0) and
    consume no config_rng draws; the H-on config draws both (spec §6)."""
    if config.alpha_h_mode == "off":
        return 0.0, 1.0
    if config.alpha_h_mode == "uniform_1_10":
        alpha_H = float(config_rng.uniform(cfg.ALPHA_H_ON_MIN, cfg.ALPHA_H_ON_MAX))
        tau_H = float(config_rng.uniform(cfg.TAU_H_MIN, cfg.TAU_H_MAX))
        return alpha_H, tau_H
    raise ValueError(f"Unknown alpha_h_mode: {config.alpha_h_mode!r}")


# =============================================================================
# Build model + data for a run
# =============================================================================
def build_model(seed: int, config: cfg.ExperimentConfig, device, init_rng, alpha_H, tau_H):
    """Instantiate model with the canonical (config-independent) init draw order.

    Draw order from ``init_rng`` — identical across the three configs of a set,
    which is what guarantees identical epoch-0 parameters (A1):
      p_rewire_pypy, p_rewire_pyinh, bias_out, output_weight_scale,
      bias_e_init, bias_i_init, (connectivity rewiring), (model weight inits).
    """
    p_rewire_pypy = float(init_rng.uniform(cfg.P_REWIRE_PyPy_MIN, cfg.P_REWIRE_PyPy_MAX))
    p_rewire_pyinh = float(init_rng.uniform(cfg.P_REWIRE_PyInh_MIN, cfg.P_REWIRE_PyInh_MAX))
    bias_out = float(init_rng.uniform(cfg.BIAS_OUT_MIN, cfg.BIAS_OUT_MAX))
    output_weight_scale = float(init_rng.uniform(cfg.OUTPUT_WEIGHT_SCALE_MIN, cfg.OUTPUT_WEIGHT_SCALE_MAX))
    bias_e_init = float(init_rng.uniform(cfg.BIAS_E_INIT_MIN, cfg.BIAS_E_INIT_MAX))
    bias_i_init = float(init_rng.uniform(cfg.BIAS_I_INIT_MIN, cfg.BIAS_I_INIT_MAX))

    # Targeted-bias pilots override the DRAWN biases.  The draws above still
    # happen and are then discarded, which is the whole point: init_rng is left
    # at exactly the position it would have had, so connectivity rewiring and
    # every model weight below are bit-identical to the unoverridden run at the
    # same seed.  Only the two bias scalars differ.  (Same discipline as the
    # w_celoc/w_cerem arms, which are consumed after every draw.)
    bias_e_override = getattr(config, "bias_e_override", None)
    bias_i_override = getattr(config, "bias_i_override", None)
    if bias_e_override is not None:
        bias_e_init = float(bias_e_override)
    if bias_i_override is not None:
        bias_i_init = float(bias_i_override)

    CeLoc, CeRem, CeLocI = M.build_connectivity(init_rng, p_rewire_pypy, p_rewire_pyinh, device)

    model = M.SparseBigWilsonCowan(
        CeLoc, CeRem, CeLocI, rng=init_rng,
        bias_e_init=bias_e_init, bias_i_init=bias_i_init,
        bias_out=bias_out, output_weight_scale=output_weight_scale,
        tau_H=tau_H, alpha_H=alpha_H, device=device,
        mnist_input_dim=784, num_output=10,
        # Carried on the config so a pilot arm varying the E->E coefficients is
        # fully described by (seed, config) and lands in the run manifest.
        # NOTE these are consumed AFTER every init_rng draw above, so varying
        # them does NOT perturb the draw order -- two arms at the same seed get
        # bit-identical connectivity, biases and weights, differing only in the
        # coefficients.  That is what makes the arms directly comparable.
        w_celoc=getattr(config, "w_celoc", None),
        w_cerem=getattr(config, "w_cerem", None),
    )
    model.to(device)
    return model


def _resolve_data_root() -> str:
    return os.environ.get("WC_DATA_ROOT", str(Path.cwd() / "mnist_data"))


def build_data(seed: int, config: cfg.ExperimentConfig, device, data_root=None):
    """Build the shuffled train loader (seeded generator ``g``), the fixed eval
    loader, and the fixed eval batch (first ``eval_images`` of mnist_val under the
    seed-123 split order — used for val metrics, the gradient probe, and topology)."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset

    if data_root is None:
        data_root = _resolve_data_root()

    full_train = datasets.MNIST(root=data_root, transform=transforms.ToTensor(),
                                download=True, train=True)

    indices = np.arange(len(full_train))
    split_rng = np.random.default_rng(config.split_seed)
    split_rng.shuffle(indices)
    train_indices = indices[:cfg.TRAIN_SIZE]
    val_indices = indices[cfg.TRAIN_SIZE:]

    mnist_train = Subset(full_train, train_indices)

    # Fixed eval subset: first `eval_images` of mnist_val in split order.
    fixed_indices = list(val_indices[:config.eval_images])
    fixed_eval_subset = Subset(full_train, fixed_indices)
    fixed_eval_loader = DataLoader(fixed_eval_subset, batch_size=config.batch_size, shuffle=False)
    fixed_eval_batch = [
        (img.view(-1).to(device), torch.tensor(int(label), device=device))
        for img, label in fixed_eval_subset
    ]

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(mnist_train, batch_size=config.batch_size,
                              shuffle=True, generator=g)
    return train_loader, g, fixed_eval_loader, fixed_eval_batch


# =============================================================================
# Per-epoch training (mirrors cell-24 inner loop) with activity capture
# =============================================================================
def _train_one_epoch(model, train_loader, optimizer, criterion, config,
                     state, capture_activity=True):
    """Train one epoch over the first ``images_per_epoch`` images, carrying the
    E/I/O/H state across images.

    Returns ``(train_loss, train_acc, E_window, I_window, t_window,
    mean_E_activity)``.  ``E_window``/``I_window`` are the last
    ``TRAIN_HEATMAP_WINDOW`` seconds of activity as numpy arrays (empty unless
    ``capture_activity``); ``mean_E_activity`` is the epoch-mean of E and is
    always returned (accumulated as a GPU scalar, independent of capture)."""
    from collections import deque

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    dt = config.dt
    max_steps_window = int(TRAIN_HEATMAP_WINDOW / dt)
    E_buf = deque(maxlen=max_steps_window)        # GPU tensors (detached)
    I_buf = deque(maxlen=max_steps_window)
    t_buf = deque(maxlen=max_steps_window)
    e_sum = torch.zeros((), device=model.device)  # running sum of mean(E), on GPU
    e_count = 0

    def activity_callback(t, E_snap, I_snap, H_snap=None):
        nonlocal e_sum, e_count
        e_sum = e_sum + E_snap.mean()             # GPU scalar, no host sync
        e_count += 1
        if capture_activity:
            E_buf.append(E_snap)                  # already detached by train_one_image
            I_buf.append(I_snap)
            t_buf.append(float(t))

    cb = activity_callback   # always on: cheap (no per-step sync); feeds mean_E_activity

    E_state = state["E"]
    I_state = state["I"]
    O_state = state["O"]
    H_state = state["H"]

    for _, (data, target) in enumerate(train_loader):
        data = data.to(model.device)
        target = target.to(model.device)
        for i in range(data.size(0)):
            if total_images >= config.images_per_epoch:
                break
            mnist_image = data[i].view(-1)
            label = target[i]
            loss, _, correct, E_state, I_state, O_state, H_state = M.train_one_image(
                model, mnist_image, label, optimizer, criterion,
                dt=dt, image_duration=config.image_duration,
                output_average_window=config.output_average_window,
                E_init=E_state, I_init=I_state, O_init=O_state, H_init=H_state,
                activity_callback=cb, activity_time_offset=state["global_train_time_s"],
                sequential_row_input=config.sequential_row_input,
                row_duration=config.row_duration if config.sequential_row_input else None,
                mnist_input_scale=config.mnist_input_scale,
            )
            if not np.isfinite(float(loss)):
                raise RuntimeError("Stopping: non-finite loss (NaN/Inf).")
            state["global_train_time_s"] += config.image_duration
            total_loss += loss
            total_correct += correct
            total_images += 1
        if total_images >= config.images_per_epoch:
            break

    state["E"], state["I"], state["O"], state["H"] = E_state, I_state, O_state, H_state
    train_loss = total_loss / max(total_images, 1)
    train_acc = total_correct / max(total_images, 1)
    mean_E_activity = float((e_sum / e_count).item()) if e_count else None

    # Single GPU->CPU transfer at the end (only when snapshots were captured).
    if capture_activity and len(E_buf):
        E_window = torch.stack(tuple(E_buf)).cpu().numpy()
        I_window = torch.stack(tuple(I_buf)).cpu().numpy()
        t_window = np.array(t_buf, dtype=float)
    else:
        E_window = np.zeros((0, model.N), dtype=np.float32)
        I_window = np.zeros((0, model.N), dtype=np.float32)
        t_window = np.zeros((0,), dtype=float)
    return train_loss, train_acc, E_window, I_window, t_window, mean_E_activity


# =============================================================================
# Clean-gradient probe (§7)
# =============================================================================
def clean_gradient_probe(model, criterion, fixed_eval_batch, config, boundary):
    """One forward+backward over the fixed eval batch at current params, from a
    clean (zero) initial state.  Returns bias/connectivity gradient summaries and
    the angle between (grad_bias_e, grad_bias_i) and the seizure-boundary normal.
    Does not step the optimizer; zeroes grads afterward.

    The probe images are independent (each runs from a zero state), so they are
    evaluated as a single [B, N] batch (``simulate_window_batched``) — one forward
    + one backward instead of B sequential 125-step graphs.  Numerically
    equivalent to the per-image mean-CE gradient; far faster and lighter on a GPU.
    """
    device = model.device
    n_imgs = min(config.grad_probe_images, len(fixed_eval_batch))
    images = torch.stack([img for img, _ in fixed_eval_batch[:n_imgs]]).to(device)   # [B, 784]
    labels = torch.stack([torch.as_tensor(lbl) for _, lbl in fixed_eval_batch[:n_imgs]]).to(device)

    model.train()
    model.zero_grad(set_to_none=True)
    avg_output = M.simulate_window_batched(
        model, images, config.dt, config.image_duration, config.output_average_window,
        sequential_row_input=config.sequential_row_input,
        row_duration=config.row_duration if config.sequential_row_input else None,
        mnist_input_scale=config.mnist_input_scale)
    loss = criterion(avg_output, labels)
    loss.backward()

    g_be = float(model.bias_e.grad.item())
    g_bi = float(model.bias_i.grad.item())
    gn_loc = float(model.CeLoc_values.grad.norm(2).item())
    gn_rem = float(model.CeRem_values.grad.norm(2).item())
    gn_loci = float(model.CeLocI_values.grad.norm(2).item())

    # Angle between gradient (g_be, g_bi) and the boundary normal (m, -1)/sqrt(m^2+1)
    m, _c = boundary
    n = np.array([m, -1.0]) / np.sqrt(m * m + 1.0)
    gvec = np.array([g_be, g_bi])
    gnorm = np.linalg.norm(gvec)
    if gnorm < 1e-12:
        angle = float("nan")
    else:
        cosang = float(np.clip(np.dot(gvec, n) / gnorm, -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cosang)))

    model.zero_grad(set_to_none=True)
    return dict(grad_bias_e=g_be, grad_bias_i=g_bi,
                grad_norm_CeLoc=gn_loc, grad_norm_CeRem=gn_rem, grad_norm_CeLocI=gn_loci,
                grad_angle_to_boundary_deg=angle)


# =============================================================================
# Boundary bookkeeping (§10)
# =============================================================================
def signed_distance_to_boundary(bias_e: float, bias_i: float, boundary) -> float:
    """Signed distance of (bE, bI) to the line bI = m*bE + c using normal
    (m, -1)/sqrt(m^2+1).  Sign convention per §10 (positive = seizure-prone side)."""
    m, c = boundary
    return (m * bias_e - bias_i + c) / np.sqrt(m * m + 1.0)


# =============================================================================
# Topology trigger (§8)
# =============================================================================
def topology_trigger(epoch: int, train_loss_hist, train_acc_hist,
                     last_topo_epoch: int, config) -> tuple[bool, str | None]:
    """Decide whether to measure topology this epoch.  ``*_hist`` include the
    current epoch's value as the last element."""
    if epoch == 0:
        return True, "epoch0"
    if (epoch + 1) % config.topo_track_interval == 0:
        return True, "scheduled"

    # Sign-flip on smoothed (trailing-window) deltas of train_loss OR train_acc,
    # with per-metric deadband and a minimum gap since the last measurement.
    if (epoch - last_topo_epoch) < config.topo_min_gap:
        return False, None

    def smoothed_delta_sign(hist, eps, w):
        # raw deltas
        if len(hist) < w + 2:
            return None, None
        deltas = np.diff(np.asarray(hist, dtype=float))
        if len(deltas) < w + 1:
            return None, None
        cur = float(np.mean(deltas[-w:]))
        prev = float(np.mean(deltas[-(w + 1):-1]))
        if abs(cur) < eps or abs(prev) < eps:
            return 0, 0  # within deadband -> treat as no flip
        return (1 if cur > 0 else -1), (1 if prev > 0 else -1)

    w = config.topo_smooth_window
    for hist, eps in ((train_loss_hist, config.topo_eps_loss),
                      (train_acc_hist, config.topo_eps_acc)):
        cur_s, prev_s = smoothed_delta_sign(hist, eps, w)
        if cur_s is not None and cur_s != 0 and prev_s != 0 and cur_s != prev_s:
            return True, "sign_flip"
    return False, None


# =============================================================================
# Parameter summaries
# =============================================================================
def _mean_std_l2(t: torch.Tensor):
    with torch.no_grad():
        return (float(t.mean().item()),
                float(t.std(unbiased=False).item()),
                float(t.norm(2).item()))


def _param_summaries(model) -> dict:
    cl_m, cl_s, cl_l2 = _mean_std_l2(model.CeLoc_values)
    cr_m, cr_s, cr_l2 = _mean_std_l2(model.CeRem_values)
    ci_m, ci_s, ci_l2 = _mean_std_l2(model.CeLocI_values)
    inh_m, inh_s, _ = _mean_std_l2(model.inh_to_exc_gain)
    mn_m, mn_s, _ = _mean_std_l2(model.mnist_weights)
    with torch.no_grad():
        tpy = model.tauPy_eff
        tih = model.tauInh_eff
    return dict(
        bias_e=float(model.bias_e.item()), bias_i=float(model.bias_i.item()),
        CeLoc_mean=cl_m, CeLoc_std=cl_s, CeLoc_l2=cl_l2,
        CeRem_mean=cr_m, CeRem_std=cr_s, CeRem_l2=cr_l2,
        CeLocI_mean=ci_m, CeLocI_std=ci_s, CeLocI_l2=ci_l2,
        inh_to_exc_gain_mean=inh_m, inh_to_exc_gain_std=inh_s,
        mnist_weights_mean=mn_m, mnist_weights_std=mn_s,
        tauPy_eff_mean=float(tpy.mean().item()), tauPy_eff_std=float(tpy.std(unbiased=False).item()),
        tauInh_eff_mean=float(tih.mean().item()), tauInh_eff_std=float(tih.std(unbiased=False).item()),
    )


# =============================================================================
# Checkpoint I/O (§11)
# =============================================================================
def _save_checkpoint(ckpt_dir: Path, epoch: int, model, optimizer, g, state,
                     trigger_state, run_id, seed, config_name, keep_last=2,
                     retain=False):
    """Write the rolling checkpoint (pruned to ``keep_last``).  When ``retain`` is
    set (a deferred-topology trigger epoch), also archive a permanent copy under
    ``checkpoints/topology/`` so the later topology pass can reload the exact model
    state topology would have been measured on, without retraining.  The archive
    subdir is outside the ``checkpoint_epoch_*.pkl`` glob, so pruning never touches
    it and the ``keep_last`` invariant on the rolling checkpoints is unchanged."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rng_states = {
        "torch_cpu": torch.random.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        rng_states["torch_cuda"] = torch.cuda.get_rng_state()

    ckpt = {
        "version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "run_id": run_id,
        "seed": seed,
        "config_name": config_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_states": rng_states,
        "dl_generator_state": g.get_state(),
        "network_state": {
            "E": state["E"].detach().cpu(),
            "I": state["I"].detach().cpu(),
            "O": state["O"].detach().cpu(),
            "H": state["H"].detach().cpu(),
            "global_train_time_s": state["global_train_time_s"],
        },
        "trigger_state": trigger_state,
    }
    path = ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pkl"
    tmp = ckpt_dir / f".checkpoint_epoch_{epoch:04d}.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    os.replace(tmp, path)

    # Archive a permanent copy at deferred-topology trigger epochs.
    if retain:
        topo_dir = ckpt_dir / "topology"
        topo_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, topo_dir / f"topo_checkpoint_epoch_{epoch:04d}.pkl")

    # keep only the most recent `keep_last` (non-recursive: skips topology/)
    ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pkl"))
    for old in ckpts[:-keep_last]:
        try:
            old.unlink()
        except OSError:
            pass


def _load_latest_checkpoint(ckpt_dir: Path):
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pkl"))
    if not ckpts:
        return None
    with open(ckpts[-1], "rb") as f:
        return pickle.load(f)


# =============================================================================
# Metrics table I/O
# =============================================================================
def _flush_metrics(rows: list[dict], run_dir: Path):
    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    parquet_ok = True
    try:
        df.to_parquet(run_dir / "metrics.parquet", index=False)
    except Exception as e:
        parquet_ok = False
        print(f"  [warn] parquet write failed ({e}); CSV mirror still written")
    df.to_csv(run_dir / "metrics.csv", index=False)
    return parquet_ok


def epoch_reconstructor(epoch_dir):
    """Return a ``reconstruct_connectivity``-compatible callable for a training
    epoch bundle written by ``run_experiment``.

    Pass to ``labeling_toolkit.analytic_features.extract(run_dir, recon, cfg, wc_figures)``
    IN PLACE OF ``param_render._reconstruct_connectivity``:

        import wc_experiment as X, param_render as PR, wc_config as cfg
        import wc_figures as WF, labeling_toolkit.analytic_features as AF
        d = Path(run_dir) / "arrays" / "epoch_0048"
        feats = AF.extract(d, X.epoch_reconstructor(d), cfg, WF)

    Why this exists: ``PR._reconstruct_connectivity`` rebuilds the model from the
    seed, which yields the connectivity the run STARTED with.  That is right for an
    exploration probe (it never trains) and wrong for a training epoch, where
    CeLoc/CeRem are precisely what gradient descent has moved.  This reads the
    trained CSR that was saved with the arrays instead, so the LFP proxy is
    computed from the same weights the dynamics were generated with.

    Signature matches the toolkit's expectation ``(seed, config_name, arm="",
    w_celoc=None, w_cerem=None)``; the arguments are accepted and ignored, since
    the answer is stored rather than derived.  The saved ``w_celoc``/``w_cerem``
    are returned, so an arm's coefficient override is honoured automatically.
    """
    import scipy.sparse as sp
    z = np.load(Path(epoch_dir) / "arrays.npz")

    def _recon(seed, config_name, arm="", w_celoc=None, w_cerem=None):
        CeLoc = sp.csr_matrix((z["CeLoc_data"], z["CeLoc_indices"], z["CeLoc_indptr"]),
                              shape=tuple(int(v) for v in z["CeLoc_shape"]))
        CeRem = sp.csr_matrix((z["CeRem_data"], z["CeRem_indices"], z["CeRem_indptr"]),
                              shape=tuple(int(v) for v in z["CeRem_shape"]))
        wL = float(z["w_celoc"]) if w_celoc is None else float(w_celoc)
        wR = float(z["w_cerem"]) if w_cerem is None else float(w_cerem)
        return CeLoc, CeRem, wL, wR

    return _recon


def _append_manifest(run_dir: Path, manifest_rows: list[dict]):
    path = run_dir / "figure_manifest.csv"
    df = pd.DataFrame(manifest_rows, columns=FIGURE_MANIFEST_COLUMNS)
    df.to_csv(path, index=False)


# =============================================================================
# Main entry point
# =============================================================================
def run_experiment(seed: int, config: cfg.ExperimentConfig,
                   n_epochs: int | None = None, resume: bool = True,
                   out_root: str = cfg.OUT_ROOT, device=None,
                   data_root: str | None = None,
                   measure_topology_flag: bool = True,
                   save_figures: bool = True,
                   verbose: bool = True) -> RunResult:
    """Run one (seed, config) for ``n_epochs``.  Idempotent: resumes from the
    latest checkpoint under the run directory if ``resume=True``."""
    if n_epochs is None:
        n_epochs = config.num_epochs
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    deterministic_ok = seed_everything(seed)
    init_rng, config_rng = make_rngs(seed)
    alpha_H, tau_H = resolve_alpha_tau(config, config_rng)

    model = build_model(seed, config, device, init_rng, alpha_H, tau_H)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Initial network state.  Set ONCE here, before the epoch loop -- this is the
    # state entering the first image of epoch 0, and from there the trajectory is
    # unbroken for all num_epochs (§4 carry-over).  It is NOT a per-epoch reset.
    #
    # H0 from init_rng (as before).  E0/I0 now match param_explore exactly:
    # U(0, ic_e_max) / U(0, ic_i_max) from a separate generator seeded
    # SeedSequence([seed, 0xE0]), so (a) a training run and a probe at the same
    # seed start from the SAME E0/I0, and (b) changing the IC cannot shift the
    # init_rng stream position, leaving network parameters bit-identical to a run
    # at any other IC.  The former E=I=0 start was a different basin for a
    # bistable network, and because state carries over it conditioned the whole
    # ~13,000 s trajectory rather than a 1 s transient.  O0 stays zero, as in the
    # probe.
    H0 = init_rng.uniform(cfg.H_INIT_MIN, cfg.H_INIT_MAX, size=model.N).astype(np.float32)
    ic_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xE0]))
    E0 = (ic_rng.random(model.N) * config.ic_e_max).astype(np.float32)
    I0 = (ic_rng.random(model.N) * config.ic_i_max).astype(np.float32)
    state = {
        "E": torch.tensor(E0, device=device),
        "I": torch.tensor(I0, device=device),
        "O": torch.zeros(model.num_output, device=device),
        "H": torch.tensor(H0, device=device),
        "global_train_time_s": 0.0,
    }

    train_loader, g, fixed_eval_loader, fixed_eval_batch = build_data(seed, config, device, data_root)

    run_id = f"seed{seed:03d}__{config.name}"
    run_dir = Path(out_root) / run_id
    ckpt_dir = run_dir / "checkpoints"
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)

    boundary = cfg.get_boundary(config.name, alpha_H)

    # Run provenance (2026-08-14).  The exploration path has always written this
    # via param_persist.save_bundle; the training path wrote no commit, no
    # environment and no config snapshot, which for a 25 h run is a real
    # reproducibility gap.  Written at run START (not at DONE) so it exists even
    # for a run that is killed mid-flight, and re-written on resume.  Reuses
    # param_persist's helpers rather than reimplementing them, so the two paths
    # cannot drift.
    try:
        import param_persist as _pp
        (run_dir / "run_params.json").write_text(json.dumps(dict(
            run_id=run_id, seed=seed, config_name=config.name,
            alpha_H=float(alpha_H), tau_H=float(tau_H),
            device=str(device), deterministic_ok=bool(deterministic_ok),
            git_commit=_pp._git_commit(), env=_pp._env_provenance(),
            derived_config=_pp.derived_config(cfg),
            config=asdict(config),
            boundary=dict(slope=float(boundary[0]), intercept=float(boundary[1]),
                          provenance=getattr(cfg, "BOUNDARY_FIT_PROVENANCE", "")),
            initial_condition=dict(
                ic_e_max=float(config.ic_e_max), ic_i_max=float(config.ic_i_max),
                h_init_min=float(cfg.H_INIT_MIN), h_init_max=float(cfg.H_INIT_MAX),
                ic_rng_seed_sequence=[int(seed), 0xE0], o_init="zeros"),
            eval_branched=True,
        ), indent=2, default=str))
    except Exception as e:
        print(f"  [warn] run_params.json not written: {e}")

    rows: list[dict] = []
    manifest_rows: list[dict] = []
    trigger_state = {"train_loss_hist": [], "train_acc_hist": [],
                     "val_loss_hist": [], "val_acc_hist": [],
                     "last_topo_epoch": -(10 ** 9),
                     "prev_on_seizure_side": None,
                     "prev_CeLoc_l2": None, "prev_CeRem_l2": None}
    start_epoch = 0

    # ---- Resume ----
    done_path = run_dir / "DONE"
    if resume and done_path.exists():
        try:
            prev_done = json.loads(done_path.read_text())
        except Exception:
            prev_done = {}
        if int(prev_done.get("n_epochs", n_epochs)) >= n_epochs:
            if verbose:
                print(f"[{run_id}] already DONE ({prev_done.get('n_epochs', '?')} epochs) "
                      f">= requested {n_epochs} — nothing to do.")
            return RunResult(run_id, seed, config.name, str(run_dir), str(device),
                             deterministic_ok, alpha_H, tau_H, n_epochs, n_epochs, True,
                             str(run_dir / "metrics.parquet"))
        # Extending the run to more epochs: drop the stale sentinel and continue
        # from the latest checkpoint below.
        done_path.unlink()

    if resume:
        ckpt = _load_latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            assert ckpt["run_id"] == run_id, f"run_id mismatch: {ckpt['run_id']} != {run_id}"
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            rs = ckpt["rng_states"]
            torch.random.set_rng_state(rs["torch_cpu"])
            np.random.set_state(rs["numpy"])
            random.setstate(rs["python"])
            if "torch_cuda" in rs and torch.cuda.is_available():
                torch.cuda.set_rng_state(rs["torch_cuda"])
            g.set_state(ckpt["dl_generator_state"])
            ns = ckpt["network_state"]
            state["E"] = ns["E"].to(device)
            state["I"] = ns["I"].to(device)
            state["O"] = ns["O"].to(device)
            state["H"] = ns["H"].to(device)
            state["global_train_time_s"] = ns["global_train_time_s"]
            trigger_state = ckpt["trigger_state"]
            start_epoch = ckpt["epoch"] + 1
            # Reload metrics rows up to the checkpointed epoch (drop any orphan row
            # flushed after the last checkpoint but before the crash).
            mp = run_dir / "metrics.parquet"
            mc = run_dir / "metrics.csv"
            if mp.exists():
                prior = pd.read_parquet(mp)
            elif mc.exists():
                prior = pd.read_csv(mc)
            else:
                prior = None
            if prior is not None:
                prior = prior[prior["epoch"] <= ckpt["epoch"]]
                rows = prior.to_dict("records")
            fm = run_dir / "figure_manifest.csv"
            if fm.exists():
                man = pd.read_csv(fm)
                man = man[man["epoch"] <= ckpt["epoch"]]
                manifest_rows = man.to_dict("records")
            if verbose:
                print(f"[{run_id}] resuming from epoch {start_epoch}")

    if verbose:
        print(f"[{run_id}] device={device} alpha_H={alpha_H:.4f} tau_H={tau_H:.4f} "
              f"deterministic_ok={deterministic_ok}")

    # ---- Epoch loop ----
    for epoch in range(start_epoch, n_epochs):
        # Capture the E/I activity window only on epochs that will save figures —
        # the predictable triggers (epoch 0 and scheduled).  Sign-flip-triggered
        # epochs still get topology + metrics, just no figures.  mean_E_activity
        # is computed every epoch regardless (GPU scalar accumulator).
        # Capture cadence, arrays, and figures are three INDEPENDENT decisions.
        # They used to be one expression --
        #     will_capture = save_figures and (epoch == 0 or (epoch+1) % topo_track_interval == 0)
        # -- which had two consequences worth not reintroducing: save_figures=False
        # silently suppressed the ARRAYS too (the scientific record depended on the
        # rendering of it), and the cadence was welded to the topology schedule, so
        # capturing every epoch also forced topology every epoch.
        capture_now = (epoch == 0 or (epoch + 1) % config.capture_interval == 0)
        save_arrays_now = bool(config.save_activity_arrays and capture_now)
        save_figs_now = bool(save_figures and capture_now)
        # The activity buffer is needed if EITHER consumer wants it.
        will_capture = save_arrays_now or save_figs_now
        train_loss, train_acc, E_window, I_window, t_window, mean_E_activity = _train_one_epoch(
            model, train_loader, optimizer, criterion, config, state, capture_activity=will_capture
        )
        trigger_state["train_loss_hist"].append(float(train_loss))
        trigger_state["train_acc_hist"].append(float(100.0 * train_acc))

        summaries = _param_summaries(model)

        # Clean-gradient probe at current params
        grad = clean_gradient_probe(model, criterion, fixed_eval_batch, config, boundary)

        # Validation on the fixed eval subset, BRANCHED off the carried trajectory
        # (2026-08-14).  Eval starts from the live post-training state, so it
        # measures the network as training left it -- but its end state is
        # DISCARDED, so its 256 held-out images (64 s of simulated time, as much as
        # the epoch's training) do not condition what epoch N+1 trains from.  The
        # training trajectory is therefore continuous train-to-train, which is what
        # §4 carry-over means.  This also reconciles the state with
        # global_train_time_s, which only ever advanced during training: the saved
        # activity time axis and the actual dynamics now agree.
        val_loss, val_acc, _, _, _, _, _, _, _, _ = M.evaluate(
            model, fixed_eval_loader, criterion,
            dt=config.dt, image_duration=config.image_duration,
            output_average_window=config.output_average_window,
            max_images=config.eval_images, verbose=False,
            E_init=state["E"], I_init=state["I"], O_init=state["O"], H_init=state["H"],
            pulsed_stimuli=config.pulsed_stimuli,
            sequential_row_input=config.sequential_row_input,
            row_duration=config.row_duration if config.sequential_row_input else None,
            mnist_input_scale=config.mnist_input_scale,
        )
        trigger_state["val_loss_hist"].append(float(val_loss))
        trigger_state["val_acc_hist"].append(float(100.0 * val_acc))

        # Boundary bookkeeping
        sd = float(signed_distance_to_boundary(summaries["bias_e"], summaries["bias_i"], boundary))
        on_side = bool(sd > 0)
        crossed = bool(trigger_state["prev_on_seizure_side"] is not None
                       and on_side != trigger_state["prev_on_seizure_side"])
        trigger_state["prev_on_seizure_side"] = on_side

        # Deltas (null at epoch 0)
        def delta(curr, hist_key):
            h = trigger_state[hist_key]
            return float(h[-1] - h[-2]) if len(h) >= 2 else None
        d_train_loss = delta(train_loss, "train_loss_hist")
        d_train_acc = delta(train_acc, "train_acc_hist")
        d_val_loss = delta(val_loss, "val_loss_hist")
        d_val_acc = delta(val_acc, "val_acc_hist")
        d_CeLoc_l2 = (None if trigger_state["prev_CeLoc_l2"] is None
                      else float(summaries["CeLoc_l2"] - trigger_state["prev_CeLoc_l2"]))
        d_CeRem_l2 = (None if trigger_state["prev_CeRem_l2"] is None
                      else float(summaries["CeRem_l2"] - trigger_state["prev_CeRem_l2"]))
        trigger_state["prev_CeLoc_l2"] = summaries["CeLoc_l2"]
        trigger_state["prev_CeRem_l2"] = summaries["CeRem_l2"]

        # Topology trigger
        measure, reason = topology_trigger(
            epoch, trigger_state["train_loss_hist"], trigger_state["train_acc_hist"],
            trigger_state["last_topo_epoch"], config)

        topo = {k: None for k in [
            "domain_n_saddle", "domain_avg_persistence", "domain_max_persistence",
            "domain_loss_min", "domain_loss_max", "domain_loss_range", "domain_loss_std",
            "hessian_n_saddle", "hessian_avg_persistence", "hessian_max_persistence",
            "hessian_eig1", "hessian_eig2"]}
        topology_measured = False
        if measure and measure_topology_flag:
            import wc_topology as T
            topo = T.measure_topology(
                model, criterion, fixed_eval_batch, device,
                seq_input=config.sequential_row_input, img_dur=config.image_duration,
                dt_sim=config.dt, scale=config.mnist_input_scale,
                grid_size=cfg.TOPO_GRID_SIZE, perturbation_range=cfg.TOPO_PERTURBATION_RANGE,
                num_samples=cfg.TOPO_NUM_SAMPLES, hessian_num_samples=cfg.TOPO_HESSIAN_NUM_SAMPLES,
                lanczos_iterations=cfg.TOPO_LANCZOS_ITERATIONS, verbose=verbose,
                # first measurement of this run → full-precision diagnostic
                # (last_topo_epoch starts at a large negative sentinel, so < 0
                #  means no topology has been measured yet).
                first_measurement=(trigger_state["last_topo_epoch"] < 0))
            topology_measured = True
            trigger_state["last_topo_epoch"] = epoch
        elif measure and not measure_topology_flag:
            # Trigger fired but measurement disabled (e.g. local smoke run)
            trigger_state["last_topo_epoch"] = epoch

        # Activity arrays on captured epochs (2026-08-14).  Written BEFORE the
        # figures so a render failure cannot cost the arrays -- they are the
        # scientific record, the PNGs are only a view of them.
        #
        # Written as a BUNDLE-SHAPED directory (arrays.npz + manifest.json) so
        # labeling_toolkit.analytic_features.extract() reads a training epoch with
        # exactly the code it uses on an exploration probe: same array keys, same
        # dtypes, and the window length (TRAIN_HEATMAP_WINDOW) already equals the
        # probe's RUN_SECONDS while _train_one_epoch records every step, so the
        # sampling matches too.
        #
        # The TRAINED connectivity is saved alongside.  This is load-bearing:
        # param_render._reconstruct_connectivity rebuilds connectivity from the
        # SEED, which gives the INITIAL weights -- correct for a probe (which never
        # trains) but wrong for epoch N of a training run, where CeLoc/CeRem are
        # exactly what learning has changed.  Scoring a trained epoch's LFP against
        # its initial connectivity would silently mis-measure the thing the study
        # is about.  Use wc_experiment.epoch_reconstructor(epoch_dir) as the
        # `reconstruct_connectivity` argument to extract().
        if save_arrays_now and E_window.size:
            ep_dir = run_dir / "arrays" / f"epoch_{epoch:04d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            # Stimulus onsets on the same absolute axis as t_window: images are
            # presented back-to-back with no gap, so an onset is a sample where t
            # crosses a multiple of image_duration.  Note the leading False: the
            # window's first sample is NOT an onset unless it happens to land on a
            # boundary -- prepending True instead would inject a phantom event at
            # t[0] in every bundle (and one off by up to a full image_duration for a
            # window that starts mid-image).
            _k = np.floor(t_window / config.image_duration + 1e-9)
            _onsets = t_window[np.concatenate(([False], np.diff(_k) > 0))]
            try:
                np.savez_compressed(
                    ep_dir / "arrays.npz",
                    E_traj=E_window.astype(np.float32),
                    I_traj=I_window.astype(np.float32),
                    times=t_window.astype(np.float64),
                    stim_onsets=_onsets.astype(np.float64),
                    # trained, scaled connectivity as CSR triplets + LFP weights
                    CeLoc_data=model.CeLoc_values.detach().cpu().numpy(),
                    CeLoc_indices=model.CeLoc_col_indices.detach().cpu().numpy(),
                    CeLoc_indptr=model.CeLoc_crow_indices.detach().cpu().numpy(),
                    CeLoc_shape=np.asarray(tuple(model.CeLoc_shape), dtype=np.int64),
                    CeRem_data=model.CeRem_values.detach().cpu().numpy(),
                    CeRem_indices=model.CeRem_col_indices.detach().cpu().numpy(),
                    CeRem_indptr=model.CeRem_crow_indices.detach().cpu().numpy(),
                    CeRem_shape=np.asarray(tuple(model.CeRem_shape), dtype=np.int64),
                    w_celoc=np.float64(model.w_celoc),
                    w_cerem=np.float64(model.w_cerem),
                )
                # manifest.json LAST -- extract() reads it, so its presence is the
                # completeness marker for the bundle (same discipline as
                # param_persist.save_bundle).
                (ep_dir / "manifest.json").write_text(json.dumps(dict(
                    run_id=run_id, seed=seed, config_name=config.name,
                    arm=config.arm, epoch=epoch,
                    alpha_H=float(alpha_H), tau_H=float(tau_H),
                    connectivity="trained", trained_connectivity_in_npz=True,
                ), indent=2))
                manifest_rows.append(dict(
                    run_id=run_id, seed=seed, config_name=config.name,
                    epoch=epoch, fig_type="arrays_bundle", path=str(ep_dir)))
            except Exception as e:
                print(f"  [warn] activity-array save failed at epoch {epoch}: {e}")

        # Figures on captured (epoch 0 + scheduled) epochs (§12)
        if save_figs_now and E_window.size:
            import wc_figures as figs
            import scipy.sparse as _sp
            epoch_fig_dir = fig_dir / f"epoch_{epoch:04d}"

            def _model_csr(crow, col, vals, shape):
                # Current (trained) excitatory connectivity as SciPy CSR for the LFP proxy.
                return _sp.csr_matrix(
                    (vals.detach().cpu().numpy(),
                     col.detach().cpu().numpy(),
                     crow.detach().cpu().numpy()),
                    shape=tuple(shape))

            CeLoc_sp = _model_csr(model.CeLoc_crow_indices, model.CeLoc_col_indices,
                                  model.CeLoc_values, model.CeLoc_shape)
            CeRem_sp = _model_csr(model.CeRem_crow_indices, model.CeRem_col_indices,
                                  model.CeRem_values, model.CeRem_shape)
            try:
                produced = figs.save_all_figures(
                    str(epoch_fig_dir), E_window, I_window, t_window, config.dt,
                    CeLoc=CeLoc_sp, CeRem=CeRem_sp,
                    image_duration=config.image_duration,
                    pulsed=config.pulsed_stimuli)
                for fig_type, path in produced:
                    manifest_rows.append(dict(
                        run_id=run_id, seed=seed, config_name=config.name,
                        epoch=epoch, fig_type=fig_type, path=str(path)))
            except Exception as e:
                print(f"  [warn] figure generation failed at epoch {epoch}: {e}")

        row = dict(
            run_id=run_id, seed=seed, config_name=config.name, input_mode=config.input_mode,
            alpha_H=alpha_H, tau_H=tau_H, device=str(device),
            deterministic_ok=deterministic_ok, epoch=epoch,
            train_loss=float(train_loss), train_acc=float(train_acc),
            val_loss=float(val_loss), val_acc=float(val_acc),
            d_train_loss=d_train_loss, d_train_acc=d_train_acc,
            d_val_loss=d_val_loss, d_val_acc=d_val_acc,
            **{k: summaries[k] for k in summaries},
            **grad,
            d_CeLoc_l2=d_CeLoc_l2, d_CeRem_l2=d_CeRem_l2,
            boundary_slope=float(boundary[0]), boundary_intercept=float(boundary[1]),
            signed_dist_to_boundary=sd, on_seizure_side=on_side, crossed_this_epoch=crossed,
            topology_measured=topology_measured, trigger_reason=(reason if measure else None),
            **topo,
            H_final_mean=float(state["H"].mean().item()), H_final_std=float(state["H"].std().item()),
            mean_E_activity=mean_E_activity,
        )
        rows.append(row)

        # Flush metrics BEFORE checkpoint (§11) so the log never leads the
        # resumable state.
        _flush_metrics(rows, run_dir)
        if manifest_rows:
            _append_manifest(run_dir, manifest_rows)
        # On a deferred-topology run (measure_topology_flag=False), archive the
        # checkpoint at trigger epochs so the later topology pass can reload it.
        retain_ckpt = bool(measure and not measure_topology_flag)
        _save_checkpoint(ckpt_dir, epoch, model, optimizer, g, state,
                         trigger_state, run_id, seed, config.name,
                         retain=retain_ckpt)

        if verbose:
            print(f"[{run_id}] epoch {epoch:3d}/{n_epochs}  "
                  f"train_loss={train_loss:.4f} acc={100*train_acc:.1f}%  "
                  f"val_loss={val_loss:.4f} acc={100*val_acc:.1f}%  "
                  f"bE={summaries['bias_e']:.3f} bI={summaries['bias_i']:.3f}  "
                  f"sd={sd:+.3f}{' TOPO' if topology_measured else ''}")

    # ---- DONE sentinel (§9) ----
    _flush_metrics(rows, run_dir)
    done_path.write_text(json.dumps({
        "run_id": run_id, "seed": seed, "config_name": config.name,
        "n_epochs": n_epochs, "device": str(device),
        "deterministic_ok": deterministic_ok, "alpha_H": alpha_H, "tau_H": tau_H,
    }, indent=2))
    if verbose:
        print(f"[{run_id}] DONE ({n_epochs} epochs).")

    return RunResult(run_id, seed, config.name, str(run_dir), str(device),
                     deterministic_ok, alpha_H, tau_H, n_epochs, n_epochs, True,
                     str(run_dir / "metrics.parquet"))


# =============================================================================
# Deferred topology pass (§8) — measure topology after a topology-off training
# run, by reloading the checkpoints archived at trigger epochs.
# =============================================================================
TOPO_METRIC_COLUMNS = [
    "domain_n_saddle", "domain_avg_persistence", "domain_max_persistence",
    "domain_loss_min", "domain_loss_max", "domain_loss_range", "domain_loss_std",
    "hessian_n_saddle", "hessian_avg_persistence", "hessian_max_persistence",
    "hessian_eig1", "hessian_eig2",
]


def _merge_topology_into_metrics(run_dir: Path, measured: dict) -> None:
    """Write measured topology back into the run's metrics table: for each measured
    epoch, flip ``topology_measured`` True and fill the 12 topology_* fields.
    Column order and the CSV mirror are preserved (matches ``_flush_metrics``)."""
    mp = run_dir / "metrics.parquet"
    mc = run_dir / "metrics.csv"
    if mp.exists():
        df = pd.read_parquet(mp)
    elif mc.exists():
        df = pd.read_csv(mc)
    else:
        print(f"  [warn] {run_dir.name}: no metrics table to merge topology into")
        return
    for epoch, topo in measured.items():
        mask = df["epoch"] == epoch
        if not mask.any():
            print(f"  [warn] {run_dir.name}: no metrics row for epoch {epoch}")
            continue
        df.loc[mask, "topology_measured"] = True
        for k in TOPO_METRIC_COLUMNS:
            df.loc[mask, k] = topo.get(k)
    df = df.reindex(columns=METRIC_COLUMNS)
    try:
        df.to_parquet(mp, index=False)
    except Exception as e:
        print(f"  [warn] parquet write failed ({e}); CSV mirror still written")
    df.to_csv(mc, index=False)


def measure_topology_for_run(run_dir, device=None, data_root: str | None = None,
                             grid_size: int = cfg.TOPO_GRID_SIZE,
                             perturbation_range: float = cfg.TOPO_PERTURBATION_RANGE,
                             num_samples: int = cfg.TOPO_NUM_SAMPLES,
                             hessian_num_samples: int = cfg.TOPO_HESSIAN_NUM_SAMPLES,
                             lanczos_iterations: int = cfg.TOPO_LANCZOS_ITERATIONS,
                             config_overrides: dict | None = None,
                             verbose: bool = True) -> dict:
    """Measure loss-landscape topology for one run from its archived trigger-epoch
    checkpoints, and merge the results into the run's metrics table.

    The model is rebuilt from ``(seed, config)`` exactly as ``run_experiment`` does
    on resume — identical architecture — then each archived checkpoint's weights
    are loaded and measured.  ``config_overrides`` (dict applied via
    ``dataclasses.replace``) is primarily for tests / quick low-fidelity
    measurement; production omits it so topology uses the frozen study settings.
    """
    import dataclasses
    import wc_topology as T

    run_dir = Path(run_dir)
    topo_dir = run_dir / "checkpoints" / "topology"
    ckpts = sorted(topo_dir.glob("topo_checkpoint_epoch_*.pkl")) if topo_dir.exists() else []
    if not ckpts:
        if verbose:
            print(f"[{run_dir.name}] no archived topology checkpoints — skipping")
        return {"run_dir": str(run_dir), "measured_epochs": [], "n_measured": 0}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Identity from the first archived checkpoint.
    with open(ckpts[0], "rb") as f:
        head = pickle.load(f)
    seed = int(head["seed"])
    config = cfg.make_config(head["config_name"])
    if config_overrides:
        config = dataclasses.replace(config, **config_overrides)

    # Rebuild the model deterministically (same path as run_experiment setup).
    seed_everything(seed)
    init_rng, config_rng = make_rngs(seed)
    alpha_H, tau_H = resolve_alpha_tau(config, config_rng)
    model = build_model(seed, config, device, init_rng, alpha_H, tau_H)
    criterion = nn.CrossEntropyLoss()
    _, _, _, fixed_eval_batch = build_data(seed, config, device, data_root)

    measured: dict[int, dict] = {}
    for i, cp in enumerate(ckpts):
        with open(cp, "rb") as f:
            ck = pickle.load(f)
        epoch = int(ck["epoch"])
        model.load_state_dict(ck["model_state_dict"])
        topo = T.measure_topology(
            model, criterion, fixed_eval_batch, device,
            seq_input=config.sequential_row_input, img_dur=config.image_duration,
            dt_sim=config.dt, scale=config.mnist_input_scale,
            grid_size=grid_size, perturbation_range=perturbation_range,
            num_samples=num_samples, hessian_num_samples=hessian_num_samples,
            lanczos_iterations=lanczos_iterations, verbose=verbose,
            first_measurement=(i == 0))
        measured[epoch] = topo
        if verbose:
            print(f"[{run_dir.name}] topology measured @ epoch {epoch}")

    _merge_topology_into_metrics(run_dir, measured)
    return {"run_dir": str(run_dir), "seed": seed, "config_name": config.name,
            "measured_epochs": sorted(measured), "n_measured": len(measured)}
