"""param_render.py — render per-run figures from saved arrays.

Uses the LONGITUDINAL montage (``wc_figures.save_heatmap`` / ``save_lfp``) so
the seizure classifier (``classifier.py`` + ``features.py``) applies unchanged:

  * heatmap_E.png — E-activity heatmap, viridis, fixed vmin=0/vmax=1
  * lfp.png       — filtered-LFP proxy at 28 ring-spaced electrodes, FIXED
                    y-span=150, high-pass order-4 zero-phase, cutoff 1 Hz

The LFP proxy needs the CeLoc/CeRem connectivity, which is NOT stored in the
bundle (it is reconstructable).  We rebuild it deterministically from the run's
seed via the same ``build_connectivity`` draw the run used, so the rendered LFP
is identical to what a live render would produce.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

import wc_config as cfg
import wc_model as M
import wc_figures as F
import wc_experiment as X


def _reconstruct_connectivity(seed: int, config_name: str):
    """Rebuild the model's *scaled* CeLoc/CeRem as SciPy CSR, matching exactly
    what the longitudinal LFP proxy consumes.

    The harness passes the model's learnable ``CeLoc_values`` (structural
    connectivity multiplied by the per-edge ``celoc_scale`` draw), NOT the raw
    structural 1.0s.  So we rebuild the full model deterministically from
    ``seed`` + ``config`` (same init draw order the run used) and extract its
    connectivity CSR the same way ``wc_experiment`` does.
    """
    import scipy.sparse as sp
    config = {c.name: c for c in cfg.make_configs()}[config_name]
    X.seed_everything(seed)
    init_rng, config_rng = X.make_rngs(seed)
    alpha_H, tau_H = X.resolve_alpha_tau(config, config_rng)
    model = X.build_model(seed, config, "cpu", init_rng, alpha_H, tau_H)

    def _model_csr(crow, col, vals, shape):
        return sp.csr_matrix(
            (vals.detach().cpu().numpy(),
             col.detach().cpu().numpy(),
             crow.detach().cpu().numpy()),
            shape=tuple(shape))

    CeLoc_csr = _model_csr(model.CeLoc_crow_indices, model.CeLoc_col_indices,
                           model.CeLoc_values, model.CeLoc_shape)
    CeRem_csr = _model_csr(model.CeRem_crow_indices, model.CeRem_col_indices,
                           model.CeRem_values, model.CeRem_shape)
    return CeLoc_csr, CeRem_csr


def render_run(fig_dir, E_traj, I_traj, times, seed, config_name,
               image_duration=None, dpi=150):
    """Render heatmap_E.png + lfp.png into ``fig_dir`` from arrays."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if image_duration is None:
        image_duration = cfg.IMAGE_DURATION

    E = np.asarray(E_traj, dtype=np.float32)
    I = np.asarray(I_traj, dtype=np.float32)
    t = np.asarray(times, dtype=np.float32)

    # E-activity heatmap (fixed 0-1 viridis)
    hm_path = fig_dir / "heatmap_E.png"
    F.save_heatmap(E, t, str(hm_path),
                   title=f"E activity — {config_name} seed{seed:03d} (last {t[-1]-t[0]:.1f}s)",
                   cbar_label="E activity", image_duration=image_duration)

    # LFP proxy (28 ring electrodes, fixed span 150) — needs connectivity
    CeLoc_csr, CeRem_csr = _reconstruct_connectivity(seed, config_name)
    lfp_path = fig_dir / "lfp.png"
    F.save_lfp(E, t, str(lfp_path), dt=cfg.DT, I_window=I,
               CeLoc=CeLoc_csr, CeRem=CeRem_csr, image_duration=image_duration)
    return str(hm_path), str(lfp_path)


def render_from_result(out_root, result, image_duration=None):
    """render_fn adapter for param_persist.run_sweep(render_fn=...)."""
    from param_persist import run_dir
    d = run_dir(out_root, result.config_name, result.seed)
    return render_run(d / "figures", result.E_traj, result.I_traj, result.times,
                      result.seed, result.config_name, image_duration=image_duration)


def render_from_bundle(out_root, config_name, seed, image_duration=None):
    """Re-render figures from a saved bundle (post-hoc)."""
    from param_persist import load_bundle, run_dir
    arrays, _params, _manifest = load_bundle(out_root, config_name, seed)
    d = run_dir(out_root, config_name, seed)
    return render_run(d / "figures", arrays["E_traj"], arrays["I_traj"],
                      arrays["times"], seed, config_name, image_duration=image_duration)
