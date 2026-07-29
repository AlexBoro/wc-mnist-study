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


def _reconstruct_connectivity(seed: int, config_name: str, arm: str = "",
                              w_celoc=None, w_cerem=None):
    """Rebuild the model's *scaled* CeLoc/CeRem as SciPy CSR, matching exactly
    what the longitudinal LFP proxy consumes.

    The harness passes the model's learnable ``CeLoc_values`` (structural
    connectivity multiplied by the per-edge ``celoc_scale`` draw), NOT the raw
    structural 1.0s.  So we rebuild the full model deterministically from
    ``seed`` + ``config`` (same init draw order the run used) and extract its
    connectivity CSR the same way ``wc_experiment`` does.
    """
    import dataclasses
    import scipy.sparse as sp
    config = {c.name: c for c in cfg.make_configs()}[config_name]
    # A pilot arm overrides the E->E coefficients. They are consumed AFTER every
    # draw from init_rng, so overriding them does not perturb draw order and the
    # rebuilt connectivity is bit-identical to the run's -- but the LFP proxy must
    # still be given the arm's coefficients, or the figure would be computed with
    # different weights than the model it depicts.
    if arm or w_celoc is not None or w_cerem is not None:
        w = cfg.PILOT_COEFF_ARMS.get(arm) if arm else None
        config = dataclasses.replace(
            config,
            w_celoc=float(w_celoc if w_celoc is not None else (w if w is not None else config.w_celoc)),
            w_cerem=float(w_cerem if w_cerem is not None else (w if w is not None else config.w_cerem)),
            arm=str(arm))
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
    return CeLoc_csr, CeRem_csr, float(model.w_celoc), float(model.w_cerem)


def render_run(fig_dir, E_traj, I_traj, times, seed, config_name,
               image_duration=None, dpi=150, arm="", span=None,
               return_stats=False):
    """Render heatmap_E.png + lfp.png into ``fig_dir`` from arrays.

    ``arm`` selects the pilot coefficient variant, which the LFP proxy needs: the
    proxy IS the model's excitatory update line restricted to the electrode rows,
    so rendering with default coefficients would depict a different network than
    the one that was simulated.

    ``return_stats=True`` returns the LFP peak-deflection statistics, which is how
    cfg.LFP_SPAN gets recalibrated from the pilot rather than set by eye.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if image_duration is None:
        image_duration = cfg.IMAGE_DURATION

    E = np.asarray(E_traj, dtype=np.float32)
    I = np.asarray(I_traj, dtype=np.float32)
    t = np.asarray(times, dtype=np.float32)

    # E-activity heatmap (fixed 0-1 viridis)
    tag = f"{config_name} seed{seed:03d}" + (f" [{arm}]" if arm else "")
    hm_path = fig_dir / "heatmap_E.png"
    F.save_heatmap(E, t, str(hm_path),
                   title=f"E activity — {tag} (last {t[-1]-t[0]:.1f}s)",
                   cbar_label="E activity", image_duration=image_duration)

    # LFP proxy (28 ring electrodes, fixed span cfg.LFP_SPAN) — needs connectivity
    CeLoc_csr, CeRem_csr, wL, wR = _reconstruct_connectivity(seed, config_name, arm=arm)
    lfp_path = fig_dir / "lfp.png"
    stats = F.save_lfp(E, t, str(lfp_path), dt=cfg.DT, I_window=I,
                       CeLoc=CeLoc_csr, CeRem=CeRem_csr,
                       image_duration=image_duration,
                       w_celoc=wL, w_cerem=wR, span=span,
                       return_stats=return_stats)
    if return_stats:
        return dict(fig_dir=str(fig_dir), heatmap=str(hm_path),
                    w_celoc=wL, w_cerem=wR, arm=arm, **stats)
    return str(hm_path), str(lfp_path)


def render_from_result(out_root, result, image_duration=None, arm="",
                       span=None, return_stats=False):
    """render_fn adapter for param_persist.run_sweep(render_fn=...)."""
    from param_persist import run_dir
    arm = arm or str(result.params.get("arm", "") or "")
    d = run_dir(out_root, result.config_name, result.seed, arm)
    return render_run(d / "figures", result.E_traj, result.I_traj, result.times,
                      result.seed, result.config_name,
                      image_duration=image_duration, arm=arm, span=span,
                      return_stats=return_stats)


def render_from_bundle(out_root, config_name, seed, image_duration=None, arm="",
                       span=None, return_stats=False):
    """Re-render figures from a saved bundle (post-hoc).

    Coefficients come from the bundle's own params.json, not from the current
    module defaults -- so re-rendering an old bundle after a code change depicts
    the network that was actually simulated.
    """
    from param_persist import load_bundle, run_dir
    arrays, params, _manifest = load_bundle(out_root, config_name, seed, arm)
    arm = arm or str(params.get("arm", "") or "")
    d = run_dir(out_root, config_name, seed, arm)
    return render_run(d / "figures", arrays["E_traj"], arrays["I_traj"],
                      arrays["times"], seed, config_name,
                      image_duration=image_duration, arm=arm, span=span,
                      return_stats=return_stats)
