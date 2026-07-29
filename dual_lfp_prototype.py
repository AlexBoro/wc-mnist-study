"""dual_lfp_prototype.py — render the Filtered LFP proxy under the CURRENT
autoscaled y-axis vs. a FIXED y-axis, for two contrasting epochs, to decide
whether to pin ylim for constant pixel amplitude.

Faithfully reuses the harness reproduction path (regenerate_figures.py):
checkpoint -> rebuild model/config -> frozen re-simulation -> LFP proxy.
The ONLY change vs. wc_figures.save_lfp is an optional ax.set_ylim.

Output: dual_lfp_compare.png (2 rows = epochs, 2 cols = autoscale|fixed).
"""
import os, sys, functools, pickle
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wc_config as cfg
import wc_figures as figs
from wc_experiment import (seed_everything, make_rngs, resolve_alpha_tau,
                           build_model, build_data)
from regenerate_figures import _load_ckpt, _model_csr, _simulate_window

RUN_ROOT = HERE.parent / ("WC_MNIST_Longitudinal-20260713T011621Z-2-001/"
                          "WC_MNIST_Longitudinal")

# (run_dir_name, epoch, human label). First two are the requested contrast pair
# (rendered in the figure); the rest are extra seizure epochs sampled ONLY to
# calibrate a headroom-safe fixed ylim against the worst realistic swing.
TARGETS = [
    ("seed000__sequential_noH", 71, "ep71 spread-PFA seizure"),
    ("seed000__sequential_noH",  0, "ep00 untrained flat baseline"),
]
# extra ictal epochs for ylim calibration only (not plotted)
CALIB_EXTRA = [
    ("seed001__sequential_noH", 47, "ep47 sustained spike-wave"),
    ("seed002__sequential_noH", 51, "ep51 sustained ictal"),
    ("seed002__sequential_noH", 63, "ep63 sustained ictal"),
    ("seed001__sequential_H",   99, "ep99 late-train"),
]
WINDOW_S = 10.0
CUTOFF_HZ = 1.0
OFFSET_STEP = 5.0          # fixed inter-electrode spacing used by save_lfp


def lfp_from_window(E_window, I_window, t_window, CeLoc, CeRem, dt):
    """Reproduce wc_figures.save_lfp's LFP + high-pass + stacking EXACTLY."""
    N = E_window.shape[1]
    electrodes, labels = figs._ring_electrodes(N, 28)
    E_count = len(electrodes)
    self_term = E_window[:, electrodes]
    inh_term = (np.asarray(I_window, float)[:, electrodes]
                if I_window is not None and getattr(I_window, "size", 0)
                else np.zeros_like(self_term))
    loc_term = (np.asarray(CeLoc[electrodes, :] @ E_window.T).T
                if CeLoc is not None else np.zeros_like(self_term))
    rem_term = (np.asarray(CeRem[electrodes, :] @ E_window.T).T
                if CeRem is not None else np.zeros_like(self_term))
    LFP = (cfg.W_PY2PY_SELF*self_term + cfg.W_CELOC*loc_term
           + cfg.W_CEREM*rem_term + cfg.W_LFP_INH*inh_term)
    dt_eff = float(t_window[1]-t_window[0]) if len(t_window) > 1 else float(dt)
    sr = 1.0/dt_eff if dt_eff > 0 else 1.0
    filt = figs.filter_eeg(LFP, CUTOFF_HZ, sr, high_or_low="high", order=4)
    offsets = np.arange(E_count) * OFFSET_STEP
    stacked = filt + offsets[None, :]
    return stacked, offsets, labels, filt


def simulate_epoch(run_name, epoch):
    run_dir = RUN_ROOT / run_name
    cp = run_dir / "checkpoints" / "topology" / f"topo_checkpoint_epoch_{epoch:04d}.pkl"
    device = torch.device("cpu")
    head = _load_ckpt(cp, device)
    seed = int(head["seed"]); config = cfg.make_config(head["config_name"])
    seed_everything(seed)
    init_rng, config_rng = make_rngs(seed)
    alpha_H, tau_H = resolve_alpha_tau(config, config_rng)
    model = build_model(seed, config, device, init_rng, alpha_H, tau_H); model.eval()
    _, _, _, fixed_eval_batch = build_data(seed, config, device,
                                           os.environ.get("WC_DATA_ROOT"))
    images = [img for img, _l in fixed_eval_batch]
    n_images = max(1, round(WINDOW_S / config.image_duration))
    ck = _load_ckpt(cp, device)
    model.load_state_dict(ck["model_state_dict"])
    E_w, I_w, t_w = _simulate_window(model, images, ck["network_state"], config, n_images)
    CeLoc = _model_csr(model.CeLoc_crow_indices, model.CeLoc_col_indices,
                       model.CeLoc_values, model.CeLoc_shape)
    CeRem = _model_csr(model.CeRem_crow_indices, model.CeRem_col_indices,
                       model.CeRem_values, model.CeRem_shape)
    stacked, offsets, labels, filt = lfp_from_window(E_w, I_w, t_w, CeLoc, CeRem, config.dt)
    return dict(stacked=stacked, offsets=offsets, labels=labels, filt=filt,
                t=t_w, config=config, E=E_w, I=I_w)


def main():
    results = []
    for run_name, epoch, lab in TARGETS:
        print(f"[sim] {run_name} epoch {epoch} ...", flush=True)
        r = simulate_epoch(run_name, epoch); r["label"] = lab
        # diagnostics: how far does the biggest trace swing vs one channel spacing?
        span = float(r["filt"].max() - r["filt"].min())
        pk = float(np.abs(r["filt"]).max())
        r["span"], r["peak"] = span, pk
        print(f"    filt peak |x|={pk:.2f}  full span={span:.2f}  "
              f"(one channel spacing = {OFFSET_STEP})  ratio={span/OFFSET_STEP:.1f}", flush=True)
        results.append(r)

    # calibrate a headroom-safe ylim against extra ictal epochs (not plotted)
    calib = []
    for run_name, epoch, lab in CALIB_EXTRA:
        cp = (RUN_ROOT / run_name / "checkpoints" / "topology"
              / f"topo_checkpoint_epoch_{epoch:04d}.pkl")
        if not cp.exists():
            print(f"[calib] missing {run_name} ep{epoch} — skip", flush=True); continue
        print(f"[calib] {run_name} ep{epoch} ...", flush=True)
        cr = simulate_epoch(run_name, epoch); cr["label"] = lab
        cr["peak"] = float(np.abs(cr["filt"]).max())
        cr["span"] = float(cr["filt"].max() - cr["filt"].min())
        calib.append(cr)
        print(f"    filt peak |x|={np.abs(cr['filt']).max():.2f}", flush=True)

    E_count = len(results[0]["offsets"])
    LADDER_TOP = (E_count - 1) * OFFSET_STEP           # 135 (top electrode offset)
    LADDER_MID = LADDER_TOP / 2.0                      # 67.5

    # --- headroom-safe range (never clips), for reference ---
    allres = results + calib
    gmin = min(float(r["stacked"].min()) for r in allres)
    gmax = max(float(r["stacked"].max()) for r in allres)
    pad = 0.08 * (gmax - gmin)
    SAFE_LO = np.floor((gmin - pad) / 5) * 5
    SAFE_HI = np.ceil((gmax + pad) / 5) * 5

    # --- requested fixed SPAN = 150, centered on ladder midpoint ---
    SPAN = 150.0
    FIXED_LO = LADDER_MID - SPAN/2.0                   # -7.5
    FIXED_HI = LADDER_MID + SPAN/2.0                   # 142.5
    print(f"[ylim] headroom-safe would be [{SAFE_LO:g},{SAFE_HI:g}] (span {SAFE_HI-SAFE_LO:g}); "
          f"using requested fixed span {SPAN:g} -> [{FIXED_LO:g}, {FIXED_HI:g}]", flush=True)

    def clip_frac(stacked):
        out = (stacked < FIXED_LO) | (stacked > FIXED_HI)
        return 100.0 * out.mean()

    fig, axes = plt.subplots(len(results), 2, figsize=(15, 5*len(results)))
    for i, r in enumerate(results):
        smin, smax = float(r["stacked"].min()), float(r["stacked"].max())
        cf = clip_frac(r["stacked"])
        for j, (mode, title) in enumerate(
                [("auto", "CURRENT (autoscaled y — pixel scale drifts)"),
                 ("fixed", f"FIXED span 150: y ∈ [{FIXED_LO:g}, {FIXED_HI:g}]")]):
            ax = axes[i, j]
            for ch in range(E_count):
                ax.plot(r["t"], r["stacked"][:, ch], lw=0.7)
            if mode == "fixed":
                ax.set_ylim(FIXED_LO, FIXED_HI)
            ax.set_yticks(r["offsets"]); ax.set_yticklabels(r["labels"], fontsize=5)
            ax.set_xlabel("time (s)"); ax.set_ylabel("Ring electrode (stacked)")
            lo, hi = ax.get_ylim()
            flag = (f"  ⚠{cf:.2f}% samples clipped" if (mode == "fixed" and cf > 0) else
                    ("  (no clipping)" if mode == "fixed" else ""))
            ax.set_title(f"{r['label']}\n{title}   (ylim span={hi-lo:.0f}){flag}", fontsize=10)
    fig.suptitle("Filtered LFP proxy — autoscaled (current) vs FIXED span-150 y-axis   "
                 "[fixed span ⇒ identical pixels-per-signal-unit on every epoch]",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig("dual_lfp_compare.png", dpi=140, bbox_inches="tight")
    print("saved dual_lfp_compare.png", flush=True)

    with open("dual_lfp_diag.txt", "w") as f:
        f.write(f"requested fixed span=150 ylim = [{FIXED_LO:g}, {FIXED_HI:g}]  "
                f"(ladder top offset={LADDER_TOP:g}, mid={LADDER_MID:g})\n")
        f.write(f"headroom-safe (never clips) ylim = [{SAFE_LO:g}, {SAFE_HI:g}]  "
                f"span={SAFE_HI-SAFE_LO:g}\n")
        for r in allres:
            smin, smax = float(r["stacked"].min()), float(r["stacked"].max())
            f.write(f"{r['label']}: filt_peak_abs={r['peak']:.3f} filt_span={r['span']:.3f} "
                    f"| stacked[{smin:.1f},{smax:.1f}] "
                    f"span150_clip%={clip_frac(r['stacked']):.3f} "
                    f"top_peak_over_142.5={max(0.0, smax-FIXED_HI):.1f}\n")
    print("saved dual_lfp_diag.txt", flush=True)


if __name__ == "__main__":
    main()
