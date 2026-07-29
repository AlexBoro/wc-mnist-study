"""regen_anchors_fixed.py — regenerate the 4 per-epoch figures for the six anchor
epochs using the PATCHED save_lfp (fixed ylim span-150), then re-run the detector's
crossing/FPFA metrics (labeling_toolkit.features.extract_features) on the fresh
fixed-scale renders. Writes anchors_fixed_metrics.json for comparison against the
current (autoscale-era) CSV values.

Faithful path: rebuild model+config from each topo checkpoint, replay the fixed
MNIST eval batch, save_all_figures (heatmap_E/I + lfp + lfp_focus), then extract.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent
RUN_ROOT = REPO / "WC_MNIST_Longitudinal-20260713T011621Z-2-001" / "WC_MNIST_Longitudinal"
# labeling_toolkit for the metric extractor
sys.path.insert(0, str(REPO / "labeling_toolkit"))

import wc_config as cfg
import wc_figures as figs
from wc_experiment import (seed_everything, make_rngs, resolve_alpha_tau,
                           build_model, build_data)
from regenerate_figures import _load_ckpt, _model_csr, _simulate_window
import features as F  # labeling_toolkit/features.py

ANCHORS = [
    ("seed001__sequential_noH", 47),
    ("seed000__sequential_noH", 71),
    ("seed002__sequential_noH", 51),
    ("seed002__sequential_noH", 63),
    ("seed001__sequential_H",   99),
    ("seed003__sequential_H",   27),
]

# render into a SEPARATE tree so we never overwrite the archived Drive-origin figures
OUT_ROOT = HERE / "anchors_fixed_figs"
WINDOW_S = 10.0
DATA_ROOT = str(HERE / "mnist_data")


def regen_one(run_name, epoch):
    device = torch.device("cpu")
    cp = RUN_ROOT / run_name / "checkpoints" / "topology" / f"topo_checkpoint_epoch_{epoch:04d}.pkl"
    head = _load_ckpt(cp, device)
    seed = int(head["seed"]); config = cfg.make_config(head["config_name"])
    seed_everything(seed)
    init_rng, config_rng = make_rngs(seed)
    alpha_H, tau_H = resolve_alpha_tau(config, config_rng)
    model = build_model(seed, config, device, init_rng, alpha_H, tau_H); model.eval()
    _, _, _, fixed_eval_batch = build_data(seed, config, device, DATA_ROOT)
    images = [img for img, _ in fixed_eval_batch]
    n_images = max(1, round(WINDOW_S / config.image_duration))
    ck = _load_ckpt(cp, device)
    model.load_state_dict(ck["model_state_dict"])
    E_w, I_w, t_w = _simulate_window(model, images, ck["network_state"], config, n_images)
    CeLoc = _model_csr(model.CeLoc_crow_indices, model.CeLoc_col_indices,
                       model.CeLoc_values, model.CeLoc_shape)
    CeRem = _model_csr(model.CeRem_crow_indices, model.CeRem_col_indices,
                       model.CeRem_values, model.CeRem_shape)
    ep_dir = OUT_ROOT / run_name / f"epoch_{epoch:04d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    figs.save_all_figures(str(ep_dir), E_w, I_w, t_w, config.dt,
                          CeLoc=CeLoc, CeRem=CeRem,
                          image_duration=config.image_duration,
                          pulsed=config.pulsed_stimuli)
    return ep_dir


def main():
    rows = []
    for run_name, epoch in ANCHORS:
        print(f"[regen] {run_name} ep{epoch} ...", flush=True)
        ep_dir = regen_one(run_name, epoch)
        heatmap = ep_dir / "heatmap_E.png"
        lfp = ep_dir / "lfp.png"
        feats = F.extract_features(str(heatmap), str(lfp))
        # keep the LFP-sensitive metrics plus a couple of heatmap sanity columns
        row = dict(run=run_name, epoch=epoch,
                   S=feats.get("S"), cross_cols=feats.get("cross_cols"),
                   n_active_channels=feats.get("n_active_channels"),
                   fast_activity_duration_s=feats.get("fast_activity_duration_s"),
                   ink_frac=feats.get("ink_frac"), max_col_frac=feats.get("max_col_frac"),
                   n_complexes=feats.get("n_complexes"))
        print(f"    fixed-scale: cross={row['cross_cols']} n_active={row['n_active_channels']} "
              f"fast_dur={row['fast_activity_duration_s']} S={row['S']}", flush=True)
        rows.append(row)
    with open(HERE / "anchors_fixed_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("saved anchors_fixed_metrics.json", flush=True)


if __name__ == "__main__":
    main()
