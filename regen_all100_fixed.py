"""regen_all100_fixed.py — full 100-epoch validation on the FIXED LFP axis.

For every (run, epoch) in regen100_manifest.json:
  1. rebuild model+config from the topo checkpoint, replay the fixed MNIST batch,
     save_all_figures with the PATCHED save_lfp (fixed ylim span-150),
  2. extract_features on the fresh heatmap_E + lfp renders (includes the new
     sync_spike_extent feature),
  3. classify() with the patched decision table (LFP-correlate gate + FIS/FSW
     sync-extent split).
Writes taxonomy_validation_100_fixed.csv (same schema as the prior CSV plus
sync_spike_extent) for the report rebuild.

Faithful path only: figures are re-simulated from frozen checkpoints, never
reused from the downscaled archived thumbnails. Renders go to a SEPARATE tree
(all100_fixed_figs/) so archived Drive figures are untouched.
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent
RUN_ROOT = REPO / "WC_MNIST_Longitudinal-20260713T011621Z-2-001" / "WC_MNIST_Longitudinal"
sys.path.insert(0, str(REPO / "labeling_toolkit"))

import wc_config as cfg
import wc_figures as figs
from wc_experiment import (seed_everything, make_rngs, resolve_alpha_tau,
                           build_model, build_data)
from regenerate_figures import _load_ckpt, _model_csr, _simulate_window
import features as F
import classifier as C

OUT_ROOT = HERE / "all100_fixed_figs"
WINDOW_S = 10.0
DATA_ROOT = str(HERE / "mnist_data")

# per-run cached (model, config, eval batch) — build_data/build_model are the
# expensive parts and are identical across epochs of the same run/seed.
_RUN_CACHE = {}


def _run_ctx(run_name, sample_cp):
    if run_name in _RUN_CACHE:
        return _RUN_CACHE[run_name]
    device = torch.device("cpu")
    head = _load_ckpt(sample_cp, device)
    seed = int(head["seed"]); config = cfg.make_config(head["config_name"])
    seed_everything(seed)
    init_rng, config_rng = make_rngs(seed)
    alpha_H, tau_H = resolve_alpha_tau(config, config_rng)
    model = build_model(seed, config, device, init_rng, alpha_H, tau_H); model.eval()
    _, _, _, fixed_eval_batch = build_data(seed, config, device, DATA_ROOT)
    images = [img for img, _ in fixed_eval_batch]
    n_images = max(1, round(WINDOW_S / config.image_duration))
    ctx = dict(model=model, config=config, images=images, n_images=n_images, device=device)
    _RUN_CACHE[run_name] = ctx
    return ctx


def regen_one(run_name, epoch):
    ep_dir_existing = OUT_ROOT / run_name / f"epoch_{epoch:04d}"
    if (ep_dir_existing / "lfp.png").exists() and (ep_dir_existing / "heatmap_E.png").exists():
        return ep_dir_existing            # resume: figures already rendered, skip re-simulation
    device = torch.device("cpu")
    cp = RUN_ROOT / run_name / "checkpoints" / "topology" / f"topo_checkpoint_epoch_{epoch:04d}.pkl"
    ctx = _run_ctx(run_name, cp)
    model, config = ctx["model"], ctx["config"]
    ck = _load_ckpt(cp, device)
    model.load_state_dict(ck["model_state_dict"])
    E_w, I_w, t_w = _simulate_window(model, ctx["images"], ck["network_state"],
                                     config, ctx["n_images"])
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
    import csv
    manifest = json.load(open(HERE / "regen100_manifest.json"))
    cols = ["run","epoch","new_label","new_conf","new_rule","prior_label","changed",
            "ink_frac","max_col_frac","band_score","n_complexes","max_IDI","min_IDI",
            "median_IDI","sustained_fraction","vertical_spread_fraction","S","cross_cols",
            "n_active_channels","fast_activity_duration_s","sync_spike_extent","mean_E",
            "val_acc","new_clause"]
    out = HERE / "taxonomy_validation_100_fixed.csv"
    f = open(out, "w", newline=""); w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    t0 = time.time()
    for i, m in enumerate(manifest):
        run_name, epoch = m["run"], int(m["epoch"])
        ep_dir = regen_one(run_name, epoch)
        feats = F.extract_features(str(ep_dir / "heatmap_E.png"),
                                   str(ep_dir / "lfp.png"),
                                   mean_E=m.get("mean_E"), val_acc=m.get("val_acc"))
        verdict = C.classify(feats)
        row = dict(run=run_name, epoch=epoch,
                   new_label=verdict["label_name"], new_conf=verdict["confidence"],
                   new_rule=verdict["rule"], new_clause=verdict["clause"],
                   prior_label=m.get("prior_label"))
        row.update({k: feats.get(k) for k in
                    ["ink_frac","max_col_frac","band_score","n_complexes","max_IDI",
                     "min_IDI","median_IDI","sustained_fraction","vertical_spread_fraction",
                     "S","cross_cols","n_active_channels","fast_activity_duration_s",
                     "sync_spike_extent","mean_E","val_acc"]})
        row["changed"] = "CHANGED" if row["new_label"] != m.get("prior_label") else ""
        w.writerow({k: row.get(k) for k in cols}); f.flush()
        if (i + 1) % 10 == 0:
            dt = time.time() - t0
            print(f"  [{i+1}/100] {dt:.0f}s elapsed", flush=True)
    f.close()
    print(f"saved {out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
