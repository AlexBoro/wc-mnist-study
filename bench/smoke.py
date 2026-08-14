import os, sys, json, time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,"~/dev/wc-mnist-study")
os.environ.setdefault("WC_DATA_ROOT","~/dev/wc-mnist-study/mnist_data")
import torch; torch.set_num_threads(1)
import numpy as np, dataclasses
import wc_config as cfg, wc_experiment as X
from pathlib import Path

OUT = Path("bench/smoke_out"); 
import shutil; shutil.rmtree(OUT, ignore_errors=True)
C = [c for c in cfg.make_configs() if c.name=="sequential_H"][0]
# tiny: 2 epochs, 6 train images, 4 eval images, topology off, figures on (drives capture)
C = dataclasses.replace(C, num_epochs=2, images_per_epoch=6, eval_images=4, topo_track_interval=1)
t0=time.time()
X.run_experiment(seed=7, config=C, out_root=str(OUT), device=torch.device("cpu"),
                 data_root=os.environ["WC_DATA_ROOT"], save_figures=True,
                 measure_topology_flag=False, resume=False)
print("wall %.1fs" % (time.time()-t0))
d = OUT/"seed007__sequential_H"
print("=== files ===")
for p in sorted(d.rglob("*")):
    if p.is_file(): print("  ", p.relative_to(d), p.stat().st_size)
print("=== run_params.json keys ===")
rp = json.loads((d/"run_params.json").read_text())
print("  ", sorted(rp.keys()))
print("   git_commit:", rp["git_commit"][:12], "| eval_branched:", rp["eval_branched"])
print("   boundary:", rp["boundary"])
print("   initial_condition:", rp["initial_condition"])
print("=== arrays ===")
for ap in sorted((d/"arrays").glob("*.npz")):
    z = np.load(ap)
    print("  ", ap.name, {k: (z[k].shape, str(z[k].dtype)) for k in z.files})
    print("      t range %.4f..%.4f | onsets %s" % (z["times"][0], z["times"][-1], z["stim_onsets"][:5]))
print("=== metrics ===")
import pandas as pd
m = pd.read_parquet(d/"metrics.parquet")
print(m[["epoch","train_loss","train_acc","val_loss","val_acc","bias_e","bias_i",
         "signed_dist_to_boundary","on_seizure_side","grad_angle_to_boundary_deg"]].to_string(index=False))

print("=== toolkit reads a training epoch ===")
import labeling_toolkit.analytic_features as AF, wc_figures as WF
import param_render as PR
for ep in ["epoch_0000","epoch_0001"]:
    ed = d/"arrays"/ep
    feats = AF.extract(ed, X.epoch_reconstructor(ed), cfg, WF)
    keys = sorted(feats.keys()) if isinstance(feats, dict) else type(feats)
    print(" ", ep, "->", {k: (round(v,4) if isinstance(v,float) else v) for k,v in list(feats.items())[:6]} if isinstance(feats,dict) else keys)
# does using the SEED-based (initial) connectivity give a different answer?
ed = d/"arrays"/"epoch_0001"
f_trained = AF.extract(ed, X.epoch_reconstructor(ed), cfg, WF)
f_initial = AF.extract(ed, PR._reconstruct_connectivity, cfg, WF)
if isinstance(f_trained, dict):
    diff = {k: (f_trained[k], f_initial.get(k)) for k in f_trained
            if isinstance(f_trained[k], float) and f_initial.get(k) != f_trained[k]}
    print("  features differing trained-vs-initial connectivity:", len(diff), "of", len(f_trained))
