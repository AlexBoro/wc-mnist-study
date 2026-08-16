"""Measure the per-epoch cost of capture (arrays + figures) vs no capture.

Answers: what does moving from every-4th-epoch capture to EVERY-epoch capture
cost in wall time? Three arms at identical settings, differing only in what is
saved, so the delta is attributable:
    arm A  capture OFF          -> pure training cost
    arm B  arrays ON, figs OFF  -> + activity buffer, GPU->CPU stack, npz write
    arm C  arrays ON, figs ON   -> + PNG rendering (the current capture path)
"""
import os, sys, time, json, shutil, dataclasses
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch; torch.set_num_threads(1)
import wc_config as cfg
from wc_experiment import run_experiment

DR = os.environ["WC_DATA_ROOT"]
BASE = [c for c in cfg.make_configs() if c.name == "sequential_H"][0]
# topo_track_interval=1 forces capture EVERY epoch, which is exactly the
# proposed change; the arms differ only in save_activity_arrays / save_figures.
ARMS = [
    ("A_no_capture",   dict(save_activity_arrays=False), False),
    ("B_arrays_only",  dict(save_activity_arrays=True),  False),
    ("C_arrays_figs",  dict(save_activity_arrays=True),  True),
]
N_EP = 4
out = {}
for name, over, figs in ARMS:
    C = dataclasses.replace(BASE, num_epochs=N_EP, images_per_epoch=48, eval_images=32,
                            topo_track_interval=1, **over)
    root = f"bench/cap_{name}"
    shutil.rmtree(root, ignore_errors=True)
    t0 = time.time()
    run_experiment(seed=0, config=C, out_root=root, device=torch.device("cpu"),
                   data_root=DR, measure_topology_flag=False, save_figures=figs,
                   resume=False, verbose=False)
    el = time.time() - t0
    def dsize(p):
        t = 0
        for r, _, fs in os.walk(p):
            for f in fs:
                t += os.path.getsize(os.path.join(r, f))
        return t
    rd = os.path.join(root, "seed000__sequential_H")
    out[name] = dict(total_s=round(el, 2), per_epoch_s=round(el / N_EP, 3),
                     arrays_bytes=dsize(os.path.join(rd, "arrays")),
                     figures_bytes=dsize(os.path.join(rd, "figures")))
    print(f"{name:<16} {el:7.2f}s total  {el/N_EP:6.3f}s/epoch  "
          f"arrays={out[name]['arrays_bytes']/1e6:6.2f}MB "
          f"figs={out[name]['figures_bytes']/1e6:6.2f}MB", flush=True)

a, b, c = out["A_no_capture"]["per_epoch_s"], out["B_arrays_only"]["per_epoch_s"], out["C_arrays_figs"]["per_epoch_s"]
out["deltas"] = dict(arrays_overhead_s_per_epoch=round(b - a, 3),
                     figures_overhead_s_per_epoch=round(c - b, 3),
                     total_capture_overhead_s_per_epoch=round(c - a, 3),
                     pct_of_no_capture=round(100 * (c - a) / a, 1) if a else None)
json.dump(out, open("bench/capture_cost.json", "w"), indent=2)
print("\ndeltas:", json.dumps(out["deltas"], indent=2))
