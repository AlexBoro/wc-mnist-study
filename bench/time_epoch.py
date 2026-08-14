"""Measure per-epoch wall time at production step counts.

Cost is strictly linear in Euler steps (one sparse matmul set per step, fixed
784-element vectors), so timing a REDUCED image count and scaling by the step
ratio is exact to within noise -- and takes minutes instead of an hour.
Reports train and eval separately because eval is now branched and its cost is
not avoidable, only its state effect.
"""
import os, sys, time
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch; torch.set_num_threads(1)
import dataclasses
import wc_config as cfg
from wc_experiment import run_experiment

N_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 16
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "sequential_H"

t0 = time.time()
C = [c for c in cfg.make_configs() if c.name == CONFIG][0]
C = dataclasses.replace(C, num_epochs=2, images_per_epoch=N_IMG, eval_images=N_IMG)
res = run_experiment(
    seed=0, config=C, out_root="bench/time_out", device="cpu",
    measure_topology_flag=False, save_figures=False, resume=False, verbose=False,
)
el = time.time() - t0

steps_measured = 2 * 2 * N_IMG * cfg.IMAGE_DURATION / cfg.DT      # 2 epochs x (train+eval)
steps_prod_epoch = (cfg.IMAGES_PER_EPOCH + cfg.EVAL_IMAGES) * cfg.IMAGE_DURATION / cfg.DT
per_step = el / steps_measured
s_epoch = per_step * steps_prod_epoch
print(f"measured    : {el:.1f} s for 2 epochs x {N_IMG} train + {N_IMG} eval images")
print(f"per step    : {per_step*1e3:.3f} ms")
print(f"PROD EPOCH  : {s_epoch:.0f} s  ({s_epoch/60:.1f} min)   [{cfg.IMAGES_PER_EPOCH} train + {cfg.EVAL_IMAGES} eval]")
print(f"100 EPOCHS  : {s_epoch*100/3600:.1f} h per run")
print(f"15 RUNS seq : {s_epoch*100*15/3600:.0f} h   |  parallel on 15 containers: {s_epoch*100/3600:.1f} h")
