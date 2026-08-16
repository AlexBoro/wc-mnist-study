"""float16 fidelity test against the ANALYTIC features, run locally.

The study bundles live on the Modal volume and job submission is currently
failing, so this generates bundles LOCALLY from the same code path and tests
against those. Weaker than testing the study's own bundles (different seeds and
epoch counts), but it exercises exactly the quantity at issue: whether float16
quantization of E_traj/I_traj moves an analytic amplitude across the Type 1
threshold. The pixel features are unaffected by construction -- figures are
rendered from the in-memory float32 arrays before the npz is written.
"""
import os, sys, glob, json, shutil, tempfile, dataclasses
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
os.environ.setdefault("MPLBACKEND", "Agg"); os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")
sys.path.insert(0, os.getcwd())
import numpy as np, torch; torch.set_num_threads(1)
import wc_config as cfg, wc_figures as WF, wc_experiment as X
from wc_experiment import run_experiment
import labeling_toolkit.analytic_features as AF
from labeling_toolkit.classifier import TH, SPIKE_AMP_FEATURE

DR = os.environ["WC_DATA_ROOT"]
ROOT = "bench/f16_runs"
shutil.rmtree(ROOT, ignore_errors=True)
# capture_interval=1 (the new field) so every epoch yields a bundle.
for name in ("sequential_H", "simultaneous", "sequential_noH"):
    C = [c for c in cfg.make_configs() if c.name == name][0]
    C = dataclasses.replace(C, num_epochs=3, images_per_epoch=48, eval_images=16,
                            capture_interval=1, save_activity_arrays=True)
    run_experiment(seed=0, config=C, out_root=ROOT, device=torch.device("cpu"),
                   data_root=DR, measure_topology_flag=False, save_figures=True,
                   resume=False, verbose=False)
    print("generated", name, flush=True)

bundles = sorted(glob.glob(os.path.join(ROOT, "*", "arrays", "epoch_*")))
print(f"\n{len(bundles)} bundles; SPIKE_AMP_FEATURE={SPIKE_AMP_FEATURE} "
      f"threshold={TH['SPIKE_AMP_MIN']}\n")

rows, cross = [], 0
for b in bundles:
    z = np.load(os.path.join(b, "arrays.npz"))
    f32 = AF.extract(b, X.epoch_reconstructor(b), cfg, WF)
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, os.path.basename(b)); os.makedirs(d)
        shutil.copy(os.path.join(b, "manifest.json"), d)
        kw = {k: z[k] for k in z.files}
        kw["E_traj"] = z["E_traj"].astype(np.float16).astype(np.float32)
        kw["I_traj"] = z["I_traj"].astype(np.float16).astype(np.float32)
        np.savez_compressed(os.path.join(d, "arrays.npz"), **kw)
        f16 = AF.extract(d, X.epoch_reconstructor(d), cfg, WF)
    a32 = float(f32.get(SPIKE_AMP_FEATURE, 0.0)); a16 = float(f16.get(SPIKE_AMP_FEATURE, 0.0))
    T = TH["SPIKE_AMP_MIN"]
    crossed = (a32 >= T) != (a16 >= T)
    cross += crossed
    r = dict(bundle="/".join(b.split(os.sep)[-3:]).replace("arrays/", ""),
             amp_f32=a32, amp_f16=a16,
             rel_err=abs(a32 - a16) / max(a32, 1e-12),
             margin_to_threshold=abs(a32 - T), crosses=bool(crossed))
    for k in ("amp_excursion", "width_ms", "n_chans_prominence_ge1", "n_sharp_transients"):
        if k in f32 and isinstance(f32[k], (int, float)):
            r[k + "_f32"], r[k + "_f16"] = float(f32[k]), float(f16.get(k, float("nan")))
    rows.append(r)
    print(f"  {r['bundle']:<38} amp {a32:9.4f} -> {a16:9.4f}  rel {r['rel_err']:.2e}"
          f"  margin {r['margin_to_threshold']:7.4f}{'  THRESHOLD CROSS' if crossed else ''}",
          flush=True)

rel = [r["rel_err"] for r in rows]
print(f"\nbundles tested         : {len(rows)}")
print(f"threshold crossings    : {cross}")
print(f"max relative amp error : {max(rel):.3e}")
print(f"min margin to threshold: {min(r['margin_to_threshold'] for r in rows):.4f}")
disc = [r for r in rows if r.get("n_chans_prominence_ge1_f32") != r.get("n_chans_prominence_ge1_f16")]
print(f"bundles where the >=1.0 channel COUNT changed: {len(disc)}")
json.dump(dict(rows=rows, n=len(rows), crossings=cross, max_rel_err=max(rel),
               chan_count_changed=len(disc)),
          open("bench/f16_local.json", "w"), indent=2)
