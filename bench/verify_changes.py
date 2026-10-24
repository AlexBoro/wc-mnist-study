import os, sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
_REPO = os.environ.get("WC_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
os.environ.setdefault("WC_DATA_ROOT", os.path.join(_REPO, "mnist_data"))
import torch; torch.set_num_threads(1)
import numpy as np
import wc_config as cfg, wc_experiment as X

print("=== 1. boundary constants + provenance ===")
print("  provenance:", cfg.BOUNDARY_FIT_PROVENANCE)
for c in ("simultaneous","sequential_noH"):
    print(f"  {c}: {cfg.get_boundary(c)}")
print("  sequential_H a=8:", cfg.get_boundary("sequential_H", 8.0))
print("  sequential_H a=2:", cfg.get_boundary("sequential_H", 2.0))

print("=== 2. new config fields ===")
C = [c for c in cfg.make_configs() if c.name=="sequential_H"][0]
print("  ic_e_max/ic_i_max:", C.ic_e_max, C.ic_i_max, "| save_activity_arrays:", C.save_activity_arrays)

print("=== 3. training E0/I0 == probe E0/I0 at same seed ===")
SEED=7
r1 = np.random.default_rng(np.random.SeedSequence([SEED,0xE0]))
E_train = (r1.random(784)*C.ic_e_max).astype(np.float32)
r2 = np.random.default_rng(np.random.SeedSequence([SEED,0xE0]))
E_probe = (r2.random(784)*cfg.IC_E_MAX).astype(np.float32)
print("  identical:", bool(np.array_equal(E_train,E_probe)),
      "| E0 mean %.4f max %.4f" % (E_train.mean(), E_train.max()))

print("=== 4. init_rng stream position unshifted ===")
dev=torch.device("cpu")
X.seed_everything(SEED); ir,cr = X.make_rngs(SEED); aH,tH = X.resolve_alpha_tau(C,cr)
m1 = X.build_model(SEED,C,dev,ir,aH,tH)
h1 = ir.uniform(cfg.H_INIT_MIN,cfg.H_INIT_MAX,size=m1.N)
X.seed_everything(SEED); ir2,cr2 = X.make_rngs(SEED); aH2,tH2 = X.resolve_alpha_tau(C,cr2)
m2 = X.build_model(SEED,C,dev,ir2,aH2,tH2)
h2 = ir2.uniform(cfg.H_INIT_MIN,cfg.H_INIT_MAX,size=m2.N)
print("  CeLoc identical:", bool(torch.equal(m1.CeLoc_values,m2.CeLoc_values)),
      "| bias_e identical:", bool(torch.equal(m1.bias_e,m2.bias_e)),
      "| H0 identical:", bool(np.array_equal(h1,h2)))

print("=== 5. evaluate does not mutate inputs in place ===")
import wc_model as M
tl,g,fel,feb = X.build_data(SEED,C,dev,os.environ["WC_DATA_ROOT"])
E=torch.rand(m1.N); I=torch.rand(m1.N); O=torch.zeros(m1.num_output); H=torch.rand(m1.N)*0.15
Ec,Hc = E.clone(),H.clone()
out = M.evaluate(m1,fel,torch.nn.CrossEntropyLoss(),dt=C.dt,image_duration=C.image_duration,
    output_average_window=C.output_average_window,max_images=2,verbose=False,
    E_init=E,I_init=I,O_init=O,H_init=H,pulsed_stimuli=C.pulsed_stimuli,
    sequential_row_input=C.sequential_row_input,
    row_duration=C.row_duration if C.sequential_row_input else None,
    mnist_input_scale=C.mnist_input_scale)
print("  caller's E unchanged:", bool(torch.equal(E,Ec)), "| H unchanged:", bool(torch.equal(H,Hc)))
print("  eval end-state differs from entry:", not bool(torch.equal(out[5],Ec)))
