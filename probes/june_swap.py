"""Run the CURRENT simulator with JUNE connectivity+coefficients. Variant selectable."""
import sys,os,json,numpy as np,torch,dataclasses
sys.path.insert(0,os.getcwd())
import wc_config as cfg
VARIANT = sys.argv[1]; OUT = sys.argv[2]
# June's declared scale ranges must be set BEFORE wc_model import (DEFAULT_INIT_RANGES freeze)
if VARIANT != "current":
    cfg.CELOC_INIT_SCALE_MIN,  cfg.CELOC_INIT_SCALE_MAX  = -0.3, 1.5
    cfg.CEREM_INIT_SCALE_MIN,  cfg.CEREM_INIT_SCALE_MAX  = -0.4, 1.5
    cfg.CELOCI_INIT_SCALE_MIN, cfg.CELOCI_INIT_SCALE_MAX = -0.7, 1.5
import wc_model as M, wc_experiment as X, param_explore as PE

_orig = M.build_connectivity_csr
def june_csr(rng, p_pp, p_pi, N=cfg.N, k_pypy=cfg.TARGET_OUTDEG_PyPy, k_pyinh=cfg.TARGET_OUTDEG_PyInh):
    L = M.generate_ws_directed_csr(N, k_pypy,  p_pp, rng)
    R = M.generate_ws_directed_csr(N, k_pypy,  p_pp, rng)
    I = M.generate_ws_directed_csr(N, k_pyinh, p_pi, rng)
    return L, R, I
def hybrid_csr(rng, p_pp, p_pi, N=cfg.N, k_pypy=cfg.TARGET_OUTDEG_PyPy, k_pyinh=cfg.TARGET_OUTDEG_PyInh):
    """CURRENT E->E split, JUNE-style CeLocI (identical here: both use generate_ws) --
       difference is only the SCALE RANGE, handled above."""
    return _orig(rng, p_pp, p_pi, N=N, k_pypy=k_pypy, k_pyinh=k_pyinh)

if VARIANT == "june_conn":       M.build_connectivity_csr = june_csr
elif VARIANT == "june_scaleonly": M.build_connectivity_csr = hybrid_csr

# June coefficients 0.15 / 0.05 for the june_conn variant
W = dict(june_conn=(0.15,0.05), june_scaleonly=(0.10,0.10), current=(0.10,0.10))[VARIANT]
cs={c.name:c for c in cfg.make_configs()}
dev=torch.device("cpu"); rows=[]
for seed in range(6):
    for cname in ("sequential_noH","simultaneous"):
        c=dataclasses.replace(cs[cname], w_celoc=W[0], w_cerem=W[1], arm="june")
        r=PE.run_one_exploration(seed,c,device=dev,verbose=False)
        E=r.E_traj.astype(np.float32)
        rows.append(dict(variant=VARIANT,seed=seed,config=cname,mean_E=float(E.mean()),
                         max_E=float(E.max()),frac_hi=float(np.mean(E>0.5))))
        json.dump(rows,open(OUT,"w"),indent=1)
print("done",VARIANT,len(rows))
