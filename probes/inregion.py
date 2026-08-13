import sys,os,json,numpy as np,torch,dataclasses
sys.path.insert(0,os.getcwd())
import wc_config as cfg, wc_experiment as X, param_explore as PE
OUT=sys.argv[1]; SEEDS=[int(x) for x in sys.argv[2].split(",")]
cs={c.name:c for c in cfg.make_configs()}
dev=torch.device("cpu"); rows=[]
def biases(seed):
    r,_=X.make_rngs(seed)
    for lo,hi in ((cfg.P_REWIRE_PyPy_MIN,cfg.P_REWIRE_PyPy_MAX),(cfg.P_REWIRE_PyInh_MIN,cfg.P_REWIRE_PyInh_MAX),
                  (cfg.BIAS_OUT_MIN,cfg.BIAS_OUT_MAX),(cfg.OUTPUT_WEIGHT_SCALE_MIN,cfg.OUTPUT_WEIGHT_SCALE_MAX)):
        r.uniform(lo,hi)
    return float(r.uniform(cfg.BIAS_E_INIT_MIN,cfg.BIAS_E_INIT_MAX)), float(r.uniform(cfg.BIAS_I_INIT_MIN,cfg.BIAS_I_INIT_MAX))
for seed in SEEDS:
    be,bi=biases(seed)
    for cname in ("sequential_noH","simultaneous"):
        r=PE.run_one_exploration(seed,cs[cname],device=dev,verbose=False)
        E=r.E_traj.astype(np.float32); lfp=E.mean(axis=1)
        rows.append(dict(seed=seed,config=cname,bias_e=be,bias_i=bi,
            mean_E=float(E.mean()), max_E=float(E.max()),
            lfp_max=float(lfp.max()), lfp_p99=float(np.percentile(lfp,99)),
            frac_t_hi=float(np.mean(lfp>0.05)), frac_hi=float(np.mean(E>0.5))))
        json.dump(rows,open(OUT,"w"),indent=1)
print("done",len(rows))
