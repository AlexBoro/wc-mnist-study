import sys,os,json
sys.path.insert(0,os.getcwd())
import wc_config as cfg
lo,hi = float(sys.argv[1]), float(sys.argv[2])
cfg.CELOC_INIT_SCALE_MIN,cfg.CELOC_INIT_SCALE_MAX=lo,hi
cfg.CEREM_INIT_SCALE_MIN,cfg.CEREM_INIT_SCALE_MAX=lo,hi
import wc_model as M
assert M.DEFAULT_INIT_RANGES["celoc_scale"]==(lo,hi), M.DEFAULT_INIT_RANGES["celoc_scale"]
import dataclasses,numpy as np,torch, param_explore as PE
cs={c.name:c for c in cfg.make_configs()}; dev=torch.device("cpu")
rows=[]
for seed in (0,1,2):
    for cname in ["simultaneous","sequential_noH","sequential_H"]:
        c=dataclasses.replace(cs[cname], w_celoc=0.10, w_cerem=0.10, arm="w100")
        r=PE.run_one_exploration(seed,c,device=dev,verbose=False)
        E=r.E_traj.astype(np.float32)
        rows.append(dict(regime=f"({lo},{hi})",seed=seed,config=cname,
                         mean_E=float(E.mean()),max_E=float(E.max()),
                         frac_hi=float(np.mean(E>0.5))))
        json.dump(rows, open(sys.argv[3],"w"), indent=1)
print("wrote", sys.argv[3], len(rows), "rows")
