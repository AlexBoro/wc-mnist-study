import sys,os,json
sys.path.insert(0,os.getcwd())
import wc_config as cfg
import numpy as np,torch,dataclasses, wc_experiment as X, param_explore as PE
cs={c.name:c for c in cfg.make_configs()}; dev=torch.device("cpu")
rows=[]
for seed in range(12):
    c=dataclasses.replace(cs["sequential_noH"], w_celoc=0.10, w_cerem=0.10, arm="w100")
    r=PE.run_one_exploration(seed,c,device=dev,verbose=False)
    E=r.E_traj.astype(np.float32); p=r.params
    rows.append(dict(seed=seed, mean_E=float(E.mean()), max_E=float(E.max()),
                     frac_hi=float(np.mean(E>0.5)),
                     bias_e=float(p.get("bias_e_mean",np.nan)),
                     bias_i=float(p.get("bias_i_mean",np.nan))))
    json.dump(rows,open(sys.argv[1],"w"),indent=1)
print("done",len(rows))
