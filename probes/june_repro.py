import sys,os,json,numpy as np,scipy.sparse as sp
sys.path.insert(0,os.getcwd())
import wc_config as cfg, wc_model as M

def june_conn(N,K_pypy,K_pyinh,p_pp,p_pi,rng):
    """June explore cell 19: THREE independent full WS graphs, CeLoc & CeRem both at p_pp."""
    L = M.generate_ws_directed_csr(N,K_pypy,p_pp,rng)
    R = M.generate_ws_directed_csr(N,K_pypy,p_pp,rng)
    I = M.generate_ws_directed_csr(N,K_pyinh,p_pi,rng)
    return L,R,I

N=784; K=156
JUNE_RANGES = dict(celoc=(-0.3,1.5), cerem=(-0.4,1.5), celoci=(-0.7,1.5))
rows=[]
for seed in range(8):
    rng=np.random.default_rng(seed)
    p_pp=float(rng.uniform(0.001,0.9)); p_pi=float(rng.uniform(0.001,0.9))
    L,R,I=june_conn(N,K,K,p_pp,p_pi,rng)
    d={}
    for nm,mat,key in (("CeLoc",L,"celoc"),("CeRem",R,"cerem"),("CeLocI",I,"celoci")):
        lo,hi=JUNE_RANGES[key]
        vals = mat.data * np.random.default_rng(seed*100+hash(nm)%97).uniform(lo,hi,size=mat.data.size)
        d[nm+"_mean"]=float(vals.mean()); d[nm+"_max"]=float(vals.max())
        d[nm+"_nnz"]=int(mat.data.size)
        d[nm+"_colsum"]=float(np.bincount(mat.indices,weights=vals,minlength=N).mean())
    d.update(seed=seed,p_pp=p_pp,p_pi=p_pi)
    rows.append(d)
json.dump(rows,open(sys.argv[1],"w"),indent=1)
print("wrote",len(rows))
