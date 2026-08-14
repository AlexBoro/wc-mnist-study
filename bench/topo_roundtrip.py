"""Round-trip check: a deferred-topology run's archived checkpoints must be
sufficient for run_topology_pass to measure topology and merge it into metrics."""
import os, sys, time, json
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch; torch.set_num_threads(1)
from wc_experiment import measure_topology_for_run

RUN = "bench/defer_out/seed000__sequential_H"
t0 = time.time()
res = measure_topology_for_run(
    RUN, device="cpu", data_root=os.environ["WC_DATA_ROOT"],
    grid_size=5, num_samples=8, hessian_num_samples=8, lanczos_iterations=4,
    config_overrides={"images_per_epoch": 6, "eval_images": 4}, verbose=True)
out = {"result": res, "elapsed_s": round(time.time()-t0, 1)}

import csv
with open(os.path.join(RUN, "metrics.csv")) as f:
    rows = list(csv.DictReader(f))
tcols = [c for c in rows[0] if c.startswith(("domain_", "hessian_"))]
out["topology_columns"] = tcols
out["rows_with_topology"] = {
    r["epoch"]: (r.get("domain_loss_min") not in (None, "", "nan")) for r in rows}
with open("bench/topo_roundtrip.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2)[:1200])
