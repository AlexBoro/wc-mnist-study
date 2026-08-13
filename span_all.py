"""Measure LFP peak deflection on ALL pilot bundles.

The 15-sample subset showed a bimodal peak_abs distribution (quiet ~2-5, active
~18-30), so a Gaussian tail extrapolation to 1500 runs is not defensible. Full
census of the 60 pilot runs instead, plus the analytic ceiling implied by the
proxy's own construction.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/work/code")
os.chdir("/work/code")

sys.path.insert(0, "/work/code/labeling_toolkit")   # must precede the import

import wc_config as cfg          # noqa: E402
import param_render as PR        # noqa: E402
import wc_figures as WF          # noqa: E402
import analytic_features as AF   # noqa: E402

BUNDLES = "/vol/bundles"
CONFIGS = [c.name for c in cfg.make_configs()]
N_SEEDS = int(os.environ.get("N_SEEDS", "20"))
EXPECT = len(CONFIGS) * N_SEEDS
rows = []
for cname in CONFIGS:
    for seed in range(N_SEEDS):
        d = os.path.join(BUNDLES, cname, f"run_{seed:03d}")
        if not os.path.exists(os.path.join(d, "arrays.npz")):
            # NOT benign. A container's /vol mount is a snapshot from mount time,
            # so bundles written by a concurrent shard's sandbox are invisible
            # here. Silently skipping produced statistics over half the corpus
            # with errors:0. Record it and let the assert below fail.
            rows.append({"config": cname, "seed": seed, "error": "MISSING arrays.npz"})
            continue
        try:
            # Measure from the arrays directly -- same filtered proxy the figure
            # draws, without paying for the PNG.
            filt, _t, _sr, offset = AF.lfp_from_arrays(
                d, PR._reconstruct_connectivity, cfg, WF)
            peak = float(np.abs(filt).max())
            rows.append({"config": cname, "seed": seed, "peak_abs": peak,
                         "span_needed": 135.0 + 2 * peak,
                         "offset": float(offset)})
        except Exception as e:                        # noqa: BLE001
            rows.append({"config": cname, "seed": seed,
                         "error": f"{type(e).__name__}: {e}"})

ok = [r for r in rows if "error" not in r]
missing = [r for r in rows if "error" in r]
assert len(ok) == EXPECT, (
    f"measured {len(ok)} of {EXPECT} expected bundles; {len(missing)} missing. "
    f"If this job reused a sandbox that was warm while another shard wrote to "
    f"/vol, its mount is stale -- rerun on a FRESH handle. First few: "
    f"{missing[:3]}")
pk = np.array([r["peak_abs"] for r in ok])
res = {"n": len(ok), "n_err": len(rows) - len(ok), "rows": rows,
       "peak_abs": {"min": float(pk.min()), "max": float(pk.max()),
                    "mean": float(pk.mean()), "median": float(np.median(pk)),
                    "p90": float(np.percentile(pk, 90)),
                    "p99": float(np.percentile(pk, 99))},
       "span_needed_max": float(135 + 2 * pk.max()),
       "frac_over_150": float((135 + 2 * pk > 150).mean()),
       "frac_over_200": float((135 + 2 * pk > 200).mean()),
       "frac_over_240": float((135 + 2 * pk > 240).mean())}
os.makedirs("/work/out", exist_ok=True)
with open("/work/out/span_all.json", "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
