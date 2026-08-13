"""Pilot gate checks 2 (LFP span) and 3 (regime occupancy).

Runs on Modal because both need E_traj from every bundle and the bundles live on
/vol -- 60 x 10.1 MB now, 15.2 GB at full scale. Shipping them home to check
them would cost more than the sweep did.

Writes out/gate23.json (small) for the parent to read.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/work/code")
os.chdir("/work/code")

import wc_config as cfg          # noqa: E402
import param_persist as PP       # noqa: E402
import param_render as PR        # noqa: E402

BUNDLES = "/vol/bundles"
CONFIGS = [c.name for c in cfg.make_configs()]
SEEDS = list(range(int(os.environ.get("N_SEEDS", "20"))))
# A container's /vol mount is a snapshot from mount time: bundles written by a
# concurrent shard's sandbox are invisible here for the life of this sandbox.
# The first version of this script skipped missing paths and reported errors:0
# over half the corpus. Assert the count instead -- and run on a FRESH handle.
_present = [(c, s) for c in CONFIGS for s in SEEDS
            if os.path.exists(os.path.join(BUNDLES, c, f"run_{s:03d}", "arrays.npz"))]
assert len(_present) == len(CONFIGS) * len(SEEDS), (
    f"see {len(_present)} of {len(CONFIGS)*len(SEEDS)} bundles on /vol -- stale "
    f"mount? rerun on a fresh handle (sandbox mounts current state at start).")
QUIET, SATUR = 0.05, 0.95
LADDER = (28 - 1) * 5.0

out = {"lfp_span": {"cfg_value": float(cfg.LFP_SPAN), "ladder": LADDER,
                    "headroom": float(cfg.LFP_SPAN) - LADDER, "rows": []},
       "occupancy": {"quiet_th": QUIET, "satur_th": SATUR, "rows": []}}

# ---- 2. LFP span calibration ------------------------------------------------
# Span must cover the electrode ladder plus the signal excursion around it. The
# y-axis is deliberately NOT autoscaled, so a fixed pixel-deflection threshold
# only transfers between figures if the span is constant AND uncliped.
worst = 0.0
for cname in CONFIGS:
    for seed in SEEDS[:5]:
        try:
            st = PR.render_from_bundle(BUNDLES, cname, seed, arm=None,
                                       return_stats=True)
        except Exception as e:                      # noqa: BLE001
            out["lfp_span"]["rows"].append(
                {"config": cname, "seed": seed, "error": f"{type(e).__name__}: {e}"})
            continue
        row = {"config": cname, "seed": seed,
               "peak_abs": float(st["peak_abs"]),
               "span_needed": float(st["span_needed"]),
               "clipped_frac": float(st["clipped_frac"])}
        out["lfp_span"]["rows"].append(row)
        worst = max(worst, row["span_needed"])
out["lfp_span"]["max_span_needed"] = worst
out["lfp_span"]["verdict"] = ("OK" if worst <= cfg.LFP_SPAN
                             else f"CLIPPING - raise LFP_SPAN to >= {int(np.ceil(worst/10)*10)}")

# ---- 3. regime occupancy on the real dynamics -------------------------------
# A pinned (saturated) network cannot show a seizure transition, so '%runs
# pinned' is the number that decides whether the operating point is usable.
for cname in CONFIGS:
    means, quiets, saturs, pinned, n = [], [], [], 0, 0
    per_seed = []
    for seed in SEEDS:
        try:
            arrays, _p, _m = PP.load_bundle(BUNDLES, cname, seed, arm=None)
        except Exception:                           # noqa: BLE001
            continue
        E = arrays["E_traj"].astype(np.float32)
        half = E[E.shape[0] // 2:]                  # 2nd half: past the transient
        means.append(float(half.mean()))
        quiets.append(float((half < QUIET).mean()))
        saturs.append(float((half > SATUR).mean()))
        pinned += int(half.mean() > SATUR)
        n += 1
        per_seed.append({"seed": seed, "mean_E": float(half.mean()),
                         "frac_quiet": float((half < QUIET).mean()),
                         "frac_satur": float((half > SATUR).mean()),
                         "max_E": float(half.max())})
    if not n:
        continue
    out["occupancy"]["rows"].append(
        {"config": cname, "n": n,
         "mean_E": float(np.mean(means)), "sd_E": float(np.std(means)),
         "pct_quiet": float(100 * np.mean(quiets)),
         "pct_satur": float(100 * np.mean(saturs)),
         "pct_runs_pinned": float(100 * pinned / n),
         "per_seed": per_seed})

os.makedirs("/work/out", exist_ok=True)
with open("/work/out/gate23.json", "w") as f:
    json.dump(out, f, indent=1)
print("LFP max_span_needed:", round(worst, 1), "|", out["lfp_span"]["verdict"])
for r in out["occupancy"]["rows"]:
    print(f"  {r['config']:16s} mean_E={r['mean_E']:.3f} "
          f"quiet={r['pct_quiet']:.1f}% satur={r['pct_satur']:.1f}% "
          f"pinned={r['pct_runs_pinned']:.0f}%")
