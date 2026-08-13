"""Choose LFP_SPAN by re-rendering at candidate spans and re-extracting the
PIXEL features the classifier actually reads.

Two failure modes pull in opposite directions:
  * span too SMALL -> the largest deflections clip, truncating exactly the
    top/bottom electrodes that the channel-extent rules count.
  * span too LARGE -> traces compress vertically and small deflections sink
    below the line-width / anti-aliasing floor that analytic_features documents,
    so subtle events stop being detected.
So the span cannot be set from clipping alone. This renders the same runs at
each candidate and reports the pixel features, letting the two effects be seen
against each other.

Runs on a set spanning both modes of the bimodal peak distribution.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/work/code")
sys.path.insert(0, "/work/code/labeling_toolkit")
os.chdir("/work/code")

import wc_config as cfg          # noqa: E402
import param_persist as PP       # noqa: E402
import param_render as PR        # noqa: E402
import features as FT            # noqa: E402
import analytic_features as AF   # noqa: E402
import wc_figures as WF          # noqa: E402

BUNDLES = "/vol/bundles"
SPANS = [150.0, 200.0, 240.0, 280.0]
# quiet-mode and active-mode representatives, measured in the span census
CASES = [("sequential_H", 1), ("sequential_noH", 2), ("sequential_H", 3),
         ("simultaneous", 0), ("simultaneous", 4), ("simultaneous", 13),
         ("sequential_noH", 4), ("sequential_H", 4)]
# the pixel features the classifier reads (12 of its 13 inputs)
PIX = ["spike_amp_max", "sync_spike_extent", "vertical_spread_fraction",
       "burst_band_frac", "burst_band_dens", "ink_frac", "max_col_frac",
       "cross_cols", "fast_dur_s", "fast_contig_dur_s", "fast_ibi_ratio",
       "n_fpfa_sequences"]

out = []
for cname, seed in CASES:
    d = os.path.join(BUNDLES, cname, f"run_{seed:03d}")
    if not os.path.exists(os.path.join(d, "arrays.npz")):
        out.append({"config": cname, "seed": seed, "error": "missing"})
        continue
    arrays, params, _m = PP.load_bundle(BUNDLES, cname, seed, arm=None)
    # analytic amplitude is span-INDEPENDENT: the invariant reference
    af = AF.extract(d, PR._reconstruct_connectivity, cfg, WF)
    rec = {"config": cname, "seed": seed,
           "amp_prominence": af.get("amp_prominence"),
           "amp_excursion": af.get("amp_excursion"),
           "n_chans_prom_ge1": af.get("n_chans_prominence_ge1"),
           "by_span": {}}
    for sp in SPANS:
        # render_run (not render_from_bundle) so figures go to a scratch dir:
        # render_from_bundle writes into the bundle's own figures/ and would
        # overwrite the pilot's rendered PNGs on /vol.
        fd = f"/tmp/spanchk/{cname}_{seed}_{int(sp)}"
        os.makedirs(fd, exist_ok=True)
        stats = PR.render_run(fd, arrays["E_traj"], arrays["I_traj"],
                              arrays["times"], seed, cname,
                              arm="", span=sp, return_stats=True)
        f = FT.extract_features(os.path.join(fd, "heatmap_E.png"),
                                os.path.join(fd, "lfp.png"),
                                mean_E=None, val_acc=None)
        row = {k: (round(float(f[k]), 5) if isinstance(f.get(k), (int, float))
                   else f.get(k)) for k in PIX if k in f}
        row["clipped_frac"] = round(float(stats.get("clipped_frac", 0.0)), 5)
        rec["by_span"][str(int(sp))] = row
    out.append(rec)

os.makedirs("/work/out", exist_ok=True)
with open("/work/out/span_sweep.json", "w") as fh:
    json.dump(out, fh, indent=1)

for r in out:
    if "error" in r:
        continue
    print(f"\n{r['config']} seed{r['seed']:02d}  amp_prom={r['amp_prominence']} "
          f"(span-independent)")
    for sp, row in r["by_span"].items():
        print(f"   span {sp:>4}: clip={row['clipped_frac']*100:6.3f}%  "
              f"spike_amp={row.get('spike_amp_max')}  "
              f"extent={row.get('sync_spike_extent')}  "
              f"vspread={row.get('vertical_spread_fraction')}  "
              f"ink={row.get('ink_frac')}")
