#!/usr/bin/env python
"""H0 counterfactual: re-run designed tau_H seeds at several H_INIT_MAX levels.

H0 is the STATE of the adaptation variable; alpha_H is its STRENGTH.  This probe
varies only H0's upper bound and holds everything else fixed, to test whether a
near-zero initial adaptation state is what permits the unopposed opening burst.

Design points ARE passed as overrides.  `bias_e`/`bias_i` are scalar learnable
parameters (`model.bias_e.item()`), one value per run -- NOT per-unit vectors --
so the value recorded in params.json is exactly the scalar the run used, and
feeding it back as an override reproduces the run.  Verified on seed 49: with
overrides the E trajectory matches the stored bundle to float16 storage
tolerance; without them the seed redraws bias_e = -0.030 instead of the designed
-0.864, giving a far more excitable network (recurrent bursting instead of a
single 1.1 s burst).  Omitting the overrides is therefore NOT a null change.

h_init_max cannot shift the RNG stream: init_rng.uniform(lo, hi, size=N) consumes
exactly N draws whatever the bounds, so all 11 weight tensors stay bit-identical
across levels (verified), isolating H0 as the only difference between arms.
"""
from __future__ import annotations
import argparse
import csv
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import wc_config as cfg          # noqa: E402
import param_persist as PP       # noqa: E402
import param_render as PR        # noqa: E402


def load_design(path):
    """Index designed (bias_e, bias_i) by (config, seed)."""
    idx = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            idx[(row["config"], int(float(row["seed"])))] = (
                float(row["bias_e"]), float(row["bias_i"]))
    return idx


def make_hook(design):
    """Apply the designed scalar biases; refuse to fall through to a draw."""
    def hook(config, config_name, seed):
        try:
            be, bi = design[(config_name, int(seed))]
        except KeyError:
            raise KeyError(f"no design point for ({config_name}, seed={seed})") from None
        return dataclasses.replace(config, bias_e_override=be, bias_i_override=bi)
    return hook


def render(out_root, result, arm=""):
    PR.render_from_result(out_root, result,
                          image_duration=cfg.IMAGE_DURATION, arm=arm)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--design", default="handoff/design_h0.csv")
    ap.add_argument("--h-init-max", type=float, required=True)
    ap.add_argument("--h-init-mode", default=None,
                    choices=["uniform", "match_e0"],
                    help="match_e0 sets H_i(0)=E_i(0) per unit (adaptation fixed point)")
    ap.add_argument("--config", default="sequential_H")
    ap.add_argument("--run-seconds", type=float, default=10.0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    design = load_design(Path(a.design))
    seeds = sorted({s for (c, s) in design if c == a.config})
    print(f"[h0probe] {len(seeds)} designed seeds x {a.config} "
          f"@ H0~U(0,{a.h_init_max:g}) -> {a.out_root}", flush=True)

    PP.run_sweep(
        a.out_root, seeds, config_names=[a.config], device=a.device,
        run_seconds=a.run_seconds, arms=None,
        config_hook=make_hook(design), render_fn=render,
        h_init_max=a.h_init_max, h_init_mode=a.h_init_mode,
        code_tag=f"h0probe_h{a.h_init_max:g}"
                 + (f"_{a.h_init_mode}" if a.h_init_mode else ""),
        repo_dir=str(Path(__file__).parent), verbose=True,
    )


if __name__ == "__main__":
    main()
