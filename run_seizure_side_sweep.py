#!/usr/bin/env python
"""20 seeds x 3 configs, biases placed on the seizure-susceptible side.

Design (see handoff/design_points.csv, built by the session that made this):

  * E->E weights are drawn from the config's own continuous init range,
    CELOC/CEREM_INIT_SCALE = (-0.6, 3.0).  Nothing here overrides that; the
    range already matches what was asked for.  Both E->E coefficient knobs
    stay at the study default W_CELOC = W_CEREM = 0.10, i.e. NO arm split --
    this sweep varies bias only.

  * Each seed gets ONE bias_e and ONE depth below the boundary.  bias_i is
    then set PER CONFIG from that config's own fitted seizure boundary,
      bias_i = m * bias_e + c - depth,
    so the three configs sit at the same distance into their own seizure
    region rather than at the same absolute bias_i.  That is the comparison
    we actually want: "matched susceptibility", not "matched coordinates".

  * Boundaries are the logistic fits on WC_MNIST_parameter_explore_FINAL.xlsx
    (bias_e_mean / bias_i_mean columns), which reproduce the 6/28/26 figure.
    Seizure side is BELOW each line, established empirically from that log
    (91-96% seizure rate below vs 0.7-4% above).

  * depth is capped per seed so bias_i stays inside BIAS_I_INIT_MIN for every
    config -- otherwise the deepest points would fall outside the box the
    model samples from.

Run:
    python run_seizure_side_sweep.py --out-root <dir> [--design handoff/design_points.csv]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from pathlib import Path

# PYTHONSAFEPATH is set in this environment, so the script directory is not on
# sys.path automatically.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import wc_config as cfg          # noqa: E402
import param_persist as PP       # noqa: E402
import param_render as PR        # noqa: E402


def load_design(path: Path) -> dict[tuple[str, int], dict]:
    """Index the design table by (config, seed)."""
    idx = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            idx[(row["config"], int(float(row["seed"])))] = dict(
                bias_e=float(row["bias_e"]),
                bias_i=float(row["bias_i"]),
                depth=float(row["depth"]),
            )
    return idx


def make_hook(design):
    """config_hook(config, config_name, seed) -> config with designed biases.

    Raises rather than silently falling through to a random draw: a missing
    design point means the sweep and the table disagree, and inheriting an
    undesigned bias would quietly corrupt the comparison.
    """
    def hook(config, config_name, seed):
        try:
            dp = design[(config_name, int(seed))]
        except KeyError:
            raise KeyError(
                f"no design point for ({config_name}, seed={seed}); "
                f"design table has {len(design)} rows"
            ) from None
        return dataclasses.replace(
            config,
            bias_e_override=dp["bias_e"],
            bias_i_override=dp["bias_i"],
        )
    return hook


def render(out_root, result, arm=""):
    """Write lfp.png + heatmap_E.png where build_contact_sheet.py looks.

    Delegates to the project renderer so the LFP proxy is built from the run's
    OWN reconstructed connectivity at the study-wide fixed span, exactly as the
    pilot and the longitudinal figures were.  Safe under a bias override: the
    overridden draws still consume init_rng, so the rebuilt connectivity is
    bit-identical to what the run simulated.
    """
    PR.render_from_result(out_root, result,
                          image_duration=cfg.IMAGE_DURATION, arm=arm)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--design", default="handoff/design_points.csv")
    ap.add_argument("--run-seconds", type=float, default=10.0)
    ap.add_argument("--configs", default="simultaneous,sequential_noH,sequential_H")
    ap.add_argument("--device", default=None)
    ap.add_argument("--h-init-max", type=float, default=None,
                    help="Override H_INIT_MAX (the adaptation STATE H0 range, "
                         "not alpha_H). Consumes the same init_rng draws, so "
                         "network parameters stay bit-identical.")
    a = ap.parse_args(argv)

    design = load_design(Path(a.design))
    config_names = [c.strip() for c in a.configs.split(",") if c.strip()]
    seeds = sorted({s for (c, s) in design if c in config_names})
    print(f"[sweep] {len(config_names)} configs x {len(seeds)} seeds "
          f"= {len(config_names) * len(seeds)} runs -> {a.out_root}", flush=True)

    PP.run_sweep(
        a.out_root, seeds, config_names=config_names, device=a.device,
        run_seconds=a.run_seconds, arms=None,          # no arm split: bias-only sweep
        config_hook=make_hook(design), render_fn=render,
        h_init_max=a.h_init_max,
        code_tag="seizure_side_20x3", repo_dir=str(Path(__file__).parent),
        verbose=True,
    )


if __name__ == "__main__":
    main()
