#!/usr/bin/env python
"""bench/match_activity_bias.py -- route 2 of the confound test: match the
sequential arms' starting ACTIVITY to simultaneous by shifting the initial
biases ALONG the seizure boundary, so the starting DISTANCE to the boundary is
unchanged.

THE CONFOUND (restated)
    Sequential configs drift 2-3x further toward the boundary than simultaneous,
    but they also start ~17x quieter, with recurrent-weight gradients at exactly
    zero for the first ~6 epochs.  In a near-silent network the only parameters
    with non-zero gradient are the biases, and raising the biases IS movement
    toward the boundary.  So (H1) "sequential input structure drives the drift"
    and (H2) "a quiet network's only descent direction is toward the boundary"
    both predict the observed result.

WHY ROUTE 1 FAILED, AND WHY THIS IS DIFFERENT
    bench/match_activity.py tried to equalize activity via mnist_input_scale and
    found it IMPOSSIBLE: sequential mean E saturates at ~0.00363 against a
    0.0434 baseline (12x short), because each row drives 28 units for ~8.9 ms
    against a 10-80 ms membrane time constant -- a recruitment failure, not a
    drive deficit (total per-image drive is identical between modes).
    The bias path is not subject to that ceiling: bias_e shifts the operating
    point of every unit continuously, not just the momentarily-driven row.

THE DESIGN PROBLEM, AND THE FIX
    Naively raising bias_e to match activity would move the run's starting point
    toward the seizure boundary -- confounding the very quantity under study
    (signed_dist_to_boundary), and guaranteeing a "closer start" that trivially
    produces a different drift.  The fix is geometric: the boundary is a LINE in
    (b_E, b_I), b_I = m*b_E + c, so shifting along its TANGENT
        t = (1, m)/sqrt(1+m^2)
    changes the operating point while leaving the signed distance EXACTLY
    invariant (verified to 6 decimals: s = 0, 1, 2, 3 all give -3.221014 for
    seed 0 sequential_noH).  Activity is not constant along that line -- raising
    both biases together raises excitability -- so there is room to match
    activity at fixed boundary distance.  That is the whole trick, and it makes
    this a genuinely single-variable intervention:
        held fixed: seed, connectivity, every model weight (bias_e_override
            leaves init_rng position untouched by design -- the draws still
            happen and are discarded), the input path, and the STARTING
            SIGNED DISTANCE to that run's own boundary
        varied:     starting activity (via position along the boundary)
    If the drift gap survives this, H1: sequential input structure matters.
    If it collapses, H2: the pilot's result was about where runs start.

    Each config has its OWN boundary (and sequential_H's depends on its drawn
    alpha_H, so it is per-seed), so the tangent is computed per run, never
    shared -- the same error that had to be corrected in the pilot figure.

RESULT (2026-08-15): ROUTE 2 ALSO FAILS -- and the two failures agree.
    The geometry works exactly: signed distance is invariant along the tangent to
    6 decimals (-3.371630 at s = 0, 1, 2, 3, 4, 5 for seed 0 sequential_noH), so
    the intervention is clean.  What fails is the lever's reach.  Measured
    (seed 0, 8 images + 4 burn-in, fraction of the simultaneous target):
        sequential_noH  s=0: 0.051x   s=1: 0.054x   s=2: 0.059x
                        s=3: 0.065x   s=4: 0.071x   s=5: 0.074x
        sequential_H    s=0: 0.049x   ...           s=5: 0.093x
    Extrapolating the local slope to parity needs s ~ 375 (sequential_noH) or
    s ~ 72 (sequential_H), i.e. b_E of roughly +163 or +37 against a sampled
    BIAS_E_INIT range of [-8, 0].  A "matched" arm there would sit so far outside
    the studied region that it could no longer speak to the sweep it is meant to
    explain.

    SO: BOTH routes to equalizing initial activity fail, for the same underlying
    reason.  The sequential arms' quiescence is not a parameter that happens to
    be set low -- it is a fixed point of the recruitment dynamics (each row
    drives 28 units for ~8.9 ms against a 10-80 ms membrane time constant, so
    activity decays before it can spread).  Raising drive saturates; raising
    excitability at fixed boundary distance barely moves it.

    CONSEQUENCE, stated plainly: at these settings "sequential input" and
    "starts quiet" are not two variables that can be separated by matching --
    the second is a consequence of the first.  Any claim that the pilot's 2-3x
    larger drift is caused by input STRUCTURE rather than by starting activity
    cannot be supported by a matched-activity control, because no such control
    exists in this model.  Honest options now:
      * report the drift difference as confounded, naming the confound (this is
        the defensible reading of the pilot);
      * change what is matched rather than what is set -- compare runs at
        matched STARTING ACTIVITY across seeds within a config (the pilot's own
        seeds already span 0.0022-0.0027) and ask whether drift scales with
        initial activity inside a config, which needs no new arms;
      * change the protocol (multi-row / overlapping windows) and accept that it
        answers a different question than the pilot asked.
    The middle option is the cheapest real test and uses data already in hand.

Usage:
    python bench/match_activity_bias.py --scan          # does activity move along t?
    python bench/match_activity_bias.py                 # solve for the shift
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import wc_config as cfg
import wc_experiment as X
from match_activity import measure_mean_E


def boundary_for(seed: int, config):
    """That run's own (slope, intercept) -- resolved through the same
    alpha_H the run would draw, since sequential_H's boundary depends on it."""
    X.seed_everything(seed)
    _init_rng, config_rng = X.make_rngs(seed)
    alpha_H, _tau_H = X.resolve_alpha_tau(config, config_rng)
    slope, intercept = cfg.get_boundary(config.name, alpha_H)
    return float(slope), float(intercept), float(alpha_H)


def drawn_biases(seed: int, config):
    """The biases this (seed, config) would draw with no override, read off the
    built model so it is the realized value rather than a re-derivation."""
    X.seed_everything(seed)
    init_rng, config_rng = X.make_rngs(seed)
    alpha_H, tau_H = X.resolve_alpha_tau(config, config_rng)
    model = X.build_model(seed, config, torch.device("cpu"), init_rng, alpha_H, tau_H)
    return float(model.bias_e.item()), float(model.bias_i.item())


def shifted_config(config, be: float, bi: float, s: float):
    """Config with biases moved to (be, bi); ``s`` recorded in the arm tag so two
    arms at one seed cannot collide on the same output directory."""
    return dataclasses.replace(config, bias_e_override=float(be),
                               bias_i_override=float(bi),
                               arm=f"bshift{s:+.3f}")


def tangent_point(be0: float, bi0: float, slope: float, s: float):
    """Move ``s`` along the boundary tangent from (be0, bi0)."""
    t = np.array([1.0, slope]) / np.sqrt(1.0 + slope ** 2)
    return float(be0 + s * t[0]), float(bi0 + s * t[1])


def signed_dist(be: float, bi: float, slope: float, intercept: float) -> float:
    return float((slope * be - bi + intercept) / np.sqrt(1.0 + slope ** 2))


def measure_at_shift(seed, config, s, data_root, n_images, burn_in):
    slope, intercept, _aH = boundary_for(seed, config)
    be0, bi0 = drawn_biases(seed, config)
    be, bi = tangent_point(be0, bi0, slope, s)
    c2 = shifted_config(config, be, bi, s)
    r = measure_mean_E(seed, c2, n_images, data_root, burn_in=burn_in)
    r.update(shift=s, bias_e=be, bias_i=bi,
             signed_dist=signed_dist(be, bi, slope, intercept),
             signed_dist_unshifted=signed_dist(be0, bi0, slope, intercept))
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--images", type=int, default=12)
    ap.add_argument("--burn-in", type=int, default=4)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--scan", action="store_true",
                    help="measure activity vs shift and stop (is the lever usable?)")
    ap.add_argument("--max-shift", type=float, default=6.0)
    ap.add_argument("--out", default="bench/match_activity_bias_result.json")
    args = ap.parse_args()

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    torch.set_num_threads(1)
    data_root = args.data_root or os.environ.get("WC_DATA_ROOT") or str(
        Path(__file__).resolve().parent.parent / "mnist_data")
    configs = {c.name: c for c in cfg.make_configs()}

    # Baseline: simultaneous, unshifted, at production settings.
    base = measure_at_shift(args.seed, configs["simultaneous"], 0.0,
                            data_root, args.images, args.burn_in)
    print(f"simultaneous baseline: mean_E {base['mean_E']:.6f}  "
          f"(b_E {base['bias_e']:+.4f}, b_I {base['bias_i']:+.4f}, "
          f"signed dist {base['signed_dist']:+.4f})\n")
    target = base["mean_E"]

    out = {"seed": args.seed, "images": args.images, "burn_in": args.burn_in,
           "target_mean_E": target, "baseline": base, "arms": {}}

    for name in ("sequential_noH", "sequential_H"):
        cfg_obj = configs[name]
        slope, intercept, alpha_H = boundary_for(args.seed, cfg_obj)
        be0, bi0 = drawn_biases(args.seed, cfg_obj)
        d0 = signed_dist(be0, bi0, slope, intercept)
        print(f"=== {name} (alpha_H {alpha_H:.4f}, boundary slope {slope:.4f}) ===")
        print(f"  unshifted: b_E {be0:+.4f}, b_I {bi0:+.4f}, signed dist {d0:+.4f}")
        trace = []
        shifts = np.arange(0.0, args.max_shift + 1e-9, 1.0) if args.scan else \
                 np.array([0.0, args.max_shift])
        for s in shifts:
            t0 = time.time()
            r = measure_at_shift(args.seed, cfg_obj, float(s), data_root,
                                 args.images, args.burn_in)
            trace.append(r)
            # The invariant that makes this test valid -- assert, don't assume.
            assert abs(r["signed_dist"] - d0) < 1e-6, \
                f"tangent shift changed the boundary distance: {r['signed_dist']} vs {d0}"
            print(f"  s={s:+.2f}  b_E {r['bias_e']:+.4f} b_I {r['bias_i']:+.4f}  "
                  f"dist {r['signed_dist']:+.6f}  mean_E {r['mean_E']:.6f}  "
                  f"({r['mean_E']/target:.3f}x target)   ({time.time()-t0:.0f}s)",
                  flush=True)
        out["arms"][name] = {"slope": slope, "intercept": intercept,
                            "alpha_H": alpha_H, "bias_e0": be0, "bias_i0": bi0,
                            "signed_dist": d0, "trace": trace}
        reach = max(r["mean_E"] for r in trace) / target
        print(f"  -> best reach along the boundary: {reach:.2f}x target\n")
        out["arms"][name]["reach"] = reach

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
