#!/usr/bin/env python
"""bench/match_activity.py -- calibrate MNIST_INPUT_SCALE so the sequential configs
start at the same mean E activity as simultaneous.

WHY THIS EXISTS (the confound)
    In the 20-epoch pilot, sequential configs start ~17x quieter than
    simultaneous (mean_E_activity 0.0025 vs 0.043 at epoch 0) and their
    recurrent-weight gradients are EXACTLY zero until epoch ~6, while the bias
    gradients are non-zero from epoch 0.  So in a near-silent network the only
    parameters gradient descent can move are the biases -- and raising the
    biases IS movement toward the seizure boundary.

    That makes the pilot's headline result ambiguous.  Sequential configs drift
    2-3x further toward the boundary than simultaneous, but two explanations
    predict exactly that:
      (H1) sequential INPUT STRUCTURE drives the network toward the boundary
      (H2) sequential runs merely START QUIETER, and a quiet network's only
           available descent direction is "raise activity", which is toward the
           boundary.  Nothing about sequence per se.
    They are separable: equalize the initial activity and re-measure the drift.
    If the gap survives, H1.  If it collapses, H2 -- and the geometric story is
    about where you start, not how you're driven.

RESULT (2026-08-15): STAGE 1 FAILED, AND THE FAILURE IS THE FINDING.
    The target is UNREACHABLE by any input scale.  Sequential mean E saturates
    at ~0.00363 -- measured response curve (12 images + 4 burn-in, seed 0,
    sequential_noH):
        scale     3 -> 0.002180        scale   200 -> 0.003583
        scale    12 -> 0.003042        scale   800 -> 0.003620
        scale    50 -> 0.003409        scale  3200 -> 0.003629
    i.e. a 1000x increase in drive buys 1.7x in activity, and every 4x step past
    scale 50 changes mean E by <=5%.  The simultaneous baseline is 0.0434 -- 12x
    above the sequential ceiling.

    The cause is structural, not a gain problem -- but state it carefully: it is
    NOT that sequential presentation loses input drive.  Summed over an image the
    two modes deliver EXACTLY the same total (I_ext sum 544.14 both, all 28 rows
    visited); the drive is time-distributed, not reduced.  What fails is
    RECRUITMENT.  compute_mnist_input(active_row=r) drives 28 of 784 neurons for
    row_duration = image_duration/28 = 8.9 ms, i.e. ~4-5 Euler steps at
    dt = 0.002, while the excitatory time constant TAUPY is 10-80 ms.  A driven
    patch therefore decays before the next row arrives rather than spreading, so
    the population mean never rises even though individual driven units activate
    fully.  Raising the amplitude past saturation only pins the driven row's own
    sigmoid -- which is why both the network-wide and driven-subset curves
    plateau.  Levers that could plausibly move this are recurrent (CeLoc/CeRem
    scale), inhibitory quenching, alpha_H damping of the inter-row residue, or
    row_duration itself -- not mnist_input_scale.

    CONSEQUENCE FOR THE CONFOUND: H1 and H2 are NOT separable via input scale.
    The low activity is not an incidental setting that happens to differ between
    arms -- it is a consequence of sequential presentation itself, so "sequential
    input" and "starts quieter" are the same intervention at these settings and
    no scale choice pulls them apart.  Options that DO separate them, in
    increasing order of how much they change the model:
      (a) match on a shared driven-subset statistic instead of network-wide mean
          E (e.g. mean E over the driven row only) -- keeps both arms untouched
          and asks whether LOCAL drive is matched;
      (b) raise the excitatory bias b_E of the sequential arm at init so its
          resting activity matches, leaving the input path alone -- but b_E is a
          boundary coordinate, so this moves the start point along the very axis
          under study and must be reported as such;
      (c) widen the sequential input (multi-row blocks / overlapping windows)
          until the driven FRACTION matches, which changes the stimulus protocol
          and so answers a different question than the pilot asked.
    Recommend (a) as the honest matched comparison and (b) only as a deliberate,
    clearly-labelled probe of the start-point dependence.

WHAT THIS SCRIPT DOES (stage 1 of 2)
    Stage 1 (this file): find, per config, the mnist_input_scale whose UNTRAINED
    epoch-0 mean E activity matches the simultaneous baseline at the production
    scale (3.0).  Pure measurement, no training, no gradients -- minutes, not
    hours.  Writes bench/match_activity_result.json.
    Stage 2 (later, once containers free up): run the matched-scale arms for
    ~20 epochs and compare boundary drift against the pilot's unmatched arms.

WHY MEASURE RATHER THAN SOLVE
    Total drive is pixel x mnist_input_scale x learnable mnist_weights (two
    multiplications), then passed through the WC sigmoid and the recurrent
    network, so activity is NOT linear in the scale -- and prior work in this
    project found the analogous Liou response compressive (rate ~ s^0.43).  A
    closed-form inversion would be wrong; a short bisection on the real dynamics
    is right.

FIDELITY REQUIREMENT
    The probe must measure the SAME quantity run_experiment logs as
    mean_E_activity, or the calibration is against a different number than the
    one that showed the 17x gap.  So it replicates that path exactly:
      * same IC draw (init_rng for H, SeedSequence([seed, 0xE0]) for E/I)
      * state CARRIED image to image (not reset per image) -- this matters,
        activity builds across the epoch
      * model.forward() per Euler step with the same row schedule
        (active_row_at) for sequential
      * mean over every step of every image, as activity_callback accumulates
      * torch.no_grad() and NO optimizer step -- the only deliberate difference,
        since we want the activity the network STARTS with, before learning
    Validated against the pilot's own epoch-0 numbers before any bisection runs;
    a mismatch aborts rather than silently calibrating the wrong thing.

Usage:
    python bench/match_activity.py --images 32 --validate-only
    python bench/match_activity.py --images 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import wc_config as cfg
import wc_experiment as X
import wc_model as M


def measure_mean_E(seed: int, config, n_images: int, data_root: str,
                   scale: float | None = None, device="cpu", burn_in: int = 4) -> dict:
    """Untrained epoch-0 mean E activity, by the same accumulation run_experiment
    uses for ``mean_E_activity``.  ``scale`` overrides config.mnist_input_scale.

    ``burn_in`` images are simulated but EXCLUDED from the average.  This is not
    cosmetic: the IC draw (E0 ~ U[0, ic_e_max]) is far above the operating level,
    and measured per-image it decays inside the FIRST image -- image 0 of a
    sequential run reads 0.0969 against a steady 0.0023, a 40x transient.  A
    short average that includes it is dominated by it (n=8 gave 0.0143, 5x the
    pilot's 256-image epoch-0 value of 0.0027), which would have calibrated the
    scale against a decaying initial condition rather than the activity the
    network actually operates at.

    The default burn_in=4 is measured directly, not inferred from a neighbouring
    window (seed 0, sequential_noH, 44-image span, varying only burn_in):
        burn_in= 0 -> 0.004485      burn_in= 8 -> 0.002298
        burn_in= 2 -> 0.002295      burn_in=16 -> 0.002360
        burn_in= 4 -> 0.002306
    Excluding no images nearly doubles the estimate, while anything from 2 to 16
    lands within 3% of the same value: the estimate is flat in burn_in once the
    first image or two is gone, and 4 sits inside that plateau at a cost of
    seconds.  run_experiment's own mean_E_activity includes the transient but
    averages it over 256 images, so it dilutes to a few percent.
    """
    import dataclasses
    if scale is not None:
        config = dataclasses.replace(config, mnist_input_scale=float(scale))
    device = torch.device(device)

    X.seed_everything(seed)
    init_rng, config_rng = X.make_rngs(seed)
    alpha_H, tau_H = X.resolve_alpha_tau(config, config_rng)
    model = X.build_model(seed, config, device, init_rng, alpha_H, tau_H)

    # Identical IC draw to run_experiment (wc_experiment.py ~line 622).
    H0 = init_rng.uniform(cfg.H_INIT_MIN, cfg.H_INIT_MAX, size=model.N).astype(np.float32)
    ic_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xE0]))
    E0 = (ic_rng.random(model.N) * config.ic_e_max).astype(np.float32)
    I0 = (ic_rng.random(model.N) * config.ic_i_max).astype(np.float32)
    E = torch.tensor(E0, device=device)
    I = torch.tensor(I0, device=device)
    O = torch.zeros(model.num_output, device=device)
    H = torch.tensor(H0, device=device)

    train_loader, _g, _fe, _fb = X.build_data(seed, config, device, data_root)

    dt = config.dt
    num_steps = int(config.image_duration / dt)
    rd = config.row_duration if config.sequential_row_input else None
    if config.sequential_row_input and rd is None:
        rd = config.image_duration / 28

    total = n_images + burn_in
    e_sum, e_count, n_done = 0.0, 0, 0
    burn_sum, burn_count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for _, (data, _target) in enumerate(train_loader):
            data = data.to(device)
            for i in range(data.size(0)):
                if n_done >= total:
                    break
                img = data[i].view(-1) * config.mnist_input_scale
                counted = n_done >= burn_in          # exclude the IC transient
                for step in range(num_steps):
                    if config.sequential_row_input:
                        row = M.active_row_at(step, dt, config.image_duration, rd)
                        I_ext = model.compute_mnist_input(img, active_row=row)
                    else:
                        I_ext = model.compute_mnist_input(img)
                    E, I, O, H = model.forward(E, I, O, I_ext, dt, H=H)
                    m = float(E.mean())
                    if counted:
                        e_sum += m; e_count += 1
                    else:
                        burn_sum += m; burn_count += 1
                n_done += 1
            if n_done >= total:
                break
    return {"mean_E": e_sum / max(e_count, 1), "n_images": n_done - burn_in,
            "n_steps": e_count, "scale": float(config.mnist_input_scale),
            "alpha_H": float(alpha_H), "burn_in": burn_in,
            "burn_in_mean_E": burn_sum / max(burn_count, 1)}


def bisect_scale(seed: int, config, target: float, data_root: str, n_images: int,
                 lo: float = 0.5, hi: float = 400.0, tol: float = 0.02,
                 max_iter: int = 18, verbose: bool = True, burn_in: int = 4) -> dict:
    """Find the scale whose untrained mean E matches ``target`` within ``tol``
    (relative).  Bisection on a monotone-increasing response; brackets first and
    reports honestly if the target is unreachable in [lo, hi] rather than
    returning the endpoint as if it were a solution."""
    f_lo = measure_mean_E(seed, config, n_images, data_root, scale=lo, burn_in=burn_in)["mean_E"]
    f_hi = measure_mean_E(seed, config, n_images, data_root, scale=hi, burn_in=burn_in)["mean_E"]
    if verbose:
        print(f"    bracket: scale {lo} -> {f_lo:.6f}   scale {hi} -> {f_hi:.6f}"
              f"   target {target:.6f}", flush=True)
    if not (f_lo <= target <= f_hi):
        return {"ok": False, "reason": "target outside bracket",
                "bracket": [lo, hi], "f_bracket": [f_lo, f_hi], "target": target}
    trace = []
    for it in range(max_iter):
        mid = 0.5 * (lo + hi)
        f = measure_mean_E(seed, config, n_images, data_root, scale=mid, burn_in=burn_in)["mean_E"]
        rel = (f - target) / target
        trace.append({"iter": it, "scale": mid, "mean_E": f, "rel_err": rel})
        if verbose:
            print(f"    it{it:2d}  scale {mid:8.3f}  mean_E {f:.6f}  rel {rel:+.4f}", flush=True)
        if abs(rel) <= tol:
            return {"ok": True, "scale": mid, "mean_E": f, "rel_err": rel,
                    "iters": it + 1, "trace": trace}
        if f < target:
            lo = mid
        else:
            hi = mid
    return {"ok": False, "reason": f"no convergence in {max_iter} iters",
            "scale": mid, "mean_E": f, "rel_err": rel, "trace": trace}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--burn-in", type=int, default=4,
                    help="images simulated but excluded (IC transient); see measure_mean_E")
    ap.add_argument("--images", type=int, default=32,
                    help="images per measurement; 32 tracks the pilot's epoch-0 "
                         "value closely and keeps a measurement to ~1 min")
    ap.add_argument("--tol", type=float, default=0.02, help="relative tolerance")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="bench/match_activity_result.json")
    ap.add_argument("--validate-only", action="store_true",
                    help="check the probe against the pilot's epoch-0 numbers and stop")
    args = ap.parse_args()

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    torch.set_num_threads(1)
    data_root = args.data_root or os.environ.get("WC_DATA_ROOT") or str(
        Path(__file__).resolve().parent.parent / "mnist_data")

    configs = {c.name: c for c in cfg.make_configs()}

    # ---- fidelity gate: the probe must reproduce the pilot's epoch-0 activity ----
    # Pilot epoch-0 mean_E_activity (seed 0, 256 images, production scale 3.0):
    PILOT_E0 = {"simultaneous": 0.0493, "sequential_noH": 0.00274, "sequential_H": 0.00272}
    print(f"probe fidelity check (seed {args.seed}, {args.images} images "
          f"+ {args.burn_in} burn-in, scale 3.0)")
    print(f"{'config':<16}{'probe':>12}{'pilot epoch0':>14}{'ratio':>9}")
    measured0 = {}
    for name in ("simultaneous", "sequential_noH", "sequential_H"):
        t0 = time.time()
        r = measure_mean_E(args.seed, configs[name], args.images, data_root,
                           burn_in=args.burn_in)
        measured0[name] = r["mean_E"]
        ratio = r["mean_E"] / PILOT_E0[name]
        print(f"{name:<16}{r['mean_E']:>12.6f}{PILOT_E0[name]:>14.6f}{ratio:>9.2f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)
    # A factor-2 window: the pilot value averages 256 images with WEIGHT UPDATES
    # interleaved, this probe averages args.images with none, so exact equality is
    # not expected -- but the 17x between-config gap must be reproduced.
    gap_probe = measured0["simultaneous"] / measured0["sequential_noH"]
    gap_pilot = PILOT_E0["simultaneous"] / PILOT_E0["sequential_noH"]
    print(f"\nbetween-config gap: probe {gap_probe:.1f}x   pilot {gap_pilot:.1f}x")
    if not (0.5 <= gap_probe / gap_pilot <= 2.0):
        print("[FAIL] probe does not reproduce the pilot's activity gap; "
              "calibrating against it would be meaningless")
        return 1
    print("[OK] probe reproduces the activity gap\n")
    if args.validate_only:
        return 0

    # ---- calibration: match the sequential configs to the simultaneous baseline ----
    target = measured0["simultaneous"]
    print(f"target mean_E = {target:.6f} (simultaneous at production scale 3.0)\n")
    out = {"seed": args.seed, "images": args.images, "burn_in": args.burn_in,
           "tol": args.tol,
           "production_scale": float(cfg.MNIST_INPUT_SCALE),
           "baseline_config": "simultaneous", "target_mean_E": target,
           "probe_epoch0": measured0, "pilot_epoch0_reference": PILOT_E0,
           "gap_probe": gap_probe, "gap_pilot": gap_pilot, "arms": {}}
    for name in ("sequential_noH", "sequential_H"):
        print(f"  calibrating {name}")
        res = bisect_scale(args.seed, configs[name], target, data_root, args.images,
                           tol=args.tol, burn_in=args.burn_in)
        out["arms"][name] = res
        if res.get("ok"):
            print(f"  -> {name}: mnist_input_scale = {res['scale']:.3f} "
                  f"(mean_E {res['mean_E']:.6f}, {res['rel_err']:+.2%})\n", flush=True)
        else:
            print(f"  -> {name}: FAILED ({res.get('reason')})\n", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    ok = all(a.get("ok") for a in out["arms"].values())
    if ok:
        print("\nMATCHED SCALES (stage 2 arms):")
        for name, a in out["arms"].items():
            print(f"  {name}: --config-override mnist_input_scale={a['scale']:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
