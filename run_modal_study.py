#!/usr/bin/env python
"""run_modal_study.py -- headless, sharded, multiprocess driver for the 100-epoch
longitudinal training runs (N seeds x 3 configs).

This is the training-path counterpart to run_modal_sweep.py.  The two solve the
same dispatch problem but make OPPOSITE calls in three places, because a
training run is ~1900x longer than an exploration probe and its intermediate
state is irreplaceable rather than cheap to recompute.

MEASURED COST (2026-08-14, 1 torch thread, CPU, sequential_H, this commit)
    945 s/epoch at IMAGES_PER_EPOCH=256 + EVAL_IMAGES=256   (14.77 ms/step,
    64000 Euler steps/epoch)             ->  26.3 h per 100-epoch run
    peak RSS 0.98 GB per process
    activity bundle 31.4 MB x 26 capture epochs -> 0.82 GB/run, 12.2 GB for 15
    topology 2.17 s per grid point x 225 points x 2 grids = ~16 min per
    measurement, x26 scheduled measurements = ~7 h/run, i.e. +27% on training

WHY PROCESSES-PER-CONTAINER AND NOT ONE RUN PER CONTAINER
    15 runs at 26.3 h is 394 CPU-hours.  One run per container would need 15
    containers against a 10-container workspace cap.  Runs are independent and
    each is single-threaded at ~1 GB, so N runs share one container at ~1x
    wall-clock each: 8 processes on one 8-CPU container finishes 8 runs in the
    time one run takes.  Two containers therefore cover 15 runs in ~26 h of
    wall-clock instead of 79 h across 5 containers.
    Thread pinning happens via env vars BEFORE torch is imported -- calling
    torch.set_num_threads() after import does not resize an existing pool.
    Size the pool from the provider_params `cpu=` you REQUESTED: neither nproc
    nor sched_getaffinity reflects it inside a Modal container.

WHY out_root IS ON THE VOLUME, UNLIKE THE SWEEP
    run_modal_sweep keeps out_root container-local and drains bundles to /vol,
    because a probe writes a 10 MB bundle every ~6 s and each write would be a
    network round-trip.  A training run writes a 3 MB checkpoint once per 15.8
    min and a 31 MB bundle once per ~63 min -- utterly negligible I/O, and
    writing it straight to /vol removes the entire relocate step that has
    already destroyed one completed shard (EXDEV at the final move).
    More decisively: a run's value is its CHECKPOINT LINEAGE.  A probe killed
    at minute 20 loses 20 minutes; a training run killed at hour 20 with
    container-local checkpoints loses 20 hours.  On /vol the same kill loses
    only the epoch in flight, and resume=True continues from the last epoch.
    Volume writes persist with no explicit commit.

WHY TOPOLOGY IS DEFERRED BY DEFAULT
    Inline topology makes a run 33 h instead of 26 h, and it is measured from
    ARCHIVED trigger-epoch checkpoints (checkpoints/topology/) which the run
    writes regardless.  So it is separable after the fact with no loss, and
    keeping it out of the training job means a topology bug cannot cost a
    26 h trajectory.  Run run_topology_pass.py afterwards, and note that pass
    needs a FRESH handle -- see the stale-mount warning below.

RESUME
    Hoisted into the parent, as in the sweep, but for a different reason: here
    each worker runs ONE (seed, config) and run_experiment's own resume already
    works, since every worker sees the same canonical --out-root on /vol.  The
    parent check exists only to avoid paying container startup for a run that
    is already DONE, and to print an honest plan before spending 26 h.

VOLUME STALE-MOUNT WARNING
    A container's /vol is a snapshot from mount time; writes committed by OTHER
    containers afterwards are invisible for the life of the sandbox, and reads
    do not error -- the paths simply are not there.  Consequences here:
      * Shard k must not census shard j's output.  This driver only ever reads
        its OWN shard's runs, so it is unaffected.
      * Any post-hoc census, classification or topology pass MUST run on a
        handle created AFTER the training jobs finished, and should assert its
        expected input count rather than skip missing paths.

EXIT-STATUS TRAP
    A `trap ... EXIT` in a Modal `command=` makes the job report exit 1 even on
    success.  This driver therefore does its log copying in Python (finally:
    block, below) and never installs a shell trap; the launching command should
    print the driver's own return code and that is the authoritative status.

Usage (one shard; submit one job per shard, one handle each):
    python run_modal_study.py --out-root /vol/wc_mnist_training \\
        --n-seeds 5 --shard 0 --n-shards 2 --workers 8 --epochs 100
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# work planning
# ---------------------------------------------------------------------------
def build_runs(seeds: list[int], config_names: list[str]) -> list[tuple[int, str]]:
    """The (seed, config) unit of work, seed-major.

    Seed-major rather than config-major so that a shard which is cut short
    still yields COMPLETE seeds -- all three configs of seed 0 finish before
    seed 1 starts.  A config-major order truncated the same way would leave
    every seed missing the same config, which supports no comparison at all.
    """
    return [(s, c) for s in seeds for c in config_names]


def shard_runs(runs: list[tuple[int, str]], shard: int, n_shards: int) -> list[tuple[int, str]]:
    """Strided slice of the run list for one shard.

    Strided, NOT contiguous -- the opposite of run_modal_sweep.shard_seeds.
    There, contiguity made a partial sweep legible on disk because every run
    was equivalent.  Here the three configs have different costs (sequential_H
    carries an extra adaptation variable) and a contiguous block could land one
    shard with all the expensive runs.  Striding interleaves configs and seeds
    so shards are cost-balanced.

    CAVEAT (found 2026-08-15, was stated wrongly here before): striding does NOT
    guarantee every shard covers all configs.  The run list is seed-major, so it
    cycles through configs with period len(configs); when gcd(n_shards,
    len(configs)) > 1 the stride locks onto that cycle and a shard collapses onto
    a single config -- e.g. 3 configs with n_shards=3 gives shard 0 all-
    simultaneous, shard 1 all-sequential_noH, shard 2 all-sequential_H.  That is
    the worst loss pattern for a paired comparison: one dead container removes an
    entire configuration across every seed, rather than costing each config
    equally.  ``coverage_warning`` below detects it and main() prints it loudly.
    Prefer an n_shards coprime with the config count (for 3 configs: 2, 4, 5).
    """
    if not (0 <= shard < n_shards):
        raise ValueError(f"shard {shard} out of range for n_shards {n_shards}")
    return runs[shard::n_shards]


def coverage_warning(runs: list[tuple[int, str]], n_shards: int,
                     config_names: list[str]) -> str | None:
    """Return a warning if striding would give some shard less than full config
    coverage, else None.  Checked across ALL shards, not just the one being run,
    so a single shard's launch still surfaces a study-wide sharding mistake."""
    if n_shards < 2 or len(config_names) < 2:
        return None
    thin = {}
    for k in range(n_shards):
        got = {c for _s, c in runs[k::n_shards]}
        if len(got) < len(config_names):
            thin[k] = sorted(got)
    if not thin:
        return None
    # Suggest by TESTING each candidate, not from a rule.  Coprimality with the
    # config count is necessary but not sufficient: a shard also needs at least
    # len(configs) runs to hold them all, so with few runs a coprime n_shards can
    # still produce a thin shard (9 runs / 4 shards -> 2-run shards).
    ok = [n for n in range(2, len(runs) + 1)
          if all(len({c for _s, c in runs[k::n]}) == len(config_names) for k in range(n))]
    hint = (f" Shard counts with full coverage for this run list: "
            f"{', '.join(map(str, ok))}." if ok else
            " No shard count above 1 gives full coverage for this run list; "
            "run it as a single shard.")
    return ("sharding gives incomplete config coverage: "
            + "; ".join(f"shard {k} only has {v}" for k, v in thin.items())
            + "." + hint
            + " One dead container would otherwise remove a whole configuration.")


def split_workers(runs: list[tuple[int, str]], n_workers: int) -> list[list[tuple[int, str]]]:
    """Round-robin runs onto workers, dropping empty workers.

    Round-robin rather than contiguous chunks for the same cost-balance reason
    as the shard stride: a worker should not get all of one config.
    """
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    chunks: list[list[tuple[int, str]]] = [[] for _ in range(min(n_workers, len(runs)))]
    if not chunks:
        return []
    for i, r in enumerate(runs):
        chunks[i % len(chunks)].append(r)
    return chunks


def run_dir_for(out_root: Path, seed: int, config_name: str) -> Path:
    """Run directory layout, mirroring wc_experiment's own run_id convention.

    Kept as a function rather than inlined so a layout change in wc_experiment
    breaks in ONE place here.  (run_experiment builds `seed{seed:03d}__{name}`.)
    """
    return out_root / f"seed{seed:03d}__{config_name}"


def filter_done(out_root: Path, runs: list[tuple[int, str]], n_epochs: int,
                verbose: bool = True) -> tuple[list[tuple[int, str]], int]:
    """Drop runs whose DONE sentinel already records >= n_epochs.

    Reads the sentinel rather than trusting its mere existence: a DONE written
    by an earlier 20-epoch pilot must NOT cause a 100-epoch run to be skipped.
    run_experiment applies the same rule internally and would resume correctly
    anyway; doing it here avoids paying process startup to learn that, and
    makes the printed plan honest about what the shard will actually compute.
    """
    todo, n_skip = [], 0
    for seed, cname in runs:
        done_path = run_dir_for(out_root, seed, cname) / "DONE"
        keep = True
        if done_path.exists():
            try:
                prev = json.loads(done_path.read_text())
                keep = int(prev.get("n_epochs", 0)) < n_epochs
            except (ValueError, OSError):
                keep = True          # unreadable sentinel: recompute, don't guess
        if keep:
            todo.append((seed, cname))
        else:
            n_skip += 1
            if verbose:
                print(f"[skip] seed{seed:03d}/{cname} already DONE at >= {n_epochs} epochs")
    return todo, n_skip


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------
# Run as a subprocess with a JSON spec on argv, exactly as run_modal_sweep does.
# The env-var thread pinning at the top is why this is a separate process and not
# a thread or a multiprocessing fork: the vars must be set before torch loads.
WORKER_SRC = r'''
import json, os, sys, time, traceback
spec = json.loads(sys.argv[1])
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
if spec["data_root"]:
    os.environ["WC_DATA_ROOT"] = spec["data_root"]
sys.path.insert(0, spec["code_dir"])

import torch
torch.set_num_threads(1)
import dataclasses
import wc_config as cfg
from wc_experiment import run_experiment

def _mk(cname):
    """Config for one run, with the capture cadence applied if requested.

    capture_interval is SEPARATE from topo_track_interval (see wc_config): setting
    it here changes only how often activity is saved, leaving the topology
    schedule alone.  None means leave the config default.
    """
    c = cfg.make_config(cname)
    if spec.get("capture_interval"):
        c = dataclasses.replace(c, capture_interval=int(spec["capture_interval"]))
    return c

ok, failed = 0, []
for seed, cname in spec["runs"]:
    t0 = time.time()
    print(f"[w] START seed={seed} config={cname}", flush=True)
    try:
        res = run_experiment(
            seed, _mk(cname),
            n_epochs=spec["epochs"],
            resume=True,                       # the whole point: survive a kill
            out_root=spec["out_root"],
            device="cpu",
            data_root=spec["data_root"],
            measure_topology_flag=spec["topology"],
            save_figures=spec["figures"],
            verbose=True,
        )
        ok += 1
        print(f"[w] DONE  seed={seed} config={cname} "
              f"({(time.time()-t0)/3600:.2f} h) done={getattr(res, 'done', None)}",
              flush=True)
    except Exception as e:
        # One run's failure must not abort its worker's remaining runs: at 26 h
        # apiece, discarding queued work to report an unrelated exception is the
        # expensive choice.  Recorded per-run and re-reported in the summary.
        failed.append({"seed": seed, "config": cname, "error": repr(e),
                       "traceback": traceback.format_exc()})
        print(f"[w] FAIL  seed={seed} config={cname}: {e!r}", flush=True)
        traceback.print_exc()

print("WORKER_SUMMARY " + json.dumps({"ok": ok, "failed": failed}), flush=True)
sys.exit(1 if failed else 0)
'''


def launch_worker(k: int, runs: list[tuple[int, str]], args, log_dir: Path):
    """Start worker k, returning (Popen, spec, open logfile handle)."""
    spec = {
        "runs": [[int(s), c] for s, c in runs],
        "out_root": str(args.out_root),
        "code_dir": str(HERE),
        "data_root": args.data_root,
        "epochs": args.epochs,
        "topology": args.topology,
        "figures": not args.no_figures,
        "capture_interval": args.capture_interval,
    }
    src = log_dir / f"worker_{k}.py"
    src.write_text(WORKER_SRC)
    lf = open(log_dir / f"worker_{k}.log", "w", buffering=1)
    p = subprocess.Popen([sys.executable, str(src), json.dumps(spec)],
                         stdout=lf, stderr=subprocess.STDOUT, cwd=str(HERE))
    return p, spec, lf


# ---------------------------------------------------------------------------
# progress
# ---------------------------------------------------------------------------
def census(out_root: Path, runs: list[tuple[int, str]], n_epochs: int) -> dict:
    """Epoch progress across this shard's runs, read from the durable tree.

    Reads metrics.csv line counts rather than the parquet: the CSV is the
    mirror that is always written (pyarrow may be absent), and a line count is
    cheap enough to call every heartbeat.  DONE is counted separately because a
    run can be at epoch 100 and still be finishing its final writes.
    """
    n_done, epochs = 0, {}
    for seed, cname in runs:
        rd = run_dir_for(out_root, seed, cname)
        if (rd / "DONE").exists():
            n_done += 1
        mcsv = rd / "metrics.csv"
        e = 0
        if mcsv.exists():
            try:
                with open(mcsv) as f:
                    e = max(0, sum(1 for _ in f) - 1)      # minus header
            except OSError:
                e = 0
        epochs[f"s{seed}/{cname}"] = e
    total = sum(epochs.values())
    return {"done": n_done, "epochs": epochs, "total_epochs": total,
            "pct": 100.0 * total / max(1, len(runs) * n_epochs)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True,
                    help="canonical run tree; put this ON THE VOLUME (/vol/...) "
                         "so checkpoints survive a container timeout")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-from", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="EXPLICIT seed list, overriding --n-seeds/--seed-from. "
                         "Use this to run a randomly drawn seed set: the master "
                         "RNG seed that produced the list belongs in the launch "
                         "script so the set is reproducible rather than ad hoc. "
                         "Note consecutive seeds are NOT biased -- over 200 "
                         "consecutive seeds, corr(seed index, drawn bias_e) is "
                         "-0.01 -- but a small consecutive BLOCK can still be "
                         "unrepresentative by chance, which is what an explicit "
                         "drawn set avoids.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8,
                    help="processes in THIS container; size from the cpu= you "
                         "requested, not from nproc (unreliable on Modal)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="default: wc_config.NUM_EPOCHS")
    ap.add_argument("--capture-interval", type=int, default=None,
                    help="how often activity is saved: 1 = EVERY epoch, 4 = the "
                         "config default. Independent of the topology schedule "
                         "(--topology / topo_track_interval), so raising capture "
                         "frequency does not raise topology cost. At float16 a "
                         "bundle is ~9 MB, so 1 costs ~1.1 GB/run at 100 epochs.")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="default: all three from wc_config.make_configs()")
    ap.add_argument("--data-root", default=None,
                    help="MNIST root; point at a staged /vol path so workers "
                         "never download concurrently")
    ap.add_argument("--topology", action="store_true",
                    help="measure topology INLINE (+~7 h/run). Default off: "
                         "run_topology_pass.py reads the archived checkpoints "
                         "afterwards for the same result")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip PNGs. NOTE: activity arrays are gated on the "
                         "same capture flag, so this also disables arrays.npz")
    ap.add_argument("--log-dir", default=None,
                    help="default <out-root>/logs_shard<k>. Keep on the Volume: "
                         "container-local worker logs die with the sandbox and "
                         "a failed shard is then undiagnosable")
    ap.add_argument("--heartbeat-s", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without spending compute")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    import wc_config as cfg
    if args.epochs is None:
        args.epochs = cfg.NUM_EPOCHS
    if args.configs is None:
        args.configs = [c.name for c in cfg.make_configs()]

    args.out_root = Path(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else args.out_root / f"logs_shard{args.shard}"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.seeds:
        # Explicit list wins. Sorted for a deterministic run order (so sharding
        # and resume are reproducible) while leaving the SET itself as drawn --
        # sorting reorders work, it does not select which seeds run.
        seeds = sorted(set(args.seeds))
        if len(seeds) != len(args.seeds):
            print(f"[warn] --seeds had duplicates; using {len(seeds)} unique")
    else:
        seeds = list(range(args.seed_from, args.seed_from + args.n_seeds))
    print(f"seeds ({len(seeds)}): {seeds}")
    all_runs = build_runs(seeds, args.configs)
    my_runs = shard_runs(all_runs, args.shard, args.n_shards)
    todo, n_skip = filter_done(args.out_root, my_runs, args.epochs)
    chunks = split_workers(todo, args.workers)

    # Cost projection from the measured per-epoch time, printed BEFORE launch so
    # a mis-sized shard is obvious while it is still free to fix.
    S_PER_EPOCH = 945.0          # measured, see module docstring
    if chunks:
        deepest = max(len(c) for c in chunks)
        est_h = deepest * args.epochs * S_PER_EPOCH / 3600.0
        if args.topology:
            est_h *= 1.27        # measured topology overhead
    else:
        est_h = 0.0

    print(f"shard {args.shard}/{args.n_shards}: {len(my_runs)} of {len(all_runs)} runs "
          f"({n_skip} already DONE, {len(todo)} to compute)")
    print(f"seeds {seeds[0]}..{seeds[-1]}  configs {args.configs}  epochs {args.epochs}")
    print(f"out_root  {args.out_root}")
    print(f"log_dir   {log_dir}")
    print(f"workers   {len(chunks)} x 1 thread   topology_inline={args.topology}  "
          f"figures={not args.no_figures}")
    for k, c in enumerate(chunks):
        print(f"  w{k}: {len(c)} runs {[f's{s}/{n}' for s, n in c]}")
    print(f"ESTIMATED WALL: {est_h:.1f} h (deepest worker) -- "
          f"set the container timeout above this")
    cov = coverage_warning(all_runs, args.n_shards, args.configs)
    if cov:
        print(f"[WARN] {cov}")
    if args.dry_run:
        return 0
    if not chunks:
        print("nothing to do: every run in this shard is already DONE")
        (args.out_root / f"shard_{args.shard}_summary.json").write_text(json.dumps(
            dict(shard=args.shard, runs=len(my_runs), skipped=n_skip,
                 computed=0, failures=0, hours=0.0, nothing_to_do=True), indent=2))
        return 0

    t0 = time.time()
    procs, rc, summaries = [], {}, {}
    interrupted = False
    try:
        for k, c in enumerate(chunks):
            p, _spec, lf = launch_worker(k, c, args, log_dir)
            procs.append((k, p, lf))
            print(f"[launch] w{k} pid={p.pid} runs={len(c)}", flush=True)

        # Heartbeat: a 26 h silent job is indistinguishable from a hung one in a
        # remote log, and at this duration that ambiguity is expensive.  Reports
        # EPOCH COUNTS off the durable tree, not just liveness, so a worker that
        # is alive but stuck shows up as a flat epoch count.
        while True:
            alive = [(k, p) for k, p, _ in procs if p.poll() is None]
            for k, p, _ in procs:
                if p.poll() is not None and k not in rc:
                    rc[k] = p.returncode
                    print(f"[done] w{k} rc={p.returncode} "
                          f"({(time.time()-t0)/3600:.2f} h)", flush=True)
            cen = census(args.out_root, my_runs, args.epochs)
            print(f"[hb] {(time.time()-t0)/3600:6.2f} h  alive={len(alive)}  "
                  f"runs_done={cen['done']}/{len(my_runs)}  "
                  f"epochs={cen['total_epochs']}/{len(my_runs)*args.epochs} "
                  f"({cen['pct']:.1f}%)", flush=True)
            if not alive:
                break
            time.sleep(args.heartbeat_s)
    except KeyboardInterrupt:
        # A container timeout or a manual stop arrives here.  Caught rather than
        # propagated so the summary below still gets written: without it, a shard
        # cut short at hour 25 leaves no record of how far it got, and the only
        # way to find out is to re-census the tree by hand.  Workers are left to
        # die with the process -- their last checkpoint is already on the Volume,
        # which is what resume needs.
        interrupted = True
        print("\n[interrupt] stopping; writing partial summary", flush=True)
    finally:
        # In Python's finally, NOT a shell EXIT trap: a trap in a Modal command=
        # makes the job report exit 1 even on success.  This block also runs on
        # KeyboardInterrupt/timeout kill, so a shard cut short still leaves its
        # logs and a summary next to the runs on the Volume.
        for _k, _p, lf in procs:
            try:
                lf.close()
            except Exception:
                pass
        for k, _p, _lf in procs:
            lp = log_dir / f"worker_{k}.log"
            if lp.exists():
                try:
                    txt = lp.read_text(errors="replace")
                    for line in txt.splitlines():
                        if line.startswith("WORKER_SUMMARY "):
                            summaries[k] = json.loads(line[len("WORKER_SUMMARY "):])
                except (OSError, ValueError):
                    pass

    el_h = (time.time() - t0) / 3600.0
    cen = census(args.out_root, my_runs, args.epochs)
    n_fail = sum(len(s.get("failed", [])) for s in summaries.values())
    failures = [f for s in summaries.values() for f in s.get("failed", [])]
    if failures:
        (log_dir / "failures.json").write_text(json.dumps(failures, indent=2))

    summary = dict(shard=args.shard, n_shards=args.n_shards,
                   runs=[[s, c] for s, c in my_runs], configs=args.configs,
                   epochs=args.epochs, skipped=n_skip, scheduled=len(todo),
                   runs_done=cen["done"], epochs_completed=cen["total_epochs"],
                   epochs_expected=len(my_runs) * args.epochs,
                   failures=n_fail, worker_rcs=rc, hours=round(el_h, 3),
                   topology_inline=args.topology, interrupted=interrupted)
    (args.out_root / f"shard_{args.shard}_summary.json").write_text(
        json.dumps(summary, indent=2))

    tag = "SHARD INTERRUPTED" if interrupted else "SHARD DONE"
    print(f"\n{tag} in {el_h:.2f} h: {cen['done']}/{len(my_runs)} runs DONE, "
          f"{cen['total_epochs']} epochs, {n_fail} failures, worker rcs={rc}")
    # A worker rc of 0 is not sufficient evidence of success -- the sweep taught
    # that a driver can print failures=0 while workers died.  Require the
    # durable tree to agree: every run of this shard must carry its DONE.
    complete = (cen["done"] == len(my_runs))
    if not complete:
        print(f"[warn] {len(my_runs) - cen['done']} run(s) of this shard lack a "
              f"DONE sentinel -- resubmit the same shard, resume will continue "
              f"them from their last checkpoint")
    return 0 if (complete and not interrupted and n_fail == 0
                 and all(v == 0 for v in rc.values())) else 1


if __name__ == "__main__":
    raise SystemExit(main())
