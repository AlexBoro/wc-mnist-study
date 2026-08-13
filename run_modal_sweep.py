#!/usr/bin/env python
"""run_modal_sweep.py -- headless, sharded, multiprocess driver for the 500x3
uniform-draw parameter exploration.

This replaces RUN_PARAM_EXPLORE.ipynb for non-Colab execution.  The notebook's
first cells mount Google Drive and read a GitHub token from `google.colab
.userdata` to clone the repo; neither exists on a remote worker, where the code
and MNIST arrive as a payload instead.  Everything downstream of those cells --
the resumable sweep, the fingerprint discipline, the arms=None default -- is
reused unchanged from param_persist.

WHY MULTIPROCESS AND NOT THREADS
    One run is ~5000 Euler steps on 784-element vectors, with three sparse COO
    matrices rebuilt per step (wc_model.step -> _get_sparse_matrix).  That is
    dispatch-latency bound, not FLOP bound: torch threads inside a run buy
    almost nothing (measured ~1x from 6 threads), while separate processes scale
    nearly linearly (measured 5.83x from 6 processes).  So each worker is a
    subprocess pinned to ONE thread, and parallelism comes from worker count.
    Thread pinning happens via env vars BEFORE torch is imported -- setting
    torch.set_num_threads() after import does not undo an already-sized pool.

WHY EACH WORKER GETS ITS OWN out_root
    param_persist.append_log_row appends a row to ONE csv at out_root, reading
    the existing header to fix column order.  Concurrent appends from N workers
    interleave partial lines and corrupt that file.  Each worker therefore
    writes to out_root/_w<k>/, and the shard merges the per-worker logs on the
    header of the first one at the end.  Bundle DIRECTORIES are disjoint by
    (config, seed) so they are safe to write in parallel; only the shared CSV is
    not.

    Consequence for resume: a worker only sees its own _w<k>/ tree, which is
    EMPTY at launch, so run_sweep's own resume check can never fire -- it would
    recompute a run whose bundle already sits in the canonical tree.  (Observed:
    a re-run of a completed 4-run shard redid all four instead of skipping them.)
    Resume is therefore hoisted into the PARENT: filter_incomplete() checks
    PP.is_complete against --bundle-root once, before launch, and workers are
    handed only the seeds that still need computing.  Doing the scan once in the
    parent is also cheaper than N workers each scanning the whole tree.

FINGERPRINT TRAP
    Do NOT pass h_init_max/h_init_mode.  run_sweep appends a suffix to the model
    fingerprint when either is not None (`..._h0.15`), so passing the very same
    value that cfg.H_INIT_MAX already holds produces a DIFFERENT fingerprint
    from a bundle written under the config default -- and resume would recompute
    every run rather than skip it.  The config default is authoritative here.

Usage (one shard of the full sweep):
    python run_modal_sweep.py --out-root /vol/wc_mnist_param_explore \\
        --shard 3 --n-shards 8 --n-seeds 500 --workers 8 --code-tag param-explore-v3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# seed sharding
# ---------------------------------------------------------------------------
def shard_seeds(n_seeds: int, shard: int, n_shards: int,
                seed_from: int = 0) -> list[int]:
    """Contiguous seed block for one shard.

    Contiguous rather than strided so that a shard's bundles land in a
    predictable seed range, which makes a partial sweep legible on disk.  The
    remainder is spread over the first `n_seeds % n_shards` shards, so block
    sizes differ by at most one and no shard is empty.
    """
    if not (0 <= shard < n_shards):
        raise ValueError(f"shard {shard} out of range for n_shards {n_shards}")
    base, extra = divmod(n_seeds, n_shards)
    start = seed_from + shard * base + min(shard, extra)
    count = base + (1 if shard < extra else 0)
    return list(range(start, start + count))


def filter_incomplete(bundle_root, seeds, config_names, expected_T,
                      fingerprints, deep_check=True, verbose=True):
    """Seeds with at least one config still to compute, plus a skip count.

    Resume lives here rather than inside the workers: each worker's out_root is
    private and empty at launch, so run_sweep's own bundle-presence check cannot
    see the canonical tree.  A seed is kept if ANY of its configs is incomplete;
    the per-config skip inside run_sweep then handles the partial case, because
    the worker's fingerprint check reads whichever tree its own out_root points
    at -- which is why complete runs must be excluded here, not there.

    Fingerprint is per config (PE.model_fingerprint), matching what run_sweep
    computes, so a bundle written under a different model definition is treated
    as incomplete and recomputed rather than inherited.
    """
    import param_persist as PP

    keep, skipped = [], 0
    for s in seeds:
        need = False
        for cn in config_names:
            if PP.is_complete(bundle_root, cn, s,
                              expected_T=(expected_T if deep_check else None),
                              arm="", expect_fingerprint=fingerprints.get(cn)):
                skipped += 1
            else:
                need = True
        if need:
            keep.append(s)
    if verbose and skipped:
        print(f"[resume] {skipped} run(s) already complete under the current "
              f"fingerprint; {len(keep)} seed(s) still need work", flush=True)
    return keep, skipped


def split_workers(seeds: list[int], n_workers: int) -> list[list[int]]:
    """Round-robin seeds over workers.

    Round-robin, not contiguous: runtime varies with the drawn parameters (a
    saturated network integrates no faster, but rendering does vary), so
    interleaving keeps the workers balanced without predicting cost.
    """
    out: list[list[int]] = [[] for _ in range(n_workers)]
    for i, s in enumerate(seeds):
        out[i % n_workers].append(s)
    return [w for w in out if w]


# ---------------------------------------------------------------------------
# worker body -- runs in a subprocess, one thread
# ---------------------------------------------------------------------------
WORKER_SRC = r'''
import json, os, sys
# Thread pinning MUST precede the torch import.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import matplotlib
matplotlib.use("Agg")          # headless: no display on a remote worker

spec = json.loads(sys.argv[1])
sys.path.insert(0, spec["code_dir"])
os.chdir(spec["code_dir"])     # MNIST resolves relative to the repo root

import torch
torch.set_num_threads(1)
import numpy as np
import wc_config as cfg
import param_persist as PP
import param_render as PR

render = None
if spec["render"]:
    def render(out_root, result, _arm=""):
        PR.render_from_result(out_root, result,
                              image_duration=cfg.IMAGE_DURATION, arm=_arm)

summary = PP.run_sweep(
    spec["out_root"],
    seeds=spec["seeds"],
    config_names=spec["configs"],
    device="cpu",
    run_seconds=spec["run_seconds"],
    record_stride=spec["record_stride"],
    dtype_out=np.float16,
    code_tag=spec["code_tag"],
    repo_dir=spec["code_dir"],
    render_fn=render,
    verbose=True,
    arms=None,               # study default W_CELOC=W_CEREM=0.10; 1500 runs not 4500
    deep_check=spec["deep_check"],
    config_hook=None,        # uniform draws: no per-seed bias design
    # h_init_max / h_init_mode deliberately omitted -- see FINGERPRINT TRAP
)
print("WORKER_SUMMARY " + json.dumps(summary), flush=True)
'''


def graft_complete(w_out: Path, bundle_root: Path, seeds, config_names,
                   expected_T, fingerprints, deep_check) -> int:
    """Symlink already-complete bundles into a worker's private tree.

    filter_incomplete() drops seeds whose every config is done, but a seed kept
    for ONE incomplete config would still have its complete siblings recomputed,
    because the worker's own out_root is empty.  Linking those bundles in lets
    run_sweep's per-config check see them and skip.  Symlinks (not copies) keep
    the 10.3 MB/run off the container disk a second time.

    Falls back silently to recomputation if the filesystem refuses symlinks --
    wasted CPU is preferable to a failed shard.
    """
    import param_persist as PP

    n = 0
    for s in seeds:
        for cn in config_names:
            if not PP.is_complete(bundle_root, cn, s,
                                  expected_T=(expected_T if deep_check else None),
                                  arm="", expect_fingerprint=fingerprints.get(cn)):
                continue
            src = PP.run_dir(bundle_root, cn, s, "")
            dst = PP.run_dir(w_out, cn, s, "")
            if dst.exists():
                continue
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(src, target_is_directory=True)
                n += 1
            except OSError:
                return n
    return n


def launch_worker(k: int, seeds: list[int], args, log_dir: Path,
                  bundle_root: Path = None, fingerprints: dict = None,
                  expected_T: int = None):
    """Start one worker subprocess. Returns (proc, spec, logfile handle)."""
    w_out = Path(args.out_root) / f"_w{k}"
    w_out.mkdir(parents=True, exist_ok=True)
    if bundle_root is not None and bundle_root != Path(args.out_root):
        graft_complete(w_out, bundle_root, seeds, args.configs, expected_T,
                       fingerprints or {}, args.deep_check)
    spec = dict(
        code_dir=str(HERE), out_root=str(w_out), seeds=seeds,
        configs=args.configs, run_seconds=args.run_seconds,
        record_stride=args.record_stride, code_tag=args.code_tag,
        deep_check=args.deep_check, render=not args.no_render,
    )
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[v] = "1"
    lf = open(log_dir / f"worker_{k}.log", "w")
    p = subprocess.Popen([sys.executable, "-c", WORKER_SRC, json.dumps(spec)],
                         stdout=lf, stderr=subprocess.STDOUT, env=env,
                         cwd=str(HERE))
    return p, spec, lf


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------
def merge_worker_logs(out_root: Path, n_workers: int) -> int:
    """Concatenate per-worker master logs into one CSV at the shard root.

    Column order is taken from the first worker log that exists and every other
    row is realigned onto that header by NAME.  Blind concatenation would
    misalign any worker whose header differs (append_log_row derives the header
    from whichever row it writes first), which is exactly the failure its own
    comment warns about.
    """
    rows: list[dict] = []
    header: list[str] | None = None
    for k in range(n_workers):
        p = out_root / f"_w{k}" / PP_MASTER_LOG
        if not p.exists():
            continue
        with open(p, newline="") as f:
            rd = csv.DictReader(f)
            if header is None and rd.fieldnames:
                header = list(rd.fieldnames)
            for r in rd:
                rows.append(r)
    if header is None:
        return 0
    for r in rows:                       # union any late-appearing columns
        for k2 in r:
            if k2 not in header:
                header.append(k2)
    dest = out_root / PP_MASTER_LOG
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k2: r.get(k2, "") for k2 in header})
    return len(rows)


def merge_failures(out_root: Path, n_workers: int) -> int:
    """Concatenate per-worker failures.jsonl so the shard reports a real count."""
    recs = []
    for k in range(n_workers):
        p = out_root / f"_w{k}" / "failures.jsonl"
        if p.exists():
            recs += [ln for ln in p.read_text().splitlines() if ln.strip()]
    if recs:
        (out_root / "failures.jsonl").write_text("\n".join(recs) + "\n")
    return len(recs)


def relocate_bundles(out_root: Path, n_workers: int, bundle_root: Path) -> int:
    """Move each worker's bundle tree into the canonical shared layout.

    Workers write to out_root/_w<k>/<config>/run_SSS/.  The canonical tree the
    notebook and classifier expect is <bundle_root>/<config>/run_SSS/.  Moving
    (not copying) keeps the 10.3 MB/run cost off the container's disk twice.
    """
    moved = 0
    for k in range(n_workers):
        wr = out_root / f"_w{k}"
        if not wr.is_dir():
            continue
        for cfg_dir in wr.iterdir():
            if not cfg_dir.is_dir():
                continue
            for run in cfg_dir.iterdir():
                if not run.is_dir() or not run.name.startswith("run_"):
                    continue
                if run.is_symlink():
                    continue        # grafted resume link, already canonical
                dest = bundle_root / cfg_dir.name / run.name
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                run.rename(dest)
                moved += 1
    return moved


PP_MASTER_LOG = "param_explore_log.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True,
                    help="shard output root (per-worker logs, merged CSV)")
    ap.add_argument("--bundle-root", default=None,
                    help="canonical bundle tree; defaults to --out-root. Point "
                         "every shard at ONE bundle-root on a shared volume so "
                         "resume and classification see the whole sweep.")
    ap.add_argument("--n-seeds", type=int, default=500)
    ap.add_argument("--seed-from", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="default: all three from wc_config.make_configs()")
    ap.add_argument("--run-seconds", type=float, default=None,
                    help="default: param_explore.RUN_SECONDS")
    ap.add_argument("--record-stride", type=int, default=1)
    ap.add_argument("--code-tag", default="param-explore-v3")
    ap.add_argument("--no-deep-check", dest="deep_check", action="store_false",
                    help="skip the E_traj row-count check on skip candidates")
    ap.add_argument("--no-render", action="store_true",
                    help="skip PNG rendering (classification needs the figures)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the shard/worker seed plan and exit")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    import wc_config as cfg
    import param_explore as PE
    if args.configs is None:
        args.configs = [c.name for c in cfg.make_configs()]
    if args.run_seconds is None:
        args.run_seconds = PE.RUN_SECONDS

    all_seeds = shard_seeds(args.n_seeds, args.shard, args.n_shards, args.seed_from)
    n_runs = len(all_seeds) * len(args.configs)

    out_root = Path(args.out_root)
    bundle_root = Path(args.bundle_root) if args.bundle_root else out_root
    out_root.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)
    log_dir = out_root / "worker_logs"
    log_dir.mkdir(exist_ok=True)

    # expected_T and the per-config fingerprints must match what run_sweep
    # computes internally, or resume compares against the wrong key.
    expected_T = len(range(0, int(round(args.run_seconds / cfg.DT)),
                           args.record_stride))
    configs_by_name = {c.name: c for c in cfg.make_configs()}
    fingerprints = {cn: PE.model_fingerprint(configs_by_name[cn])["hash"]
                    for cn in args.configs}

    seeds, n_skip = filter_incomplete(
        bundle_root, all_seeds, args.configs, expected_T, fingerprints,
        deep_check=args.deep_check, verbose=True)
    chunks = split_workers(seeds, args.workers) if seeds else []

    print(f"shard {args.shard}/{args.n_shards}: {len(all_seeds)} seeds "
          f"[{all_seeds[0]}..{all_seeds[-1]}] x {len(args.configs)} configs "
          f"= {n_runs} runs ({n_skip} already complete)")
    print(f"workers: {len(chunks)} x 1 thread   configs: {args.configs}")
    print(f"code_tag={args.code_tag}  run_seconds={args.run_seconds}  "
          f"record_stride={args.record_stride}  deep_check={args.deep_check}  "
          f"expected_T={expected_T}  fingerprints={sorted(set(fingerprints.values()))}")
    for k, c in enumerate(chunks):
        print(f"  w{k}: {len(c)} seeds {c[:4]}{'...' if len(c) > 4 else ''}")
    if args.dry_run:
        return 0
    if not chunks:
        print("nothing to do: every run in this shard is already complete")
        (out_root / f"shard_{args.shard}_summary.json").write_text(json.dumps(
            dict(shard=args.shard, n_runs=n_runs, skipped=n_skip, log_rows=0,
                 bundles=0, failures=0, minutes=0.0, nothing_to_do=True), indent=2))
        return 0

    t0 = time.time()
    procs = []
    for k, c in enumerate(chunks):
        procs.append((k,) + launch_worker(k, c, args, log_dir, bundle_root,
                                          fingerprints, expected_T))
        print(f"[launch] w{k} pid={procs[-1][1].pid}", flush=True)

    # Progress heartbeat: a long silent job is indistinguishable from a hung one
    # in a remote log, and the harvest only happens at the end.
    rc = {}
    while True:
        alive = [(k, p) for k, p, _s, _lf in procs if p.poll() is None]
        for k, p, _s, _lf in procs:
            if p.poll() is not None and k not in rc:
                rc[k] = p.returncode
                print(f"[done] w{k} rc={p.returncode} "
                      f"({(time.time()-t0)/60:.1f}m)", flush=True)
        if not alive:
            break
        n_done = sum(1 for _ in bundle_root.rglob("manifest.json"))
        print(f"[hb] {(time.time()-t0)/60:6.1f}m  workers_alive={len(alive)}  "
              f"bundles_in_tree={n_done}", flush=True)
        time.sleep(60)
    for _k, _p, _s, lf in procs:
        lf.close()

    n_rows = merge_worker_logs(out_root, len(chunks))
    n_fail = merge_failures(out_root, len(chunks))
    n_moved = relocate_bundles(out_root, len(chunks), bundle_root)

    el = (time.time() - t0) / 60
    print(f"\nSHARD DONE in {el:.1f}m: {n_rows} log rows, {n_moved} bundles "
          f"relocated, {n_fail} failures, worker rcs={rc}")
    summary = dict(shard=args.shard, n_shards=args.n_shards, seeds=seeds,
                   configs=args.configs, n_runs=n_runs, skipped=n_skip,
                   log_rows=n_rows, bundles=n_moved, failures=n_fail,
                   worker_rcs=rc, minutes=round(el, 2), code_tag=args.code_tag)
    (out_root / f"shard_{args.shard}_summary.json").write_text(
        json.dumps(summary, indent=2))
    return 0 if all(v == 0 for v in rc.values()) and n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
