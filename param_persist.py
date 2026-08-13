"""param_persist.py — per-run reproducibility bundle + resumable driver.

Bundle layout (§5 of param_explore_spec.md), under ``OUT_ROOT``:

    <config_name>/run_<seed>/
        arrays.npz      E_traj, I_traj, times, stim_onsets, mnist_indices
        params.json     every sampled hyperparameter + summary stats + runtime
        manifest.json   full reconstruction key (git commit, seeds, versions, GPU)

A run is COMPLETE iff all three files exist and arrays.npz loads with the
expected shapes.  The driver skips complete runs (resumable) and appends a row
to a master log CSV (flushed per row) as runs finish.
"""
from __future__ import annotations

import os
import sys
import dataclasses
import json
import time
import glob
import platform
import subprocess
from pathlib import Path

import numpy as np

import wc_config as cfg
import param_explore as PE

MASTER_LOG = "param_explore_log.csv"


# ---------------------------------------------------------------------------
# environment / provenance
# ---------------------------------------------------------------------------
def _git_commit(repo_dir=None) -> str:
    """Best-effort git commit SHA of the code (empty string if unavailable)."""
    try:
        cwd = repo_dir or Path(__file__).resolve().parent
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd),
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    # Colab fallback: env var set by the notebook after clone
    return os.environ.get("WC_CODE_COMMIT", "")


def _env_provenance() -> dict:
    import scipy, torch
    gpu = ""
    try:
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return dict(
        python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
        cuda=(torch.version.cuda or ""), gpu_name=gpu,
        platform=platform.platform(),
    )


# ---------------------------------------------------------------------------
# paths / completeness
# ---------------------------------------------------------------------------
def run_dir(out_root, config_name, seed, arm="") -> Path:
    """Bundle directory for one run.

    ``arm`` names a coefficient variant (see cfg.PILOT_COEFF_ARMS).  It becomes a
    directory level ABOVE config_name, so an armed sweep writes

        out_root/arm/config_name/run_SSS/

    and an unarmed one keeps the original out_root/config_name/run_SSS/ layout
    byte-for-byte.  Without this, three arms at one seed+config would resolve to
    the same directory: the first would be written, the other two skipped as
    "already complete", and the sweep would silently return one arm's data
    labelled as three.
    """
    root = Path(out_root)
    if arm:
        root = root / str(arm)
    return root / config_name / f"run_{seed:03d}"


def bundle_fingerprint(out_root, config_name, seed, arm="") -> str:
    """The model fingerprint recorded in an existing bundle ("" if absent)."""
    man = run_dir(out_root, config_name, seed, arm) / "manifest.json"
    try:
        with open(man) as f:
            m = json.load(f)
    except Exception:
        return ""
    fp = m.get("model_fingerprint") or {}
    return str(fp.get("hash", "")) if isinstance(fp, dict) else str(fp)


def _png_usable(p) -> bool:
    """True iff p is a PNG that was written to completion.

    Existence alone is not enough.  A worker killed mid-``savefig`` leaves a
    partial PNG on disk: correct 8-byte signature, plausible size, no terminating
    chunk.  Three such bundles sat on the durable tree looking complete to a
    stat-only check while being undecodable (the classifier derives 12 of its 13
    inputs from lfp.png pixels, so those runs were unusable).

    Checks the trailing IEND chunk, which is the LAST thing a complete PNG
    contains.  Layout: 4-byte big-endian length (zero for IEND), the 4-byte type
    tag, then a 4-byte CRC -- so the tag sits at b[-8:-4], NOT b[-12:-8].  I got
    that offset wrong once and it reported every valid PNG as truncated.
    Reads 8 bytes, not the image: no decode, cheap enough for the resume scan.
    """
    try:
        with open(p, "rb") as fh:
            fh.seek(-8, os.SEEK_END)
            return fh.read(4) == b"IEND"   # b[-8:-4] is the tag; b[-4:] is the CRC
    except (OSError, ValueError):
        return False          # absent, or shorter than 8 bytes


def is_complete(out_root, config_name, seed, expected_T=None, arm="",
                expect_fingerprint=None, require_figures=True) -> bool:
    """Fast completion check for the resumable driver.

    ``save_bundle`` writes arrays.npz -> params.json -> manifest.json (manifest
    LAST), so all three existing is a sound "run finished" marker. The default
    path is STAT-ONLY (three ``exists()`` metadata calls) — critically, it does
    NOT open arrays.npz. Opening a multi-MB npz reads the zip central directory,
    which over the Google Drive FUSE mount is a network round-trip PER RUN; doing
    that 1500x (resume skip-scan or a completeness sweep) is what makes the naive
    version crawl. Pass ``expected_T`` only when you want a deep integrity check
    (opens the npz and verifies the E_traj row count).

    ``expect_fingerprint`` (a 12-char hash from PE.model_fingerprint) makes the
    check refuse to treat a bundle built by a DIFFERENT model definition as
    complete: a mismatch returns False, so the run is redone under the current
    definition rather than silently inherited from the old one. Reading the
    fingerprint opens manifest.json, which is small (unlike arrays.npz), so this
    stays cheap over the Drive mount.
    """
    d = run_dir(out_root, config_name, seed, arm)
    arr, par, man = d / "arrays.npz", d / "params.json", d / "manifest.json"
    if not (arr.exists() and par.exists() and man.exists()):
        return False
    if require_figures:
        # 12 of the 13 classifier inputs are pixel-derived from lfp.png, so a
        # bundle without figures is not a usable run however intact its arrays
        # are.  Needed for bundles written before figures preceded the manifest:
        # those can carry a manifest and no figures, and without this check
        # resume calls them complete and skips them permanently.  Two stat calls,
        # no decode -- cheap enough to leave on by default.
        figs = d / "figures"
        for n in ("heatmap_E.png", "lfp.png"):
            if not _png_usable(figs / n):
                return False
    if expect_fingerprint:
        got = bundle_fingerprint(out_root, config_name, seed, arm)
        if got != str(expect_fingerprint):
            return False
    if expected_T is None:
        return True
    try:
        with np.load(arr) as z:
            if "E_traj" not in z or "I_traj" not in z:
                return False
            if z["E_traj"].shape[0] != expected_T:
                return False
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------
def _call_render(render_fn, out_root, result, arm):
    """Invoke a render_fn that may or may not accept an ``arm`` keyword.

    Decided by signature inspection, NOT by catching TypeError.  The previous
    try/except TypeError form was wrong in a way that mattered: any TypeError
    raised INSIDE a render that does accept `arm` was indistinguishable from a
    signature mismatch, so the render was silently re-run positionally and the
    real error surfaced (if at all) as a confusing second failure.  It also hid
    the actual worker signature bug for a full sweep -- the driver's own wrapper
    took `_arm`, not `arm`, so every call took the fallback path.
    """
    import inspect

    try:
        params = inspect.signature(render_fn).parameters
        takes_arm = ("arm" in params or
                     any(p.kind is inspect.Parameter.VAR_KEYWORD
                         for p in params.values()))
    except (TypeError, ValueError):
        takes_arm = False        # builtins/C callables: no signature available
    if takes_arm:
        return render_fn(out_root, result, arm=arm)
    return render_fn(out_root, result)


def save_bundle(out_root, result: PE.ExplorationResult, code_tag="",
                repo_dir=None, compress=True, arm="", model_fingerprint=None,
                render_fn=None) -> Path:
    """Write one run's bundle: arrays.npz -> params.json -> FIGURES -> manifest.json.

    ``render_fn(out_root, result, arm=...)`` is called here, between params.json
    and manifest.json, rather than by the caller after this returns.  That
    ordering is load-bearing and not stylistic:

    The multiprocess driver's parent drains finished runs to the shared Volume
    mid-flight, and it decides "finished" by the presence of manifest.json (see
    run_modal_sweep.relocate_bundles).  When rendering happened AFTER this
    function returned, every run had a window in which the manifest existed but
    the figures did not -- and the parent, doing exactly what it was told, moved
    the directory out from under the still-running render.  savefig then raised
    FileNotFoundError inside the worker.  Observed on 2026-08-14: of the 32
    workers whose exit code was recorded (8 of 10 shards; shards 6 and 9 were
    cancelled before reporting), 20 died this way -- and the shard-level failure
    counter reported 0 for every one of them, because the crash killed the worker
    rather than being caught per-run.

    Rendering before the manifest closes the window: no manifest means the
    directory is not eligible for relocation, so the render always writes into a
    stable path.  The invariant the whole driver leans on -- "manifest.json
    exists => this run is complete and self-contained" -- becomes true, where
    before it was true only of the arrays.

    A render failure therefore leaves NO manifest, so the run is not complete and
    resume will redo it.  That is deliberate: the classifier reads pixels out of
    lfp.png, so a bundle without figures is not a usable run.  A deterministic
    render bug will be retried on every restart, which is loud (it shows up in
    failures.jsonl and the failure count) rather than silently dropping runs from
    the corpus -- the failure mode this replaced.
    """
    d = run_dir(out_root, result.config_name, result.seed, arm)
    d.mkdir(parents=True, exist_ok=True)

    # arrays
    arr_path = d / "arrays.npz"
    saver = np.savez_compressed if compress else np.savez
    saver(arr_path, E_traj=result.E_traj, I_traj=result.I_traj,
          times=result.times, stim_onsets=result.stim_onsets,
          mnist_indices=np.asarray(result.mnist_indices, dtype=np.int64))

    # params
    with open(d / "params.json", "w") as f:
        json.dump(result.params, f, indent=2)

    # figures — BEFORE the manifest, see the docstring.  Called via the shared
    # adapter so a render_fn that does not accept `arm` is handled by signature
    # inspection rather than by catching TypeError (a TypeError raised deep
    # inside the render would otherwise be misread as a signature mismatch and
    # trigger a bogus positional retry).
    if render_fn is not None:
        _call_render(render_fn, out_root, result, arm)

    # manifest — full reconstruction key
    manifest = dict(
        run_id=result.run_id, seed=result.seed, config_name=result.config_name,
        input_mode=result.input_mode, alpha_H=result.alpha_H, tau_H=result.tau_H,
        git_commit=_git_commit(repo_dir), code_tag=code_tag,
        arm=str(arm or result.params.get("arm", "")),
        # Full model definition + its hash. Compared on resume so a mid-sweep
        # code change cannot silently mix two definitions in one output tree.
        model_fingerprint=(model_fingerprint if model_fingerprint is not None
                           else PE.model_fingerprint(cfg.make_config(result.config_name))),
        w_celoc=float(result.params.get("w_celoc", cfg.W_CELOC)),
        w_cerem=float(result.params.get("w_cerem", cfg.W_CEREM)),
        seeds=dict(master_seed=result.seed,
                   split_seed=int(cfg.SPLIT_SEED),
                   note="init_rng & config_rng spawned from SeedSequence(master_seed).spawn(2); "
                        "torch/np/py global seeded via wc_experiment.seed_everything(master_seed)"),
        run_seconds=float(result.params.get("run_seconds", PE.RUN_SECONDS)),
        dt=float(cfg.DT), N=int(result.params["N"]),
        num_output=int(result.params["num_output"]),
        mnist_input_scale=float(cfg.MNIST_INPUT_SCALE),
        record_T=int(result.E_traj.shape[0]),
        arrays_dtype=str(result.E_traj.dtype),
        runtime_s=result.runtime_s,
        n_mnist_images=len(result.mnist_indices),
        env=_env_provenance(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(d / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return d


def load_bundle(out_root, config_name, seed, arm=""):
    d = run_dir(out_root, config_name, seed, arm)
    with np.load(d / "arrays.npz") as z:
        arrays = {k: z[k] for k in z.files}
    params = json.load(open(d / "params.json"))
    manifest = json.load(open(d / "manifest.json"))
    return arrays, params, manifest


# ---------------------------------------------------------------------------
# master log (append per row, flush)
# ---------------------------------------------------------------------------
_LOG_COLS = None


def append_log_row(out_root, result: PE.ExplorationResult, arm=""):
    """Append one row to the sweep-root master log.

    The log stays at ``out_root`` (NOT under the arm directory) so one CSV spans
    every arm; ``arm`` and the E->E coefficients are columns, which is what makes
    the arms comparable in a single read.
    """
    import csv
    path = Path(out_root) / MASTER_LOG
    row = dict(result.params)  # includes seed, config_name, alpha_H, tau_H, runtime_s, summaries
    row["run_id"] = result.run_id
    row["arm"] = str(arm or result.params.get("arm", ""))
    row["runtime_s"] = float(result.runtime_s)
    # Column order must come from the FILE's existing header, not from this row's
    # keys.  Using sorted(row.keys()) per call silently misaligns every column
    # from the point where any row's key set differs from the header's -- and a
    # 1500-run sweep resumed after a code change is exactly where that happens.
    if path.exists() and path.stat().st_size > 0:
        with open(path, "r", newline="") as f:
            header = next(csv.reader(f), []) or sorted(row.keys())
        extra = sorted(set(row) - set(header))
        if extra:
            # Never reorder a partially-written log: park unknown keys in a JSON
            # side column if it exists, else drop them and say so once.
            if "extra_json" in header:
                row["extra_json"] = json.dumps({k: row[k] for k in extra}, sort_keys=True)
            else:
                print(f"[warn] master log lacks columns {extra}; not recorded in CSV "
                      f"(they are in params.json)", flush=True)
        row = {k: row.get(k, "") for k in header}
        write_header = False
    else:
        header = sorted(set(row) | {"extra_json"})
        row = {k: row.get(k, "") for k in header}
        write_header = True
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()


# ---------------------------------------------------------------------------
# resumable driver
# ---------------------------------------------------------------------------
def run_sweep(out_root, seeds, config_names=None, device=None,
              run_seconds=PE.RUN_SECONDS, record_stride=1, dtype_out=np.float16,
              code_tag="", repo_dir=None, render_fn=None, verbose=True,
              arms=None, deep_check=True, config_hook=None,
              h_init_max=None, h_init_mode=None):
    """Resumable sweep over arms x configs x seeds.

    ``arms`` is a mapping ``{arm_name: coefficient}`` (e.g. cfg.PILOT_COEFF_ARMS)
    setting both E->E coefficients for that arm; pass None for a single unarmed
    sweep using the config defaults, which reproduces the original layout and
    behaviour exactly.

    Resume is by bundle presence AND model fingerprint: a bundle written under a
    different model definition does NOT count as complete, so a mid-sweep code
    change causes those runs to be redone rather than silently inherited.  A
    disconnect costs at most the in-flight run.

    ``config_hook(config, config_name, seed) -> config`` lets a caller vary the
    config PER SEED -- used by the targeted-bias pilot, where each seed carries
    its own designed (bias_e_override, bias_i_override).  The fingerprint is
    recomputed after the hook, inside the seed loop, so resume still refuses to
    inherit a bundle written at a different design point.  With no hook the
    fingerprint is identical for every seed and behaviour is unchanged.

    ``deep_check=True`` verifies E_traj row count on skip candidates.  That opens
    arrays.npz, which over the Drive FUSE mount is a network round-trip per run;
    set False for a fast skip-scan once you trust the tree.
    """
    if config_names is None:
        config_names = [c.name for c in cfg.make_configs()]
    configs = {c.name: c for c in cfg.make_configs()}
    expected_T = len(range(0, int(round(run_seconds / cfg.DT)), record_stride))
    arm_items = list(arms.items()) if arms else [("", None)]

    total = len(arm_items) * len(config_names) * len(seeds)
    done = skipped = failed = redone = 0
    t_start = time.time()
    Path(out_root).mkdir(parents=True, exist_ok=True)

    for arm_name, coeff in arm_items:
        for cname in config_names:
            config = configs[cname]
            if coeff is not None:
                config = dataclasses.replace(
                    config, w_celoc=float(coeff), w_cerem=float(coeff))
            # arm is metadata on the config so run_id/params.json carry it
            config = dataclasses.replace(config, arm=str(arm_name))
            base_config = config
            for seed in seeds:
                config = (base_config if config_hook is None
                          else config_hook(base_config, cname, seed))
                fp = PE.model_fingerprint(config)
                # H0 range is NOT part of the model definition, so it does not enter
                # model_fingerprint -- but a bundle written at a different h_init_max
                # is a different experiment and must not be inherited by resume.
                if h_init_max is not None:
                    fp = dict(fp)
                    fp["hash"] = f"{fp['hash']}_h{float(h_init_max):g}"
                if h_init_mode:
                    fp = dict(fp)
                    fp["hash"] = f"{fp['hash']}_m{h_init_mode}"
                if is_complete(out_root, cname, seed,
                               expected_T=(expected_T if deep_check else None),
                               arm=arm_name, expect_fingerprint=fp["hash"]):
                    skipped += 1
                    continue
                stale = bundle_fingerprint(out_root, cname, seed, arm_name)
                if stale and stale != fp["hash"]:
                    redone += 1
                    if verbose:
                        print(f"[refresh] {arm_name}/{cname}/run_{seed:03d}: "
                              f"fingerprint {stale} != {fp['hash']}, recomputing",
                              flush=True)
                try:
                    result = PE.run_one_exploration(
                        seed, config, device=device, run_seconds=run_seconds,
                        record_stride=record_stride, dtype_out=dtype_out,
                        h_init_max=h_init_max, h_init_mode=h_init_mode,
                        verbose=False)
                    # render_fn is handed to save_bundle so the figures land
                    # BEFORE manifest.json, which is what makes a drained bundle
                    # complete-or-absent rather than complete-looking-but-
                    # figureless.  See save_bundle's docstring.  Consequence: a
                    # render failure now fails the RUN (no manifest, redone on
                    # resume) instead of leaving a figure-less bundle that resume
                    # skips forever and the pixel classifier cannot read.
                    save_bundle(out_root, result, code_tag=code_tag,
                                repo_dir=repo_dir, arm=arm_name,
                                model_fingerprint=fp, render_fn=render_fn)
                    append_log_row(out_root, result, arm=arm_name)
                except Exception as e:
                    # One bad run must not abandon the other 1499.
                    failed += 1
                    print(f"[FAIL] arm={arm_name or '-'} {cname} seed={seed}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    _append_failure(out_root, arm_name, cname, seed, e)
                    continue
                done += 1
                if verbose and (done % 10 == 0 or done == 1):
                    el = time.time() - t_start
                    rate = el / max(done, 1)
                    remaining = (total - skipped - done) * rate
                    print(f"[{done}/{total-skipped} new | {skipped} skipped] "
                          f"{result.run_id} rt={result.runtime_s:.1f}s "
                          f"elapsed={el/60:.1f}m eta={remaining/60:.1f}m", flush=True)

    print(f"DONE: {done} new, {skipped} skipped, {failed} failed, "
          f"{redone} refreshed (fingerprint change), total complete="
          f"{done+skipped}/{total} in {(time.time()-t_start)/60:.1f}m", flush=True)
    return dict(done=done, skipped=skipped, failed=failed, redone=redone,
                total=total)


def _append_failure(out_root, arm, config_name, seed, exc):
    """Record a failed run so the sweep summary is not silently short."""
    import traceback
    path = Path(out_root) / "failures.jsonl"
    rec = dict(arm=str(arm), config_name=config_name, seed=int(seed),
               error=f"{type(exc).__name__}: {exc}",
               traceback=traceback.format_exc()[-2000:],
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()


def sweep_status(out_root, seeds, config_names=None, arms=None,
                 expect_fingerprint=None):
    """What is present on disk, without running anything.

    Returns per-(arm, config) counts of complete / missing / stale-fingerprint,
    which is what you want after a Colab disconnect before deciding to resume.
    """
    if config_names is None:
        config_names = [c.name for c in cfg.make_configs()]
    configs = {c.name: c for c in cfg.make_configs()}
    arm_items = list(arms.items()) if arms else [("", None)]
    rows = []
    for arm_name, coeff in arm_items:
        for cname in config_names:
            config = configs[cname]
            if coeff is not None:
                config = dataclasses.replace(config, w_celoc=float(coeff),
                                             w_cerem=float(coeff))
            want = expect_fingerprint or PE.model_fingerprint(config)["hash"]
            comp = miss = stale = 0
            for seed in seeds:
                if is_complete(out_root, cname, seed, arm=arm_name,
                               expect_fingerprint=want):
                    comp += 1
                elif bundle_fingerprint(out_root, cname, seed, arm_name):
                    stale += 1
                else:
                    miss += 1
            rows.append(dict(arm=arm_name, config_name=cname, complete=comp,
                             missing=miss, stale=stale, total=len(seeds)))
    return rows
