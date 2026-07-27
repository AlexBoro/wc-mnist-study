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
def run_dir(out_root, config_name, seed) -> Path:
    return Path(out_root) / config_name / f"run_{seed:03d}"


def is_complete(out_root, config_name, seed, expected_T=None) -> bool:
    d = run_dir(out_root, config_name, seed)
    arr, par, man = d / "arrays.npz", d / "params.json", d / "manifest.json"
    if not (arr.exists() and par.exists() and man.exists()):
        return False
    try:
        with np.load(arr) as z:
            if "E_traj" not in z or "I_traj" not in z:
                return False
            if expected_T is not None and z["E_traj"].shape[0] != expected_T:
                return False
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------
def save_bundle(out_root, result: PE.ExplorationResult, code_tag="",
                repo_dir=None, compress=True) -> Path:
    d = run_dir(out_root, result.config_name, result.seed)
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

    # manifest — full reconstruction key
    manifest = dict(
        run_id=result.run_id, seed=result.seed, config_name=result.config_name,
        input_mode=result.input_mode, alpha_H=result.alpha_H, tau_H=result.tau_H,
        git_commit=_git_commit(repo_dir), code_tag=code_tag,
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


def load_bundle(out_root, config_name, seed):
    d = run_dir(out_root, config_name, seed)
    with np.load(d / "arrays.npz") as z:
        arrays = {k: z[k] for k in z.files}
    params = json.load(open(d / "params.json"))
    manifest = json.load(open(d / "manifest.json"))
    return arrays, params, manifest


# ---------------------------------------------------------------------------
# master log (append per row, flush)
# ---------------------------------------------------------------------------
_LOG_COLS = None


def append_log_row(out_root, result: PE.ExplorationResult):
    import csv
    path = Path(out_root) / MASTER_LOG
    row = dict(result.params)  # includes seed, config_name, alpha_H, tau_H, runtime_s, summaries
    row["run_id"] = result.run_id
    write_header = not path.exists()
    # Stable column order: union of keys seen so far (persist header on first write)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()


# ---------------------------------------------------------------------------
# resumable driver
# ---------------------------------------------------------------------------
def run_sweep(out_root, seeds, config_names=None, device=None,
              run_seconds=PE.RUN_SECONDS, record_stride=1, dtype_out=np.float16,
              code_tag="", repo_dir=None, render_fn=None, verbose=True):
    """Resumable 500×3 sweep.

    Iterates ``config_names`` × ``seeds``; skips complete bundles; saves a bundle
    (+ optional figures via ``render_fn(out_root, result)``) and appends a log row
    per finished run.  A disconnect costs at most the in-flight run.
    """
    if config_names is None:
        config_names = [c.name for c in cfg.make_configs()]
    configs = {c.name: c for c in cfg.make_configs()}
    expected_T = len(range(0, int(round(run_seconds / cfg.DT)), record_stride))

    total = len(config_names) * len(seeds)
    done = skipped = 0
    t_start = time.time()
    for cname in config_names:
        config = configs[cname]
        for seed in seeds:
            if is_complete(out_root, cname, seed, expected_T=expected_T):
                skipped += 1
                continue
            result = PE.run_one_exploration(
                seed, config, device=device, run_seconds=run_seconds,
                record_stride=record_stride, dtype_out=dtype_out, verbose=False)
            save_bundle(out_root, result, code_tag=code_tag, repo_dir=repo_dir)
            if render_fn is not None:
                try:
                    render_fn(out_root, result)
                except Exception as e:
                    print(f"[warn] render failed for {result.run_id}: {e}", flush=True)
            append_log_row(out_root, result)
            done += 1
            if verbose and (done % 10 == 0 or done == 1):
                el = time.time() - t_start
                rate = el / max(done, 1)
                remaining = (total - skipped - done) * rate
                print(f"[{done}/{total-skipped} new | {skipped} skipped] "
                      f"{result.run_id} rt={result.runtime_s:.1f}s "
                      f"elapsed={el/60:.1f}m eta={remaining/60:.1f}m", flush=True)
    print(f"DONE: {done} new, {skipped} skipped, total complete="
          f"{done+skipped}/{total} in {(time.time()-t_start)/60:.1f}m", flush=True)
    return dict(done=done, skipped=skipped, total=total)
