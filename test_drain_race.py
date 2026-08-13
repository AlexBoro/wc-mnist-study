"""Regression tests for the mid-flight drain race that killed the 2026-08-14 sweep.

Failure being locked down: run_sweep wrote manifest.json and THEN rendered, while
the multiprocess parent's heartbeat treated manifest.json as "run finished" and
moved the directory to the shared Volume.  A run drained inside that window had
its directory pulled out from under the still-running savefig, so the worker died
with FileNotFoundError -- 34 of 40 workers across 10 shards -- and the shard's
failure counter read 0 because the crash killed the worker instead of being
caught per-run.

These tests use fakes rather than real simulations: the race is in the ordering of
writes, not in the model, and a real run costs ~35 s.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import param_persist as PP
import run_modal_sweep as RMS


class FakeResult:
    """Minimal stand-in for PE.ExplorationResult as save_bundle consumes it."""

    def __init__(self, seed=0, config_name="simultaneous", T=8):
        self.seed = seed
        self.config_name = config_name
        self.run_id = f"{config_name}_run{seed:03d}"
        self.input_mode = "sequential"
        self.alpha_H = 0.0
        self.tau_H = 1.0
        self.runtime_s = 0.1
        self.E_traj = np.zeros((T, 4), dtype=np.float16)
        self.I_traj = np.zeros((T, 4), dtype=np.float16)
        self.times = np.arange(T, dtype=np.float32)
        self.stim_onsets = np.zeros(1, dtype=np.float32)
        self.mnist_indices = [0]
        self.params = {"N": 4, "num_output": 1, "seed": seed,
                       "config_name": config_name, "run_seconds": 1.0}


def _fp():
    return {"hash": "deadbeefcafe", "spec": {}}


def test_manifest_is_written_after_figures(tmp_path):
    """The invariant the drain depends on: manifest.json exists => figures exist."""
    order = []

    def render_fn(out_root, result, arm=""):
        d = PP.run_dir(out_root, result.config_name, result.seed, arm)
        order.append(("render", (d / "manifest.json").exists()))
        (d / "figures").mkdir(parents=True, exist_ok=True)
        for n in ("heatmap_E.png", "lfp.png"):
            (d / "figures" / n).write_bytes(b"x")

    d = PP.save_bundle(tmp_path, FakeResult(), model_fingerprint=_fp(),
                       render_fn=render_fn)
    assert order == [("render", False)], (
        "render ran after the manifest existed -- the drain window is open again")
    assert (d / "manifest.json").exists()
    assert (d / "figures" / "lfp.png").exists()


def test_render_failure_leaves_no_manifest(tmp_path):
    """A failed render must not leave a complete-looking, figure-less bundle.

    Such a bundle is unusable (12 of 13 classifier inputs are pixel-derived) yet
    resume would skip it forever, silently shrinking the corpus.
    """
    def boom(out_root, result, arm=""):
        raise RuntimeError("render exploded")

    with pytest.raises(RuntimeError):
        PP.save_bundle(tmp_path, FakeResult(), model_fingerprint=_fp(),
                       render_fn=boom)
    d = PP.run_dir(tmp_path, "simultaneous", 0, "")
    assert not (d / "manifest.json").exists()
    assert not PP.is_complete(tmp_path, "simultaneous", 0,
                              expect_fingerprint="deadbeefcafe")


def test_relocate_skips_run_without_manifest(tmp_path):
    """relocate_bundles must ignore an in-flight run (no manifest yet)."""
    w = tmp_path / "_w0" / "simultaneous" / "run_000"
    w.mkdir(parents=True)
    (w / "arrays.npz").write_bytes(b"x")
    b = tmp_path / "bundles"
    assert RMS.relocate_bundles(tmp_path, 1, b) == 0
    (w / "manifest.json").write_text("{}")
    assert RMS.relocate_bundles(tmp_path, 1, b) == 1
    assert (b / "simultaneous" / "run_000" / "manifest.json").exists()


def test_is_complete_rejects_figureless_bundle(tmp_path):
    """Bundles written by the old ordering carry a manifest but no figures."""
    d = PP.run_dir(tmp_path, "simultaneous", 0, "")
    d.mkdir(parents=True)
    (d / "arrays.npz").write_bytes(b"x")
    (d / "params.json").write_text("{}")
    (d / "manifest.json").write_text(json.dumps(
        {"model_fingerprint": {"hash": "deadbeefcafe"}}))
    assert not PP.is_complete(tmp_path, "simultaneous", 0,
                              expect_fingerprint="deadbeefcafe")
    assert PP.is_complete(tmp_path, "simultaneous", 0,
                          expect_fingerprint="deadbeefcafe",
                          require_figures=False)
    (d / "figures").mkdir()
    for n in ("heatmap_E.png", "lfp.png"):
        (d / "figures" / n).write_bytes(b"x")
    assert PP.is_complete(tmp_path, "simultaneous", 0,
                          expect_fingerprint="deadbeefcafe")


def test_call_render_does_not_mistake_inner_typeerror(tmp_path):
    """A TypeError from INSIDE the render must propagate, not trigger a retry.

    The old `try: f(arm=...) except TypeError: f(...)` form could not tell a
    signature mismatch from a real error, so it silently re-ran the render.
    """
    calls = []

    def render_fn(out_root, result, arm=""):
        calls.append(arm)
        raise TypeError("unrelated bug deep inside plotting")

    with pytest.raises(TypeError):
        PP._call_render(render_fn, tmp_path, FakeResult(), "armA")
    assert calls == ["armA"], f"render was retried: {calls}"


def test_call_render_passes_arm_to_worker_wrapper():
    """The driver's own wrapper must accept `arm` -- it took `_arm` and lost it."""
    import inspect
    src = RMS.WORKER_SRC
    ns = {}
    got = {}

    def fake(out_root, result, arm=""):
        got["arm"] = arm

    # the wrapper as the worker defines it, extracted by signature contract
    assert "def render(out_root, result, arm=\"\")" in src, (
        "worker render wrapper no longer takes `arm`; _call_render will drop it")
    PP._call_render(fake, "o", FakeResult(), "armB")
    assert got["arm"] == "armB"
