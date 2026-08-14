"""Regression tests for derived-config provenance (the lead-in / span gate).

Failure being locked down: LEAD_IN_S and LFP_SPAN change what LABEL a run gets
without changing its dynamics, and neither was recorded per-run. Worse, both are
read at TWO moments -- render time (pixel features come from the stored lfp.png)
and classify time (analytic features are measured from arrays.npz under whatever
cfg.LEAD_IN_S is in force then). A config edit between a sweep and a later
re-classification therefore produced feature rows whose two halves were measured
under different windows, with nothing on disk to reveal it.

That is not a cosmetic difference. Measured on a reference bundle, moving the
lead-in 0.0 -> 1.0 s takes amp_prominence from 1.482 to 1.091, straddling
TH['SPIKE_AMP_MIN'] = 1.0 -- i.e. it flips the epileptiform gate and changes the
label.

These tests use hand-written manifests rather than real bundles: the mechanism is
in what gets recorded and compared, not in the model.
"""
import json
from pathlib import Path

import pytest

import classify_runs as CR
import param_persist as PP
from backfill_derived_config import backfill_one


class FakeCfg:
    LFP_SPAN = 240.0
    LEAD_IN_S = 1.0
    OUTPUT_AVERAGE_WINDOW = 0.01
    IMAGE_DURATION = 0.25


def _manifest(tmp_path, derived=None, git_commit="bc58019"):
    d = tmp_path / "simultaneous" / "run_000"
    d.mkdir(parents=True)
    m = {"run_id": "seed000__simultaneous", "seed": 0,
         "config_name": "simultaneous", "git_commit": git_commit}
    if derived is not None:
        m["derived_config"] = derived
    (d / "manifest.json").write_text(json.dumps(m, indent=2))
    return d


# --- what gets recorded ----------------------------------------------------

def test_derived_config_records_the_label_bearing_knobs():
    dc = PP.derived_config(FakeCfg)
    assert dc["lfp_span"] == 240.0
    assert dc["lead_in_s"] == 1.0
    assert dc["source"] == "recorded"


def test_derived_config_is_not_in_the_model_fingerprint():
    """These MUST stay out of the fingerprint.

    is_complete() recomputes any run whose fingerprint disagrees, so folding a
    render knob in would mark an entire corpus stale and re-simulate ~18 GB of
    trajectories in order to change a picture.
    """
    import param_explore as PE
    import wc_config as cfg
    fp = PE.model_fingerprint(cfg.make_config("simultaneous"))
    flat = json.dumps(fp).lower()
    assert "lfp_span" not in flat
    assert "lead_in" not in flat


# --- the three-state comparison -------------------------------------------

def test_verdict_match_and_conflict():
    assert CR._derived_verdict({"lead_in_s": 1.0}, 1.0) == CR.DERIVED_MATCH
    assert CR._derived_verdict({"lead_in_s": 1.0}, 2.5) == CR.DERIVED_CONFLICT


def test_absent_record_is_unknown_not_match():
    """The load-bearing case: no record must NOT read as agreement.

    A boolean comparison would have to call this either a conflict (blocking
    every legacy bundle) or a match (silently accepting a window mismatch --
    the exact failure this exists to prevent).
    """
    assert CR._derived_verdict({}, 1.0) == CR.DERIVED_UNKNOWN
    assert CR._derived_verdict({"lfp_span": 240.0}, 1.0) == CR.DERIVED_UNKNOWN
    assert CR._derived_verdict({"lead_in_s": None}, 1.0) == CR.DERIVED_UNKNOWN


def test_unknown_is_refused_alongside_conflict():
    """Callers gate on == MATCH, so UNKNOWN must not pass the gate."""
    for rec in ({}, {"lead_in_s": 2.5}):
        assert CR._derived_verdict(rec, 1.0) != CR.DERIVED_MATCH


def test_float_reprs_round_trip_as_match():
    """Tolerance absorbs JSON/CSV float repr only -- not real differences."""
    assert CR._derived_verdict({"lead_in_s": 1.0000000000001}, 1.0) == CR.DERIVED_MATCH
    assert CR._derived_verdict({"lead_in_s": 1.001}, 1.0) == CR.DERIVED_CONFLICT


def test_span_is_recorded_but_not_compared():
    """lfp_span has no measure-time counterpart: this process reads the stored
    PNG rather than redrawing it, so there is no current value to disagree with.
    A span-only difference must not be reported as a conflict."""
    assert CR._derived_verdict(
        {"lead_in_s": 1.0, "lfp_span": 150.0}, 1.0) == CR.DERIVED_MATCH


# --- backfill -------------------------------------------------------------

def test_backfill_marks_values_as_reconstructed(tmp_path):
    """An inferred value must never be indistinguishable from an observed one."""
    d = _manifest(tmp_path)
    assert backfill_one(d) == "written"
    dc = json.loads((d / "manifest.json").read_text())["derived_config"]
    assert dc["source"] == "reconstructed"
    assert dc["lead_in_s"] == 1.0
    assert "reconstruction_evidence" in dc


def test_backfill_uses_render_era_span_not_the_committed_one(tmp_path):
    """The 60 pilot runs commit 6ebe0b5, whose config says LFP_SPAN = 150.0 --
    but their figures were re-rendered at 240 after the recalibration, and the
    classifier reads the stored figure. Taking the committed value would write a
    false 150 onto exactly the runs whose provenance was hardest to establish."""
    d = _manifest(tmp_path, git_commit="6ebe0b5")
    backfill_one(d)
    dc = json.loads((d / "manifest.json").read_text())["derived_config"]
    assert dc["lfp_span"] == 240.0
    assert "note" in dc and "150" in dc["note"]


def test_backfill_will_not_clobber_a_recorded_block(tmp_path):
    d = _manifest(tmp_path, derived={"lead_in_s": 1.0, "source": "recorded"})
    assert backfill_one(d) == "skipped_recorded"
    dc = json.loads((d / "manifest.json").read_text())["derived_config"]
    assert dc["source"] == "recorded"


def test_backfill_is_idempotent(tmp_path):
    d = _manifest(tmp_path)
    assert backfill_one(d) == "written"
    assert backfill_one(d) == "skipped_present"


def test_backfill_reports_missing_manifest(tmp_path):
    assert backfill_one(tmp_path / "nonexistent") == "no_manifest"
