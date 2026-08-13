"""Contract tests for the targeted-bias override path (added 2026-08-10).

The override exists so a pilot can place (bias_e, bias_i) on a DESIGNED grid
relative to the 6/28/26 seizure boundary instead of accepting whatever the seed
happened to draw.  That is only sound if adding it changed nothing else.  These
tests pin the two properties the design depends on:

  A. NO-OP WHEN ABSENT.  With no override, every value drawn from init_rng is
     what it was before the change.  Guarded by hardcoded expected biases for
     seeds 0 and 4 -- seed 4's pair also appears in the 2026-08-01 handoff, so
     the constant is checkable against a document outside this repo.

  B. STREAM POSITION PRESERVED.  With an override, the biases change and
     NOTHING else does: identical connectivity (edge counts, CSR structure, and
     the value arrays themselves) and identical MNIST weights.  This is what
     makes seed a pure connectivity replicate and the bias an independent axis.
     It works because build_model still performs both uniform draws and then
     discards them, so init_rng is left at the same position.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

import wc_config as cfg
import wc_experiment as X

DEV = torch.device("cpu")

# The biases seeds 0 and 4 draw under the canonical order.  Independently
# reproducible from SeedSequence(seed).spawn(2)[0] with the documented draw
# sequence; seed 4's pair is also recorded in SESSION_HANDOFF_2026-08-01.md.
EXPECTED_DRAWN = {
    0: (-4.616, -4.224),
    4: (-1.695, -10.780),
}


def _build(seed, config):
    X.seed_everything(seed)
    init_rng, config_rng = X.make_rngs(seed)
    alpha_H, tau_H = X.resolve_alpha_tau(config, config_rng)
    return X.build_model(seed, config, DEV, init_rng, alpha_H, tau_H)


@pytest.mark.parametrize("seed", sorted(EXPECTED_DRAWN))
def test_no_override_reproduces_canonical_draws(seed):
    """A: absent an override, the drawn biases are unchanged."""
    m = _build(seed, cfg.make_config("simultaneous"))
    be = float(m.bias_e.detach().cpu())
    bi = float(m.bias_i.detach().cpu())
    exp_e, exp_i = EXPECTED_DRAWN[seed]
    assert be == pytest.approx(exp_e, abs=1e-3), f"bias_e drifted: {be} != {exp_e}"
    assert bi == pytest.approx(exp_i, abs=1e-3), f"bias_i drifted: {bi} != {exp_i}"


@pytest.mark.parametrize("config_name", ["simultaneous", "sequential_noH", "sequential_H"])
def test_override_sets_biases_and_perturbs_nothing_else(config_name):
    """B: the override changes the two biases and leaves the network identical."""
    seed = 7
    base_cfg = cfg.make_config(config_name)
    ovr_cfg = dataclasses.replace(base_cfg, bias_e_override=-1.25, bias_i_override=-6.5)

    base = _build(seed, base_cfg)
    ovr = _build(seed, ovr_cfg)

    # the biases ARE overridden
    assert float(ovr.bias_e.detach().cpu()) == pytest.approx(-1.25, abs=1e-6)
    assert float(ovr.bias_i.detach().cpu()) == pytest.approx(-6.5, abs=1e-6)
    # and they actually differ from the draw, or the test proves nothing
    assert float(base.bias_e.detach().cpu()) != pytest.approx(-1.25, abs=1e-6)

    # connectivity is bit-identical: structure AND values, all three matrices
    for name in ("CeLoc", "CeRem", "CeLocI"):
        for part in ("crow_indices", "col_indices", "values"):
            a = getattr(base, f"{name}_{part}").detach().cpu().numpy()
            b = getattr(ovr, f"{name}_{part}").detach().cpu().numpy()
            assert a.shape == b.shape, f"{name}_{part} shape changed"
            assert np.array_equal(a, b), f"{name}_{part} values changed"

    # every other learnable parameter is bit-identical too
    for pname in ("mnist_weights", "inh_to_exc_gain", "output_weights"):
        if not hasattr(base, pname):
            continue
        a = getattr(base, pname).detach().cpu().numpy()
        b = getattr(ovr, pname).detach().cpu().numpy()
        assert np.array_equal(a, b), f"{pname} changed under override"


def test_partial_override_leaves_the_other_bias_drawn():
    """Overriding one bias must not silently move the other."""
    seed = 0
    base = _build(seed, cfg.make_config("simultaneous"))
    part = _build(seed, dataclasses.replace(cfg.make_config("simultaneous"),
                                            bias_e_override=-2.0))
    assert float(part.bias_e.detach().cpu()) == pytest.approx(-2.0, abs=1e-6)
    assert float(part.bias_i.detach().cpu()) == pytest.approx(
        float(base.bias_i.detach().cpu()), abs=1e-9)


def test_fingerprint_records_designed_biases():
    """Two design points at one seed must not be resume-compatible, and an
    unoverridden config must keep its historical fingerprint hash."""
    import param_explore as PE
    base = cfg.make_config("simultaneous")
    a = dataclasses.replace(base, bias_e_override=-1.0, bias_i_override=-5.0)
    b = dataclasses.replace(base, bias_e_override=-3.0, bias_i_override=-9.0)
    h0 = PE.model_fingerprint(base)
    assert "bias_e_override" not in h0, "unoverridden fingerprint must be unchanged"
    assert PE.model_fingerprint(a)["hash"] != PE.model_fingerprint(b)["hash"]
    assert PE.model_fingerprint(a)["hash"] != h0["hash"]
