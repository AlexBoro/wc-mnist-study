"""analytic_features.py — LFP features measured from arrays.npz, not from pixels.

WHY THIS EXISTS
---------------
``features.py`` measures the LFP by opening the rendered ``lfp.png``, building an
ink mask, and inferring per-channel traces from vertical runs of ink.  That was
the only option when figures were the sole per-run artifact.  It carries three
costs that matter for the Type 0/1 boundary:

  1. Trace MERGING.  A large deflection fuses with its neighbours into one ink
     run, so amplitude must be attributed by centroid ownership rather than
     measured.  Near the 1.0-spacing bar this attribution is exactly where the
     signal is weakest.
  2. RASTERISATION.  A 1500x750 canvas gives ~19 px per channel, so amplitude is
     quantised to ~0.05 spacing and sub-pixel transients are lost entirely.
  3. ANTI-ALIASING and line width put a floor under apparent deflection size.

Every run now saves ``arrays.npz``, and the stacked plot uses a FIXED offset
(``wc_figures.LFP_STACK_OFFSET``), so the user's criterion — "the amplitude of
the transient is sufficient to cross the baseline of the channel above" — is
exactly computable: reconstruct the same LFP proxy the figure plots, apply the
same high-pass filter, and divide deflections by that offset.

WHAT IS MEASURED
----------------
``amp_excursion``  peak |deviation from 0| per channel, in spacing units.  The
                   literal reading of the criterion, and the analytic counterpart
                   of ``features.spike_amp_max``.
``amp_prominence`` peak PROMINENCE in spacing units.  Prominence measures a
                   transient against its own LOCAL baseline, so a slowly drifting
                   trace cannot accumulate excursion and read as a spike.  On the
                   2026-08-12 sweep this separated Type 0 from Type 1 better than
                   excursion (9% vs 15% class overlap), though neither is clean.
``width_ms``       full width at half prominence of that peak — the "sharp" half
                   of "sharp transients standing out from the background".

Both amplitude measures are reported so the Type 1 threshold can be fitted to
adjudicated calls rather than assumed; ``classifier.py`` reads whichever
``SPIKE_AMP_FEATURE`` names.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths

# Peaks below this prominence (in spacing units) are not candidate transients.
# Well under the 1.0 criterion: this only bounds the search, it is not a decision.
_MIN_PROM_SPACING = 0.15

# A transient wider than this is a wave, not a spike.  Used only for the
# descriptive n_sharp count, never as a gate.
_SHARP_MAX_MS = 200.0


def lfp_from_arrays(run_dir, reconstruct_connectivity, cfg, wc_figures,
                    lead_in=None):
    """Rebuild the filtered LFP proxy exactly as ``wc_figures.save_lfp`` plots it.

    Returns ``(filt, times, sample_rate, offset)`` where ``filt`` is
    ``[n_samples, n_electrodes]`` in filtered signal units and ``offset`` is the
    inter-channel spacing those units must be divided by.

    The coefficient handling mirrors ``save_lfp``: the proxy IS the model's
    excitatory update line restricted to the electrode rows, so the run's own
    ``w_celoc``/``w_cerem`` are used, not the config defaults.

    ``lead_in`` (seconds, default ``cfg.LEAD_IN_S``) drops the opening window
    BEFORE filtering, matching ``param_render.render_run`` so the analytic and
    pixel features describe the same window.  This matters: the classifier reads
    the analytic amplitude (``classifier.SPIKE_AMP_FEATURE``), so trimming only
    the rendered figures would leave the epileptiform gate measuring the opening
    transient.  Pass ``lead_in=0.0`` for the untrimmed measure.
    """
    run_dir = Path(run_dir)
    z = np.load(run_dir / "arrays.npz")
    E = z["E_traj"].astype(float)
    I = z["I_traj"].astype(float)
    t = z["times"].astype(float)

    # Measurement window -- see the ``lead_in`` note above. Guarded identically
    # to render_run so a lead-in longer than the run cannot empty the arrays.
    _lead = float(getattr(cfg, "LEAD_IN_S", 0.0) if lead_in is None else lead_in)
    if _lead > 0.0 and len(t) > 1:
        _k = int(np.searchsorted(t, t[0] + _lead))
        if 0 < _k < len(t) - 1:
            E, I, t = E[_k:], I[_k:], t[_k:]
    man = json.loads((run_dir / "manifest.json").read_text())
    seed, cname = int(man["seed"]), str(man["config_name"])
    arm = str(man.get("arm", "") or "")

    CeLoc, CeRem, wL, wR = reconstruct_connectivity(seed, cname, arm=arm)
    electrodes, _ = wc_figures._ring_electrodes(E.shape[1], 28)

    self_term = E[:, electrodes]
    inh_term = I[:, electrodes]
    loc_term = np.asarray(CeLoc[electrodes, :] @ E.T).T
    rem_term = np.asarray(CeRem[electrodes, :] @ E.T).T
    LFP = (cfg.W_PY2PY_SELF * self_term + wL * loc_term
           + wR * rem_term + cfg.W_LFP_INH * inh_term)

    dt_eff = float(t[1] - t[0]) if len(t) > 1 else float(cfg.DT)
    sr = 1.0 / dt_eff if dt_eff > 0 else 1.0
    filt = wc_figures.filter_eeg(LFP, 1.0, sr, high_or_low="high", order=4)
    return filt, t, sr, float(wc_figures.LFP_STACK_OFFSET)


def transient_features(filt, sr, offset):
    """Per-run discrete-transient features, all amplitudes in spacing units.

    Scans both polarities on every channel: an epileptiform transient may deflect
    either way and the criterion is on |amplitude|.
    """
    n_ch = filt.shape[1]
    exc = np.abs(filt).max(axis=0) / offset

    best_prom = 0.0
    best_width = float("nan")
    best_chan = -1
    n_sharp = 0
    prom_per_ch = np.zeros(n_ch)

    for ch in range(n_ch):
        y = filt[:, ch]
        for sgn in (1.0, -1.0):
            sig = sgn * y
            pk, _ = find_peaks(sig, prominence=_MIN_PROM_SPACING * offset)
            if not len(pk):
                continue
            pr = peak_prominences(sig, pk)[0] / offset
            wd = peak_widths(sig, pk, rel_height=0.5)[0] / sr * 1000.0
            if pr.max() > prom_per_ch[ch]:
                prom_per_ch[ch] = float(pr.max())
            n_sharp += int(np.sum((pr >= 1.0) & (wd <= _SHARP_MAX_MS)))
            i = int(np.argmax(pr))
            if pr[i] > best_prom:
                best_prom = float(pr[i])
                best_width = float(wd[i])
                best_chan = ch

    return dict(
        amp_excursion=round(float(exc.max()), 3),
        amp_prominence=round(best_prom, 3),
        width_ms=None if np.isnan(best_width) else round(best_width, 1),
        peak_channel=best_chan,
        n_chans_excursion_ge1=int((exc >= 1.0).sum()),
        n_chans_prominence_ge1=int((prom_per_ch >= 1.0).sum()),
        n_sharp_transients=int(n_sharp),
    )


def extract(run_dir, reconstruct_connectivity, cfg, wc_figures, lead_in=None):
    """Full analytic feature dict for one run directory.

    ``lead_in`` is forwarded to ``lfp_from_arrays``; None means cfg.LEAD_IN_S.
    """
    filt, _t, sr, offset = lfp_from_arrays(
        run_dir, reconstruct_connectivity, cfg, wc_figures, lead_in=lead_in)
    out = transient_features(filt, sr, offset)
    out["lfp_offset"] = offset
    return out
