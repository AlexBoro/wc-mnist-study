"""wc_figures.py — Per-measurement-epoch activity figures (§12).

Saves four figure types from the last training-window E/I activity:
  heatmap_E.png, heatmap_I.png  — "raster-style" spatiotemporal heatmaps of E/I,
                                  fixed 0-1 color scale + colorbar + stim marks
                                  (the notebook's seizure-annotation view)
  lfp.png                       — filtered LFP proxy at 28 ring-spaced electrodes
                                  (weighted E/I + local/remote synaptic drive,
                                  zero-phase high-pass, ring-labeled, stim marks)
  lfp_focus.png                 — NEW focus-targeted virtual-electrode LFP at the
                                  heatmap-detected hotspot (unfiltered + highpass),
                                  addressing the sparse-electrode blind spot (§12)

matplotlib is imported lazily (Agg backend) so importing this module is cheap and
headless-safe.  Each ``save_*`` returns the output path; ``save_all_figures``
returns a list of (fig_type, path) for the manifest.

The figure *interface* (the fig_type names and the manifest columns) is the
contract consumed by the later epileptiform-classification step, so it must stay
stable even if the rendering details evolve.  The pre-refactor dot-scatter
"raster" (E>0.5) was removed: it had no notebook antecedent, misrepresents the
WC rate model, and rendered blank; ``labeling_toolkit`` consumes the E heatmap
(kind='heatmap'), which is what the notebook called its "raster-style heatmap".
"""

from __future__ import annotations

import os
from math import gcd

import numpy as np

import wc_config as cfg

# MNIST row length: images enter the ring as img.view(-1), so pixel (r, c) maps
# to unit MNIST_ROW_LEN*r + c.  Used only to de-alias electrode placement.
MNIST_ROW_LEN = 28

FIG_TYPES = ("heatmap_E", "heatmap_I", "lfp", "lfp_focus")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _highpass(sig: np.ndarray, dt: float, cutoff_hz: float = 1.0) -> np.ndarray:
    """High-pass filter a 1-D signal.  Uses scipy butter/filtfilt if available,
    else a moving-average detrend fallback."""
    sig = np.asarray(sig, dtype=float)
    try:
        from scipy.signal import butter, filtfilt
        fs = 1.0 / dt
        nyq = 0.5 * fs
        wn = min(max(cutoff_hz / nyq, 1e-4), 0.99)
        b, a = butter(2, wn, btype="high")
        if len(sig) > 9:
            return filtfilt(b, a, sig)
    except Exception:
        pass
    # Fallback: subtract a centered moving average
    win = max(3, int(round((1.0 / cutoff_hz) / dt)))
    win = min(win, len(sig) if len(sig) % 2 == 1 else len(sig) - 1)
    if win < 3:
        return sig - sig.mean()
    kernel = np.ones(win) / win
    trend = np.convolve(sig, kernel, mode="same")
    return sig - trend


def filter_eeg(eeg: np.ndarray, cutoff: float, sample_rate: float,
               high_or_low: str = "high", order: int = 4) -> np.ndarray:
    """Zero-phase (forward-backward) IIR filter over axis 0 (notebook-verbatim).

    Falls back to per-channel mean subtraction if scipy is unavailable or the
    signal is too short for the filter.
    """
    eeg = np.asarray(eeg, dtype=float)
    if eeg.ndim == 1:
        eeg = eeg[:, None]
    try:
        from scipy.signal import butter, lfilter
        ny = sample_rate / 2.0
        wn = np.clip(np.asarray(cutoff, dtype=float) / ny, 1e-4, 0.99)
        b, a = butter(order, wn, btype=high_or_low)
        if eeg.shape[0] > 3 * max(len(a), len(b)):
            fwd = lfilter(b, a, eeg, axis=0)
            back = lfilter(b, a, fwd[::-1, :], axis=0)
            return back[::-1, :]
    except Exception:
        pass
    return eeg - eeg.mean(axis=0, keepdims=True)


def _ring_electrodes(num_cells: int, n_elec: int = 28):
    """Evenly-spaced virtual electrodes on the 1-D recurrent ring.

    The WC units live on a directed ring lattice (unit ``i`` wired to
    ``i ± 1 … ± k/2  (mod N)``), so **ring distance on the linear index** — not any
    2-D grid — is what determines connectivity locality.  We therefore place
    ``n_elec`` electrodes at uniform ring spacing and label them by ring position.

    (Deviation from the notebook's 5x5 ``_select_torus_electrodes``, which imposed
    a 28x28 pixel-grid metric unrelated to the ring; see spec §12.)

    Stride: uniform ring spacing at N=784, n_elec=28 gives stride exactly 28 --
    which is also the image row length, since MNIST enters as ``img.view(-1)`` so
    pixel (r, c) -> unit 28r + c.  The stride therefore ALIASES with the row
    length and every electrode lands in image column 0:

        units 0, 28, 56, ... 756  ->  pixels (0,0), (1,0), (2,0), ... (27,0)

    Column 0 carries essentially no MNIST ink (mean occupancy ~0.000), so the
    electrodes sat on the units receiving the least direct stimulus in the image.
    Using stride 29 (coprime with 28) instead walks the electrodes diagonally
    across the image, visiting a different column each row, so the sampled units
    span the ink distribution rather than one empty edge.

    This changes WHICH units are plotted, so LFP figures before and after are not
    comparable and labels must be re-validated.  Ring spacing is preserved to
    within one unit, so connectivity locality -- the thing the ring metric is for
    -- is unaffected.

    Returns ``(linear_indices, labels)``.
    """
    n_elec = max(1, min(int(n_elec), int(num_cells)))
    stride = num_cells / n_elec
    row_len = MNIST_ROW_LEN
    de_alias = (num_cells % row_len == 0                       # ring indexes an image
                and abs(stride - round(stride)) < 1e-9         # integer stride
                and int(round(stride)) > 1
                and gcd(int(round(stride)), row_len) != 1)     # and it aliases
    if de_alias:
        s = int(round(stride))
        while gcd(s, row_len) != 1 and s < num_cells:
            s += 1
        idx = (np.arange(n_elec) * s % num_cells).astype(int)
    else:
        idx = (np.round(np.arange(n_elec) * stride) % num_cells).astype(int)
    idx = sorted(set(int(i) for i in idx))
    labels = [f"u{i:03d}" for i in idx]
    return idx, labels


def _stim_onsets_in_window(t0: float, t1: float, image_duration_s: float,
                           pulsed: bool = False, pulse_on_s: float = 1.0,
                           pulse_off_s: float = 1.0, align_t0: float = 0.0) -> np.ndarray:
    """MNIST image onset times within [t0, t1] (absolute time). Notebook-verbatim."""
    try:
        t0 = float(t0); t1 = float(t1); image_duration_s = float(image_duration_s)
    except Exception:
        return np.array([], dtype=float)
    if image_duration_s <= 0 or t1 <= t0:
        return np.array([], dtype=float)
    eps = 1e-9
    if not pulsed:
        k0 = int(np.ceil((t0 - align_t0) / image_duration_s - eps))
        k1 = int(np.floor((t1 - align_t0) / image_duration_s + eps))
        if k1 < k0:
            return np.array([], dtype=float)
        return (align_t0 + np.arange(k0, k1 + 1) * image_duration_s).astype(float)
    images_per_block = max(int(round(pulse_on_s / image_duration_s)), 1)
    block_dur = images_per_block * image_duration_s + pulse_off_s
    if block_dur <= 0:
        return np.array([], dtype=float)
    b0 = int(np.floor((t0 - align_t0) / block_dur)) - 1
    b1 = int(np.ceil((t1 - align_t0) / block_dur)) + 1
    onsets = []
    for b in range(b0, b1 + 1):
        block_start = align_t0 + b * block_dur
        for j in range(images_per_block):
            tt = block_start + j * image_duration_s
            if (tt + eps) >= t0 and (tt - eps) <= t1:
                onsets.append(tt)
    return np.array(sorted(onsets), dtype=float) if onsets else np.array([], dtype=float)


def _add_stim_ticks(ax, stim_times, color="black", height_frac=0.03, offset_frac=0.015):
    """Short vertical ticks at stim onset times, near the bottom of the axes."""
    stim_times = np.asarray(stim_times, dtype=float).ravel()
    stim_times = stim_times[np.isfinite(stim_times)]
    if stim_times.size == 0:
        return
    y0, y1 = ax.get_ylim()
    yr = (y1 - y0) if (y1 != y0) else 1.0
    y_base = y0 + offset_frac * yr
    ax.vlines(stim_times, y_base, y_base + height_frac * yr,
              colors=color, linewidth=1.0, zorder=20, clip_on=True)


def detect_hotspot(E_window: np.ndarray, half_width: int = 4) -> tuple[int, list[int]]:
    """Heatmap-detected hotspot: the peak unit and a contiguous ring window around it.

    Returns ``(center_unit_index, member_unit_indices)`` where the members are the
    ``2*half_width + 1`` units centered on the peak along the 1-D recurrent ring,
    with wraparound (``(center + d) % N``).

    The window is taken on the *ring* — the WC connectivity metric
    (``wc_model.generate_ring_lattice_directed``; unit ``i`` wired to
    ``i ± 1 … ± k/2 mod N``) — so a focus is a contiguous ring arc, consistent
    with the ring-spaced ``save_lfp`` electrodes.  (Earlier versions used a 3x3
    patch on the sqrt(N) *pixel* grid, which mixed ring-adjacent units with
    ring-distant ones ~sqrt(N) away; see §12.)  ``half_width=4`` keeps the old
    9-unit footprint while making it geometrically meaningful; the ring's local
    connectivity range (±k/2 = ±78) is far wider, so a small arc keeps the
    electrode focal rather than smearing across the ring.
    """
    mean_act = E_window.mean(axis=0)            # [N]
    N = mean_act.shape[0]
    center = int(np.argmax(mean_act))
    # Clamp the half-width so the window never wraps onto itself when N is tiny
    # (2*half_width+1 <= N); a no-op for the real N=784.  Guards against duplicate
    # members that would over-weight units in the mean-E readout.
    hw = min(int(half_width), (N - 1) // 2)
    members = [int((center + d) % N) for d in range(-hw, hw + 1)]
    return center, members


def save_heatmap(activity: np.ndarray, times: np.ndarray, path: str, title: str,
                 cbar_label: str = "activity", image_duration: float | None = None,
                 pulsed: bool = False):
    """Spatiotemporal "raster-style" heatmap: units (rows) x time (cols).

    Notebook-faithful (§12): a **fixed absolute** ``vmin=0, vmax=1`` viridis image
    (E/I are sigmoid rates in (0,1), so the scale is comparable across epochs),
    with a colorbar and MNIST stimulus-onset hashmarks.  This is the seizure-
    annotation substrate consumed by ``labeling_toolkit`` (kind='heatmap').
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 4))
    if activity.size == 0:
        ax.text(0.5, 0.5, "no activity recorded", ha="center", va="center")
    else:
        extent = [float(times[0]) if len(times) else 0.0,
                  float(times[-1]) if len(times) else 1.0,
                  0, activity.shape[1]]
        im = ax.imshow(activity.T, aspect="auto", origin="lower",
                       extent=extent, cmap="viridis", interpolation="nearest",
                       vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax, label=cbar_label)
        if image_duration and len(times) > 1:
            stim = _stim_onsets_in_window(times[0], times[-1], image_duration, pulsed=pulsed)
            _add_stim_ticks(ax, stim, color="black")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("unit")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# Inter-channel spacing of the stacked LFP traces, in FILTERED SIGNAL UNITS.
# The taxonomy's amplitude criterion ("the transient crosses the baseline of
# the channel above") is expressed in units of this spacing, so any measurement
# of that criterion -- pixel-based or analytic -- must divide by THIS number.
LFP_STACK_OFFSET = 5.0


def save_lfp(E_window: np.ndarray, times: np.ndarray, path: str, dt: float,
             I_window: np.ndarray | None = None, CeLoc=None, CeRem=None,
             image_duration: float | None = None, pulsed: bool = False,
             cutoff_hz: float = 1.0, w_celoc: float | None = None,
             w_cerem: float | None = None, span: float | None = None,
             return_stats: bool = False):
    """Sparse filtered LFP proxy at 28 ring-spaced virtual electrodes (§12).

    Reconstructs the biophysical LFP proxy at evenly-spaced electrodes on the 1-D
    recurrent ring (the WC connectivity metric — see ``_ring_electrodes``):

        LFP = W_PY2PY_SELF*self_term + W_CELOC*loc_term
              + W_CEREM*rem_term + W_LFP_INH*inh_term

    The three excitatory coefficients are EXACTLY the model's Py2Py coefficients
    (wc_config), because this proxy is the model's own excitatory update line
    restricted to the electrode rows.  ``w_celoc``/``w_cerem`` override them for
    a run whose model was built with non-default coefficients — pass
    ``model.w_celoc``/``model.w_cerem`` so the proxy tracks the model.

    where ``loc_term``/``rem_term`` are the local/remote excitatory synaptic drive
    into each electrode (``CeLoc``/``CeRem`` are SciPy CSR [N,N] connectivity).
    The stacked traces are zero-phase high-pass filtered (order 4), labeled by
    ring position, and annotated with MNIST stimulus onsets.  If ``I_window`` or
    the connectivity is missing, the corresponding term is dropped rather than
    failing.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 5))
    if E_window.size == 0:
        ax.text(0.5, 0.5, "no activity recorded", ha="center", va="center")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Electrode (stacked)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    N = E_window.shape[1]
    electrodes, labels = _ring_electrodes(N, 28)
    E_count = len(electrodes)

    self_term = E_window[:, electrodes]
    if I_window is not None and getattr(I_window, "size", 0):
        inh_term = np.asarray(I_window, dtype=float)[:, electrodes]
    else:
        inh_term = np.zeros_like(self_term)
    if CeLoc is not None:
        loc_term = np.asarray(CeLoc[electrodes, :] @ E_window.T).T
    else:
        loc_term = np.zeros_like(self_term)
    if CeRem is not None:
        rem_term = np.asarray(CeRem[electrodes, :] @ E_window.T).T
    else:
        rem_term = np.zeros_like(self_term)

    wL = cfg.W_CELOC if w_celoc is None else float(w_celoc)
    wR = cfg.W_CEREM if w_cerem is None else float(w_cerem)
    LFP = (cfg.W_PY2PY_SELF * self_term + wL * loc_term
           + wR * rem_term + cfg.W_LFP_INH * inh_term)

    dt_eff = float(times[1] - times[0]) if len(times) > 1 else float(dt)
    sample_rate = 1.0 / dt_eff if dt_eff > 0 else 1.0
    filt = filter_eeg(LFP, cutoff_hz, sample_rate, high_or_low="high", order=4)

    offsets = np.arange(E_count) * LFP_STACK_OFFSET
    stacked = filt + offsets[None, :]
    for ch in range(E_count):
        ax.plot(times, stacked[:, ch], lw=0.7)

    # FIXED y-axis: a constant span centered on the electrode-ladder midpoint so
    # the pixel-per-signal-unit scale is IDENTICAL on every epoch (no per-plot
    # autoscale). This makes a fixed pixel-deflection threshold transferable and
    # keeps flat epochs readable. Span 150 (calibrated 2026-07-25): flat & moderate
    # seizures do not clip; only the extreme tips of the largest ictal peaks
    # (ep47/ep99) truncate on the top/bottom electrodes — acceptable since those
    # deflections are already unambiguously supra-threshold.
    #
    # The span now lives in wc_config (LFP_SPAN) rather than here, because the
    # E->E rework changes the signal amplitude: 150 was calibrated against the
    # OLD coefficients, and holding it fixed while the drive changes would silently
    # change pixels-per-signal-unit, breaking the transferability the fixed axis
    # exists to provide. Recalibrate from the pilot's measured peak deflections and
    # set cfg.LFP_SPAN so pixels-per-signal-unit matches the old figures.
    _ladder_mid = (E_count - 1) * 5.0 / 2.0
    _span = float(cfg.LFP_SPAN if span is None else span)
    ax.set_ylim(_ladder_mid - _span / 2.0, _ladder_mid + _span / 2.0)

    # MNIST stimulus onset marks (needs ylim, so after plotting)
    if image_duration and len(times) > 1:
        stim = _stim_onsets_in_window(times[0], times[-1], image_duration, pulsed=pulsed)
        _add_stim_ticks(ax, stim, color="black")

    ax.set_yticks(offsets)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Ring electrode (stacked)")
    ax.set_title(f"Filtered LFP proxy — {E_count} ring electrodes (high-pass > {cutoff_hz:g} Hz)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if return_stats:
        # Peak deflection of the FILTERED, un-offset traces -- the quantity the
        # span must accommodate. Reported so cfg.LFP_SPAN can be recalibrated
        # from the pilot's measured amplitudes rather than set by eye.
        # The span must accommodate the LADDER (fixed at (E_count-1)*5.0) PLUS the
        # signal excursion above/below it -- the ladder dominates, so a
        # recommendation based on peak amplitude alone is badly wrong.
        # pixels-per-signal-unit depends ONLY on _span, so holding _span fixed
        # already preserves the old figures' scale; the live question is whether
        # deflections clip.
        _ladder_h = (E_count - 1) * 5.0
        _peak = float(np.abs(filt).max())
        return dict(path=path, span=_span, ladder_height=_ladder_h,
                    peak_abs=_peak,
                    p99_abs=float(np.percentile(np.abs(filt), 99)),
                    per_electrode_peak=np.abs(filt).max(axis=0).tolist(),
                    span_needed=_ladder_h + 2.0 * _peak,
                    headroom=_span - _ladder_h,
                    clipped_frac=float(np.mean(
                        np.abs(stacked - _ladder_mid) > _span / 2.0)))
    return path


def save_lfp_focus(E_window: np.ndarray, times: np.ndarray, path: str, dt: float):
    """Focus-targeted virtual-electrode LFP at the detected hotspot (§12).

    Top: unfiltered mean-E of the hotspot ring window.  Bottom: high-pass filtered
    version (reveals fast epileptiform components the sparse electrode array can
    miss).
    """
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    if E_window.size == 0:
        ax1.text(0.5, 0.5, "no activity recorded", ha="center", va="center")
    else:
        _, members = detect_hotspot(E_window)
        virtual = E_window[:, members].mean(axis=1)
        ax1.plot(times, virtual, lw=0.8, color="C3")
        ax1.set_ylabel("E (raw)")
        ax1.set_title(f"Focus virtual-electrode LFP (hotspot ring window, {len(members)} units)")
        ax2.plot(times, _highpass(virtual, dt), lw=0.8, color="C1")
        ax2.set_ylabel("E (highpass)")
    ax2.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def save_all_figures(out_dir: str, E_window: np.ndarray, I_window: np.ndarray,
                     times: np.ndarray, dt: float,
                     CeLoc=None, CeRem=None, image_duration: float | None = None,
                     pulsed: bool = False):
    """Render the four per-epoch figure types into ``out_dir``.

    ``image_duration``/``pulsed`` add MNIST stimulus-onset marks to the heatmaps
    and LFP.  ``CeLoc``/``CeRem`` (SciPy CSR [N,N] excitatory connectivity) and
    ``I_window`` feed the notebook-faithful LFP proxy in ``save_lfp``.  Returns a
    list of (fig_type, absolute_path) for the manifest.
    """
    os.makedirs(out_dir, exist_ok=True)
    E_window = np.asarray(E_window, dtype=float)
    I_window = np.asarray(I_window, dtype=float)
    times = np.asarray(times, dtype=float)

    rows = []
    rows.append(("heatmap_E", save_heatmap(E_window, times,
                 os.path.join(out_dir, "heatmap_E.png"), "Excitatory activity (E)",
                 cbar_label="E activity", image_duration=image_duration, pulsed=pulsed)))
    rows.append(("heatmap_I", save_heatmap(I_window, times,
                 os.path.join(out_dir, "heatmap_I.png"), "Inhibitory activity (I)",
                 cbar_label="I activity", image_duration=image_duration, pulsed=pulsed)))
    rows.append(("lfp", save_lfp(E_window, times,
                 os.path.join(out_dir, "lfp.png"), dt,
                 I_window=I_window, CeLoc=CeLoc, CeRem=CeRem,
                 image_duration=image_duration, pulsed=pulsed)))
    rows.append(("lfp_focus", save_lfp_focus(E_window, times,
                 os.path.join(out_dir, "lfp_focus.png"), dt)))
    return rows
