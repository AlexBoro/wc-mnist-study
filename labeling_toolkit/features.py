"""features.py — unified feature extractor for the WC-MNIST epileptiform taxonomy.

ONE function, extract_features(heatmap_path, lfp_path, mean_E=None, val_acc=None),
computes every signal the decision table (classifier.py) reads. See
taxonomy_spec.md for the criterion each feature serves. All figure geometry
assumptions are documented inline; thresholds live in classifier.py, not here
(this module measures, it does not decide).

Figures are the harness's original-resolution PNGs:
  - heatmap_E.png : "Excitatory activity (E)" — unit index (y) vs time (x),
    viridis colormap, colourbar on the right ~last 12% of width.
  - lfp.png       : "Filtered LFP proxy — 28 ring electrodes (stacked)".
Window is a fixed 10 s epoch (x-axis spans exactly 10 s).
"""
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d

N_CHANNELS = 28          # stacked LFP ring electrodes; also heatmap channel count
WINDOW_S   = 10.0        # every epoch figure spans 10 s

# Fixed heatmap interior crop on the uniform 600x1500 canvas (screen_v82 BOX):
# excludes the left axis/labels AND the viridis colourbar on the right. Using a
# fixed BOX (not a fractional crop) is load-bearing — a fractional crop reaches
# into the colourbar and forces max_col_frac -> 1.0 for every figure. Verified to
# reproduce the authoritative CSV ink_frac/max_col_frac exactly.
BOX = dict(y0=58, y1=509, x0=108, x1=1199)


def _max_circular_run(mask):
    """Longest run of consecutive True in a 1-D boolean array treated as a RING
    (index 0 neighbours index N-1). Returns an int in [0, len(mask)].

    All-True -> full length N (the ring is entirely lit). Otherwise the longest
    wrap-around run equals the longest linear run in the doubled array, and can
    never exceed N because at least one False breaks it.
    """
    n = len(mask)
    if n == 0:
        return 0
    if mask.all():
        return int(n)
    best = cur = 0
    for v in np.concatenate([mask, mask]):   # scan twice to catch seam-straddling runs
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return int(min(best, n))


# ---------------------------------------------------------------- heatmap ----
def _heatmap_arrays(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    H, W, _ = a.shape
    G, B = a[:, :, 1], a[:, :, 2]
    # active-pixel rule (viridis bright: high green, yellow highlights); same
    # rule as screen_v82 for continuity of ink_frac / max_col_frac.
    bright_full = (G > 110) & (G >= B)
    # interior crop: use the fixed BOX when the canvas matches the 600x1500
    # convention; otherwise fall back to a conservative fractional interior.
    if (H, W) == (600, 1500):
        y0b, y1b, x0b, x1b = BOX["y0"], BOX["y1"], BOX["x0"], BOX["x1"]
    else:
        y0b, y1b = int(0.10 * H), int(0.97 * H)
        x0b, x1b = int(0.072 * W), int(0.80 * W)
    interior = bright_full[y0b:y1b, x0b:x1b]
    return a, bright_full, interior, (x0b, x1b, y0b, y1b), H, W

def heatmap_features(path):
    a, bright_full, interior, box, H, W = _heatmap_arrays(path)
    x0b, x1b, y0b, y1b = box
    ink_frac = interior.mean()
    coldens = interior.mean(0)            # per-column active fraction (height-normalized)
    rowdens = interior.mean(1)            # per-row active fraction
    max_col_frac = float(np.percentile(coldens, 99)) if coldens.size else 0.0
    band_score   = float(np.percentile(rowdens, 99)) if rowdens.size else 0.0   # ADVISORY

    feats = dict(ink_frac=round(ink_frac, 4),
                 max_col_frac=round(max_col_frac, 4),
                 band_score=round(band_score, 4),
                 n_complexes=0, max_IDI=None, min_IDI=None, median_IDI=None,
                 sustained_fraction=0.0, vertical_spread_fraction=0.0)
    colprof = interior.sum(0).astype(float)
    if colprof.max() == 0:
        return feats
    tspan = interior.shape[1]
    def t(c): return c / tspan * WINDOW_S             # interior column -> seconds

    # --- temporal: segment vertical bursts (complexes) over time ---
    # Threshold is the MAX of a relative cut (0.30 of the peak) and an ABSOLUTE
    # floor (fraction of interior height). The absolute floor is load-bearing:
    # on a flat/silent epoch the relative cut alone segments baseline noise into
    # dozens of phantom "complexes" (peak colprof ~7-12 rows), which then trips
    # the sustained-spike-wave ictal rule. Real complexes recruit >=180 rows;
    # ABS_FLOOR_FRAC*height (~27 rows) cleanly separates the two populations.
    ABS_FLOOR_FRAC = 0.06
    sm = uniform_filter1d(colprof, 7)
    thr = max(sm.max() * 0.30, ABS_FLOOR_FRAC * interior.shape[0])
    above = sm > thr
    bursts = []   # (start_c, end_c, peak_c)
    i = 0
    while i < len(sm):
        if above[i]:
            j = i
            while j < len(sm) and above[j]:
                j += 1
            peak = i + int(np.argmax(sm[i:j]))
            bursts.append((i, j - 1, peak))
            i = j
        else:
            i += 1
    n = len(bursts)
    feats["n_complexes"] = n
    # inter-discharge intervals (seconds) between consecutive complexes
    idis = [t(bursts[k][0]) - t(bursts[k - 1][1]) for k in range(1, n)]
    if idis:
        feats["max_IDI"]    = round(max(idis), 3)
        feats["min_IDI"]    = round(min(idis), 3)
        feats["median_IDI"] = round(float(np.median(idis)), 3)
    # sustained_fraction: fraction of window above threshold (how much of the
    # window is discharge vs quiescence) -> high for ictal, low for discrete.
    feats["sustained_fraction"] = round(float(above.mean()), 3)

    # --- spatial: vertical spread fraction at the strongest burst ---
    # FSW = partial contiguous spread; GSW = essentially full vertical range.
    # Coordinates are interior-relative (bursts index into `interior`).
    if n:
        strongest = max(bursts, key=lambda bb: sm[bb[2]])
        c0, c1 = max(0, strongest[2] - 3), min(interior.shape[1], strongest[2] + 4)
        col_slice = interior[:, c0:c1].any(1)   # any active pixel per row in the burst
        feats["vertical_spread_fraction"] = round(float(col_slice.mean()), 3)
    return feats

# ------------------------------------------------------------------- LFP ----
def lfp_features(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    H, W, _ = a.shape
    mx = a.max(2); mn = a.min(2); sat = mx - mn
    white = (mx > 235) & (sat < 18)
    ink = (sat >= 35) & (mx > 60) & (~white)   # coloured trace ink
    colsum = ink.sum(0); rowsum = ink.sum(1)
    feats = dict(S=0.0, cross_cols=0, n_active_channels=0,
                 fast_activity_duration_s=0.0, sync_spike_extent=0)
    if colsum.max() == 0:
        return feats
    xs = np.where(colsum > colsum.max() * 0.05)[0]
    ys = np.where(rowsum > rowsum.max() * 0.05)[0]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    tspan = x1 - x0

    # FIXED inter-channel spacing (spec: never per-run estimate)
    S = (y1 - y0) / N_CHANNELS
    feats["S"] = round(S, 1)

    # crossings: columns whose longest vertical ink run >= 0.85 S (minimal onset crop)
    xa = int(x0 + 0.02 * tspan)
    region = ink[y0:y1, xa:x1]
    cross = 0
    for col in region.T:
        best = run = gap = 0
        for v in col:
            if v:
                run += 1; gap = 0; best = max(best, run)
            else:
                gap += 1
                if gap > 1:
                    run = 0
        if best >= 0.85 * S:
            cross += 1
    feats["cross_cols"] = int(cross)

    # --- network-spread FPFA (2 vs 5): channels with sustained fast activity ---
    # Partition the vertical extent into N_CHANNELS bands (one per electrode).
    # Bands overlap slightly (stacked traces deflect into neighbours), so a raw
    # per-column ink count is noisy. The clean discriminator is per-channel
    # TEMPORAL VARIABILITY: a flat channel has near-constant column ink (low std),
    # a channel carrying fast activity swings (high std). Empirically the median
    # per-channel std separates spread-PFA seizure (~2.6) from sustained
    # spike-wave (~1.1) from flat baseline (~0.2).
    band_edges = np.linspace(y0, y1, N_CHANNELS + 1).astype(int)
    STD_ACTIVE = 1.5          # per-channel colcount std above this = fast-active channel
    DILATE_S   = 0.4          # temporal dilation so out-of-phase oscillation reads continuous
    dil = max(1, int(DILATE_S / WINDOW_S * (x1 - x0)))
    per_ch_colcount = np.zeros((N_CHANNELS, x1 - x0), dtype=float)
    ch_std = np.zeros(N_CHANNELS)
    for ch in range(N_CHANNELS):
        lo, hi = band_edges[ch], band_edges[ch + 1]
        cc = ink[lo:hi, x0:x1].sum(0).astype(float)
        per_ch_colcount[ch] = cc
        ch_std[ch] = cc.std()
    active_channels = ch_std > STD_ACTIVE
    feats["n_active_channels"] = int(active_channels.sum())
    # For the >4-channels-for->=2s test (2 vs 5): mark per-channel deflection
    # columns, then temporally DILATE (rolling OR over ~0.4 s) so an out-of-phase
    # rhythmic channel reads active across its whole oscillation period rather
    # than only at instantaneous peaks. Count simultaneous active channels per
    # column; report the longest contiguous span (s) with >4 active. Sustained
    # seizures span ~10 s; brief Type-4 GSW bursts stay well under 2 s.
    if active_channels.any():
        loc = np.zeros((N_CHANNELS, x1 - x0), dtype=bool)
        for ch in np.where(active_channels)[0]:
            cc = per_ch_colcount[ch]
            on = cc > (cc.mean() + 0.4 * cc.std())
            loc[ch] = np.convolve(on.astype(int), np.ones(dil), "same") > 0
        n_loc = loc[active_channels].sum(0)
        spread = n_loc > 4
        best = cur = 0
        for v in spread:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        feats["fast_activity_duration_s"] = round(best / (x1 - x0) * WINDOW_S, 2)

    # --- synchronous spike spatial extent (FIS vs FSW) ---
    # FIS vs FSW turns on how far a single DISCRETE spike spreads across the
    # electrode ring: a focal spike touches <=4 spatially-adjacent channels
    # near-simultaneously; a spread spike touches >4. Unlike n_active_channels
    # (which counts any channel active ANYWHERE in the 10s window, so temporally
    # separate spikes on different channels inflate it), this measures the max
    # number of CONTIGUOUS channels co-spiking within a +-40ms window.
    #   per (channel, column): a discrete deflection = colcount well above the
    #   channel's own baseline (mean + SYNC_K*std). Dilate each channel's spike
    #   train by +-CO_WIN_S so spikes within the co-window count as simultaneous.
    #   Then at every column, take the longest run of contiguous active channels;
    #   sync_spike_extent = max of that run over all columns. The electrodes lie
    #   on a TORUS/RING, so channel 0 and channel N-1 are physical neighbours:
    #   adjacency WRAPS across the seam (a spread straddling the last and first
    #   stacked traces is one contiguous discharge, not two).
    CO_WIN_S = 0.04           # +-40 ms simultaneity co-window
    SYNC_K   = 1.5            # deflection stands >=1.5 std above the channel baseline
    cw = max(1, int(CO_WIN_S / WINDOW_S * (x1 - x0)))
    spike_on = np.zeros((N_CHANNELS, x1 - x0), dtype=bool)
    for ch in range(N_CHANNELS):
        cc = per_ch_colcount[ch]
        if cc.std() < 1e-6 or cc.max() < 0.5 * S:
            continue          # flat / no real deflection in this band
        on = cc > (cc.mean() + SYNC_K * cc.std())
        spike_on[ch] = np.convolve(on.astype(int), np.ones(2 * cw + 1), "same") > 0
    max_extent = 0
    if spike_on.any():
        for t in range(spike_on.shape[1]):
            max_extent = max(max_extent, _max_circular_run(spike_on[:, t]))
    feats["sync_spike_extent"] = int(max_extent)
    return feats

# ============================================================================
# REVISED TAXONOMY (user definitions, 2026-08-11) — contiguity-aware features
# ============================================================================
# The revised definitions turn on quantities the original extractor does not
# measure. Three changes are load-bearing; each was validated against the
# 100-card adjudication set (borderline_adjudication_2026-08-11.csv):
#
# 1. FREQUENCY SELECTIVITY. "Paroxysmal FAST activity" is defined by being fast.
#    The original per-channel test (std of column ink count -> n_active_channels)
#    is frequency-BLIND: a channel making large SLOW excursions scores as high as
#    one carrying genuine fast oscillation. On the adjudicated set this is the
#    dominant error mode — epochs the user scores FPFA carry a few genuinely fast
#    channels plus many large-slow ones, and the variability test counts all of
#    them, reading 10-18 "active" channels where the definition caps FPFA at 4.
#    lfp_fast_features() instead counts direction reversals of each channel's
#    trace centreline per second and requires >= FAST_HZ.
#
# 2. CONTIGUITY, NOT TOTAL COUNT. User ruling 2026-08-11: "the distinction
#    between FPFA and seizure hinges on the number of CONTIGUOUS channels showing
#    FPFA, not the total number of channels." Contiguity is measured on the RING
#    (channel 0 adjoins channel N-1). The reported quantity is the joint
#    spatial-temporal one the definition names: fast_contig_dur_s[k-1] = longest
#    span (s) during which >= k contiguous channels were simultaneously fast.
#
# 3. GAP-TOLERANT HEATMAP BAND. A spike-wave burst occupies a vertical BAND of
#    unit indices but is SPECKLED — individual rows inside the band are dark, so
#    a strict contiguous run breaks at every dark row and drastically under-reads
#    the span (measured 16 indices on a burst spanning the whole network). The
#    definition says the burst "spans a contiguous range of unit indices" — a
#    band, not every index lit. burst_band_features() tolerates internal gaps up
#    to GAP_FRAC of the interior height.
#
# Legacy features (n_active_channels, fast_activity_duration_s,
# vertical_spread_fraction, ...) are still computed and returned UNCHANGED, so
# existing consumers and archived CSVs stay comparable. The revised classifier
# reads the new ones.

N_UNITS       = 784     # heatmap unit indices (network size)
FAST_HZ       = 6.0     # centreline reversal rate at/above which a channel is "fast"
FAST_WIN_S    = 0.5     # sliding window for the rate estimate
FAST_DILATE_S = 0.3     # temporal dilation so a briefly-interrupted oscillation reads continuous
GAP_FRAC      = 0.02    # heatmap band gap tolerance, as a fraction of interior height


def _longest_true_span(mask):
    """Longest run of consecutive True in a 1-D boolean array (NOT circular)."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return int(best)


def _longest_band(mask, gap_tol):
    """Longest run of True allowing internal gaps of up to `gap_tol` False.

    Implements "spans a contiguous range" for a SPECKLED burst: the burst occupies
    a band of unit indices, but not every index inside the band is lit.
    """
    n = len(mask)
    best = 0
    i = 0
    while i < n:
        if mask[i]:
            last = i
            gap = 0
            j = i
            while j < n:
                if mask[j]:
                    last = j
                    gap = 0
                else:
                    gap += 1
                    if gap > gap_tol:
                        break
                j += 1
            best = max(best, last - i + 1)
            i = last + 1
        else:
            i += 1
    return int(best)


def _segments(mask):
    """List of (start, end) index pairs for each maximal True run."""
    segs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            segs.append((i, j)); i = j
        else:
            i += 1
    return segs


def _longest_band_span(mask, gap_tol):
    """As _longest_band, but also returns the winning band's (start, end) indices
    so its internal density can be measured."""
    n = len(mask)
    best = (0, 0, -1)
    i = 0
    while i < n:
        if mask[i]:
            last = i
            gap = 0
            j = i
            while j < n:
                if mask[j]:
                    last = j
                    gap = 0
                else:
                    gap += 1
                    if gap > gap_tol:
                        break
                j += 1
            if last - i + 1 > best[0]:
                best = (last - i + 1, i, last)
            i = last + 1
        else:
            i += 1
    return best


def _lfp_geometry(path):
    """Shared LFP figure geometry: ink mask cropped to the plot interior, span,
    and the FIXED inter-channel spacing S (extent / N_CHANNELS — never estimated
    per-run, per spec). Returns (sub, T, S, px_per_s); sub is None if blank.
    """
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    mx = a.max(2); mn = a.min(2); sat = mx - mn
    white = (mx > 235) & (sat < 18)
    ink = (sat >= 35) & (mx > 60) & (~white)
    colsum = ink.sum(0); rowsum = ink.sum(1)
    if colsum.max() == 0:
        return None, 0, 0.0, 0.0
    xs = np.where(colsum > colsum.max() * 0.05)[0]
    ys = np.where(rowsum > rowsum.max() * 0.05)[0]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    T = int(x1 - x0)
    S = (y1 - y0) / N_CHANNELS
    return ink[y0:y1, x0:x1], T, S, (T / WINDOW_S if T else 0.0)


def _centrelines(sub, T, S, n_ch=N_CHANNELS):
    """Per-channel trace centreline and deflection amplitude by OWNERSHIP.

    Geometry (measured on the 1500x750 canvas): S ~19 px; a quiescent trace is a
    2-3 px ink run on its lane centre; a deflection is a longer run whose vertical
    extent is the transient's amplitude. A large deflection MERGES with the
    neighbouring traces into one run — which is why a naive per-lane ink count
    lights up neighbours that are actually flat, and why involvement must be
    attributed rather than counted.

    Rule: in each column every maximal ink run is assigned to the lane containing
    its CENTROID — the channel that owns the transient. A lane merely crossed by
    someone else's excursion is not credited.

    Returns (cline, amp): centreline row per (channel, column), NaN where the
    channel owns no run; and peak |deviation from baseline| in units of S, so
    amp >= 1.0 means the transient reached the baseline of the adjacent channel.
    """
    H = sub.shape[0]
    rel_centres = (np.arange(n_ch) + 0.5) * S
    cline = np.full((n_ch, T), np.nan)
    amp = np.zeros((n_ch, T))
    for t in range(T):
        col = sub[:, t]
        if not col.any():
            continue
        i = 0
        while i < H:
            if col[i]:
                j = i
                while j < H and col[j]:
                    j += 1
                top, bot = i, j - 1
                cen = 0.5 * (top + bot)
                ch = int(np.argmin(np.abs(rel_centres - cen)))
                if np.isnan(cline[ch, t]) or abs(cen - rel_centres[ch]) < abs(cline[ch, t] - rel_centres[ch]):
                    cline[ch, t] = cen
                dev = max(abs(top - rel_centres[ch]), abs(bot - rel_centres[ch])) / S
                if dev > amp[ch, t]:
                    amp[ch, t] = dev
                i = j
            else:
                i += 1
    return cline, amp


def lfp_fast_features(path, n_ch=N_CHANNELS):
    """Frequency-selective fast activity, ring contiguity, burst structure, amplitude.

    Returns
      n_fast_channels    channels carrying fast activity anywhere in the window
      fast_max_contig    max contiguous fast channels at any instant (ring-wrapped)
      fast_dur_s         longest span (s) with >= 1 fast channel
      fast_contig_dur_s  list of n_ch floats; entry k-1 = longest span (s) with
                         >= k CONTIGUOUS channels simultaneously fast. This is the
                         joint criterion the revised definitions name; the
                         FPFA-vs-seizure test reads entry SEIZ_CONTIG_CHANS-1.
      fast_ibi_ratio     median inter-burst interval / median burst duration on the
                         channel that is fast longest. Type 2 requires "inter-burst
                         intervals of the same duration or shorter than the burst"
                         -> ratio <= 1. 0.0 if a single uninterrupted run; None if
                         no fast activity.
      n_fpfa_sequences   number of channels each carrying a qualifying FPFA
                         sequence in its own right (>=2 s fast, intervals no
                         longer than its bursts). USER: "Multiple independent
                         FPFA sequences can coexist in the same epoch."
      fpfa_best_dur_s    total fast duration on the best such channel
      fpfa_best_ratio    that channel's interval/burst ratio
      spike_amp_max      largest deflection amplitude in units of S. The user's
                         "standing out from the background" criterion is that the
                         transient crosses the baseline of the channel above,
                         i.e. amplitude >= 1.0.
    """
    out = dict(n_fast_channels=0, fast_max_contig=0, fast_dur_s=0.0,
               fast_contig_dur_s=[0.0] * n_ch, fast_ibi_ratio=None,
               spike_amp_max=0.0, n_fpfa_sequences=0, fpfa_best_dur_s=0.0,
               fpfa_best_ratio=None)
    sub, T, S, px_per_s = _lfp_geometry(path)
    if sub is None or T < 8:
        return out
    cline, amp = _centrelines(sub, T, S, n_ch)
    out["spike_amp_max"] = round(float(amp.max()), 2)

    # --- oscillation RATE per channel: centreline direction reversals per second ---
    w = max(3, int(FAST_WIN_S * px_per_s))
    rate = np.zeros((n_ch, T))
    for ch in range(n_ch):
        y = cline[ch]
        good = ~np.isnan(y)
        if good.sum() < w:
            continue
        idx = np.arange(T)
        y = uniform_filter1d(np.interp(idx, idx[good], y[good]), 3)
        d = np.diff(y)
        sgn = np.sign(np.where(np.abs(d) < 0.35, 0.0, d))   # ignore flat pixel noise
        rev = np.zeros(T)
        last = 0.0
        for t in range(len(sgn)):
            if sgn[t] != 0:
                if last != 0.0 and sgn[t] != last:
                    rev[t] = 1.0
                last = sgn[t]
        # two reversals per oscillation cycle
        rate[ch] = np.convolve(rev, np.ones(w), "same") / (2.0 * w / px_per_s)

    fast = rate >= FAST_HZ
    dil = max(1, int(FAST_DILATE_S * px_per_s))
    for ch in range(n_ch):
        fast[ch] = np.convolve(fast[ch].astype(int), np.ones(dil), "same") > 0
    out["n_fast_channels"] = int(fast.any(1).sum())

    runs = np.array([_max_circular_run(fast[:, t]) for t in range(T)])
    out["fast_max_contig"] = int(runs.max())
    out["fast_dur_s"] = round(_longest_true_span(runs >= 1) / px_per_s, 2)
    out["fast_contig_dur_s"] = [round(_longest_true_span(runs >= k) / px_per_s, 2)
                                for k in range(1, n_ch + 1)]

    # --- PER-CHANNEL qualifying FPFA sequences ---
    # USER: "Multiple independent FPFA sequences can coexist in the same epoch."
    # The Type 2 test must therefore ask whether ANY SINGLE channel carries a
    # qualifying sequence (>= FPFA_MIN_DUR_S of fast activity whose inter-burst
    # intervals are no longer than its bursts), not whether some epoch-wide
    # aggregate does. Measuring the interval ratio on one arbitrarily-chosen
    # channel mixes independent sequences together and reads their interleaving
    # as long intervals: on one user-scored FPFA epoch that produced a ratio of
    # 1.28 (failing the rule) where the actual carrier channel runs 5.26 s at a
    # ratio of 0.61 (passing comfortably).
    n_qual = 0
    best_dur = 0.0
    best_ratio = None
    for ch in range(n_ch):
        segs = _segments(fast[ch])
        if not segs:
            continue
        dur = sum(b - a for a, b in segs) / px_per_s
        if len(segs) == 1:
            ratio = 0.0
        else:
            bd = [(b - a) / px_per_s for a, b in segs]
            ibi = [(segs[k][0] - segs[k - 1][1]) / px_per_s for k in range(1, len(segs))]
            mb = float(np.median(bd))
            ratio = (float(np.median(ibi)) / mb) if mb > 0 else None
        qual = dur >= 2.0 and ratio is not None and ratio <= 1.0
        n_qual += int(qual)
        if (qual, dur) > (best_dur > 0 and best_ratio is not None and best_ratio <= 1.0, best_dur):
            best_dur, best_ratio = dur, ratio
    out["n_fpfa_sequences"] = n_qual
    out["fpfa_best_dur_s"] = round(best_dur, 2)
    out["fpfa_best_ratio"] = None if best_ratio is None else round(best_ratio, 2)

    # --- legacy epoch-level burst structure (retained for comparability) ---
    if fast.any():
        ch = int(np.argmax(fast.sum(1)))
        on = fast[ch]
        segs, i = [], 0
        while i < T:
            if on[i]:
                j = i
                while j < T and on[j]:
                    j += 1
                segs.append((i, j)); i = j
            else:
                i += 1
        if len(segs) == 1:
            out["fast_ibi_ratio"] = 0.0                     # continuous
        elif len(segs) >= 2:
            bd  = [(b - a_) / px_per_s for a_, b in segs]
            ibi = [(segs[k][0] - segs[k - 1][1]) / px_per_s for k in range(1, len(segs))]
            mb = float(np.median(bd))
            out["fast_ibi_ratio"] = round(float(np.median(ibi)) / mb, 2) if mb > 0 else None
    return out


def burst_band_features(path, n_units=N_UNITS):
    """Vertical extent of the strongest heatmap burst as a gap-tolerant BAND.

    Serves the Type 3 (>=25% and <85% of the network) vs Type 4 (>=85%) gate.
      burst_band_frac  band span as a fraction of the network
      burst_band_idx   the same in unit indices (frac * N_UNITS)
      burst_any_frac   fraction of ALL rows active in the burst (strict, no band)
      burst_band_dens  fraction of rows INSIDE the band that are lit (coherence)
    """
    _, _bright, interior, _box, _H, _W = _heatmap_arrays(path)
    out = dict(burst_band_frac=0.0, burst_band_idx=0, burst_any_frac=0.0)
    colprof = interior.sum(0).astype(float)
    if colprof.max() == 0:
        return out
    sm = uniform_filter1d(colprof, 7)
    peak = int(np.argmax(sm))
    c0, c1 = max(0, peak - 3), min(interior.shape[1], peak + 4)
    rows = interior[:, c0:c1].any(1)
    nrows = len(rows)
    band, b0, b1 = _longest_band_span(rows, max(1, int(GAP_FRAC * nrows)))
    out["burst_any_frac"]  = round(float(rows.mean()), 3)
    out["burst_band_frac"] = round(band / nrows, 3)
    out["burst_band_idx"]  = int(round(band / nrows * n_units))
    # DENSITY of the winning band: what fraction of the rows INSIDE it are lit.
    # Load-bearing guard. Gap tolerance is needed because a real burst is speckled,
    # but applied alone it will also stitch together sparse HORIZONTAL streaks that
    # merely coincide in time, reporting a 254-index "burst" on an epoch whose
    # heatmap has no coherent vertical column at all (observed on two epochs the
    # user scored FPFA). A genuine "transient vertical burst" fills the range it
    # spans; the density floor in classifier.TH separates the two.
    out["burst_band_dens"] = round(float(rows[b0:b1 + 1].mean()), 3) if band else 0.0
    return out


DISCHARGE_SLEW = 12.0   # S/s; rate of rise separating a spike/sharp wave from a
                        # slow swing of the same amplitude
DISCHARGE_AMP  = 1.0    # in units of S: the user's "crosses the baseline of the
                        # channel above"


def lfp_involvement_features(path, n_ch=N_CHANNELS):
    """MORPHOLOGY-AGNOSTIC epileptiform involvement, for the generalized Type 5.

    User definition (2026-08-11 rev 2): "A seizure is any form of sustained
    epileptic activity that has spread over much of the network." Involvement can
    therefore no longer mean "fast" only — a spike-wave or polyspike discharge
    train is epileptiform without being fast, and restricting Type 5 to the fast
    arm would make the seizure label insensitive to exactly that.

    A channel counts as involved at time t when EITHER
      (a) it is fast there (>= FAST_HZ centreline reversals/s), OR
      (b) a SHARP DISCHARGE is present: excursion amplitude >= DISCHARGE_AMP
          (run-edge deviation from baseline, in units of S) AND rate of rise
          >= DISCHARGE_SLEW.

    The slew requirement in (b) is load-bearing and is the guard against the
    failure mode that produced the previous classifier's 3:1 seizure over-call.
    Amplitude alone does NOT make activity epileptiform: these networks produce
    large SLOW swings that reach the same amplitude as a spike. Slew is what
    distinguishes them. Note also that amplitude is measured from the ink run's
    EDGE, not its centreline — a tall spike's centroid moves only about half the
    run's extent, so a centreline-based amplitude test silently never fires.

    Deliberately NOT done: bridging across inter-discharge gaps. It was prototyped
    and rejected — joining discharges separated by clear intervals inflated a
    user-scored FPFA epoch from 1.39 s to 4.29 s of 7-contiguous involvement, i.e.
    straight over the seizure bar. A clear interval breaks sustainedness.

    Returns
      involve_contig_dur_s  list of n_ch floats; entry k-1 = longest span (s) with
                            >= k CONTIGUOUS channels simultaneously involved. The
                            Type 5 test reads entry SEIZ_CONTIG_CHANS-1.
      n_involved_channels   channels involved anywhere in the window
      involve_max_contig    max contiguous involved channels at any instant (ring)
      n_discharge_channels  channels involved via the discharge arm only (no fast)
    """
    out = dict(involve_contig_dur_s=[0.0] * n_ch, n_involved_channels=0,
               involve_max_contig=0, n_discharge_channels=0)
    sub, T, S, pps = _lfp_geometry(path)
    if sub is None or T < 8:
        return out
    cline, amp = _centrelines(sub, T, S, n_ch)
    idx = np.arange(T)
    fast = np.zeros((n_ch, T), bool)
    sharp = np.zeros((n_ch, T), bool)
    w = max(3, int(FAST_WIN_S * pps))
    for ch in range(n_ch):
        y = cline[ch]
        good = ~np.isnan(y)
        if good.sum() < w:
            continue
        yy = uniform_filter1d(np.interp(idx, idx[good], y[good]), 3)
        d = np.diff(yy)
        sgn = np.sign(np.where(np.abs(d) < 0.35, 0.0, d))
        rev = np.zeros(T)
        last = 0.0
        for t in range(len(sgn)):
            if sgn[t] != 0:
                if last != 0.0 and sgn[t] != last:
                    rev[t] = 1.0
                last = sgn[t]
        fast[ch] = (np.convolve(rev, np.ones(w), "same") / (2.0 * w / pps)) >= FAST_HZ
        slew = np.abs(np.gradient(yy / S)) * pps
        sharp[ch] = (amp[ch] >= DISCHARGE_AMP) & (slew >= DISCHARGE_SLEW)
    dil = max(1, int(FAST_DILATE_S * pps))
    for ch in range(n_ch):
        fast[ch] = np.convolve(fast[ch].astype(int), np.ones(dil), "same") > 0
        sharp[ch] = np.convolve(sharp[ch].astype(int), np.ones(dil), "same") > 0
    inv = fast | sharp
    out["n_involved_channels"] = int(inv.any(1).sum())
    out["n_discharge_channels"] = int((sharp & ~fast).any(1).sum())
    runs = np.array([_max_circular_run(inv[:, t]) for t in range(T)])
    out["involve_max_contig"] = int(runs.max())
    out["involve_contig_dur_s"] = [round(_longest_true_span(runs >= k) / pps, 2)
                                   for k in range(1, n_ch + 1)]
    return out


# --------------------------------------------------------------- combine ----
def extract_features(heatmap_path, lfp_path, mean_E=None, val_acc=None):
    f = {}
    f.update(heatmap_features(heatmap_path))
    f.update(lfp_features(lfp_path))
    # revised-taxonomy features (frequency-selective, contiguity-aware)
    f.update(burst_band_features(heatmap_path))
    f.update(lfp_fast_features(lfp_path))
    f.update(lfp_involvement_features(lfp_path))
    if mean_E is not None:  f["mean_E"]  = round(float(mean_E), 4)
    if val_acc is not None: f["val_acc"] = round(float(val_acc), 4)
    return f
