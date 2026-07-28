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

# --------------------------------------------------------------- combine ----
def extract_features(heatmap_path, lfp_path, mean_E=None, val_acc=None):
    f = {}
    f.update(heatmap_features(heatmap_path))
    f.update(lfp_features(lfp_path))
    if mean_E is not None:  f["mean_E"]  = round(float(mean_E), 4)
    if val_acc is not None: f["val_acc"] = round(float(val_acc), 4)
    return f
