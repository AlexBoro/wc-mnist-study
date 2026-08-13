"""Prototype: morphology-agnostic epileptiform involvement.

Type 5 (user, 2026-08-11 rev 2) is now ANY form of sustained epileptiform activity
spread over >=7 contiguous channels. So involvement can no longer mean "fast" only
-- a spike-wave discharge train is epileptiform without being fast.

The trap: the previous classifier's 3:1 seizure over-call came from treating ANY
large deflection as activity. A slow swing is not epileptiform. So involvement =
fast oscillation OR a SHARP discharge (supra-baseline AND fast-rising), never
amplitude alone.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from labeling_toolkit import features as F

def involvement(path, n_ch=F.N_CHANNELS, slew_min=6.0, amp_min=1.0, bridge_frac=1.0):
    """Returns (involved mask [n_ch x T], px_per_s, diagnostics).

    A channel is involved at time t if it is fast there, or if a sharp discharge
    is present. Discharge = centreline excursion reaching amp_min (in units of the
    inter-channel spacing S -- the user's "crosses the baseline of the channel
    above") with slew >= slew_min S/s, which is what separates a spike from a slow
    swing of the same amplitude.

    Bridging uses the user's OWN interval rule rather than a fitted constant:
    consecutive discharges are joined only when the gap is no longer than the
    events it separates ("inter-burst intervals of the same duration or shorter
    than the burst"). A clear interval therefore breaks sustainedness.
    """
    sub, T, S, pps = F._lfp_geometry(path)
    if sub is None or T < 8:
        return np.zeros((n_ch, 1), bool), 1.0, {}
    cline, amp = F._centrelines(sub, T, S, n_ch)

    # --- fast arm: reuse the installed detector's logic ---
    w = max(3, int(F.FAST_WIN_S * pps))
    fast = np.zeros((n_ch, T), bool)
    sharp = np.zeros((n_ch, T), bool)
    idx = np.arange(T)
    for ch in range(n_ch):
        y = cline[ch]; good = ~np.isnan(y)
        if good.sum() < w:
            continue
        yy = uniform_filter1d(np.interp(idx, idx[good], y[good]), 3)
        d = np.diff(yy)
        sgn = np.sign(np.where(np.abs(d) < 0.35, 0.0, d))
        rev = np.zeros(T); last = 0.0
        for t in range(len(sgn)):
            if sgn[t] != 0:
                if last != 0.0 and sgn[t] != last:
                    rev[t] = 1.0
                last = sgn[t]
        rate = np.convolve(rev, np.ones(w), "same") / (2.0 * w / pps)
        fast[ch] = rate >= F.FAST_HZ
        # --- sharp-discharge arm: slew in units of S per second ---
        slew = np.abs(np.gradient(yy / S)) * pps
        dev = np.abs(yy - np.nanmedian(yy)) / S
        sharp[ch] = (slew >= slew_min) & (dev >= amp_min * 0.5)
        # require the excursion to actually reach amp_min somewhere in each event
        segs = _segments(sharp[ch])
        for a, b in segs:
            if dev[a:b].max() < amp_min:
                sharp[ch, a:b] = False

    dil = max(1, int(F.FAST_DILATE_S * pps))
    for ch in range(n_ch):
        fast[ch] = np.convolve(fast[ch].astype(int), np.ones(dil), "same") > 0

    inv = fast | sharp
    # bridge gaps no longer than the flanking events (user's interval rule)
    for ch in range(n_ch):
        inv[ch] = _bridge(inv[ch], bridge_frac)
    return inv, pps, dict(fast=fast, sharp=sharp)


def _segments(mask):
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


def _bridge(mask, frac):
    """Join consecutive True runs when the gap between them is <= frac * the
    shorter flanking run. Implements "intervals of the same duration or shorter
    than the burst"; a clear (longer) interval is left as a break."""
    segs = _segments(mask)
    if len(segs) < 2:
        return mask
    out = mask.copy()
    for k in range(1, len(segs)):
        a0, a1 = segs[k - 1]; b0, b1 = segs[k]
        gap = b0 - a1
        if gap <= frac * min(a1 - a0, b1 - b0):
            out[a1:b0] = True
    return out


def contig_dur(inv, pps, k):
    """Longest span (s) with >= k contiguous involved channels simultaneously."""
    T = inv.shape[1]
    runs = np.array([F._max_circular_run(inv[:, t]) for t in range(T)])
    best = cur = 0
    for v in (runs >= k):
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best / pps
