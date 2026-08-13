"""classifier.py — declarative decision table for the WC-MNIST taxonomy.

The ENTIRE classification logic is the ordered RULES list below. classify(feats)
walks it top-down and returns the FIRST matching rule's verdict, with provenance
(which rule fired + the taxonomy clause it implements). Adding or changing a type
= editing one row here; nothing is nested or hidden. Features come from
features.py::extract_features.

REVISED 2026-08-11 to the user's updated definitions. What changed and why:

  * CONTIGUITY replaces total channel count. User ruling: "the distinction
    between FPFA and seizure hinges on the number of CONTIGUOUS channels showing
    FPFA, not the total number of channels." The old ictal_spread_pfa rule tested
    n_active_channels > 4 (a TOTAL count, frequency-blind) and produced the
    over-calling documented in CLASSIFIER_AUDIT_2026-08-11.md. It is replaced by
    a joint spatial-temporal test on fast_contig_dur_s.

  * The "outlasts the stimulus" criterion is REMOVED entirely. User ruling:
    "There should be no mention of whether a discharge outlasting the stimulus.
    This is not relevant in the current classification scheme."

  * HIGHEST TYPE WINS. User ruling: "Some epochs will show multiple types of
    epileptiform activity. In this case, the higher score should be used." Rule
    order therefore descends 5 -> 4 -> 3 -> 2 -> 1 -> 0 strictly, and the first
    match is the highest applicable type. (This is a real change: the old table
    interleaved arms and could return a lower type than one a later arm matched.)

  * Type 3/4 read the gap-tolerant heatmap BAND (burst_band_frac), not the strict
    contiguous run, and are bounded by the user's explicit network fractions.

Validation: reproduces all 100 user calls in borderline_adjudication_2026-08-11.csv
(see validate_revised_taxonomy.py). Thresholds are the user's stated numbers; none
were fitted.

Label codes: 0 none, 1 FIS, 2 FPFA, 3 FSW, 4 GSW, 5 seizure(ictal), -1 review.
"""

# ---- named thresholds (ALL tunables live here, each with its provenance) ----
# "USER" = a number the user stated in the 2026-08-11 definitions, not a fit.
# "MEAS" = a measurement convention chosen to implement a user criterion.
# Which measured feature supplies the Type 1 amplitude. "spike_amp_max" is the
# legacy pixel measure read off the rendered lfp.png; "amp_prominence" and
# "amp_excursion" come from analytic_features.py, measured from arrays.npz.
# USER 2026-08-12: switch to the arrays. Prominence is the default because it
# measures a transient against its own local baseline, so slow drift cannot
# accumulate into an apparent spike (excursion can).
SPIKE_AMP_FEATURE = "amp_prominence"

TH = dict(
    # -- Type 2 FPFA (focal paroxysmal fast activity) --
    FPFA_MIN_DUR_S   = 2.0,   # USER: "Minimum duration: 2 consecutive seconds";
                              #       shorter is scored Type 1 (FIS)
    FPFA_MAX_CHANS   = 6,     # USER (corrected 2026-08-11): "Involves up to 6
                              #       adjacent channels in the LFP plot"; ">=7
                              #       contiguous -> seizure". With 6 the cap and
                              #       the promotion bar are now ADJACENT: no
                              #       undefined band remains (the earlier value 4
                              #       left 5-6 unspecified).
    FPFA_IBI_RATIO   = 1.0,   # USER: "inter-burst intervals of the same duration
                              #       or shorter than the burst" -> IBI/burst <= 1

    # -- Type 5 seizure --
    SEIZ_CONTIG_CHANS = 7,    # USER: ">=7 contiguous channels involved" both as the
                              #       FPFA->seizure promotion and as "spread over
                              #       much of the network"
    SEIZ_SUSTAIN_S    = 2.0,  # USER: "'Sustained' means at least 2 seconds"

    # -- Type 1 FIS / spike-wave promotion --
    SPIKE_AMP_MIN    = 1.0,   # USER: "amplitude ... sufficient to cross the
                              #       baseline of the channel above" -> >= 1.0 S
    SW_SPREAD_CHANS  = 7,     # USER: "time-locked spread to 7 [or] more channels"
                              #       promotes a spike to a spike-wave form

    # -- Type 3 FSW / Type 4 GSW extent (user's explicit network fractions) --
    FSW_BAND_LO      = 0.25,  # USER: ">=25% ... of the network" (196 indices)
    GSW_BAND_LO      = 0.85,  # USER: ">85% of the network" (>=672 indices);
                              #       FSW upper bound is this value
    FSW_CHANS_LO     = 7,     # USER: "involves 7 to 23 channels on the LFP plot"
    GSW_CHANS_LO     = 24,    # USER: ">=24 channels on the LFP plot"
    BAND_DENS_MIN    = 0.30,  # MEAS: a "transient vertical burst" must FILL the range
                              #       it spans. Gap tolerance is needed (real bursts
                              #       are speckled) but alone it stitches sparse
                              #       horizontal streaks into a phantom 254-index
                              #       burst on epochs with no vertical column at all.
                              #       Adjudicated set: user spike-wave/seizure bursts
                              #       are >=0.34 dense, the two phantom bursts 0.21.

    # -- Type 0 silence floor --
    INK_SILENT       = 0.02,  # heatmap essentially blank
    MCF_QUIET        = 0.15,  # no vertical recruitment
    CROSS_QUIET      = 20,    # few/no LFP crossings
)

N_UNITS = 784   # network size; the index bounds above are fractions of this


# --------------------------------------------------------------- helpers ----
def _involve_dur(f, k):
    """Longest span (s) with >= k CONTIGUOUS channels simultaneously INVOLVED
    (fast activity or sharp discharge — any epileptiform morphology).

    Reads involve_contig_dur_s. Falls back to the fast-only measure for feature
    dicts produced before the generalized involvement detector existed, so older
    tables still classify (slightly less sensitively) rather than raising.
    """
    v = f.get("involve_contig_dur_s")
    if not v:
        return _contig_dur(f, k)
    return 0.0 if (k < 1 or k > len(v)) else float(v[k - 1])


def _contig_dur(f, k):
    """Longest span (s) with >= k CONTIGUOUS channels simultaneously fast.

    Reads fast_contig_dur_s (features.lfp_fast_features), a list whose entry k-1
    is that span. Returns 0.0 when the feature is absent, so a feature dict from
    the pre-revision extractor degrades to 'no contiguous fast activity' rather
    than raising.
    """
    v = f.get("fast_contig_dur_s")
    if not v or k < 1 or k > len(v):
        return 0.0
    return float(v[k - 1])


def _has_epileptiform_correlate(f):
    """Labels 1-5 require a morphological correlate on the LFP: either a discrete
    transient large enough to stand out from the background (the user's amplitude
    criterion: spike_amp_max >= 1.0 S, i.e. it crosses the baseline of the channel
    above) or genuine fast activity somewhere on the ring.

    NOTE the deliberate omission: heatmap activity alone — however hot, however
    banded — is NOT epileptiform without an LFP correlate. Retained from the
    2026-07-25 user ruling. The "outlasts the stimulus" clause that used to sit
    here is REMOVED per the 2026-08-11 ruling.

    CHANGED 2026-08-12 (USER): the second arm was ``n_fast_channels > 0`` — ANY
    fast activity, of any duration, promoted a run out of Type 0. The user's Type 2
    definition makes sub-2 s fast activity Type 1 only if the amplitude criterion
    is met, so bare fast activity is no longer a ticket out of Type 0; the arm now
    requires a QUALIFYING FPFA sequence (>=2 s, intervals no longer than bursts),
    which is the only thing fast activity alone can legitimately promote to.
    """
    return (_spike_amp(f) >= TH["SPIKE_AMP_MIN"]
            or f.get("n_fpfa_sequences", 0) >= 1)


def _spike_amp(f):
    """The Type 1 amplitude, in units of inter-channel spacing.

    SPIKE_AMP_FEATURE selects the source.  "spike_amp_max" is the pixel measure
    from features.py; the analytic_features keys are measured from arrays.npz and
    are exact (see analytic_features module docstring).  Falls back to the pixel
    measure when the analytic feature is absent, so runs that predate the
    analytic extractor still classify.
    """
    v = f.get(SPIKE_AMP_FEATURE)
    if v is None:
        v = f.get("spike_amp_max", 0.0)
    return float(v or 0.0)


def _fis_present(f):
    """Type 1 (USER, 2026-08-12): a discrete transient whose amplitude crosses the
    baseline of the channel above, with time-locked spread below the spike-wave
    promotion bar.

    The second arm implements the user's Type 2 clause verbatim: "Shorter
    duration sequences scored as type 1 (FIS) IF THE AMPLITUDE CRITERION FOR TYPE 1
    TRANSIENTS IS MET".  Sub-2 s fast activity is therefore NOT self-promoting --
    it must independently clear the amplitude bar.  Before 2026-08-12 this arm had
    no amplitude test, which admitted 9 of 15 FIS calls on the seizure-side sweep
    on grounds unrelated to the Type 1 definition.
    """
    amp_ok = _spike_amp(f) >= TH["SPIKE_AMP_MIN"]
    if not amp_ok:
        return False
    if _sw_chans(f) < TH["SW_SPREAD_CHANS"]:
        return True
    # Spread reaches the spike-wave bar but the spike-wave arms above declined it
    # (band extent/density failed): still a Type 1 transient, not a Type 0.
    return 0.0 < f.get("fast_dur_s", 0.0) < TH["FPFA_MIN_DUR_S"]


def _seizure(f):
    """Type 5, generalized (USER, 2026-08-11 rev 2): "A seizure is any form of
    sustained epileptic activity that has spread over much of the network",
    with "sustained" = at least 2 s and "spread over much of the network" =
    >= 7 contiguous channels involved.

    MORPHOLOGY-BLIND BY DESIGN. This single criterion replaces the previous pair
    of arms (paroxysmal-fast + spike-wave complex count). The user's intent is
    that the seizure label be SENSITIVE to any sufficiently widespread and
    sustained epileptiform activity, so what the activity looks like is not
    tested: involvement counts fast oscillation and sharp discharges alike (see
    features.lfp_involvement_features). This also retires the unvalidated
    complex-count arm, which relied on heatmap-derived n_complexes/max_IDI rather
    than trace-side complexes and mis-promoted the one card it ever fired on.

    Duration and spatial extent are tested JOINTLY: the quantity is the span for
    which >= 7 contiguous channels were involved AT THE SAME TIME. Testing them
    separately is the error behind the earlier over-calling — many FPFA epochs
    have activity throughout the window and >= 7 channels involved at some point,
    without ever having 7 contiguous involved together for 2 s.

    Validated on the 100-card adjudication set: all 66 user-scored FPFA epochs
    reach at most 1.76 s, all 28 user-scored seizures at least 3.53 s. The 2 s
    bar sits in a gap with ~1.8 s of margin on either side.
    """
    return _involve_dur(f, TH["SEIZ_CONTIG_CHANS"]) >= TH["SEIZ_SUSTAIN_S"]


def _band_ok(f):
    """The heatmap burst is a COHERENT vertical burst: it spans enough of the
    network AND actually fills the range it spans (see TH['BAND_DENS_MIN']).
    Feature dicts predating burst_band_dens pass the density test by default.
    """
    d = f.get("burst_band_dens")
    return d is None or d >= TH["BAND_DENS_MIN"]


def _band_frac(f):
    """Vertical extent of the strongest heatmap burst as a fraction of the network,
    gap-tolerant (features.burst_band_features). Falls back to the legacy strict
    measure if absent.
    """
    v = f.get("burst_band_frac")
    if v is None:
        v = f.get("vertical_spread_fraction", 0.0)
    return float(v or 0.0)


def _sw_chans(f):
    """Channels involved in a time-locked spike-wave discharge (LFP side).
    sync_spike_extent measures the max CONTIGUOUS channels co-spiking within the
    simultaneity co-window, which is the "time-locked spread ... to adjacent
    channels" the definitions describe (synchronously or with phase delay).
    """
    return f.get("sync_spike_extent", 0)


def _fpfa_present(f):
    """Type 2 core: fast activity of at least the minimum duration, in bursting
    sequences whose inter-burst intervals are no longer than the bursts (or
    continuous).

    Channel extent is not re-tested here: the ceiling ("up to 6 adjacent
    channels") and the promotion bar (">=7 contiguous -> seizure") are adjacent,
    so anything exceeding the ceiling has already been caught by the seizure rule
    above under first-match-wins. Multiple independent FPFA sequences may coexist
    in one epoch (USER), which is why the test asks whether a qualifying sequence
    is present rather than imposing a whole-epoch channel budget.
    """
    n = f.get("n_fpfa_sequences")
    if n is not None:
        return n >= 1
    # fallback for feature dicts predating the per-channel measure
    if f.get("fast_dur_s", 0.0) < TH["FPFA_MIN_DUR_S"]:
        return False
    r = f.get("fast_ibi_ratio")
    return r is None or r <= TH["FPFA_IBI_RATIO"]


# The earlier undefined 5-6 contiguous channel band is CLOSED: the user corrected
# the FPFA ceiling from 4 to 6 on 2026-08-11, making it adjacent to the 7-contiguous
# seizure bar. No epoch can now fall between the two definitions.


# ------------------------------------------------------------------- RULES ----
# Each rule: (name, predicate(f)->bool, label, confidence, clause).
# Order is STRICTLY DESCENDING BY TYPE so that "the higher score should be used"
# falls out of first-match-wins. Within a type, arms are alternatives (OR).
RULES = [
    # --- gate: labels 1-5 require an LFP correlate -> otherwise Type 0 ---
    ("no_lfp_correlate", lambda f: not _has_epileptiform_correlate(f), 0, "high",
     "Type0: no LFP correlate (no supra-baseline transient, no fast activity)"),

    # --- Type 5: seizure (two alternative criteria; either suffices) ---
    ("seizure", _seizure, 5, "high",
     "Type5: sustained (>=2 s) epileptiform activity, any morphology, over >=7 CONTIGUOUS channels"),

    # --- Type 4: generalized spike-wave (essentially the full vertical range) ---
    ("gsw_generalized",
     lambda f: ((_band_frac(f) >= TH["GSW_BAND_LO"] and _band_ok(f))
                or _sw_chans(f) >= TH["GSW_CHANS_LO"]),
     4, "med",
     "Type4: burst involves >85% of the network (>=672 indices / >=24 channels)"),

    # --- Type 3: focal spike-wave (partial contiguous range, spread on the LFP) ---
    ("fsw_focal",
     lambda f: (_band_frac(f) >= TH["FSW_BAND_LO"] and _band_frac(f) < TH["GSW_BAND_LO"]
                and _band_ok(f)
                and _sw_chans(f) >= TH["FSW_CHANS_LO"]),
     3, "med",
     "Type3: burst spans 25-85% of the network with spike-wave spread to >=7 channels"),

    # --- Type 2: focal paroxysmal fast activity ---
    ("fpfa_focal", _fpfa_present, 2, "med",
     "Type2: fast activity >=2 s, inter-burst intervals no longer than the bursts"),

    # --- Type 1: focal interictal spikes (incl. focal polyspikes) ---
    # Reaches here having failed the seizure/spike-wave/FPFA arms: a discrete
    # transient standing out from the background (>=1.0 S) whose time-locked
    # spread stays below the spike-wave promotion bar, OR fast activity too brief
    # to be FPFA ("Shorter duration scored as type 1").
    ("fis_focal", _fis_present, 1, "med",
     "Type1: focal interictal spike/polyspike (supra-baseline, spread <7 channels)"),

    # --- Type 0: none (guarded silence) ---
    ("none_silent",
     lambda f: (f.get("ink_frac", 1) < TH["INK_SILENT"]
                and f.get("cross_cols", 999) < TH["CROSS_QUIET"]
                and f.get("max_col_frac", 1) < TH["MCF_QUIET"]),
     0, "high",
     "Type0: no epileptiform activity (silent heatmap, quiet LFP)"),

    # --- fall-through: marginal -> human review ---
    ("review_marginal", lambda f: True, -1, "low",
     "marginal features: route to human adjudication"),
]

LABEL_NAMES = {0: "none (0)", 1: "FIS (1)", 2: "FPFA (2)", 3: "FSW (3)",
               4: "GSW (4)", 5: "seizure (5)", -1: "epileptiform (review)"}


def classify(feats):
    """Return dict(label, label_name, confidence, rule, clause) for a feature dict."""
    for name, pred, label, conf, clause in RULES:
        try:
            hit = pred(feats)
        except Exception:
            hit = False
        if hit:
            return dict(label=label, label_name=LABEL_NAMES[label],
                        confidence=conf, rule=name, clause=clause)
    return dict(label=-1, label_name=LABEL_NAMES[-1], confidence="low",
                rule="review_marginal", clause="fallthrough")
