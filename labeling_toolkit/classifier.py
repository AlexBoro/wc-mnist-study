"""classifier.py — declarative decision table for the WC-MNIST taxonomy.

The ENTIRE classification logic is the ordered RULES list below. classify(feats)
walks it top-down and returns the FIRST matching rule's verdict, with provenance
(which rule fired + the taxonomy clause it implements). Adding or changing a type
= editing one row here; nothing is nested or hidden. See taxonomy_spec.md for the
definitions each rule encodes; features come from features.py::extract_features.

Design invariants (from taxonomy_spec.md):
  * Morphology is primary; mean_E / val_acc corroborate confidence only.
  * Conservative screen: marginal feature values -> 'review', never a forced call.
  * Tie-break when genuinely arbitrary: err toward Type 0.

Label codes: 0 none, 1 FIS, 2 FPFA, 3 FSW, 4 GSW, 5 seizure(ictal), -1 review.
"""

# ---- named thresholds (ALL tunables live here, each with its calibration note) ----
TH = dict(
    # -- Type 0 silence floor --
    INK_SILENT      = 0.02,    # heatmap essentially blank
    MCF_QUIET       = 0.15,    # no vertical recruitment (FIX 1: guards the none arm)
    CROSS_QUIET     = 20,      # few/no LFP crossings

    # -- vertical recruitment / spread --
    MCF_RECRUIT     = 0.40,    # max_col_frac >= this = real vertical recruitment (3/4/5)
    VSPREAD_FULL    = 0.75,    # vertical_spread_fraction >= this = generalized (GSW); below = focal (FSW)

    # -- ictal (Type 5) --
    SEIZ_COMPLEXES  = 4,       # >4 complexes ...
    IDI_DISCERN     = 1.0,     # ... with NO IDI >= this (s) = sustained no-IDI = ictal spike-wave
    SUSTAINED_FRAC  = 0.35,    # fraction-of-window discharging; high = sustained
    SPREAD_CHANS    = 4,       # >4 LFP channels ...
    SPREAD_DUR_S    = 2.0,     # ... showing fast activity for >= this (s) = network-spread PFA = ictal
    CROSS_ICTAL     = 200,     # very high crossing count corroborates sustained spike-wave

    # -- energy / function corroboration (NOT definitional) --
    MEANE_ICTAL     = 0.35,    # mean_E at/above ictal range
    VALACC_COLLAPSE = 0.25,    # val_acc collapsed toward chance corroborates ictal

    # -- FPFA / FIS --
    INK_ACTIVE      = 0.02,    # above silence floor = some activity present
    CROSS_FIS_LO    = 1,       # FIS crossing marginal band low ...
    CROSS_FIS_HI    = 6,       # ... high (1-6 = human-review zone)

    # -- LFP-correlate gate (labels 1-5 REQUIRE an LFP correlate; else Type 0) --
    RHYTHM_CHANS    = 3,       # >=this channels showing sustained fast activity ...
    RHYTHM_DUR_S    = 2.0,     # ... for >=this (s) = rhythmic-oscillation correlate

    # -- FIS vs FSW: synchronous-spike spatial extent --
    SYNC_CHANS      = 4,       # discrete spike touching <=this adjacent channels = FIS; >this = FSW
    SYNC_SPIKE_MIN  = 2,       # a synchronous discrete spike on >=this adjacent channels is a
                               # valid LFP correlate on its own (a single focal spike crosses too
                               # few image columns to clear CROSS_FIS_HI, but IS a real FIS)
)


def _has_discrete_spikes(f):
    """A discrete epileptiform spike is present on the LFP. Two independent detectors,
    EITHER sufficing:
      - cross_cols > CROSS_FIS_HI : column-crossing count, tuned for spike-WAVE trains
        (dozens+ crossings). Under-counts a single focal spike (crosses ~3-4 columns).
      - sync_spike_extent >= SYNC_SPIKE_MIN : synchronous sharp transient on >=2 adjacent
        channels. This IS a focal interictal spike by definition and catches the single
        spike the crossing counter misses (seed003 ep15 failure mode, user 2026-07-25).
    """
    return (f.get("cross_cols", 0) > TH["CROSS_FIS_HI"]
            or f.get("sync_spike_extent", 0) >= TH["SYNC_SPIKE_MIN"])


def _has_lfp_correlate(f):
    """Labels 1-5 require a morphological correlate ON THE LFP: either discrete
    spikes (_has_discrete_spikes) or rhythmic oscillation (sustained fast activity on
    >=RHYTHM_CHANS channels for >=RHYTHM_DUR_S). Heatmap bands / persistent activity
    that outlast the stimulus WITHOUT any LFP pathology are NOT epileptiform
    (user ruling 2026-07-25) -> such epochs fall through this gate to Type 0.
    """
    rhythm = (f.get("n_active_channels", 0) >= TH["RHYTHM_CHANS"]
              and f.get("fast_activity_duration_s", 0.0) >= TH["RHYTHM_DUR_S"])
    return _has_discrete_spikes(f) or rhythm


def _ictal_spikewave(f):
    """>4 complexes with no clearly discernible IDI (sustained spike-wave).

    A spike-wave COMPLEX by definition recruits substantial vertical range, so
    the count must be corroborated by recruitment (max_col_frac >= MCF_RECRUIT).
    Without this guard, the simultaneous-config low-amplitude rhythmic background
    (12 sub-second bursts at max_col_frac ~0.12, ink ~0.065, val_acc preserved
    ~0.85) is mis-counted as a sustained seizure. True spike-wave seizures sit at
    max_col_frac >= 0.77; the guard cleanly separates the two populations.
    """
    n = f.get("n_complexes", 0)
    if n <= TH["SEIZ_COMPLEXES"]:
        return False
    # recruitment guard: complexes must recruit real vertical range
    if f.get("max_col_frac", 0) < TH["MCF_RECRUIT"]:
        return False
    # 'no clearly discernible IDI': the largest gap between complexes stays below
    # the discernible-IDI threshold (LFP doesn't return to baseline between them).
    max_idi = f.get("max_IDI")
    if max_idi is None:
        return False
    return max_idi < TH["IDI_DISCERN"]


def _ictal_spread_pfa(f):
    """>4 LFP channels showing paroxysmal fast activity for >= 2 s."""
    return (f.get("n_active_channels", 0) > TH["SPREAD_CHANS"]
            and f.get("fast_activity_duration_s", 0.0) >= TH["SPREAD_DUR_S"])


def _energy_corroborates_ictal(f):
    me, va = f.get("mean_E"), f.get("val_acc")
    hot = (me is not None and me >= TH["MEANE_ICTAL"])
    collapsed = (va is not None and va <= TH["VALACC_COLLAPSE"])
    return hot or collapsed


# ------------------------------------------------------------------- RULES ----
# Each rule: (name, predicate(f)->bool, label, confidence, clause).
# Order matters: first match wins. Seizure arms precede interictal precede none.
RULES = [
    # --- LFP-correlate GATE (checked FIRST): no LFP spikes/rhythm -> Type 0 ---
    # Enforces "labels 1-5 require an LFP correlate". A heatmap that is hot with
    # bands or persistent post-stimulus activity but shows NO discrete spikes and
    # NO rhythmic oscillation on the LFP traces is NOT epileptiform -> Type 0,
    # regardless of ink_frac / band_score. This supersedes every epileptiform arm.
    ("no_lfp_correlate", lambda f: not _has_lfp_correlate(f), 0, "high",
     "Type0: no LFP correlate (no discrete spikes, no rhythmic oscillation) -> not epileptiform"),

    # --- Type 5: ictal (checked FIRST so a real seizure is never swallowed) ---
    ("ictal_spikewave_sustained", _ictal_spikewave, 5, "high",
     "Type5: >4 spike-wave complexes, no discernible IDI (sustained)"),

    ("ictal_spread_pfa", _ictal_spread_pfa, 5, "high",
     "Type5: paroxysmal fast activity spread over >4 channels for >=2 s"),

    # sustained recruitment corroborated by energy collapse, even if the
    # crossing/complex detectors under-count (ep51/ep63 failure mode: modest
    # amplitude sustained ictal that crossings can't see). Requires real
    # recruitment (ink) AND energy/function corroboration -> med confidence.
    ("ictal_energy_corroborated",
     lambda f: (f.get("ink_frac", 0) >= 0.25
                and f.get("sustained_fraction", 0) >= TH["SUSTAINED_FRAC"]
                and _energy_corroborates_ictal(f)),
     5, "med",
     "Type5: sustained recruitment + energy/function collapse (crossing-silent ictal)"),

    # --- Type 3/4: interictal spike-wave (discrete, real recruitment) ---
    # Generalized (full vertical spread) vs focal (partial). Discrete = fails the
    # ictal no-IDI/count test above, so any recruitment reaching here is interictal.
    ("gsw_generalized",
     lambda f: (f.get("max_col_frac", 0) >= TH["MCF_RECRUIT"]
                and f.get("vertical_spread_fraction", 0) >= TH["VSPREAD_FULL"]),
     4, "med",
     "Type4: discrete generalized spike-wave, full vertical spread, clear IDIs"),

    ("fsw_focal",
     lambda f: f.get("max_col_frac", 0) >= TH["MCF_RECRUIT"],
     3, "med",
     "Type3: discrete focal spike-wave, partial vertical spread, clear IDIs"),

    # --- Type 2: FPFA (focal, sustained-but-not-spread fast activity) ---
    # LFP shows fast activity on a channel subset that does NOT meet the network-
    # spread ictal bar. band_score is ADVISORY only, never sole trigger.
    ("fpfa_focal",
     lambda f: (f.get("n_active_channels", 0) >= 2
                and f.get("fast_activity_duration_s", 0.0) >= 0.5
                and f.get("n_active_channels", 0) <= TH["SPREAD_CHANS"]),
     2, "med",
     "Type2: focal paroxysmal fast activity (run-like, outlasts stim, not spread)"),

    # --- Type 1/3: discrete spikes without strong recruitment, split by LFP extent ---
    # These reach here having failed the ictal + recruited-spike-wave arms: discrete
    # crossings present (cross_cols>CROSS_FIS_HI), low heatmap recruitment
    # (max_col_frac<MCF_RECRUIT). The FIS-vs-FSW boundary is the LFP channel-count
    # rule (user 2026-07-25): a near-simultaneous spike spanning <=SYNC_CHANS adjacent
    # channels is FOCAL interictal (FIS); spanning >SYNC_CHANS is FOCAL spike-wave (FSW).
    ("fis_focal",
     lambda f: (_has_discrete_spikes(f)
                and f.get("max_col_frac", 0) < TH["MCF_RECRUIT"]
                and f.get("sync_spike_extent", 0) <= TH["SYNC_CHANS"]),
     1, "med",
     "Type1: focal interictal spikes (discrete spike, <=4-channel synchronous extent)"),

    ("fsw_discrete_spread",
     lambda f: (_has_discrete_spikes(f)
                and f.get("max_col_frac", 0) < TH["MCF_RECRUIT"]
                and f.get("sync_spike_extent", 0) > TH["SYNC_CHANS"]),
     3, "med",
     "Type3: focal spike-wave (discrete spike, >4-channel synchronous extent)"),

    # --- Type 0: none (FIX-1 guarded silence) ---
    ("none_silent",
     lambda f: (f.get("ink_frac", 1) < TH["INK_SILENT"]
                and f.get("cross_cols", 999) < TH["CROSS_QUIET"]
                and f.get("max_col_frac", 1) < TH["MCF_QUIET"]),
     0, "high",
     "Type0: no epileptiform activity (silent, guarded on max_col_frac)"),

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
    # unreachable (last rule always matches) but keep the contract total
    return dict(label=-1, label_name=LABEL_NAMES[-1], confidence="low",
                rule="review_marginal", clause="fallthrough")
