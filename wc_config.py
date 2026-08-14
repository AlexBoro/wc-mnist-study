"""wc_config.py — Frozen hyperparameters and experiment configuration.

All numeric constants below are copied byte-for-byte from the v82 notebook
cell 7 ("Hyperparameters (fixed & randomized)") and cell 12 (connectivity target
degrees).  Acceptance Test A2 (``model_diff_check.py``) diffs these against the
notebook to guarantee no drift.

The notebook drew randomized hyperparameters from a single global ``rng`` and
relied on cell-execution order.  This module instead exposes the *ranges* as
constants; ``wc_experiment.run_experiment`` draws the concrete values from an
explicit, seed-derived RNG (see §5 of the engineering spec).  Distributions and
ranges are unchanged — only the RNG plumbing differs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

# =============================================================================
# Network size (notebook cell 12)
# =============================================================================
N_SIDE = 28
N = N_SIDE * N_SIDE                       # 784 Wilson-Cowan units (matches MNIST)
TARGET_OUTDEG_PyPy = 156                  # Py -> Py  (even, ring-lattice degree)
TARGET_OUTDEG_PyInh = 156                 # Py -> Inh

# =============================================================================
# Simulation timing (cell 7)
# =============================================================================
DT = 1.0 / 500                            # Euler timestep (s)
IMAGE_DURATION = 0.25                     # seconds (250 ms) per image
OUTPUT_AVERAGE_WINDOW = 0.01              # seconds (10 ms) output-averaging window
MNIST_INPUT_SCALE = 3.0                   # external-input gain on pixel intensities

# Opening window excluded from FEATURE EXTRACTION only -- never from simulation
# or from arrays.npz.  Two independent artefacts live in the first ~1 s:
#   1. The probe IC (E0 ~ U(0, IC_E_MAX)) puts most of the population above
#      threshold, so sample 0 is relaxation from the initial condition, not a
#      response to the first stimulus.  Measured: 760/784 units fire in the
#      opening transient, then activity drops 25-fold.
#   2. save_lfp's zero-phase 4th-order 1 Hz high-pass rings on the step-like
#      opening.  Measured step response: 0.62 s to fall to 10% of peak, 0.71 s
#      to 5%, 1.63 s to 1%.  Padding does NOT shorten this (it is the filter's
#      genuine step response, not an initialisation artefact).
# 1.0 s clears (1) outright and puts (2) below ~5% of peak, at a cost of 10% of
# a 10 s run.  Verified on the 8 near-silent safe-side runs: labels are
# IDENTICAL at 0.5 s and 1.0 s (amp_prominence differs by <=0.007), but at 1.0 s
# no run's labelling peak falls inside the ringing window -- at 0.5 s two did
# (seeds 1104, peak at t=0.88 s).  Resolved at CALL time, so a notebook can set
# cfg.LEAD_IN_S with no re-import.
LEAD_IN_S = 1.0                           # seconds excluded from measurement

# =============================================================================
# Core training control (cell 7 / §4 of the spec)
# =============================================================================
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

# =============================================================================
# Randomized-hyperparameter ranges (cell 7).  Drawn from init_rng per run.
# =============================================================================
P_REWIRE_PyPy_MIN,  P_REWIRE_PyPy_MAX  = 0.001, 0.9
P_REWIRE_PyInh_MIN, P_REWIRE_PyInh_MAX = 0.001, 0.9

BIAS_OUT_MIN, BIAS_OUT_MAX = -4.0, -2.0
OUTPUT_WEIGHT_SCALE_MIN, OUTPUT_WEIGHT_SCALE_MAX = 0.005, 0.02

# Learnable per-unit time-constant *initialization* ranges (cell 7, dated 1/10/26)
TAUPY_MIN,  TAUPY_MAX  = 0.01, 0.08
TAUINH_MIN, TAUINH_MAX = 0.01, 0.08
TAUOUT_MIN, TAUOUT_MAX = 1.0 / 25 * 1.0, 1.0 / 25 * 1.0   # degenerate -> constant 1/25

# Slow adaptation variable H (cell 7, v57/v63)
#
# 2026-08-12: WIDENED from (0.5, 3.0) to (0.01, 10.0) to match the 6/28/26
# reference exploration.  Measured in WC_MNIST_parameter_explore_FINAL.xlsx
# (sequential_H, n=500): tau_H spans 0.0193-9.9547, with 42% of runs above 3.0
# and a floor two orders of magnitude below 0.5.  The old (0.5, 3.0) cap made
# adaptation uniformly SLOW relative to event timescales, so H integrated over
# a whole event and terminated it regardless of how hard the biases drove --
# which is the suspected reason alpha_H displaced bias position as the variable
# controlling seizure here (rho(alpha_H,sz) = -0.73 vs -0.13 in the reference,
# while rho(depth-past-boundary, sz) = +0.02 here vs +0.44 in the reference).
#
# NOTE on the reference distribution: it is a MIXTURE, not a clean uniform.
# 160/500 reference runs sit at exactly tau_H = 1.0; the other 340 are ~uniform
# on (0.019, 9.955).
#
# CORRECTED 2026-08-12.  An earlier version of this note attributed the 1.0
# spike to "the resolve_alpha_tau OFF-path default leaking into H-on runs".
# That is WRONG.  The H-off configs carry 0/500 rows at tau_H == 1.0 (they
# sample tau_H normally and simply never use it), and all 160 pinned rows have
# alpha_H >= 1.0, i.e. adaptation genuinely on.  The real cause is a CODE
# VERSION boundary: the pinned rows are log rows 0-159, one contiguous block,
# and their word_doc filenames identify them as v57 (Jan 15), which held tau_H
# fixed at 1.0; rows 160-499 are v63 (Jan 16+), which sampled it.  The
# "mixture" is two code versions concatenated into one spreadsheet.
#
# CONSEQUENCE for comparisons: this repo ports v63's declarations, so v63 is
# the reference target and tau_H is CONFOUNDED WITH VERSION (rho = 0.65).  The
# subgroup contrast that earlier appeared here as evidence for a tau_H effect
# (0.978 sampled vs 0.810 pinned, Fisher p = 0.032) does NOT survive adding a
# version indicator, which flips the tau_H coefficient negative (LR chi2 =
# 5.73, p = 0.017).  Within v63 alone, rho(tau_H, seizure) = -0.030 (p = 0.85),
# because 43/44 in-band v63 runs seize -- the reference is at ceiling in-band
# and cannot resolve a tau_H effect at all.  Compare against v63 only, never
# the pooled rate and never the "sampled tau_H" subgroup.
#
# The widening to (0.01, 10.0) is justified by OUR OWN 60-run experiment, which
# is internally controlled: the alpha_H-matched paired arm flipped 8 seeds to
# seizure and 0 back (McNemar p = 0.008) and restored rho(depth, seizure) from
# +0.015 to +0.250.  We sample uniformly and do NOT reproduce the 1.0 spike,
# which is an artifact of superseded reference code.
TAU_H_MIN, TAU_H_MAX = 0.01, 10.0
# Initial STATE of the adaptation variable H (not its strength -- that is
# ALPHA_H_ON_* below).  Explore path: H0 ~ U(H_INIT_MIN, H_INIT_MAX) per unit,
# drawn from init_rng (param_explore.py, wc_experiment.py, timing_calibration.py).
#
# NO ASYMMETRY between probe and training.  A note here previously claimed the
# TRAINING path started H at exactly 0; that was WRONG and is retracted.  The
# `H = torch.zeros(N)` fallbacks inside wc_model.py's inner functions are
# UNREACHABLE from either entry point: wc_experiment.run_experiment draws
# H0 from the same init_rng over the same U(H_INIT_MIN, H_INIT_MAX) range and
# threads it through, and both existing call sites pass it.  The 6/28/26
# reference notebook randomizes H0 in its training driver the same way.
#
# The genuine probe/training difference is in the OTHER state variables (E0/I0,
# see IC_E_MAX/IC_I_MAX below), which is faithful to the reference.
#
# 2026-08-13: H_INIT_MAX raised 0.01 -> 0.15 on the 22-seed dose response
# (h0_dose_summary.csv).  At H0 ~ 0 adaptation starts fully discharged and the
# network must charge it up; that charging transient IS the opening burst that
# was being labelled GSW.  Raising the bound removes those artefactual
# transients while leaving genuine sustained seizures intact -- the class
# transition matrix is strictly lower-triangular and no sustained run was lost.
# Separation (seizure median / GSW median) improves 8.3x -> 20.1x.  0.15 is
# equilibrium-consistent: mean H0 = 0.075 against an E operating point of 0.138.
# Do NOT raise further -- at 1.0 the t=0 adaptation term overlaps |bias_e| and
# flattens every class to silence (separation 1.2x, 2/22 runs active).
# NOTE: this invalidates bundles generated at 0.01 as a like-for-like set.
H_INIT_MIN, H_INIT_MAX = 0.0, 0.15
# Initial state of the free-running probe: E0 ~ U(0, IC_E_MAX), I0 ~ U(0, IC_I_MAX).
# These are the 6/28/26 notebook's values (run_exploration_simulation, IC_E_MAX=IC_I_MAX=1.0).
# The repo previously started the probe at E=I=0, which is a different basin for a
# bistable network and suppressed seizures in the sequential configurations.
IC_E_MAX, IC_I_MAX = 1.0, 1.0

# H-current strength.  OFF configs use 0.0 (H-current is off iff alpha_H == 0.0).
# The H-on config samples Uniform(ALPHA_H_ON_MIN, ALPHA_H_ON_MAX) (spec §2/§6,
# restoring the notebook's commented-out 1..10 range).
ALPHA_H_ON_MIN, ALPHA_H_ON_MAX = 1.0, 10.0

# Initial-value ranges for learnable parameters (cell 7, dated 1/6/26 & 1/10/26)
BIAS_E_INIT_MIN, BIAS_E_INIT_MAX = -8.0, 0.0
BIAS_I_INIT_MIN, BIAS_I_INIT_MAX = -12.0, 0.0
MNIST_WEIGHT_MIN, MNIST_WEIGHT_MAX = -12.0, 13.0
INH2PY_GAIN_MIN, INH2PY_GAIN_MAX = 12.0, 37.0
# Widened and EQUALIZED (see the drive note under W_CELOC below).  Equal means
# for the two E->E populations are half of what makes drive flat in p_rewire;
# equal coefficients are the other half.
#
# 2026-08-10: REVERTED to (-0.6, 3.0).  A 2026-08-01 edit had widened these
# to (0.0, 5.0); that change is retracted because it moved TWO things at
# once and the two are not separable after the fact:
#   (a) drive:  per-unit E->E drive 18.72 -> 39.00, i.e. 2.13x the 6/28/26
#       reference networks, when the goal was to MATCH them; and
#   (b) ensemble: a 0.0 lower bound makes NO E->E edge negative, deleting
#       the negative-weight fraction that is present in June (16.7% of
#       CeLoc, 21.1% of CeRem) and in (-0.6, 3.0) (16.7%).
# (-0.6, 3.0) is the drive-MATCHED range: half as many E->E edges as June
# (156 vs 312, one split ring vs two full WS graphs), each ~2x stronger.
#   June   156*0.15*mean U(-0.3,1.5) + 156*0.05*mean U(-0.4,1.5) = 18.33
#   here   156*0.10*mean U(-0.6,3.0)                             = 18.72
# KNOWN RESIDUAL: matching the mean does not match the variance.  Fewer,
# stronger edges give a wider per-unit drive spread: CV 0.069 here vs
# 0.056 in June (sd ratio 1.26).  Recorded, not corrected.
CELOC_INIT_SCALE_MIN,  CELOC_INIT_SCALE_MAX  = -0.6, 3.0
CEREM_INIT_SCALE_MIN,  CEREM_INIT_SCALE_MAX  = -0.6, 3.0
CELOCI_INIT_SCALE_MIN, CELOCI_INIT_SCALE_MAX = -0.7, 1.5

# =============================================================================
# Recurrent E->E drive coefficients
# =============================================================================
# The Py2Py update line, in one place:
#
#     Py2Py = W_PY2PY_SELF * E + W_CELOC * (CeLoc @ E) + W_CEREM * (CeRem @ E)
#
# These were bare literals at six sites (three model paths, the LFP proxy in
# wc_figures, model_diff_check, dual_lfp_prototype).  The LFP proxy's excitatory
# part is EXACTLY this expression restricted to the electrode rows, so the proxy
# and the model must use the same numbers or the proxy silently stops measuring
# the model it claims to measure.  Naming them makes that coupling mechanical.
#
# W_CELOC vs W_CEREM and the p_rewire confound: Branch A splits ONE ring of
# TARGET_OUTDEG_PyPy edges between CeLoc and CeRem according to p_rewire, so
# per-unit drive is
#
#     W_PY2PY_SELF + K * [ W_CELOC * mean(CeLoc) * (1-p) + W_CEREM * mean(CeRem) * p ]
#
# whose slope in p is K * (W_CEREM*mean(CeRem) - W_CELOC*mean(CeLoc)).  With
# unequal coefficients that slope is nonzero and p_rewire silently carries an
# excitation difference (corr = -1.0 exactly).  Equal coefficients AND equal
# weight means make the slope exactly zero at every K, which is the flat
# drive-vs-p that Branch B always had by construction.
W_PY2PY_SELF = 10.0
# Equal, and set to the value that reproduces Branch B's per-unit excitation:
#   Branch B dealt a FULL ring into CeLoc AND another into CeRem (312 in-edges
#   per unit at every p) so its drive was 10 + 156*0.15*0.600 + 156*0.05*0.550
#   = 28.33, already flat in p.  Branch A has 156 in-edges, so matching that
#   level needs 10 + 156*w*1.2 = 28.33 -> w = 0.0979; 0.10 gives 28.72 (1.014x).
# The pilot brackets this with three arms: 0.075 (the CURRENT Branch A drive at
# p->0, i.e. 'changed the tilt, nothing else'), 0.10 (Branch-B-matched), and
# 0.15 (1.34x Branch B).  The study-wide default is the matched value.
W_CELOC = 0.10
W_CEREM = 0.10

# The three pilot arms, so the notebook and the analysis agree on what they are.
PILOT_COEFF_ARMS = {
    "w075": 0.075,   # 24.04 = current Branch A drive at p->0 (0.849x Branch B)
    "w100": 0.100,   # 28.72 = Branch-B-matched (1.014x)
    "w150": 0.150,   # 38.08 = 1.344x Branch B
}

# Inhibitory coefficient of the LFP proxy only (wc_figures.save_lfp); the model
# itself uses the learnable per-unit inh_to_exc_gain, not a fixed scalar.
W_LFP_INH = 25.0

# Fixed y-axis span of the LFP ladder, in signal units (wc_figures.save_lfp).
# The axis is deliberately NOT autoscaled: a constant span makes
# pixels-per-signal-unit identical on every figure, which is what lets a fixed
# pixel-deflection threshold transfer across runs and keeps flat epochs readable.
#
# 150.0 was calibrated 2026-07-25 against the OLD E->E coefficients (0.15/0.05).
# The rework changed signal amplitude, so 150 was PROVISIONAL and is now
# superseded.
#
# RECALIBRATED 2026-08-14 from the 20-seed x 3-config pilot (60 runs, all
# measured -- see the volume-visibility note below). Measured peak |deflection|
# of the filtered, un-offset traces:
#     min 2.15   median 10.97   p90 28.52   p99 34.05   max 34.48
# The distribution is BIMODAL, not tailed: 27 quiet runs at 2.2-6.1 and 33 active
# runs at 10.2-34.5, with the gap doing the separating. So the span is set from
# the active mode's ceiling, not from a Gaussian tail extrapolation (which
# projected a spurious 243 off the pooled sd).
#
# Required span = ladder (27*5.0 = 135) + 2 * peak, since the traces excurse both
# ways around the ladder. Max required = 135 + 2*34.48 = 204.0, and 55% of pilot
# runs exceeded 150 -- i.e. the old value clipped the majority of runs.
#
# 240 gives 52.5 per side: 1.52x the largest peak observed in 60 runs. The margin
# is deliberate because the full sweep is 25x larger and will sample further out
# than the pilot did, and because clipping is not a cosmetic defect here: 12 of
# the classifier's 13 inputs are PIXEL-derived from lfp.png (only amp_prominence
# is analytic), and the channel-extent rules -- sync_spike_extent,
# involve_contig_dur_s, vertical_spread_fraction, burst_band_frac -- count
# exactly the top/bottom electrodes that clipping truncates. A clipped figure
# therefore mislabels runs, it does not merely look worse.
#
# Changing this is a RE-RENDER, not a re-simulate: arrays.npz is untouched.
LFP_SPAN = 240.0

# =============================================================================
# Dataset split (cell 11 / §4): deterministic 50k/10k, split RNG seed 123
# =============================================================================
SPLIT_SEED = 123
TRAIN_SIZE = 50000                        # remaining 10k -> mnist_val

# =============================================================================
# Fixed study settings (§4)
# =============================================================================
NUM_EPOCHS = 100
IMAGES_PER_EPOCH = 256                    # resampled from shuffled train loader each epoch
EVAL_IMAGES = 256                         # fixed held-out subset, identical every epoch

# Default output root (Colab Drive).  Overridable per call / via WC_OUT_ROOT env.
OUT_ROOT = "/content/drive/MyDrive/WC_MNIST_Longitudinal"

# =============================================================================
# Topology-trigger calibration (§8 / §17 open items)
# =============================================================================
TOPO_TRACK_INTERVAL = 4                   # scheduled cadence: (epoch+1) % 4 == 0
TOPO_SMOOTH_WINDOW = 2                    # trailing window for delta smoothing
TOPO_EPS_LOSS = 0.01                      # sign-flip deadband on train_loss
TOPO_EPS_ACC = 1.0                        # sign-flip deadband on train_acc (pct-pt)
TOPO_MIN_GAP = 2                          # min epochs between topology measurements

# Topology grid/Hessian config (cell 16 TOPOLOGY_CONFIG, frozen)
TOPO_GRID_SIZE = 15
TOPO_PERTURBATION_RANGE = 2.0
TOPO_NUM_SAMPLES = 500                    # grid loss-eval images
TOPO_HESSIAN_NUM_SAMPLES = 100
TOPO_LANCZOS_ITERATIONS = 50

# =============================================================================
# Seizure-boundary constants
# Logistic regression (C=1e4) on realized (bias_e, bias_i); bI = m*bE + c.
# Applied at analysis time only — simulations never depend on these, so a
# refit can always be applied post hoc to metrics.parquet (bias_e, bias_i,
# grad_bias_e, grad_bias_i are all recorded per epoch).
#
# 2026-08-14 REFIT on the 1500-run v3 hand-scored corpus (500 seeds x 3
# configs; param_explore_v3_classified.csv joined to all_runs_params.csv).
# The previous values were the 28 Jun 2026 fit, which predates both the v3
# corpus and hand-scoring.  Re-scoring every run changes seizure side for:
#     sequential_H    29/500 (5.8%)   hand-score agreement 93.6% -> 96.6%
#     simultaneous    12/500 (2.4%)                        96.4% -> 96.0%
#     sequential_noH   1/500 (0.2%)                        99.0% -> 99.2%
# 22 of the 29 H-on flips fall in the 6-10 stratum, whose intercept moves
# most (-3.37 -> -6.82).  That is the configuration this study cares about,
# which is why this was updated before the 100-epoch runs rather than
# recomputed afterwards.  See boundary_refit_v3.csv / boundary_drift.png.
# =============================================================================
BOUNDARY_FIT_PROVENANCE = "v3-handscored-1500run-2026-08-14"
BOUNDARY_SIMULTANEOUS = (2.357697, 0.036471)     # (slope m, intercept c)
BOUNDARY_SEQUENTIAL_NOH = (2.005077, -2.522236)
BOUNDARY_SEQUENTIAL_H_AGGREGATE = (1.950845, -4.499028)
# alpha_H-stratified boundaries for the H-on config: stratum -> (m, c)
BOUNDARY_SEQUENTIAL_H_STRATA = {
    "0-3": (1.880765, -3.176108),
    "3-6": (2.146298, -3.221631),
    "6-10": (1.408629, -6.818616),
}


def get_boundary(config_name: str, alpha_H: float = 0.0) -> tuple[float, float]:
    """Return (slope m, intercept c) for the seizure boundary of a config.

    For ``sequential_H`` the alpha_H-stratified boundary is used (the stratum
    matching the run's sampled alpha_H); falls back to the aggregate fit if the
    value is out of the calibrated range.
    """
    if config_name == "simultaneous":
        return BOUNDARY_SIMULTANEOUS
    if config_name == "sequential_noH":
        return BOUNDARY_SEQUENTIAL_NOH
    if config_name == "sequential_H":
        if 0.0 <= alpha_H < 3.0:
            return BOUNDARY_SEQUENTIAL_H_STRATA["0-3"]
        if 3.0 <= alpha_H < 6.0:
            return BOUNDARY_SEQUENTIAL_H_STRATA["3-6"]
        if 6.0 <= alpha_H <= 10.0:
            return BOUNDARY_SEQUENTIAL_H_STRATA["6-10"]
        return BOUNDARY_SEQUENTIAL_H_AGGREGATE
    raise ValueError(f"Unknown config_name: {config_name!r}")


# =============================================================================
# Experiment configuration
# =============================================================================
@dataclass
class ExperimentConfig:
    """One configuration within a set (§2).

    A *set* is three of these sharing a seed; they differ only in ``input_mode``
    and ``alpha_h_mode``.  Everything else is a fixed study-wide hyperparameter,
    carried here so a run is fully described by ``(seed, ExperimentConfig)``.
    """

    name: str                              # "simultaneous" | "sequential_noH" | "sequential_H"
    input_mode: str                        # "simultaneous" | "sequential"
    alpha_h_mode: str                      # "off" | "uniform_1_10"

    # ---- Recurrent E->E drive coefficients ----
    # Carried on the config (not read from the module at use time) so a pilot
    # arm that varies them is fully described by (seed, ExperimentConfig) and
    # lands in the run manifest.  Defaults are the study-wide values.
    w_celoc: float = W_CELOC
    w_cerem: float = W_CEREM

    # Label for a coefficient variant (a key of PILOT_COEFF_ARMS, or "" for the
    # study defaults).  Pure metadata -- it never enters the dynamics -- but it
    # reaches run_id, params.json and the manifest, which is what keeps three
    # arms at one seed from resolving to the same output directory.
    arm: str = ""

    # ---- Targeted bias overrides (2026-08-10) ----
    # When None (the default), bias_e_init / bias_i_init are DRAWN from
    # init_rng exactly as before and nothing about a run changes.  When set,
    # the draws still happen -- so the RNG stream position, and therefore the
    # connectivity and every downstream weight, is bit-identical to the
    # unoverridden run at the same seed -- but the drawn values are discarded
    # and these are used instead.
    #
    # This is what lets a pilot place (bias_e, bias_i) on a DESIGNED grid
    # relative to the 6/28/26 seizure boundary while `seed` reverts to what it
    # should be: a pure connectivity replicate.  Selecting seeds on their
    # drawn biases instead would confound the two, and would oversample the
    # corner of the region (seeds landing in-region for all three configs have
    # bias_e median -0.76 vs the June seizure-side median of -1.25).
    #
    # The realized values reach params.json via _collect_params, which reads
    # model.bias_e / model.bias_i rather than the draw, so a run always records
    # the biases it actually used.
    bias_e_override: Optional[float] = None
    bias_i_override: Optional[float] = None

    # ---- Epoch-0 initial condition (2026-08-14) ----
    # E0 ~ U(0, ic_e_max), I0 ~ U(0, ic_i_max) per unit, drawn from a SEPARATE
    # generator seeded SeedSequence([seed, 0xE0]) -- deliberately not init_rng,
    # so changing the IC cannot shift the init_rng stream position and network
    # parameters stay bit-identical to a run at any other IC.  Same discipline
    # and same generator seeding as param_explore.run_exploration_simulation, so
    # a training run and a probe at the same seed start from the SAME E0/I0.
    #
    # This closes the last probe/training asymmetry.  The training path formerly
    # started at E=I=0; for a bistable network that is a different basin, not a
    # transient (it dropped the probe's past-boundary rate in the sequential
    # configurations to 0.11 vs the notebook's 0.94).  Because training state
    # carries over across all 100 epochs, epoch 0's basin is the initial
    # condition for the whole ~13,000 s trajectory, not a 1 s transient.
    #
    # O0 is zero in both paths (the output layer has no free-running dynamics
    # before the first stimulus reaches it).
    ic_e_max: float = IC_E_MAX
    ic_i_max: float = IC_I_MAX

    # ---- Activity-array capture (2026-08-14) ----
    # Save the TRAIN_HEATMAP_WINDOW of E/I as arrays.npz alongside the PNG, on
    # the same epochs the heatmap is rendered.  Without this the window goes
    # straight to a figure and is discarded, so training epochs cannot be run
    # through labeling_toolkit -- "the network approached the seizure boundary"
    # stays a claim about a line in bias space and never becomes an observation
    # of seizure-like activity.  Keys match param_persist.save_bundle
    # (E_traj/I_traj/times/stim_onsets) so the same feature code reads both.
    save_activity_arrays: bool = True

    # ---- Fixed hyperparameters (§4) ----
    num_epochs: int = NUM_EPOCHS
    images_per_epoch: int = IMAGES_PER_EPOCH
    eval_images: int = EVAL_IMAGES
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    dt: float = DT
    image_duration: float = IMAGE_DURATION
    output_average_window: float = OUTPUT_AVERAGE_WINDOW
    mnist_input_scale: float = MNIST_INPUT_SCALE
    split_seed: int = SPLIT_SEED

    # ---- Stimulus scheduling ----
    # PULSED_STIMULI was a sampled knob in the notebook; the longitudinal study
    # fixes it OFF so the eval/train protocol is identical across runs (§4).
    pulsed_stimuli: bool = False

    # ---- Topology trigger (§8) ----
    topo_track_interval: int = TOPO_TRACK_INTERVAL
    topo_smooth_window: int = TOPO_SMOOTH_WINDOW
    topo_eps_loss: float = TOPO_EPS_LOSS
    topo_eps_acc: float = TOPO_EPS_ACC
    topo_min_gap: int = TOPO_MIN_GAP

    # ---- Clean-gradient probe (§7) ----
    # Images from the fixed eval subset used for the per-epoch gradient probe.
    # 64 (a prefix of the fixed eval subset) estimates the bias-gradient direction
    # well at a fraction of the cost of the full 256; raise toward EVAL_IMAGES for
    # a lower-variance estimate.
    grad_probe_images: int = 64

    @property
    def sequential_row_input(self) -> bool:
        return self.input_mode == "sequential"

    @property
    def row_duration(self) -> float:
        return self.image_duration / 28

    def to_dict(self) -> dict:
        return asdict(self)


def make_configs() -> list[ExperimentConfig]:
    """The three paired configurations of a set (§2)."""
    return [
        ExperimentConfig(
            name="simultaneous",
            input_mode="simultaneous",
            alpha_h_mode="off",
        ),
        ExperimentConfig(
            name="sequential_noH",
            input_mode="sequential",
            alpha_h_mode="off",
        ),
        ExperimentConfig(
            name="sequential_H",
            input_mode="sequential",
            alpha_h_mode="uniform_1_10",
        ),
    ]


def make_config(name: str) -> ExperimentConfig:
    """Return the single config with the given name."""
    for c in make_configs():
        if c.name == name:
            return c
    raise ValueError(f"Unknown config name: {name!r}")
