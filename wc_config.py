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
TAU_H_MIN, TAU_H_MAX = 0.5, 3.0
H_INIT_MIN, H_INIT_MAX = 0.0, 0.01

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
# The rework changes signal amplitude, so this value is PROVISIONAL: the pilot
# reports measured peak deflections (save_lfp(..., return_stats=True)) and this
# is then set so pixels-per-signal-unit matches the old figures. Leaving it at
# 150 while the drive changes would silently rescale every figure.
LFP_SPAN = 150.0

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
# Seizure-boundary constants (§10, FINAL labels 28 Jun 2026)
# Logistic regression (C=1e4) on (bias_e_mean, bias_i_mean); bI = m*bE + c.
# Applied at analysis time only — simulations never depend on these.
# =============================================================================
BOUNDARY_SIMULTANEOUS = (2.24, -0.78)            # (slope m, intercept c)
BOUNDARY_SEQUENTIAL_NOH = (2.10, -2.45)
BOUNDARY_SEQUENTIAL_H_AGGREGATE = (2.19, -2.81)
# alpha_H-stratified boundaries for the H-on config: stratum -> (m, c)
BOUNDARY_SEQUENTIAL_H_STRATA = {
    "0-3": (2.29, -2.52),
    "3-6": (2.10, -2.22),
    "6-10": (2.06, -3.37),
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
