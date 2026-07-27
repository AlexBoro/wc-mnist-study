# WC-MNIST Parameter-Exploration Study — Clean Redo Specification

**Status:** canonical spec for the 2026-07 redo. Supersedes the accumulated
`WC_MNIST_parameter_explore_FINAL.xlsx` study (1500 runs assembled from 1786
runs across many notebook versions), which had no single reproducible generator.
This spec produces the same 500×3 design from one pinned code version so that
every run is fully reconstructable.

---

## 1. Purpose

Sweep hyperparameters of the Wilson–Cowan MNIST network and record the
free-running dynamics of each sampled network, so that each run can be
classified under the **uniform seizure taxonomy** (`classifier.py` +
`features.py` + `taxonomy_spec.md`) exactly as the longitudinal study is.
No training, no gradient descent, no epoch loop — one 10-second simulation
of a freshly-initialised random network per run.

## 2. Design

| Axis | Value |
|---|---|
| Configurations | 3: `simultaneous`, `sequential_noH`, `sequential_H` |
| Runs per configuration | 500 |
| Total runs | 1500 |
| Per run | one 10.0 s free-running simulation, **no training** |

The three configurations reuse `wc_config.make_configs()` and differ only in:

| Config | `input_mode` | `alpha_h_mode` | H-current |
|---|---|---|---|
| `simultaneous`   | `simultaneous` | `off`         | `alpha_H = 0` |
| `sequential_noH` | `sequential`   | `off`         | `alpha_H = 0` |
| `sequential_H`   | `sequential`   | `uniform_1_10`| `alpha_H ~ U(1,10)` |

Naming matches the longitudinal harness so the taxonomy stays consistent
(`±S` = sequential row-input, `±H` = H-current).

## 3. Simulation

- **Model:** `wc_model.SparseBigWilsonCowan` (dynamics FROZEN — `step()` is
  used verbatim; no code change to the dynamics).
- **Units:** N = 784 (28×28), num_output = 10.
- **Timestep:** `dt = 1/500 s`. `RUN_SECONDS = 10.0` → 5000 Euler steps.
- **Integration:** the frozen `model.step(E, I, O, I_ext, dt, H)` Euler update,
  including the v57 slow-adaptation feedback `input_e = ... − alpha_H·H`,
  `tau_H·dH/dt = −H + E`.
- **Initial conditions:** E = I = O = H = 0 at t=0 (same cold-start as the
  training windows). The *network parameters* are the random draw; the state
  starts at rest and is driven by the MNIST input.
- **Stimulus (MNIST as external drive, cycles through the full 10 s):**
  - `simultaneous`: a new whole image is presented every
    `IMAGE_DURATION = 0.25 s` → `I_ext[:784] = scaled_pixels · mnist_weights`.
    10 s / 0.25 s = 40 image presentations per run.
  - `sequential`: within each 0.25 s image, the 28 rows are presented one at a
    time for `row_duration = 0.25/28 s` each; row *r* drives units `[28r : 28r+28]`.
  - `mnist_input_scale = 3.0` (`cfg.MNIST_INPUT_SCALE`).
  - Images are drawn from the seed-123 MNIST train split, cycled in order; the
    exact image indices used are recorded in the manifest.
- **Recorded outputs:** full `E_traj` and `I_traj` over all 5000 steps
  ([5000, 784] float32), `times` [5000], and the stimulus onset times. Storing
  the full arrays makes the electrode montage a *derived rendering choice*, so
  the figures can be regenerated on any montage without re-simulating.

## 4. Sampling distributions (per run, pinned)

Drawn from a per-run numpy `Generator` seeded from `(seed)` via `build_model`'s
fixed draw order (identical across the three configs of a set → the three share
epoch-0 network parameters; they differ ONLY in input_mode and alpha_H). Ranges
are the **refactor / FINAL ranges** (confirmed by the user 2026-07-27), i.e. the
ranges that generated the published seizure-boundary study — NOT the narrower
v36-notebook ranges.

| Parameter | Distribution | Source constant |
|---|---|---|
| `p_rewire_py_py`  | U(0.001, 0.9)   | `P_REWIRE_PyPy_MIN/MAX` |
| `p_rewire_py_inh` | U(0.001, 0.9)   | `P_REWIRE_PyInh_MIN/MAX` |
| `bias_out`        | U(−4.0, −2.0)   | `BIAS_OUT_MIN/MAX` |
| `output_weight_scale` | U(0.005, 0.02) | `OUTPUT_WEIGHT_SCALE_MIN/MAX` |
| `bias_e_init`     | U(−8.0, 0.0)    | `BIAS_E_INIT_MIN/MAX` |
| `bias_i_init`     | U(−12.0, 0.0)   | `BIAS_I_INIT_MIN/MAX` |
| `tauPy` (per-unit)| U(0.01, 0.08)   | `TAUPY_MIN/MAX` |
| `tauInh` (per-unit)| U(0.01, 0.08)  | `TAUINH_MIN/MAX` |
| `tauOut`          | constant 1/25   | `TAUOUT_MIN/MAX` (degenerate) |
| `mnist_weight` (per-unit) | U(−12, 13) | `MNIST_WEIGHT_MIN/MAX` |
| `inh2py_gain` (per-unit)  | U(12, 37)  | `INH2PY_GAIN_MIN/MAX` |
| `celoc_scale`     | U(−0.3, 1.5)    | `CELOC_INIT_SCALE_MIN/MAX` |
| `cerem_scale`     | U(−0.4, 1.5)    | `CEREM_INIT_SCALE_MIN/MAX` |
| `celoci_scale`    | U(−0.7, 1.5)    | `CELOCI_INIT_SCALE_MIN/MAX` |
| `alpha_H`         | 0.0 (off configs) / U(1.0, 10.0) (`sequential_H`) | `ALPHA_H_ON_MIN/MAX` |
| `tau_H`           | U(0.5, 3.0) — **nominal** (overridable knob) | `TAU_H_MIN/MAX` |

**`tau_H` note:** the FINAL data has `tau_H` up to ~10 (an accumulation
artifact across notebook versions). The user confirmed the clean nominal
range **U(0.5, 3.0)** for the redo. `tau_H` only affects `sequential_H`
(the two off-configs have `alpha_H = 0`, so H is decoupled from the dynamics).

Connectivity: `build_connectivity(rng, p_rewire_py_py, p_rewire_py_inh, device)`
→ small-world CeLoc (local) + CeRem (long-range) + CeLocI, ring-lattice degree
`TARGET_OUTDEG_PyPy = TARGET_OUTDEG_PyInh = 156`.

## 5. Per-run reproducibility bundle

Written under `OUT_ROOT/<config_name>/run_<seed>/`:

- **`arrays.npz`** — `E_traj` [5000,784] f32, `I_traj` [5000,784] f32,
  `times` [5000] f32, `stim_onsets` [f32], `mnist_indices` [int]. (Optionally
  downcast E/I to f16 if size is a concern; f32 default.)
- **`params.json`** — every sampled hyperparameter (scalars + per-unit array
  summary stats: min/max/mean/std for tauPy/tauInh/bias/mnist_weight/gains/
  CeLoc/CeRem/CeLocI scales), the config name, `alpha_H`, `tau_H`, `seed`,
  `runtime_s`.
- **`manifest.json`** — full reconstruction key:
  - `git_commit` (SHA of the pinned module) and `code_tag`
  - `config_name`, `seed`, `run_id`
  - RNG seeds: numpy, torch, python (and the derived `init_rng` seed)
  - `RUN_SECONDS`, `dt`, `N`, `num_output`, `mnist_input_scale`, `split_seed`
  - library versions: python, numpy, scipy, torch, CUDA; `gpu_name`
  - timestamp

A run is **complete** iff all three files exist and `arrays.npz` loads with the
expected shapes. The driver **skips** complete runs (resumable). A per-run row
is appended to a master log (`param_explore_log.csv`, flushed per row) as runs
finish.

**Reconstruction guarantee:** given `manifest.json` (code_tag + seeds) the run
reproduces bit-for-bit on the same GPU/library stack; given `arrays.npz` the
figures and classification reproduce without re-simulating.

## 6. Figure rendering (uniform with the longitudinal classifier)

Rendered from `arrays.npz`, NOT captured live, using the longitudinal montage
so `classifier.py`/`features.py` apply unchanged:

- **`heatmap_E.png`** — E-activity heatmap, viridis, **fixed `vmin=0, vmax=1`**
  (`wc_figures.save_heatmap`-style), colorbar.
- **`lfp.png`** — filtered-LFP proxy at **28 ring-spaced electrodes**
  (`wc_figures._ring_electrodes`, replacing the docx 5×5=25-electrode torus),
  stacked with `offsets = arange(28)·5.0`, high-pass order-4 zero-phase filter,
  cutoff 1 Hz, on the **FIXED y-span = 150.0** centered on the ladder midpoint
  (`_ladder_mid = (28−1)·5/2`), MNIST stim onset ticks, dpi=150.

This makes the metrics (`cross_cols`, `sync_spike_extent`, `n_active_channels`,
`fast_activity_duration_s`) pixel-transferable from the longitudinal calibration.

## 7. Classification (post-Colab)

The uniform pipeline (`classifier.py` + `features.py`) is applied to every run
→ one **master CSV**: taxonomy label (0–5), all metrics, all sampled
hyperparameters, config, seed. Labels 1–5 require an LFP morphological
correlate (discrete spikes via `cross_cols>6` OR `sync_spike_extent≥2`, or
rhythmic oscillation); FIS vs FSW split on synchronous-spike extent
(≤`SYNC_CHANS=4` → FIS, > → FSW); err toward Type 0 on arbitrary calls.

## 8. Interactive adjudication HTML (post-Colab)

A `taxonomy_validation`-analogue over a **randomly-selected sample of runs plus
ALL review-flagged runs** (not all 1500 — 1500 cards won't open in a browser),
one card per run (heatmap + LFP + label + metrics), with the same adjudication
controls (radio buttons, localStorage, CSV export, filters). localStorage key
bumped to a fresh namespace.
