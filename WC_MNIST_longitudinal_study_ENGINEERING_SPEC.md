# WC-MNIST Longitudinal Study — Engineering Handoff Spec

**Audience:** Claude Code (implementation), against the v82 notebook family in
`Loss Surface Analysis/`.
**Compiled:** 28 June 2026.
**Status:** Approved design. Implement the module + driver + tests + plotting.
Do **not** run the 60-run study (that runs on Colab) and do **not** alter model
dynamics (cell-14 math is frozen; see Acceptance Test A2).

---

## 1. Purpose

Convert the single-config, hand-driven v82 notebook into a parameterized,
reproducible, automated harness that runs **paired 100-epoch simulations** and
records per-epoch parameter evolution, performance, loss-surface topology, and
activity figures.

Scientific goal: explain **why these networks evolve toward regions of increasing
seizure susceptibility during training.** The "what" (networks drift toward and
across the seizure boundary) is established cross-sectionally in Results v22; this
study makes it **longitudinal** — one fixed training duration with per-epoch
tracking — so we can measure *when* and *why* the drift happens, when performance
plateaus/decays, and how the loss-surface geometry changes along the path.

Secondary question (lower priority): why networks become less effective once they
seize.

---

## 2. Experimental design

- A **set** = 3 paired runs sharing one seed, one run per configuration.
- **Configurations (3):**
  1. `simultaneous` — all pixels at once, H-current OFF (αH = 0).
  2. `sequential_noH` — row-by-row input, H-current OFF (αH = 0).
  3. `sequential_H` — row-by-row input, H-current ON (αH ~ Uniform(1.0, 10.0)).
- **Initial batch:** 5 sets (15 runs). **Eventual:** ~20 sets (60 runs).
- All runs: **100 epochs**, same fixed settings (§4).
- Within a set, the **initial network is identical across all three configs**
  (same weights, topology, biases, taus); the *only* differences are input mode
  and αH. This is guaranteed by the RNG design in §5, not by luck.
- Treat **seed (set) as the unit of analysis** for population statistics; the
  three configs in a set are paired, not independent.

H-current is OFF iff `alpha_H == 0.0`.

---

## 3. Module API

Deliver a Python package (not a notebook) plus a thin Colab entry cell that imports it.

### 3.1 `wc_model.py`
- Verbatim port of notebook **cell 14** (`SparseBigWilsonCowan`) and the
  train/eval helpers (`train_one_image`, `train_epoch`, `evaluate`) from cell 22.
- **No change to dynamics, defaults, or sampled ranges.** Acceptance Test A2
  diffs the math.

### 3.2 `wc_config.py`
- `@dataclass ExperimentConfig`: `name`, `input_mode ∈ {"simultaneous","sequential"}`,
  `alpha_h_mode ∈ {"off","uniform_1_10"}`, plus all fixed hyperparameters (§4).
- Factory `make_configs() -> list[ExperimentConfig]` returns the three configs above.
- Frozen numeric constants (ranges, dt, durations) live here, copied byte-for-byte
  from notebook cell 7.

### 3.3 `wc_experiment.py`
```python
def run_experiment(seed: int, config: ExperimentConfig,
                   n_epochs: int = 100, resume: bool = True,
                   out_root: str = OUT_ROOT) -> RunResult:
    """Run one (seed, config) for n_epochs. Idempotent: resumes from the
    latest checkpoint under the run's directory if resume=True."""
```
- Builds `run_id` (§9), resolves dirs, restores checkpoint if present, runs the
  epoch loop, writes per-epoch Parquet rows (§7), saves figures + manifest on
  measurement epochs (§8/§12), checkpoints every epoch (§11).

### 3.4 `run_study.py`
```python
def run_study(n_sets: int, start_seed: int = 0,
              configs=None, n_epochs: int = 100) -> None:
    """For each seed in [start_seed, start_seed+n_sets), for each config,
    call run_experiment(...). Sequential on one GPU (see §13).
    Fully resumable: re-running skips completed runs and continues partial ones."""
```
- Entry point for the whole study. A completed run is marked by a
  `run_id/DONE` sentinel written after epoch 100 + final flush.

---

## 4. Fixed settings (frozen unless noted)

| Setting | Value | Notes |
|---|---|---|
| `NUM_EPOCHS` | 100 | |
| `IMAGES_PER_EPOCH` | 256 | resampled from shuffled train loader each epoch (unchanged) |
| `EVAL_IMAGES` | 256 | **fixed held-out subset**, identical every epoch (changed from 40) |
| `BATCH_SIZE` | 64 | |
| `LEARNING_RATE` | 1e-3 | |
| `dt` | 1/500 s | |
| `IMAGE_DURATION` | 0.25 s | sequential ROW_DURATION = 0.25/28 |
| Train/val/test split | 50k / 10k / 10k | split RNG seed 123 (unchanged) |
| Validation | every epoch, **carry-over state** (E/I/O/H continuous from train→val→next train) | leave dynamics untouched |

The fixed eval subset = first 256 images of `mnist_val` under the seed-123 split
order (deterministic, identical across all runs and epochs). Used for eval
metrics **and** the per-epoch clean-gradient probe (§7).

---

## 5. RNG design (reproducibility)

Goal: bit-exact within a (seed, config), and identical initialization across the
three configs of a set.

- **Two independent streams per run, derived from the set seed:**
  - `init_rng` — seeds **all network initialization** (topology/rewiring, weights,
    bias_e/bias_i, tau vectors, H init, mnist weights, CeLoc/CeRem/CeLocI, output).
    Derived **only** from the set seed — *never* from config knobs. This guarantees
    the three configs in a set start from the identical network.
  - `config_rng` — draws αH for `sequential_H` (Uniform(1,10)); returns 0.0 for
    the OFF configs **without consuming an init draw**. Also draws `PULSED_STIMULI`
    if retained. Kept separate so toggling config never shifts the init stream.
- Seed all global libraries deterministically at run start: `np.random.seed`,
  `random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`.
- Determinism flags: `torch.backends.cudnn.deterministic=True`,
  `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True, warn_only=True)`,
  and set env `CUBLAS_WORKSPACE_CONFIG=:4096:8` **before** torch import.
- **Acceptance criterion (A1):** bit-exact loss/acc curves on **CPU** across two
  runs of the same (seed, config). On **GPU**, accept curves identical within a
  documented tolerance if sparse/scatter kernels are nondeterministic; log which
  mode (`device`, `deterministic_ok: bool`) each run used.
- The notebook's cell-7 (`rng = default_rng(seed)`) + cell-9 (global reseed)
  pattern is replaced by this explicit two-stream design invoked once inside
  `run_experiment`, eliminating cell-order fragility.

---

## 6. Configuration of αH and input mode

- `simultaneous`: `SEQUENTIAL_ROW_INPUT=False`, `alpha_H=0.0`.
- `sequential_noH`: `SEQUENTIAL_ROW_INPUT=True`, `alpha_H=0.0`.
- `sequential_H`: `SEQUENTIAL_ROW_INPUT=True`, `alpha_H=config_rng.uniform(1.0,10.0)`.
- `TAU_H` ~ Uniform(0.5, 3.0) from `config_rng` for the H-on config (unchanged
  range). For OFF configs τH is irrelevant (αH=0) but draw-and-discard to keep
  stream bookkeeping uniform, or simply don't draw — either is fine since
  `config_rng` never feeds init. Record τH and αH in the log regardless.

---

## 7. Per-epoch logging schema (Parquet)

One row per (run, epoch). One Parquet file per run: `run_id/metrics.parquet`,
appended/rewritten each epoch (also mirror to CSV for quick inspection). A
combined loader concatenates all runs for analysis.

**Identity / config**
`run_id, seed, config_name, input_mode, alpha_H, tau_H, device, deterministic_ok, epoch`

**Performance**
`train_loss, train_acc, val_loss, val_acc,`
`d_train_loss, d_train_acc, d_val_loss, d_val_acc` (epoch-to-epoch deltas; null at epoch 0)

**Bias parameters (the drift)**
`bias_e, bias_i`

**Clean gradients** (one extra forward+backward on the fixed eval batch at current params)
`grad_bias_e, grad_bias_i,`
`grad_norm_CeLoc, grad_norm_CeRem, grad_norm_CeLocI,`
`grad_angle_to_boundary_deg` (angle between (grad_bias_e, grad_bias_i) and the
config's seizure-boundary normal; the core "why it drifts toward seizure" metric)

**Connectivity / heterogeneity summaries** (per learnable group: mean, std, L2 norm)
`CeLoc_mean, CeLoc_std, CeLoc_l2,`
`CeRem_mean, CeRem_std, CeRem_l2,`
`CeLocI_mean, CeLocI_std, CeLocI_l2,`
`inh_to_exc_gain_mean, inh_to_exc_gain_std,`
`mnist_weights_mean, mnist_weights_std,`
`tauPy_eff_mean, tauPy_eff_std, tauInh_eff_mean, tauInh_eff_std`
(plus `d_CeLoc_l2, d_CeRem_l2` deltas — local vs remote growth-rate comparison is
a primary analysis target: does training strengthen long-range E→E faster than local?)

**Seizure-boundary bookkeeping** (constants in §10)
`boundary_slope, boundary_intercept, signed_dist_to_boundary` (positive = seizure-prone
side), `on_seizure_side` (bool), `crossed_this_epoch` (bool)

**Topology** (null on non-measurement epochs)
`topology_measured` (bool), `trigger_reason ∈ {"scheduled","sign_flip","epoch0"}`,
`domain_n_saddle, domain_avg_persistence, domain_max_persistence,`
`domain_loss_min, domain_loss_max, domain_loss_range, domain_loss_std,`
`hessian_n_saddle, hessian_avg_persistence, hessian_max_persistence,`
`hessian_eig1, hessian_eig2`

**Activity / seizure markers**
`H_final_mean, H_final_std` (and any cheap seizure proxy already computed)

---

## 8. Topology trigger logic

Measure topology (both domain + Hessian, §10) when **any** of:
1. `epoch == 0`, or
2. `(epoch+1) % 4 == 0` (scheduled cadence), or
3. **sign-flip trigger:** the smoothed epoch-to-epoch delta of `train_loss`
   **or** `train_acc` changes sign vs the previous epoch.

Sign-flip details:
- Smooth deltas over a trailing 2-epoch window before testing sign.
- **Deadband:** ignore flips where `|delta| < eps` (`eps` per metric, calibrated
  on the first set; start with `eps_loss = 0.01`, `eps_acc = 1.0` pct-pt).
- **Min gap:** at least 2 epochs since the last topology measurement, so a noisy
  stretch can't trigger every epoch.
- Record `trigger_reason`. The dense `transition_epoch` every-epoch window from
  v79 is **OFF** (replaced by this trigger).

Figures (§12) are saved on the **same** measurement epochs.

**First-measurement diagnostic (built into the topology recorder).** On a run's
first topology measurement only (avoid per-measurement console spam), print
`loss_range`, `loss_std`, and `eig1` in **full precision (scientific notation)**
for each config, then apply a **hard-zero tripwire**: if any config's domain
`loss_range` is exactly `0.0` (not merely small), emit a **WARNING** (do not
raise — a genuinely silent network can legitimately produce a constant loss) that
says the ±perturbation may not be applied to `bias_e`/`bias_i` or the fixed eval
batch may be returning a constant, and points to the spot check below. Tiny-
but-nonzero values (~1e-4 domain `loss_range` for the sequential chance plateau,
matching the original v82 run) are normal and print as `0.0000` at 4 decimals —
not a bug. Spot check when the warning fires: set `bias_e` to two very different
values and confirm the loss differs in sequential mode. These fields are already
persisted in Parquet (§7); this adds live visibility + a tripwire only, no schema
change.

---

## 9. Run IDs and file layout

```
OUT_ROOT = /content/drive/MyDrive/WC_MNIST_Longitudinal
  run_id = f"seed{seed:03d}__{config_name}"     # e.g. seed000__sequential_H
  {OUT_ROOT}/{run_id}/
      metrics.parquet            # per-epoch table (§7), rewritten each epoch
      metrics.csv                # mirror for quick inspection
      checkpoints/
          checkpoint_epoch_0099.pkl   # keep_last=2 (resume only)
      figures/
          epoch_0004/  heatmap_E.png  heatmap_I.png  lfp.png  lfp_focus.png
          epoch_0008/  ...
      figure_manifest.csv        # (run_id, seed, config, epoch, fig_type, path)
      DONE                       # sentinel written after epoch 100 + final flush
  {OUT_ROOT}/_combined/
      all_runs_metrics.parquet   # built by analysis loader, not the driver
      all_figures_manifest.csv
```

---

## 10. Seizure-boundary constants (current, FINAL labels 28 Jun 2026)

Logistic regression (`C=1e4`) on `bias_e_mean`, `bias_i_mean`, seizure =
`Sz_Txnmy==5`, expressed as `bI = m·bE + c`. Applied at **analysis time** and for
the per-epoch `signed_dist_to_boundary` field; simulations do not depend on them.

| Config | slope m | intercept c | fit acc |
|---|---|---|---|
| simultaneous | 2.24 | −0.78 | 94.2% |
| sequential_noH | 2.10 | −2.45 | 98.6% |
| sequential_H (aggregate) | 2.19 | −2.81 | 98.4% |

**sequential_H αH-stratified** (use the stratum matching each run's αH):

| αH stratum | slope m | intercept c | fit acc |
|---|---|---|---|
| 0–3 (low) | 2.29 | −2.52 | 100% |
| 3–6 (med) | 2.10 | −2.22 | 100% |
| 6–10 (high) | 2.06 | −3.37 | 98.2% |

Signed distance uses the boundary normal `(m, −1)/√(m²+1)`; sign convention:
positive = seizure-prone side (above/right of the line in (bE, bI)).

The per-epoch `signed_dist_to_boundary` field and `get_boundary()` follow the
αH-stratified strata above. **Exception:** the trajectory figure
(`plot_trajectories.py`) draws a single fixed reference boundary per config for
readability, so `sequential_H` uses the **aggregate** fit (2.19, −2.81) there
rather than the strata. The drawn line is therefore independent of per-run αH and
does not exactly correspond to the stratified `signed_dist_to_boundary` values.

---

## 11. Checkpoint contract (resume)

Reuse the v81 full-state checkpoint, **slimmed**: it carries only what resume
needs, not the accumulated heavy figure arrays (those go to disk per epoch, §12).

Each checkpoint contains: `epoch`, `model_state_dict`, `optimizer_state_dict`,
all four RNG states (`torch_cpu`, `torch_cuda`, `numpy`, `python`), the
DataLoader generator state, network state (`E,I,O,H` tensors + `global_train_time_s`),
the two topology trackers (to_dict), `seed`, `config_name`, `version`.

- Checkpoint **every epoch** (cheap once heavy arrays are externalized);
  `keep_last=2`.
- Per-epoch metrics are flushed to `metrics.parquet` **before** the checkpoint so
  a crash never leaves the log ahead of the resumable state.
- On resume: restore model/optimizer/RNG/generator/network state/trackers, set
  `start_epoch = checkpoint['epoch']+1`, continue. Verify resumed `run_id` matches.

---

## 12. Figures + manifest

On each measurement epoch (§8), save and index:
- `heatmap_E.png`, `heatmap_I.png` — the notebook's "raster-style" spatiotemporal
  heatmaps of E/I over the last training window: **fixed `vmin=0, vmax=1`** viridis
  (E/I are sigmoid rates in (0,1), so the scale is comparable across epochs), with a
  colorbar and MNIST stimulus-onset hashmarks. `heatmap_E` is the seizure-annotation
  view consumed by `labeling_toolkit` (kind='heatmap').
- `lfp.png` — filtered LFP proxy (weighted E/I + local/remote synaptic drive,
  zero-phase high-pass, stim marks) at **28 evenly-spaced electrodes on the 1-D
  recurrent ring**. **Deviation from the notebook:** the notebook sampled a 5×5
  torus grid (`_select_torus_electrodes`), but the WC connectivity is a directed
  *ring* lattice (`generate_ring_lattice_directed`; unit `i` ↔ `i±1…±k/2 mod N`),
  so ring distance on the linear index — not the 28×28 pixel grid — is what governs
  locality. Ring-spaced electrodes sample that metric uniformly; grid electrodes
  landed in 5 clusters and labeled units by their unrelated MNIST-pixel `(row,col)`.
- `lfp_focus.png` — **NEW focus-targeted virtual-electrode LFP** (heatmap-detected
  hotspot, unfiltered + highpass), addressing the known sparse-electrode blind spot
  (existing notebook TODO). Add this now since figures feed classification. The
  hotspot is a **contiguous ring window** (peak unit ± 4 on the ring, 9 units),
  consistent with the ring-spaced `lfp.png` electrodes — not the earlier 3×3
  pixel-grid patch, which mixed ring-adjacent and ring-distant units.

There is **no** dot-scatter "raster" figure: the notebook never had one (its
"raster" was the E heatmap above), a per-unit spike raster misrepresents the WC
rate model, and `labeling_toolkit` consumes only `heatmap`/`lfp`.

Each figure adds a row to `figure_manifest.csv`:
`run_id, seed, config_name, epoch, fig_type, path`. Raw E/I time-series arrays are
retained **only for measurement epochs** (not all 100) to bound storage.

The manifest is the interface to the later epileptiform-classification step
(§14) — its schema must stay stable.

---

## 13. Execution model / parallelism

- One Colab GPU → runs execute **sequentially** inside `run_study`. Do not attempt
  intra-session multi-run parallelism on a single GPU.
- Real parallelism = multiple Colab sessions/notebooks each running `run_study`
  over a disjoint seed range, or a multi-GPU/local host (driver would then fan out
  one run per device). Design `run_study` so disjoint seed ranges write to
  disjoint `run_id`s and never collide.
- Robust to Colab timeout / session loss via per-epoch checkpoint + idempotent
  resume (§11).

---

## 14. Out of scope for Claude Code

- **Running** the 5 (then ~20) sets — done on Colab by the user.
- **Epileptiform classification** of the saved figures — done later in Cowork
  using the existing `labeling_toolkit/`, consuming `figure_manifest.csv`.
- **Final analysis plots** beyond the plotting scripts in §15.

---

## 15. Deliverables

1. `wc_model.py`, `wc_config.py`, `wc_experiment.py`, `run_study.py` (§3).
2. `repro_test.py` — Acceptance Test A1 (determinism).
3. `model_diff_check.py` — Acceptance Test A2 (cell-14 math unchanged).
4. `smoke_test.py` — 2-epoch run of all three configs at seed 0; asserts schema,
   files, manifest, resume.
5. `timing_calibration.py` — times one epoch (train+eval+one topology measurement)
   on the target GPU and extrapolates to 100 epochs × N runs (§16).
6. `plot_trajectories.py` — builds the figure-1-style (bE, bI) panels with
   **figure-2-style epoch-by-epoch paths overlaid**, colored by epoch and/or
   accuracy, with the §10 boundaries drawn; reads `_combined` Parquet.
7. `analysis_loader.py` — concatenates per-run Parquet into `_combined`.
8. `README.md` — how to launch on Colab, resume, and add seed ranges.

---

## 16. Acceptance tests

- **A1 Determinism:** same (seed, config) twice → identical `metrics.parquet`
  loss/acc columns (bit-exact on CPU; documented tolerance on GPU). Also verify
  the three configs of a set share identical epoch-0 model parameters (init RNG).
- **A2 Model unchanged:** automated diff of the ported `SparseBigWilsonCowan.step`/
  `forward` and sampled-range constants against notebook cell 14 / cell 7 — must be
  semantically identical (whitespace/format only).
- **A3 Smoke:** 2-epoch ×3-config run completes; Parquet schema (§7) present;
  figures + manifest written on epoch-0; checkpoint resumes to identical state.
- **A4 Resume:** kill mid-run, re-launch, confirm it continues from last epoch
  with identical continuation (compare against an uninterrupted reference).
- **A5 Timing:** `timing_calibration.py` prints projected wall-clock for
  100 epochs × 15 and × 60 runs; gate the full launch on this.

---

## 17. Open / calibration items (resolve on the first set)

- Sign-flip deadband `eps_loss`, `eps_acc` (start 0.01 / 1.0 pct-pt; tune so
  triggers track real inflections, not noise).
- GPU determinism tolerance (A1) — confirm whether full bit-exactness is
  achievable on the Colab GPU or a tolerance is required.
- Per-epoch wall-clock at IMAGES_PER_EPOCH=256 / EVAL_IMAGES=256 + one topology
  measurement (A5) — confirm the 60-run budget is acceptable before scaling.
- Confirm `lfp_focus` hotspot-detection definition with the taxonomy work before
  the big batch (it feeds classification).
- **Epoch-0 topology sanity check** — now implemented in the topology recorder
  (§8, "First-measurement diagnostic"): a full-precision print + hard-zero
  WARNING tripwire. Calibration action: on the first set, eyeball the printed
  values and confirm the sequential configs are tiny-nonzero (not a hard `0.0`);
  if the warning fires, run the bias-perturbation spot check before scaling.
  Context: simultaneous clears the 4-decimal rounding threshold because it drives
  all 784 neurons; sequential_H ≈ sequential_noH at epoch 0 is expected since H≈0
  at init and the two share initialization.

---

## 18. Notes for analysis (not implementation, for context)

The headline "why" metric is `grad_angle_to_boundary_deg`: if the loss gradient
in (bE, bI) consistently points toward the seizure-prone normal across epochs and
configs, that *is* the mechanistic statement that gradient descent on the task
drives networks toward seizure susceptibility. Pair it with `signed_dist_to_boundary`
(the trajectory), the local-vs-remote E→E growth comparison (`d_CeRem_l2` vs
`d_CeLoc_l2`), and the loss-surface topology at each inflection to assemble the
explanation.
