# WC-MNIST Longitudinal Study — harness

Parameterized, reproducible, resumable harness for the paired 100-epoch
Wilson–Cowan / MNIST simulations described in
`WC_MNIST_longitudinal_study_ENGINEERING_SPEC.md`. The model dynamics are a
verbatim port of the **v82 notebook** (`7_+S_-H_v82_…ipynb`, cell 14); only RNG
plumbing and orchestration changed. See "Model fidelity" below.

> Scope: this repo runs the simulations and writes per-epoch metrics, figures,
> and checkpoints. Running the full 60-run study (Colab) and the downstream
> epileptiform classification are **out of scope** (spec §14).

---

## Files

| File | Role |
|---|---|
| `wc_config.py` | `ExperimentConfig` dataclass, `make_configs()`, frozen cell-7 constants, §10 seizure-boundary constants. |
| `wc_model.py` | Verbatim `SparseBigWilsonCowan` (cell 14), connectivity builder (cells 12–13), `train_one_image`/`train_epoch`/`evaluate` (cell 22). |
| `wc_topology.py` | Domain + Hessian loss-landscape topology (cell 16), trimmed to the §7 metrics. Lazy `gudhi` / `scipy.eigsh`. |
| `wc_figures.py` | The four per-epoch figures: `heatmap_E/I` (fixed 0-1 viridis + stim marks), `lfp` (28 ring-spaced electrodes), `lfp_focus` (§12). Lazy `matplotlib`. |
| `wc_experiment.py` | `run_experiment(seed, config, …)` — RNG, epoch loop, §7 logging, gradient probe, boundary bookkeeping, topology trigger, checkpoint/resume, figures + manifest. On a topology-off run it **archives** trigger-epoch checkpoints; `measure_topology_for_run()` reloads them for the deferred topology pass. |
| `run_study.py` | `run_study(n_sets, start_seed, …)` driver over seeds × configs. |
| `run_topology_pass.py` | `run_topology_pass(out_root, runs, …)` — deferred topology: reloads archived checkpoints and merges topology into each run's metrics (see "Deferred topology" below). |
| `regenerate_figures.py` | `regenerate_figures(out_root, runs, …)` — re-render figures for completed runs from archived checkpoints (frozen re-sim; raw E/I windows aren't saved, so figures are a reproducible frozen readout, not the original training window). No retraining. |
| `analysis_loader.py` | Concatenates per-run Parquet/manifests into `_combined/`. |
| `plot_trajectories.py` | (bE, bI) trajectory panels with boundaries (§15.6). |
| `repro_test.py` | **A1** determinism + cross-config init identity. |
| `model_diff_check.py` | **A2** ported math == notebook cell 14 / cell 7. |
| `smoke_test.py` | **A3** schema/files/figures + **A4** resume identity. |
| `timing_calibration.py` | **A5** per-epoch timing → 100-epoch × {15,60}-run projection. |
| `RUN_WC_MNIST_STUDY.ipynb` | Colab notebook — runnable cells (mount, **pull code from GitHub**, trial, timing, training pass, deferred topology pass, resume, combine) with inline instructions. The recommended way to run on Colab; Step 3 clones this repo so code isn't uploaded to Drive. |

---

## Install

Python ≥ 3.8. Core: `torch`, `torchvision`, `numpy`, `scipy`, `pandas`,
`pyarrow`, `matplotlib`. Topology (optional, Colab): `gudhi`. On Colab everything
is present except `gudhi` (`pip install gudhi` once). See `requirements.txt`.

```bash
pip install -r requirements.txt
# or: pip install torch torchvision numpy scipy pandas pyarrow matplotlib gudhi
```

MNIST is downloaded automatically to `$WC_DATA_ROOT` (default `./mnist_data`).

---

## Launch on Colab

**Getting the code there:** the recommended path is `RUN_WC_MNIST_STUDY.ipynb`, whose
Step 3 clones this repo from GitHub to `/content/wc_mnist_code` each session (read-only
PAT in Colab secret `GH_PAT`) — code is **not** uploaded to Drive; only results are.
Setup is one-time (see the notebook's Step 3; details in memory `wc-mnist-git-sync-plan`).
The snippet below is the underlying programmatic call once the code is importable:

```python
# one cell — mounts Drive, runs 5 sets (seeds 0..4) × 3 configs × 100 epochs
from google.colab import drive; drive.mount('/content/drive')
import run_study
run_study.run_study(n_sets=5, start_seed=0)        # OUT_ROOT defaults to Drive
```

or from a shell:

```bash
python run_study.py --n-sets 5                      # seeds 0..4, all 3 configs
```

Output goes to `OUT_ROOT` (`wc_config.OUT_ROOT`,
`/content/drive/MyDrive/WC_MNIST_Longitudinal`). Override with
`--out-root /some/path`.

## Resume (Colab timeouts / session loss)

Re-run the **same command**. Each `(seed, config)`:
- is **skipped** if its `DONE` sentinel exists for ≥ the requested epochs;
- otherwise **resumes** from the latest per-epoch checkpoint and continues.

Checkpoints are written every epoch (`keep_last=2`) and restore model, optimizer,
all four RNG states, the DataLoader generator, and the E/I/O/H network state, so
a resumed run continues **bit-for-bit** (verified by A4).

## Add seed ranges (real parallelism)

One GPU runs runs **sequentially**. For true parallelism, launch multiple
sessions over **disjoint** seed ranges into the same `out_root` — they write
disjoint `run_id`s and never collide:

```bash
# session 1                      # session 2
python run_study.py --n-sets 5   python run_study.py --n-sets 5 --start-seed 5
```

Run a single config or fewer epochs:

```bash
python run_study.py --n-sets 1 --configs sequential_H --epochs 100
```

After runs finish, build the combined tables and the trajectory figure:

```bash
python analysis_loader.py   --out-root <OUT_ROOT>     # _combined/all_runs_metrics.parquet
python plot_trajectories.py --out-root <OUT_ROOT>     # _combined/trajectories.png
```

---

## Deferred topology (two-phase runs)

Topology is the most expensive part of a run and you rarely want it on every
seed, so it can be **deferred**. Run the training pass with topology off:

```bash
python run_study.py --n-sets 5 --no-topology          # add --no-figures to defer figures too
```

Trigger-epoch checkpoints (epoch 0, every 4th, sign-flips) are then archived to
`checkpoints/topology/`. Measure topology later, on just the seeds you want, by
reloading them — no retraining:

```bash
python run_topology_pass.py --out-root <OUT_ROOT>                       # all runs with archives
python run_topology_pass.py --out-root <OUT_ROOT> --runs seed000__sequential_H
```

This fills the `topology_*` columns and flips `topology_measured` in each run's
`metrics.parquet`/`.csv`. Because runs are deterministic and the archived
checkpoint holds the exact weights topology would have been measured on inline,
the result matches an all-in-one run. (`RUN_WC_MNIST_STUDY.ipynb` Steps 8 / 8b
wrap the two phases.) The trigger cadence is unchanged whether topology is on or
deferred — `run_study.py` (topology on) still measures inline.

On a run's **first** topology measurement (either path), `measure_topology` prints
`domain_loss_range`/`domain_loss_std`/`hessian_eig1` at full precision and emits a
**WARNING** (it never raises) if the domain `loss_range` is *exactly* `0.0` — a
sanity check that the ±bias perturbation is actually reaching the model. A
tiny-nonzero range (~1e-4, which prints as `0.0000` at 4 dp) is normal for the
sequential chance plateau and does **not** trip it (spec §8).

---

## Output layout

```
OUT_ROOT/seed000__sequential_H/
    metrics.parquet            # per-epoch table (§7), rewritten each epoch
    metrics.csv                # mirror
    checkpoints/checkpoint_epoch_0099.pkl   # keep_last=2
    checkpoints/topology/topo_checkpoint_epoch_*.pkl  # archived at trigger epochs (topology-off runs)
    figures/epoch_0004/{heatmap_E,heatmap_I,lfp,lfp_focus}.png
    figure_manifest.csv        # (run_id, seed, config_name, epoch, fig_type, path)
    DONE                       # sentinel (json: n_epochs, device, alpha_H, …)
OUT_ROOT/_combined/            # built by analysis_loader, not the driver
    all_runs_metrics.parquet
    all_figures_manifest.csv
```

The three configs of a set: `simultaneous` (all pixels, αH=0),
`sequential_noH` (row-by-row, αH=0), `sequential_H` (row-by-row,
αH∼U(1,10)). H-current is off iff `alpha_H == 0.0`.

---

## Acceptance tests

```bash
python model_diff_check.py     # A2  math unchanged vs notebook (reads co-located v82 .ipynb; torch only)
python repro_test.py           # A1  determinism + init identity (CPU bit-exact)
python smoke_test.py           # A3 + A4  schema/figures + resume identity
python timing_calibration.py --device cuda    # A5  wall-clock projection
```

`repro_test.py` / `smoke_test.py` use a shortened `image_duration` so they finish
off-Colab in well under a minute — determinism, schema, and resume do not depend
on the amount of dynamics. The full study uses the frozen `wc_config` defaults.
On GPU, run `repro_test.py --device cuda --tol 1e-5` (sparse/scatter kernels may
not be bit-exact; the run logs `device` and `deterministic_ok`).

---

## Model fidelity (why you can trust the port)

- `SparseBigWilsonCowan.step` / `.forward` are **byte-for-byte** the cell-14
  dynamics. `model_diff_check.py` (A2) parses the notebook and asserts the
  methods are AST-identical and that one `step` is numerically bit-identical to
  an inline reference of the cell-14 equations.
- All 38 sampled-range constants match cell 7 (A2). The **one** intentional
  change is restoring `alpha_H ∼ Uniform(1,10)` for the H-on config (the
  notebook had it forced to 0); A2 checks that against the spec, not the notebook.
- RNG (spec §5): two numpy streams are spawned from `SeedSequence(seed)` —
  `init_rng` seeds **all** initialization (config-independent, so a set's three
  configs share one init, verified by A1b); `config_rng` draws `alpha_H`/`tau_H`
  only. Global libraries are seeded from the set seed and determinism flags set;
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before `torch` import.
- **Sparse operator is COO, not CSR.** `_get_sparse_matrix` builds each step's
  connectivity as a `sparse_coo_tensor`. This is a storage/dispatch choice only
  — `step`/`forward` source is unchanged, and COO is bit-identical to CSR on the
  value gradients — but it is required because CSR sparse-dense **autograd** on
  CUDA raises `expected scalar type Long but found Bool` on current torch builds.
  COO is the supported path. (Verified equivalent on CPU; A2 still passes.)

## Performance notes (GPU)

- **Training and validation are batch-1 / sequential by design** — images carry
  E/I/O/H state forward (§4), so they cannot be batched without changing the
  dynamics. This is the irreducible per-epoch cost (~minutes/epoch on an L4).
- **The clean-gradient probe and topology are batched** (`[B, N]` via
  `wc_model.simulate_window_batched`, the notebook's evaluator). These images are
  independent (each runs from a zero state), so a measurement that would be tens
  of millions of single-image sims (e.g. domain grid = 15×15 × 500 × 125 steps)
  runs as one `[500, N]` pass per grid point — orders of magnitude faster on a
  GPU, and numerically equivalent to the per-image loop (~1e-10).
- **Activity snapshots stay on-GPU**, transferred to CPU once per captured epoch
  (not per Euler step), and are captured only on figure epochs. `mean_E_activity`
  is accumulated as a GPU scalar so it is still logged every epoch.
- **Figures are written on epoch 0 + scheduled (every-4th) epochs**, not on
  sign-flip-only measurement epochs. This is a deliberate perf trade vs §12
  (those epochs still get topology + metrics, just no figures) so the activity
  window isn't captured on epochs whose measurement status isn't known until
  after training. To restore figures on every measurement epoch, capture
  unconditionally in `_train_one_epoch` (cheap now that there's no per-step sync).
- Per-epoch knobs in `wc_config.py`: `TOPO_NUM_SAMPLES`/`TOPO_GRID_SIZE` (topology
  cost), `grad_probe_images` (=64; gradient-probe cost). `TOPO_NUM_SAMPLES` (500)
  is currently **clamped to `EVAL_IMAGES`** (256) because topology samples come
  from the fixed eval subset — raising it above 256 has no effect unless
  `EVAL_IMAGES` is raised too. If per-epoch training dominates, run more parallel
  sessions over disjoint seed ranges rather than shrinking the protocol.
- Run `timing_calibration.py --device cuda` on the actual GPU first and gate the
  full launch on it (§17).

## Calibration items to resolve on the first set (spec §17)

- Sign-flip deadbands `topo_eps_loss=0.01`, `topo_eps_acc=1.0` (in
  `wc_config`); tune so triggers track real inflections, not noise.
- GPU determinism tolerance for A1.
- Per-epoch wall-clock at the full `IMAGES_PER_EPOCH=256` / `EVAL_IMAGES=256` +
  one topology measurement (run `timing_calibration.py` on the Colab GPU and gate
  the 60-run launch on it).
- `lfp_focus` hotspot definition — confirm with the taxonomy work before the big
  batch (it feeds classification).
- **`EVAL_IMAGES` is not a free knob.** The fixed eval subset feeds val metrics,
  the topology sample pool, the gradient probe, *and* the post-val state carries
  into the next epoch's training (§4), so lowering it changes training
  trajectories and topology quality — keep it fixed study-wide. Decoupling val's
  ending state from the next epoch's training is **intentionally not done**: that
  carry-over is a wanted biological analogy (no reset — the next task starts from
  wherever the system currently is). Topology getting its own sample pool
  (analysis-only, no dynamics coupling) is still fine if ever needed.
