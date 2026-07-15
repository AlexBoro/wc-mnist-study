# CLAUDE.md — WC-MNIST longitudinal study harness

Context for Claude Code. Authoritative design = `WC_MNIST_longitudinal_study_ENGINEERING_SPEC.md`.
Runner notebook = `RUN_WC_MNIST_STUDY.ipynb` (Colab). Do NOT change model dynamics
(acceptance test A2 enforces this).

## Code sync — GitHub, not Drive (as of 14 Jul 2026)

This folder is a **git repo** pushed to `github.com/AlexBoro/wc-mnist-study` (private).
The notebook's **Step 3** clones / `reset --hard`s the repo to `/content/wc_mnist_code`
each session (read-only PAT in Colab secret `GH_PAT`), so **code no longer lives on
Drive** — only results do (`MyDrive/WC_MNIST_Longitudinal/`). Per-change loop: edit +
commit locally → push (GitHub Desktop "Push origin") → any Colab tab re-runs Step 3 to
pick it up. The v82 `.ipynb` (A2's reference) is tracked in the repo, so it clones
automatically — no separate upload. Pin `REF` to a tag before a batch so parallel
sessions run identical code. Verified end-to-end on Colab 14 Jul 2026. See memory
`wc-mnist-git-sync-plan`.

## Pre-run review (2 Jul 2026) — status before committing to Steps 7 & 8

Verified present and working:
- Step 5 gate runs `repro_test.py` (A1: determinism + identical cross-config
  init) and `smoke_test.py` (A3: full §7 Parquet schema, epoch-0 figures +
  manifest, checkpoint, DONE; A4: interrupt/resume matches bit-for-bit).
- Step 6 = 2-epoch 1-set trial; Step 7 times with `--no-topology` (matches Step 8).
- Determinism flags in place (`use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`).
- `model_diff_check.py` (A2) confirmed working: AST equivalence of
  `step()`/`forward()` vs notebook cell 14 PASSES, and its 38 sampled-range
  constants PASS. Only sub-check 3 (one numeric step) needs torch, present on Colab.

## Fixes applied (2 Jul 2026, via Claude Code) — both DONE

1. **A2 re-enabled in Step 5.** `RUN_WC_MNIST_STUDY.ipynb` Step 5 cell now runs
   `model_diff_check.py` (line uncommented); the Step 5 markdown is relabeled
   **mandatory** and spells out that A2 needs the v82 `.ipynb`
   (`7_+S_-H_v82_..._v82.ipynb`) present in `CODE_DIR`, and cell-0's prerequisites
   now list it. (**Resolved 14 Jul 2026:** the v82 `.ipynb` is tracked in the git
   repo, so it clones into `CODE_DIR` automatically — no manual Drive upload; A2
   finds it co-located.)

2. **Epoch-0 topology tripwire implemented.** `wc_topology.measure_topology` gained
   a `first_measurement` param; on the first measurement it prints
   `domain_loss_range`/`domain_loss_std`/`hessian_eig1` in full precision (`!r`)
   and emits a WARNING (never raises) if `domain_loss_range` is exactly `0.0`.
   Both callers in `wc_experiment.py` pass it: inline path
   `first_measurement=(trigger_state["last_topo_epoch"] < 0)`, Step 8b path
   `first_measurement=(i == 0)`. Compiles clean; no schema change (fields already
   in Parquet §7). Tiny-nonzero (~1e-4, prints 0.0000 at 4 dp) does NOT trip it.

Note: ENGINEERING_SPEC.md §8/§17 (edited earlier in Cowork) describe this tripwire;
code now matches the spec.

## Figure rework (13 Jul 2026, via Claude Code)

Reworked the per-epoch figures in `wc_figures.py` (+ call site in
`wc_experiment.py`); all deviations documented in spec §12. Model dynamics
untouched (A2 unaffected; smoke A3/A4 pass). New `.py` count: **14**.

- **`lfp.png`** — the refactor had reduced it to raw mean-E; restored to the
  notebook's weighted LFP proxy (`10·E + 0.15·loc + 0.05·rem + 25·I`, order-4
  zero-phase high-pass, electrode labels, stim marks). Then **switched electrode
  sampling from the 5×5 pixel grid to 28 evenly-spaced electrodes on the 1-D
  recurrent ring** — WC connectivity is a ring lattice
  (`generate_ring_lattice_directed`), so ring distance (linear index) governs
  locality, not the 28×28 MNIST-pixel grid. `_select_torus_electrodes` →
  `_ring_electrodes`.
- **`heatmap_E/I`** — auto-normalized `hot` → fixed `vmin=0,vmax=1` viridis +
  colorbar + stim marks. This is the notebook's "raster-style heatmap" and the
  seizure-annotation view `labeling_toolkit` consumes (kind='heatmap').
- **`raster.png` removed** — the dot-scatter (E>0.5) had no notebook antecedent,
  misrepresents the rate model, rendered blank. `FIG_TYPES` is now 4; the
  manifest no longer emits a `raster` row.
- **`lfp_focus.png`** — `detect_hotspot` patch switched from a 3×3 pixel-grid
  neighborhood to a **contiguous ring window** (peak ± 4, 9 units, wraparound),
  so the focus electrode is a real ring arc consistent with `save_lfp`. mean-E
  readout (not the weighted proxy) is unchanged — it's a detection tool by design.
- **`wc_experiment.py`** figure block reconstructs CeLoc/CeRem as SciPy CSR and
  passes `I_window` + stim params to the figures.
- **`plot_trajectories.py`** — single fixed reference boundary per config
  (sequential_H uses the aggregate fit, not the αH strata) + shared x/y limits.

New **`regenerate_figures.py`**: re-renders figures for completed runs from the
archived measurement-epoch checkpoints. Raw E/I windows aren't saved, so it
re-simulates a frozen window (driven by the fixed eval subset) — a reproducible
per-epoch readout, NOT bit-identical to the original training window. Notebook
**Step 11** runs it; default window = 10 s.

**Rollout status.** Code sync is now via git (see "Code sync" above): the reworked
`wc_figures.py` + notebook are committed and on GitHub, so Colab gets them by re-running
Step 3 — no manual `.py` upload. STILL PENDING: runs that *already completed* on Drive
with the old figure code need **Step 11** (`regenerate_figures`) on Colab to re-render
with the new figures; that hasn't been run yet. During testing, one run
(`seed000__sequential_noH`) was regenerated in the *local download snapshot* only
(and with the pre-ring grid electrodes) — not the authoritative Drive results.

## Notes
- Two-phase workflow (train with topology deferred → Step 8b measures topology
  from archived checkpoints) is an intentional change from the spec's inline
  topology, to cut training cost. Fine to keep.
- The epoch-0 output showing `0.0000` for the sequential configs and small
  nonzero (`loss_range=0.0027`, `eig1=0.0218`) for simultaneous is EXPECTED
  (chance plateau; sequential drives only 28/784 neurons; simultaneous clears the
  4-dp rounding threshold). sequential_H ≈ sequential_noH at epoch 0 is expected
  (H≈0 at init; shared init).
