# WC-MNIST session handoff — 2026-08-13

Written at a pause point, immediately before the 500×3 parameter exploration.
Read this first on resume; it supersedes `SESSION_HANDOFF_2026-08-01.md` on every
point where the two disagree.

---

## 1. Where we are

The blocking defect of the 2026-08-01 session — near-silent runs on the
**non-seizing** side of the boundary being labelled as generalized epileptiform
discharge — is **closed**. It had two independent causes, both in measurement
rather than in the model, and both are now fixed in code:

1. **The probe initial condition.** `E0 ~ U(0, IC_E_MAX)` puts most of the
   population above threshold, so sample 0 is relaxation *from the initial
   condition*, not a response to the first stimulus. Measured on the safe-side
   runs: 760/784 units fire in the opening transient, then activity drops
   25-fold. This opening burst was what the classifier was scoring.
2. **The high-pass filter's step response.** `wc_figures.save_lfp` applies a
   zero-phase 4th-order 1 Hz high-pass, and the step-like opening makes it ring.
   Measured step response: **0.62 s to fall to 10% of peak, 0.71 s to 5%,
   1.63 s to 1%.**

The fix is a **measurement window**, `cfg.LEAD_IN_S = 1.0`, excluded from feature
extraction only — never from simulation and never from `arrays.npz`.

Nothing about the model, and no classifier threshold, was changed.

---

## 2. Retractions from this session

Two conclusions were reported to the user and then reversed. Both are recorded
here so neither is inherited as fact.

**Retracted #1: "the epileptiform gate survives the lead-in fix."** That check
merged only the freshly recomputed field-proxy and band features onto each run's
existing label row. The composite extractor runs several feature groups, so the
untouched groups were still carrying opening-transient values and propping the
gate up. Running the real `extract_features` → `classify` chain showed the gate
falls too.

**Retracted #2: "a `lead_in` parameter on `render_run` is sufficient."** It is
not. There are **two independent consumers** of the trajectories:

| path | input | features |
|---|---|---|
| `param_render.render_run` | arrays in memory | pixel features (from `heatmap_E.png`, `lfp.png`) |
| `labeling_toolkit/analytic_features.lfp_from_arrays` | `arrays.npz` on disk | analytic features |

`classify_run` overlays the analytic keys **on top of** the pixel ones, and
`classifier.SPIKE_AMP_FEATURE = "amp_prominence"` is analytic. Editing only
`render_run` would have cleaned the figures while leaving the gate measuring the
opening transient. Both paths now trim.

---

## 3. Rejected alternatives (do not redo these)

**Do not trim the stored bundle.** `param_persist` computes an `expected_T` row
count and its `deep_check` rejects any bundle whose `E_traj` row count differs.
Trimmed bundles would be judged incomplete and re-simulated indefinitely; it
would also stale the stored summary statistics and discard data irreversibly.
Store the full run; exclude at measurement time.

**Do not crop already-rendered images.** Cropping shifts the axes-box detection
in the pixel feature extractor, so margin beside the colorbar is admitted as
plot interior. Slice the arrays, then render.

**Do not switch the LFP high-pass to a padded `filtfilt`.** Tested directly:
padding does **not** shorten the ring (0.622 s either way), because the ring is
the filter's genuine step response, not an initialisation artefact. Switching
would shift amplitudes enough to flip two of the eight test runs across the
1.0 threshold while buying nothing.

**Filter ordering does not matter.** Slice-then-filter vs filter-then-slice agree
to 0.028 spacing units on all 8 runs — no label changes, no peak-channel changes.
The simpler slice-then-filter ordering is what is implemented.

---

## 4. Why 1.0 s and not 0.5 s

Labels are **identical** at 0.5 s and 1.0 s (`amp_prominence` differs by
≤ 0.007). The difference is the ring:

- At 0.5 s, a residue of 0.16–0.27 spacing units survives into the retained
  window, and on **seed 1104 in both sequential configs** the labelling peak
  falls at t = 0.88 s — *inside* the 0.62 s ringing window. Those amplitudes
  would be partly filter artefact.
- At 1.0 s, cause (1) is cleared outright, cause (2) is below ~5% of peak, and
  no run's labelling peak falls inside the ringing window.

Cost: 10% of a 10 s run.

---

## 5. Current label split on the safe-side probe (8 runs, production path)

Full chain, `LEAD_IN_S = 1.0`, `SPIKE_AMP_FEATURE = "amp_prominence"`,
`TH["SPIKE_AMP_MIN"] = 1.0`:

| config | seed | amp_prominence | label |
|---|---|---|---|
| sequential_noH | 1101 | 1.114 | FIS (1) |
| sequential_noH | 1102 | 0.616 | none (0) |
| sequential_noH | 1103 | 1.098 | FIS (1) |
| sequential_noH | 1104 | 0.791 | none (0) |
| sequential_H | 1101 | 1.025 | FIS (1) |
| sequential_H | 1102 | 0.608 | none (0) |
| sequential_H | 1103 | 1.094 | FIS (1) |
| sequential_H | 1104 | 0.785 | none (0) |

**4 FIS / 4 none.** No GSW — both arms of the generalized rule now fail on all 8,
which was the point of the exercise.

Read this table with two caveats. The four FIS runs sit at 1.025–1.114 against a
threshold of 1.0 — **marginal, and clustered on the same two seeds across both
configs**, which suggests seed-specific structure rather than a config effect.
And threshold semantics belong to the user: changing `SPIKE_AMP_MIN`
reinterprets every previously labelled run, so **do not tune it on n = 8.** The
500×3 sweep will supply an amplitude distribution; decide then.

---

## 6. Corpus comparability — read before comparing labels

Two distinct problems, one per corpus — verified by reading the headers, not from
memory:

- **`taxonomy_validation_100_fixed.csv`** (24 columns) has **no
  `amp_prominence` column at all**, and no `spike_amp_max` either. It predates
  the analytic features entirely, so its labels rest on a different amplitude
  measure than any new run will.
- **`safeside_labels.csv`** (43 columns) *does* carry `amp_prominence` — but the
  values were measured with **no lead-in** (`LEAD_IN_S` did not exist). Its
  numbers are therefore inflated by exactly the transient this session removed:
  the file's first row reads 8.178, and the safe-side runs re-measured through
  the production path now read 0.61–1.11. The rows in §5 above supersede it for
  these 8 runs.

Old and new labels are **not directly comparable** in either case. The user has
accepted this: the 500×3 run is a full re-run, so provenance of the old corpora
is not a blocker. Do not pool them with new output.

Also still true from 2026-08-01: `H_INIT_MAX` was raised 0.01 → 0.15, which
**invalidates bundles generated at 0.01 as a like-for-like set.**

---

## 7. Code changes committed this session

| file | change |
|---|---|
| `wc_config.py` | new `LEAD_IN_S = 1.0` with the full measurement rationale; `H_INIT_MIN, H_INIT_MAX = 0.0, 0.15` (from the prior session, previously uncommitted) |
| `param_render.py` | `lead_in=None` kwarg on `render_run`, `render_from_result`, `render_from_bundle`; guarded slice before `save_lfp` so the filter runs on the trimmed signal |
| `labeling_toolkit/analytic_features.py` | `lead_in=None` on `lfp_from_arrays` and `extract`; same guarded slice on the `arrays.npz` load path |
| `RUN_PARAM_EXPLORE.ipynb` | stale knob comment corrected (E→E scale is `U(-0.6, 3.0)` for both CeLoc and CeRem; notes the lead-in) |
| `.gitignore` | ignore `safeside/` (125 MB probe bundle tree, regenerable) |

Design notes:

- `LEAD_IN_S` is resolved at **call time** (`float(cfg.LEAD_IN_S if lead_in is
  None else lead_in)`), following the `LFP_SPAN` convention, so a notebook can
  override it with no re-import. Verified: setting `cfg.LEAD_IN_S = 0.0`
  changes run 1104's `amp_prominence` 0.791 → 1.277 through the production
  `classify_runs._analytic_feats` path.
- Both slices are **guarded** (`if 0 < k < len(t) - 1`) so a lead-in longer than
  the run renders untrimmed rather than emptying the arrays.
- `getattr(cfg, "LEAD_IN_S", 0.0)` in the toolkit keeps older callers working.
- `analytic_features.py` exists in **one** copy — not vendored — and `cfg` is
  passed in by `classify_runs`, so the constant propagates.
- Pass `lead_in=0.0` anywhere to reproduce the untrimmed measure.

All 27 unit tests pass (`test_bias_override.py`, `test_connectivity_invariants.py`).

---

## 8. State of the 500×3 run

Ready. Nothing outstanding in code.

**The sweep is 500 seeds × 3 configurations = 1500 runs, with parameters drawn
from uniform distributions WITHOUT regard to where the seizure boundary is
thought to lie** (user's specification, 2026-08-13). Every range is already what
`wc_config.py` does — verified, no edits needed:

| quantity | range | source |
|---|---|---|
| `bias_e` | `U(-8, 0)` | `BIAS_E_INIT_MIN/MAX` |
| `bias_i` | `U(-12, 0)` | `BIAS_I_INIT_MIN/MAX` |
| `H0` (per unit) | `U(0, 0.15)` | `H_INIT_MIN/MAX` |
| configs | simultaneous, sequential_noH, sequential_H | `cfg.make_configs()` |

`bias_e_override` / `bias_i_override` default to `None`, so biases are drawn, not
prescribed.

**Notebook knobs for this run:**

```
N_SEEDS = 500
ARMS    = None      # NOT cfg.PILOT_COEFF_ARMS
```

`ARMS` is unrelated to biases — it splits each run across three values of the
E→E coefficient (`w075`/`w100`/`w150`), which would make this 4500 runs. `None`
uses the study default `W_CELOC = W_CEREM = 0.10`. At 1500 runs: ≈ 1.5–2 h,
≈ 7 GB.

**`handoff/design_500x3.csv` is NOT this sweep — do not use it.** It prescribes
bias pairs selected to sit past the boundary (353 of 500 seeds, 71%), which is
the opposite of drawing without regard to the boundary. Same for
`build_boundary_map.py`, which merges that design on `["config","seed"]` and
plots the DESIGN coordinates: with random draws, plot each run's realized
`bias_e`/`bias_i` from the master log instead (`_collect_params` records them
from `model.bias_e`). Plotting design coordinates would also mask a failed
override — the point lands where it was meant to regardless.

**Expect ~31% of seeds in the seizure region** (~465 of 1500). That is the
correct behaviour for an unbiased sample, and it is exactly why the 2026-08-01
six-seed smoke test looked dead. A low seizure count is not a defect — check the
bias draws against the fitted boundary before concluding anything is broken.

**Other notes:**

- Run `N_SEEDS = 20` and the four §7 notebook checks before raising to 500:
  completeness, LFP span calibration, regime occupancy, flat-drive.
- E→E init scale: `CELOC_INIT_SCALE = CEREM_INIT_SCALE = U(-0.6, 3.0)`
- `cfg.LFP_SPAN` is provisional — recalibrate from the pilot in §7 check 2,
  which is what `render_run(return_stats=True)` exists for. Changing it does not
  change the model fingerprint.

The first decision after the pilot is the amplitude threshold, using the sweep's
`amp_prominence` distribution. It is the user's call.

---

## 9. Figures produced this session

Diagnostic, saved as artifacts in the Claude Science project, not in the repo:

- `filter_order_test.png` / `.csv` — filter step response; both orderings per
  run; padded-filter comparison
- `fis_run_lfp.png` — the FIS run's field proxy with the labelling transient
  annotated
- `leadin_relabel.csv` — per-run labels, rules and analytic amplitudes, all 8 runs
