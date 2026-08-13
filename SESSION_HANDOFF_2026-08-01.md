# WC-MNIST session handoff — 2026-08-01

Written at a pause point. Read this first on resume; it supersedes any earlier
summary in the conversation.

---

## 1. Where we actually are (one paragraph)

We are **not** running the 500×3 sweep yet, and **nothing has been pushed**. The
session went looking for why the current code's `sequential` runs looked dead,
spent most of a day on hypotheses that were all built on a sampling artifact,
and then — prompted by the user asking "can we reproduce the 6/28 figure?" —
discovered the real, much narrower defect: **the current model is all-or-none
where the June reference was graded.** That is the open problem.

---

## 2. The correction that matters most

**Retracted:** "current-code sequential is quiescent / broken."

The smoke tests used seeds 0–5. Five of the six drew bias values **outside the
seizure region entirely**:

| seed | bias_e | bias_i | in seizure region? |
|---|---|---|---|
| 0 | -4.62 | -4.22 | no |
| 1 | -7.23 | -2.25 | no |
| 2 | -3.76 | -2.47 | no |
| 3 | -6.09 | -7.43 | no |
| 5 | -7.01 | -5.22 | no |
| 4 | -1.70 | -10.78 | yes |

Only ~31% of seeds land in-region, so P(0 hits in 6) ≈ 0.11. Ordinary bad luck.
Every conclusion drawn from those runs was diagnosing quiescent parameter space,
not a broken model.

**Method rule for the future: before concluding a regime is broken, check the
bias draws of the seeds you tested.** `handoff/inregion.py` does this.

---

## 3. The real finding — loss of dynamic range

Running seeds that *do* land in-region (n=18, `sequential_noH`), current code
produces unambiguous seizures. But the transition is a **step, not a ramp**:

- every seed with `bias_e > -1.19` → saturated (`mean_E` 0.94–0.99, LFP high ~98% of run)
- every seed with `bias_e < -1.54` → flat (`mean_E` ≤ 0.0072, `lfp_max` ≤ 0.018)
- **nothing in between**

Seizure rate by `bias_e` band, June reference vs current:

| bias_e band | June (n=1000) | current (n=18) |
|---|---|---|
| [-8, -4) | 4.8% | 0% |
| [-4, -3) | 22.4% | (no seeds) |
| [-3, -2) | 29.2% | 0% |
| [-2, -1.5) | 27.3% | 0% |
| [-1.5, -1) | 51.1% | 100% |
| [-1, -0.5) | 64.1% | (no seeds) |
| [-0.5, 0) | 57.7% | 100% |

June ramps smoothly and produces seizures even at `bias_e = -6`. That gradient
is what makes the 6/28 figure look like a *boundary* — a probabilistic
transition with scatter on both sides — instead of a step.

**Corollary that has not been checked yet:** June's sequential panel has 198 FIS
(type 1) vs 112 seizures. Current code produces essentially no intermediate
states, so **the FIS class would likely come out near-empty.** That is a
falsifiable prediction and a good cheap test on resume.

Leading suspicion: the widened `(0.0, 5.0)` E→E scale pushes the network past
criticality the moment it activates. Unverified.

---

## 4. Hypotheses tested and closed (do not redo these)

| hypothesis | verdict | evidence |
|---|---|---|
| Input path differs from June | **NO** | `compute_mnist_input` masks to the row using same weights/gain; no compensation anywhere. Per-step sequential drive is 1/28 of simultaneous, but summed over 28 rows equals it **exactly** (544.14 both) — drive is time-distributed, not lost. |
| Stimulus constants differ | **NO** | `MNIST_INPUT_SCALE`=3.0, `IMAGE_DURATION`=0.25, `ROW_DURATION`=IMAGE_DURATION/28, `DT`=1/500 — all byte-identical to June. Sampled bias/gain/weight ranges identical. |
| Row scheduling is wrong | **NO** | all 28 rows visited correctly. |
| E→E strength too weak | **NO** | current `CeLoc` mean 2.505 vs June 0.644 (~4×) and still quiescent at those seeds. |
| Weight-distribution tail / CeLocI | **NO** | see below. |

### The CeLocI test — valid method, worthless result

Reproduced June's exact explore-cell connectivity (three independent full WS
graphs; `CeLoc` and `CeRem` both at `p_rewire_py_py`, so 312 E→E in-edges/unit
vs current 156), with June's declared ranges and coefficients, swapped into the
current simulator. **Reproduction validated** against the logged xlsx:

| | reproduced | logged June |
|---|---|---|
| CeLoc mean | 0.647 | 0.644 |
| CeRem mean | 0.594 | 0.590 |
| CeLocI mean | 0.426 | 0.429 |

Result: no change (sequential median `mean_E` 0.00181 vs 0.00188 vs 0.00208).
**But this ran on the same quiescent seeds**, so it never actually tested the
hypothesis. The construction reproduction is trustworthy and reusable; the
dynamical conclusion is not. Rerun on in-region seeds if the question resurfaces.

---

## 5. Reference facts pinned this session

**June (v63) declared scale ranges** — dated 1/10/26 in the notebook, *not* the
commented-out `(0.01, 1.5)` predecessor:

```
CELOC_INIT_SCALE  = -0.3, 1.5
CEREM_INIT_SCALE  = -0.4, 1.5
CELOCI_INIT_SCALE = -0.7, 1.5
```

**Coefficients:** June `Py2Py = 10.0*E + 0.15*(CeLoc@E) + 0.05*(CeRem@E)`;
current uses `W_CELOC = W_CEREM = 0.10`.
`Py2Inh = 0.1*(CeLocI@E) + 15.0*E` is **identical** in both.

**Connectivity:** June's explore cell built CeLoc and CeRem as two *separate*
full WS graphs at the same p (312 E→E in-edges/unit). Current
`build_connectivity_csr` splits *one* ring lattice via a Bernoulli mask (156).

**6/28 figure provenance:** `Figure1_type_and_boundary_FINAL_6_28_26.png` is
backed by `WC MNIST Paper/WC_MNIST_parameter_explore_FINAL.xlsx` sheet `Data`,
1500 rows × 71 cols (`SEQUENTIAL_ROW_INPUT` = 1.0 for 1000, 0.0 for 500). Label
column `Sz_Txnmy`. Logged rates: sequential seizure 20.7%, any epileptiform
59.9%; simultaneous seizure 31.6%. Sequential boundary ≈
`bias_e > -0.2 + 0.375*bias_i`.

---

## 6. Uncommitted state — IMPORTANT

Repo `~/dev/wc-mnist-study`, branch `main`, HEAD `b3932d5`,
**in sync with origin (0 ahead / 0 behind)**. Three files have real
(non-whitespace) changes, uncommitted:

1. **`wc_config.py`** — E→E scale widened `(-0.6, 3.0)` → `(0.0, 5.0)`, both
   ranges held equal, with an explanatory comment block. Note the lower bound
   is now 0.0, so **no E→E edge is negative** — the ~17% negative fraction
   present in both June and the `(-0.6, 3.0)` ensemble is gone. Deliberate
   change of ensemble, not just strength.
2. **`test_connectivity_invariants.py`** — `test_init_scale_ranges_straddle_zero`
   rewritten as `test_init_scale_ranges_sign_contract` (pins the new
   non-negative E→E ranges as intentional; keeps straddle-zero alive for the
   untouched inhibitory pathway). Full suite passes.
3. **`RUN_PARAM_EXPLORE.ipynb`** — `N_SEEDS` 10 → 20;
   `OUT_ROOT` → `/content/drive/MyDrive/wc_mnist_param_explore_v3_scale0_5`.

**The whole worktree also shows as dirty from CRLF line-ending noise** (10072
insertions / 10072 deletions; `git diff --ignore-all-space` shows zero). Use
`--ignore-all-space` when inspecting, and edit with byte-preserving I/O so line
endings survive.

⚠️ **Given the section-3 finding, the `(0.0, 5.0)` change is now suspect** — it
is the leading candidate for *causing* the all-or-none behaviour. Do not push it
as-is without deciding that question first.

---

## 7. Suggested first move on resume

The honest test of the user's question, rather than another mechanism guess:
run ~60 seeds sampled across the **full** bias range under (a) current config
and (b) June's ranges + coefficients, classify with the v82 detector, and put
the result next to the 6/28 panel. ~40 s/run ⇒ roughly 80 minutes. This either
reproduces the graded boundary or localises the loss to the config change.

Cheap prerequisite worth doing first: check whether current code produces **any**
FIS-like intermediate runs at all (section 3 corollary).

---

## 8. Artifacts and files

Saved artifacts (in the project store):
- `sequential_reframe.png` — the two-panel reframe figure (in-region seeds vs 6/28 boundary; seizure rate vs bias_e)
- `inregion_seed_scan.csv` — 18 in-region seeds, bias draws + activity metrics
- `celoci_hypothesis_test.csv` — per-seed results for both June variants
- `regime_smoke_test.csv` (v2) — old-vs-new regime comparison (⚠ quiescent seeds)
- `wc_probe_harness.tar.gz` — the probe scripts below

Probe scripts (were in the **ephemeral** workspace `handoff/`; now bundled into
`wc_probe_harness.tar.gz` — untar to recover):
- `inregion.py` — runs given seeds, reports bias draws + LFP/activity metrics. **The most reusable one.**
- `june_repro.py` — reproduces June's three-graph connectivity, validated against the xlsx
- `june_swap.py` — monkeypatches `M.build_connectivity_csr` to run June connectivity in the current simulator
- `regime_probe.py`, `seq_scan.py` + assorted `*.json` result files

**Kernel gotcha:** `wc_model.DEFAULT_INIT_RANGES` is frozen at import time, so
mutating `wc_config` afterwards is a silent no-op. Set config values *before*
importing `wc_model` (see the top of `june_swap.py`).
