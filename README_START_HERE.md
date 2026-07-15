# Start here (for Claude Code)

This folder stages a refactor task.

**Goal:** turn the v82 notebook into a parameterized, reproducible Python harness
for a longitudinal WC-MNIST training study, exactly as described in the spec.

## Files
- `WC_MNIST_longitudinal_study_ENGINEERING_SPEC.md` — the authoritative spec.
  Implement against it. Read it fully before writing any code.
- `7_+S_-H_v82_MNIST_Classifier_BPTT_sequential_v82.ipynb` — the source notebook
  to port the model and training/eval helpers from. **Do not change model dynamics.**

## Kickoff prompt to use
> Read `WC_MNIST_longitudinal_study_ENGINEERING_SPEC.md`. Implement the Python
> module, driver, tests, and plotting scripts exactly as specified, porting the
> model from the v82 notebook in this folder **without changing any model
> dynamics**. Start with the model port (`wc_model.py`) and the reproducibility
> test, and show me your plan before writing files.

## Scope reminders
- Claude Code writes code only. It does NOT run the 60-run study (that runs on
  Colab) and does NOT need a GPU or the multi-GB data files.
- Epileptiform classification of output figures happens later in Cowork using the
  existing `labeling_toolkit/`, via the `figure_manifest.csv` the harness writes.
- Deliverables and acceptance tests are listed in spec §15 and §16.
