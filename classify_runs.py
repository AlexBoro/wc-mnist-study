"""classify_runs.py — classify every param-explore run into a master CSV.

Walks the param-explore output tree (OUT_ROOT/<config>/run_<seed>/), and for each
run with a figure pair (heatmap_E.png + lfp.png) runs the CANONICAL v82-derived
extractor + decision table:

    labeling_toolkit.features.extract_features(heatmap, lfp, mean_E, val_acc)
    labeling_toolkit.classifier.classify(feats)

producing ONE flat master CSV row per run: identity (config, seed, run_id) +
every extracted feature + the taxonomy label, confidence, firing rule, and a
`review` flag (label == -1). The system stays: one canonical spec, one feature
extractor, one flat decision table; marginal calls route to review, and the none
label is the conservative default.

mean_E / val_acc: this is a NO-TRAINING sweep, so there is no val_acc, and mean_E
of the E trajectory lives in arrays.npz (not needed for figure-based screening).
Both are passed as None; the extractor handles their absence — the energy-collapse
ictal corroboration arm simply does not fire, which is correct for this study.

Resumable: an existing master row for a run is kept unless --force. CSV is
rewritten atomically after each batch so a mid-run interrupt loses nothing.

Usage:
    python classify_runs.py --out-root /path/to/wc_mnist_param_explore \\
                            --toolkit /path/to/labeling_toolkit \\
                            --csv master_classification.csv
"""
from __future__ import annotations

import os
import sys
import csv
import json
import argparse
from pathlib import Path


CONFIGS = ["simultaneous", "sequential_noH", "sequential_H"]

# Columns emitted, in order. Identity first, then features, then verdict.
ID_COLS = ["run_id", "config_name", "seed", "input_mode", "alpha_H", "tau_H"]
FEAT_COLS = [
    "ink_frac", "max_col_frac", "band_score", "n_complexes",
    "max_IDI", "min_IDI", "median_IDI", "sustained_fraction",
    "vertical_spread_fraction",
    "S", "cross_cols", "n_active_channels", "fast_activity_duration_s",
    "sync_spike_extent",
]
VERDICT_COLS = ["label", "label_name", "confidence", "rule", "clause", "review"]
ALL_COLS = ID_COLS + FEAT_COLS + VERDICT_COLS


def _load_toolkit(toolkit_dir):
    """Import the canonical features + classifier from the toolkit directory."""
    toolkit_dir = str(Path(toolkit_dir).resolve())
    if toolkit_dir not in sys.path:
        sys.path.insert(0, toolkit_dir)
    import features as F      # noqa: E402
    import classifier as C    # noqa: E402
    return F, C


def _iter_runs(out_root):
    """Yield (config_name, seed, run_dir) for every run dir present, sorted."""
    out_root = Path(out_root)
    for cname in CONFIGS:
        cdir = out_root / cname
        if not cdir.is_dir():
            continue
        for rd in sorted(cdir.glob("run_*")):
            try:
                seed = int(rd.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            yield cname, seed, rd


def _read_identity(run_dir, cname, seed):
    """Pull identity fields from params.json (falls back to dir name)."""
    ident = dict(run_id=f"seed{seed:03d}__{cname}", config_name=cname, seed=seed,
                 input_mode="", alpha_H="", tau_H="")
    pj = run_dir / "params.json"
    if pj.exists():
        try:
            p = json.load(open(pj))
            for k in ("input_mode", "alpha_H", "tau_H"):
                if k in p:
                    ident[k] = p[k]
        except Exception:
            pass
    return ident


def _find_fig(run_dir, name):
    """Locate a figure PNG for a run: figures are written under
    run_<seed>/figures/ by the Colab bundle writer, but early mock trees put
    them directly in run_<seed>/. Check both."""
    for cand in (run_dir / "figures" / name, run_dir / name):
        if cand.exists():
            return cand
    return run_dir / name  # non-existent; caller checks .exists()


def classify_run(run_dir, cname, seed, F, C):
    """Extract + classify one run. Returns a full master row dict, or None if
    the figure pair is missing."""
    hm = _find_fig(run_dir, "heatmap_E.png")
    lfp = _find_fig(run_dir, "lfp.png")
    if not (hm.exists() and lfp.exists()):
        return None
    feats = F.extract_features(str(hm), str(lfp), mean_E=None, val_acc=None)
    verdict = C.classify(feats)
    row = _read_identity(run_dir, cname, seed)
    for k in FEAT_COLS:
        row[k] = feats.get(k, "")
    row["label"] = verdict["label"]
    row["label_name"] = verdict["label_name"]
    row["confidence"] = verdict["confidence"]
    row["rule"] = verdict["rule"]
    row["clause"] = verdict["clause"]
    row["review"] = 1 if verdict["label"] == -1 else 0
    return row


def _write_csv(csv_path, rows_by_id):
    """Atomically write all rows, ordered by (config, seed)."""
    ordered = sorted(rows_by_id.values(),
                     key=lambda r: (CONFIGS.index(r["config_name"])
                                    if r["config_name"] in CONFIGS else 99,
                                    int(r["seed"])))
    tmp = str(csv_path) + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALL_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(ordered)
    os.replace(tmp, csv_path)


def run(out_root, toolkit_dir, csv_path, force=False, flush_every=25, verbose=True):
    F, C = _load_toolkit(toolkit_dir)
    csv_path = Path(csv_path)

    # resume: load existing rows keyed by run_id
    rows = {}
    if csv_path.exists() and not force:
        for r in csv.DictReader(open(csv_path)):
            rows[r["run_id"]] = r
        if verbose:
            print(f"resuming: {len(rows)} rows already classified")

    all_runs = list(_iter_runs(out_root))
    n_total = len(all_runs)
    done = skipped = missing = 0
    for i, (cname, seed, rd) in enumerate(all_runs, 1):
        rid = f"seed{seed:03d}__{cname}"
        if rid in rows and not force:
            skipped += 1
            continue
        row = classify_run(rd, cname, seed, F, C)
        if row is None:
            missing += 1
            continue
        rows[rid] = row
        done += 1
        if done % flush_every == 0:
            _write_csv(csv_path, rows)
            if verbose:
                print(f"[{i}/{n_total}] classified={done} skipped={skipped} "
                      f"missing_figs={missing}", flush=True)

    _write_csv(csv_path, rows)
    # label histogram
    from collections import Counter
    hist = Counter(int(r["label"]) for r in rows.values())
    if verbose:
        print(f"DONE: {len(rows)} rows -> {csv_path.name} "
              f"(new={done}, skipped={skipped}, missing_figs={missing})")
        print("label histogram:", dict(sorted(hist.items())))
    return dict(total=len(rows), new=done, skipped=skipped,
                missing_figs=missing, hist=dict(hist))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True,
                    help="param-explore output tree (…/wc_mnist_param_explore)")
    ap.add_argument("--toolkit", required=True,
                    help="path to labeling_toolkit/ (features.py + classifier.py)")
    ap.add_argument("--csv", default="master_classification.csv")
    ap.add_argument("--force", action="store_true", help="re-classify all runs")
    args = ap.parse_args()
    run(args.out_root, args.toolkit, args.csv, force=args.force)
