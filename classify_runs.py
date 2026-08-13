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
    # rev-2 taxonomy (2026-08-11): the features the current RULES actually
    # decide on.  The block above is kept so older sheets still parse.
    "involve_contig_dur_s", "n_fpfa_sequences", "fpfa_best_dur_s",
    "fpfa_best_ratio", "fast_dur_s", "fast_contig_dur_s",
    "fast_ibi_ratio", "spike_amp_max", "burst_band_dens", "burst_band_frac",
    "n_fast_channels",
    # analytic (arrays.npz) LFP measures — see labeling_toolkit/analytic_features.py
    "amp_prominence", "amp_excursion", "width_ms",
    "n_chans_prominence_ge1", "n_chans_excursion_ge1", "n_sharp_transients",
]
VERDICT_COLS = ["label", "label_name", "confidence", "rule", "clause", "review"]
# Array-valued features: which channel-count index to export, and it must match
# the threshold the RULES test at, or the sheet would show a number the verdict
# was not based on. Sourced from the classifier so the two cannot drift.
ARRAY_FEAT_AT = {
    "involve_contig_dur_s": None,   # filled from classifier.TH at load
    "fast_contig_dur_s": None,
}

ALL_COLS = ID_COLS + FEAT_COLS + VERDICT_COLS
ALL_COLS = [c for c in ALL_COLS if c not in ARRAY_FEAT_AT] + [
    f"{k}_at{v}" for k, v in ARRAY_FEAT_AT.items() if v]


def _load_toolkit(toolkit_dir):
    """Import the canonical features + classifier from the toolkit directory."""
    toolkit_dir = str(Path(toolkit_dir).resolve())
    if toolkit_dir not in sys.path:
        sys.path.insert(0, toolkit_dir)
    import features as F      # noqa: E402
    import classifier as C    # noqa: E402
    # Bind the array-feature export indices to the thresholds the RULES test at,
    # so the exported column can never report a channel count the verdict was
    # not based on.
    TH = getattr(C, "TH", {})
    ARRAY_FEAT_AT["involve_contig_dur_s"] = TH.get("SEIZ_CONTIG_CHANS")
    ARRAY_FEAT_AT["fast_contig_dur_s"] = TH.get("SEIZ_CONTIG_CHANS")
    global ALL_COLS
    ALL_COLS = [c for c in ID_COLS + FEAT_COLS + VERDICT_COLS
                if c not in ARRAY_FEAT_AT] + [
        f"{k}_at{v}" for k, v in ARRAY_FEAT_AT.items() if v]
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


_ANALYTIC = {"mod": None, "recon": None, "cfg": None, "figs": None, "ok": None}


def _analytic_feats(run_dir):
    """Analytic (array-measured) LFP features for one run, or {} if unavailable.

    Imported lazily and cached: the extractor needs the project modules
    (param_render for connectivity reconstruction, wc_config, wc_figures), which
    the toolkit itself does not depend on. Missing arrays.npz is NOT an error --
    sweeps predating the extractor fall back to the pixel measure.
    """
    from pathlib import Path as _P
    if not (_P(run_dir) / "arrays.npz").exists():
        return {}
    if _ANALYTIC["ok"] is None:
        try:
            # _load_toolkit already put --toolkit on sys.path, so the toolkit
            # module is importable flat. The PROJECT modules it needs live beside
            # this script, which PYTHONSAFEPATH keeps off sys.path -- add it.
            import sys as _sys
            from pathlib import Path as _PP
            _here = str(_PP(__file__).resolve().parent)
            if _here not in _sys.path:
                _sys.path.insert(0, _here)
            import analytic_features as AF
            import param_render as PR
            import wc_config as _cfg
            import wc_figures as _WF
            _ANALYTIC.update(mod=AF, recon=PR._reconstruct_connectivity,
                             cfg=_cfg, figs=_WF, ok=True)
        except Exception as e:
            print(f"  [analytic features unavailable: {e}]")
            _ANALYTIC["ok"] = False
    if not _ANALYTIC["ok"]:
        return {}
    return _ANALYTIC["mod"].extract(
        run_dir, _ANALYTIC["recon"], _ANALYTIC["cfg"], _ANALYTIC["figs"])


def classify_run(run_dir, cname, seed, F, C):
    """Extract + classify one run. Returns a full master row dict, or None if
    the figure pair is missing."""
    hm = _find_fig(run_dir, "heatmap_E.png")
    lfp = _find_fig(run_dir, "lfp.png")
    if not (hm.exists() and lfp.exists()):
        return None
    feats = F.extract_features(str(hm), str(lfp), mean_E=None, val_acc=None)
    # Analytic LFP features, measured from arrays.npz rather than the rendered
    # PNG. These OVERLAY the pixel features: the pixel keys stay in the row (so
    # the two measures can be compared on the sheet) but the classifier reads
    # whichever key classifier.SPIKE_AMP_FEATURE names. A run without arrays.npz
    # (older sweeps) simply keeps the pixel measure.
    feats.update(_analytic_feats(run_dir))
    verdict = C.classify(feats)
    row = _read_identity(run_dir, cname, seed)
    for k in FEAT_COLS:
        v = feats.get(k, "")
        # Some features are per-channel-count ARRAYS (e.g. involve_contig_dur_s
        # is indexed by "how many contiguous channels"). Writing the whole list
        # into a CSV cell makes the sheet unreadable and unsortable, so export
        # the value at the threshold the rules actually decide on, and name the
        # column accordingly. The full array stays available in the feature dict.
        if isinstance(v, (list, tuple)):
            k_at = ARRAY_FEAT_AT.get(k)
            if k_at is None or not v:
                continue
            row[f"{k}_at{k_at}"] = (round(float(v[k_at - 1]), 4)
                                    if 1 <= k_at <= len(v) else "")
            continue
        row[k] = v
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


def run(out_root, toolkit_dir, csv_path, force=False, flush_every=25, verbose=True,
        overrides=None):
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

    # --- human-adjudication overrides: stamped as the final pass, after every
    # auto label is written. Off by default -> a bare run gives the pristine
    # auto-only master (the number to cite for raw detector agreement). With
    # --overrides, the verified-label master used for downstream analysis.
    ov_summary = None
    if overrides:
        try:
            # apply_overrides.py sits next to this file at the repo root; make the
            # import robust to the caller's cwd (the script is often run from a
            # different directory so toolkit relative-imports resolve).
            import sys as _sys, os as _os
            _here = _os.path.dirname(_os.path.abspath(__file__))
            if _here not in _sys.path:
                _sys.path.insert(0, _here)
            import apply_overrides as _AO
            ov_summary = _AO.apply(str(csv_path), overrides, verbose=verbose)
        except Exception as e:  # never let an override problem corrupt the auto master
            print(f"WARNING: override pass failed ({e!r}); auto master is intact, "
                  f"overrides NOT applied")

    # label histogram (reflects overrides if applied)
    from collections import Counter
    final_rows = {r["run_id"]: r for r in csv.DictReader(open(csv_path))}
    hist = Counter(int(r["label"]) for r in final_rows.values())
    if verbose:
        print(f"DONE: {len(rows)} rows -> {csv_path.name} "
              f"(new={done}, skipped={skipped}, missing_figs={missing})")
        print("label histogram:", dict(sorted(hist.items())))
    return dict(total=len(rows), new=done, skipped=skipped,
                missing_figs=missing, hist=dict(hist), overrides=ov_summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True,
                    help="param-explore output tree (…/wc_mnist_param_explore)")
    ap.add_argument("--toolkit", required=True,
                    help="path to labeling_toolkit/ (features.py + classifier.py)")
    ap.add_argument("--csv", default="master_classification.csv")
    ap.add_argument("--force", action="store_true", help="re-classify all runs")
    ap.add_argument("--overrides", default=None,
                    help="human_overrides.csv ledger; when given, human verdicts "
                         "are stamped onto the master as a final idempotent pass "
                         "(see apply_overrides.py). Omit for the pristine auto-only master.")
    args = ap.parse_args()
    run(args.out_root, args.toolkit, args.csv, force=args.force,
        overrides=args.overrides)
