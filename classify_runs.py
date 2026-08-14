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
ID_COLS = ["run_id", "config_name", "seed", "input_mode", "alpha_H", "tau_H",
           "fingerprint", "code_tag"]
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
# Render/measurement provenance, recorded per row.
#
# `*_render` is what the bundle's manifest says was in force when lfp.png was
# DRAWN; `*_measure` is what this process used when the ANALYTIC features were
# measured from arrays.npz. They are separate columns because they are separate
# events -- a config edit between the sweep and a re-classification makes one
# feature row half-measured under each window, and a single column could not
# show that. `derived_verdict` is the comparison (see _derived_verdict).
DERIVED_COLS = ["lfp_span_render", "lead_in_s_render",
                "lead_in_s_measure", "derived_verdict"]
# Array-valued features: which channel-count index to export, and it must match
# the threshold the RULES test at, or the sheet would show a number the verdict
# was not based on. Sourced from the classifier so the two cannot drift.
ARRAY_FEAT_AT = {
    "involve_contig_dur_s": None,   # filled from classifier.TH at load
    "fast_contig_dur_s": None,
}

ALL_COLS = ID_COLS + FEAT_COLS + VERDICT_COLS + DERIVED_COLS
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
    ALL_COLS = [c for c in ID_COLS + FEAT_COLS + VERDICT_COLS + DERIVED_COLS
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
    """Pull identity fields from params.json (falls back to dir name).

    `fingerprint` and `code_tag` are carried through deliberately: run_id is
    seed<NNN>__<config> and therefore NOT unique across parameter-exploration
    sweeps, so the labels CSV must record WHICH sweep it describes. The
    adjudication tooling reads these columns to refuse cross-corpus joins.
    """
    ident = dict(run_id=f"seed{seed:03d}__{cname}", config_name=cname, seed=seed,
                 input_mode="", alpha_H="", tau_H="", fingerprint="", code_tag="")
    pj = run_dir / "params.json"
    if pj.exists():
        try:
            p = json.load(open(pj))
            for k in ("input_mode", "alpha_H", "tau_H", "fingerprint", "code_tag"):
                if k in p:
                    ident[k] = p[k]
        except Exception:
            pass
    return ident


def _corpus_str(fingerprint, code_tag):
    """Canonical corpus identity string (delegates to apply_overrides)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from apply_overrides import corpus_id
    return corpus_id(code_tag, fingerprint)


def _corpus_verdict(a, b):
    """Delegate to apply_overrides -- ONE authority for corpus comparison.

    Do not reimplement: the comparison is subtle (a matching model fingerprint
    is NOT a corpus match, because the fingerprint excludes tau_H's sampling
    range and both corpora therefore carry a3dee49abc12). Two copies of that
    logic would drift.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from apply_overrides import corpus_verdict
    return corpus_verdict(a, b)


def _rows_corpus(rows):
    """Corpus identity shared by a list of CSV dict rows, or "" if unknown.
    Raises if the rows disagree -- such a file already mixes corpora."""
    seen = {_corpus_str(r.get("fingerprint"), r.get("code_tag")) for r in rows}
    seen.discard("")
    if len(seen) > 1:
        raise SystemExit(f"labels file mixes corpora: {sorted(seen)[:4]}")
    return seen.pop() if seen else ""


def _tree_corpus(out_root):
    """Corpus identity of a bundle tree, read from the runs' params.json.
    Samples up to 50 runs; raises if they disagree."""
    seen = set()
    for i, (cname, seed, rd) in enumerate(_iter_runs(out_root)):
        pj = Path(rd) / "params.json"
        if pj.exists():
            try:
                p = json.load(open(pj))
                c = _corpus_str(p.get("fingerprint"), p.get("code_tag"))
                if c:
                    seen.add(c)
            except Exception:
                pass
        if i >= 49:
            break
    if len(seen) > 1:
        raise SystemExit(f"bundle tree mixes corpora: {sorted(seen)[:4]}")
    return seen.pop() if seen else ""


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


def _bundle_derived(run_dir):
    """The bundle's recorded render-time derived_config ({} if absent).

    Reads manifest.json directly rather than importing param_persist: this
    module is run against a toolkit path and must not require the project's
    sweep modules to be importable. Delegating would be cleaner but couples the
    classifier to param_persist for a two-line read.
    """
    man = Path(run_dir) / "manifest.json"
    try:
        with open(man) as f:
            m = json.load(f)
    except Exception:
        return {}
    dc = m.get("derived_config")
    return dict(dc) if isinstance(dc, dict) else {}


DERIVED_MATCH = "match"
DERIVED_CONFLICT = "conflict"
DERIVED_UNKNOWN = "unknown"

# Absolute tolerance for comparing recorded floats. These are config constants
# written and re-read as JSON/CSV, not measurements, so this only absorbs
# float-repr round-tripping -- it is not a "close enough" allowance.
_DERIVED_TOL = 1e-9


def _derived_verdict(recorded, current):
    """Three-state comparison of render-time vs measure-time derived config.

    Returns DERIVED_MATCH / DERIVED_CONFLICT / DERIVED_UNKNOWN.

    Three states rather than a boolean, for the same reason the corpus gate has
    three: a bundle written before ``derived_config`` existed carries no record,
    and "no record" is NOT agreement. A boolean would force that case to be
    either a false conflict (blocking every legacy bundle) or a false match
    (silently accepting a window mismatch, which is the failure this whole
    change exists to prevent). Callers gate on ``== DERIVED_MATCH``, so UNKNOWN
    is refused alongside CONFLICT unless explicitly allowed.

    Only ``lead_in_s`` is compared. ``lfp_span`` cannot be: it is a RENDER-time
    quantity with no measure-time counterpart -- this process reads the stored
    PNG rather than redrawing it, so there is no current value to disagree with.
    It is recorded for audit, and a span change is caught by re-rendering, not
    here.
    """
    if not recorded:
        return DERIVED_UNKNOWN
    rec = recorded.get("lead_in_s")
    if rec is None:
        return DERIVED_UNKNOWN
    try:
        if abs(float(rec) - float(current)) <= _DERIVED_TOL:
            return DERIVED_MATCH
    except (TypeError, ValueError):
        return DERIVED_UNKNOWN
    return DERIVED_CONFLICT


def _measure_lead_in():
    """The lead-in this process will use for ANALYTIC features.

    Read from the same cfg object ``_analytic_feats`` hands to
    ``AF.extract``, which resolves ``cfg.LEAD_IN_S`` at call time -- so this
    reports what is actually in force, not what the config file says on disk.
    Returns None when the analytic path is unavailable (no arrays.npz anywhere,
    older sweeps): there is then no measure-time window to compare.
    """
    cfg = _ANALYTIC.get("cfg")
    if cfg is None:
        return None
    return float(getattr(cfg, "LEAD_IN_S", 0.0))


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

    # Derived-config provenance. Recorded whether or not it agrees -- the row
    # carries the evidence and `run` decides what to do about it, so a refusal
    # is auditable rather than a bare count.
    rec = _bundle_derived(run_dir)
    meas = _measure_lead_in()
    row["lfp_span_render"] = rec.get("lfp_span", "")
    row["lead_in_s_render"] = rec.get("lead_in_s", "")
    row["lead_in_s_measure"] = "" if meas is None else meas
    row["derived_verdict"] = (DERIVED_UNKNOWN if meas is None
                              else _derived_verdict(rec, meas))
    return row


def _write_csv(csv_path, rows_by_id):
    """Atomically write all rows, ordered by (config, seed)."""
    ordered = sorted(rows_by_id.values(),
                     key=lambda r: (CONFIGS.index(r["config_name"])
                                    if r["config_name"] in CONFIGS else 99,
                                    int(r["seed"])))
    tmp = str(csv_path) + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALL_COLS, extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(ordered)
    os.replace(tmp, csv_path)


def run(out_root, toolkit_dir, csv_path, force=False, flush_every=25, verbose=True,
        overrides=None, allow_config_drift=False):
    F, C = _load_toolkit(toolkit_dir)
    csv_path = Path(csv_path)

    # Resume: load existing rows keyed by run_id.
    #
    # PROVENANCE GUARD. run_id is seed<NNN>__<config>, which is identical across
    # different parameter-exploration sweeps. So an existing CSV from ANOTHER
    # sweep would resume "successfully": every run_id matches, every row is
    # skipped, and the output silently carries the OTHER corpus's labels. Refuse
    # to resume unless the existing file's corpus matches the bundle tree's.
    rows = {}
    if csv_path.exists() and not force:
        existing = list(csv.DictReader(open(csv_path)))
        tree_corpus = _tree_corpus(out_root)
        file_corpus = _rows_corpus(existing)
        from_apply = _corpus_verdict(tree_corpus, file_corpus)
        if from_apply == "conflict":
            raise SystemExit(
                f"REFUSING to resume: {csv_path} describes corpus "
                f"{file_corpus!r} but the bundle tree at {out_root} is "
                f"{tree_corpus!r}. run_id collides across sweeps, so resuming "
                f"would keep the other corpus's labels. Use --force to "
                f"re-classify from scratch, or write to a different --csv.")
        elif from_apply != "match":
            print(f"WARNING cannot CONFIRM that {csv_path} belongs to the bundle "
                  f"tree at {out_root}: file={file_corpus or '<none>'} "
                  f"tree={tree_corpus or '<none>'}. A matching model fingerprint "
                  f"is not sufficient (it excludes tau_H's sampling range, so "
                  f"different sweeps share one) -- only a code_tag on both sides "
                  f"confirms a match. Resuming anyway, since resume only skips "
                  f"rows it can re-derive; pass --force to reclassify from "
                  f"scratch if this file might be from another sweep.")
        for r in existing:
            rows[r["run_id"]] = r
        if verbose:
            print(f"resuming: {len(rows)} rows already classified"
                  + (f" (corpus {file_corpus})" if file_corpus else ""))

    all_runs = list(_iter_runs(out_root))
    n_total = len(all_runs)
    done = skipped = missing = 0
    # Rows whose render-time and measure-time derived config disagree (or cannot
    # be confirmed to agree). COUNTED, not silently dropped: a bare skip would
    # make a partly-measured corpus look like a smaller clean one, which is the
    # same failure mode as dropping unscorable runs from a rate.
    drift = {}
    for i, (cname, seed, rd) in enumerate(all_runs, 1):
        rid = f"seed{seed:03d}__{cname}"
        if rid in rows and not force:
            skipped += 1
            continue
        row = classify_run(rd, cname, seed, F, C)
        if row is None:
            missing += 1
            continue
        # DERIVED-CONFIG GATE. The features in this row are measured from two
        # sources under two windows: pixel features from the stored lfp.png
        # (drawn at sweep time) and analytic features from arrays.npz (measured
        # now). If the lead-in differs between those moments the row mixes them,
        # and the effect is not cosmetic -- amp_prominence moves 1.482 -> 1.091
        # on a reference bundle between 0.0 s and 1.0 s, straddling
        # TH['SPIKE_AMP_MIN'] = 1.0. Refuse rather than record a row whose label
        # no single configuration would have produced. Gate on == MATCH so
        # UNKNOWN is refused too; --allow-config-drift downgrades to a warning.
        dv = row.get("derived_verdict")
        if dv != DERIVED_MATCH and not allow_config_drift:
            drift.setdefault(dv or DERIVED_UNKNOWN, []).append(rid)
            continue
        rows[rid] = row
        done += 1
        if done % flush_every == 0:
            _write_csv(csv_path, rows)
            if verbose:
                print(f"[{i}/{n_total}] classified={done} skipped={skipped} "
                      f"missing_figs={missing}", flush=True)

    # Refusal report BEFORE the write, so a run that refused everything cannot
    # leave a fresh CSV that looks like a successful classification.
    if drift:
        n_ref = sum(len(v) for v in drift.values())
        print(f"REFUSED {n_ref} run(s) on derived-config drift "
              f"(measure-time lead_in_s = {_measure_lead_in()!r}):")
        for state, ids in sorted(drift.items()):
            ex = ", ".join(ids[:4]) + (" ..." if len(ids) > 4 else "")
            print(f"  {state}: {len(ids)} -- {ex}")
        print("  'conflict' = the bundle was RENDERED under a different lead_in_s "
              "than this process measures with; its pixel and analytic features "
              "would not share a window.\n"
              "  'unknown'  = the bundle predates derived_config in the manifest, "
              "so agreement cannot be CONFIRMED (absence of a record is not a "
              "match).\n"
              "  Fix by restoring the config the bundle was rendered under, or "
              "re-render the bundles under the current config. Pass "
              "--allow-config-drift to record these rows anyway (the per-row "
              "*_render / *_measure / derived_verdict columns keep the evidence).")
        if not rows:
            raise SystemExit(
                "REFUSING to write: every run was refused on derived-config "
                f"drift ({n_ref} run(s)); no rows would be classified. Leaving "
                f"{csv_path} untouched.")

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
    ap.add_argument("--allow-config-drift", action="store_true",
                    help="record rows whose render-time and measure-time derived "
                         "config disagree (or cannot be confirmed to agree) "
                         "instead of refusing them. The per-row *_render / "
                         "*_measure / derived_verdict columns are written either "
                         "way; this only decides whether such rows are KEPT.")
    args = ap.parse_args()
    run(args.out_root, args.toolkit, args.csv, force=args.force,
        overrides=args.overrides, allow_config_drift=args.allow_config_drift)
