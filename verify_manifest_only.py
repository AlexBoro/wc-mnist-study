"""Corpus-wide derived-config verification WITHOUT feature extraction.

Answers one question over all 1500 bundles: would the gate in classify_runs
admit this bundle? It does so by calling the gate's OWN functions
(_bundle_derived / _measure_lead_in / _derived_verdict) rather than
reimplementing the comparison -- a reimplementation could agree with my
reading of the rule while disagreeing with the shipped code, which is exactly
the thing under test.

Cost: reads manifest.json per run. It never touches arrays.npz, so this is
seconds, not the ~1 h that `classify_runs --force` costs (that re-extracts
analytic features for every run).

Exit non-zero on any bundle the gate would refuse, or on a short count.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "code")
sys.path.insert(0, "code/labeling_toolkit")

import classify_runs as CR  # noqa: E402

ROOT = Path("/vol/bundles")
EXPECTED = 1500

mans = sorted(ROOT.glob("*/run_*/manifest.json"))
print(f"manifests found: {len(mans)}")

# _measure_lead_in reads a LAZILY-primed cache: classify_run calls
# _analytic_feats (which imports wc_config and populates _ANALYTIC["cfg"])
# BEFORE computing the verdict, so in the real pipeline the config is always in
# force by then. A verifier that skips that priming gets cfg=None and reads
# every bundle as UNKNOWN -- an artifact of the verifier, not a corpus fault.
# Prime it exactly the way the pipeline does: one real bundle with arrays.npz.
_probe = next((m.parent for m in mans if (m.parent / "arrays.npz").exists()), None)
if _probe is None:
    print("FATAL: no bundle with arrays.npz — cannot prime the analytic path")
    sys.exit(2)
CR._analytic_feats(_probe)
current = CR._measure_lead_in()
print(f"primed analytic path from: {_probe}")
print(f"measure-time lead_in_s (live config): {current}")
print(f"gate tolerance: {CR._DERIVED_TOL}")
if current is None:
    print("FATAL: analytic path unavailable even after priming — every verdict "
          "would be UNKNOWN for want of a measure-time value, which says nothing "
          "about the corpus. Refusing to report a vacuous pass.")
    sys.exit(2)

verdicts = Counter()
spans = Counter()
leads = Counter()
sources = Counter()
by_config = Counter()
offenders = []
rows = []

for m in mans:
    run_dir = m.parent
    rec = CR._bundle_derived(run_dir)
    v = CR._derived_verdict(rec, current)
    verdicts[v] += 1
    spans[rec.get("lfp_span")] += 1
    leads[rec.get("lead_in_s")] += 1
    sources[rec.get("source")] += 1
    d = json.loads(m.read_text())
    cn = d.get("config_name", "?")
    by_config[(cn, v)] += 1
    rows.append(dict(
        run_id=d.get("run_id", ""), config_name=cn,
        git_commit=str(d.get("git_commit", ""))[:7],
        lead_in_s_render=rec.get("lead_in_s", ""),
        lead_in_s_measure=current,
        lfp_span_render=rec.get("lfp_span", ""),
        source=rec.get("source", ""),
        derived_verdict=v,
    ))
    if v != CR.DERIVED_MATCH:
        offenders.append((str(run_dir), v, rec.get("lead_in_s")))

print(f"verdicts: {dict(verdicts)}")
print(f"lead_in_s recorded: {dict(leads)}")
print(f"lfp_span recorded: {dict(spans)}")
print(f"source: {dict(sources)}")
print("per-config verdicts:")
for (cn, v), n in sorted(by_config.items()):
    print(f"  {cn:18s} {v:9s} {n}")

with open("out/manifest_verify.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)

# Positive control: the gate must still REFUSE a drifted value. A verifier that
# only ever sees passes cannot distinguish "everything matches" from "the
# comparison is a no-op returning match".
ctl_conflict = CR._derived_verdict({"lead_in_s": float(current) + 0.5}, current)
ctl_unknown = CR._derived_verdict({}, current)
print(f"control drifted(+0.5) -> {ctl_conflict!r} (want conflict)")
print(f"control no-record      -> {ctl_unknown!r} (want unknown)")

fail = []
if len(mans) != EXPECTED:
    fail.append(f"expected {EXPECTED} manifests, found {len(mans)}")
if offenders:
    fail.append(f"{len(offenders)} bundle(s) would be REFUSED: {offenders[:5]}")
if ctl_conflict != CR.DERIVED_CONFLICT:
    fail.append(f"positive control failed: drift gave {ctl_conflict!r}")
if ctl_unknown != CR.DERIVED_UNKNOWN:
    fail.append(f"positive control failed: no-record gave {ctl_unknown!r}")

if fail:
    print("MANIFEST VERIFY FAILED:")
    for x in fail:
        print("  -", x)
    sys.exit(1)
print(f"MANIFEST VERIFY OK — all {len(mans)} bundles would be admitted by the gate")
