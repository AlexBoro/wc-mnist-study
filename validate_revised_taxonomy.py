"""validate_revised_taxonomy.py — check the revised classifier against the user's
100-card adjudication (borderline_adjudication_2026-08-11.csv).

Runs labeling_toolkit.features.extract_features + classifier.classify on each
recovered figure pair and compares to the user's own call. Prints a confusion
matrix and lists every disagreement with the features that drove it.

Usage:  python validate_revised_taxonomy.py <cards_dir> <adjudication_csv> [out_csv]
  cards_dir/<run_id>/{heatmap_E.png,lfp.png}
"""
import sys, csv
from pathlib import Path
from labeling_toolkit import features as F, classifier as C

def main(cards_dir, adj_csv, out_csv=None):
    cards = Path(cards_dir)
    with open(adj_csv, newline="") as fh:
        adj = list(csv.DictReader(fh))
    rows, agree = [], 0
    for r in adj:
        rid = r["run_id"]
        d = cards / rid
        hm, lfp = d / "heatmap_E.png", d / "lfp.png"
        if not (hm.exists() and lfp.exists()):
            print(f"MISSING figures: {rid}"); continue
        f = F.extract_features(str(hm), str(lfp))
        v = C.classify(f)
        yours = int(r["your_code"])
        ok = (v["label"] == yours)
        agree += ok
        rows.append(dict(run_id=rid, group=r["group"], yours=yours,
                         new=v["label"], rule=v["rule"], match=int(ok),
                         old_auto=int(r["auto_code"]),
                         fast_dur_s=f.get("fast_dur_s"),
                         contig7_dur_s=(f.get("fast_contig_dur_s") or [0]*28)[6],
                         fast_max_contig=f.get("fast_max_contig"),
                         n_fast=f.get("n_fast_channels"),
                         ibi_ratio=f.get("fast_ibi_ratio"),
                         spike_amp_max=f.get("spike_amp_max"),
                         band_frac=f.get("burst_band_frac"),
                         band_idx=f.get("burst_band_idx"),
                         sync=f.get("sync_spike_extent"),
                         n_complexes=f.get("n_complexes"), max_IDI=f.get("max_IDI")))
    n = len(rows)
    print(f"\nagreement with user: {agree}/{n} = {agree/n:.1%}")
    labs = sorted({r["yours"] for r in rows} | {r["new"] for r in rows})
    print("\nconfusion (rows = your call, cols = new classifier)")
    print("      " + "".join(f"{c:>6}" for c in labs))
    for a in labs:
        line = f"{a:>4}: "
        for b in labs:
            line += f"{sum(1 for r in rows if r['yours']==a and r['new']==b):>6}"
        print(line)
    bad = [r for r in rows if not r["match"]]
    if bad:
        print(f"\n{len(bad)} disagreement(s):")
        for r in bad:
            print(f"  {r['run_id']}  yours={r['yours']} new={r['new']} via {r['rule']}")
            print(f"      contig7={r['contig7_dur_s']}s max_contig={r['fast_max_contig']} "
                  f"fast_dur={r['fast_dur_s']}s ibi={r['ibi_ratio']} amp={r['spike_amp_max']} "
                  f"band={r['band_idx']}idx sync={r['sync']}")
    # old-vs-new on the same cards
    old_agree = sum(1 for r in rows if r["old_auto"] == r["yours"])
    print(f"\nprevious classifier on the same cards: {old_agree}/{n} = {old_agree/n:.1%}")
    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"wrote {out_csv}")
    return rows

if __name__ == "__main__":
    main(*sys.argv[1:])
