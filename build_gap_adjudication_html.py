"""build_gap_adjudication_html.py — hand-scoring page for CLASSIFIER GAP runs.

Companion to build_adjudication_html.py, but a different job. That page audits
runs the classifier DID call (agree / override). This page handles the runs it
COULD NOT call: rows whose firing rule is the `review_marginal` catch-all, i.e.
every taxonomy rule returned False and the fallthrough caught them.

Those runs are not threshold-edge cases. They are gaps in the taxonomy or in
its implementation, and they are the numerator of the "percentage of runs that
required hand scoring" number reported in the results section. This page exists
so that number is auditable: one card per gap run, both figures, the full rule
trace showing which test each arm failed, and a scoring control.

Export contract (deliberately compatible with apply_overrides.seed_from_export):
    run_id, config, model_label, model_code, decision, your_label, your_code,
    agreed, reason, gap_rule, gap_verdict, scored_at

    * A taxonomy score (0-5) exports as agreed='no' with your_code set, so
      `apply_overrides.seed_from_export` picks it up unchanged and it flows into
      the verified master via the documented path.
    * A NON-taxonomy verdict ('needs_new_type' / 'unscorable') exports with
      your_code BLANK and agreed=<verdict>. seed_from_export keeps only
      agreed=='no' rows, so these can never inject a bogus code into the master
      (_norm_code passes non-digit strings straight through, which would).
      They are still counted by gap_tally.py via the gap_verdict column.

Taxonomy: 0 none · 1 FIS · 2 FPFA · 3 FSW · 4 GSW · 5 seizure

Usage:
    python build_gap_adjudication_html.py \
        --csv master_classification.csv \
        --out-root /path/to/bundles \
        --html gap_adjudication.html \
        --toolkit /path/to/labeling_toolkit      # optional: adds rule trace
"""
from __future__ import annotations

import os
import sys
import csv
import json
import base64
import argparse
import importlib.util

# The catch-all rule name in labeling_toolkit/classifier.py. A run firing this
# rule is by definition one no taxonomy arm claimed.
GAP_RULE = "review_marginal"

LABEL_NAMES = {"0": "none", "1": "FIS", "2": "FPFA", "3": "FSW",
               "4": "GSW", "5": "seizure", "-1": "review"}

# Features worth showing on every card, with the bar each is tested against.
# (feature key, display label, note)
SHOWN_FEATS = [
    ("amp_prominence",           "amp_prominence",   "epileptiform bar 1.0"),
    ("fast_dur_s",               "fast_dur_s",       "FPFA needs >=2.0 s"),
    ("n_fpfa_sequences",         "n_fpfa_sequences", "FPFA needs >=1"),
    ("sync_spike_extent",        "sync_spike_extent", "FSW >=7 ch, GSW >=24 ch"),
    ("involve_contig_dur_s_at7", "contig_dur@7",     "seizure needs >=2.0 s"),
    ("cross_cols",               "cross_cols",       "quiet if <20"),
    ("ink_frac",                 "ink_frac",         "silent if <0.02"),
    ("max_col_frac",             "max_col_frac",     "quiet if <0.15"),
]


def _load_toolkit(path):
    """Import the classifier from an explicit path (optional).

    Probes both layouts: <path>/classifier.py (path points at the package)
    and <path>/labeling_toolkit/classifier.py (path points at the repo root).
    Import failures are reported, not swallowed -- a silent None here means
    the page ships with no rule trace and nothing says so.
    """
    if not path:
        return None
    cands = [os.path.join(path, "classifier.py"),
             os.path.join(path, "labeling_toolkit", "classifier.py")]
    src = next((c for c in cands if os.path.isfile(c)), None)
    if src is None:
        print(f"WARNING --toolkit {path!r}: no classifier.py at "
              + " or ".join(cands) + "; rule trace disabled", file=sys.stderr)
        return None
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(src)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    spec = importlib.util.spec_from_file_location("_gap_classifier", src)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(f"WARNING --toolkit import failed ({type(exc).__name__}: {exc}); "
              "rule trace disabled", file=sys.stderr)
        return None
    if not hasattr(mod, "RULES"):
        print(f"WARNING {src} has no RULES; rule trace disabled", file=sys.stderr)
        return None
    return mod


def _find_fig(out_root, config, seed, name):
    """Probe both known layouts: run_<seed>/figures/<name> and run_<seed>/<name>."""
    for cand in (
        os.path.join(out_root, config, f"run_{int(seed):03d}", "figures", name),
        os.path.join(out_root, config, f"run_{int(seed):03d}", name),
    ):
        if os.path.isfile(cand):
            return cand
    return None


def _b64(path):
    if not path:
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _rule_trace(mod, feats):
    """Evaluate every rule against a feature dict -> [(rule, fired)] + notes."""
    if mod is None or not hasattr(mod, "RULES"):
        return None
    out = []
    for entry in mod.RULES:
        name, pred = entry[0], entry[1]
        try:
            fired = bool(pred(feats))
        except Exception as exc:
            fired = f"error: {type(exc).__name__}"
        out.append({"rule": name, "fired": fired})
    extra = {}
    for fn, key in (("_band_frac", "band_frac"), ("_sw_chans", "sw_chans"),
                    ("_spike_amp", "spike_amp"), ("_band_ok", "band_ok")):
        f = getattr(mod, fn, None)
        if f is None:
            continue
        try:
            v = f(feats)
            extra[key] = bool(v) if isinstance(v, bool) else round(float(v), 4)
        except Exception:
            pass
    return {"rules": out, "derived": extra}


def collect(csv_path, out_root, toolkit=None, gap_rule=GAP_RULE):
    """Return (cards, stats) for every run whose firing rule is the catch-all."""
    mod = _load_toolkit(toolkit)
    cards, total = [], 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            if (row.get("rule") or "").strip() != gap_rule:
                continue
            config = (row.get("config_name") or row.get("config_dir") or "").strip()
            seed = row.get("seed")
            feats = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    feats[k] = float(v)
                except (TypeError, ValueError):
                    pass
            shown = []
            for key, disp, note in SHOWN_FEATS:
                if key in feats:
                    val = feats[key]
                    txt = f"{val:g}" if val == int(val) else f"{val:.2f}"
                    shown.append({"k": disp, "v": txt, "note": note})
            cards.append({
                "run_id": row.get("run_id") or f"seed{int(seed):03d}__{config}",
                "config": config,
                "seed": int(float(seed)),
                "model_code": (row.get("label") or "").strip(),
                "model_label": (row.get("label_name") or "").strip(),
                "confidence": (row.get("confidence") or "").strip(),
                "rule": (row.get("rule") or "").strip(),
                "clause": (row.get("clause") or "").strip(),
                "feats": shown,
                "trace": _rule_trace(mod, feats),
                "lfp": _b64(_find_fig(out_root, config, seed, "lfp.png")),
                "heat": _b64(_find_fig(out_root, config, seed, "heatmap_E.png")),
            })
    cards.sort(key=lambda c: (c["config"], c["seed"]))
    stats = {"total_runs": total, "gap_runs": len(cards),
             "gap_pct": (100.0 * len(cards) / total) if total else 0.0,
             "by_config": {}}
    for c in cards:
        stats["by_config"][c["config"]] = stats["by_config"].get(c["config"], 0) + 1
    return cards, stats


PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>WC-MNIST — classifier gap hand-scoring</title>
<style>
 body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
      margin:0;background:#f6f7f9;color:#1a1a1a}
 header{position:sticky;top:0;z-index:20;background:#fff;border-bottom:1px solid #d8dbe0;
        padding:12px 20px;box-shadow:0 1px 4px rgba(0,0,0,.06)}
 h1{margin:0 0 4px;font-size:17px}
 .sub{color:#555;font-size:12.5px}
 .bar{margin-top:9px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 button{font:inherit;padding:6px 13px;border:1px solid #b9bfc7;background:#fff;border-radius:6px;
        cursor:pointer}
 button:hover{background:#eef1f5}
 button.primary{background:#1f5fbf;color:#fff;border-color:#1f5fbf;font-weight:600}
 button.primary:hover{background:#1a51a3}
 #prog{font-weight:600}
 #prog .done{color:#1a7f37}
 .wrap{padding:18px 20px 60px;max-width:1680px;margin:0 auto}
 .card{background:#fff;border:1px solid #dfe2e7;border-radius:9px;margin-bottom:22px;overflow:hidden}
 .card.scored{border-color:#1a7f37;box-shadow:0 0 0 1px #1a7f37 inset}
 .chead{display:flex;justify-content:space-between;align-items:baseline;gap:14px;
        padding:11px 15px;background:#fafbfc;border-bottom:1px solid #e7e9ee;flex-wrap:wrap}
 .rid{font-weight:700;font-size:15px}
 .auto{color:#8a5a00;font-size:12.5px}
 .figs{display:grid;grid-template-columns:1.52fr 1fr;gap:10px;padding:12px 14px}
 .figs img{width:100%;height:auto;display:block;border:1px solid #e4e6eb;border-radius:5px;
           cursor:zoom-in;background:#fff}
 .figs img.zoom{position:fixed;inset:14px;width:auto;height:auto;max-width:calc(100vw - 28px);
                max-height:calc(100vh - 28px);margin:auto;z-index:100;cursor:zoom-out;
                box-shadow:0 8px 40px rgba(0,0,0,.45);object-fit:contain}
 .figcap{font-size:11.5px;color:#666;text-align:center;margin-top:3px}
 .meta{padding:2px 15px 12px}
 table.f{border-collapse:collapse;font-size:12px}
 table.f td{padding:2px 12px 2px 0;white-space:nowrap}
 table.f td.k{color:#444;font-weight:600}
 table.f td.n{color:#777}
 details.tr{margin:6px 15px 12px;font-size:12px}
 details.tr summary{cursor:pointer;color:#1f5fbf}
 .rules{display:flex;flex-wrap:wrap;gap:5px;margin-top:7px}
 .rules span{padding:2px 7px;border-radius:4px;background:#eef0f3;border:1px solid #dcdfe4;
             font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}
 .rules span.hit{background:#fde8b0;border-color:#e3c25f;font-weight:700}
 .deriv{margin-top:7px;color:#555;font-family:ui-monospace,Menlo,monospace;font-size:11px}
 .score{padding:11px 15px;background:#fbfcfd;border-top:1px solid #e7e9ee;
         display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 .score label{display:inline-flex;align-items:center;gap:5px;padding:5px 11px;border-radius:6px;
              border:1px solid #c6ccd4;background:#fff;cursor:pointer;font-size:13px}
 .score label:hover{background:#eef1f5}
 .score label.sel{background:#1f5fbf;border-color:#1f5fbf;color:#fff;font-weight:700}
 .score label.sel.alt{background:#8a4fbf;border-color:#8a4fbf}
 .score input{margin:0}
 .why{flex:1 1 340px;min-width:240px;padding:6px 9px;border:1px solid #c6ccd4;border-radius:6px;
      font:inherit;font-size:12.5px}
 .hint{font-size:11.5px;color:#777;padding:0 15px 11px}
 #tallybox{background:#fff;border:1px solid #dfe2e7;border-radius:9px;padding:14px 16px;
           margin-bottom:22px}
 #tallybox h2{margin:0 0 7px;font-size:14px}
 #tallybox table{border-collapse:collapse;font-size:13px}
 #tallybox td{padding:2px 16px 2px 0}
</style></head><body>
<header>
  <h1>WC-MNIST — classifier gap hand-scoring</h1>
  <div class="sub">Runs whose firing rule is the <code>__GAPRULE__</code> catch-all: every taxonomy
  arm returned False. These are gaps in the taxonomy or its implementation, not threshold-edge
  cases. Your scores here are the numerator of the reported hand-scored percentage.</div>
  <div class="bar">
    <span id="prog"></span>
    <button class="primary" onclick="exportCsv()">Export CSV</button>
    <button onclick="jumpNext()">Next unscored &rarr;</button>
    <button onclick="clearAll()">Clear all</button>
    <span class="sub">Keys <b>0</b>-<b>5</b> score the card under the cursor, <b>N</b> = needs new
    type, <b>U</b> = unscorable, <b>Enter</b> = next unscored.</span>
  </div>
</header>
<div class="wrap">
  <div id="tallybox"></div>
  <div id="cards"></div>
</div>
<script>
const CARDS = __CARDS__;
// Sweep identity, baked in at build time. Exported on every scored row so the
// resulting ledger is self-identifying: run_id alone is NOT unique across
// parameter-exploration sweeps, and apply_overrides.py refuses rows whose
// corpus stamp disagrees with the master being stamped.
const CORPUS = "__CORPUS__";
const STATS = __STATS__;
const KEY   = "wcmnist_gap_adjudication_v1";
const NAMES = {"0":"none","1":"FIS","2":"FPFA","3":"FSW","4":"GSW","5":"seizure"};
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { store = {}; }
function save(){ localStorage.setItem(KEY, JSON.stringify(store)); }

function tally(){
  const n = CARDS.length;
  let scored = 0, byv = {};
  CARDS.forEach(c => { const r = store[c.run_id];
    if (r && r.v) { scored++; byv[r.v] = (byv[r.v]||0)+1; } });
  const rows = Object.keys(byv).sort().map(k =>
    `<tr><td>${k in NAMES ? k+" &mdash; "+NAMES[k] : k}</td><td><b>${byv[k]}</b></td></tr>`).join("");
  document.getElementById("tallybox").innerHTML =
    `<h2>Gap tally</h2><table>
     <tr><td>runs in corpus</td><td><b>${STATS.total_runs}</b></td></tr>
     <tr><td>runs the classifier could not call</td><td><b>${STATS.gap_runs}</b>
         (${STATS.gap_pct.toFixed(2)}% of corpus)</td></tr>
     <tr><td>hand-scored on this page</td><td><b>${scored}</b> / ${n}</td></tr>
     </table>${rows ? "<h2 style='margin-top:10px'>Your scores</h2><table>"+rows+"</table>" : ""}`;
  document.getElementById("prog").innerHTML =
    scored === n ? `<span class="done">&#10003; all ${n} scored &mdash; export now</span>`
                 : `${scored} / ${n} scored`;
}

function render(){
  document.getElementById("cards").innerHTML = CARDS.map((c,i) => {
    const feats = c.feats.map(f =>
      `<tr><td class="k">${f.k}</td><td>${f.v}</td><td class="n">${f.note}</td></tr>`).join("");
    let trace = "";
    if (c.trace){
      const rr = c.trace.rules.map(r =>
        `<span class="${r.fired===true?'hit':''}">${r.rule}: ${r.fired}</span>`).join("");
      const dv = Object.entries(c.trace.derived)
        .map(([k,v]) => `${k}=${v}`).join("   ");
      trace = `<details class="tr"><summary>rule trace &mdash; which arm failed</summary>
               <div class="rules">${rr}</div><div class="deriv">${dv}</div></details>`;
    }
    const opts = ["0","1","2","3","4","5"].map(k =>
      `<label data-v="${k}"><input type="radio" name="s${i}" value="${k}">${k} ${NAMES[k]}</label>`
      ).join("") +
      `<label data-v="needs_new_type" class="alt"><input type="radio" name="s${i}"
        value="needs_new_type">N &mdash; needs new type</label>
       <label data-v="unscorable" class="alt"><input type="radio" name="s${i}"
        value="unscorable">U &mdash; unscorable</label>`;
    const img = (b64, cap) => b64
      ? `<div><img src="data:image/png;base64,${b64}" onclick="this.classList.toggle('zoom')">
         <div class="figcap">${cap}</div></div>`
      : `<div class="figcap">[${cap} missing]</div>`;
    return `<div class="card" id="card${i}" data-run="${c.run_id}" data-config="${c.config}"
              data-mcode="${c.model_code}" data-mlabel="${c.model_label}" data-rule="${c.rule}">
      <div class="chead">
        <span class="rid">${c.run_id}</span>
        <span class="auto">auto: ${c.model_label} &middot; rule <code>${c.rule}</code>
              &middot; ${c.clause}</span>
      </div>
      <div class="figs">
        ${img(c.lfp,  "Filtered LFP proxy &mdash; 28 ring electrodes (click to zoom)")}
        ${img(c.heat, "Excitatory rate &mdash; viridis, fixed 0&ndash;1 (click to zoom)")}
      </div>
      <div class="meta"><table class="f">${feats}</table></div>
      ${trace}
      <div class="score">${opts}
        <input class="why" id="why${i}" placeholder="reason / note (optional, exported)">
      </div>
      <div class="hint">A score of 0&ndash;5 exports as an override and flows into the verified
        master via apply_overrides. &ldquo;Needs new type&rdquo; and &ldquo;unscorable&rdquo; export
        with a blank code so they can never inject a non-taxonomy value, and are tallied
        separately.</div>
    </div>`;
  }).join("");

  CARDS.forEach((c,i) => {
    const card = document.getElementById("card"+i);
    const rec = store[c.run_id] || {};
    if (rec.why) document.getElementById("why"+i).value = rec.why;
    card.querySelectorAll('input[type=radio]').forEach(r => {
      if (rec.v && r.value === rec.v) r.checked = true;
      r.onchange = () => { setv(c.run_id, r.value); paint(i); };
    });
    document.getElementById("why"+i).oninput = e => {
      store[c.run_id] = Object.assign({}, store[c.run_id], {why: e.target.value}); save(); };
    paint(i);
  });
  tally();
}

function setv(run_id, v){
  store[run_id] = Object.assign({}, store[run_id],
                                {v: v, at: new Date().toISOString()});
  save(); tally();
}
function paint(i){
  const card = document.getElementById("card"+i);
  const rec = store[CARDS[i].run_id] || {};
  card.classList.toggle("scored", !!rec.v);
  card.querySelectorAll('.score label').forEach(l =>
    l.classList.toggle("sel", l.dataset.v === rec.v));
}
function jumpNext(){
  for (let i=0;i<CARDS.length;i++){
    const rec = store[CARDS[i].run_id];
    if (!rec || !rec.v){ document.getElementById("card"+i)
      .scrollIntoView({behavior:"smooth", block:"center"}); return; }
  }
  document.getElementById("tallybox").scrollIntoView({behavior:"smooth"});
}
function clearAll(){
  if (!confirm("Clear all scores on this page?")) return;
  store = {}; save(); render();
}
document.addEventListener("keydown", ev => {
  if (ev.target.tagName === "INPUT" && ev.target.type === "text") return;
  if (ev.key === "Enter"){ jumpNext(); return; }
  const k = ev.key.toLowerCase();
  let v = null;
  if ("012345".includes(ev.key)) v = ev.key;
  else if (k === "n") v = "needs_new_type";
  else if (k === "u") v = "unscorable";
  if (!v) return;
  // score the card nearest the viewport centre
  const mid = window.innerHeight / 2;
  let best = null, bd = 1e9;
  CARDS.forEach((c,i) => {
    const r = document.getElementById("card"+i).getBoundingClientRect();
    const d = Math.abs((r.top + r.bottom)/2 - mid);
    if (d < bd){ bd = d; best = i; }
  });
  if (best === null) return;
  const card = document.getElementById("card"+best);
  const inp = card.querySelector(`input[value="${v}"]`);
  if (inp){ inp.checked = true; setv(CARDS[best].run_id, v); paint(best); }
});

function exportCsv(){
  const head = ["run_id","config","model_label","model_code","decision","your_label",
                "your_code","agreed","reason","gap_rule","gap_verdict","scored_at",
                "corpus"];
  const rows = [head];
  CARDS.forEach((c,i) => {
    const rec = store[c.run_id];
    if (!rec || !rec.v) return;
    const v = rec.v;
    const taxo = v in NAMES;
    rows.push([
      c.run_id, c.config, c.model_label, c.model_code,
      taxo ? "hand_score" : v,
      taxo ? NAMES[v] : "",
      taxo ? v : "",                       // blank for non-taxonomy verdicts
      taxo ? "no" : v,                     // only 'no' is consumed by seed_from_export
      rec.why || (taxo ? "hand-scored gap run: classifier fell through to "+c.rule : v),
      c.rule, v, rec.at || "", CORPUS
    ]);
  });
  if (rows.length === 1){ alert("Nothing scored yet."); return; }
  const esc = s => { s = (s===null||s===undefined) ? "" : String(s);
    return /[",\\n]/.test(s) ? '"'+s.replace(/"/g,'""')+'"' : s; };
  const csv = rows.map(r => r.map(esc).join(",")).join("\\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {type:"text/csv"}));
  a.download = `wcmnist_gap_scores_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
}
render();
</script></body></html>
"""


def build(cards, stats, html_path, gap_rule=GAP_RULE, corpus=""):
    if not corpus:
        print("WARNING building a page with NO corpus stamp: its export will be "
              "refused by apply_overrides.py unless --allow-unstamped is passed. "
              "Pass --params (or a --csv carrying fingerprint/code_tag).")
    page = (PAGE.replace("__CARDS__", json.dumps(cards))
                .replace("__STATS__", json.dumps(stats))
                .replace("__CORPUS__", corpus)
                .replace("__GAPRULE__", gap_rule))
    tmp = html_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(page)
    os.replace(tmp, html_path)
    return html_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="master_classification.csv")
    ap.add_argument("--out-root", required=True,
                    help="bundle tree root: <out-root>/<config>/run_<seed>/[figures/]")
    ap.add_argument("--html", default="gap_adjudication.html")
    ap.add_argument("--toolkit", default=None,
                    help="path to labeling_toolkit (adds the per-run rule trace)")
    ap.add_argument("--gap-rule", default=GAP_RULE,
                    help=f"firing rule that marks a gap run (default {GAP_RULE})")
    ap.add_argument("--params", default=None,
                    help="all_runs_params.csv -- identifies the corpus, which is "
                         "stamped into every exported score row")
    ap.add_argument("--corpus", default=None,
                    help="override the corpus stamp (normally read from --csv/--params)")
    a = ap.parse_args()

    cards, stats = collect(a.csv, a.out_root, a.toolkit, a.gap_rule)
    if not cards:
        print(f"no runs with rule={a.gap_rule!r} in {a.csv}")
        return
    corpus = a.corpus
    if corpus is None:
        try:
            from apply_overrides import corpus_of
            corpus = corpus_of(a.csv, a.params)
        except Exception as e:
            print(f"WARNING could not identify corpus: {type(e).__name__}: {e}")
            corpus = ""
    print(f"corpus: {corpus or '<none>'}")
    missing = [c["run_id"] for c in cards if not c["lfp"] or not c["heat"]]
    build(cards, stats, a.html, a.gap_rule, corpus=corpus)
    print(f"{stats['gap_runs']} gap run(s) of {stats['total_runs']} "
          f"({stats['gap_pct']:.2f}%) -> {a.html}")
    print("by config:", stats["by_config"])
    if missing:
        print(f"WARNING figures missing for {len(missing)}: {missing[:8]}")


if __name__ == "__main__":
    main()
