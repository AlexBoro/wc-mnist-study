"""build_adjudication_html.py — self-contained interactive adjudication page.

Reads the master classification CSV (from classify_runs.py) and the per-run
figure pair (heatmap_E.png + lfp.png), and emits ONE self-contained HTML file:
each card shows both figures (base64-embedded), the auto label + confidence +
firing rule + clause + key metrics, and radio controls to AGREE or override to
any taxonomy code. Adjudications persist in localStorage (fresh namespace key so
this page never collides with the epoch-validation page), export to CSV, and can
be filtered (all / review / seizure / to-do / overrides).

Card SAMPLE = every ANCHOR (any non-none auto-call + any review-flagged run)
UNION a random sample of the none-calls (--sample N, seeded via --seed). Every
positive claim the classifier makes gets a human look; the none-calls get a
random audit — mirroring the detector-validation page.

Taxonomy: 0 none · 1 FIS · 2 FPFA · 3 FSW · 4 GSW · 5 seizure · R review

Usage:
    python build_adjudication_html.py \\
        --csv master_classification.csv \\
        --out-root /path/to/wc_mnist_param_explore \\
        --html adjudication.html --sample 120 --seed 0
"""
from __future__ import annotations

import os
import csv
import base64
import random
import argparse
import html as _html
from pathlib import Path

CONFIGS = ["simultaneous", "sequential_noH", "sequential_H"]

# taxonomy code -> (display name, chip colour)
TAX = {
    "0": ("0 none",     "#41ab5d"),
    "1": ("1 FIS",      "#78c679"),
    "2": ("2 FPFA",     "#fe9929"),
    "3": ("3 FSW",      "#ec7014"),
    "4": ("4 GSW",      "#cc4c02"),
    "5": ("5 seizure",  "#a50f15"),
    "R": ("review",     "#6a51a3"),
}
STORE_KEY = "wcmnist_paramexplore_adjudication_v1"

# metrics surfaced on each card (label -> pretty)
METRIC_KEYS = [
    ("ink_frac", "ink"), ("max_col_frac", "max_col"), ("band_score", "band"),
    ("n_complexes", "cplx"), ("cross_cols", "cross"),
    ("n_active_channels", "chans"), ("fast_activity_duration_s", "fast_s"),
    ("sync_spike_extent", "sync"), ("vertical_spread_fraction", "vspread"),
]


def _code_of(row):
    """Taxonomy code string for a master-CSV row (review sentinel -> 'R')."""
    try:
        lab = int(float(row.get("label", "")))
    except (ValueError, TypeError):
        return "R"
    if lab == -1:
        return "R"
    return str(lab)


def _b64_img(path):
    try:
        b = Path(path).read_bytes()
    except OSError:
        return None
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")


def _run_dir(out_root, config_name, seed):
    return Path(out_root) / config_name / f"run_{int(seed):03d}"


def _find_fig(run_dir, name):
    """Figures live under run_<seed>/figures/ (Colab bundle layout) or directly
    in run_<seed>/ (early mock trees). Check both."""
    for cand in (run_dir / "figures" / name, run_dir / name):
        if cand.exists():
            return cand
    return run_dir / name


def _is_anchor(r):
    """A card we always show for human verification: any review-flagged run OR
    any abnormal (non-none) auto-call. The classifier's positive claims are
    exactly what most needs a human look; none-calls get a random audit."""
    if str(r.get("review", "0")) == "1":
        return True
    code = _code_of(r)
    return code not in ("0",)  # anything not 'none (0)' (incl. review 'R')


def _select(rows, sample_n, seed):
    """Return the ordered subset: all anchors (review-flagged + every non-none
    auto-call) + a random sample of the none-calls."""
    review = [r for r in rows if _is_anchor(r)]
    review_ids = {r["run_id"] for r in review}
    rest = [r for r in rows if r["run_id"] not in review_ids]
    rng = random.Random(seed)
    if sample_n is not None and sample_n < len(rest):
        rest = rng.sample(rest, sample_n)
    chosen = review + rest
    chosen.sort(key=lambda r: (CONFIGS.index(r["config_name"])
                               if r["config_name"] in CONFIGS else 99,
                               int(float(r["seed"]))))
    # tag which are review (anchors) for the filter
    for r in chosen:
        r["_is_review"] = r["run_id"] in review_ids
    return chosen, len(review), len(chosen)


def _card_html(row, out_root):
    rid = row["run_id"]
    cname = row["config_name"]
    seed = row["seed"]
    code = _code_of(row)
    name, colour = TAX.get(code, (code, "#777"))
    conf = row.get("confidence", "")
    rule = row.get("rule", "")
    clause = row.get("clause", "")
    is_review = row.get("_is_review", False)

    rd = _run_dir(out_root, cname, seed)
    hm = _b64_img(_find_fig(rd, "heatmap_E.png"))
    lfp = _b64_img(_find_fig(rd, "lfp.png"))

    metrics = " ".join(
        f"{lbl}={row[k]}" for k, lbl in METRIC_KEYS
        if row.get(k, "") not in ("", None)
    )

    opts = []
    for c in ["0", "1", "2", "3", "4", "5", "R"]:
        nm, col = TAX[c]
        opts.append(
            f'<label data-val="{c}" onclick="pick(this.closest(\'.card\'),\'{c}\')" '
            f'style="border-color:{col}">{_html.escape(nm)}</label>'
        )
    agree = ('<label data-val="AGREE" class="agreeopt" '
             'onclick="pick(this.closest(\'.card\'),\'AGREE\')">&#10003; agree</label>')

    imgs = ""
    if hm:
        imgs += (f'<figure><img src="{hm}"><figcaption>heatmap E '
                 f'(viridis 0&ndash;1)</figcaption></figure>')
    if lfp:
        imgs += (f'<figure><img src="{lfp}"><figcaption>LFP '
                 f'(28-ring montage)</figcaption></figure>')
    if not imgs:
        imgs = '<div class="diff">figure pair missing for this run</div>'

    return f"""
<div class="card" data-run="{_html.escape(rid)}" data-config="{_html.escape(cname)}"
     data-modelcode="{code}" data-modelname="{_html.escape(name)}"
     data-review="{1 if is_review else 0}">
  <h3>{_html.escape(rid)}
      <span class="tag" style="background:{colour}">{_html.escape(name)}</span>
      <span class="conf">conf: {_html.escape(str(conf))}</span>
      {'<span class="conf" style="color:#6a51a3">&#9733; review</span>' if is_review else ''}
  </h3>
  <div class="clause">{_html.escape(str(clause))}</div>
  <div class="metrics"><span class="rule">{_html.escape(str(rule))}</span> &nbsp; {_html.escape(metrics)}</div>
  <div class="imgs">{imgs}</div>
  <div class="adj">
    <span class="lab">Your call:</span>
    <span class="opts">{agree}{''.join(opts)}</span>
    <span class="state"></span>
  </div>
</div>"""


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f4f6;color:#1a1a1a}
.wrap{max-width:1100px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 4px} .sub{color:#555;font-size:13px;margin-bottom:12px}
.legend{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 12px}
.chip{padding:4px 10px;border-radius:12px;color:#fff;font-size:12px;font-weight:600}
.toolbar{position:sticky;top:0;z-index:20;background:#fff;border-radius:8px;padding:12px 16px;margin-bottom:18px;
  box-shadow:0 1px 4px rgba(0,0,0,.12);display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.toolbar .grp{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.fbtn{border:1px solid #ccc;background:#f7f7f9;border-radius:6px;padding:5px 10px;font-size:12px;cursor:pointer}
.fbtn.active{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
.abtn{border:0;border-radius:6px;padding:6px 12px;font-size:12px;font-weight:600;cursor:pointer;color:#fff}
.abtn.exp{background:#2c7fb8} .abtn.clr{background:#999}
.prog{font-size:13px;font-weight:600;margin-left:auto}
.prog b{color:#2c7fb8}
.card{background:#fff;border-radius:8px;padding:14px 16px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.08);border-left:6px solid #ccc}
.card.done{box-shadow:0 0 0 2px #41ab5d inset,0 1px 3px rgba(0,0,0,.08)}
.card h3{margin:0 0 6px;font-size:15px}
.tag{display:inline-block;padding:2px 9px;border-radius:10px;color:#fff;font-size:12px;font-weight:700;vertical-align:middle}
.conf{font-size:11px;color:#666;margin-left:6px}
.rule{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#444;background:#f0f0f3;padding:1px 6px;border-radius:4px}
.clause{font-size:12px;color:#555;margin:4px 0 8px}
.metrics{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#333;margin:6px 0}
.imgs{display:flex;gap:10px;flex-wrap:wrap} .imgs figure{margin:0;flex:1;min-width:340px}
.imgs img{width:100%;border:1px solid #ddd;border-radius:4px} .imgs figcaption{font-size:11px;color:#777;text-align:center;margin-top:2px}
.diff{background:#fff3cd;border:1px solid #ffe08a;border-radius:6px;padding:6px 10px;font-size:12px;margin:6px 0;color:#7a5c00}
.adj{margin-top:10px;padding:10px;border-top:1px solid #eee;background:#fafafb;border-radius:6px}
.adj .lab{font-size:12px;font-weight:600;color:#333;margin-right:8px}
.opts{display:inline-flex;flex-wrap:wrap;gap:4px;vertical-align:middle}
.opts label{font-size:12px;border:1px solid #ccc;border-radius:6px;padding:4px 9px;cursor:pointer;user-select:none;background:#fff}
.opts label.agreeopt{background:#eaf5ea;border-color:#41ab5d;font-weight:700}
.opts label.sel{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
.state{margin-left:10px;font-size:12px;font-weight:600}
.state.agree{color:#41ab5d} .state.override{color:#cc4c02}
"""

JS = """
const STORE_KEY = '%STORE_KEY%';
function loadState(){ try{return JSON.parse(localStorage.getItem(STORE_KEY))||{};}catch(e){return {};} }
function saveState(s){ localStorage.setItem(STORE_KEY, JSON.stringify(s)); }
let STATE = loadState();
function cardKey(card){ return card.dataset.run; }
function labelName(code){
  return {'0':'0 none','1':'1 FIS','2':'2 FPFA','3':'3 FSW','4':'4 GSW','5':'5 seizure','R':'review'}[code]||code;
}
function applySelection(card){
  const chosen = STATE[cardKey(card)];
  card.querySelectorAll('.opts label').forEach(l=>{
    l.classList.remove('sel');
    if(chosen!==undefined && l.dataset.val===chosen) l.classList.add('sel');
  });
  const st = card.querySelector('.state');
  card.classList.toggle('done', chosen!==undefined);
  if(chosen===undefined){ st.textContent=''; st.className='state'; }
  else if(chosen==='AGREE'){ st.textContent='\\u2713 agreed'; st.className='state agree'; }
  else { st.textContent='\\u2192 override: '+labelName(chosen); st.className='state override'; }
}
function pick(card, val){
  const key = cardKey(card);
  if(STATE[key]===val){ delete STATE[key]; } else { STATE[key]=val; }
  saveState(STATE); applySelection(card); updateProgress();
}
function updateProgress(){
  const cards=[...document.querySelectorAll('.card')];
  let done=0, agree=0, over=0;
  cards.forEach(c=>{ const v=STATE[cardKey(c)]; if(v!==undefined){done++; if(v==='AGREE')agree++; else over++;} });
  document.getElementById('prog').innerHTML =
    `<b>${done}</b>/${cards.length} adjudicated &nbsp;\\u00b7&nbsp; ${agree} agree &nbsp;\\u00b7&nbsp; ${over} override`;
}
let FILTER='all';
function setFilter(f, btn){
  FILTER=f;
  document.querySelectorAll('.fbtn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.card').forEach(c=>{
    const v=STATE[cardKey(c)]; let show=true;
    if(f==='review') show = c.dataset.review==='1';
    else if(f==='seizure') show = c.dataset.modelcode==='5';
    else if(f==='none') show = c.dataset.modelcode==='0';
    else if(f==='todo') show = (v===undefined);
    else if(f==='override') show = (v!==undefined && v!=='AGREE');
    else if(CONFIGS.includes(f)) show = c.dataset.config===f;
    c.style.display = show ? '' : 'none';
  });
}
const CONFIGS=['simultaneous','sequential_noH','sequential_H'];
function exportCSV(){
  const cards=[...document.querySelectorAll('.card')];
  const rows=[['run_id','config','model_label','model_code','decision','your_label','your_code','agreed']];
  cards.forEach(c=>{
    const v=STATE[cardKey(c)];
    const model=c.dataset.modelname, mcode=c.dataset.modelcode;
    let decision='', yourlbl='', ycode='', agreed='';
    if(v===undefined){ decision='(not adjudicated)'; }
    else if(v==='AGREE'){ decision='agree'; yourlbl=model; ycode=mcode; agreed='yes'; }
    else { decision='override'; yourlbl=labelName(v); ycode=v; agreed='no'; }
    rows.push([c.dataset.run, c.dataset.config, model, mcode, decision, yourlbl, ycode, agreed]);
  });
  const csv=rows.map(r=>r.map(x=>`"${String(x).replace(/"/g,'""')}"`).join(',')).join('\\n');
  const blob=new Blob([csv],{type:'text/csv'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download=`wcmnist_paramexplore_adjudication_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
}
function clearAll(){
  if(!confirm('Clear ALL your adjudication decisions? This cannot be undone.')) return;
  STATE={}; saveState(STATE);
  document.querySelectorAll('.card').forEach(applySelection); updateProgress();
}
window.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('.card').forEach(applySelection); updateProgress();
});
"""


def build(csv_path, out_root, html_path, sample_n=120, seed=0, verbose=True):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    chosen, n_review, n_cards = _select(rows, sample_n, seed)

    # label histogram over the FULL master, for the header
    from collections import Counter
    hist = Counter(_code_of(r) for r in rows)

    legend = "".join(
        f'<span class="chip" style="background:{col}">{_html.escape(nm)}</span>'
        for _, (nm, col) in TAX.items()
    )
    filters = "".join(
        f'<button class="fbtn{" active" if f=="all" else ""}" '
        f'onclick="setFilter(\'{f}\',this)">{_html.escape(lbl)}</button>'
        for f, lbl in [("all", "all"), ("review", "review only"),
                       ("seizure", "seizure"), ("none", "none"),
                       ("todo", "to-do"), ("override", "overrides"),
                       ("simultaneous", "simult."),
                       ("sequential_noH", "seq_noH"),
                       ("sequential_H", "seq_H")]
    )
    cards = "\n".join(_card_html(r, out_root) for r in chosen)
    hist_txt = " · ".join(f"{TAX.get(k,(k,''))[0]}: {v}"
                          for k, v in sorted(hist.items()))

    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>WC-MNIST param-explore — adjudication</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>WC-MNIST parameter-explore &mdash; adjudication</h1>
<div class="sub">{n_cards} cards shown ({n_review} anchors: every abnormal + review-flagged auto-call,
always shown &middot; plus a random audit sample of none-calls, seed={seed}). Full master:
{len(rows)} runs. Auto-label histogram: {_html.escape(hist_txt)}.</div>
<div class="legend">{legend}</div>
<div class="toolbar">
  <div class="grp">{filters}</div>
  <div class="grp">
    <button class="abtn exp" onclick="exportCSV()">Export CSV</button>
    <button class="abtn clr" onclick="clearAll()">Clear all</button>
  </div>
  <div class="prog" id="prog"></div>
</div>
{cards}
</div><script>{JS.replace('%STORE_KEY%', STORE_KEY)}</script></body></html>"""

    Path(html_path).write_text(page, encoding="utf-8")
    if verbose:
        mb = os.path.getsize(html_path) / 1e6
        print(f"wrote {html_path} ({mb:.1f} MB) — {n_cards} cards "
              f"({n_review} review + {n_cards - n_review} sampled)")
        print("full-master histogram:", dict(sorted(hist.items())))
    return dict(cards=n_cards, review=n_review, total=len(rows),
                hist=dict(hist), path=str(html_path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="master_classification.csv")
    ap.add_argument("--out-root", required=True,
                    help="param-explore output tree (…/wc_mnist_param_explore)")
    ap.add_argument("--html", default="adjudication.html")
    ap.add_argument("--sample", type=int, default=120,
                    help="random sample of non-review runs (default 120)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.csv, args.out_root, args.html, sample_n=args.sample, seed=args.seed)
