"""Scrollable contact sheet over a param-explore sweep's figures.

Builds ONE self-contained .html file holding downscaled thumbnails of every
run's figures, laid out so the comparison the pilot exists to make is the
one you read across a row.

LAYOUT.  Row = (config, seed); column = coefficient arm.  The three arms of a
given seed+config share bit-identical initialization and differ only in E->E
drive, so reading left-to-right along a row isolates the coefficient's effect
with everything else held fixed.  Reading down a column shows seed-to-seed
spread within one arm.  Any other arrangement (e.g. grouping by arm first)
buries the paired comparison.

THUMBNAILS ARE NOT MEASURABLE.  Images are downscaled and JPEG-recompressed to
keep one file openable in a browser; pixel deflections read off them are NOT
calibration-grade.  Every card links to its full-resolution PNG, and the
detector must always run against those.  This mirrors the lesson from
detector_validation_seeds0-4.html, where measuring the embedded thumbnails
would have mis-set the LFP thresholds.

USAGE
    python build_contact_sheet.py --out-root /content/drive/MyDrive/wc_mnist_param_explore_v2
    python build_contact_sheet.py --out-root ... --dynamics     # adds mean_E, sortable

--dynamics reads E_traj from every arrays.npz (~31 MB/run) to compute regime
statistics, so it is opt-in; results are cached to contact_sheet_dynamics.csv
at the sweep root and reused on later builds.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import os
from pathlib import Path

MASTER_LOG = "param_explore_log.csv"
DYN_CACHE = "contact_sheet_dynamics.csv"

# Thumbnail width in px.  460 keeps a 3-arm row readable side by side on a
# 1500px screen while holding the whole pilot under ~15 MB.
THUMB_W = 460
JPEG_Q = 72

FIG_NAMES = ("lfp.png", "heatmap_E.png")


# --------------------------------------------------------------------------
# tree walking
# --------------------------------------------------------------------------
def find_fig(run_dir: Path, name: str) -> Path | None:
    """Figures live under run_<seed>/figures/ (real Colab bundle layout) or
    directly in run_<seed>/ (early mock trees).  Check both, as the other
    drivers do."""
    for cand in (run_dir / "figures" / name, run_dir / name):
        if cand.exists():
            return cand
    return None


def discover_runs(out_root: Path) -> list[dict]:
    """Walk out_root and return one record per run directory found.

    Handles both the armed layout (out_root/arm/config/run_SSS) and the
    unarmed one (out_root/config/run_SSS) by looking for the run_* level and
    reading the arm off params.json rather than inferring it from depth --
    the bundle is authoritative about its own arm.
    """
    runs = []
    for manifest in sorted(out_root.rglob("manifest.json")):
        rd = manifest.parent
        if not rd.name.startswith("run_"):
            continue
        params = {}
        pj = rd / "params.json"
        if pj.exists():
            try:
                params = json.loads(pj.read_text())
            except Exception:
                params = {}
        try:
            seed = int(rd.name.split("_")[1])
        except Exception:
            seed = int(params.get("seed", -1))
        cfg_name = str(params.get("config_name") or rd.parent.name)
        arm = str(params.get("arm", "") or "")
        if not arm:
            # Unarmed sweeps have no arm level; armed ones do.  Only trust the
            # directory when params.json did not say.
            rel = rd.relative_to(out_root).parts
            arm = rel[0] if len(rel) >= 3 else ""
        runs.append(dict(
            run_dir=rd, seed=seed, config_name=cfg_name, arm=arm,
            params=params,
            figs={n: find_fig(rd, n) for n in FIG_NAMES},
        ))
    return runs


def read_master_log(out_root: Path) -> dict[tuple, dict]:
    """Index the sweep-root master log by (arm, config_name, seed)."""
    path = out_root / MASTER_LOG
    if not path.exists():
        return {}
    idx = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (str(row.get("arm", "") or ""),
                       str(row.get("config_name", "")),
                       int(float(row.get("seed", -1))))
            except Exception:
                continue
            idx[key] = row
    return idx


# --------------------------------------------------------------------------
# optional dynamics stats (cached -- reading E_traj is the expensive part)
# --------------------------------------------------------------------------
def compute_dynamics(runs: list[dict], out_root: Path, hi=0.75, lo=0.05,
                     verbose=True) -> dict[tuple, dict]:
    """mean_E and saturation/quiescence occupancy per run, cached to disk.

    These are the statistics that decide whether an arm is usable at all (a
    network pinned near saturation cannot show a transition), so they are what
    the sheet sorts on.  E_traj is ~16 MB per run, hence the cache.
    """
    cache_path = out_root / DYN_CACHE
    cache: dict[tuple, dict] = {}
    if cache_path.exists():
        with open(cache_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    cache[(row["arm"], row["config_name"], int(row["seed"]))] = row
                except Exception:
                    continue

    import numpy as np
    fresh = 0
    for i, r in enumerate(runs):
        key = (r["arm"], r["config_name"], r["seed"])
        if key in cache:
            continue
        npz = r["run_dir"] / "arrays.npz"
        if not npz.exists():
            continue
        try:
            with np.load(npz) as z:
                E = z["E_traj"]
                cache[key] = dict(
                    arm=key[0], config_name=key[1], seed=key[2],
                    mean_E=f"{float(E.mean()):.6f}",
                    max_E=f"{float(E.max()):.6f}",
                    frac_high=f"{float((E > hi).mean()):.6f}",
                    frac_low=f"{float((E < lo).mean()):.6f}",
                    std_E=f"{float(E.std()):.6f}",
                )
            fresh += 1
            if verbose and fresh % 10 == 0:
                print(f"  dynamics: {fresh} computed ({i+1}/{len(runs)} scanned)",
                      flush=True)
        except Exception as e:
            print(f"  [warn] dynamics failed for {r['run_dir'].name}: "
                  f"{type(e).__name__}: {e}", flush=True)

    if fresh and cache:
        cols = ["arm", "config_name", "seed", "mean_E", "max_E", "frac_high",
                "frac_low", "std_E"]
        with open(cache_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for v in cache.values():
                w.writerow({k: v.get(k, "") for k in cols})
        if verbose:
            print(f"  dynamics cache written: {cache_path.name} "
                  f"({fresh} new, {len(cache)} total)")
    return cache


# --------------------------------------------------------------------------
# thumbnails
# --------------------------------------------------------------------------
def thumb_b64(path: Path | None, width=THUMB_W, quality=JPEG_Q) -> tuple[str, int]:
    """Downscaled JPEG data-URI for one PNG.  Returns ('', 0) if unreadable."""
    if path is None or not path.exists():
        return "", 0
    try:
        from PIL import Image
    except ImportError:
        # No PIL: embed the PNG as-is rather than failing the whole sheet.
        raw = path.read_bytes()
        return "data:image/png;base64," + base64.b64encode(raw).decode(), len(raw)
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width > width:
                h = max(1, round(im.height * width / im.width))
                im = im.resize((width, h), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
        raw = buf.getvalue()
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode(), len(raw)
    except Exception:
        return "", 0


# --------------------------------------------------------------------------
# html
# --------------------------------------------------------------------------
CSS = """
:root { --bg:#12141a; --card:#1c2028; --edge:#2c323d; --fg:#e6e9ef;
        --dim:#8b94a3; --hot:#ff6b6b; --cool:#4dabf7; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:13px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
header { position:sticky; top:0; z-index:20; background:#0e1015ee;
         backdrop-filter:blur(8px); border-bottom:1px solid var(--edge);
         padding:10px 16px; }
h1 { margin:0 0 6px; font-size:15px; font-weight:600; letter-spacing:.2px; }
.meta { color:var(--dim); font-size:11px; }
.controls { display:flex; gap:14px; align-items:center; flex-wrap:wrap;
            margin-top:8px; }
.controls label { color:var(--dim); font-size:11px; text-transform:uppercase;
                  letter-spacing:.5px; }
select, button { background:var(--card); color:var(--fg);
                 border:1px solid var(--edge); border-radius:5px;
                 padding:4px 8px; font-size:12px; }
button { cursor:pointer; }
button:hover { border-color:var(--cool); }
.warn { margin:0; padding:7px 16px; background:#3a2a12;
        border-bottom:1px solid #5a4520; color:#ffd08a; font-size:11.5px; }
main { padding:14px 16px 60px; }
.rowgroup { margin-bottom:22px; border:1px solid var(--edge);
            border-radius:8px; background:#161920; overflow:hidden; }
.rowhead { display:flex; justify-content:space-between; align-items:baseline;
           padding:7px 12px; background:#1a1e26;
           border-bottom:1px solid var(--edge); }
.rowhead .id { font-weight:600; font-size:13px; }
.rowhead .st { color:var(--dim); font-size:11px; font-family:ui-monospace,monospace; }
.arms { display:grid; gap:1px; background:var(--edge); }
.cell { background:var(--card); padding:8px; min-width:0; }
.cell .armname { font-family:ui-monospace,monospace; font-size:11px;
                 color:var(--cool); margin-bottom:5px; display:flex;
                 justify-content:space-between; }
.cell .armname .w { color:var(--dim); }
.cell img { width:100%; display:block; border-radius:3px; cursor:zoom-in;
            background:#fff; }
.cell .figlabel { color:var(--dim); font-size:10px; margin:4px 0 2px;
                  text-transform:uppercase; letter-spacing:.4px; }
.cell .missing { color:var(--hot); font-size:11px; padding:18px 0;
                 text-align:center; border:1px dashed #4a3540; border-radius:3px; }
.stats { font-family:ui-monospace,monospace; font-size:10.5px; color:var(--dim);
         margin-top:5px; line-height:1.5; }
.stats b { color:var(--fg); font-weight:600; }
.pin { color:var(--hot); }
a.full { color:var(--cool); font-size:10px; text-decoration:none; }
a.full:hover { text-decoration:underline; }
#lb { position:fixed; inset:0; background:#000000f2; display:none;
      z-index:100; cursor:zoom-out; }
#lb img { position:absolute; top:50%; left:50%;
          transform:translate(-50%,-50%); max-width:98vw; max-height:98vh; }
#lb .cap { position:fixed; bottom:10px; left:0; right:0; text-align:center;
           color:var(--dim); font-size:12px; font-family:ui-monospace,monospace; }
.hidden { display:none !important; }
"""

JS = """
const rows = Array.from(document.querySelectorAll('.rowgroup'));
function apply() {
  const cf = document.getElementById('f-config').value;
  const fg = document.getElementById('f-fig').value;
  const sk = document.getElementById('f-sort').value;
  rows.forEach(r => {
    const ok = (cf === '*' || r.dataset.config === cf);
    r.classList.toggle('hidden', !ok);
  });
  document.querySelectorAll('.figblock').forEach(b => {
    b.classList.toggle('hidden', fg !== '*' && b.dataset.fig !== fg);
  });
  const main = document.getElementById('main');
  const vis = rows.slice().sort((a, b) => {
    if (sk === 'natural')
      return a.dataset.order.localeCompare(b.dataset.order);
    const av = parseFloat(a.dataset[sk]), bv = parseFloat(b.dataset[sk]);
    if (isNaN(av) && isNaN(bv)) return 0;
    if (isNaN(av)) return 1;
    if (isNaN(bv)) return -1;
    return bv - av;
  });
  vis.forEach(r => main.appendChild(r));
  const n = rows.filter(r => !r.classList.contains('hidden')).length;
  document.getElementById('count').textContent = n + ' of ' + rows.length + ' rows';
}
['f-config','f-fig','f-sort'].forEach(id =>
  document.getElementById(id).addEventListener('change', apply));
const lb = document.getElementById('lb');
document.addEventListener('click', e => {
  if (e.target.tagName === 'IMG' && e.target.closest('.cell')) {
    lb.querySelector('img').src = e.target.src;
    lb.querySelector('.cap').textContent = e.target.dataset.cap || '';
    lb.style.display = 'block';
  } else if (e.target.closest('#lb')) { lb.style.display = 'none'; }
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') lb.style.display = 'none';
});
apply();
"""


def build_html(runs, log_idx, dyn, out_root: Path, title: str,
               sat_thresh=0.75, verbose=True) -> tuple[str, dict]:
    """Assemble the sheet.  Rows are (config, seed); columns are arms."""
    arms = sorted({r["arm"] for r in runs})
    configs = sorted({r["config_name"] for r in runs})
    # Group into paired rows so a row is one seed+config across all arms.
    groups: dict[tuple, dict] = {}
    for r in runs:
        groups.setdefault((r["config_name"], r["seed"]), {})[r["arm"]] = r

    total_bytes = 0
    n_missing = 0
    parts = []
    order_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1]))

    for gi, key in enumerate(order_keys):
        cname, seed = key
        byarm = groups[key]
        # Row-level sort keys: worst-case (max) across the row's arms, so
        # sorting surfaces the seed+config where ANY arm misbehaves.
        def rowstat(field):
            vals = []
            for a, r in byarm.items():
                d = dyn.get((a, cname, seed))
                if d and d.get(field) not in ("", None):
                    try:
                        vals.append(float(d[field]))
                    except Exception:
                        pass
            return max(vals) if vals else float("nan")

        r_mean = rowstat("mean_E")
        r_high = rowstat("frac_high")
        cells = []
        for arm in arms:
            r = byarm.get(arm)
            if r is None:
                cells.append('<div class="cell"><div class="armname">'
                             f'{html.escape(arm)}</div>'
                             '<div class="missing">no run</div></div>')
                n_missing += 1
                continue
            p = r["params"]
            wl = p.get("w_celoc", "")
            wr = p.get("w_cerem", "")
            wtxt = (f"w={float(wl):.3f}/{float(wr):.3f}"
                    if wl != "" and wr != "" else "")
            figparts = []
            for fname in FIG_NAMES:
                uri, nb = thumb_b64(r["figs"].get(fname))
                total_bytes += nb
                lab = "LFP (28 ring electrodes)" if fname == "lfp.png" else "E activity"
                if not uri:
                    figparts.append(
                        f'<div class="figblock" data-fig="{fname}">'
                        f'<div class="figlabel">{lab}</div>'
                        '<div class="missing">figure missing</div></div>')
                    n_missing += 1
                    continue
                fp = r["figs"][fname]
                try:
                    rel = os.path.relpath(fp, out_root)
                except Exception:
                    rel = str(fp)
                cap = f"{cname} seed{seed:03d} [{arm}] {fname}"
                figparts.append(
                    f'<div class="figblock" data-fig="{fname}">'
                    f'<div class="figlabel">{lab} '
                    f'<a class="full" href="{html.escape(rel)}" '
                    f'target="_blank">full-res &#8599;</a></div>'
                    f'<img src="{uri}" loading="lazy" '
                    f'data-cap="{html.escape(cap)}"></div>')

            d = dyn.get((arm, cname, seed))
            if d:
                fh = float(d["frac_high"])
                pin = ' <span class="pin">&#9679; pinned</span>' if fh > 0.5 else ""
                stat = (f'<div class="stats">mean_E <b>{float(d["mean_E"]):.3f}</b> '
                        f'&nbsp; max <b>{float(d["max_E"]):.3f}</b><br>'
                        f'frac&gt;{sat_thresh:g} <b>{fh:.3f}</b> '
                        f'&nbsp; frac&lt;0.05 <b>{float(d["frac_low"]):.3f}</b>'
                        f'{pin}</div>')
            else:
                lg = log_idx.get((arm, cname, seed), {})
                bits = [f'{k} <b>{lg[k]}</b>' for k in ("bias_e", "bias_i")
                        if lg.get(k)]
                stat = ('<div class="stats">' + " &nbsp; ".join(bits) + '</div>'
                        if bits else "")
            cells.append(
                f'<div class="cell"><div class="armname"><span>'
                f'{html.escape(arm)}</span><span class="w">'
                f'{html.escape(wtxt)}</span></div>'
                + "".join(figparts) + stat + '</div>')

        fp_txt = ""
        anyr = next(iter(byarm.values()))
        fpv = anyr["params"].get("fingerprint", "")
        if fpv:
            fp_txt = f'fp {html.escape(str(fpv))}'
        parts.append(
            f'<section class="rowgroup" data-config="{html.escape(cname)}" '
            f'data-order="{cname}|{seed:04d}" data-mean_E="{r_mean}" '
            f'data-frac_high="{r_high}">'
            f'<div class="rowhead"><span class="id">{html.escape(cname)} '
            f'&nbsp;seed {seed:03d}</span><span class="st">{fp_txt}</span></div>'
            f'<div class="arms" style="grid-template-columns:'
            f'repeat({max(1,len(arms))},1fr)">' + "".join(cells)
            + '</div></section>')
        if verbose and (gi + 1) % 10 == 0:
            print(f"  rows: {gi+1}/{len(order_keys)} "
                  f"({total_bytes/1e6:.1f} MB of thumbnails)", flush=True)

    cfg_opts = "".join(f'<option value="{html.escape(c)}">{html.escape(c)}</option>'
                       for c in configs)
    sort_opts = ('<option value="natural">config, then seed</option>'
                 '<option value="frac_high">saturation (worst arm first)</option>'
                 '<option value="mean_E">mean E (highest first)</option>')
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<header><h1>{html.escape(title)}</h1>
<div class="meta">{len(order_keys)} seed&times;config rows &nbsp;&middot;&nbsp;
{len(runs)} runs &nbsp;&middot;&nbsp; arms: {html.escape(", ".join(arms) or "(unarmed)")}
&nbsp;&middot;&nbsp; <span id="count"></span></div>
<div class="controls">
<label>config <select id="f-config"><option value="*">all</option>{cfg_opts}</select></label>
<label>figure <select id="f-fig"><option value="*">both</option>
<option value="lfp.png">LFP only</option>
<option value="heatmap_E.png">heatmap only</option></select></label>
<label>sort <select id="f-sort">{sort_opts}</select></label>
</div></header>
<p class="warn"><b>Thumbnails are downscaled JPEGs &mdash; not calibration-grade.</b>
Do not read pixel deflections off them or set detector thresholds from them.
Click any panel to enlarge, or use the full-res link to open the original PNG.</p>
<main id="main">{''.join(parts)}</main>
<div id="lb"><img><div class="cap"></div></div>
<script>{JS}</script></body></html>"""
    return doc, dict(rows=len(order_keys), runs=len(runs), arms=arms,
                     configs=configs, thumb_mb=total_bytes / 1e6,
                     missing=n_missing)


# --------------------------------------------------------------------------
def main(argv=None):
    global THUMB_W
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True,
                    help="sweep root (holds param_explore_log.csv)")
    ap.add_argument("--html", default="contact_sheet.html",
                    help="output filename, written under --out-root")
    ap.add_argument("--dynamics", action="store_true",
                    help="read E_traj per run for mean_E / saturation stats "
                         "(~31 MB/run; cached to contact_sheet_dynamics.csv)")
    ap.add_argument("--config", default=None,
                    help="restrict to one config_name")
    ap.add_argument("--arm", default=None, help="restrict to one arm")
    ap.add_argument("--max-seed", type=int, default=None,
                    help="only seeds <= this (keeps the file small on big sweeps)")
    ap.add_argument("--thumb-width", type=int, default=THUMB_W)
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)

    THUMB_W = int(a.thumb_width)

    out_root = Path(a.out_root)
    if not out_root.exists():
        raise SystemExit(f"out_root does not exist: {out_root}")

    print(f"scanning {out_root} ...")
    runs = discover_runs(out_root)
    if a.config:
        runs = [r for r in runs if r["config_name"] == a.config]
    if a.arm:
        runs = [r for r in runs if r["arm"] == a.arm]
    if a.max_seed is not None:
        runs = [r for r in runs if r["seed"] <= a.max_seed]
    if not runs:
        raise SystemExit("no run directories found (looked for */manifest.json "
                         "under run_*/). Has the sweep written anything yet?")
    print(f"  {len(runs)} runs")

    log_idx = read_master_log(out_root)
    print(f"  master log rows: {len(log_idx)}")

    dyn = {}
    if a.dynamics:
        print("computing dynamics stats (reads E_traj; cached) ...")
        dyn = compute_dynamics(runs, out_root)

    title = a.title or f"WC-MNIST param-explore contact sheet — {out_root.name}"
    print("building html ...")
    doc, info = build_html(runs, log_idx, dyn, out_root, title)
    dest = out_root / a.html
    dest.write_text(doc, encoding="utf-8")
    size_mb = dest.stat().st_size / 1e6
    print(f"\nwrote {dest}")
    print(f"  {info['rows']} rows x {len(info['arms']) or 1} arms, "
          f"{info['runs']} runs")
    print(f"  thumbnails {info['thumb_mb']:.1f} MB -> file {size_mb:.1f} MB")
    if info["missing"]:
        print(f"  [note] {info['missing']} missing figures/runs marked in red")
    # Measured 31 KB/run at --thumb-width 460, so the 90-run pilot is ~3 MB but
    # the full 4500-run sweep would be ~140 MB in a single file.
    if size_mb > 40:
        print(f"  [warn] {size_mb:.0f} MB is large for one html; browsers get "
              "sluggish past ~50 MB and may refuse past ~150 MB.\n"
              "         Split it: --config <name>, --max-seed <n>, or a smaller "
              "--thumb-width.")
    return dest


if __name__ == "__main__":
    main()
