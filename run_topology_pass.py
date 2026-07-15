"""run_topology_pass.py — Deferred topology measurement (§8).

Second phase of the split study: after a **topology-off** training pass
(``run_study.py --no-topology``), this reloads the checkpoints that were archived
at each topology-trigger epoch (``checkpoints/topology/``) and measures the
loss-landscape topology on them, merging the results back into each run's
``metrics.parquet`` / ``metrics.csv``.

Because runs are deterministic and the archived checkpoint holds the exact model
state topology would have been measured on inline, this reproduces the same
topology as an all-in-one run — but lets you defer the expensive measurement and
run it only on the seeds you care about.

Needs ``gudhi`` (persistence) and SciPy (Hessian eigenvalues); without ``gudhi``
the loss-grid and eigenvalue fields are still written and the persistence fields
come back null (see ``wc_topology``).

CLI:
    python run_topology_pass.py --out-root DIR                 # all runs with archives
    python run_topology_pass.py --out-root DIR --runs seed000__sequential_H
    python run_topology_pass.py --out-root DIR --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import wc_config as cfg
from wc_experiment import measure_topology_for_run


def run_topology_pass(out_root: str, runs=None, device=None,
                      data_root: str | None = None, verbose: bool = True) -> list[dict]:
    """Measure topology for the given runs (or every run under ``out_root`` that
    has an archived-checkpoint dir).  Returns a list of per-run result dicts."""
    out_root = Path(out_root)
    if runs:
        run_dirs = [out_root / r for r in runs]
    else:
        run_dirs = [d for d in sorted(out_root.iterdir())
                    if d.is_dir() and (d / "checkpoints" / "topology").exists()]

    results = []
    for rd in run_dirs:
        res = measure_topology_for_run(rd, device=device, data_root=data_root, verbose=verbose)
        results.append(res)
        if verbose:
            print(f"[{rd.name}] measured {res['n_measured']} epoch(s): {res['measured_epochs']}")
    total = sum(r["n_measured"] for r in results)
    if verbose:
        print(f"\nTopology pass complete: {total} measurement(s) across {len(results)} run(s).")
    return results


def _parse_runs(spec: str | None):
    if not spec:
        return None
    return [s.strip() for s in spec.split(",") if s.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Deferred topology pass over archived checkpoints.")
    ap.add_argument("--out-root", default=cfg.OUT_ROOT)
    ap.add_argument("--runs", default=None,
                    help="comma-separated run_ids (default: all runs with archived checkpoints)")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--data-root", default=None,
                    help="MNIST root (default: $WC_DATA_ROOT or ./mnist_data)")
    args = ap.parse_args()

    run_topology_pass(out_root=args.out_root, runs=_parse_runs(args.runs),
                      device=args.device, data_root=args.data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
