"""run_study.py — Driver for the whole longitudinal study (§3.4 / §13).

For each seed in ``[start_seed, start_seed + n_sets)`` and each config, call
``run_experiment``.  Runs execute **sequentially** (one GPU per session, §13).
Fully resumable: completed runs (``DONE`` sentinel) are skipped and partial runs
continue from their latest checkpoint.

Real parallelism = launch multiple sessions over **disjoint** seed ranges; each
writes to disjoint ``run_id``s under the same ``out_root`` and never collides.

CLI:
    python run_study.py --n-sets 5                  # seeds 0..4, all 3 configs
    python run_study.py --n-sets 5 --start-seed 5   # second session, seeds 5..9
    python run_study.py --n-sets 1 --configs sequential_H --epochs 100
"""

from __future__ import annotations

import argparse

import wc_config as cfg
from wc_experiment import run_experiment


def run_study(n_sets: int, start_seed: int = 0, configs=None, n_epochs: int = 100,
              out_root: str = cfg.OUT_ROOT, device=None, data_root: str | None = None,
              measure_topology: bool = True, save_figures: bool = True,
              verbose: bool = True) -> list:
    """Run ``n_sets`` sets sequentially.  Returns the list of RunResult objects.

    Re-running is safe: each (seed, config) skips if already DONE, otherwise
    resumes from its latest checkpoint.
    """
    if configs is None:
        configs = cfg.make_configs()

    results = []
    total = n_sets * len(configs)
    done = 0
    for seed in range(start_seed, start_seed + n_sets):
        for config in configs:
            done += 1
            if verbose:
                print(f"\n{'#' * 70}\n# [{done}/{total}] seed={seed} config={config.name}\n{'#' * 70}")
            res = run_experiment(
                seed, config, n_epochs=n_epochs, resume=True,
                out_root=out_root, device=device, data_root=data_root,
                measure_topology_flag=measure_topology, save_figures=save_figures,
                verbose=verbose,
            )
            results.append(res)
    if verbose:
        n_done = sum(1 for r in results if r.done)
        print(f"\nStudy complete: {n_done}/{len(results)} runs DONE under {out_root}")
    return results


def _parse_configs(spec: str | None):
    if not spec:
        return None
    names = [s.strip() for s in spec.split(",") if s.strip()]
    return [cfg.make_config(n) for n in names]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the WC-MNIST longitudinal study.")
    ap.add_argument("--n-sets", type=int, required=True, help="number of seeds (sets) to run")
    ap.add_argument("--start-seed", type=int, default=0, help="first seed (for disjoint ranges)")
    ap.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    ap.add_argument("--out-root", default=cfg.OUT_ROOT)
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--data-root", default=None, help="MNIST root (default: $WC_DATA_ROOT or ./mnist_data)")
    ap.add_argument("--configs", default=None,
                    help="comma-separated subset of config names (default: all 3)")
    ap.add_argument("--no-topology", action="store_true", help="skip topology measurement")
    ap.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = ap.parse_args()

    run_study(
        n_sets=args.n_sets, start_seed=args.start_seed,
        configs=_parse_configs(args.configs), n_epochs=args.epochs,
        out_root=args.out_root, device=args.device, data_root=args.data_root,
        measure_topology=not args.no_topology, save_figures=not args.no_figures,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
