"""timing_calibration.py — Acceptance Test A5: wall-clock projection.

Times the cost of one epoch's components on the target device — training,
validation, the clean-gradient probe, and one topology measurement — then
projects the wall-clock for the full study (100 epochs x {15, 60} runs).

Topology runs only on measurement epochs (§8: epoch 0 + every 4th + sign-flips),
so it is projected with an estimated measurement-epoch count rather than all 100.

Gate the full launch on this (§17): confirm the 60-run budget is acceptable
before scaling.

CLI:
    python timing_calibration.py --device cuda
    python timing_calibration.py --device cpu --no-topology      # quick CPU sanity
    python timing_calibration.py --device cuda --topo-epochs 30
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

import wc_config as cfg
import wc_model as M
import wc_experiment as E


def _time(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def calibrate(config_name: str, device: str, data_root: str | None,
              measure_topology: bool, topo_epochs: int,
              images_per_epoch: int | None, eval_images: int | None) -> dict:
    config = cfg.make_config(config_name)
    if images_per_epoch is not None:
        config = __import__("dataclasses").replace(config, images_per_epoch=images_per_epoch)
    if eval_images is not None:
        config = __import__("dataclasses").replace(config, eval_images=eval_images,
                                                   grad_probe_images=eval_images)
    dev = torch.device(device)

    E.seed_everything(0)
    init_rng, config_rng = E.make_rngs(0)
    alpha_H, tau_H = E.resolve_alpha_tau(config, config_rng)
    model = E.build_model(0, config, dev, init_rng, alpha_H, tau_H)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader, g, fixed_eval_loader, fixed_eval_batch = E.build_data(0, config, dev, data_root)

    H0 = init_rng.uniform(cfg.H_INIT_MIN, cfg.H_INIT_MAX, size=model.N).astype("float32")
    state = {"E": torch.zeros(model.N, device=dev), "I": torch.zeros(model.N, device=dev),
             "O": torch.zeros(model.num_output, device=dev), "H": torch.tensor(H0, device=dev),
             "global_train_time_s": 0.0}
    boundary = cfg.get_boundary(config.name, alpha_H)

    print(f"Calibrating: config={config_name} device={device} "
          f"images/epoch={config.images_per_epoch} eval={config.eval_images} "
          f"image_duration={config.image_duration}s ({int(config.image_duration/config.dt)} steps/img)")

    t_train, _ = _time(E._train_one_epoch, model, train_loader, optimizer, criterion,
                       config, state, capture_activity=True)
    t_val, _ = _time(M.evaluate, model, fixed_eval_loader, criterion,
                     dt=config.dt, image_duration=config.image_duration,
                     output_average_window=config.output_average_window,
                     max_images=config.eval_images, verbose=False,
                     E_init=state["E"], I_init=state["I"], O_init=state["O"], H_init=state["H"],
                     sequential_row_input=config.sequential_row_input,
                     row_duration=config.row_duration if config.sequential_row_input else None,
                     mnist_input_scale=config.mnist_input_scale)
    t_probe, _ = _time(E.clean_gradient_probe, model, criterion, fixed_eval_batch, config, boundary)

    t_topo = 0.0
    if measure_topology:
        import wc_topology as T
        t_topo, _ = _time(T.measure_topology, model, criterion, fixed_eval_batch, dev,
                          seq_input=config.sequential_row_input, img_dur=config.image_duration,
                          dt_sim=config.dt, scale=config.mnist_input_scale,
                          grid_size=cfg.TOPO_GRID_SIZE, perturbation_range=cfg.TOPO_PERTURBATION_RANGE,
                          num_samples=cfg.TOPO_NUM_SAMPLES, hessian_num_samples=cfg.TOPO_HESSIAN_NUM_SAMPLES,
                          lanczos_iterations=cfg.TOPO_LANCZOS_ITERATIONS, verbose=False)

    return dict(t_train=t_train, t_val=t_val, t_probe=t_probe, t_topo=t_topo,
                n_epochs=cfg.NUM_EPOCHS, topo_epochs=topo_epochs)


def project(timings: dict):
    per_epoch = timings["t_train"] + timings["t_val"] + timings["t_probe"]
    n_epochs = timings["n_epochs"]
    topo_total = timings["t_topo"] * timings["topo_epochs"]
    per_run = per_epoch * n_epochs + topo_total

    print("\nPer-component wall-clock (one epoch):")
    print(f"  train epoch       : {timings['t_train']:8.2f} s")
    print(f"  validation        : {timings['t_val']:8.2f} s")
    print(f"  clean-grad probe  : {timings['t_probe']:8.2f} s")
    print(f"  topology (1 meas) : {timings['t_topo']:8.2f} s "
          f"(x{timings['topo_epochs']} measurement epochs)")
    print(f"\nPer run ({n_epochs} epochs + {timings['topo_epochs']} topology measurements): "
          f"{per_run/60:.1f} min")
    print("\nProjected wall-clock (sequential, one device):")
    for n_runs in (15, 60):
        total_h = per_run * n_runs / 3600.0
        print(f"  {n_runs:2d} runs : {total_h:6.2f} h  ({total_h/24:.2f} days)")
    print("\nNote: sign-flip triggers add topology measurements beyond the scheduled "
          "count; re-run with --topo-epochs to bound the worst case.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sequential_H")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--no-topology", action="store_true")
    ap.add_argument("--topo-epochs", type=int, default=26,
                    help="estimated #measurement epochs/run (scheduled ~26; add sign-flips)")
    ap.add_argument("--images", type=int, default=None, help="override images/epoch (CPU sanity)")
    ap.add_argument("--eval-images", type=int, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    timings = calibrate(args.config, device, args.data_root,
                        not args.no_topology, args.topo_epochs,
                        args.images, args.eval_images)
    project(timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
