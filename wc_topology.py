"""wc_topology.py — Loss-landscape topology measurement (notebook cell 16).

Focused port of the dual-method (domain + Hessian) loss-landscape topology
analysis.  Produces exactly the metrics in the §7 Parquet schema:

    domain_n_saddle, domain_avg_persistence, domain_max_persistence,
    domain_loss_min, domain_loss_max, domain_loss_range, domain_loss_std,
    hessian_n_saddle, hessian_avg_persistence, hessian_max_persistence,
    hessian_eig1, hessian_eig2

The notebook's richer v79/v82 extras (E-I balance scan, eigenvector bias
projection, per-parameter energy decomposition, evolution plots) are *not* part
of the longitudinal schema and are omitted here.

Heavy dependencies are imported lazily:
  * ``gudhi``  — cubical persistence (domain & Hessian metrics).
  * ``scipy.sparse.linalg.eigsh`` — top-2 Hessian eigenvalues/vectors.
If ``gudhi`` is unavailable the loss-grid statistics and Hessian eigenvalues are
still returned; persistence fields come back as ``None`` with a printed note.
This lets the harness run end-to-end off-Colab while the full topology is
computed where the libraries are installed.

Loss is evaluated with the faithful per-sample ``model.forward`` path (identical
dynamics to training).  The notebook's v80 batched evaluator is a pure
performance optimization with the same semantics; topology values are analysis
metadata and are not covered by Acceptance Tests A1/A2.
"""

from __future__ import annotations

import numpy as np
import torch

import wc_model as M
import wc_config as cfg


# =============================================================================
# Parameter <-> flat-vector helpers (cell 16, verbatim)
# =============================================================================
def params_to_vector(model) -> torch.Tensor:
    return torch.cat([p.data.flatten() for p in model.parameters() if p.requires_grad])


def vector_to_params(model, vector: torch.Tensor):
    idx = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.data.copy_(vector[idx:idx + numel].view_as(p))
            idx += numel


# =============================================================================
# Loss-at-current-params on a fixed batch (faithful per-sample forward)
# =============================================================================
def _batch_images_labels(fixed_batch, num_samples, device):
    n = min(num_samples, len(fixed_batch))
    images = torch.stack([img for img, _ in fixed_batch[:n]]).to(device)            # [B, 784]
    labels = torch.stack([torch.as_tensor(lbl) for _, lbl in fixed_batch[:n]]).to(device)
    return images, labels


def _loss_at_point(model, criterion, fixed_batch, device, *,
                   seq_input, img_dur, dt_sim, scale, num_samples):
    """Mean cross-entropy over ``num_samples`` of ``fixed_batch`` at current params.

    Evaluated as a single [B, N] batch (``simulate_window_batched``) rather than
    one image at a time — this is the per-grid-point hot path and the dominant
    cost of topology, so batching it is the key speedup (the original notebook
    used the same batched evaluator)."""
    model.eval()
    images, labels = _batch_images_labels(fixed_batch, num_samples, device)
    with torch.no_grad():
        avg = M.simulate_window_batched(model, images, dt_sim, img_dur, cfg.OUTPUT_AVERAGE_WINDOW,
                                        sequential_row_input=seq_input, row_duration=img_dur / 28,
                                        mnist_input_scale=scale)
        loss = criterion(avg, labels).item()
    model.train()
    return loss


def _grad_at_point(model, criterion, fixed_batch, device, *,
                   seq_input, img_dur, dt_sim, scale, num_samples):
    """Flat gradient of mean CE w.r.t. all trainable params (for Hessian HVP),
    evaluated on a single [B, N] batch."""
    params = [p for p in model.parameters() if p.requires_grad]
    images, labels = _batch_images_labels(fixed_batch, num_samples, device)
    model.zero_grad(set_to_none=True)
    avg = M.simulate_window_batched(model, images, dt_sim, img_dur, cfg.OUTPUT_AVERAGE_WINDOW,
                                    sequential_row_input=seq_input, row_duration=img_dur / 28,
                                    mnist_input_scale=scale)
    loss = criterion(avg, labels)
    loss.backward()
    grad = torch.cat([p.grad.flatten() for p in params]).detach().clone()
    model.zero_grad(set_to_none=True)
    return grad


# =============================================================================
# Top-2 Hessian eigenvectors via Lanczos + finite-difference HVP (cell 16)
# =============================================================================
def compute_top_hessian_eigenvectors(model, criterion, fixed_batch, device, *,
                                     seq_input, img_dur, dt_sim, scale,
                                     num_samples, top_k=2,
                                     lanczos_iterations=50, epsilon=1e-4,
                                     max_matvec=300):
    from scipy.sparse.linalg import LinearOperator, eigsh

    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)

    def hessian_vector_product(v_np):
        v = torch.from_numpy(v_np).float().to(device)

        def grad_with_delta(sign):
            originals = [p.data.clone() for p in params]
            try:
                idx = 0
                for p in params:
                    numel = p.numel()
                    p.data.add_(sign * epsilon * v[idx:idx + numel].view_as(p))
                    idx += numel
                return _grad_at_point(model, criterion, fixed_batch, device,
                                      seq_input=seq_input, img_dur=img_dur,
                                      dt_sim=dt_sim, scale=scale, num_samples=num_samples)
            finally:
                for p, orig in zip(params, originals):
                    p.data.copy_(orig)
                model.zero_grad(set_to_none=True)

        gp = grad_with_delta(+1.0).cpu().numpy()
        gm = grad_with_delta(-1.0).cpu().numpy()
        return (gp - gm) / (2 * epsilon)

    H_op = LinearOperator((num_params, num_params), matvec=hessian_vector_product)
    # Bound the iteration count.  The notebook used maxiter=num_params (~2.5e5),
    # which can spin for minutes when the loss landscape is near-flat (degenerate
    # Hessian).  A bounded maxiter makes eigsh raise on non-convergence; the
    # caller catches it and records the topology fields as None for that epoch.
    eigvals, eigvecs = eigsh(H_op, k=top_k, which="LA",
                             ncv=min(lanczos_iterations, num_params),
                             tol=1e-3, maxiter=max_matvec)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    vectors = [torch.from_numpy(eigvecs[:, i]).float().to(device) for i in range(top_k)]
    return vectors, eigvals.tolist()


# =============================================================================
# Persistence metrics from a 2-D loss grid (cell 16, verbatim logic)
# =============================================================================
def compute_persistence_metrics(loss_grid: np.ndarray) -> dict:
    """0-dim cubical persistence via gudhi.  Returns persistence fields = None
    if gudhi is unavailable (loss stats are still computed)."""
    loss_min = float(loss_grid.min())
    loss_max = float(loss_grid.max())
    out = {
        "loss_min": loss_min,
        "loss_max": loss_max,
        "loss_range": loss_max - loss_min,
        "loss_std": float(loss_grid.std()),
        "n_saddle": None,
        "avg_persistence": None,
        "max_persistence": None,
    }
    try:
        import gudhi
    except ImportError:
        return out

    cc = gudhi.CubicalComplex(top_dimensional_cells=loss_grid.flatten(),
                              dimensions=loss_grid.shape)
    cc.compute_persistence()
    pairs = [(b, d) for b, d in cc.persistence_intervals_in_dimension(0) if not np.isinf(d)]
    if not pairs:
        out.update(n_saddle=0, avg_persistence=0.0, max_persistence=0.0)
    else:
        persistences = [d - b for b, d in pairs]
        out.update(n_saddle=len(pairs),
                   avg_persistence=float(np.mean(persistences)),
                   max_persistence=float(np.max(persistences)))
    return out


# =============================================================================
# Domain + Hessian landscape sampling
# =============================================================================
def _sample_domain_grid(model, criterion, fixed_batch, device, *,
                        grid_size, perturbation_range, num_samples,
                        seq_input, img_dur, dt_sim, scale):
    be = dict(model.named_parameters())["bias_e"]
    bi = dict(model.named_parameters())["bias_i"]
    be0 = be.item()
    bi0 = bi.item()
    be_vals = np.linspace(be0 - perturbation_range, be0 + perturbation_range, grid_size)
    bi_vals = np.linspace(bi0 - perturbation_range, bi0 + perturbation_range, grid_size)
    grid = np.zeros((grid_size, grid_size))
    try:
        for i, p1 in enumerate(be_vals):
            for j, p2 in enumerate(bi_vals):
                with torch.no_grad():
                    be.fill_(float(p1))
                    bi.fill_(float(p2))
                grid[j, i] = _loss_at_point(model, criterion, fixed_batch, device,
                                            seq_input=seq_input, img_dur=img_dur,
                                            dt_sim=dt_sim, scale=scale, num_samples=num_samples)
    finally:
        with torch.no_grad():
            be.fill_(be0)
            bi.fill_(bi0)
    return grid


def _sample_hessian_grid(model, criterion, fixed_batch, device, *,
                         grid_size, perturbation_range, num_samples,
                         hessian_num_samples, seq_input, img_dur, dt_sim, scale,
                         lanczos_iterations):
    vectors, eigvals = compute_top_hessian_eigenvectors(
        model, criterion, fixed_batch, device,
        seq_input=seq_input, img_dur=img_dur, dt_sim=dt_sim, scale=scale,
        num_samples=hessian_num_samples, top_k=2,
        lanczos_iterations=lanczos_iterations,
    )
    d1, d2 = vectors[0], vectors[1]
    theta0 = params_to_vector(model)
    a_vals = np.linspace(-perturbation_range, perturbation_range, grid_size)
    grid = np.zeros((grid_size, grid_size))
    try:
        for i, a1 in enumerate(a_vals):
            for j, a2 in enumerate(a_vals):
                vector_to_params(model, theta0 + float(a1) * d1 + float(a2) * d2)
                grid[j, i] = _loss_at_point(model, criterion, fixed_batch, device,
                                            seq_input=seq_input, img_dur=img_dur,
                                            dt_sim=dt_sim, scale=scale, num_samples=num_samples)
    finally:
        vector_to_params(model, theta0)
    return grid, eigvals


# =============================================================================
# Public entry point
# =============================================================================
def measure_topology(model, criterion, fixed_batch, device, *,
                     seq_input, img_dur, dt_sim, scale,
                     grid_size, perturbation_range, num_samples,
                     hessian_num_samples, lanczos_iterations,
                     verbose=True, first_measurement=False):
    """Measure domain + Hessian topology; return the §7 topology fields (flat dict).

    On any failure (e.g. Lanczos non-convergence, gudhi missing) the affected
    fields are returned as ``None`` and a note is printed; the harness records
    the row regardless so the per-epoch table stays rectangular.
    """
    result = dict(
        domain_n_saddle=None, domain_avg_persistence=None, domain_max_persistence=None,
        domain_loss_min=None, domain_loss_max=None, domain_loss_range=None, domain_loss_std=None,
        hessian_n_saddle=None, hessian_avg_persistence=None, hessian_max_persistence=None,
        hessian_eig1=None, hessian_eig2=None,
    )

    # ---- Domain method ----
    try:
        dgrid = _sample_domain_grid(
            model, criterion, fixed_batch, device,
            grid_size=grid_size, perturbation_range=perturbation_range,
            num_samples=num_samples, seq_input=seq_input, img_dur=img_dur,
            dt_sim=dt_sim, scale=scale,
        )
        dm = compute_persistence_metrics(dgrid)
        result.update(
            domain_n_saddle=dm["n_saddle"],
            domain_avg_persistence=dm["avg_persistence"],
            domain_max_persistence=dm["max_persistence"],
            domain_loss_min=dm["loss_min"], domain_loss_max=dm["loss_max"],
            domain_loss_range=dm["loss_range"], domain_loss_std=dm["loss_std"],
        )
        if verbose:
            print(f"    [topology] domain: n_saddle={dm['n_saddle']} "
                  f"loss_range={dm['loss_range']:.4f}")
    except Exception as e:
        print(f"    [topology] domain method failed: {e}")

    # ---- Hessian method ----
    try:
        hgrid, eigvals = _sample_hessian_grid(
            model, criterion, fixed_batch, device,
            grid_size=grid_size, perturbation_range=perturbation_range,
            num_samples=num_samples, hessian_num_samples=hessian_num_samples,
            seq_input=seq_input, img_dur=img_dur, dt_sim=dt_sim, scale=scale,
            lanczos_iterations=lanczos_iterations,
        )
        hm = compute_persistence_metrics(hgrid)
        result.update(
            hessian_n_saddle=hm["n_saddle"],
            hessian_avg_persistence=hm["avg_persistence"],
            hessian_max_persistence=hm["max_persistence"],
            hessian_eig1=float(eigvals[0]), hessian_eig2=float(eigvals[1]),
        )
        if verbose:
            print(f"    [topology] hessian: eig1={eigvals[0]:.4f} eig2={eigvals[1]:.4f} "
                  f"n_saddle={hm['n_saddle']}")
    except Exception as e:
        print(f"    [topology] hessian method failed: {e}")

    # First-measurement diagnostic + hard-zero tripwire (spec §8).  The routine
    # print above shows loss_range at 4 dp, so a tiny-nonzero range (~1e-4, normal
    # for the sequential chance plateau where only 28/784 neurons are driven)
    # renders as 0.0000 and is indistinguishable from a genuine hard zero.  On a
    # run's first measurement, re-print the key domain/Hessian values in full
    # precision and WARN (never raise) if the domain loss_range is *exactly* 0.0 —
    # that signals the +/-bias perturbation isn't reaching bias_e/bias_i or the
    # fixed eval batch is constant, not merely a flat landscape.
    if first_measurement:
        # Show floats in scientific notation (spec §8) so a ~1e-4 range is clearly
        # distinct from a hard 0.0; pass None through unchanged, since any method
        # that failed above leaves its fields None.
        def _fp(x):
            return f"{x:.6e}" if isinstance(x, float) else repr(x)
        lr = result["domain_loss_range"]
        print(f"    [topology] first-measurement diagnostic (full precision): "
              f"domain_loss_range={_fp(lr)} "
              f"domain_loss_std={_fp(result['domain_loss_std'])} "
              f"hessian_eig1={_fp(result['hessian_eig1'])}")
        if lr == 0.0:
            print("    [topology] WARNING: domain loss_range is EXACTLY 0.0 — the "
                  "bias_e/bias_i perturbation may not be applied, or the fixed eval "
                  "batch is constant. Spot-check: set bias_e to two very different "
                  "values and confirm the loss changes in this input mode. "
                  "(A tiny-nonzero range printing as 0.0000 at 4 dp is normal and "
                  "does NOT trip this.)")

    return result
