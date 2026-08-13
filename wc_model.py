"""wc_model.py — Verbatim port of the v82 Wilson-Cowan model and train/eval helpers.

Ported from the v82 notebook:
  * ``SparseBigWilsonCowan``         — cell 14 (dynamics frozen; see A2 / model_diff_check.py)
  * ``build_connectivity``           — cells 12-13 (small-world CeLoc/CeRem/CeLocI)
  * ``train_one_image`` / ``train_epoch`` / ``evaluate`` — cell 22

DYNAMICS ARE FROZEN.  ``SparseBigWilsonCowan.step`` and ``.forward`` reproduce
the cell-14 math coefficient-for-coefficient.  The only deviation from the
notebook is RNG plumbing: every stochastic initialization draw is routed through
an explicit ``numpy.random.Generator`` passed in by the caller, instead of the
notebook's reliance on globally-seeded ``np.random``/``torch`` state.  This
removes cell-order fragility and guarantees (a) bit-exact reproducibility within
a (seed, config) and (b) identical initialization across the three configs of a
set (engineering spec §5).  Distributions, ranges, shapes and the draw *content*
are unchanged — only the stream source differs.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

import wc_config as cfg


# =============================================================================
# Connectivity construction (notebook cells 12-13)
# =============================================================================
def generate_ring_lattice_directed(N: int, k: int) -> sp.csr_matrix:
    """Directed ring-lattice on N nodes with out-degree k (cell 12, verbatim)."""
    assert k % 2 == 0, "k must be even for the ring-lattice construction."
    rows = []
    cols = []
    data = []
    half_k = k // 2

    for i in range(N):
        for offset in range(1, half_k + 1):
            j_fwd = (i + offset) % N
            j_back = (i - offset) % N

            # forward edge
            rows.append(j_fwd)   # target index
            cols.append(i)       # source index
            data.append(1.0)

            # backward edge
            rows.append(j_back)
            cols.append(i)
            data.append(1.0)

    mat = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
    mat.sum_duplicates()
    return mat


def generate_ws_directed_csr(N: int, k: int, p_rewire: float, rng) -> sp.csr_matrix:
    """Directed Watts-Strogatz small-world network (cell 12, verbatim except rng).

    The notebook used a hard-coded ``np.random.default_rng(seed=12345)``; per spec
    §5 the rewiring stream is now the seed-derived ``init_rng`` passed in here.
    """
    base = generate_ring_lattice_directed(N, k)
    rows, cols = base.nonzero()
    rows_ws = rows.astype(np.int64).copy()
    cols_ws = cols.astype(np.int64).copy()

    for idx in range(len(rows_ws)):
        if rng.random() < p_rewire:
            src = cols_ws[idx]
            # draw a new target different from the source
            while True:
                j_new = int(rng.integers(0, N))
                if j_new != src:
                    rows_ws[idx] = j_new
                    break

    data_ws = np.ones_like(rows_ws, dtype=np.float32)
    mat_ws = sp.csr_matrix((data_ws, (rows_ws, cols_ws)), shape=(N, N), dtype=np.float32)
    mat_ws.sum_duplicates()
    return mat_ws


def build_connectivity_csr(rng, p_rewire_pypy: float, p_rewire_pyinh: float,
                           N: int = cfg.N,
                           k_pypy: int = cfg.TARGET_OUTDEG_PyPy,
                           k_pyinh: int = cfg.TARGET_OUTDEG_PyInh):
    """Build (CeLoc_csr, CeRem_csr, CeLocI_csr) small-world matrices (cell 12).

    ``rng`` is the seed-derived init Generator.  Draw order (fixed, config-
    independent): CeLoc/CeRem split mask -> remote-edge targets -> CeLocI rewiring.
    """
    K_PyPy = int(2 * round(k_pypy / 2))
    K_PyInh = int(2 * round(k_pyinh / 2))

    # ---- Py->Py connectivity = CeLoc (local) + CeRem (long-range) ----
    ring = generate_ring_lattice_directed(N, K_PyPy)
    rows, cols = ring.nonzero()
    n_edges = len(rows)

    mask_rem = rng.random(n_edges) < p_rewire_pypy

    # CeLoc: edges that remain local
    rows_loc = rows[~mask_rem]
    cols_loc = cols[~mask_rem]
    data_loc = np.ones_like(rows_loc, dtype=np.float32)
    CeLoc_csr = sp.csr_matrix((data_loc, (rows_loc, cols_loc)), shape=(N, N), dtype=np.float32)
    CeLoc_csr.eliminate_zeros()

    # CeRem: rewired edges
    rows_rem = rows[mask_rem].astype(np.int64).copy()
    cols_rem = cols[mask_rem].astype(np.int64).copy()

    for idx in range(len(rows_rem)):
        src = cols_rem[idx]
        while True:
            new_row = int(rng.integers(0, N))
            if new_row != src:
                rows_rem[idx] = new_row
                break

    data_rem = np.ones_like(rows_rem, dtype=np.float32)
    CeRem_csr = sp.csr_matrix((data_rem, (rows_rem, cols_rem)), shape=(N, N), dtype=np.float32)
    CeRem_csr.sum_duplicates()

    # ---- Py->Inh connectivity as a separate small-world network ----
    CeLocI_csr = generate_ws_directed_csr(N, K_PyInh, p_rewire_pyinh, rng)

    return CeLoc_csr, CeRem_csr, CeLocI_csr


def csr_to_torch_sparse(csr, device):
    """Convert a SciPy CSR matrix to a torch.sparse_csr_tensor (cell 13, verbatim).

    This CSR tensor is only used at init to extract crow/col/value arrays; it is
    never used in an autograd matmul (the per-step operator is rebuilt as COO in
    ``_get_sparse_matrix``).  The PyTorch "beta state" notice is therefore benign
    and suppressed to keep run logs clean.
    """
    import warnings
    indptr = torch.from_numpy(csr.indptr.astype(np.int64))
    indices = torch.from_numpy(csr.indices.astype(np.int64))
    data = torch.from_numpy(csr.data.astype(np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.sparse_csr_tensor(indptr, indices, data, size=csr.shape, device=device)


def build_connectivity(rng, p_rewire_pypy: float, p_rewire_pyinh: float, device,
                       N: int = cfg.N):
    """Convenience: build small-world connectivity as torch sparse CSR tensors."""
    CeLoc_csr, CeRem_csr, CeLocI_csr = build_connectivity_csr(
        rng, p_rewire_pypy, p_rewire_pyinh, N=N
    )
    CeLoc = csr_to_torch_sparse(CeLoc_csr, device)
    CeRem = csr_to_torch_sparse(CeRem_csr, device)
    CeLocI = csr_to_torch_sparse(CeLocI_csr, device)
    return CeLoc, CeRem, CeLocI


# Default initialization ranges (cell 7).  A2 verifies these match the notebook.
DEFAULT_INIT_RANGES = dict(
    taupy=(cfg.TAUPY_MIN, cfg.TAUPY_MAX),
    tauinh=(cfg.TAUINH_MIN, cfg.TAUINH_MAX),
    tauout=(cfg.TAUOUT_MIN, cfg.TAUOUT_MAX),
    bias_e=(cfg.BIAS_E_INIT_MIN, cfg.BIAS_E_INIT_MAX),
    bias_i=(cfg.BIAS_I_INIT_MIN, cfg.BIAS_I_INIT_MAX),
    mnist_weight=(cfg.MNIST_WEIGHT_MIN, cfg.MNIST_WEIGHT_MAX),
    inh2py_gain=(cfg.INH2PY_GAIN_MIN, cfg.INH2PY_GAIN_MAX),
    celoc_scale=(cfg.CELOC_INIT_SCALE_MIN, cfg.CELOC_INIT_SCALE_MAX),
    cerem_scale=(cfg.CEREM_INIT_SCALE_MIN, cfg.CEREM_INIT_SCALE_MAX),
    celoci_scale=(cfg.CELOCI_INIT_SCALE_MIN, cfg.CELOCI_INIT_SCALE_MAX),
)


# =============================================================================
# Model (notebook cell 14) — DYNAMICS FROZEN
# =============================================================================
def active_row_at(step, dt, image_duration, row_duration=None, n_rows=28):
    """Which image row is active at Euler `step`, on the canonical schedule.

    row_duration/dt need not be a whole number of steps (for the standard
    250 ms image and dt=2 ms it is 4.4643).  Quantising it to an integer step
    count -- ``steps_per_row = int(row_duration/dt)`` -- loses 0.929 ms per row,
    which accumulates to 26 ms over 28 rows: the scan then finishes the image at
    step 112 of 125 and the ``min(..., n_rows-1)`` clamp parks the remaining 13
    steps on the near-empty bottom row (34 ms there instead of 8 ms, and ~10%
    less total drive per image).

    Recomputing the row from continuous time instead never quantises
    row_duration, so boundaries stay at true multiples of it and rows take 4 or
    5 steps as needed, summing to exactly one image.  This is the schedule the
    canonical notebook's ``run_exploration_simulation`` uses; its training path
    used the quantised form, so the two disagreed.  Both paths call this now, so
    a probe and a training run drive the network identically -- required for
    comparing trained networks against a boundary fit from exploration runs.

    `step` may be image-local or a global counter; the modulo handles both.
    """
    if row_duration is None:
        row_duration = image_duration / n_rows
    time_in_image = (step * dt) % image_duration
    return min(int(time_in_image / row_duration), n_rows - 1)


class SparseBigWilsonCowan(nn.Module):
    """Large Wilson-Cowan network with sparse connectivity and learnable parameters.

    Verbatim port of cell 14.  ``step``/``forward`` are frozen (A2).  Stochastic
    initialization is drawn from the explicit ``rng`` (numpy Generator) rather
    than global state; ranges/shapes/distributions are unchanged.
    """

    def __init__(
        self,
        CeLoc,
        CeRem,
        CeLocI,
        rng,                              # numpy Generator — all stochastic init
        tauPy=1.0 / 25,
        tauInh=0.5 / 25,
        tauOut=1.0 / 25,
        tau_H=1.0,
        alpha_H=0.5,
        bias_e=-2.5,
        bias_i=-3.0,
        bias_out=-3.0,
        mnist_input_dim=784,
        num_output=10,
        output_weight_scale=0.01,
        device=None,
        bias_e_init=None,
        bias_i_init=None,
        init_ranges=None,
        w_celoc=None,
        w_cerem=None,
    ):
        super().__init__()

        # Recurrent E->E drive coefficients (see wc_config).  Stored on the
        # model so every forward path — step(), forward(), forward_batched() —
        # and the LFP proxy read the SAME numbers, and so a pilot arm that
        # varies them is captured by the model instance rather than by module
        # state.  Plain floats, not buffers: they are fixed hyperparameters,
        # not learnable and not part of the state dict.
        self.w_py2py_self = float(cfg.W_PY2PY_SELF)
        self.w_celoc = float(cfg.W_CELOC if w_celoc is None else w_celoc)
        self.w_cerem = float(cfg.W_CEREM if w_cerem is None else w_cerem)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Cache of COO row-index tensors keyed by the id() of each stored
        # crow_indices attribute (see _get_sparse_matrix).  Populated lazily on
        # first use; the crow attributes live for the model's lifetime so their
        # ids are stable.
        self._coo_index_cache = {}

        if init_ranges is None:
            init_ranges = DEFAULT_INIT_RANGES
        R = init_ranges

        # Dimensions
        self.N = int(CeLoc.shape[0])
        self.num_output = int(num_output)
        self.mnist_input_dim = int(mnist_input_dim)
        self.mnist_neurons = min(self.N, self.mnist_input_dim)

        # ------------------------------------------------------------
        # Slow adaptation variable H (v57) - fixed (non-learnable)
        # ------------------------------------------------------------
        self.tau_H = float(tau_H)
        self.alpha_H = float(alpha_H)

        # ------------------------------------------------------------
        # Learnable per-unit time constants (softplus positivity)
        # ------------------------------------------------------------
        self.tau_floor = 1e-4

        def inv_softplus(x: torch.Tensor) -> torch.Tensor:
            # inverse of softplus for x>0
            return torch.log(torch.expm1(x))

        # Initialize taus from ranges (drawn from the explicit rng)
        tauPy_init = rng.uniform(R["taupy"][0], R["taupy"][1], size=self.N).astype(np.float32)
        tauInh_init = rng.uniform(R["tauinh"][0], R["tauinh"][1], size=self.N).astype(np.float32)
        tauOut_init = rng.uniform(R["tauout"][0], R["tauout"][1], size=self.num_output).astype(np.float32)

        # raw params live on device
        tauPy_t = torch.tensor(tauPy_init, device=self.device, dtype=torch.float32)
        tauInh_t = torch.tensor(tauInh_init, device=self.device, dtype=torch.float32)
        tauOut_t = torch.tensor(tauOut_init, device=self.device, dtype=torch.float32)

        # ensure strictly > tau_floor before inverse
        eps = 1e-8
        self.tauPy_raw = nn.Parameter(inv_softplus(torch.clamp(tauPy_t - self.tau_floor, min=eps)))
        self.tauInh_raw = nn.Parameter(inv_softplus(torch.clamp(tauInh_t - self.tau_floor, min=eps)))
        self.tauOut_raw = nn.Parameter(inv_softplus(torch.clamp(tauOut_t - self.tau_floor, min=eps)))

        # ------------------------------------------------------------
        # Bias terms (v37-style initialization)
        # ------------------------------------------------------------
        be = float(rng.uniform(R["bias_e"][0], R["bias_e"][1])) if (bias_e_init is None) else float(bias_e_init)
        bi = float(rng.uniform(R["bias_i"][0], R["bias_i"][1])) if (bias_i_init is None) else float(bias_i_init)

        self.bias_e = nn.Parameter(torch.tensor(be, dtype=torch.float32, device=self.device))
        self.bias_i = nn.Parameter(torch.tensor(bi, dtype=torch.float32, device=self.device))

        # Keep bias_out fixed (v37 behavior)
        self.bias_out = float(bias_out)

        # ------------------------------------------------------------
        # MNIST input weights (v37-style random init)
        # ------------------------------------------------------------
        mnist_init = torch.tensor(
            rng.uniform(R["mnist_weight"][0], R["mnist_weight"][1], size=self.mnist_neurons).astype(np.float32),
            device=self.device,
        )
        self.mnist_weights = nn.Parameter(mnist_init)

        # ------------------------------------------------------------
        # Inhibitory-to-excitatory gain (v37-style)
        # ------------------------------------------------------------
        inh_init = torch.tensor(
            rng.uniform(R["inh2py_gain"][0], R["inh2py_gain"][1], size=self.N).astype(np.float32),
            device=self.device,
        )
        self.inh_to_exc_gain = nn.Parameter(inh_init)

        # ------------------------------------------------------------
        # Sparse connectivity (CSR + randomized scaling, v37-style)
        # ------------------------------------------------------------
        def to_csr(mat):
            if isinstance(mat, torch.Tensor) and mat.is_sparse_csr:
                return mat
            if isinstance(mat, torch.Tensor) and mat.is_sparse:
                return mat.to_sparse_csr()
            return torch.tensor(mat, dtype=torch.float32, device=self.device).to_sparse_csr()

        CeLoc_csr = to_csr(CeLoc).to(self.device)
        CeRem_csr = to_csr(CeRem).to(self.device)
        CeLocI_csr = to_csr(CeLocI).to(self.device)

        CeLoc_vals = CeLoc_csr.values().clone()
        CeRem_vals = CeRem_csr.values().clone()
        CeLocI_vals = CeLocI_csr.values().clone()

        # Randomized per-edge scaling (drawn from rng, same uniform ranges as cell 7)
        CeLoc_vals = CeLoc_vals * self._rand_like(CeLoc_vals, rng, R["celoc_scale"])
        CeRem_vals = CeRem_vals * self._rand_like(CeRem_vals, rng, R["cerem_scale"])
        CeLocI_vals = CeLocI_vals * self._rand_like(CeLocI_vals, rng, R["celoci_scale"])

        self.CeLoc_crow_indices = CeLoc_csr.crow_indices()
        self.CeLoc_col_indices = CeLoc_csr.col_indices()
        self.CeLoc_shape = CeLoc_csr.shape
        self.CeLoc_values = nn.Parameter(CeLoc_vals)

        self.CeRem_crow_indices = CeRem_csr.crow_indices()
        self.CeRem_col_indices = CeRem_csr.col_indices()
        self.CeRem_shape = CeRem_csr.shape
        self.CeRem_values = nn.Parameter(CeRem_vals)

        self.CeLocI_crow_indices = CeLocI_csr.crow_indices()
        self.CeLocI_col_indices = CeLocI_csr.col_indices()
        self.CeLocI_shape = CeLocI_csr.shape
        self.CeLocI_values = nn.Parameter(CeLocI_vals)

        # Output weights (Gaussian, drawn from rng, scaled identically to cell 14)
        out_w = rng.standard_normal((self.num_output, self.N)).astype(np.float32) * float(output_weight_scale)
        self.output_weights = nn.Parameter(torch.tensor(out_w, device=self.device))

    def _rand_like(self, ref: torch.Tensor, rng, rng_range):
        """Uniform(rng_range) tensor shaped like ``ref`` drawn from the numpy rng."""
        lo, hi = rng_range
        arr = rng.uniform(lo, hi, size=tuple(ref.shape)).astype(np.float32)
        return torch.tensor(arr, device=ref.device, dtype=torch.float32)

    # ------------------------------------------------------------
    # Effective taus (positive)
    # ------------------------------------------------------------
    @property
    def tauPy_eff(self):
        return self.tau_floor + F.softplus(self.tauPy_raw)

    @property
    def tauInh_eff(self):
        return self.tau_floor + F.softplus(self.tauInh_raw)

    @property
    def tauOut_eff(self):
        return self.tau_floor + F.softplus(self.tauOut_raw)

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _get_sparse_matrix(self, crow, col, vals, shape):
        # Build the sparse operator as a COO tensor rather than CSR.  The frozen
        # step()/forward() math is unchanged (they still call this and do
        # ``A @ E``), but sparse-dense matmul *autograd* on CUDA is robust for
        # COO and buggy for CSR: on some CUDA/torch builds the CSR backward
        # raises "expected scalar type Long but found Bool".  COO @ dense is
        # numerically identical (verified bit-exact on the value gradients), so
        # this is a storage/dispatch change, not a dynamics change (A2 still
        # passes — step()/forward() source is untouched).
        #
        # The COO row indices (expanded from the CSR crow pointers) are constant
        # for the model's lifetime, so they are computed once and cached; only
        # ``vals`` (the learnable Parameter) changes between steps.
        key = id(crow)
        indices = self._coo_index_cache.get(key)
        if indices is None:
            counts = crow[1:] - crow[:-1]
            row = torch.repeat_interleave(
                torch.arange(shape[0], device=crow.device, dtype=col.dtype), counts)
            indices = torch.stack([row, col])
            self._coo_index_cache[key] = indices
        return torch.sparse_coo_tensor(indices, vals, size=shape, device=self.device)

    def activation(self, x, gain=1.0, threshold=4.0):
        return torch.sigmoid(gain * (x - threshold))

    def compute_mnist_input(self, mnist_image, active_row=None):
        """Compute the input to the network from an MNIST image (cell 14, verbatim).

        mnist_image: shape (784,) - flattened MNIST image [0, 1]
        active_row: int or None - if specified, only pixels in this row (0-27)
                    receive input.  When None, all pixels at once.
        Returns: shape (N,) - input current to each neuron
        """
        I_ext = torch.zeros(self.N, device=self.device)

        if active_row is None:
            # Original behavior: all pixels at once
            I_ext[:self.mnist_neurons] = mnist_image[:self.mnist_neurons] * self.mnist_weights
        else:
            # Sequential row input: only activate neurons corresponding to active_row
            row_start = active_row * 28
            row_end = row_start + 28
            # Clamp to valid range
            row_start = max(0, min(row_start, self.mnist_neurons))
            row_end = max(0, min(row_end, self.mnist_neurons))
            if row_start < row_end:
                I_ext[row_start:row_end] = mnist_image[row_start:row_end] * self.mnist_weights[row_start:row_end]

        return I_ext

    # ============================================================
    # FROZEN DYNAMICS — cell 14 step()/forward(). Do not modify.
    # >>> WC_DYNAMICS_BEGIN <<<
    # ============================================================
    @torch.no_grad()
    def step(self, E, I, O, I_ext, dt, H=None):
        """One Euler step of the dynamics (no gradients)."""
        # Force consistent dtype/device (prevents Float/Double matmul issues)
        E = E.to(device=self.device, dtype=torch.float32)
        I = I.to(device=self.device, dtype=torch.float32)
        O = O.to(device=self.device, dtype=torch.float32)
        I_ext = I_ext.to(device=self.device, dtype=torch.float32)

        # Initialize H if not provided (v57)
        if H is None:
            H = torch.zeros(self.N, device=self.device, dtype=torch.float32)
        else:
            H = H.to(device=self.device, dtype=torch.float32)

        # Reconstruct sparse matrices with learnable weights
        CeLoc = self._get_sparse_matrix(
            self.CeLoc_crow_indices, self.CeLoc_col_indices,
            self.CeLoc_values, self.CeLoc_shape
        )
        CeRem = self._get_sparse_matrix(
            self.CeRem_crow_indices, self.CeRem_col_indices,
            self.CeRem_values, self.CeRem_shape
        )
        CeLocI = self._get_sparse_matrix(
            self.CeLocI_crow_indices, self.CeLocI_col_indices,
            self.CeLocI_values, self.CeLocI_shape
        )

        # Connectivity-based inputs for E-I network
        Py2Py = (self.w_py2py_self * E
                 + self.w_celoc * (CeLoc @ E)
                 + self.w_cerem * (CeRem @ E))

        # I -> E (local, per-neuron, learnable gain)
        Inh2Py = self.inh_to_exc_gain * I

        Py2Inh = 0.1 * (CeLocI @ E) + 15.0 * E

        # Net inputs (v57: added -alpha_H * H for slow adaptation feedback)
        input_e = Py2Py - Inh2Py + I_ext + self.bias_e - self.alpha_H * H
        input_i = Py2Inh + self.bias_i

        # Output layer input
        input_out = torch.matmul(self.output_weights, E) + self.bias_out

        # Activations
        S_e = self.activation(input_e)
        S_i = self.activation(input_i)
        S_out = self.activation(input_out)

        # Euler integration
        dE_dt = (-E + S_e) / self.tauPy_eff
        dI_dt = (-I + S_i) / self.tauInh_eff
        dO_dt = (-O + S_out) / self.tauOut_eff

        # v57: Slow adaptation dynamics: tau_H * dH/dt = -H + E
        dH_dt = (-H + E) / self.tau_H

        E_new = E + dt * dE_dt
        I_new = I + dt * dI_dt
        O_new = O + dt * dO_dt
        H_new = H + dt * dH_dt

        return E_new, I_new, O_new, H_new

    def forward(self, E, I, O, I_ext, dt, H=None):
        """Forward pass WITH gradients (for training)."""
        # Force consistent dtype/device (prevents Float/Double matmul issues)
        E = E.to(device=self.device, dtype=torch.float32)
        I = I.to(device=self.device, dtype=torch.float32)
        O = O.to(device=self.device, dtype=torch.float32)
        I_ext = I_ext.to(device=self.device, dtype=torch.float32)

        # Initialize H if not provided (v57)
        if H is None:
            H = torch.zeros(self.N, device=self.device, dtype=torch.float32)
        else:
            H = H.to(device=self.device, dtype=torch.float32)

        # Reconstruct sparse matrices with learnable weights
        CeLoc = self._get_sparse_matrix(
            self.CeLoc_crow_indices, self.CeLoc_col_indices,
            self.CeLoc_values, self.CeLoc_shape
        )
        CeRem = self._get_sparse_matrix(
            self.CeRem_crow_indices, self.CeRem_col_indices,
            self.CeRem_values, self.CeRem_shape
        )
        CeLocI = self._get_sparse_matrix(
            self.CeLocI_crow_indices, self.CeLocI_col_indices,
            self.CeLocI_values, self.CeLocI_shape
        )

        # Connectivity-based inputs for E-I network
        Py2Py = (self.w_py2py_self * E
                 + self.w_celoc * (CeLoc @ E)
                 + self.w_cerem * (CeRem @ E))

        # I -> E (local, per-neuron, learnable gain)
        Inh2Py = self.inh_to_exc_gain * I

        Py2Inh = 0.1 * (CeLocI @ E) + 15.0 * E

        # Net inputs (v57: added -alpha_H * H for slow adaptation feedback)
        input_e = Py2Py - Inh2Py + I_ext + self.bias_e - self.alpha_H * H
        input_i = Py2Inh + self.bias_i

        # Output layer input
        input_out = torch.matmul(self.output_weights, E) + self.bias_out

        # Activations
        S_e = self.activation(input_e)
        S_i = self.activation(input_i)
        S_out = self.activation(input_out)

        # Euler integration
        dE_dt = (-E + S_e) / self.tauPy_eff
        dI_dt = (-I + S_i) / self.tauInh_eff
        dO_dt = (-O + S_out) / self.tauOut_eff

        # v57: Slow adaptation dynamics: tau_H * dH/dt = -H + E
        dH_dt = (-H + E) / self.tau_H

        E_new = E + dt * dE_dt
        I_new = I + dt * dI_dt
        O_new = O + dt * dO_dt
        H_new = H + dt * dH_dt

        return E_new, I_new, O_new, H_new
    # >>> WC_DYNAMICS_END <<<


# =============================================================================
# Training / evaluation helpers (notebook cell 22) — verbatim except that the
# notebook global ``MNIST_INPUT_SCALE`` is now an explicit parameter.
# =============================================================================
def train_one_image(model, mnist_image, label, optimizer, criterion,
                    dt=0.002, image_duration=0.25, output_average_window=0.01,
                    E_init=None, I_init=None, O_init=None, H_init=None,
                    activity_callback=None, activity_time_offset=0.0,
                    sequential_row_input=False, row_duration=None,
                    mnist_input_scale=cfg.MNIST_INPUT_SCALE):
    """Train on a single MNIST image using image-boundary truncated BPTT (cell 22)."""
    optimizer.zero_grad()

    # --- Initialize states from previous image (truncate graph) ---
    N = model.N
    num_output = model.num_output

    if E_init is None:
        E = torch.zeros(N, device=model.device)
    else:
        E = E_init.detach()

    if I_init is None:
        I = torch.zeros(N, device=model.device)
    else:
        I = I_init.detach()

    if O_init is None:
        O = torch.zeros(num_output, device=model.device)
    else:
        O = O_init.detach()

    # v57: Initialize H (slow adaptation)
    if H_init is None:
        H = torch.zeros(N, device=model.device)
    else:
        H = H_init.detach()

    # Make them leaf variables with gradients for this image window
    E = E.clone().requires_grad_(True)
    I = I.clone().requires_grad_(True)
    O = O.clone().requires_grad_(True)
    H = H.clone().requires_grad_(True)  # v57: H also needs gradients for BPTT

    num_steps = int(image_duration / dt)
    buffer_steps = int(output_average_window / dt)

    # Buffer for averaging output over last window
    output_buffer = []

    if sequential_row_input:
        # Sequential row-by-row presentation
        if row_duration is None:
            row_duration = image_duration / 28
        for step in range(num_steps):
            current_row = active_row_at(step, dt, image_duration, row_duration)

            # Compute input with only the active row
            I_ext = model.compute_mnist_input(
                mnist_image.to(model.device) * mnist_input_scale,
                active_row=current_row
            )

            E, I, O, H = model.forward(E, I, O, I_ext, dt, H=H)  # v57: added H

            # Optional callback to record activity without tracking gradients
            if activity_callback is not None:
                t = activity_time_offset + (step + 1) * dt
                activity_callback(t, E.detach(), I.detach(), H.detach())  # v57: added H

            # Store outputs in the final averaging window
            if step >= num_steps - buffer_steps:
                output_buffer.append(O)
    else:
        # Original behavior: full image at once
        I_ext = model.compute_mnist_input(mnist_image.to(model.device) * mnist_input_scale)

        for step in range(num_steps):
            E, I, O, H = model.forward(E, I, O, I_ext, dt, H=H)  # v57: added H

            # Optional callback to record activity without tracking gradients
            if activity_callback is not None:
                t = activity_time_offset + (step + 1) * dt
                activity_callback(t, E.detach(), I.detach(), H.detach())  # v57: added H

            # Store outputs in the final averaging window
            if step >= num_steps - buffer_steps:
                output_buffer.append(O)

    # Average output over final window
    avg_output = torch.stack(output_buffer).mean(dim=0)

    # Compute loss
    loss = criterion(avg_output.unsqueeze(0), label.unsqueeze(0))

    # Backward pass through all timesteps of this image
    loss.backward()

    # Update weights
    optimizer.step()

    # Check if prediction was correct
    predicted = torch.argmax(avg_output)
    correct = (predicted == label).item()

    # Detach final states so they can be reused for the next image
    E_final = E.detach()
    I_final = I.detach()
    O_final = O.detach()
    H_final = H.detach()  # v57: slow adaptation

    return loss.item(), avg_output.detach().cpu(), correct, E_final, I_final, O_final, H_final


def train_epoch(model, train_loader, optimizer, criterion,
                dt=0.002, image_duration=0.25, output_average_window=0.01,
                max_images=None, verbose=True,
                E_init=None, I_init=None, O_init=None, H_init=None,
                sequential_row_input=False, row_duration=None,
                mnist_input_scale=cfg.MNIST_INPUT_SCALE):
    """Train for one epoch over the dataset (cell 22)."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    # Persistent states across the whole epoch
    E_state = E_init
    I_state = I_init
    O_state = O_init
    H_state = H_init  # v57: slow adaptation

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(model.device)
        target = target.to(model.device)

        batch_size = data.size(0)

        for i in range(batch_size):
            mnist_image = data[i].view(-1)
            label = target[i]

            loss, _, correct, E_state, I_state, O_state, H_state = train_one_image(
                model, mnist_image, label, optimizer, criterion,
                dt=dt,
                image_duration=image_duration,
                output_average_window=output_average_window,
                E_init=E_state,
                I_init=I_state,
                O_init=O_state,
                H_init=H_state,  # v57: slow adaptation
                sequential_row_input=sequential_row_input,
                row_duration=row_duration,
                mnist_input_scale=mnist_input_scale,
            )

            total_loss += loss
            total_correct += correct
            total_images += 1

            if max_images is not None and total_images >= max_images:
                break

        if verbose and (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                  f"Images: {total_images}, "
                  f"Loss: {total_loss/total_images:.4f}, "
                  f"Acc: {100*total_correct/total_images:.2f}%")

        if max_images is not None and total_images >= max_images:
            break

    avg_loss = total_loss / max_images if (max_images is not None and max_images > 0) else total_loss / max(total_images, 1)
    accuracy = total_correct / max(total_images, 1)

    return avg_loss, accuracy, E_state, I_state, O_state, H_state  # v57: added H_state


def evaluate(model, test_loader, criterion,
             dt=0.002, image_duration=0.25, output_average_window=0.01,
             max_images=None, verbose=True,
             E_init=None, I_init=None, O_init=None, H_init=None,
             pulsed_stimuli=False,
             sequential_row_input=False, row_duration=None,
             record_H_history=False,
             mnist_input_scale=cfg.MNIST_INPUT_SCALE):
    """Evaluate model with continuous E/I/O/H dynamics (cell 22)."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    predictions = []
    true_labels = []
    outputs_list = []

    N = model.N
    num_output = model.num_output

    # Initialize persistent states for this evaluation run
    if E_init is None:
        E_state = torch.zeros(N, device=model.device)
    else:
        E_state = E_init.to(model.device)

    if I_init is None:
        I_state = torch.zeros(N, device=model.device)
    else:
        I_state = I_init.to(model.device)

    if O_init is None:
        O_state = torch.zeros(num_output, device=model.device)
    else:
        O_state = O_init.to(model.device)

    # v57: Initialize H (slow adaptation)
    if H_init is None:
        H_state = torch.zeros(N, device=model.device)
    else:
        H_state = H_init.to(model.device)

    # v57: H history tracking for plots
    H_history_times = []
    H_history_means = []
    H_history_stds = []
    eval_time_offset = 0.0  # Track cumulative time during evaluation

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(model.device)
            target = target.to(model.device)

            batch_size = data.size(0)

            for i in range(batch_size):
                mnist_image = data[i].view(-1)
                label = target[i]

                # Initialize states from previous image's final state
                E = E_state.clone()
                I = I_state.clone()
                O = O_state.clone()
                H = H_state.clone()  # v57: slow adaptation

                num_steps = int(image_duration / dt)
                buffer_steps = int(output_average_window / dt)

                output_buffer = []

                if sequential_row_input:
                    # Sequential row-by-row presentation
                    if row_duration is None:
                        _row_duration = image_duration / 28
                    else:
                        _row_duration = row_duration

                    for step in range(num_steps):
                        current_row = active_row_at(step, dt, image_duration, _row_duration)
                        I_ext = model.compute_mnist_input(
                            mnist_image * mnist_input_scale,
                            active_row=current_row
                        )
                        E, I, O, H = model.step(E, I, O, I_ext, dt, H=H)  # v57: added H

                        # v57: Record H statistics if requested
                        if record_H_history:
                            t = eval_time_offset + step * dt
                            H_history_times.append(t)
                            H_history_means.append(float(H.mean().item()))
                            H_history_stds.append(float(H.std().item()))

                        if step >= num_steps - buffer_steps:
                            output_buffer.append(O)
                else:
                    # Original behavior: full image at once
                    I_ext = model.compute_mnist_input(mnist_image * mnist_input_scale)

                    for step in range(num_steps):
                        E, I, O, H = model.step(E, I, O, I_ext, dt, H=H)  # v57: added H

                        # v57: Record H statistics if requested
                        if record_H_history:
                            t = eval_time_offset + step * dt
                            H_history_times.append(t)
                            H_history_means.append(float(H.mean().item()))
                            H_history_stds.append(float(H.std().item()))

                        if step >= num_steps - buffer_steps:
                            output_buffer.append(O)

                avg_output = torch.stack(output_buffer).mean(dim=0)

                loss = criterion(avg_output.unsqueeze(0), label.unsqueeze(0))
                total_loss += loss.item()

                predicted = torch.argmax(avg_output)
                correct = (predicted == label).item()
                total_correct += correct
                total_images += 1

                predictions.append(predicted.cpu().item())
                true_labels.append(label.cpu().item())
                outputs_list.append(avg_output.cpu().numpy())

                # Update persistent states for continuity across images
                E_state = E.clone()
                I_state = I.clone()
                O_state = O.clone()
                H_state = H.clone()  # v57: slow adaptation

                # v57: Update time offset for H history
                eval_time_offset += image_duration

                # Optional 1s OFF period after blocks of images (pulsed stimulus mode)
                if pulsed_stimuli:
                    images_per_block = max(int(round(1.0 / image_duration)), 1)
                    if (total_images) % images_per_block == 0 and (max_images is None or total_images < max_images):
                        off_duration = 1.0  # seconds of no external input
                        num_off_steps = int(off_duration / dt)
                        I_ext_off = torch.zeros_like(I_ext)
                        for off_step in range(num_off_steps):
                            E_state, I_state, O_state, H_state = model.step(E_state, I_state, O_state, I_ext_off, dt, H=H_state)  # v57: added H

                            # v57: Record H statistics during OFF period if requested
                            if record_H_history:
                                t = eval_time_offset + off_step * dt
                                H_history_times.append(t)
                                H_history_means.append(float(H_state.mean().item()))
                                H_history_stds.append(float(H_state.std().item()))

                        # v57: Update time offset after OFF period
                        eval_time_offset += off_duration

                if max_images is not None and total_images >= max_images:
                    break

            if max_images is not None and total_images >= max_images:
                break

    avg_loss = total_loss / max(total_images, 1)
    accuracy = total_correct / max(total_images, 1)

    if verbose:
        print(f"Evaluation: Loss: {avg_loss:.4f}, Accuracy: {100*accuracy:.2f}%")

    # v57: Prepare H history if recorded
    if record_H_history:
        H_history = {
            'times': H_history_times,
            'means': H_history_means,
            'stds': H_history_stds
        }
    else:
        H_history = None

    return avg_loss, accuracy, predictions, true_labels, outputs_list, E_state, I_state, O_state, H_state, H_history


# =============================================================================
# Batched evaluation (notebook cell 16 `_topology_simulate_batched`) — used for
# the clean-gradient probe and topology, where images are independent (each runs
# from a zero state), so they can be evaluated in parallel as a [B, N] batch.
#
# This is NOT used for training/validation (those carry state image-to-image and
# must stay sequential).  forward_batched reproduces the per-sample step()/forward()
# math element-for-element for a batch; the only difference is the leading batch
# dimension, so it is numerically equivalent (verified against forward()).
# =============================================================================
def forward_batched(model, E, I, O, H, I_ext, dt):
    """One Euler step for a batch.  E, I, H: [B, N]; O: [B, Nout]; I_ext: [B, N]."""
    CeLoc = model._get_sparse_matrix(model.CeLoc_crow_indices, model.CeLoc_col_indices,
                                     model.CeLoc_values, model.CeLoc_shape)
    CeRem = model._get_sparse_matrix(model.CeRem_crow_indices, model.CeRem_col_indices,
                                     model.CeRem_values, model.CeRem_shape)
    CeLocI = model._get_sparse_matrix(model.CeLocI_crow_indices, model.CeLocI_col_indices,
                                      model.CeLocI_values, model.CeLocI_shape)

    ET = E.T.contiguous()                                   # [N, B]
    Py2Py = (model.w_py2py_self * E
             + model.w_celoc * (CeLoc @ ET).T
             + model.w_cerem * (CeRem @ ET).T)
    Inh2Py = model.inh_to_exc_gain * I                      # [N] * [B,N]
    Py2Inh = 0.1 * (CeLocI @ ET).T + 15.0 * E

    input_e = Py2Py - Inh2Py + I_ext + model.bias_e - model.alpha_H * H
    input_i = Py2Inh + model.bias_i
    input_out = E @ model.output_weights.T + model.bias_out  # [B,N]@[N,Nout]

    S_e = model.activation(input_e)
    S_i = model.activation(input_i)
    S_out = model.activation(input_out)

    E_new = E + dt * (-E + S_e) / model.tauPy_eff
    I_new = I + dt * (-I + S_i) / model.tauInh_eff
    O_new = O + dt * (-O + S_out) / model.tauOut_eff
    H_new = H + dt * (-H + E) / model.tau_H                 # old E, matches forward()
    return E_new, I_new, O_new, H_new


def simulate_window_batched(model, images, dt, image_duration, output_average_window,
                            sequential_row_input=False, row_duration=None,
                            mnist_input_scale=cfg.MNIST_INPUT_SCALE):
    """Run one image-window for a batch of images from a zero state, returning the
    window-averaged output logits [B, Nout].  ``images``: [B, 784] on device,
    unscaled.  Honors gradients if called inside a grad-enabled context."""
    device = model.device
    B = images.shape[0]
    N = model.N
    Nout = model.num_output
    mn = model.mnist_neurons
    num_steps = int(image_duration / dt)
    buffer_steps = max(1, int(output_average_window / dt))

    scaled = images * mnist_input_scale
    eff = min(mn, images.shape[1])

    if sequential_row_input:
        rd = row_duration if row_duration is not None else image_duration / 28
        I_ext_rows = torch.zeros(28, B, N, device=device)
        for r in range(28):
            rs = r * 28
            re = min(rs + 28, eff)
            if rs < re:
                I_ext_rows[r, :, rs:re] = scaled[:, rs:re] * model.mnist_weights[rs:re]
    else:
        I_ext_batch = torch.zeros(B, N, device=device)
        I_ext_batch[:, :eff] = scaled[:, :eff] * model.mnist_weights[:eff]

    E = torch.zeros(B, N, device=device)
    I = torch.zeros(B, N, device=device)
    O = torch.zeros(B, Nout, device=device)
    H = torch.zeros(B, N, device=device)
    buf = []
    for step in range(num_steps):
        Ix = I_ext_rows[active_row_at(step, dt, image_duration, rd)] if sequential_row_input else I_ext_batch
        E, I, O, H = forward_batched(model, E, I, O, H, Ix, dt)
        if step >= num_steps - buffer_steps:
            buf.append(O)
    return torch.stack(buf).mean(dim=0)                     # [B, Nout]
