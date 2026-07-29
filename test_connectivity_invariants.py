"""Invariant tests for the Py->Py / Py->Inh connectivity construction.

WHY THIS FILE EXISTS
--------------------
``repro_test.py`` and ``smoke_test.py`` cover reproducibility and end-to-end
running, but neither asserts anything about the *structure* of CeLoc / CeRem /
CeLocI.  A connectivity rework can therefore change network dynamics silently:
the runs still complete, the seeds still reproduce each other, and the seizure
boundary quietly moves.

Every assertion below was verified against the code at commit 9491d19 (the
pre-rework baseline) across p_rewire in {0, 0.001, 0.15, 0.4, 0.6, 0.893, 1.0}
and seeds {0, 1, 7}.  They are *characterization* tests: they pin current
behaviour so that an intentional change shows up as a failing test you then
update deliberately, rather than as an unexplained shift in results.

Run:  python test_connectivity_invariants.py
      (or: pytest test_connectivity_invariants.py -q)

If you are changing connectivity ON PURPOSE, expect failures here.  Read the
docstring of the failing test, decide whether the new behaviour is what you
want, and update the assertion in the same commit as the code change.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Some environments run with PYTHONSAFEPATH set, which omits the script's own
# directory from sys.path; make the repo importable either way.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import wc_config as cfg          # noqa: E402
import wc_model as M             # noqa: E402

# Parameter grid used for every structural test.  0.0 and 1.0 are included
# deliberately: they are outside the sampled range but are the degenerate cases
# where an off-by-one in the split mask would show up.
P_GRID = (0.0, 0.001, 0.15, 0.4, 0.6, 0.893, 1.0)
SEEDS = (0, 1, 7)

N = cfg.N                                    # 784
K_PYPY = int(2 * round(cfg.TARGET_OUTDEG_PyPy / 2))    # 156
K_PYINH = int(2 * round(cfg.TARGET_OUTDEG_PyInh / 2))  # 156


def _build(p_pypy, p_pyinh=None, seed=0):
    """Build the three CSR matrices from a fresh seeded generator."""
    if p_pyinh is None:
        p_pyinh = p_pypy
    rng = np.random.default_rng(seed)
    return M.build_connectivity_csr(rng, p_pypy, p_pyinh)


def _tiny_model(seed=0, **kw):
    """A CPU model on the real connectivity, for coefficient-plumbing checks."""
    import torch
    rng = np.random.default_rng(seed)
    dev = torch.device("cpu")
    CeLoc, CeRem, CeLocI = M.build_connectivity(rng, 0.3, 0.3, dev)
    return M.SparseBigWilsonCowan(CeLoc, CeRem, CeLocI, rng=rng, device=dev, **kw)


# ---------------------------------------------------------------------------
# 1. The ring lattice we start from
# ---------------------------------------------------------------------------

def test_ring_lattice_shape_and_size():
    """The directed ring lattice deals exactly k edges per unit, no duplicates.

    784 units * 156 out-edges = 122,304 stored entries.  If this number changes,
    either N/k changed in wc_config or the ring construction started producing
    collisions -- both of which change total recurrent drive.
    """
    ring = M.generate_ring_lattice_directed(N, K_PYPY)
    assert ring.shape == (N, N)
    assert ring.nnz == N * K_PYPY == 122304, f"ring nnz={ring.nnz}"
    # Every entry is a single edge: the ring never deals the same pair twice.
    assert ring.data.max() == 1.0, f"ring has duplicate entries, max={ring.data.max()}"
    # No self-connections.
    assert (ring.diagonal() == 0).all(), "ring lattice has self-loops"
    # Out-degree is exactly k for every unit (columns are sources).
    colsum = np.asarray(ring.sum(axis=0)).ravel()
    assert colsum.min() == colsum.max() == K_PYPY, f"out-degree range {colsum.min()}-{colsum.max()}"


def test_ring_lattice_requires_even_k():
    """Odd k is rejected -- the construction adds forward and backward edges in pairs."""
    try:
        M.generate_ring_lattice_directed(64, 7)
    except AssertionError:
        return
    raise AssertionError("odd k should raise AssertionError")


# ---------------------------------------------------------------------------
# 2. The CeLoc / CeRem split
# ---------------------------------------------------------------------------

def test_dealt_weight_is_conserved_per_unit():
    """CeLoc + CeRem column sums equal exactly k for EVERY unit, at every p.

    This is the strongest statement about the split: rewiring moves an edge's
    *target*, never its source, so each unit deals exactly K_PYPY units of
    outgoing weight regardless of p.  Duplicate collapse sums colliding entries,
    which preserves the column sum.

    Total recurrent excitation depends on this quantity, so a break here means
    drive changed.
    """
    for p in P_GRID:
        for seed in SEEDS:
            CeLoc, CeRem, _ = _build(p, seed=seed)
            colsum = (np.asarray(CeLoc.sum(axis=0)).ravel()
                      + np.asarray(CeRem.sum(axis=0)).ravel())
            assert colsum.min() == colsum.max() == K_PYPY, (
                f"p={p} seed={seed}: per-unit dealt weight ranges "
                f"{colsum.min()}-{colsum.max()}, expected exactly {K_PYPY}")


def test_celoc_never_has_duplicate_entries():
    """CeLoc entries are always exactly 1.0 -- it holds un-rewired ring edges only.

    The ring deals each (target, source) pair once, and the split mask only
    *removes* edges from CeLoc, so no collisions are possible.  CeRem, by
    contrast, accumulates collisions from random rewiring (entries of 2, 3, 4).

    This asymmetry is why mean stored weight differs between the two matrices,
    and why CeLoc_values_max is pinned at CELOC_INIT_SCALE_MAX in the run logs.
    """
    for p in P_GRID:
        for seed in SEEDS:
            CeLoc, _, _ = _build(p, seed=seed)
            if CeLoc.nnz:
                assert CeLoc.data.max() == 1.0, (
                    f"p={p} seed={seed}: CeLoc has a collapsed entry of "
                    f"{CeLoc.data.max()} -- it should never collide")


def test_cerem_entries_are_positive_integers():
    """CeRem stored values are integral edge counts before scaling is applied.

    The scale draw happens later, in the model __init__, per STORED ENTRY.  If
    values here are non-integral, scaling has leaked into construction.
    """
    for p in (0.4, 0.893):
        for seed in SEEDS:
            _, CeRem, _ = _build(p, seed=seed)
            if CeRem.nnz:
                assert np.allclose(CeRem.data, np.round(CeRem.data))
                assert CeRem.data.min() >= 1.0


def test_split_is_monotone_in_p():
    """CeRem grows and CeLoc shrinks as p rises; the endpoints are degenerate.

    At p=0 every edge stays local (CeRem empty); at p=1 every edge is rewired
    (CeLoc empty).  Nothing in between should invert that ordering.
    """
    loc_sums, rem_sums = [], []
    for p in P_GRID:
        CeLoc, CeRem, _ = _build(p, seed=0)
        loc_sums.append(CeLoc.sum())
        rem_sums.append(CeRem.sum())
    assert loc_sums[0] == N * K_PYPY, "p=0 should leave all edges local"
    assert rem_sums[0] == 0, "p=0 should produce an empty CeRem"
    assert loc_sums[-1] == 0, "p=1 should empty CeLoc"
    assert rem_sums[-1] == N * K_PYPY, "p=1 should move all edges to CeRem"
    assert all(a >= b for a, b in zip(loc_sums, loc_sums[1:])), f"CeLoc not monotone: {loc_sums}"
    assert all(a <= b for a, b in zip(rem_sums, rem_sums[1:])), f"CeRem not monotone: {rem_sums}"


def test_no_self_connections_anywhere():
    """Rewiring rejects new_row == src, so no matrix ever gains a self-loop.

    A self-loop would add to a unit's own recurrent drive on top of the explicit
    self-term in the WC update, double-counting it.
    """
    for p in (0.15, 0.6, 0.893):
        for seed in SEEDS:
            CeLoc, CeRem, CeLocI = _build(p, seed=seed)
            for name, mat in (("CeLoc", CeLoc), ("CeRem", CeRem), ("CeLocI", CeLocI)):
                assert (mat.diagonal() == 0).all(), f"{name} has self-loops at p={p} seed={seed}"


# ---------------------------------------------------------------------------
# 3. Py->Inh
# ---------------------------------------------------------------------------

def test_celoci_conserves_dealt_weight():
    """CeLocI is a full Watts-Strogatz network: every unit deals K_PYINH edges.

    Unlike the CeLoc/CeRem pair, CeLocI keeps rewired and un-rewired edges in a
    SINGLE matrix, so its column sum is k (not split) and it does accumulate
    duplicates.
    """
    for p in (0.0, 0.25, 0.9):
        for seed in SEEDS:
            _, _, CeLocI = _build(0.4, p_pyinh=p, seed=seed)
            colsum = np.asarray(CeLocI.sum(axis=0)).ravel()
            assert colsum.min() == colsum.max() == K_PYINH, (
                f"p_pyinh={p} seed={seed}: CeLocI out-degree "
                f"{colsum.min()}-{colsum.max()}, expected {K_PYINH}")


# ---------------------------------------------------------------------------
# 4. Determinism and the documented draw order
# ---------------------------------------------------------------------------

def test_same_seed_gives_identical_matrices():
    """Construction is a pure function of (rng seed, p_pypy, p_pyinh)."""
    a = _build(0.4, 0.3, seed=42)
    b = _build(0.4, 0.3, seed=42)
    for name, x, y in zip(("CeLoc", "CeRem", "CeLocI"), a, b):
        assert x.nnz == y.nnz, f"{name} nnz differs between identical seeds"
        assert np.array_equal(x.indices, y.indices), f"{name} indices differ"
        assert np.array_equal(x.indptr, y.indptr), f"{name} indptr differs"
        assert np.allclose(x.data, y.data), f"{name} data differs"


def test_draw_order_isolates_pypy_from_pyinh():
    """Changing p_rewire_pyinh must not perturb CeLoc/CeRem.

    build_connectivity_csr documents a fixed draw order: split mask -> remote
    targets -> CeLocI rewiring.  Because CeLocI draws LAST, the Py->Py matrices
    are independent of p_pyinh at a fixed seed.  This is what makes it possible
    to vary Py->Inh topology without also perturbing recurrent excitation.
    """
    base = _build(0.4, 0.3, seed=42)
    alt = _build(0.4, 0.9, seed=42)
    for name, x, y in zip(("CeLoc", "CeRem"), base[:2], alt[:2]):
        assert np.array_equal(x.indices, y.indices), (
            f"{name} changed when only p_rewire_pyinh changed -- draw order broken")
        assert np.allclose(x.data, y.data), f"{name} data changed with p_pyinh"


def test_different_seeds_give_different_wiring():
    """Guard against an rng that is silently ignored (which would make all
    'independent seeds' identical and collapse the study's error bars)."""
    a = _build(0.4, seed=0)
    b = _build(0.4, seed=1)
    assert not np.array_equal(a[1].indices, b[1].indices), "CeRem identical across seeds"


# ---------------------------------------------------------------------------
# 5. The scale ranges (config contract) and the weights they produce
# ---------------------------------------------------------------------------

def test_init_scale_ranges_straddle_zero():
    """All three scale ranges include negative values -- this is load-bearing.

    A minority of nominally excitatory recurrent synapses initialize INHIBITORY.
    Several analyses assumed strictly positive ranges and were wrong.  If these
    bounds change, mean edge weight and therefore total drive change with them.

    The two E->E fractions are now both 1/6 by construction: the ranges were
    equalized to U[-0.6, 3.0] so their means match, which is half of what makes
    drive flat in p_rewire (equal coefficients are the other half).  CeLocI is
    untouched at U[-0.7, 1.5].
    """
    for lo, hi, frac, name in (
        (cfg.CELOC_INIT_SCALE_MIN, cfg.CELOC_INIT_SCALE_MAX, 0.167, "CeLoc"),
        (cfg.CEREM_INIT_SCALE_MIN, cfg.CEREM_INIT_SCALE_MAX, 0.167, "CeRem"),
        (cfg.CELOCI_INIT_SCALE_MIN, cfg.CELOCI_INIT_SCALE_MAX, 0.318, "CeLocI"),
    ):
        assert lo < 0.0 < hi, f"{name} range [{lo}, {hi}] no longer straddles zero"
        assert abs((-lo / (hi - lo)) - frac) < 0.005, (
            f"{name} negative fraction changed: {-lo / (hi - lo):.3f} vs {frac}")


def test_ee_ranges_are_identical():
    """The two E->E scale ranges are now the SAME range, not merely similar.

    They used to differ in their lower bound (CeLoc U[-0.3, 1.5] mean 0.600;
    CeRem U[-0.4, 1.5] mean 0.550), which is what made an edge moved from local
    to remote lose strength and gave p_rewire a drive gradient to smuggle.  They
    are now identical, so mean(CeLoc) == mean(CeRem) exactly.
    """
    assert cfg.CELOC_INIT_SCALE_MIN == cfg.CEREM_INIT_SCALE_MIN, (
        "E->E lower bounds diverged -- means no longer match, so drive is no "
        "longer flat in p_rewire")
    assert cfg.CELOC_INIT_SCALE_MAX == cfg.CEREM_INIT_SCALE_MAX, (
        "E->E upper bounds diverged -- same consequence")


def test_ee_drive_is_flat_in_p_rewire():
    """The point of the whole rework: per-unit E->E drive does not depend on p.

    Branch A splits ONE ring of TARGET_OUTDEG_PyPy edges between CeLoc and CeRem
    according to p_rewire, so per-unit drive is

        W_PY2PY_SELF + K*[ W_CELOC*mean(CeLoc)*(1-p) + W_CEREM*mean(CeRem)*p ]

    whose slope in p is K*(W_CEREM*mean(CeRem) - W_CELOC*mean(CeLoc)).  Equal
    coefficients AND equal means make that slope exactly zero, which is the flat
    drive-vs-p that Branch B had by construction (it dealt a full ring into each
    of CeLoc and CeRem, so both in-degrees were K at every p).

    A nonzero slope means p_rewire again carries an unmodelled excitation
    difference, and any effect attributed to topology conceals it.
    """
    mL = (cfg.CELOC_INIT_SCALE_MIN + cfg.CELOC_INIT_SCALE_MAX) / 2
    mR = (cfg.CEREM_INIT_SCALE_MIN + cfg.CEREM_INIT_SCALE_MAX) / 2
    slope = cfg.TARGET_OUTDEG_PyPy * (cfg.W_CEREM * mR - cfg.W_CELOC * mL)
    assert abs(slope) < 1e-9, (
        f"drive slope in p_rewire is {slope:.4f}, not zero -- p_rewire is "
        f"confounded with excitation again (w {cfg.W_CELOC}/{cfg.W_CEREM}, "
        f"means {mL}/{mR})")

    # And empirically, on the real generator with real weight draws.
    drives = []
    for p in (0.001, 0.3, 0.6, 0.9):
        CeLoc, CeRem, _ = _build(p, seed=11)
        rng = np.random.default_rng(4)
        vL = CeLoc.data * rng.uniform(cfg.CELOC_INIT_SCALE_MIN, cfg.CELOC_INIT_SCALE_MAX, CeLoc.nnz)
        vR = CeRem.data * rng.uniform(cfg.CEREM_INIT_SCALE_MIN, cfg.CEREM_INIT_SCALE_MAX, CeRem.nnz)
        drives.append(cfg.W_CELOC * vL.sum() / CeLoc.shape[0]
                      + cfg.W_CEREM * vR.sum() / CeRem.shape[0])
    spread = (max(drives) - min(drives)) / np.mean(drives)
    assert spread < 0.02, (
        f"measured mean drive varies {100*spread:.1f}% across p_rewire "
        f"(expected <2% sampling noise): {[round(d, 3) for d in drives]}")


def test_scaled_celoc_weights_stay_within_scale_range():
    """Because CeLoc entries are all 1.0, its scaled weights lie exactly in
    [CELOC_INIT_SCALE_MIN, CELOC_INIT_SCALE_MAX].

    This is the source of the logged CeLoc_values_max ~= 1.4996 in every run --
    with ~100k edges the largest draw saturates the ceiling.  CeRem has no such
    bound: its scaled max is (max duplicate count) * CEREM_INIT_SCALE_MAX.
    """
    lo, hi = cfg.CELOC_INIT_SCALE_MIN, cfg.CELOC_INIT_SCALE_MAX
    for p in (0.05, 0.4, 0.8):
        CeLoc, CeRem, _ = _build(p, seed=9)
        rng = np.random.default_rng(99)
        w_loc = CeLoc.data * rng.uniform(lo, hi, size=CeLoc.nnz).astype(np.float32)
        assert w_loc.min() >= lo - 1e-5, f"p={p}: CeLoc weight {w_loc.min()} below {lo}"
        assert w_loc.max() <= hi + 1e-5, f"p={p}: CeLoc weight {w_loc.max()} above {hi}"
        # And the ceiling really is approached, which is what the logs show.
        assert w_loc.max() > hi - 0.01, f"p={p}: CeLoc max {w_loc.max()} did not saturate"


def test_ee_coefficients_reach_the_model():
    """Every forward path reads the E->E coefficients from the model instance.

        Py2Py = w_py2py_self * E + w_celoc * (CeLoc @ E) + w_cerem * (CeRem @ E)

    The previous version of this test scraped the source for the literals
    ``10.0``/``0.15``/``0.05``.  They are now named config values that a pilot
    arm may legitimately override, so scraping is both wrong and useless.  What
    still needs guarding is that the coefficients are not hard-coded ANYWHERE in
    the update paths — a stray literal would silently ignore a per-run override.
    """
    import inspect
    for src in (inspect.getsource(M.SparseBigWilsonCowan),
                inspect.getsource(M.forward_batched)):
        for ln in src.splitlines():
            if "Py2Py" not in ln and "w_celoc" not in ln and "w_cerem" not in ln:
                continue
            for lit in ("0.15", "0.05"):
                assert lit not in ln, (
                    f"bare E->E coefficient {lit} in an update path -- a per-run "
                    f"coefficient override would be silently ignored.  Line: {ln.strip()}")

    m = _tiny_model(w_celoc=0.123, w_cerem=0.456)
    assert m.w_celoc == 0.123 and m.w_cerem == 0.456, "override did not reach the model"
    d = _tiny_model()
    assert (d.w_celoc, d.w_cerem) == (cfg.W_CELOC, cfg.W_CEREM), "defaults drifted from config"


def test_lfp_proxy_matches_the_model_update():
    """The LFP proxy IS the model's excitatory update line at the electrode rows.

    wc_figures.save_lfp reconstructs

        LFP = W_PY2PY_SELF*self + W_CELOC*loc + W_CEREM*rem + W_LFP_INH*inh

    If the model's coefficients change and the proxy's do not, every LFP figure
    silently stops measuring the network it is labelled with -- the failure mode
    that motivated naming these constants.  This asserts the proxy carries no
    bare literals and defaults to the same config values the model does.
    """
    import inspect
    import wc_figures as FG
    src = inspect.getsource(FG.save_lfp)
    body = src.split('"""')[-1]          # skip the docstring
    for ln in body.splitlines():
        if "LFP =" in ln or "wL =" in ln or "wR =" in ln:
            for lit in ("0.15", "0.05", "10.0", "25.0"):
                assert lit not in ln, (
                    f"bare coefficient {lit} in the LFP proxy -- proxy can "
                    f"desynchronize from the model.  Line: {ln.strip()}")
    sig = inspect.signature(FG.save_lfp).parameters
    for name in ("w_celoc", "w_cerem"):
        assert name in sig, f"save_lfp cannot track a per-run coefficient: no {name} parameter"


# ---------------------------------------------------------------------------
# 6. Regression guards on measured quantities
# ---------------------------------------------------------------------------

def test_collapse_reduces_nnz_but_not_weight():
    """Duplicate collapse is the ONLY mechanism that makes nnz < edges dealt.

    Collapse sums colliding (target, source) triplets into one stored entry, so
    stored entries fall while total weight is preserved exactly.  Measured at
    p=0.6 seed=3: CeRem 73,342 dealt -> 69,229 stored, CeLoc 48,962 -> 48,962.
    """
    CeLoc, CeRem, _ = _build(0.6, seed=3)
    assert CeLoc.nnz == CeLoc.sum(), "CeLoc should have no collapse"
    assert CeRem.nnz < CeRem.sum(), "CeRem should show collapse at p=0.6"
    # Weight preserved despite fewer entries.
    assert abs((CeLoc.sum() + CeRem.sum()) - N * K_PYPY) < 1e-3


def test_sum_duplicates_call_is_a_noop():
    """The explicit CeRem_csr.sum_duplicates() in build_connectivity_csr does
    nothing: sp.csr_matrix already summed duplicate triplets at construction.

    Kept as a test so that if someone 'fixes' construction to NOT collapse, this
    fails and points at the real collapse site.
    """
    rows = np.array([3, 3, 7, 3])
    cols = np.array([0, 0, 0, 1])
    mat = sp.csr_matrix((np.ones(4, dtype=np.float32), (rows, cols)), shape=(8, 8))
    assert mat.nnz == 3, "constructor no longer collapses duplicate triplets"
    assert mat[3, 0] == 2.0
    before = mat.data.copy()
    mat.sum_duplicates()
    assert np.array_equal(before, mat.data), "sum_duplicates() was not a no-op"


if __name__ == "__main__":
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}\n          {exc}")
        except Exception as exc:                       # noqa: BLE001
            failures.append((name, exc))
            print(f"  ERROR {name}\n          {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
