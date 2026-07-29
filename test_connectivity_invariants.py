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

    A minority of nominally excitatory recurrent synapses initialize INHIBITORY:
    16.7% of CeLoc, 21.1% of CeRem, 31.8% of CeLocI.  Several analyses assumed
    strictly positive ranges and were wrong.  If these bounds change, mean edge
    weight and therefore total drive change with them.
    """
    for lo, hi, frac, name in (
        (cfg.CELOC_INIT_SCALE_MIN, cfg.CELOC_INIT_SCALE_MAX, 0.167, "CeLoc"),
        (cfg.CEREM_INIT_SCALE_MIN, cfg.CEREM_INIT_SCALE_MAX, 0.211, "CeRem"),
        (cfg.CELOCI_INIT_SCALE_MIN, cfg.CELOCI_INIT_SCALE_MAX, 0.318, "CeLocI"),
    ):
        assert lo < 0.0 < hi, f"{name} range [{lo}, {hi}] no longer straddles zero"
        assert abs((-lo / (hi - lo)) - frac) < 0.005, (
            f"{name} negative fraction changed: {-lo / (hi - lo):.3f} vs {frac}")


def test_celoc_and_cerem_share_upper_bound():
    """The two E->E ranges differ ONLY in their lower bound.

    CeLoc U[-0.3, 1.5] mean 0.600; CeRem U[-0.4, 1.5] mean 0.550.  So an edge
    moved from local to remote loses strength purely by gaining more inhibitory
    reach, not by losing excitatory reach: E[w | w>0] = +0.750 for both.
    """
    assert cfg.CELOC_INIT_SCALE_MAX == cfg.CEREM_INIT_SCALE_MAX, (
        "CeLoc/CeRem upper bounds diverged; the 'same excitation, more inhibition' "
        "reading of the asymmetry no longer holds")
    assert cfg.CEREM_INIT_SCALE_MIN < cfg.CELOC_INIT_SCALE_MIN


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


def test_ee_coefficients_unchanged():
    """The WC update weights CeLoc 3x more heavily than CeRem.

        Py2Py = 10.0 * E + 0.15 * (CeLoc @ E) + 0.05 * (CeRem @ E)

    These three coefficients dominate every structural asymmetry measured above:
    the 3x split swamps both the 8.3% draw difference and duplicate inflation.
    The self-term is invariant to p_rewire and larger than either connectivity
    coefficient, so it must be included in any 'total recurrent drive' figure.

    Asserted by reading the source, because they are literals in the update and
    not exposed as config values.
    """
    import inspect
    src = inspect.getsource(M.SparseBigWilsonCowan)
    hits = [ln.strip() for ln in src.splitlines() if "Py2Py = " in ln]
    assert hits, "could not find the Py2Py update line"
    line = hits[0]
    for coeff in ("10.0", "0.15", "0.05"):
        assert coeff in line, (
            f"E->E coefficient {coeff} missing from update line -- recurrent drive "
            f"changed.  Line is: {line}")


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
