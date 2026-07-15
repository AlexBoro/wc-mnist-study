"""model_diff_check.py — Acceptance Test A2: model dynamics unchanged.

Proves the ported model is *semantically identical* to the v82 notebook, three
independent ways:

  1. AST equivalence — ``SparseBigWilsonCowan.step`` and ``.forward`` in
     ``wc_model.py`` compile to the same abstract syntax tree (modulo
     docstrings / comments / whitespace) as cell 14 of the notebook.
  2. Constant diff — the sampled-range constants in ``wc_config.py`` match the
     literals in cell 7, byte-for-byte.  (The single intentional spec change —
     restoring alpha_H ~ Uniform(1, 10) for the H-on config — is checked
     separately against the spec value, not the notebook.)
  3. Numeric equivalence — one ``step`` of the ported model equals an inline
     reference implementation of the exact cell-14 equations, bit-for-bit.

Run:  python model_diff_check.py [path/to/v82_notebook.ipynb]
Exit code 0 on success, 1 on any mismatch.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_NOTEBOOK = HERE / "7_+S_-H_v82_MNIST_Classifier_BPTT_sequential_v82.ipynb"


# =============================================================================
# Notebook helpers
# =============================================================================
def load_notebook_cells(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def find_cell_containing(cells: list[str], needle: str) -> str:
    for src in cells:
        if needle in src:
            return src
    raise AssertionError(f"No code cell contains {needle!r}")


# =============================================================================
# 1. AST equivalence of step()/forward()
# =============================================================================
def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    """Drop a leading string-expression statement (docstring)."""
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        return body[1:]
    return body


def extract_methods(class_src: str, class_name: str, method_names) -> dict[str, ast.FunctionDef]:
    """Return {method_name: FunctionDef} from a source string containing a class.

    The notebook cell wraps the class at module level, so we can parse the whole
    cell.  wc_model.py is parsed the same way.
    """
    tree = ast.parse(class_src)
    cls = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            cls = node
            break
    assert cls is not None, f"class {class_name} not found"
    out = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in method_names:
            out[node.name] = node
    missing = set(method_names) - set(out)
    assert not missing, f"methods not found in {class_name}: {missing}"
    return out


def normalized_body_dump(fn: ast.FunctionDef) -> str:
    """ast.dump of a function's executable body (docstring stripped).

    ``ast.dump`` ignores comments, formatting, whitespace, and line
    continuations entirely — two functions with the same dump are semantically
    identical Python.  Line/column info is excluded.
    """
    stripped = _strip_docstring(fn.body)
    module = ast.Module(body=stripped, type_ignores=[])
    return ast.dump(module, annotate_fields=True, include_attributes=False)


def check_ast_equivalence(nb_cells: list[str]) -> list[str]:
    problems: list[str] = []

    nb_cell14 = find_cell_containing(nb_cells, "class SparseBigWilsonCowan")
    module_src = (HERE / "wc_model.py").read_text(encoding="utf-8")

    nb_methods = extract_methods(nb_cell14, "SparseBigWilsonCowan", ["step", "forward"])
    mod_methods = extract_methods(module_src, "SparseBigWilsonCowan", ["step", "forward"])

    for name in ("step", "forward"):
        nb_dump = normalized_body_dump(nb_methods[name])
        mod_dump = normalized_body_dump(mod_methods[name])
        if nb_dump == mod_dump:
            print(f"  [OK]  {name}(): AST identical to notebook cell 14")
        else:
            problems.append(f"{name}(): AST differs from notebook cell 14")
            # Show the first differing statement for debugging
            nb_stmts = [ast.dump(s) for s in _strip_docstring(nb_methods[name].body)]
            mod_stmts = [ast.dump(s) for s in _strip_docstring(mod_methods[name].body)]
            for i, (a, b) in enumerate(zip(nb_stmts, mod_stmts)):
                if a != b:
                    problems.append(f"    first diff at statement #{i}:")
                    problems.append(f"      notebook: {a[:200]}")
                    problems.append(f"      module:   {b[:200]}")
                    break
            if len(nb_stmts) != len(mod_stmts):
                problems.append(f"    statement count differs: notebook={len(nb_stmts)} module={len(mod_stmts)}")
    return problems


# =============================================================================
# 2. Constant-range diff against cell 7
# =============================================================================
_SAFE_NODES = (ast.Expression, ast.UnaryOp, ast.UAdd, ast.USub, ast.BinOp,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Constant, ast.Tuple,
               ast.Load)


def _safe_eval(node: ast.AST):
    """Evaluate a numeric-literal expression (numbers, +-*/, tuples) safely."""
    for sub in ast.walk(node):
        if not isinstance(sub, _SAFE_NODES):
            raise ValueError(f"unsafe node {type(sub).__name__}")
    return eval(compile(ast.Expression(node), "<const>", "eval"), {"__builtins__": {}}, {})


def extract_assignments(cell_src: str, names: set[str]) -> dict[str, float]:
    """Pull simple numeric assignments (incl. tuple unpacking) for ``names``."""
    tree = ast.parse(cell_src)
    found: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            # tuple unpack:  A, B = 0.01, 0.08
            if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                for t, v in zip(target.elts, node.value.elts):
                    if isinstance(t, ast.Name) and t.id in names:
                        try:
                            found[t.id] = _safe_eval(v)
                        except ValueError:
                            pass
            # scalar:  LEARNING_RATE = 1e-3
            elif isinstance(target, ast.Name) and target.id in names:
                try:
                    found[target.id] = _safe_eval(node.value)
                except ValueError:
                    pass
    return found


def check_constants(nb_cells: list[str]) -> list[str]:
    import wc_config as cfg

    problems: list[str] = []
    # Anchor on cell 7's unique @title (an earlier cell merely *lists* these
    # names to clear them from globals, so we cannot match on the name alone).
    cell7 = find_cell_containing(nb_cells, "Hyperparameters (fixed & randomized)")

    # Map: notebook constant name -> wc_config attribute name
    name_map = {
        "P_REWIRE_PyPy_MIN": "P_REWIRE_PyPy_MIN", "P_REWIRE_PyPy_MAX": "P_REWIRE_PyPy_MAX",
        "P_REWIRE_PyInh_MIN": "P_REWIRE_PyInh_MIN", "P_REWIRE_PyInh_MAX": "P_REWIRE_PyInh_MAX",
        "BIAS_OUT_MIN": "BIAS_OUT_MIN", "BIAS_OUT_MAX": "BIAS_OUT_MAX",
        "OUTPUT_WEIGHT_SCALE_MIN": "OUTPUT_WEIGHT_SCALE_MIN",
        "OUTPUT_WEIGHT_SCALE_MAX": "OUTPUT_WEIGHT_SCALE_MAX",
        "LEARNING_RATE": "LEARNING_RATE",
        "MNIST_INPUT_SCALE": "MNIST_INPUT_SCALE",
        "IMAGE_DURATION": "IMAGE_DURATION",
        "OUTPUT_AVERAGE_WINDOW": "OUTPUT_AVERAGE_WINDOW",
        "TAUPY_MIN": "TAUPY_MIN", "TAUPY_MAX": "TAUPY_MAX",
        "TAUINH_MIN": "TAUINH_MIN", "TAUINH_MAX": "TAUINH_MAX",
        "TAUOUT_MIN": "TAUOUT_MIN", "TAUOUT_MAX": "TAUOUT_MAX",
        "TAU_H_MIN": "TAU_H_MIN", "TAU_H_MAX": "TAU_H_MAX",
        "H_INIT_MIN": "H_INIT_MIN", "H_INIT_MAX": "H_INIT_MAX",
        "BIAS_E_INIT_MIN": "BIAS_E_INIT_MIN", "BIAS_E_INIT_MAX": "BIAS_E_INIT_MAX",
        "BIAS_I_INIT_MIN": "BIAS_I_INIT_MIN", "BIAS_I_INIT_MAX": "BIAS_I_INIT_MAX",
        "MNIST_WEIGHT_MIN": "MNIST_WEIGHT_MIN", "MNIST_WEIGHT_MAX": "MNIST_WEIGHT_MAX",
        "INH2PY_GAIN_MIN": "INH2PY_GAIN_MIN", "INH2PY_GAIN_MAX": "INH2PY_GAIN_MAX",
        "CELOC_INIT_SCALE_MIN": "CELOC_INIT_SCALE_MIN", "CELOC_INIT_SCALE_MAX": "CELOC_INIT_SCALE_MAX",
        "CEREM_INIT_SCALE_MIN": "CEREM_INIT_SCALE_MIN", "CEREM_INIT_SCALE_MAX": "CEREM_INIT_SCALE_MAX",
        "CELOCI_INIT_SCALE_MIN": "CELOCI_INIT_SCALE_MIN", "CELOCI_INIT_SCALE_MAX": "CELOCI_INIT_SCALE_MAX",
        "BATCH_SIZE": "BATCH_SIZE",
        "dt": "DT",
    }

    nb_vals = extract_assignments(cell7, set(name_map))
    for nb_name, cfg_name in sorted(name_map.items()):
        if nb_name not in nb_vals:
            problems.append(f"constant {nb_name}: not found in notebook cell 7")
            continue
        nb_v = nb_vals[nb_name]
        cfg_v = getattr(cfg, cfg_name)
        if abs(float(nb_v) - float(cfg_v)) > 1e-12:
            problems.append(f"constant {nb_name}: notebook={nb_v} != wc_config.{cfg_name}={cfg_v}")
    if not problems:
        print(f"  [OK]  {len(name_map)} sampled-range constants match notebook cell 7")

    # The one intentional, spec-mandated difference (spec §2/§6): the notebook
    # had ALPHA_H_MIN = ALPHA_H_MAX = 0.0 (H-current forced off); the study
    # restores Uniform(1, 10) for the H-on config.  Verify against the spec.
    if (cfg.ALPHA_H_ON_MIN, cfg.ALPHA_H_ON_MAX) == (1.0, 10.0):
        print("  [OK]  alpha_H (H-on) restored to Uniform(1.0, 10.0) per spec §2/§6")
    else:
        problems.append(
            f"alpha_H (H-on) range = ({cfg.ALPHA_H_ON_MIN}, {cfg.ALPHA_H_ON_MAX}); spec requires (1.0, 10.0)"
        )
    return problems


# =============================================================================
# 3. Numeric equivalence of one step() against an inline reference
# =============================================================================
def check_numeric_equivalence() -> list[str]:
    import numpy as np
    import torch
    import wc_model as M
    import wc_config as cfg

    problems: list[str] = []
    dev = torch.device("cpu")
    rng = np.random.default_rng(20260629)

    CeLoc, CeRem, CeLocI = M.build_connectivity(rng, 0.3, 0.25, dev)
    model = M.SparseBigWilsonCowan(
        CeLoc, CeRem, CeLocI, rng=rng,
        bias_e_init=-2.5, bias_i_init=-3.5, bias_out=-3.0,
        output_weight_scale=0.01, tau_H=1.3, alpha_H=4.0, device=dev,
    )

    gen = torch.Generator().manual_seed(7)
    E = torch.rand(model.N, generator=gen)
    I = torch.rand(model.N, generator=gen)
    O = torch.rand(model.num_output, generator=gen)
    H = torch.rand(model.N, generator=gen)
    img = torch.rand(784, generator=gen)
    I_ext = model.compute_mnist_input(img * cfg.MNIST_INPUT_SCALE)
    dt = cfg.DT

    # ---- Ported model ----
    E1, I1, O1, H1 = model.step(E, I, O, I_ext, dt, H=H)

    # ---- Inline reference: the exact cell-14 equations ----
    with torch.no_grad():
        CeLoc_m = model._get_sparse_matrix(model.CeLoc_crow_indices, model.CeLoc_col_indices,
                                           model.CeLoc_values, model.CeLoc_shape)
        CeRem_m = model._get_sparse_matrix(model.CeRem_crow_indices, model.CeRem_col_indices,
                                           model.CeRem_values, model.CeRem_shape)
        CeLocI_m = model._get_sparse_matrix(model.CeLocI_crow_indices, model.CeLocI_col_indices,
                                            model.CeLocI_values, model.CeLocI_shape)

        Py2Py = 10.0 * E + 0.15 * (CeLoc_m @ E) + 0.05 * (CeRem_m @ E)
        Inh2Py = model.inh_to_exc_gain * I
        Py2Inh = 0.1 * (CeLocI_m @ E) + 15.0 * E
        input_e = Py2Py - Inh2Py + I_ext + model.bias_e - model.alpha_H * H
        input_i = Py2Inh + model.bias_i
        input_out = torch.matmul(model.output_weights, E) + model.bias_out

        def sig(x):
            return torch.sigmoid(1.0 * (x - 4.0))

        dE = (-E + sig(input_e)) / model.tauPy_eff
        dI = (-I + sig(input_i)) / model.tauInh_eff
        dO = (-O + sig(input_out)) / model.tauOut_eff
        dH = (-H + E) / model.tau_H
        E_ref = E + dt * dE
        I_ref = I + dt * dI
        O_ref = O + dt * dO
        H_ref = H + dt * dH

    for name, a, b in [("E", E1, E_ref), ("I", I1, I_ref), ("O", O1, O_ref), ("H", H1, H_ref)]:
        if not torch.equal(a, b):
            max_abs = float((a - b).abs().max())
            problems.append(f"numeric: {name}_new differs from reference (max|Δ|={max_abs:.3e})")
    if not problems:
        print("  [OK]  step() output bit-identical to inline cell-14 reference")
    return problems


# =============================================================================
# Driver
# =============================================================================
def main() -> int:
    nb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NOTEBOOK
    if not nb_path.exists():
        print(f"ERROR: notebook not found: {nb_path}")
        return 1

    print(f"A2 model-diff check against: {nb_path.name}\n")
    nb_cells = load_notebook_cells(nb_path)

    problems: list[str] = []
    print("[1] AST equivalence of step()/forward():")
    problems += check_ast_equivalence(nb_cells)
    print("\n[2] Sampled-range constants vs cell 7:")
    problems += check_constants(nb_cells)
    print("\n[3] Numeric equivalence of one step():")
    problems += check_numeric_equivalence()

    print("\n" + "=" * 60)
    if problems:
        print("A2 FAILED — model dynamics differ from the notebook:")
        for p in problems:
            print("  - " + p)
        return 1
    print("A2 PASSED — ported model is semantically identical to v82 cell 14/cell 7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
