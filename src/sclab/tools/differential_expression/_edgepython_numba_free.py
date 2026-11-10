"""Import edgePython where numba is not available (Pyodide / WebAssembly).

edgePython compiles its per-gene loops with numba, which has no WebAssembly
build. Without numba, :func:`import_edgepython` puts a stand-in ``numba``
module in ``sys.modules`` (``njit`` returns the function unchanged), imports
edgePython, and replaces the kernels that the pseudobulk pipeline reaches with
the vectorised NumPy versions in :mod:`._edgepython_kernels` (plain-Python
loops over genes would take hours). With numba installed nothing is changed.
"""

import inspect
import sys
import types
from importlib.util import find_spec

from . import _edgepython_kernels as K

# (module, kernel, its parameters in edgePython 0.5.1, NumPy replacement)
KERNELS = [
    ("glm_kernels", "fit_leven_mat", "y offset disp w design beta_start maxit tol", K.fit_leven_mat),
    ("glm_kernels", "fit_one_group_mat", "y offset disp w maxit tol beta_start", K.fit_one_group_mat),
    ("glm_kernels", "fit_one_way_mat", "y offset disp w group ngroups maxit tol beta_start", K.fit_one_way_mat),
    ("glm_kernels", "nb_deviance_sum_mat", "y mu disp w", K.nb_deviance_sum_mat),
    ("glm_kernels", "adj_profile_ll_mat", "y mu disp w design adjust", K.adj_profile_ll_mat),
    ("glm_kernels", "_dsytf2_upper", "a", None),  # used by K's fallback, not replaced
    ("dispersion_lowlevel", "_maximize_interpolant_kernel", "x y_mat ngenes npts result", K._maximize_interpolant_kernel),
    ("ql_weights", "_compute_adjust_inner", "y mu design disp_mat w_mat prior dev_out df_out s2_out", K._compute_adjust_inner),
    ("ql_weights", "_hat_values", "design zwpt", K._hat_values),
    ("weighted_lowess", "_find_limits", "seed_idx nseeds xs ws n span_weight", K._find_limits),
    # called by name from the plain-Python _lowess_iterations
    ("weighted_lowess", "_lowess_fit", "xs ys ws rw curpt left right dist", K._lowess_fit),
]  # fmt: skip
TABLES = ["_pois_alpha_weights", "_pois_kappa_weights"] + [
    f"_nb_{ak}_{case}_{i}" for ak in "ak" for case in (1, 2) for i in (1, 2, 3)
    if not (case == 2 and i == 3)
]  # fmt: skip

_patched = False


def _numba_stub() -> types.ModuleType:
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

    stub = types.ModuleType("numba")
    stub.__doc__ = "sclab stand-in: numba is not installed, njit is a no-op."
    stub.njit = stub.jit = njit
    stub.prange = range
    return stub


def _has_numba() -> bool:
    try:
        return find_spec("numba") is not None
    except ValueError:  # a stand-in module without a spec, as ours
        return False


def import_edgepython():
    """``import edgepython``, with NumPy kernels when numba is not installed."""
    global _patched
    if not _patched and not _has_numba():
        if sys.modules.get("numba") is None:  # absent, or blocked with None
            sys.modules["numba"] = _numba_stub()
        _patch()
        _patched = True
    import edgepython

    return edgepython


def _patch():
    import importlib

    import edgepython

    def changed(what):
        return ImportError(
            f"edgepython {getattr(edgepython, '__version__', '?')}: {what}. sclab "
            "runs edgepython without numba by replacing kernels written for "
            "edgepython 0.5.1; install numba, or an edgepython version sclab supports."
        )

    q = importlib.import_module("edgepython.ql_weights")
    for name in TABLES:
        if not hasattr(q, name):
            raise changed(f"edgepython.ql_weights.{name} not found")
    swaps = []
    for mod, name, params, new in KERNELS:
        old = getattr(importlib.import_module(f"edgepython.{mod}"), name, None)
        if old is None:
            raise changed(f"edgepython.{mod}.{name} not found")
        found = " ".join(inspect.signature(old).parameters)
        if found != params:
            raise changed(f"edgepython.{mod}.{name}({found}) is not ({params})")
        if new is not None:
            swaps.append((old, new))
    # rebind every module-level reference (from-imports included)
    for modname, module in list(sys.modules.items()):
        if modname == "edgepython" or modname.startswith("edgepython."):
            for attr, value in list(vars(module).items()):
                for old, new in swaps:
                    if value is old:
                        setattr(module, attr, new)
