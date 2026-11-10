"""edgePython without numba: sclab's NumPy kernels against edgePython's numba ones."""

import copy
import importlib
import pickle
import subprocess
import sys
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sclab.tools.differential_expression import _edgepython_kernels as K
from sclab.tools.differential_expression import _edgepython_numba_free as shim
from sclab.tools.differential_expression import (
    edgepython_is_available,
    pseudobulk_edgepython,
)

pytestmark = pytest.mark.skipif(
    not edgepython_is_available() or find_spec("numba") is None,
    reason="needs edgepython with numba (the reference)",
)


def _counts(n_genes=800, n_groups=4, n_reps=6, seed=0):
    """Negative binomial pseudobulk counts, 10% of genes DE between groups."""
    rng = np.random.default_rng(seed)
    mu = np.exp(rng.normal(1.5, 2.2, n_genes))
    lfc = np.zeros((n_genes, n_groups))
    de = rng.random(n_genes) < 0.1
    lfc[de] = rng.normal(0, 1.5, (de.sum(), n_groups))
    lib = rng.uniform(0.5, 2.0, n_groups * n_reps)
    group = np.repeat(np.arange(n_groups), n_reps)
    m = mu[:, None] * np.exp(lfc[:, group]) * lib * 50
    r = 1 / (0.05 + 1.0 / np.sqrt(mu + 1))[:, None]
    return rng.negative_binomial(r, r / (r + m)).astype(float)


def _adata(nested=False, **kw) -> AnnData:
    """Replicas crossed with groups (~ 0 + group + replica) or nested (~ 0 + group)."""
    counts = _counts(**kw)
    n_groups, n_reps = kw.get("n_groups", 4), kw.get("n_reps", 6)
    grp = np.repeat([f"g{i}" for i in range(n_groups)], n_reps)
    replica = np.tile([f"r{i}" for i in range(n_reps)], n_groups)
    obs = pd.DataFrame(
        {
            "grp": pd.Categorical(grp),
            "replica": pd.Categorical(np.char.add(grp, replica) if nested else replica),
        },
        index=[f"s{i}" for i in range(counts.shape[1])],
    )
    adata = AnnData(
        counts.T, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(len(counts))])
    )
    for g in obs["grp"].cat.categories:
        for c in ("pct_expr", "num_expr", "tot_expr"):
            adata.var[f"{c}_{g}"] = 1.0
    return adata


KW = dict(condition_group="g1", reference_group="g0", aggregate=False)
DESIGNS = {
    "crossed": {},
    "nested": dict(nested=True, n_reps=3),
    "tiny": dict(n_groups=2, n_reps=2),
}


@pytest.fixture(scope="module", params=list(DESIGNS))
def captured(request):
    """Inputs of every kernel call during a numba run, and the run's top table."""
    calls = {}
    patched = []
    for mod, name, _, new in shim.KERNELS:
        if new is None or name == "_lowess_fit":  # inlined in numba's caller
            continue
        old = getattr(importlib.import_module(f"edgepython.{mod}"), name)

        def wrap(*args, _old=old, _name=name):
            calls.setdefault(_name, []).append(copy.deepcopy(args))
            return _old(*args)

        for module in [m for k, m in sys.modules.items() if k.startswith("edgepython")]:
            for attr, value in list(vars(module).items()):
                if value is old:
                    patched.append((module, attr, old))
                    setattr(module, attr, wrap)
    try:
        adata = _adata(**DESIGNS[request.param])
        tt = pseudobulk_edgepython(adata, "grp", **KW)["g1_vs_g0"]
    finally:
        for module, attr, old in patched:
            setattr(module, attr, old)
    return request.param, adata, calls, tt


def _as_arrays(result, args):
    if result is None:  # results written into the array arguments
        return [a for a in args if isinstance(a, np.ndarray)]
    return list(result) if isinstance(result, tuple) else [result]


def test_kernels_match_numba(captured):
    design, _, calls, _ = captured
    expected = {
        "fit_leven_mat",
        "adj_profile_ll_mat",
        "_maximize_interpolant_kernel",
        "fit_one_group_mat",
        "_compute_adjust_inner",
        "_hat_values",
        "_find_limits",
    }
    if design != "crossed":  # one-way layouts
        expected = expected - {"fit_leven_mat"} | {
            "fit_one_way_mat",
            "nb_deviance_sum_mat",
        }
    assert expected <= set(calls)
    kernels = {name: (mod, new) for mod, name, _, new in shim.KERNELS}
    for name, arg_list in calls.items():
        mod, new = kernels[name]
        old = getattr(importlib.import_module(f"edgepython.{mod}"), name)
        for args in arg_list[:10]:
            a_old, a_new = copy.deepcopy(args), copy.deepcopy(args)
            r_old = _as_arrays(old(*a_old), a_old)
            r_new = _as_arrays(new(*a_new), a_new)
            for x, y in zip(r_old, r_new, strict=True):
                if name == "fit_leven_mat" and x.dtype == bool:
                    # 'failed' marks genes whose damping search stalls at the
                    # optimum, which X'WX's rounding can decide
                    assert (x != y).mean() < 0.01
                    continue
                rtol = (
                    1e-9 if name in ("adj_profile_ll_mat", "fit_leven_mat") else 1e-12
                )
                np.testing.assert_allclose(y, x, rtol=rtol, atol=1e-12, err_msg=name)


def test_lowess_fit_matches_numba():
    from edgepython.weighted_lowess import _lowess_fit

    rng = np.random.default_rng(0)
    xs = np.sort(rng.normal(size=300))
    ys, ws, rw = rng.normal(size=300), rng.uniform(0.5, 2, 300), rng.uniform(0, 1, 300)
    xs[100:105] = xs[100]  # ties
    for curpt, left, right, dist in [
        (50, 10, 90, 0.8),
        (100, 99, 110, 0.0),
        (0, 0, 40, 1.0),
    ]:
        args = (xs, ys, ws, rw, curpt, left, right, dist)
        assert K._lowess_fit(*args) == pytest.approx(_lowess_fit(*args), rel=1e-12)


def test_bunch_kaufman_pivoting_fallback():
    """Matrices that need interchanges or 2x2 pivots take edgePython's scalar path."""
    from edgepython.glm_kernels import _dsytf2_upper

    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 5, 5))
    a = a + a.transpose(0, 2, 1)  # symmetric, indefinite
    a[:100] = a[:100] @ a[:100].transpose(0, 2, 1)  # positive definite
    a[0] = np.diag([10.0, 1, 1, 1, 1])
    a[0, 3, 4] = a[0, 4, 3] = 3.0  # an interchange
    expected = a.copy()
    for m in expected:
        _dsytf2_upper(m)
    np.testing.assert_allclose(
        K._dsytf2_diagonal(a.copy()),
        np.diagonal(expected, axis1=1, axis2=2),
        rtol=1e-12,
    )


NO_NUMBA = """
import pickle, sys
sys.modules["numba"] = None  # as on Pyodide
from sclab.tools.differential_expression import pseudobulk_edgepython, _edgepython_kernels as K
adata = pickle.loads(sys.stdin.buffer.read())
tt = pseudobulk_edgepython(adata, "grp", **{kw})["g1_vs_g0"]
import edgepython.glm_kernels as gk
assert gk.fit_leven_mat is K.fit_leven_mat
sys.stdout.buffer.write(pickle.dumps(tt))
"""


def test_top_table_matches_numba_without_numba(captured):
    """End to end in a process where numba cannot be imported."""
    _, adata, _, expected = captured
    out = subprocess.run(
        [sys.executable, "-c", NO_NUMBA.format(kw=KW)],
        input=pickle.dumps(adata),
        capture_output=True,
        check=True,
    )
    tt = pickle.loads(out.stdout)
    assert list(tt.index) == list(expected.index)
    for col, rtol in [
        ("logFC", 1e-8),
        ("logCPM", 1e-10),
        ("PValue", 1e-6),
        ("FDR", 1e-6),
    ]:
        np.testing.assert_allclose(
            tt[col], expected[col], rtol=rtol, atol=1e-300, err_msg=col
        )


def test_numba_users_unaffected():
    edgepython = shim.import_edgepython()
    assert type(edgepython.glm_kernels.fit_leven_mat).__module__.startswith("numba")
    assert "sclab" not in (sys.modules["numba"].__doc__ or "")


@pytest.mark.parametrize(
    "change",
    [
        lambda m: delattr(m, "fit_leven_mat"),
        lambda m: setattr(m, "adj_profile_ll_mat", lambda y, mu, disp: None),
    ],
    ids=["missing", "signature"],
)
def test_changed_edgepython_fails_loudly(change, monkeypatch):
    import edgepython.glm_kernels as gk

    before = {k: v for k, v in vars(gk).items() if not k.startswith("__")}
    with monkeypatch.context() as mp:
        change(_Proxy(mp, gk))
        with pytest.raises(ImportError, match="install numba"):
            shim._patch()
    assert {k: v for k, v in vars(gk).items() if not k.startswith("__")} == before


class _Proxy:
    """delattr/setattr on a module through monkeypatch (undone after the test)."""

    def __init__(self, mp, module):
        object.__setattr__(self, "_mp", mp)
        object.__setattr__(self, "_m", module)

    def __delattr__(self, name):
        self._mp.delattr(self._m, name)

    def __setattr__(self, name, value):
        self._mp.setattr(self._m, name, value)
