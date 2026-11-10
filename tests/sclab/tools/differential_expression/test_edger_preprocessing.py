"""sclab's filterByExpr + TMM (used without edgePython) against edgePython's."""

import numpy as np
import pandas as pd
import pytest

from sclab.tools.differential_expression import _edger_preprocessing as lite
from sclab.tools.differential_expression import edgepython_is_available
from sclab.tools.differential_expression._edgepython_numba_free import (
    import_edgepython,
)

pytestmark = pytest.mark.skipif(
    not edgepython_is_available(), reason="edgepython not installed"
)


def _counts(n_genes, n_groups, n_reps, seed):
    rng = np.random.default_rng(seed)
    mu = np.exp(rng.normal(1.5, 2.2, n_genes))  # many lowly expressed genes
    lib = rng.uniform(0.5, 2.0, n_groups * n_reps)
    counts = rng.poisson(mu[:, None] * lib * 20).astype(float)
    counts[:5] = 0  # all-zero genes
    return counts


def _designs(n_groups, n_reps):
    group = pd.Series(np.repeat([f"g{i}" for i in range(n_groups)], n_reps))
    replica = pd.Series(np.tile([f"r{i}" for i in range(n_reps)], n_groups))
    nested = pd.get_dummies(group).astype(float)
    designs = {"none": None, "nested": nested.values}
    if n_reps > 1:
        crossed = nested.join(pd.get_dummies(replica, drop_first=True).astype(float))
        designs["crossed"] = crossed.values
    return designs


# (groups, replicas): typical, tiny, a single replica, many samples
SHAPES = [(4, 6), (2, 2), (3, 1), (2, 1), (6, 8), (2, 12)]


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_matches_edgepython(shape, seed):
    edgepython = import_edgepython()
    counts = _counts(3000, *shape, seed=seed)
    for name, design in _designs(*shape).items():
        expected = np.asarray(edgepython.filter_by_expr(counts, design=design))
        keep = lite.filter_by_expr(counts, design=design)
        np.testing.assert_array_equal(keep, expected, err_msg=name)
        assert 0 < keep.sum() < len(keep)
    kept = counts[keep]
    np.testing.assert_array_equal(
        lite.calc_norm_factors(kept), edgepython.calc_norm_factors(kept)
    )


def test_degenerate_norm_factors():
    edgepython = import_edgepython()
    for counts in (np.zeros((10, 4)), np.arange(10.0)[:, None], np.ones((10, 3))):
        np.testing.assert_array_equal(
            lite.calc_norm_factors(counts), edgepython.calc_norm_factors(counts)
        )
