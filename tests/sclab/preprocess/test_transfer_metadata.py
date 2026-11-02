import warnings

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData

from sclab.preprocess import propagate_metadata, transfer_metadata
from sclab.preprocess._transfer_metadata import (
    _assign_categorical,
    _assign_numerical,
    _assign_numerical_periodic,
)

from ._graph import graph_adata


@pytest.fixture
def adata() -> AnnData:
    """Cells 0-2 are the labeled source group ``s``; cells 3-4 are targets ``t``.

    Cell 3 has neighbors 0, 1, 2 (connectivities 0.5, 0.5, 1.0).
    Cell 4 has neighbors 1, 2, 3 (cell 3 is a target, so unlabeled).
    """
    edges = {
        (3, 0): (1.0, 0.5),
        (3, 1): (2.0, 0.5),
        (3, 2): (4.0, 1.0),
        (4, 1): (1.0, 0.5),
        (4, 2): (1.0, 0.5),
        (4, 3): (1.0, 1.0),
    }
    adata = graph_adata(np.zeros((5, 1)), edges, k=3)
    adata.obs["group"] = ["s", "s", "s", "t", "t"]
    adata.obs["num"] = [1.0, 2.0, 4.0, np.nan, np.nan]
    adata.obs["cat"] = pd.Categorical(["x", "y", "y", None, None])
    return adata


# ---------------------------------------------------------------------------
# assign helpers
# ---------------------------------------------------------------------------


def test_assign_numerical_weighted_mean_and_error():
    values = pd.Series([1.0, 2.0, 4.0])
    weights = np.array([0.5, 0.5, 1.0])

    mean, err = _assign_numerical(values, weights)

    # hand computed: sum_w = 2, sum_w2 = 1.5, n_eff = 4 / 1.5
    assert mean == pytest.approx(2.75)
    var = (0.5 * 1.75**2 + 0.5 * 0.75**2 + 1.0 * 1.25**2) * 2 / (4 - 1.5)
    assert err == pytest.approx(np.sqrt(var / (4 / 1.5)))


def test_assign_numerical_single_effective_neighbor_has_nan_error():
    mean, err = _assign_numerical(pd.Series([3.0]), np.array([1.0]))
    assert mean == 3.0
    assert np.isnan(err)


def test_assign_numerical_zero_weights_is_nan():
    mean, err = _assign_numerical(pd.Series([3.0, 4.0]), np.array([0.0, 0.0]))
    assert np.isnan(mean) and np.isnan(err)


def test_assign_categorical_weighted_majority():
    values = pd.Series(["x", "y", "y"])
    winner, share = _assign_categorical(values, np.array([0.5, 0.5, 1.0]))
    assert winner == "y"
    assert share == pytest.approx(0.75)


def test_assign_numerical_periodic_wraps_around_boundary():
    values = pd.Series([0.9, 0.1, 0.95])
    mean, _ = _assign_numerical_periodic(values, np.ones(3), vmin=0.0, vmax=1.0)

    # circular mean around 0.9: offsets 0, +0.2, +0.05
    assert mean == pytest.approx(0.9 + 0.25 / 3)
    assert 0.0 <= mean < 1.0


@pytest.mark.parametrize("pair", [(0.95, 0.05), (0.95, 0.10)])
def test_assign_numerical_periodic_two_points_straddling_boundary(pair):
    mean, _ = _assign_numerical_periodic(
        pd.Series(pair), np.ones(2), vmin=0.0, vmax=1.0
    )

    expected = (pair[0] + (pair[1] + 1 - pair[0]) / 2) % 1.0  # 0.0 and 0.025
    d = abs(mean - expected) % 1.0
    assert min(d, 1 - d) < 1e-9


def test_assign_numerical_periodic_balanced_straddle():
    values = pd.Series([0.90, 0.95, 0.02, 0.05])
    mean, _ = _assign_numerical_periodic(values, np.ones(4), vmin=0.0, vmax=1.0)

    assert mean == pytest.approx(0.98, abs=1e-2)


def test_assign_numerical_periodic_uniform_spread_is_nan():
    values = pd.Series([0.0, 0.25, 0.5, 0.75])
    mean, err = _assign_numerical_periodic(values, np.ones(4), vmin=0.0, vmax=1.0)

    assert np.isnan(mean) and np.isnan(err)


# ---------------------------------------------------------------------------
# transfer_metadata
# ---------------------------------------------------------------------------


def test_transfer_numeric_writes_keys_and_values(adata: AnnData):
    transfer_metadata(adata, "group", "s", "num", min_neighs=2)

    assert "transferred_num" in adata.obs
    assert "transferred_num_error" in adata.obs

    t = adata.obs["transferred_num"]
    # source cells keep their value
    np.testing.assert_allclose(t.iloc[:3], [1.0, 2.0, 4.0])
    # cell 3: weighted mean with connectivity weights
    assert t.iloc[3] == pytest.approx(2.75)
    # cell 4: only cells 1, 2 are labeled -> mean (2 + 4) / 2
    assert t.iloc[4] == pytest.approx(3.0)

    err = adata.obs["transferred_num_error"]
    assert err.iloc[:3].isna().all()
    assert err.iloc[3] > 0 and err.iloc[4] > 0


def test_transfer_does_not_modify_source_column(adata: AnnData):
    before = adata.obs["num"].copy()
    transfer_metadata(adata, "group", "s", "num", min_neighs=2)
    pd.testing.assert_series_equal(adata.obs["num"], before)


def test_transfer_min_neighs_leaves_nan(adata: AnnData):
    transfer_metadata(adata, "group", "s", "num", min_neighs=3)

    t = adata.obs["transferred_num"]
    assert t.iloc[3] == pytest.approx(2.75)  # 3 labeled neighbors
    assert np.isnan(t.iloc[4])  # only 2 labeled neighbors
    assert np.isnan(adata.obs["transferred_num_error"].iloc[4])


def test_transfer_default_min_neighs_is_five(adata: AnnData):
    transfer_metadata(adata, "group", "s", "num")
    assert adata.obs["transferred_num"].iloc[3:].isna().all()


def test_transfer_weight_by_constant(adata: AnnData):
    transfer_metadata(adata, "group", "s", "num", min_neighs=2, weight_by="constant")
    assert adata.obs["transferred_num"].iloc[3] == pytest.approx((1 + 2 + 4) / 3)


def test_transfer_weight_by_distance(adata: AnnData):
    transfer_metadata(adata, "group", "s", "num", min_neighs=2, weight_by="distance")

    w = np.array([1 / 1.0, 1 / 2.0, 1 / 4.0])
    expected = (w * np.array([1.0, 2.0, 4.0])).sum() / w.sum()
    assert adata.obs["transferred_num"].iloc[3] == pytest.approx(expected)


def test_transfer_categorical(adata: AnnData):
    transfer_metadata(adata, "group", "s", "cat", min_neighs=2)

    t = adata.obs["transferred_cat"]
    assert isinstance(t.dtype, pd.CategoricalDtype)
    assert t.iloc[:3].tolist() == ["x", "y", "y"]
    assert t.iloc[3] == "y"
    assert t.iloc[4] == "y"

    assert "transferred_cat_proportion" in adata.obs
    assert "transferred_cat_error" not in adata.obs
    prop = adata.obs["transferred_cat_proportion"]
    assert prop.iloc[3] == pytest.approx(0.75)
    assert prop.iloc[4] == pytest.approx(1.0)


def test_transfer_periodic_numeric(adata: AnnData):
    adata.obs["phase"] = [0.9, 0.1, 0.95, np.nan, np.nan]

    transfer_metadata(
        adata,
        "group",
        "s",
        "phase",
        periodic=True,
        vmin=0.0,
        vmax=1.0,
        min_neighs=2,
        weight_by="constant",
    )

    t = adata.obs["transferred_phase"]
    # circular mean across the 1 -> 0 boundary, not the arithmetic 0.65
    assert t.iloc[3] == pytest.approx(0.9 + 0.25 / 3)


def test_transfer_non_periodic_ignores_wrap(adata: AnnData):
    adata.obs["phase"] = [0.9, 0.1, 0.95, np.nan, np.nan]
    transfer_metadata(adata, "group", "s", "phase", min_neighs=2, weight_by="constant")
    assert adata.obs["transferred_phase"].iloc[3] == pytest.approx(
        (0.9 + 0.1 + 0.95) / 3
    )


def test_transfer_boolean_keeps_source_values(adata: AnnData):
    adata.obs["flag"] = [True, False, False, False, False]

    transfer_metadata(adata, "group", "s", "flag", min_neighs=2)

    t = adata.obs["transferred_flag"]
    assert t.iloc[:3].tolist() == [True, False, False]
    assert not t.iloc[3]  # weighted vote 0.5 True vs 1.5 False
    assert t.dtype == "boolean"
    assert adata.obs["flag"].dtype == bool


def test_transfer_boolean_under_neighbored_cells_are_na(adata: AnnData):
    adata.obs["flag"] = [True, False, False, False, False]

    # cell 3 has 3 labeled neighbors, cell 4 only 2
    transfer_metadata(adata, "group", "s", "flag", min_neighs=3)

    t = adata.obs["transferred_flag"]
    assert t.iloc[:3].tolist() == [True, False, False]
    assert not t.iloc[3]
    assert pd.isna(t.iloc[4])


def test_transfer_boolean_emits_no_future_warning(adata: AnnData):
    adata.obs["flag"] = [True, False, False, False, False]

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        transfer_metadata(adata, "group", "s", "flag", min_neighs=2)


def test_transfer_unsupported_dtype_raises(adata: AnnData):
    adata.obs["label"] = ["a", "b", "c", "d", "e"]  # object dtype
    with pytest.raises(ValueError, match="Unsupported dtype"):
        transfer_metadata(adata, "group", "s", "label")


def test_transfer_invalid_weight_by_raises(adata: AnnData):
    with pytest.raises(ValueError, match="weight_by"):
        transfer_metadata(adata, "group", "s", "num", weight_by="bogus")  # type: ignore[arg-type]


def test_transfer_unknown_column_or_group_key_raises(adata: AnnData):
    with pytest.raises(KeyError):
        transfer_metadata(adata, "group", "s", "missing")
    with pytest.raises(KeyError):
        transfer_metadata(adata, "missing", "s", "num")


def test_transfer_without_neighbors_raises():
    adata = AnnData(np.zeros((3, 2)))
    adata.obs["group"] = ["s", "s", "t"]
    adata.obs["num"] = [1.0, 2.0, np.nan]
    with pytest.raises(KeyError):
        transfer_metadata(adata, "group", "s", "num")


def test_transfer_with_scanpy_neighbors_graph():
    rng = np.random.default_rng(0)
    n = 40
    # two well-separated clusters; label = cluster id on the source group only
    X = np.vstack([rng.normal(0, 0.1, (n // 2, 3)), rng.normal(10, 0.1, (n // 2, 3))])
    adata = AnnData(X)
    adata.obs["cluster"] = pd.Categorical(["a"] * (n // 2) + ["b"] * (n // 2))
    adata.obs["group"] = ["s", "t"] * (n // 2)
    sc.pp.neighbors(adata, n_neighbors=8, use_rep="X", method="gauss")

    transfer_metadata(adata, "group", "s", "cluster", min_neighs=1)

    transferred = adata.obs["transferred_cluster"]
    assert transferred.notna().all()
    assert (transferred == adata.obs["cluster"]).all()
    assert (adata.obs["transferred_cluster_proportion"].dropna() == 1.0).all()


# ---------------------------------------------------------------------------
# propagate_metadata
# ---------------------------------------------------------------------------


def test_propagate_numeric_fills_nans_in_place(adata: AnnData):
    propagate_metadata(adata, "num", min_neighs=2)

    num = adata.obs["num"]
    np.testing.assert_allclose(num.iloc[:3], [1.0, 2.0, 4.0])
    assert num.iloc[3] == pytest.approx(2.75)
    assert num.iloc[4] == pytest.approx(3.0)

    err = adata.obs["num_error"]
    assert err.iloc[:3].isna().all()
    assert err.iloc[3:].notna().all()


def test_propagate_does_not_chain_through_filled_cells(adata: AnnData):
    # cell 4 neighbors cell 3, which is unlabeled at call time; with
    # min_neighs=3 cell 4 only sees 2 labeled neighbors and stays NaN even
    # though cell 3 is filled in the same call
    propagate_metadata(adata, "num", min_neighs=3)
    assert not np.isnan(adata.obs["num"].iloc[3])
    assert np.isnan(adata.obs["num"].iloc[4])


def test_propagate_default_min_neighs_leaves_nan(adata: AnnData):
    propagate_metadata(adata, "num")
    assert adata.obs["num"].iloc[3:].isna().all()
    np.testing.assert_allclose(adata.obs["num"].iloc[:3], [1.0, 2.0, 4.0])


def test_propagate_categorical(adata: AnnData):
    propagate_metadata(adata, "cat", min_neighs=2)

    cat = adata.obs["cat"]
    assert isinstance(cat.dtype, pd.CategoricalDtype)
    assert cat.tolist() == ["x", "y", "y", "y", "y"]
    assert "cat_proportion" in adata.obs
    assert adata.obs["cat_proportion"].iloc[3] == pytest.approx(0.75)


def test_propagate_periodic(adata: AnnData):
    adata.obs["phase"] = [0.9, 0.1, 0.95, np.nan, np.nan]

    propagate_metadata(
        adata,
        "phase",
        periodic=True,
        min_neighs=3,
        weight_by="constant",
    )

    assert adata.obs["phase"].iloc[3] == pytest.approx(0.9 + 0.25 / 3)


def test_propagate_fully_labeled_is_noop(adata: AnnData):
    adata.obs["num"] = [1.0, 2.0, 4.0, 5.0, 6.0]
    propagate_metadata(adata, "num", min_neighs=1)

    np.testing.assert_allclose(adata.obs["num"], [1.0, 2.0, 4.0, 5.0, 6.0])
    # no cell needed filling, so no cell has an error estimate
    assert adata.obs["num_error"].isna().all()


def test_propagate_unsupported_dtype_raises(adata: AnnData):
    adata.obs["label"] = ["a", "b", "c", None, None]
    with pytest.raises(ValueError, match="Unsupported dtype"):
        propagate_metadata(adata, "label")


def test_propagate_unknown_column_raises(adata: AnnData):
    with pytest.raises(KeyError):
        propagate_metadata(adata, "missing")
