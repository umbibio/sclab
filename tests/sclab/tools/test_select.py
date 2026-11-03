import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sclab


@pytest.fixture
def adata() -> AnnData:
    xy = np.array([[0, 0], [1, 1], [2, 2], [1, 3], [np.nan, 1], [3, 0]], float)
    obs = pd.DataFrame(
        {"a": xy[:, 0], "b": pd.array([0, 1, 2, 3, 1, 0], dtype="Int64")},
        index=[f"c{i}" for i in range(6)],
    )
    obs.loc["c4", "b"] = pd.NA
    a = AnnData(np.zeros((6, 1)), obs=obs)
    a.obsm["X_emb"] = np.column_stack([xy, -xy[:, 0]])
    return a


def test_box_edges_inclusive_and_bounds_unordered(adata):
    mask = sclab.tools.select_box(adata, "X_emb", x=(2, 1), y=(1, 2))
    assert mask.tolist() == [False, True, True, False, False, False]


def test_box_single_axis(adata):
    mask = sclab.tools.select_box(adata, "X_emb", x=(1, 3))
    assert mask.tolist() == [False, True, True, True, False, True]
    assert sclab.tools.select_box(adata, "X_emb", x=(-3, -2), dims=(2,)).sum() == 2
    with pytest.raises(ValueError):
        sclab.tools.select_box(adata, "X_emb", x=(0, 1), y=(0, 1), dims=(0,))


def test_box_on_obs_columns_and_dataframe_obsm(adata):
    mask = sclab.tools.select_box(adata, None, x=(0, 1), y=(0, 1), dims=("a", "b"))
    assert mask.tolist() == [True, True, False, False, False, False]
    adata.obsm["df"] = pd.DataFrame(adata.obsm["X_emb"], index=adata.obs_names)
    np.testing.assert_array_equal(
        sclab.tools.select_box(adata, "df", x=(1, 3), y=(0, 2)),
        sclab.tools.select_box(adata, "X_emb", x=(1, 3), y=(0, 2)),
    )


def test_polygon_implicitly_closed_nan_excluded(adata):
    triangle = [(-0.5, -0.5), (2.5, -0.5), (2.5, 2.5)]  # below the diagonal
    mask = sclab.tools.select_polygon(adata, "X_emb", triangle)
    assert mask.tolist() == [True, True, True, False, False, False]
    same = sclab.tools.select_polygon(adata, None, np.array(triangle), dims=("a", "b"))
    np.testing.assert_array_equal(mask, same)


def test_polygon_self_intersecting_uses_even_odd():
    # pentagram drawn in one stroke: its centre is covered twice, so outside
    angles = np.deg2rad(90 + 144 * np.arange(5))
    star = np.column_stack([np.cos(angles), np.sin(angles)])
    a = AnnData(np.zeros((3, 1)))
    a.obsm["X"] = np.array([[0, 0], [0, 0.7], [0, -0.9]])
    assert sclab.tools.select_polygon(a, "X", star).tolist() == [False, True, False]


def test_polygon_validation(adata):
    with pytest.raises(ValueError):
        sclab.tools.select_polygon(adata, "X_emb", [(0, 0), (1, 1)])
    with pytest.raises(ValueError):
        sclab.tools.select_polygon(adata, "X_emb", [(0, 0), (1, 1), (1, 0)], dims=(0,))
