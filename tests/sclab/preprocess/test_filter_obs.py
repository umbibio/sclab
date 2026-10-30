import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from sclab.preprocess import filter_obs

# row sums:        10, 2, 6, 0, 8
# genes detected:   3, 1, 2, 0, 3
X = np.array(
    [
        [5, 3, 2, 0],
        [0, 2, 0, 0],
        [3, 0, 3, 0],
        [0, 0, 0, 0],
        [4, 0, 2, 2],
    ],
    dtype=float,
)


@pytest.fixture(params=["dense", "sparse"])
def adata(request) -> AnnData:
    data = csr_matrix(X) if request.param == "sparse" else X.copy()
    adata = AnnData(data)
    adata.obs_names = ["a", "b", "c", "d", "e"]
    adata.obs["tag"] = list("vwxyz")
    adata.layers["counts"] = X * 10
    return adata


def kept(adata: AnnData) -> list[str]:
    return adata.obs_names.tolist()


def test_no_criteria_is_noop(adata: AnnData):
    filter_obs(adata)
    assert kept(adata) == list("abcde")


def test_returns_none(adata: AnnData):
    assert filter_obs(adata, min_counts=5) is None


def test_min_counts(adata: AnnData):
    filter_obs(adata, min_counts=5)
    assert kept(adata) == ["a", "c", "e"]


def test_min_counts_is_inclusive(adata: AnnData):
    filter_obs(adata, min_counts=6)
    assert kept(adata) == ["a", "c", "e"]


def test_min_genes(adata: AnnData):
    filter_obs(adata, min_genes=2)
    assert kept(adata) == ["a", "c", "e"]


def test_min_genes_strict_threshold(adata: AnnData):
    filter_obs(adata, min_genes=3)
    assert kept(adata) == ["a", "e"]


def test_max_counts(adata: AnnData):
    filter_obs(adata, max_counts=8)
    assert kept(adata) == ["b", "c", "d", "e"]


def test_max_cells_keeps_highest_counts(adata: AnnData):
    filter_obs(adata, max_cells=2)
    assert kept(adata) == ["a", "e"]


def test_max_cells_larger_than_n_obs_keeps_all(adata: AnnData):
    filter_obs(adata, max_cells=100)
    assert kept(adata) == list("abcde")


def test_max_cells_ties_at_boundary_are_all_kept():
    # ranks use method="min": cells tied at the cutoff are all retained
    adata = AnnData(np.array([[5.0], [3.0], [3.0], [1.0]]))
    adata.obs_names = ["a", "b", "c", "d"]
    filter_obs(adata, max_cells=2)
    assert kept(adata) == ["a", "b", "c"]


def test_criteria_are_combined(adata: AnnData):
    # min_counts removes b, d; max_counts removes a
    filter_obs(adata, min_counts=3, max_counts=9)
    assert kept(adata) == ["c", "e"]


def test_criteria_are_evaluated_on_original_data(adata: AnnData):
    # max_cells ranks all cells, even those removed by other criteria:
    # top-2 by counts are a (10) and e (8); min_counts=9 then removes e
    filter_obs(adata, max_cells=2, min_counts=9)
    assert kept(adata) == ["a"]


def test_layer_is_used_for_counts(adata: AnnData):
    # layer is 10x X, so thresholds scale accordingly
    filter_obs(adata, layer="counts", min_counts=50)
    assert kept(adata) == ["a", "c", "e"]

    adata2 = adata.copy()
    # same threshold on X would remove everything
    filter_obs(adata2, min_counts=50)
    assert adata2.n_obs == 0


def test_subsets_obs_layers_and_matrix_consistently(adata: AnnData):
    filter_obs(adata, min_counts=5)

    assert adata.n_obs == 3
    assert adata.obs["tag"].tolist() == ["v", "x", "z"]
    assert adata.layers["counts"].shape == (3, 4)
    dense_X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    np.testing.assert_array_equal(dense_X, X[[0, 2, 4]])
    np.testing.assert_array_equal(adata.layers["counts"], X[[0, 2, 4]] * 10)


def test_removing_everything_gives_empty_adata(adata: AnnData):
    filter_obs(adata, min_counts=1000)
    assert adata.n_obs == 0
    assert adata.n_vars == 4


def test_single_cell_adata():
    adata = AnnData(np.array([[1.0, 0.0, 0.0]]))
    filter_obs(adata, min_genes=2)
    assert adata.n_obs == 0

    adata = AnnData(np.array([[1.0, 1.0, 0.0]]))
    filter_obs(adata, min_genes=2, min_counts=2)
    assert adata.n_obs == 1


def test_unknown_layer_raises(adata: AnnData):
    with pytest.raises(KeyError):
        filter_obs(adata, layer="nope", min_counts=1)
