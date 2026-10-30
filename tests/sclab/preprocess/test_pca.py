import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

from sclab.preprocess import pca

N_OBS, N_VARS, N_COMPS = 40, 12, 4


@pytest.fixture
def adata() -> AnnData:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N_OBS, N_VARS))
    X[: N_OBS // 2] += 3.0  # batch effect on the first half
    adata = AnnData(X)
    adata.obs["batch"] = ["ref"] * (N_OBS // 2) + ["other"] * (N_OBS // 2)
    adata.layers["alt"] = rng.normal(size=X.shape) * 5 + 1
    adata.var["use"] = [True] * 6 + [False] * 6
    return adata


def test_writes_expected_keys_and_shapes(adata: AnnData):
    pca(adata, n_comps=N_COMPS)

    assert adata.obsm["X_pca"].shape == (N_OBS, N_COMPS)
    assert adata.varm["PCs"].shape == (N_VARS, N_COMPS)
    assert adata.uns["pca"]["variance"].shape == (N_COMPS,)
    assert adata.uns["pca"]["variance_ratio"].shape == (N_COMPS,)
    assert "reference_batch" not in adata.uns["pca"]


def test_pcs_match_scanpy(adata: AnnData):
    ref = adata.copy()
    sc.pp.pca(ref, n_comps=N_COMPS, svd_solver="arpack")

    pca(adata, n_comps=N_COMPS)

    np.testing.assert_allclose(adata.varm["PCs"], ref.varm["PCs"])
    np.testing.assert_allclose(adata.uns["pca"]["variance"], ref.uns["pca"]["variance"])


def test_pcs_are_orthonormal_and_variance_sorted(adata: AnnData):
    pca(adata, n_comps=N_COMPS)

    PCs = adata.varm["PCs"]
    np.testing.assert_allclose(PCs.T @ PCs, np.eye(N_COMPS), atol=1e-6)
    var = adata.uns["pca"]["variance"]
    assert np.all(np.diff(var) <= 0)


def test_embedding_is_projection_of_X_on_pcs(adata: AnnData):
    pca(adata, n_comps=N_COMPS)
    np.testing.assert_allclose(adata.obsm["X_pca"], adata.X @ adata.varm["PCs"])


def test_zero_center_centers_embedding_only(adata: AnnData):
    pca(adata, n_comps=N_COMPS)
    uncentered = adata.obsm["X_pca"].copy()
    # the data was shifted, so the raw projection is not centered
    assert np.abs(uncentered.mean(axis=0)).max() > 1e-3

    pca(adata, n_comps=N_COMPS, zero_center=True)
    centered = adata.obsm["X_pca"]

    np.testing.assert_allclose(centered.mean(axis=0), 0, atol=1e-8)
    np.testing.assert_allclose(centered, uncentered - uncentered.mean(axis=0))


def test_mask_var_restricts_loadings(adata: AnnData):
    pca(adata, n_comps=N_COMPS, mask_var="use")

    PCs = adata.varm["PCs"]
    assert PCs.shape == (N_VARS, N_COMPS)
    assert np.all(PCs[6:] == 0)
    assert np.abs(PCs[:6]).max() > 0
    # embedding only depends on the masked genes
    np.testing.assert_allclose(adata.obsm["X_pca"], adata.X[:, :6] @ PCs[:6])


def test_sparse_input(adata: AnnData):
    sparse = adata.copy()
    sparse.X = csr_matrix(sparse.X)
    pca(sparse, n_comps=N_COMPS)

    assert sparse.obsm["X_pca"].shape == (N_OBS, N_COMPS)
    assert np.isfinite(sparse.obsm["X_pca"]).all()
    # sparse path is not zero-centered, so only check that it is a valid orthonormal basis
    PCs = sparse.varm["PCs"]
    np.testing.assert_allclose(PCs.T @ PCs, np.eye(N_COMPS), atol=1e-6)


def test_reference_batch_fits_on_reference_cells_only(adata: AnnData):
    ref_only = adata[adata.obs["batch"] == "ref"].copy()
    sc.pp.pca(ref_only, n_comps=N_COMPS, svd_solver="arpack")

    pca(adata, n_comps=N_COMPS, batch_key="batch", reference_batch="ref")

    np.testing.assert_allclose(adata.varm["PCs"], ref_only.varm["PCs"])
    assert adata.uns["pca"]["reference_batch"] == "ref"
    # all cells (not only reference cells) are projected
    assert adata.obsm["X_pca"].shape == (N_OBS, N_COMPS)
    np.testing.assert_allclose(adata.obsm["X_pca"], adata.X @ adata.varm["PCs"])


def test_reference_batch_differs_from_global_pca(adata: AnnData):
    glob = adata.copy()
    pca(glob, n_comps=N_COMPS)
    pca(adata, n_comps=N_COMPS, batch_key="batch", reference_batch="ref")

    assert not np.allclose(np.abs(glob.varm["PCs"]), np.abs(adata.varm["PCs"]))


def test_reference_batch_without_batch_key_raises(adata: AnnData):
    with pytest.raises((KeyError, ValueError)):
        pca(adata, n_comps=N_COMPS, reference_batch="ref")


def test_unknown_reference_batch_raises(adata: AnnData):
    with pytest.raises(ValueError):
        pca(adata, n_comps=N_COMPS, batch_key="batch", reference_batch="nope")


def test_too_many_components_raises(adata: AnnData):
    with pytest.raises(ValueError, match="n_components"):
        pca(adata, n_comps=N_VARS + 5)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: with layer=..., PCs are fitted on the layer but X_pca is "
    "computed from adata.X",
)
def test_layer_embedding_is_projection_of_layer(adata: AnnData):
    pca(adata, layer="alt", n_comps=N_COMPS, zero_center=True)

    L = adata.layers["alt"]
    expected = (L - L.mean(axis=0)) @ adata.varm["PCs"]
    np.testing.assert_allclose(adata.obsm["X_pca"], expected, atol=1e-8)
