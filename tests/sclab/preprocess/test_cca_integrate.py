from functools import partial

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from sclab.preprocess import cca_integrate, cca_integrate_pair
from sclab.preprocess import _cca_integrate as cca_integrate_module
from sclab.preprocess._cca import cca
from sclab.preprocess._cca_integrate import mcca_integrate
from sclab.preprocess._mcca import mcca_sample_space

N_OBS, N_VARS = 60, 20


@pytest.fixture
def adata() -> AnnData:
    """Two interleaved batches sharing a 3-d latent structure."""
    rng = np.random.default_rng(0)
    W = rng.normal(size=(3, N_VARS))
    X = rng.normal(size=(N_OBS, 3)) @ W + 0.1 * rng.normal(size=(N_OBS, N_VARS))
    adata = AnnData(X)
    adata.obs_names = [f"cell{i}" for i in range(N_OBS)]
    adata.obs["batch"] = pd.Categorical(["A", "B"] * (N_OBS // 2))
    adata.obsm["emb"] = rng.normal(size=(N_OBS, 6))
    adata.layers["alt"] = X * 2 + 1
    adata.var["use"] = [True] * 12 + [False] * 8
    return adata


@pytest.fixture
def sparse_serial(monkeypatch):
    """Avoid spawning one process per CPU in the sparse cross-covariance."""
    monkeypatch.setattr(cca_integrate_module, "cca", partial(cca, n_jobs=1))


def idx(adata: AnnData, group: str) -> np.ndarray:
    return np.flatnonzero((adata.obs["batch"] == group).to_numpy())


# ---------------------------------------------------------------------------
# cca_integrate_pair
# ---------------------------------------------------------------------------


def test_pair_writes_default_keys(adata: AnnData):
    cca_integrate_pair(adata, "batch", "A", "B", n_components=5, svd_solver="full")

    assert adata.obsm["X_cca"].shape == (N_OBS, 5)
    assert not np.isnan(adata.obsm["X_cca"]).any()

    uns = adata.uns["X_cca"]
    assert set(uns) == {"A-B", "A", "B"}
    assert uns["A-B"]["sigma"].shape == (5,)
    assert np.all(np.diff(uns["A-B"]["sigma"]) <= 0)
    assert uns["A"].shape == (N_OBS // 2, 5)
    assert uns["B"].shape == (N_OBS // 2, 5)


def test_pair_default_basis_name_follows_basis(adata: AnnData):
    cca_integrate_pair(adata, "batch", "A", "B", basis="emb", n_components=3)
    assert "emb_cca" in adata.obsm
    assert "emb_cca" in adata.uns

    cca_integrate_pair(adata, "batch", "A", "B", basis="alt", n_components=3)
    assert "alt_cca" in adata.obsm


def test_pair_custom_adjusted_basis(adata: AnnData):
    cca_integrate_pair(adata, "batch", "A", "B", adjusted_basis="mine", n_components=3)
    assert "mine" in adata.obsm
    assert "X_cca" not in adata.obsm


def test_pair_rows_are_scattered_in_original_cell_order(adata: AnnData):
    cca_integrate_pair(
        adata, "batch", "A", "B", n_components=4, svd_solver="full", normalize=False
    )

    ia, ib = idx(adata, "A"), idx(adata, "B")
    U, s, V = cca(adata.X[ia], adata.X[ib], n_components=4, svd_solver="full")

    np.testing.assert_allclose(adata.obsm["X_cca"][ia], U)
    np.testing.assert_allclose(adata.obsm["X_cca"][ib], V)
    np.testing.assert_allclose(adata.uns["X_cca"]["A-B"]["sigma"], s)


def test_pair_normalize_gives_unit_rows(adata: AnnData):
    cca_integrate_pair(
        adata, "batch", "A", "B", n_components=4, svd_solver="full", normalize=True
    )
    norms = np.linalg.norm(adata.obsm["X_cca"], axis=1)
    np.testing.assert_allclose(norms, 1.0)


def test_pair_normalize_false_gives_orthonormal_columns_per_group(adata: AnnData):
    cca_integrate_pair(
        adata, "batch", "A", "B", n_components=4, svd_solver="full", normalize=False
    )
    for g in "AB":
        E = adata.obsm["X_cca"][idx(adata, g)]
        np.testing.assert_allclose(E.T @ E, np.eye(4), atol=1e-8)


def test_pair_mask_var_restricts_features(adata: AnnData):
    cca_integrate_pair(
        adata,
        "batch",
        "A",
        "B",
        mask_var="use",
        n_components=3,
        svd_solver="full",
        normalize=False,
    )

    ia, ib = idx(adata, "A"), idx(adata, "B")
    X = adata.X[:, :12]
    U, _, V = cca(X[ia], X[ib], n_components=3, svd_solver="full")
    np.testing.assert_allclose(adata.obsm["X_cca"][ia], U)
    np.testing.assert_allclose(adata.obsm["X_cca"][ib], V)


def test_pair_layer_basis(adata: AnnData):
    # z-scoring makes the dense CCA invariant to the affine layer transform
    cca_integrate_pair(
        adata,
        "batch",
        "A",
        "B",
        basis="alt",
        n_components=3,
        svd_solver="full",
        adjusted_basis="from_layer",
    )
    cca_integrate_pair(
        adata,
        "batch",
        "A",
        "B",
        n_components=3,
        svd_solver="full",
        adjusted_basis="from_X",
    )
    np.testing.assert_allclose(
        np.abs(adata.obsm["from_layer"]), np.abs(adata.obsm["from_X"]), atol=1e-6
    )


def test_pair_obsm_basis(adata: AnnData):
    cca_integrate_pair(
        adata, "batch", "A", "B", basis="emb", n_components=3, svd_solver="full"
    )
    assert adata.obsm["emb_cca"].shape == (N_OBS, 3)


def test_pair_sparse_input(adata: AnnData, sparse_serial):
    adata.X = sp.csr_matrix(adata.X)
    cca_integrate_pair(adata, "batch", "A", "B", n_components=3, svd_solver="full")

    assert adata.obsm["X_cca"].shape == (N_OBS, 3)
    assert np.isfinite(adata.obsm["X_cca"]).all()


def test_pair_randomized_is_reproducible_with_seed(adata: AnnData):
    kw = dict(n_components=3, svd_solver="randomized", random_state=1)
    cca_integrate_pair(adata, "batch", "A", "B", adjusted_basis="r1", **kw)
    cca_integrate_pair(adata, "batch", "A", "B", adjusted_basis="r2", **kw)
    np.testing.assert_array_equal(adata.obsm["r1"], adata.obsm["r2"])


def test_integrate_is_reproducible_by_default(adata: AnnData):
    cca_integrate(adata, "batch", adjusted_basis="r1", n_components=3)
    cca_integrate(adata, "batch", adjusted_basis="r2", n_components=3)
    np.testing.assert_array_equal(adata.obsm["r1"], adata.obsm["r2"])


def test_pair_leaves_other_groups_nan(adata: AnnData):
    adata.obs["batch3"] = pd.Categorical(["A", "B", "C"] * (N_OBS // 3))

    cca_integrate_pair(adata, "batch3", "A", "B", adjusted_basis="e", n_components=3)

    in_pair = adata.obs["batch3"].isin(["A", "B"]).to_numpy()
    assert np.isnan(adata.obsm["e"][~in_pair]).all()
    assert np.isfinite(adata.obsm["e"][in_pair]).all()
    assert set(adata.uns["e"]) == {"A-B", "A", "B"}


def test_pair_reuses_existing_embedding_of_same_width(adata: AnnData):
    adata.obs["batch3"] = pd.Categorical(["A", "B", "C"] * (N_OBS // 3))
    cca_integrate_pair(adata, "batch3", "A", "B", adjusted_basis="e", n_components=3)
    first = adata.obsm["e"].copy()

    cca_integrate_pair(adata, "batch3", "B", "C", adjusted_basis="e", n_components=3)

    # cells of A are untouched by the second call; B and C are now filled
    a = (adata.obs["batch3"] == "A").to_numpy()
    np.testing.assert_array_equal(adata.obsm["e"][a], first[a])
    assert np.isfinite(adata.obsm["e"]).all()
    assert {"A-B", "B-C"} <= set(adata.uns["e"])


def test_pair_replaces_embedding_with_different_width(adata: AnnData):
    cca_integrate_pair(adata, "batch", "A", "B", adjusted_basis="e", n_components=5)
    cca_integrate_pair(adata, "batch", "A", "B", adjusted_basis="e", n_components=3)
    assert adata.obsm["e"].shape == (N_OBS, 3)


def test_pair_does_not_modify_input_matrix(adata: AnnData):
    X = adata.X.copy()
    cca_integrate_pair(adata, "batch", "A", "B", n_components=3)
    np.testing.assert_array_equal(adata.X, X)


def test_pair_unknown_group_raises(adata: AnnData):
    with pytest.raises(KeyError):
        cca_integrate_pair(adata, "batch", "A", "Z", n_components=3)


def test_pair_unknown_basis_raises(adata: AnnData):
    with pytest.raises(ValueError, match="Unknown basis"):
        cca_integrate_pair(adata, "batch", "A", "B", basis="nope", n_components=3)


def test_pair_invalid_svd_solver_raises(adata: AnnData):
    with pytest.raises(ValueError, match="svd_solver"):
        cca_integrate_pair(adata, "batch", "A", "B", svd_solver="nope")


def test_pair_unknown_key_raises(adata: AnnData):
    with pytest.raises(KeyError):
        cca_integrate_pair(adata, "nope", "A", "B")


def test_pair_n_components_larger_than_group_size_raises_clear_error(
    adata: AnnData,
):
    small = adata[:10].copy()  # 5 cells per batch
    with pytest.raises(ValueError, match="n_components"):
        cca_integrate_pair(small, "batch", "A", "B", n_components=8)


def test_pair_partial_solver_requires_n_components_below_group_size(
    adata: AnnData,
):
    small = adata[:10].copy()  # 5 cells per batch
    with pytest.raises(ValueError, match="partial"):
        cca_integrate_pair(
            small, "batch", "A", "B", n_components=5, svd_solver="partial"
        )


# ---------------------------------------------------------------------------
# cca_integrate
# ---------------------------------------------------------------------------


def test_integrate_two_groups_matches_pair(adata: AnnData):
    kw = dict(n_components=3, svd_solver="full")
    cca_integrate(adata, "batch", adjusted_basis="auto", **kw)

    first, second = adata.obs["batch"].unique()[:2]
    cca_integrate_pair(adata, "batch", first, second, adjusted_basis="pair", **kw)

    np.testing.assert_allclose(adata.obsm["auto"], adata.obsm["pair"])
    assert "auto" in adata.uns


def test_integrate_pair_order_follows_first_appearance(adata: AnnData):
    adata.obs["batch"] = pd.Categorical(["B", "A"] * (N_OBS // 2))
    cca_integrate(adata, "batch", n_components=3)

    assert "B-A" in adata.uns["X_cca"]
    assert "A-B" not in adata.uns["X_cca"]


def test_integrate_more_than_two_groups_not_implemented(adata: AnnData):
    adata.obs["batch3"] = pd.Categorical(["A", "B", "C"] * (N_OBS // 3))
    with pytest.raises(NotImplementedError):
        cca_integrate(adata, "batch3")


def test_integrate_single_group_not_implemented(adata: AnnData):
    adata.obs["one"] = "A"
    with pytest.raises(NotImplementedError):
        cca_integrate(adata, "one")


def test_integrate_reference_batch_raises(adata: AnnData):
    with pytest.raises(ValueError, match="reference_batch is not supported"):
        cca_integrate(adata, "batch", reference_batch="B", n_components=3)
    assert "X_cca" not in adata.obsm


# ---------------------------------------------------------------------------
# mcca_integrate (not exported from sclab.preprocess)
# ---------------------------------------------------------------------------


@pytest.fixture
def adata3(adata: AnnData) -> AnnData:
    adata.obs["batch3"] = pd.Categorical(["A", "B", "C"] * (N_OBS // 3))
    return adata


def test_mcca_integrate_writes_embedding_and_metadata(adata3: AnnData):
    mcca_integrate(adata3, "batch3", n_components=4, n_jobs=1)

    assert adata3.obsm["X_mcca"].shape == (N_OBS, 4)
    assert np.isfinite(adata3.obsm["X_mcca"]).all()
    np.testing.assert_allclose(np.linalg.norm(adata3.obsm["X_mcca"], axis=1), 1.0)

    uns = adata3.uns["X_mcca"]
    assert uns["method"] == "mcca_sample_space"
    assert sorted(uns["groups"]) == ["A", "B", "C"]
    assert uns["eigvals"].shape == (4,)
    assert np.all(np.diff(uns["eigvals"]) <= 0)


def test_mcca_integrate_scatters_rows_back_to_cell_order(adata3: AnnData):
    mcca_integrate(adata3, "batch3", n_components=3, normalize=False, n_jobs=1)

    groups = adata3.uns["X_mcca"]["groups"]
    ixs = [np.flatnonzero((adata3.obs["batch3"] == g).to_numpy()) for g in groups]
    Z, _ = mcca_sample_space(
        [adata3.X[i] for i in ixs], n_components=3, normalize=False, n_jobs=1
    )

    # eigenvectors are defined up to sign
    got = np.vstack([adata3.obsm["X_mcca"][i] for i in ixs])
    np.testing.assert_allclose(np.abs(got), np.abs(Z), atol=1e-6)


def test_mcca_integrate_custom_basis_and_mask(adata3: AnnData):
    mcca_integrate(
        adata3,
        "batch3",
        basis="emb",
        adjusted_basis="mine",
        n_components=2,
        n_jobs=1,
    )
    assert adata3.obsm["mine"].shape == (N_OBS, 2)

    mcca_integrate(adata3, "batch3", mask_var="use", n_components=2, n_jobs=1)
    assert adata3.obsm["X_mcca"].shape == (N_OBS, 2)


def test_mcca_integrate_single_group_raises(adata: AnnData):
    adata.obs["one"] = "A"
    with pytest.raises(ValueError, match="at least two groups"):
        mcca_integrate(adata, "one", n_jobs=1)


def test_mcca_integrate_unknown_basis_raises(adata3: AnnData):
    with pytest.raises(ValueError, match="Unknown basis"):
        mcca_integrate(adata3, "batch3", basis="nope", n_jobs=1)
