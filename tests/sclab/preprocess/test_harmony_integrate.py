import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sclab.preprocess import harmony_integrate
from sclab.preprocess._harmony import moe_correct_ridge, run_harmony, safe_entropy

N_OBS = 200


@pytest.fixture
def batch_idx() -> np.ndarray:
    return np.random.default_rng(1).integers(0, 2, N_OBS)


@pytest.fixture
def adata(batch_idx: np.ndarray) -> AnnData:
    """Two clusters (separated along dim 0) with a batch shift along dim 1."""
    rng = np.random.default_rng(0)
    cluster = rng.integers(0, 2, N_OBS)
    centers = np.array([[0, 0, 0, 0, 0], [6, 0, 0, 0, 0]], dtype=float)
    Z = centers[cluster] + rng.normal(size=(N_OBS, 5)) * 0.5
    Z[batch_idx == 1] += np.array([0, 5, 0, 0, 0.0])

    adata = AnnData(np.zeros((N_OBS, 3)))
    adata.obsm["X_pca"] = Z
    adata.obs["batch"] = pd.Categorical(np.where(batch_idx == 0, "A", "B"))
    adata.obs["cluster"] = cluster
    return adata


def batch_gap(E: np.ndarray, batch_idx: np.ndarray) -> float:
    return float(np.linalg.norm(E[batch_idx == 0].mean(0) - E[batch_idx == 1].mean(0)))


# ---------------------------------------------------------------------------
# harmony_integrate
# ---------------------------------------------------------------------------


def test_writes_default_key_and_keeps_input(adata: AnnData):
    original = adata.obsm["X_pca"].copy()
    harmony_integrate(adata, "batch", verbose=False)

    assert adata.obsm["X_pca_harmony"].shape == original.shape
    assert np.isfinite(adata.obsm["X_pca_harmony"]).all()
    np.testing.assert_array_equal(adata.obsm["X_pca"], original)


def test_custom_basis_and_adjusted_basis(adata: AnnData):
    adata.obsm["other"] = adata.obsm["X_pca"][:, :3].copy()

    harmony_integrate(adata, "batch", basis="other", verbose=False)
    assert adata.obsm["other_harmony"].shape == (N_OBS, 3)

    harmony_integrate(adata, "batch", adjusted_basis="mine", verbose=False)
    assert adata.obsm["mine"].shape == (N_OBS, 5)


def test_reduces_batch_separation_and_keeps_cluster_structure(
    adata: AnnData, batch_idx: np.ndarray
):
    before = batch_gap(adata.obsm["X_pca"], batch_idx)
    harmony_integrate(adata, "batch", verbose=False)
    E = adata.obsm["X_pca_harmony"]

    assert batch_gap(E, batch_idx) < 0.5 * before

    # clusters remain separated along dim 0
    c = adata.obs["cluster"].to_numpy()
    assert abs(E[c == 0, 0].mean() - E[c == 1, 0].mean()) > 3


def test_is_deterministic_given_random_state(adata: AnnData):
    harmony_integrate(adata, "batch", adjusted_basis="r1", verbose=False)
    harmony_integrate(adata, "batch", adjusted_basis="r2", verbose=False)
    np.testing.assert_allclose(adata.obsm["r1"], adata.obsm["r2"])


def test_list_key_equals_str_key(adata: AnnData):
    harmony_integrate(adata, "batch", adjusted_basis="s", verbose=False)
    harmony_integrate(adata, ["batch"], adjusted_basis="l", verbose=False)
    np.testing.assert_allclose(adata.obsm["s"], adata.obsm["l"])


def test_multiple_keys(adata: AnnData):
    adata.obs["donor"] = pd.Categorical(["u", "v"] * (N_OBS // 2))
    harmony_integrate(adata, ["batch", "donor"], verbose=False)
    assert adata.obsm["X_pca_harmony"].shape == (N_OBS, 5)


def test_reference_batch_cells_are_unchanged(adata: AnnData, batch_idx: np.ndarray):
    Z = adata.obsm["X_pca"]
    harmony_integrate(adata, "batch", reference_batch="A", verbose=False)

    E = adata.obsm["X_pca_harmony"]
    np.testing.assert_allclose(E[batch_idx == 0], Z[batch_idx == 0])
    assert np.abs(E[batch_idx == 1] - Z[batch_idx == 1]).max() > 0.1
    assert batch_gap(E, batch_idx) < batch_gap(Z, batch_idx)


def test_reference_batch_accepts_list(adata: AnnData, batch_idx: np.ndarray):
    harmony_integrate(
        adata, "batch", reference_batch="A", adjusted_basis="s", verbose=False
    )
    harmony_integrate(
        adata, "batch", reference_batch=["A"], adjusted_basis="l", verbose=False
    )
    np.testing.assert_allclose(adata.obsm["s"], adata.obsm["l"])


def test_all_batches_as_reference_freezes_everything(adata: AnnData):
    harmony_integrate(adata, "batch", reference_batch=["A", "B"], verbose=False)
    np.testing.assert_allclose(adata.obsm["X_pca_harmony"], adata.obsm["X_pca"])


def test_kwargs_are_forwarded_to_run_harmony(adata: AnnData):
    harmony_integrate(adata, "batch", adjusted_basis="a", verbose=False, nclust=3)
    harmony_integrate(
        adata, "batch", adjusted_basis="b", verbose=False, nclust=3, theta=0.0
    )
    # theta=0 disables the diversity penalty, so the result must differ
    assert not np.allclose(adata.obsm["a"], adata.obsm["b"])


def test_unknown_basis_raises(adata: AnnData):
    with pytest.raises(KeyError):
        harmony_integrate(adata, "batch", basis="nope")


def test_unknown_key_raises(adata: AnnData):
    with pytest.raises(KeyError):
        harmony_integrate(adata, "nope")


def test_list_key_with_reference_batch(adata: AnnData, batch_idx: np.ndarray):
    harmony_integrate(adata, ["batch"], reference_batch="A", verbose=False)

    E = adata.obsm["X_pca_harmony"]
    np.testing.assert_allclose(E[batch_idx == 0], adata.obsm["X_pca"][batch_idx == 0])


def test_multi_key_with_reference_batch_raises(adata: AnnData):
    adata.obs["batch2"] = adata.obs["batch"]
    with pytest.raises(ValueError, match="single batch key"):
        harmony_integrate(adata, ["batch", "batch2"], reference_batch="A")


@pytest.mark.xfail(
    strict=True,
    reason="BUG: run_harmony default nclust = round(N/30) is 1 for N < 45, and then "
    "sigma stays a float and Harmony indexes it",
)
def test_small_dataset_default_nclust():
    rng = np.random.default_rng(0)
    adata = AnnData(np.zeros((40, 2)))
    adata.obsm["X_pca"] = rng.normal(size=(40, 4))
    adata.obs["batch"] = pd.Categorical(["A", "B"] * 20)

    harmony_integrate(adata, "batch", verbose=False)

    assert adata.obsm["X_pca_harmony"].shape == (40, 4)


# ---------------------------------------------------------------------------
# run_harmony and helpers
# ---------------------------------------------------------------------------


def test_cosine_normalization_preserves_direction():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(100, 4))
    Z[:10] = -np.abs(Z[:10]) - 0.1  # all-negative cells
    meta = pd.DataFrame({"b": pd.Categorical(["A", "B"] * 50)})

    seen = {}

    def spy(data, K):
        seen["Z_cos"] = np.array(data)
        return data[:K].copy()

    run_harmony(Z, meta, "b", verbose=False, nclust=3, cluster_fn=spy)

    Z_cos = seen["Z_cos"]
    expected = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    np.testing.assert_allclose(Z_cos, expected, atol=1e-8)


def test_cosine_normalization_unchanged_when_every_max_is_positive():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(100, 4))
    Z[np.arange(100), Z.argmax(axis=1)] += 1.0  # every cell's max is positive
    assert (Z.max(axis=1) > 0).all()
    meta = pd.DataFrame({"b": pd.Categorical(["A", "B"] * 50)})

    seen = {}

    def spy(data, K):
        seen["Z_cos"] = np.array(data)
        return data[:K].copy()

    run_harmony(Z, meta, "b", verbose=False, nclust=3, cluster_fn=spy)

    old = Z / Z.max(axis=1, keepdims=True)  # previous formula
    old = old / np.linalg.norm(old, axis=1, keepdims=True)
    np.testing.assert_allclose(seen["Z_cos"], old, atol=1e-8)


@pytest.fixture
def small():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(100, 4))
    meta = pd.DataFrame({"b": pd.Categorical(["A", "B"] * 50)})
    return Z, meta


def test_run_harmony_output_is_dims_by_cells(small):
    Z, meta = small
    ho = run_harmony(Z, meta, "b", verbose=False)

    assert ho.Z_corr.shape == (4, 100)
    assert ho.Z_orig.shape == (4, 100)
    assert ho.K == 3  # round(100 / 30)
    assert ho.R.shape == (3, 100)
    np.testing.assert_allclose(ho.R.sum(axis=0), 1.0)
    assert len(ho.objective_harmony) >= 2


def test_run_harmony_accepts_transposed_input(small):
    Z, meta = small
    a = run_harmony(Z, meta, "b", verbose=False)
    b = run_harmony(Z.T, meta, "b", verbose=False)
    np.testing.assert_allclose(a.Z_corr, b.Z_corr)


def test_run_harmony_mismatched_cells_raises(small):
    Z, meta = small
    with pytest.raises(AssertionError, match="same number of cells"):
        run_harmony(Z[:50], meta, "b", verbose=False)


def test_run_harmony_nclust_and_theta_lamb_expansion(small):
    Z, meta = small
    ho = run_harmony(Z, meta, "b", verbose=False, nclust=2, theta=[2.0], lamb=[0.5])

    assert ho.K == 2
    np.testing.assert_allclose(ho.theta, [2.0, 2.0])
    # lamb is a diagonal matrix with a zero for the (unpenalized) intercept
    np.testing.assert_allclose(np.diag(ho.lamb), [0.0, 0.5, 0.5])


def test_run_harmony_tau_shrinks_theta(small):
    Z, meta = small
    ho = run_harmony(Z, meta, "b", verbose=False, nclust=2, tau=10)
    assert np.all(ho.theta < 1.0)
    assert np.all(ho.theta > 0.0)


def test_run_harmony_custom_cluster_fn(small):
    Z, meta = small

    def fixed_centroids(data, K):
        return data[:K].copy()

    ho = run_harmony(Z, meta, "b", verbose=False, nclust=3, cluster_fn=fixed_centroids)
    assert ho.Z_corr.shape == (4, 100)


def test_run_harmony_frozen_cells_are_not_corrected(small):
    Z, meta = small
    frozen = np.arange(100) < 50
    ho = run_harmony(Z, meta, "b", verbose=False, reference_values=frozen)
    np.testing.assert_allclose(ho.Z_corr[:, frozen], Z.T[:, frozen])


def test_safe_entropy_handles_zero():
    out = safe_entropy(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(out, [0.0, 0.5 * np.log(0.5), 0.0])


def test_moe_correct_ridge_removes_batch_mean_shift():
    # one cluster that takes all the mass; batch B is shifted by +3 in dim 0
    rng = np.random.default_rng(0)
    N = 100
    batch = np.array([0, 1] * (N // 2))
    Z = rng.normal(size=(2, N)) * 0.1
    Z[0, batch == 1] += 3.0

    phi = np.vstack([batch == 0, batch == 1]).astype(float)
    phi_moe = np.vstack([np.ones(N), phi])
    R = np.ones((1, N))
    lamb = np.diag([0.0, 1e-6, 1e-6])

    Z_cos, Z_corr, W, _ = moe_correct_ridge(
        Z,
        Z / np.linalg.norm(Z, axis=0),
        Z.copy(),
        R,
        np.zeros((3, 2)),
        1,
        np.zeros((3, N)),
        phi_moe,
        lamb,
    )

    gap_before = Z[0, batch == 1].mean() - Z[0, batch == 0].mean()
    gap_after = Z_corr[0, batch == 1].mean() - Z_corr[0, batch == 0].mean()
    assert gap_before == pytest.approx(3.0, abs=0.1)
    assert abs(gap_after) < 0.05
    np.testing.assert_allclose(np.linalg.norm(Z_cos, axis=0), 1.0)
    np.testing.assert_allclose(W[0], 0.0)  # intercept is never removed
