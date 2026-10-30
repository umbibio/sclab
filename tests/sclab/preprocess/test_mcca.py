import numpy as np
import pytest
import scipy.sparse as sp

from sclab.preprocess._cca import _cross_covariance_dense
from sclab.preprocess._mcca import mcca_sample_space


@pytest.fixture
def views():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(3, 15))
    sizes = [20, 25, 18]
    return [rng.normal(size=(n, 3)) @ W + 0.1 * rng.normal(size=(n, 15)) for n in sizes]


def test_shapes_and_sorted_eigenvalues(views):
    Z, eigvals = mcca_sample_space(views, n_components=4, n_jobs=1)

    assert Z.shape == (sum(v.shape[0] for v in views), 4)
    assert eigvals.shape == (4,)
    assert np.all(np.diff(eigvals) <= 0)


def test_eigenvectors_are_orthonormal(views):
    Z, _ = mcca_sample_space(views, n_components=4, n_jobs=1)
    np.testing.assert_allclose(Z.T @ Z, np.eye(4), atol=1e-8)


def test_normalize_gives_unit_rows(views):
    Z, _ = mcca_sample_space(views, n_components=4, normalize=True, n_jobs=1)
    np.testing.assert_allclose(np.linalg.norm(Z, axis=1), 1.0)


def test_two_views_eigenvalues_equal_cross_covariance_singular_values(views):
    X, Y = views[:2]
    C = _cross_covariance_dense(X, Y)
    s = np.linalg.svd(C, compute_uv=False)

    _, eigvals = mcca_sample_space([X, Y], n_components=3, n_jobs=1)

    # block matrix [[0, C], [C^T, 0]] has eigenvalues +/- singular values of C
    np.testing.assert_allclose(eigvals, s[:3], rtol=1e-6)


def test_two_views_embedding_blocks_are_singular_vectors_scaled(views):
    X, Y = views[:2]
    Z, _ = mcca_sample_space([X, Y], n_components=2, n_jobs=1)
    U, _, Vt = np.linalg.svd(_cross_covariance_dense(X, Y), full_matrices=False)

    n1 = X.shape[0]
    # eigenvectors are [u; v] / sqrt(2) up to a per-column sign
    np.testing.assert_allclose(np.abs(Z[:n1]), np.abs(U[:, :2]) / np.sqrt(2), atol=1e-6)
    np.testing.assert_allclose(
        np.abs(Z[n1:]), np.abs(Vt.T[:, :2]) / np.sqrt(2), atol=1e-6
    )


def test_sparse_and_mixed_inputs(views):
    sparse_views = [sp.csr_matrix(v) for v in views]
    Z, eigvals = mcca_sample_space(sparse_views, n_components=3, n_jobs=1)
    assert Z.shape == (63, 3)
    assert np.isfinite(eigvals).all()

    mixed = [sparse_views[0], views[1], sparse_views[2]]
    Z, _ = mcca_sample_space(mixed, n_components=3, n_jobs=1)
    assert Z.shape == (63, 3)


def test_n_components_is_capped_at_total_cells_minus_one():
    rng = np.random.default_rng(1)
    Xs = [rng.normal(size=(2, 5)), rng.normal(size=(2, 5))]
    Z, eigvals = mcca_sample_space(Xs, n_components=30, n_jobs=1)
    assert Z.shape == (4, 3)
    assert eigvals.shape == (3,)


def test_single_view_raises(views):
    with pytest.raises(ValueError, match="at least two views"):
        mcca_sample_space(views[:1], n_jobs=1)


def test_feature_mismatch_raises():
    with pytest.raises(ValueError, match="same number of features"):
        mcca_sample_space([np.ones((4, 3)), np.ones((4, 5))], n_jobs=1)
