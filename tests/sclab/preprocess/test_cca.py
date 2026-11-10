import numpy as np
import pytest
import scipy.sparse as sp

from sclab.preprocess._cca import (
    _cross_covariance_dense,
    _cross_covariance_sparse,
    _dense_scale,
    cca,
)


@pytest.fixture
def data():
    """Two batches sharing a 3-dimensional latent structure (n1=30, n2=25, p=20)."""
    rng = np.random.default_rng(0)
    W = rng.normal(size=(3, 20))
    X = rng.normal(size=(30, 3)) @ W + 0.1 * rng.normal(size=(30, 20))
    Y = rng.normal(size=(25, 3)) @ W + 0.1 * rng.normal(size=(25, 20))
    return X, Y


# ---------------------------------------------------------------------------
# cross-covariance
# ---------------------------------------------------------------------------


def test_dense_scale_gives_unit_std():
    A = np.random.default_rng(1).normal(3, 5, size=(50, 4))
    scaled = _dense_scale(A)
    np.testing.assert_allclose(scaled.std(axis=0, ddof=1), 1.0, atol=1e-6)


def test_cross_covariance_dense_matches_manual_formula(data):
    X, Y = data
    p = X.shape[1]

    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    Ys = (Y - Y.mean(0)) / Y.std(0, ddof=1)
    expected = Xs @ Ys.T / (p - 1)

    C = _cross_covariance_dense(X, Y)
    assert C.shape == (30, 25)
    np.testing.assert_allclose(C, expected, atol=1e-6)


def test_cross_covariance_sparse_matches_centered_product(data):
    X, Y = data
    p = X.shape[1]

    # the sparse path centers but does not scale
    expected = (X - X.mean(0)) @ (Y - Y.mean(0)).T / (p - 1)

    C = _cross_covariance_sparse(sp.csr_matrix(X), sp.csr_matrix(Y), n_jobs=1)
    assert C.shape == (30, 25)
    np.testing.assert_allclose(C, expected, atol=1e-10)


def test_cross_covariance_feature_mismatch_raises():
    with pytest.raises(ValueError, match="same number of features"):
        _cross_covariance_dense(np.ones((4, 3)), np.ones((4, 5)))
    with pytest.raises(ValueError, match="same number of features"):
        _cross_covariance_sparse(
            sp.csr_matrix(np.ones((4, 3))), sp.csr_matrix(np.ones((4, 5))), n_jobs=1
        )


# ---------------------------------------------------------------------------
# cca
# ---------------------------------------------------------------------------


def test_cca_shapes_and_sorted_singular_values(data):
    X, Y = data
    U, s, V = cca(X, Y, n_components=4, svd_solver="full")

    assert U.shape == (30, 4)
    assert V.shape == (25, 4)
    assert s.shape == (4,)
    assert np.all(np.diff(s) <= 0)
    assert np.all(s >= 0)


def test_cca_full_without_normalize_gives_orthonormal_columns(data):
    X, Y = data
    U, _, V = cca(X, Y, n_components=4, svd_solver="full", normalize=False)

    np.testing.assert_allclose(U.T @ U, np.eye(4), atol=1e-8)
    np.testing.assert_allclose(V.T @ V, np.eye(4), atol=1e-8)


def test_cca_normalize_gives_unit_norm_rows(data):
    X, Y = data
    U, _, V = cca(X, Y, n_components=4, svd_solver="full", normalize=True)

    np.testing.assert_allclose(np.linalg.norm(U, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(V, axis=1), 1.0)


def test_cca_reconstructs_cross_covariance(data):
    X, Y = data
    C = _cross_covariance_dense(X, Y)
    k = min(C.shape)

    U, s, V = cca(X, Y, n_components=k, svd_solver="full")

    np.testing.assert_allclose(U @ np.diag(s) @ V.T, C, atol=1e-8)


def test_cca_default_n_components_is_min_cells(data):
    X, Y = data
    U, s, V = cca(X, Y, svd_solver="full")
    assert U.shape[1] == V.shape[1] == s.size == 25


@pytest.mark.parametrize("solver", ["partial", "randomized"])
def test_cca_solvers_agree_with_full_on_leading_components(data, solver):
    X, Y = data
    _, s_full, _ = cca(X, Y, n_components=3, svd_solver="full")
    _, s, _ = cca(X, Y, n_components=3, svd_solver=solver, random_state=0)

    np.testing.assert_allclose(s, s_full, rtol=1e-3)


def test_cca_randomized_is_reproducible_with_seed(data):
    X, Y = data
    a = cca(X, Y, n_components=3, svd_solver="randomized", random_state=7)
    b = cca(X, Y, n_components=3, svd_solver="randomized", random_state=7)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_cca_partial_is_reproducible_with_seed(data):
    X, Y = data
    a = cca(X, Y, n_components=3, svd_solver="partial", random_state=7)
    b = cca(X, Y, n_components=3, svd_solver="partial", random_state=7)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_cca_matched_cells_get_similar_embeddings():
    # batch 2 is a (noisy) permutation of batch 1: matched cells must land close
    rng = np.random.default_rng(3)
    W = rng.normal(size=(3, 20))
    X = rng.normal(size=(40, 3)) @ W + 0.01 * rng.normal(size=(40, 20))
    perm = rng.permutation(40)
    Y = X[perm] + 0.001 * rng.normal(size=(40, 20))

    U, _, V = cca(X, Y, n_components=3, svd_solver="full", normalize=True)

    # Y[j] is X[perm[j]], so V[j] should be close to U[perm[j]]
    cos = (V * U[perm]).sum(axis=1)
    assert np.all(np.abs(cos) > 0.9)
    assert np.median(np.abs(cos)) > 0.99


def test_cca_sparse_input_runs_and_matches_centered_svd(data):
    X, Y = data
    U, s, V = cca(
        sp.csr_matrix(X), sp.csr_matrix(Y), n_components=3, svd_solver="full", n_jobs=1
    )

    C = (X - X.mean(0)) @ (Y - Y.mean(0)).T / (X.shape[1] - 1)
    np.testing.assert_allclose(s, np.linalg.svd(C, compute_uv=False)[:3], atol=1e-8)
    assert U.shape == (30, 3) and V.shape == (25, 3)


def test_cca_invalid_solver_raises(data):
    X, Y = data
    with pytest.raises(ValueError, match="svd_solver"):
        cca(X, Y, n_components=3, svd_solver="nope")  # type: ignore[arg-type]


def test_cca_feature_mismatch_raises():
    with pytest.raises(ValueError, match="same number of features"):
        cca(np.ones((5, 3)), np.ones((5, 4)), n_components=2)
