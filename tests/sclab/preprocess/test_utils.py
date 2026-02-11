import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from numpy import ndarray
from scipy.sparse import csr_matrix

from sclab.preprocess._utils import (
    _filter_knn_matrix,
    _symmetrize_sparse_matrix,
    get_neighbors_adjacency_matrix,
)


@pytest.fixture
def adata():
    X = np.array(
        [
            [0.36954737, 0.18118702],
            [0.20753973, 0.34154967],
            [0.00000000, 0.28895249],
            [0.51198892, 0.52502303],
            [0.31929784, 0.21606453],
        ]
    )
    adata = AnnData(X)
    sc.pp.neighbors(adata, method="gauss", n_neighbors=4)

    return adata


def assert_matrix_properties(
    matrix: ndarray | csr_matrix,
    expected_shape: tuple,
    sparse: bool = True,
    symmetric: bool = False,
):
    """Helper function to check common matrix properties."""
    # Check shape
    assert matrix.shape == expected_shape, "Matrix shape mismatch"

    # Check sparsity
    if sparse:
        assert isinstance(matrix, csr_matrix), "Matrix should be sparse CSR"

    # Check symmetry
    if symmetric:
        assert np.allclose(matrix.toarray(), matrix.toarray().T), (
            "Matrix should be symmetric"
        )


def assert_neighbor_count(
    matrix: ndarray | csr_matrix,
    n_neighbors: int,
    exact: bool = True,
):
    """Helper function to check neighbor counts."""
    n_neighbors_per_cell = np.sum(matrix > 0, axis=1)
    if exact:
        assert np.all(n_neighbors_per_cell == n_neighbors), (
            f"Each cell should have exactly {n_neighbors} neighbors"
        )
    else:
        assert np.all(n_neighbors_per_cell >= n_neighbors), (
            f"Each cell should have at least {n_neighbors} neighbors"
        )


def assert_matrices_equal(
    matrix1: ndarray | csr_matrix,
    matrix2: ndarray | csr_matrix,
    message="",
):
    """Helper function to compare matrices."""

    # Convert to dense if needed
    m1 = matrix1.toarray() if isinstance(matrix1, csr_matrix) else matrix1
    m2 = matrix2.toarray() if isinstance(matrix2, csr_matrix) else matrix2

    assert np.allclose(m1, m2), f"{message}: Matrix values should match"


def assert_filtered_matrix_properties(
    filtered: csr_matrix,
    original: csr_matrix,
    n_neighbors,
):
    """Helper function to check properties of filtered k-nn matrices."""
    # Check shape preservation
    assert_matrix_properties(filtered, original.shape)

    # Check number of neighbors
    n_neighbors_per_cell = np.sum(filtered > 0, axis=1)
    assert np.all(n_neighbors_per_cell <= n_neighbors - 1), (
        f"Each cell should have at most {n_neighbors - 1} neighbors after filtering"
    )

    # Check that non-zero values in filtered exist in original
    filtered_dense = filtered.toarray()
    original_dense = original.toarray()
    assert np.all((filtered_dense > 0) <= (original_dense > 0)), (
        "Filtered matrix should only contain edges present in original"
    )


def test_get_neighbors_adjacency_matrix_basic(adata):
    # Get adjacency matrix with default parameters (directed, unweighted)
    adj_matrix = get_neighbors_adjacency_matrix(adata)

    # Test matrix properties
    assert_matrix_properties(adj_matrix, (adata.n_obs, adata.n_obs))

    # Test unweighted property
    assert np.all(adj_matrix.data == 1), "Unweighted matrix should only contain ones"

    # Test neighbor count
    assert_neighbor_count(adj_matrix, 3)

    # Test directedness
    assert not np.allclose(adj_matrix.toarray(), adj_matrix.toarray().T), (
        "Default adjacency matrix should be directed (not symmetric)"
    )


def test_get_neighbors_adjacency_matrix_weighted(adata: AnnData):
    # Get weighted adjacency matrix
    adj_matrix = get_neighbors_adjacency_matrix(adata, weighted=True)
    adj_matrix_dense = adj_matrix.toarray()

    # Test matrix properties
    assert_matrix_properties(adj_matrix, (adata.n_obs, adata.n_obs))

    # Test weight properties
    unweighted = get_neighbors_adjacency_matrix(adata, weighted=False)
    unweighted_dense = unweighted.toarray()
    assert np.all((adj_matrix_dense > 0) == (unweighted_dense == 1)), (
        "Weighted matrix should have same sparsity pattern as unweighted"
    )
    assert np.all(adj_matrix.data > 0), "Weights should be positive"
    assert np.all(adj_matrix.data <= 1), "Weights should be normalized (<=1)"

    # Test neighbor count
    assert_neighbor_count(adj_matrix, 3)

    # Test directedness
    assert not np.allclose(adj_matrix_dense, adj_matrix_dense.T), (
        "Default adjacency matrix should be directed (not symmetric)"
    )


def test_get_neighbors_adjacency_matrix_fewer_neighbors(adata: AnnData):
    # Get adjacency matrix with fewer neighbors
    adj_matrix = get_neighbors_adjacency_matrix(adata, n_neighbors=3)

    # Test matrix properties
    assert_matrix_properties(adj_matrix, (adata.n_obs, adata.n_obs))

    # Test neighbor count
    assert_neighbor_count(adj_matrix, 2)


def test_get_neighbors_adjacency_matrix_undirected(adata: AnnData):
    # Get undirected adjacency matrix
    adj_matrix = get_neighbors_adjacency_matrix(adata, directed=False)
    adj_matrix_dense = adj_matrix.toarray()

    # Test matrix properties
    assert_matrix_properties(adj_matrix, (adata.n_obs, adata.n_obs), symmetric=True)

    # Test unweighted property
    assert np.all(adj_matrix.data == 1), "Unweighted matrix should only contain ones"

    # Compare with directed matrix
    directed = get_neighbors_adjacency_matrix(adata, directed=True)
    directed_dense = directed.toarray()

    # Check edge preservation
    assert np.all((directed_dense > 0) <= (adj_matrix_dense > 0)), (
        "Undirected matrix should preserve all outgoing edges from directed matrix"
    )
    assert np.all((directed_dense.T > 0) <= (adj_matrix_dense > 0)), (
        "Undirected matrix should preserve all incoming edges from directed matrix"
    )

    # Test neighbor count (allowing for more due to symmetrization)
    assert_neighbor_count(adj_matrix, 3, exact=False)


def test_symmetrize_sparse_matrix():
    N = 5
    # Create an upper triangular matrix for testing
    triangular_dense = np.triu(np.arange(N**2).reshape(N, N))
    triangular = csr_matrix(triangular_dense)

    # Apply symmetrization
    symmetrized = _symmetrize_sparse_matrix(triangular)

    # Test matrix properties
    assert_matrix_properties(symmetrized, (N, N), symmetric=True)

    # Check that upper triangular values are preserved
    u_idxs = np.triu_indices(N)
    assert np.allclose(triangular_dense[u_idxs], symmetrized.toarray()[u_idxs]), (
        "Upper triangular values should be preserved"
    )


def test_filter_knn_matrix_weights():
    # Create a test matrix
    connectivities = csr_matrix(
        [
            [0.0, 0.8, 0.2, 0.1],
            [0.7, 0.0, 0.3, 0.5],
            [0.2, 0.4, 0.0, 0.9],
            [0.1, 0.5, 0.8, 0.0],
        ]
    )

    # Expected result with n_neighbors=3 (keeping top 2 weights per row)
    expected_dense = np.array(
        [
            [0.0, 0.8, 0.2, 0.0],  # 0.8, 0.2 are top 2
            [0.7, 0.0, 0.0, 0.5],  # 0.7, 0.5 are top 2
            [0.0, 0.4, 0.0, 0.9],  # 0.9, 0.4 are top 2
            [0.0, 0.5, 0.8, 0.0],  # 0.8, 0.5 are top 2
        ]
    )
    expected = csr_matrix(expected_dense)

    # Apply filtering
    filtered = _filter_knn_matrix(connectivities, n_neighbors=3, mode="weights")

    # Check filtered matrix properties
    assert_filtered_matrix_properties(filtered, connectivities, n_neighbors=3)

    # Check exact values
    assert_matrices_equal(
        filtered, expected, message="Filtered weights matrix should match expected"
    )


def test_filter_knn_matrix_distances():
    # Create a test matrix with distances (smaller values = closer)
    distances = csr_matrix(
        [
            [0.0, 0.2, 0.8, 0.9],
            [0.3, 0.0, 0.7, 0.5],
            [0.8, 0.6, 0.0, 0.1],
            [0.9, 0.5, 0.2, 0.0],
        ]
    )

    # Expected result with n_neighbors=2 (keeping 2 smallest distances per row)
    expected_dense = np.array(
        [
            [0.0, 0.2, 0.8, 0.0],  # 0.2, 0.8 are smallest
            [0.3, 0.0, 0.0, 0.5],  # 0.3, 0.5 are smallest
            [0.0, 0.6, 0.0, 0.1],  # 0.1, 0.6 are smallest
            [0.0, 0.5, 0.2, 0.0],  # 0.2, 0.5 are smallest
        ]
    )
    expected = csr_matrix(expected_dense)

    # Apply filtering
    filtered = _filter_knn_matrix(distances, n_neighbors=3, mode="distances")

    # Check filtered matrix properties
    assert_filtered_matrix_properties(filtered, distances, n_neighbors=3)

    # Check exact values
    assert_matrices_equal(
        filtered, expected, message="Filtered distances matrix should match expected"
    )
