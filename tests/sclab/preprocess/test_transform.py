import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

from sclab.preprocess import pool_neighbors


@pytest.fixture
def simple_adata():
    # Create a simple AnnData object for testing
    X = csr_matrix(np.array([[1, 0, 3], [4, 5, 0], [7, 0, 9]]))
    adata = AnnData(X)  # type: ignore

    # Create a simple connectivity matrix
    connectivities = csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))

    # Create a simple distance matrix
    distances = csr_matrix(np.array([[0, 0.5, 0], [0.5, 0, 0.3], [0, 0.3, 0]]))

    adata.obsp["connectivities"] = connectivities
    adata.obsp["distances"] = distances

    return adata


@pytest.mark.skip
def test_pool_neighbors_basic(simple_adata: AnnData):
    """Test basic functionality of pool_neighbors with default parameters."""
    original_X: csr_matrix = simple_adata.X.copy()
    pool_neighbors(simple_adata)

    # Check that the output key was added
    assert "X_pooled" in simple_adata.layers
    transformed_X: csr_matrix = simple_adata.layers["X_pooled"]

    # Check that the output is different from input
    assert not np.array_equal(original_X.toarray(), transformed_X.toarray())

    # Check that the original X wasn't modified
    assert np.array_equal(original_X.toarray(), simple_adata.X.toarray())

    # Check that the output maintains sparsity
    assert issparse(transformed_X)


@pytest.mark.skip
def test_pool_neighbors_with_layer(adata):
    """Test pool_neighbors with a specific layer."""
    for layer in ["X", "counts"]:
        # Add a layer
        original_X = adata.layers[layer] = adata.X.copy()
        pool_neighbors(adata, layer=layer)

        # Check that the output key was added
        assert f"{layer}_pooled" in adata.layers
        transformed_X = adata.layers[f"{layer}_pooled"]

        # Check that the output is different from input
        assert not np.array_equal(original_X.toarray(), transformed_X.toarray())

        # Check that the original X wasn't modified
        assert np.array_equal(original_X.toarray(), adata.layers[layer].toarray())

        # Check that the output maintains sparsity
        assert issparse(transformed_X)


@pytest.mark.skip
def test_pool_neighbors_assertions(input_adata):
    """Test pool_neighbors with missing matrices."""
    adata = input_adata.copy()

    # Remove one of the matrices
    del adata.obsp["connectivities"]

    with pytest.raises(AssertionError):
        pool_neighbors(adata)


@pytest.mark.skip
def test_pool_neighbors_connectivities(simple_adata):
    """Test pool_neighbors using connectivities instead of distances."""
    original_X = simple_adata.X.copy()
    pool_neighbors(simple_adata, pooling_mode="connectivities")

    # Check that the output is different from input
    assert not np.array_equal(original_X.toarray(), simple_adata.X.toarray())


@pytest.mark.skip
def test_pool_neighbors_custom_key(simple_adata):
    """Test pool_neighbors with custom neighbors key and output key."""
    # Create custom connectivity matrices
    custom_connectivities = csr_matrix(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]))
    simple_adata.obsp["custom_connectivities"] = custom_connectivities

    pool_neighbors(
        simple_adata,
        neighbors_key="custom",
        key_added="pooled_custom",
        pooling_mode="connectivities",
    )

    # Check that the new key was added
    assert "pooled_custom" in simple_adata.layers


@pytest.mark.skip
def test_pool_neighbors_invalid_input():
    """Test pool_neighbors with invalid inputs."""
    # Create invalid AnnData without necessary matrices
    invalid_adata = AnnData(csr_matrix(np.array([[1, 2], [3, 4]])))

    with pytest.raises(ValueError):
        pool_neighbors(invalid_adata)


@pytest.mark.skip
def test_pool_neighbors_n_neighbors(simple_adata):
    """Test pool_neighbors with custom number of neighbors."""
    pool_neighbors(simple_adata, n_neighbors=1)
    pool_neighbors(simple_adata, n_neighbors=2)

    # Different n_neighbors should produce different results
    result1 = simple_adata.X.copy()
    pool_neighbors(simple_adata, n_neighbors=1)
    result2 = simple_adata.X.copy()

    assert not np.array_equal(result1.toarray(), result2.toarray())
