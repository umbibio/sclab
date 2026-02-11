import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sclab.preprocess import subset_obs, subset_var


@pytest.fixture
def adata():
    # Create a simple AnnData object for testing
    obs_names = ["a", "b", "c", "d", "e"]
    obs_types = ["type1", "type1", "type2", "type2", "type2"]

    var_names = ["gene1", "gene2", "gene3", "gene4", "gene5"]
    var_classes = ["class1", "class1", "class2", "class2", "class2"]

    # Create random data matrix
    X = np.random.rand(len(obs_names), len(var_names))

    # Create obs and var dataframes
    obs = pd.DataFrame(index=obs_names, data={"cell_type": obs_types})
    var = pd.DataFrame(index=var_names, data={"gene_classes": var_classes})

    # Create AnnData object with all components
    adata_ = AnnData(X=X, obs=obs, var=var)
    return adata_


# Tests for subset_obs
def test_subset_obs_with_pd_index(adata: AnnData):
    """Test subsetting observations using a pandas Index."""
    subset = pd.Index(["b", "c", "d"])
    subset_obs(adata, subset)
    assert adata.obs_names.tolist() == ["b", "c", "d"]


def test_subset_obs_with_names(adata: AnnData):
    """Test subsetting observations using a list of names."""
    subset = ["a", "b"]
    subset_obs(adata, subset)
    assert adata.obs_names.tolist() == ["a", "b"]


def test_subset_obs_with_indices(adata: AnnData):
    """Test subsetting observations using integer indices."""
    subset = [0, 1]
    subset_obs(adata, subset)
    assert adata.obs_names.tolist() == ["a", "b"]


def test_subset_obs_with_boolean_mask(adata: AnnData):
    """Test subsetting observations using a boolean mask."""
    subset = [True, False, True, False, False]
    subset_obs(adata, subset)
    assert adata.obs_names.tolist() == ["a", "c"]


def test_subset_obs_with_query_string(adata: AnnData):
    """Test subsetting observations using a query string."""
    subset = "cell_type == 'type1'"
    subset_obs(adata, subset)
    assert adata.obs_names.tolist() == ["a", "b"]


def test_subset_obs_invalid_input(adata: AnnData):
    """Test subsetting observations with invalid input."""
    # Test invalid names
    with pytest.raises(KeyError):
        subset_obs(adata, ["a", "b", "f"])

    # Test invalid indices
    with pytest.raises(IndexError, match="Integer indices must be between"):
        subset_obs(adata, [-1, 0])

    with pytest.raises(IndexError, match="Integer indices must be between"):
        subset_obs(adata, [0, 5])

    # Test invalid boolean mask length
    with pytest.raises(IndexError, match="Boolean mask length"):
        subset_obs(adata, [True, False, True])


# Tests for subset_var
def test_subset_var_with_pd_index(adata: AnnData):
    """Test subsetting variables using a pandas Index."""
    subset = pd.Index(["gene2", "gene3", "gene4"])
    subset_var(adata, subset)
    assert adata.var_names.tolist() == ["gene2", "gene3", "gene4"]


def test_subset_var_with_names(adata: AnnData):
    """Test subsetting variables using a list of names."""
    subset = ["gene1", "gene2"]
    subset_var(adata, subset)
    assert adata.var_names.tolist() == ["gene1", "gene2"]


def test_subset_var_with_indices(adata: AnnData):
    """Test subsetting variables using integer indices."""
    subset = [0, 1]
    subset_var(adata, subset)
    assert adata.var_names.tolist() == ["gene1", "gene2"]


def test_subset_var_with_boolean_mask(adata: AnnData):
    """Test subsetting variables using a boolean mask."""
    subset = [True, False, True, False, False]
    subset_var(adata, subset)
    assert adata.var_names.tolist() == ["gene1", "gene3"]


def test_subset_var_with_query_string(adata: AnnData):
    """Test subsetting variables using a query string."""
    subset = "gene_classes == 'class1'"
    subset_var(adata, subset)
    assert adata.var_names.tolist() == ["gene1", "gene2"]


def test_subset_var_invalid_input(adata: AnnData):
    """Test subsetting variables with invalid input."""
    # Test invalid names
    with pytest.raises(KeyError):
        subset_var(adata, ["gene1", "gene6"])

    # Test invalid indices
    with pytest.raises(IndexError, match="Integer indices must be between"):
        subset_var(adata, [-1, 0])

    with pytest.raises(IndexError, match="Integer indices must be between"):
        subset_var(adata, [0, 5])

    # Test invalid boolean mask length
    with pytest.raises(IndexError, match="Boolean mask length"):
        subset_var(adata, [True, False, True])
