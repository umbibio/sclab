import pytest
from anndata import AnnData

import sclab.preprocess

from ..test_utils import simple_loop_adata


@pytest.fixture
def adata():
    return simple_loop_adata()


def test_default_effects(adata: AnnData):
    sclab.preprocess.preprocess(adata)

    assert "counts" in adata.layers
    assert "counts_log1p" in adata.layers
    assert "counts_normt_log1p_scale" in adata.layers

    assert "highly_variable" in adata.var
    assert "total_counts" in adata.obs
    assert "n_genes" in adata.obs
    assert "n_genes_by_counts" in adata.obs


def test_groupby(adata: AnnData):
    sclab.preprocess.preprocess(adata, group_by="quadrant")

    assert "highly_variable" in adata.var
    assert "highly_variable_I" in adata.var
    assert "highly_variable_II" in adata.var
    assert "highly_variable_III" in adata.var
    assert "highly_variable_IV" in adata.var


def test_mormw(adata: AnnData):
    sclab.preprocess.preprocess(adata, normalization_method="weighted")

    assert "counts_normw_log1p_scale" in adata.layers
