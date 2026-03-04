import numpy as np
import pytest
from anndata import AnnData

import sclab.preprocess

from ..test_utils import simple_loop_adata


@pytest.fixture
def adata():
    return simple_loop_adata()


def test_default_effects(adata: AnnData):
    sclab.preprocess.qc(adata)

    assert "counts" in adata.layers
    assert "qc_tmp_current_X" not in adata.layers
    assert "barcode_rank" in adata.obs
    assert "total_counts" in adata.obs
    assert "n_genes" in adata.obs
    assert "n_genes_by_counts" in adata.obs


def test_min_counts_filters_cells(adata: AnnData):
    """Cells with total counts below min_counts should be removed."""
    n_obs_before = adata.n_obs

    rowsums = np.asarray(adata.X.sum(axis=1)).squeeze()
    min_counts = (rowsums.min() + rowsums.max()) // 2

    sclab.preprocess.qc(adata, min_counts=min_counts)

    assert adata.n_obs < n_obs_before
    assert (adata.obs["total_counts"] >= min_counts).all()


def test_max_rank_filters_cells(adata: AnnData):
    max_rank = 5
    sclab.preprocess.qc(adata, max_rank=max_rank)

    assert adata.n_obs <= max_rank
    assert (adata.obs["barcode_rank"] < max_rank).all()
