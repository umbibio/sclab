import pytest
from anndata import AnnData

import sclab.preprocess

from ..test_utils import simple_loop_adata


@pytest.fixture
def adata():
    return simple_loop_adata()


def test_sparse(adata: AnnData):
    sclab.preprocess.normalize_weighted(adata)


def test_dense(adata: AnnData):
    adata.X = adata.X.toarray()
    sclab.preprocess.normalize_weighted(adata)


def test_batch(adata: AnnData):
    sclab.preprocess.normalize_weighted(adata, batch_key="quadrant")
