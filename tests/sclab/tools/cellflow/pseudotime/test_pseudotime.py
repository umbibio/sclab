import numpy as np
import pytest
from anndata import AnnData

import sclab.preprocess
from sclab.tools.cellflow import pseudotime
from sclab.tools.cellflow.pseudotime_tools import periodic_parameter

from ....test_utils import simple_loop_adata


@pytest.fixture
def adata():
    adata = simple_loop_adata()

    sclab.preprocess.preprocess(adata, compute_hvg=False)
    sclab.preprocess.pca(adata, n_comps=5)

    return adata


def test_default(adata: AnnData):
    data = adata.obsm["X_pca"][:, :2]
    adata.obs["phi"] = periodic_parameter(data) / 2 / np.pi

    pseudotime(adata, "X_pca", "phi", (0, 1), 3)

    assert "pseudotime" in adata.uns
    assert "pseudotime" in adata.obs
    assert "pseudotime_path" in adata.obsm
    assert adata.uns["pseudotime"]["t_range"] == [0, 1]
    assert adata.uns["pseudotime"]["periodic"] is False
    assert adata.uns["pseudotime"]["params"]["use_rep"] == "X_pca"
    assert adata.uns["pseudotime"]["params"]["t_key"] == "phi"
    assert adata.uns["pseudotime"]["params"]["t_range"] == [0, 1]
    assert adata.uns["pseudotime"]["params"]["periodic"] is False


def test_periodic(adata: AnnData):
    data = adata.obsm["X_pca"][:, :2]
    adata.obs["phi"] = periodic_parameter(data)
    phi_range = (0, 2 * np.pi)

    pseudotime(adata, "X_pca", "phi", phi_range, 3, periodic=True)

    assert "pseudotime" in adata.uns
    assert "pseudotime" in adata.obs
    assert "pseudotime_path" in adata.obsm
    assert adata.uns["pseudotime"]["t_range"] == [0, 1]
    assert adata.uns["pseudotime"]["periodic"] is True
    assert adata.uns["pseudotime"]["params"]["use_rep"] == "X_pca"
    assert adata.uns["pseudotime"]["params"]["t_key"] == "phi"
    assert adata.uns["pseudotime"]["params"]["t_range"] == list(phi_range)
    assert adata.uns["pseudotime"]["params"]["periodic"] is True
