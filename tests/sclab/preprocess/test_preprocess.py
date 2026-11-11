import numpy as np
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


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_scale_per_group_integer_counts(adata: AnnData, dtype):
    adata.layers["counts"] = adata.X.astype(dtype)  # integer counts, no other step
    sclab.preprocess.preprocess(
        adata,
        group_by="quadrant",
        compute_hvg=False,
        normalization_method="none",
        log1p=False,
        scale=True,
    )

    scaled = adata.layers["counts_scale"]
    scaled = scaled.toarray() if hasattr(scaled, "toarray") else scaled
    assert scaled.dtype.kind == "f"
    counts = adata.layers["counts"]
    counts = counts.toarray() if hasattr(counts, "toarray") else counts
    for idx in adata.obs.groupby("quadrant", observed=True).indices.values():
        block, raw = scaled[idx], counts[idx].astype(float)
        sd = raw.std(axis=0, ddof=1)
        keep = sd > 0
        # scale(zero_center=False) divides by the per-group std: not truncated
        np.testing.assert_allclose(block[:, keep], raw[:, keep] / sd[keep], rtol=1e-4)
