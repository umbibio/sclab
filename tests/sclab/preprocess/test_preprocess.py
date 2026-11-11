import anndata
import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import issparse

import sclab.preprocess
from sclab.utils import layer_keys, x_name

from ..test_utils import simple_loop_adata


@pytest.fixture
def adata():
    return simple_loop_adata()


def test_default_effects(adata: AnnData):
    sclab.preprocess.preprocess(adata)

    # no copy of X nor log1p(counts): only the counts and the data before scaling
    assert layer_keys(adata) == ["counts", "counts_normt_log1p"]
    assert x_name(adata) == "counts_normt_log1p_scale"

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

    assert layer_keys(adata) == ["counts", "counts_normw_log1p"]
    assert x_name(adata) == "counts_normw_log1p_scale"


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

    assert x_name(adata) == "counts_scale" and layer_keys(adata) == ["counts"]
    scaled = adata.X
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


def _dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)


@pytest.mark.parametrize(
    "kwargs, unscaled",
    [
        ({}, "counts_normt_log1p"),
        ({"group_by": "quadrant"}, "counts_normt_log1p"),
        ({"regress_total_counts": True}, "counts_normt_log1p_regr"),
        ({"normalization_method": "weighted"}, "counts_normw_log1p"),
    ],
)
def test_unscaled_layer_only_with_scale(adata: AnnData, kwargs, unscaled):
    unscaled_run = adata.copy()
    sclab.preprocess.preprocess(unscaled_run, scale=False, **kwargs)
    assert layer_keys(unscaled_run) == ["counts"]  # X is not duplicated
    assert x_name(unscaled_run) == unscaled

    sclab.preprocess.preprocess(adata, scale=True, **kwargs)
    assert layer_keys(adata) == ["counts", unscaled]
    assert x_name(adata) == f"{unscaled}_scale"
    layer = adata.layers[unscaled]
    assert layer.dtype == np.float32
    assert issparse(layer) == issparse(unscaled_run.X)  # sparse unless regressed
    assert issparse(layer) or "regress_total_counts" in kwargs
    # the data before scaling: X of a run without scaling
    np.testing.assert_array_equal(_dense(layer), _dense(unscaled_run.X))


@pytest.mark.parametrize("group_by", [None, "quadrant"])
def test_old_style_layers_kept_and_results_unchanged(adata: AnnData, group_by):
    """A dataset preprocessed by older sclab has ``counts_log1p`` and a copy of
    X; preprocessing it again keeps them and gives the same results."""
    new = adata.copy()
    sclab.preprocess.preprocess(new, group_by=group_by)

    old = adata.copy()
    old.layers["counts_log1p"] = np.log1p(old.layers.get("counts", old.X))
    old.layers["counts_normt_log1p_scale"] = old.X.copy()
    sclab.preprocess.preprocess(old, group_by=group_by)

    assert layer_keys(old) == [
        "counts_log1p",
        "counts_normt_log1p_scale",
        "counts",
        "counts_normt_log1p",
    ]
    np.testing.assert_array_equal(_dense(old.X), _dense(new.X))
    hv = [c for c in new.var if c.startswith("highly_variable")]
    assert len(hv) > 1
    for column in hv:  # HVGs on the old layer and on log1p computed on the fly
        np.testing.assert_array_equal(old.var[column], new.var[column])


def test_x_name_survives_h5ad_round_trip(adata: AnnData, tmp_path):
    sclab.preprocess.preprocess(adata)
    path = tmp_path / "a.h5ad"
    adata.write_h5ad(path)
    read = anndata.read_h5ad(path)
    assert x_name(read) == "counts_normt_log1p_scale"
    assert list(read.uns["sclab"]["citations"]) == list(adata.uns["sclab"]["citations"])
