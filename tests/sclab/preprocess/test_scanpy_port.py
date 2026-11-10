"""sclab's native port of the scanpy preprocessing functions vs scanpy.

Realistic synthetic counts (negative binomial, cell types, three batches,
unexpressed genes), float32, sparse CSR and dense. The tolerances are those
reached on scanpy 1.12.x; most results are bit-identical.
"""

import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from sclab.preprocess import _scanpy_port as native
from sclab.preprocess import highly_variable_genes, pca, preprocess, qc

sc = pytest.importorskip("scanpy")
pytest.importorskip("skmisc")

RTOL = 1e-5  # float32 rounding: observed differences are 0 or ~1e-7
ATOL = 1e-6


def make_counts(n_obs=900, n_vars=700, sparse=True, seed=0) -> AnnData:
    rng = np.random.default_rng(seed)
    mu = rng.gamma(0.6, 2.0, n_vars)
    lib = rng.lognormal(0, 0.5, n_obs)
    types = rng.integers(0, 4, n_obs)
    effect = rng.lognormal(0, 1.0, (4, n_vars)) * (rng.random((4, n_vars)) < 0.1) + 1
    lam = lib[:, None] * mu[None, :] * effect[types]
    X = rng.negative_binomial(2, 2 / (2 + lam)).astype(np.float32)
    X[:, :5] = 0  # unexpressed genes
    obs = pd.DataFrame(
        {"batch": pd.Categorical(rng.choice(["b1", "b2", "b3"], n_obs))},
        index=[f"c{i}" for i in range(n_obs)],
    )
    adata = AnnData(
        sp.csr_matrix(X) if sparse else X,
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    adata.layers["counts"] = adata.X.copy()
    return adata


def dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def assert_frames_close(a: pd.DataFrame, b: pd.DataFrame, rtol=RTOL, atol=ATOL):
    assert list(a.columns) == list(b.columns)
    assert a.index.equals(b.index)
    for c in a.columns:
        assert a[c].dtype == b[c].dtype, c
        if a[c].dtype == bool or not pd.api.types.is_numeric_dtype(a[c]):
            np.testing.assert_array_equal(a[c], b[c], err_msg=c)
        else:
            np.testing.assert_allclose(a[c], b[c], rtol=rtol, atol=atol, err_msg=c)


@pytest.fixture(params=["csr", "dense"])
def lognorm(request):
    """Counts after QC filters, normalize_total and log1p, by scanpy."""
    adata = make_counts(sparse=request.param == "csr")
    sc.pp.filter_cells(adata, min_genes=50)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


@pytest.mark.parametrize("sparse", [True, False])
def test_qc_metrics_and_filters(sparse):
    a = make_counts(sparse=sparse)
    b = a.copy()
    sc.pp.calculate_qc_metrics(a, percent_top=None, log1p=False, inplace=True)
    native.calculate_qc_metrics(b, percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_cells(a, min_genes=50)
    native.filter_cells(b, min_genes=50)
    sc.pp.filter_genes(a, min_cells=3)
    native.filter_genes(b, min_cells=3)
    assert_frames_close(a.obs, b.obs, rtol=0, atol=0)
    assert_frames_close(a.var, b.var, rtol=0, atol=0)


@pytest.mark.parametrize("sparse", [True, False])
def test_normalize_total_and_log1p(sparse):
    a = make_counts(sparse=sparse)
    b = a.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    native.normalize_total(b, target_sum=1e4)
    assert type(a.X) is type(b.X) and a.X.dtype == b.X.dtype == np.float32
    np.testing.assert_allclose(dense(a.X), dense(b.X), rtol=RTOL)
    sc.pp.log1p(a)
    native.log1p(b)
    np.testing.assert_allclose(dense(a.X), dense(b.X), rtol=RTOL)
    assert a.uns["log1p"] == b.uns["log1p"]
    # arrays, integer input
    counts = dense(make_counts().X).astype(int)
    np.testing.assert_allclose(sc.pp.log1p(counts.copy()), native.log1p(counts.copy()))


@pytest.mark.parametrize("flavor", ["seurat", "cell_ranger"])
@pytest.mark.parametrize("batch_key", [None, "batch"])
@pytest.mark.parametrize("n_top_genes", [None, 200])
def test_hvg_dispersion_flavors(lognorm, flavor, batch_key, n_top_genes):
    kwargs = dict(flavor=flavor, batch_key=batch_key, n_top_genes=n_top_genes)
    a = sc.pp.highly_variable_genes(lognorm, inplace=False, **kwargs)
    b = native.highly_variable_genes(lognorm, inplace=False, **kwargs)
    drop = dict(columns="mean_bin", errors="ignore")  # labels of the mean bins
    assert_frames_close(a.drop(**drop), b.drop(**drop))
    assert b["highly_variable"].sum() > 0


@pytest.mark.parametrize("flavor", ["seurat_v3", "seurat_v3_paper"])
@pytest.mark.parametrize("batch_key", [None, "batch"])
def test_hvg_seurat_v3_flavors(lognorm, flavor, batch_key):
    kwargs = dict(layer="counts", flavor=flavor, batch_key=batch_key, n_top_genes=200)
    a = sc.pp.highly_variable_genes(lognorm, inplace=False, **kwargs)
    b = native.highly_variable_genes(lognorm, inplace=False, **kwargs)
    np.testing.assert_array_equal(a["highly_variable"], b["highly_variable"])
    if batch_key is None:
        assert_frames_close(a, b)
    else:
        # scanpy 1.12.5 computes the per-batch variances with Welford's
        # algorithm (sclab as scanpy 1.12.0, which it matches exactly); the
        # loess fit amplifies those 1e-15 differences (observed: <= 2e-3)
        np.testing.assert_allclose(a["variances_norm"], b["variances_norm"], rtol=1e-2)


def test_hvg_inplace_writes_what_scanpy_writes(lognorm):
    a, b = lognorm.copy(), lognorm.copy()
    for pp, adata in [(sc.pp, a), (native, b)]:
        pp.highly_variable_genes(adata, flavor="seurat", batch_key="batch")
        pp.highly_variable_genes(
            adata, layer="counts", flavor="seurat_v3_paper", n_top_genes=300
        )
    assert_frames_close(a.var, b.var)
    assert a.uns["hvg"] == b.uns["hvg"]


def test_regress_out_and_scale(lognorm):
    a, b = lognorm.copy(), lognorm.copy()
    sc.pp.regress_out(a, keys=["n_genes"], n_jobs=1)
    native.regress_out(b, keys=["n_genes"])
    assert a.X.dtype == b.X.dtype
    np.testing.assert_allclose(a.X, b.X, rtol=RTOL, atol=ATOL)
    sc.pp.scale(a, zero_center=False)
    native.scale(b, zero_center=False)
    np.testing.assert_allclose(a.X, b.X, rtol=RTOL, atol=ATOL)

    a, b = lognorm.copy(), lognorm.copy()
    sc.pp.scale(a, zero_center=False)
    native.scale(b, zero_center=False)
    assert type(a.X) is type(b.X) and a.X.dtype == b.X.dtype
    np.testing.assert_allclose(dense(a.X), dense(b.X), rtol=RTOL, atol=ATOL)
    assert_frames_close(a.var, b.var)


def test_regress_out_singular_design(lognorm):
    """A collinear key: scanpy fits a GLM per gene, the port least squares."""
    a, b = lognorm.copy(), lognorm.copy()
    for adata in (a, b):
        adata.X = dense(adata.X)
        adata.obs["n2"] = adata.obs["n_genes"] * 2
    sc.pp.regress_out(a, keys=["n_genes", "n2"], n_jobs=1)
    native.regress_out(b, keys=["n_genes", "n2"])
    assert a.X.dtype == b.X.dtype
    np.testing.assert_allclose(a.X, b.X, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("mask_var", [None, "hvg"])
@pytest.mark.parametrize("zero_center", [True, False])
def test_pca(lognorm, mask_var, zero_center):
    lognorm.var["hvg"] = np.arange(lognorm.n_vars) % 3 == 0
    a, b = lognorm.copy(), lognorm.copy()
    kwargs = dict(
        n_comps=20,
        mask_var=mask_var,
        zero_center=zero_center,
        svd_solver="arpack",
        random_state=0,
    )
    sc.pp.pca(a, **kwargs)
    native.pca(b, **kwargs)
    assert a.obsm["X_pca"].dtype == b.obsm["X_pca"].dtype == np.float32
    assert a.varm["PCs"].dtype == b.varm["PCs"].dtype
    np.testing.assert_allclose(a.obsm["X_pca"], b.obsm["X_pca"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(a.varm["PCs"], b.varm["PCs"], rtol=RTOL, atol=ATOL)
    for key in ("variance", "variance_ratio"):
        np.testing.assert_allclose(a.uns["pca"][key], b.uns["pca"][key], rtol=RTOL)
    assert a.uns["pca"]["params"] == b.uns["pca"]["params"]


def test_pca_default_mask_is_highly_variable(lognorm):
    lognorm.var["highly_variable"] = np.arange(lognorm.n_vars) % 2 == 0
    a, b = lognorm.copy(), lognorm.copy()
    sc.pp.pca(a, n_comps=10)
    native.pca(b, n_comps=10)
    assert b.uns["pca"]["params"]["mask_var"] == "highly_variable"
    assert a.uns["pca"]["params"] == b.uns["pca"]["params"]
    np.testing.assert_allclose(a.varm["PCs"], b.varm["PCs"], rtol=RTOL, atol=ATOL)


def test_pca_sparse_is_not_densified(monkeypatch):
    adata = make_counts()
    sc.pp.log1p(adata)
    monkeypatch.setattr(sp.csr_matrix, "toarray", None)  # would raise if called
    native.pca(adata, n_comps=5)
    assert adata.obsm["X_pca"].shape == (adata.n_obs, 5)


# --- the library functions: backend="native" vs backend="scanpy" ------------

PIPELINES = {
    "default": (dict(), dict(), dict(n_comps=20)),
    "grouped": (
        dict(min_cells=3),
        dict(group_by="batch", normalization_method="weighted", regress_n_genes=True),
        dict(
            n_comps=10,
            mask_var="highly_variable",
            batch_key="batch",
            reference_batch="b1",
            zero_center=True,
        ),
    ),
    "raw-cell-ranger": (
        dict(max_rank=700),
        dict(
            normalization_method="none",
            log1p=False,
            scale=False,
            regress_total_counts=True,
            hvg_methods=["cell_ranger", "seurat_v3"],
            hvg_combine="intersection",
        ),
        dict(n_comps=10, mask_var="highly_variable"),
    ),
}


@pytest.mark.parametrize("sparse", [True, False], ids=["csr", "dense"])
@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_library_backends_agree(sparse, pipeline):
    qc_kwargs, pp_kwargs, pca_kwargs = PIPELINES[pipeline]
    results = []
    for backend in ("scanpy", "native"):
        adata = make_counts(sparse=sparse, seed=1)
        qc(adata, backend=backend, **qc_kwargs)
        preprocess(adata, backend=backend, **pp_kwargs)
        pca(adata, backend=backend, **pca_kwargs)
        results.append(adata)
    a, b = results
    assert_frames_close(a.obs, b.obs)
    assert_frames_close(a.var, b.var)
    assert list(a.layers) == list(b.layers)
    for key in a.layers:
        assert type(a.layers[key]) is type(b.layers[key])
        assert a.layers[key].dtype == b.layers[key].dtype
        np.testing.assert_allclose(
            dense(a.layers[key]), dense(b.layers[key]), rtol=RTOL, atol=ATOL
        )
    np.testing.assert_allclose(dense(a.X), dense(b.X), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(a.obsm["X_pca"], b.obsm["X_pca"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(a.varm["PCs"], b.varm["PCs"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        a.uns["pca"]["variance"], b.uns["pca"]["variance"], rtol=RTOL
    )
    assert list(a.uns["sclab"]["citations"]) == list(b.uns["sclab"]["citations"])


# --- combining HVG methods -----------------------------------------------------


@pytest.mark.parametrize("combine", ["union", "intersection"])
def test_hvg_combine(combine):
    adata = make_counts()
    methods = ["seurat", "cell_ranger", "seurat_v3"]
    highly_variable_genes(adata, methods, combine, n_top_genes=150, backend="native")
    flags = adata.var[[f"highly_variable_{m}" for m in methods]].to_numpy()
    assert (flags.sum(axis=0) == 150).all()
    expected = flags.any(axis=1) if combine == "union" else flags.all(axis=1)
    np.testing.assert_array_equal(adata.var["highly_variable"], expected)
    assert 0 < expected.sum() < adata.n_vars
    assert "counts_log1p" not in adata.layers  # computed on the fly, not kept
    citations = list(adata.uns["sclab"]["citations"])
    assert {"satija2015", "zheng2017", "stuart2019", "scanpy"} <= set(citations)

    df = highly_variable_genes(
        adata, methods, combine, n_top_genes=150, inplace=False, backend="native"
    )
    pd.testing.assert_frame_equal(df, adata.var[df.columns], check_names=False)


def test_hvg_v3_takes_the_size_of_the_first_selection():
    adata = make_counts()
    df = highly_variable_genes(adata, inplace=False, backend="native")
    n = df["highly_variable_seurat"].sum()
    assert df["highly_variable_seurat_v3_paper"].sum() == n
    alone = highly_variable_genes(adata, "seurat_v3", inplace=False, backend="native")
    assert alone["highly_variable"].sum() == 2000 or adata.n_vars < 2000


def test_preprocess_with_one_hvg_method_name():
    adata = make_counts()
    preprocess(adata, hvg_methods="cell_ranger", backend="native")
    np.testing.assert_array_equal(
        adata.var["highly_variable"], adata.var["highly_variable_cell_ranger"]
    )
    assert "zheng2017" in list(adata.uns["sclab"]["citations"])


def test_hvg_bad_arguments():
    adata = make_counts()
    with pytest.raises(ValueError, match="methods"):
        highly_variable_genes(adata, ["seurat", "nope"])
    with pytest.raises(ValueError, match="methods"):
        highly_variable_genes(adata, [])
    with pytest.raises(ValueError, match="combine"):
        highly_variable_genes(adata, combine="xor")


def test_backend_argument(monkeypatch):
    adata = make_counts()
    with pytest.raises(ValueError, match="backend"):
        qc(adata, backend="numba")
    monkeypatch.setitem(sys.modules, "scanpy", None)
    assert native.resolve_backend("auto") == "native"
    with pytest.raises(ImportError, match="Needs scanpy"):
        qc(adata, backend="scanpy")
    qc(adata)  # auto: native
    assert "total_counts" in adata.obs
