"""numpy/scipy/scikit-learn ports of the ``scanpy.pp`` functions that sclab's
QC, Preprocess, PCA and Neighbors run, for environments without scanpy (e.g. Pyodide).

Ported from scanpy 1.12.5 (https://github.com/scverse/scanpy, BSD-3-Clause,
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab; see
THIRD_PARTY_NOTICES.md), files ``src/scanpy/preprocessing/``:

- ``_qc.py``: ``calculate_qc_metrics`` (``describe_obs``, ``describe_var``)
- ``_simple.py``: ``filter_cells``, ``filter_genes``, ``log1p``, ``regress_out``
- ``_normalization.py``: ``normalize_total``
- ``_scale.py``: ``scale``
- ``_highly_variable_genes.py``: ``highly_variable_genes`` (flavors seurat,
  cell_ranger, seurat_v3, seurat_v3_paper; with ``batch_key``)
- ``_pca/__init__.py``: ``pca`` (scikit-learn PCA with the ARPACK solver)

and ``src/scanpy/neighbors/`` (``__init__.py``, ``_common.py``): ``neighbors``
with exact kNN (scikit-learn) and umap-learn's connectivities
(:mod:`._fuzzy_simplicial_set`, ported from umap-learn 0.5.12).

The numba kernels are replaced by vectorized numpy/scipy code. Only the options
sclab uses are ported, with ``scanpy.pp``'s call signatures, so the library
calls ``pp.<function>(...)`` with ``pp`` either ``scanpy.pp`` or this module
(see :func:`get_pp`). The results match scanpy's within float rounding (see
``tests/sclab/preprocess/test_scanpy_port.py``).

References: Wolf, F. A., Angerer, P., & Theis, F. J. (2018). SCANPY:
large-scale single-cell gene expression data analysis. Genome Biology, 19, 15.
https://doi.org/10.1186/s13059-017-1382-0 -- and, for the methods,
Satija et al. (2015) (seurat), Zheng et al. (2017) (cell_ranger), Stuart et al.
(2019) (seurat_v3), Pedregosa et al. (2011) and Lehoucq et al. (1998) (PCA:
scikit-learn, ARPACK), McInnes et al. (2018) (neighbors: UMAP connectivities);
see :mod:`sclab._citations`.
"""

import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.sparse import issparse

from .._compat import missing, require

BACKENDS = ("auto", "native", "scanpy")
_CHUNK = 2048  # rows per block in dense reductions: bounds the float64 copies


def resolve_backend(backend: str) -> str:
    """``"scanpy"`` or ``"native"``; ``"auto"`` is scanpy if it is installed."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "scanpy":
        require("scanpy")
    if backend == "auto":
        return "scanpy" if missing("scanpy") is None else "native"
    return backend


def native_backend(params: dict) -> dict:
    """A step's codegen kwargs: ``backend="native"`` if its recorded run was
    native (a scanpy run keeps the default ``"auto"``)."""
    return {"backend": "native"} if params.get("backend") == "native" else {}


def get_pp(backend: str):
    """``scanpy.pp`` or this module, as :func:`resolve_backend` says."""
    if resolve_backend(backend) == "scanpy":
        import scanpy as sc

        return sc.pp
    return sys.modules[__name__]


# --- reductions --------------------------------------------------------------


def _sum(x, axis: int, dtype=None) -> np.ndarray:
    return np.asarray(x.sum(axis=axis, dtype=dtype)).ravel()


def _nnz(x, axis: int) -> np.ndarray:
    """Stored nonzeros per row (``axis=1``) or column (``axis=0``)."""
    if not issparse(x):
        return np.count_nonzero(x, axis=axis)
    if x.format not in ("csr", "csc"):
        x = x.tocsr()
    if axis == (1 if x.format == "csr" else 0):
        return np.diff(x.indptr)
    return np.bincount(x.indices, minlength=x.shape[1 - axis])


def _mean_var(x) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and variance (ddof=1) in float64, as
    ``fast_array_utils.stats.mean_var(x, axis=0, correction=1)`` (scanpy's):
    sums in float64 of the values and of their squares, taken in x's dtype.
    """
    n, m = x.shape
    if issparse(x):
        x = x.tocsr()
        s = np.bincount(x.indices, weights=x.data, minlength=m)
        ss = np.bincount(x.indices, weights=x.data * x.data, minlength=m)
    else:
        s = np.sum(x, axis=0, dtype=np.float64)
        ss = np.sum(x**2, axis=0, dtype=np.float64)
    mean = s / n
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (ss / n - mean**2) * (n / (n - 1))
    return mean, var


# --- QC: _qc.py, _simple.py ---------------------------------------------------


def calculate_qc_metrics(
    adata: AnnData, *, percent_top=None, log1p: bool = False, inplace: bool = True
) -> None:
    """``scanpy.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False,
    inplace=True)`` on ``adata.X`` (the only mode ported).

    Adds ``n_genes_by_counts`` and ``total_counts`` to ``adata.obs``;
    ``n_cells_by_counts``, ``mean_counts``, ``pct_dropout_by_counts`` and
    ``total_counts`` to ``adata.var``. Removes explicit zeros of a sparse X.
    """
    if percent_top or log1p or not inplace:
        raise NotImplementedError("only percent_top=None, log1p=False, inplace=True")
    x = adata.X
    if issparse(x):
        x.eliminate_zeros()
    n_cells = _nnz(x, 0)
    adata.obs["n_genes_by_counts"] = _nnz(x, 1)
    adata.obs["total_counts"] = _sum(x, 1)
    adata.var["n_cells_by_counts"] = n_cells
    total = _sum(x, 0)
    adata.var["mean_counts"] = total / x.shape[0]
    adata.var["pct_dropout_by_counts"] = (1 - n_cells / x.shape[0]) * 100
    adata.var["total_counts"] = total


def filter_cells(adata: AnnData, *, min_genes: int) -> None:
    """``scanpy.pp.filter_cells(adata, min_genes=...)``: keep cells with at least
    ``min_genes`` positive values; adds ``adata.obs["n_genes"]``."""
    number = _sum(adata.X > 0, 1)
    adata.obs["n_genes"] = number
    adata._inplace_subset_obs(number >= min_genes)


def filter_genes(adata: AnnData, *, min_cells: int) -> None:
    """``scanpy.pp.filter_genes(adata, min_cells=...)``: keep genes positive in
    at least ``min_cells`` cells; adds ``adata.var["n_cells"]``."""
    number = _sum(adata.X > 0, 0)
    adata.var["n_cells"] = number
    adata._inplace_subset_var(number >= min_cells)


# --- normalization and transforms: _normalization.py, _simple.py, _scale.py ---


def normalize_total(adata: AnnData, *, target_sum: float | None = None) -> None:
    """``scanpy.pp.normalize_total(adata, target_sum=...)`` on ``adata.X``:
    each cell's counts are scaled to sum to ``target_sum`` (default: the median
    of the nonzero cell totals). Integer X becomes float32; cells with zero
    counts are left as they are."""
    x = adata.X
    if issparse(x) and x.format == "csc":
        x = x.tocsr()
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32)
    if issparse(x):  # scanpy's numba kernel sums each row in float64
        counts = _sum(x, 1, dtype=np.float64).astype(x.dtype)
    else:
        counts = _sum(x, 1)
    if target_sum is None:
        target_sum = np.median(counts[counts > 0])
    counts = counts / target_sum
    if not np.all(counts > 0):
        warnings.warn("Some cells have zero counts", UserWarning, stacklevel=2)
    counts = counts + (counts == 0)  # no division by zero
    if issparse(x):
        x.data = x.data / np.repeat(counts, np.diff(x.indptr))
    else:
        np.true_divide(x, counts[:, None], out=x)
    adata.X = x


def log1p(data):
    """``scanpy.pp.log1p`` (natural log, not chunked).

    An AnnData's X is transformed (and ``uns["log1p"]`` set); an array or
    sparse matrix is transformed in place if float (else as a float64 copy)
    and returned.
    """
    if isinstance(data, AnnData):
        data.X = log1p(data.X)
        data.uns["log1p"] = {"base": None}
        return None
    if issparse(data):
        if data.format not in ("csr", "csc"):
            data = data.tocsr()
        if data.dtype not in (np.float32, np.float64):
            data = data.astype(np.float64)
        np.log1p(data.data, out=data.data)
        return data
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(float)
    np.log1p(data, out=data)
    return data


def scale(data, *, zero_center: bool = True):
    """``scanpy.pp.scale(data, zero_center=False)`` (the only mode ported):
    divide each gene by its standard deviation (ddof=1; 1 where it is 0).

    An AnnData's X is scaled in place and ``var["mean"]``, ``var["std"]`` set;
    an array or sparse matrix is scaled in place (integer input: a float64
    copy) and returned.
    """
    if zero_center:
        raise NotImplementedError("only zero_center=False is ported")
    if isinstance(data, AnnData):
        data.X, data.var["mean"], data.var["std"] = _scale(data.X)
        return None
    return _scale(data)[0]


def _scale(x):
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64)
    mean, var = _mean_var(x)
    std = np.sqrt(var)
    std[std == 0] = 1
    if issparse(x):
        if x.format == "csr":
            cols = x.indices
        else:  # csc
            cols = np.repeat(np.arange(x.shape[1]), np.diff(x.indptr))
        x.data /= std[cols]
    else:
        np.true_divide(x, std, out=x)
    return x, mean, std


def regress_out(adata: AnnData, keys, *, n_jobs: int | None = None) -> None:
    """``scanpy.pp.regress_out(adata, keys)`` for numeric ``adata.obs`` keys:
    replace X (densified) by the residuals of a per-gene linear regression on
    the keys plus an intercept. ``n_jobs`` is ignored.
    """
    keys = [keys] if isinstance(keys, str) else list(keys)
    if isinstance(adata.obs[keys[0]].dtype, pd.CategoricalDtype):
        raise NotImplementedError("regressing out a categorical key is not ported")
    x = adata.X
    if issparse(x):
        x = x.toarray()
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32 if x.dtype.itemsize <= 4 else np.float64)
    regressors = adata.obs[keys].copy()
    regressors.insert(0, "ones", 1.0)
    r = regressors.to_numpy()
    gram = r.T @ r
    if np.linalg.det(gram) != 0:  # scanpy's numpy_regress_out, by row blocks
        rtx = sum(
            r[i : i + _CHUNK].T @ x[i : i + _CHUNK] for i in range(0, len(x), _CHUNK)
        )
        coeff = np.linalg.inv(gram) @ rtx
        for i in range(0, len(x), _CHUNK):
            x[i : i + _CHUNK] -= r[i : i + _CHUNK] @ coeff
        res = x
    else:  # scanpy fits a Gaussian GLM per gene (statsmodels): the same residuals
        res = np.asarray(x, dtype=np.float64)
        varying = (x != x[:1]).any(axis=0)  # constant genes are kept as they are
        y = res[:, varying]
        res[:, varying] = y - r @ (np.linalg.pinv(r) @ y)
    adata.X = res


# --- highly variable genes: _highly_variable_genes.py -------------------------


def _get_rep(adata: AnnData, layer: str | None):
    return adata.X if layer is None else adata.layers[layer]


@dataclass
class _Cutoffs:  # scanpy's defaults
    min_disp: float = 0.5
    max_disp: float = np.inf
    min_mean: float = 0.0125
    max_mean: float = 3

    def in_bounds(self, mean, dispersion_norm):
        return (
            (mean > self.min_mean)
            & (mean < self.max_mean)
            & (dispersion_norm > self.min_disp)
            & (dispersion_norm < self.max_disp)
        )


def highly_variable_genes(
    adata: AnnData,
    *,
    layer: str | None = None,
    n_top_genes: int | None = None,
    flavor: str = "seurat",
    inplace: bool = True,
    batch_key: str | None = None,
) -> pd.DataFrame | None:
    """``scanpy.pp.highly_variable_genes`` with its defaults for the cutoffs
    (used when ``n_top_genes`` is None), ``n_bins=20``, ``span=0.3``,
    ``subset=False``, ``check_values=True``.

    Flavors ``"seurat"`` and ``"cell_ranger"`` (log data; Satija et al. 2015,
    Zheng et al. 2017) and ``"seurat_v3"``, ``"seurat_v3_paper"`` (counts;
    Stuart et al. 2019; need ``skmisc``). With ``inplace`` the results go to
    ``adata.var`` (and ``adata.uns["hvg"]``) as scanpy writes them, else they
    are returned as a DataFrame.
    """
    if flavor in ("seurat_v3", "seurat_v3_paper"):
        return _hvg_seurat_v3(
            adata,
            flavor=flavor,
            layer=layer,
            n_top_genes=2000 if n_top_genes is None else n_top_genes,
            batch_key=batch_key,
            inplace=inplace,
        )
    if flavor not in ("seurat", "cell_ranger"):
        raise ValueError(f"Did not recognize flavor {flavor}")
    cutoff = _Cutoffs() if n_top_genes is None else n_top_genes
    x = _get_rep(adata, layer)
    log_base = adata.uns.get("log1p", {}).get("base")
    kwargs = dict(flavor=flavor, cutoff=cutoff, log_base=log_base)
    if not batch_key:
        df = _hvg_single_batch(x, adata.var_names, **kwargs)
    else:
        df = _hvg_batched(x, adata.var_names, adata.obs[batch_key], **kwargs)
    if not inplace:
        return df
    adata.uns["hvg"] = {"flavor": flavor}
    adata.var["highly_variable"] = df["highly_variable"]
    adata.var["means"] = df["means"]
    adata.var["dispersions"] = df["dispersions"]
    adata.var["dispersions_norm"] = df["dispersions_norm"].astype(np.float32)
    if batch_key is not None:
        adata.var["highly_variable_nbatches"] = df["highly_variable_nbatches"]
        adata.var["highly_variable_intersection"] = df["highly_variable_intersection"]
    return None


def _hvg_single_batch(
    x, var_names, *, flavor, cutoff, log_base, filter_unexpressed_genes=False
) -> pd.DataFrame:
    if filter_unexpressed_genes:
        filt = _sum(x > 0, 0) >= 1
    else:
        filt = np.ones(x.shape[1], dtype=bool)
    n_removed = np.sum(~filt)
    if n_removed:
        x = x[:, filt]
    if flavor == "seurat":
        x = x.copy()
        if log_base is not None:
            x *= np.log(log_base)
        np.expm1(x.data if issparse(x) else x, out=x.data if issparse(x) else x)

    mean, var = _mean_var(x)
    mean[mean == 0] = 1e-12
    dispersion = var / mean
    if flavor == "seurat":  # logarithmized mean as in Seurat
        dispersion[dispersion == 0] = np.nan
        dispersion = np.log(dispersion)
        mean = np.log1p(mean)

    df = pd.DataFrame(dict(means=mean, dispersions=dispersion))
    df["mean_bin"] = _get_mean_bins(df["means"], flavor, 20)
    disp_stats = _get_disp_stats(df, flavor)
    df["dispersions_norm"] = (df["dispersions"] - disp_stats["avg"]) / disp_stats["dev"]
    df["highly_variable"] = _subset_genes(
        mean=mean,
        dispersion_norm=df["dispersions_norm"].to_numpy(),
        cutoff=cutoff,
        n_vars=int(filt.sum()),
    )
    df.index = var_names[filt]
    if n_removed > 0:  # zeros for the genes filtered out
        missing_hvg = pd.DataFrame(
            np.zeros((n_removed, len(df.columns))), columns=df.columns
        )
        missing_hvg["highly_variable"] = missing_hvg["highly_variable"].astype(bool)
        missing_hvg.index = var_names[~filt]
        df = pd.concat([df, missing_hvg]).loc[var_names]
    return df


def _get_mean_bins(means: pd.Series, flavor: str, n_bins: int) -> pd.Series:
    if flavor == "seurat":
        bins = n_bins
    else:  # cell_ranger
        bins = np.r_[-np.inf, np.percentile(means, np.arange(10, 105, 5)), np.inf]
    rv = pd.cut(means, bins=bins)
    return rv.cat.set_categories(rv.cat.categories.astype("string"), rename=True)


def _get_disp_stats(df: pd.DataFrame, flavor: str) -> pd.DataFrame:
    disp_grouped = df.groupby("mean_bin", observed=True)["dispersions"]
    if flavor == "seurat":
        disp_bin_stats = disp_grouped.agg(avg="mean", dev="std")
        _postprocess_dispersions_seurat(disp_bin_stats)
    else:  # cell_ranger
        disp_bin_stats = disp_grouped.agg(avg="median", dev=_mad)
    return disp_bin_stats.loc[df["mean_bin"]].set_index(df.index)


def _postprocess_dispersions_seurat(disp_bin_stats: pd.DataFrame) -> None:
    # a bin with a single gene has nan std: its normalized dispersion is set to 1
    one_gene_per_bin = disp_bin_stats["dev"].isnull()
    if not one_gene_per_bin.any():
        return
    disp_bin_stats.loc[one_gene_per_bin, "dev"] = disp_bin_stats.loc[
        one_gene_per_bin, "avg"
    ]
    disp_bin_stats.loc[one_gene_per_bin, "avg"] = 0


def _mad(a) -> float:
    """``statsmodels.robust.mad(a)``: the normal-consistent median absolute
    deviation; 0.6744897501960817 is ``scipy.stats.norm.ppf(0.75)``."""
    a = np.asarray(a, dtype=float)
    with warnings.catch_warnings():  # "Mean of empty slice"
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.median(np.abs(a - np.median(a)) / 0.6744897501960817)


def _subset_genes(*, mean, dispersion_norm, cutoff, n_vars: int) -> np.ndarray:
    if isinstance(cutoff, _Cutoffs):
        dispersion_norm = np.nan_to_num(dispersion_norm)  # similar to Seurat
        return cutoff.in_bounds(mean, dispersion_norm)
    n_top_genes = min(cutoff, n_vars)
    disp_cut_off = _nth_highest(dispersion_norm, n_top_genes)
    return np.nan_to_num(dispersion_norm, nan=-np.inf) >= disp_cut_off


def _nth_highest(x: np.ndarray, n: int) -> float:
    x = x[~np.isnan(x)]
    if n > x.size:
        msg = (
            f"`n_top_genes` (={n}) > number of normalized dispersions (={x.size}), "
            "returning all genes with normalized dispersions."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        n = x.size
    x[::-1].sort()
    return x[n - 1]


def _hvg_batched(x, var_names, batch: pd.Series, *, cutoff, **kwargs) -> pd.DataFrame:
    batch = batch.astype("category")  # scanpy sanitizes adata.obs (in place)
    batches = batch.cat.categories
    dfs = [
        _hvg_single_batch(
            x[(batch == b).to_numpy()],
            var_names,
            cutoff=cutoff,
            filter_unexpressed_genes=True,
            **kwargs,
        )
        for b in batches
    ]
    df = pd.concat(dfs, axis=0)
    df["highly_variable"] = df["highly_variable"].astype(int)
    df = df.groupby(df.index, observed=True).agg(
        dict(
            means="mean",
            dispersions="mean",
            dispersions_norm="mean",
            highly_variable="sum",
        )
    )
    df["highly_variable_nbatches"] = df["highly_variable"]
    df["highly_variable_intersection"] = df["highly_variable_nbatches"] == len(batches)
    if not isinstance(cutoff, _Cutoffs):  # n_top_genes
        # sort genes by how often they are selected as hvg within each batch and
        # break ties with normalized dispersion across batches
        df = df.sort_values(
            ["highly_variable_nbatches", "dispersions_norm"],
            ascending=False,
            na_position="last",
        )
        df["highly_variable"] = np.arange(df.shape[0]) < cutoff
        df = df.loc[var_names]
    else:
        df["dispersions_norm"] = df["dispersions_norm"].fillna(0)  # similar to Seurat
        df["highly_variable"] = cutoff.in_bounds(df["means"], df["dispersions_norm"])
    return df


def _nonnegative_integers(x) -> bool:
    data = x.data if issparse(x) else x
    if np.signbit(data).any():
        return False
    if np.issubdtype(data.dtype, np.integer):
        return True
    return not np.any((data % 1) != 0)


def _clip_square_sum(x, clip_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene sums of ``min(x, clip_val)**2`` and ``min(x, clip_val)``."""
    m = x.shape[1]
    if issparse(x):
        x = x.tocsr()
        v = np.minimum(x.data.astype(np.float64), clip_val[x.indices])
        return (
            np.bincount(x.indices, weights=v * v, minlength=m),
            np.bincount(x.indices, weights=v, minlength=m),
        )
    sq, s = np.zeros(m), np.zeros(m)
    for i in range(0, x.shape[0], _CHUNK):
        c = np.minimum(np.asarray(x[i : i + _CHUNK], dtype=np.float64), clip_val)
        s += c.sum(axis=0)
        sq += np.square(c, out=c).sum(axis=0)
    return sq, s


def _hvg_seurat_v3(
    adata: AnnData, *, flavor, layer, n_top_genes, batch_key, inplace
) -> pd.DataFrame | None:
    require("skmisc")
    from skmisc.loess import loess

    df = pd.DataFrame(index=adata.var_names)
    data = _get_rep(adata, layer)
    if not _nonnegative_integers(data):
        msg = f"`{flavor=!r}` expects raw count data, but non-integers were found."
        warnings.warn(msg, UserWarning, stacklevel=3)

    df["means"], df["variances"] = _mean_var(data)
    if batch_key is None:
        batch_info = np.zeros(adata.shape[0], dtype=int)
    else:
        batch_info = adata.obs[batch_key].to_numpy()
    norm_gene_vars = []
    for b in np.unique(batch_info):
        data_batch = data[batch_info == b]
        if batch_key is None:
            mean, var = df["means"].to_numpy(), df["variances"].to_numpy()
        else:
            mean, var = _mean_var(data_batch)

        estimat_var = np.zeros(data.shape[1], dtype=np.float64)
        if (not_const := var > 0).any():
            y = np.log10(var[not_const])
            x = np.log10(mean[not_const])
            model = loess(x, y, span=0.3, degree=2)
            model.fit()
            estimat_var[not_const] = model.outputs.fitted_values
        reg_std = np.sqrt(10**estimat_var)

        # clip large values as in Seurat
        n_obs = data_batch.shape[0]
        clip_val = reg_std * np.sqrt(n_obs) + mean
        squared_batch_counts_sum, batch_counts_sum = _clip_square_sum(
            data_batch, clip_val
        )
        norm_gene_var = (1 / ((n_obs - 1) * np.square(reg_std))) * (
            (n_obs * np.square(mean))
            + squared_batch_counts_sum
            - 2 * batch_counts_sum * mean
        )
        norm_gene_vars.append(norm_gene_var.reshape(1, -1))

    norm_gene_vars = np.concatenate(norm_gene_vars, axis=0)
    # argsort twice gives ranks, small rank means most variable
    ranked_norm_gene_vars = np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1)
    # this is done in SelectIntegrationFeatures() in Seurat v3
    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    num_batches_high_var = np.sum(
        (ranked_norm_gene_vars < n_top_genes).astype(int), axis=0
    )
    ranked_norm_gene_vars[ranked_norm_gene_vars >= n_top_genes] = np.nan
    ma_ranked = np.ma.masked_invalid(ranked_norm_gene_vars)
    median_ranked = np.ma.median(ma_ranked, axis=0).filled(np.nan)

    df = df.assign(
        gene_name=df.index,
        highly_variable_nbatches=num_batches_high_var,
        highly_variable_rank=median_ranked,
        variances_norm=np.mean(norm_gene_vars, axis=0),
    )
    if flavor == "seurat_v3":
        sort_cols = ["highly_variable_rank", "highly_variable_nbatches"]
        sort_ascending = [True, False]
    else:  # seurat_v3_paper
        sort_cols = ["highly_variable_nbatches", "highly_variable_rank"]
        sort_ascending = [False, True]
    sorted_index = (
        df[sort_cols]
        .sort_values(sort_cols, ascending=sort_ascending, na_position="last")
        .index
    )
    df["highly_variable"] = False
    df.loc[sorted_index[: int(n_top_genes)], "highly_variable"] = True

    if not inplace:
        if batch_key is None:
            df = df.drop(["highly_variable_nbatches"], axis=1)
        return df
    adata.uns["hvg"] = {"flavor": flavor}
    for key in ["highly_variable", "highly_variable_rank", "means", "variances"]:
        adata.var[key] = df[key].to_numpy()
    adata.var["variances_norm"] = (
        df["variances_norm"].to_numpy().astype("float64", copy=False)
    )
    if batch_key is not None:
        adata.var["highly_variable_nbatches"] = df[
            "highly_variable_nbatches"
        ].to_numpy()
    return None


# --- PCA: _pca/__init__.py ------------------------------------------------------

_EMPTY = object()  # scanpy's default mask_var: "highly_variable" if in adata.var


def pca(
    adata: AnnData,
    n_comps: int | None = None,
    *,
    layer: str | None = None,
    zero_center: bool = True,
    svd_solver: str | None = None,
    random_state=0,
    mask_var=_EMPTY,
) -> None:
    """``scanpy.pp.pca`` on an AnnData, not chunked, with the ARPACK solver.

    ``zero_center=True`` fits :class:`sklearn.decomposition.PCA`
    (``svd_solver="arpack"``; sparse input is centered implicitly, not
    densified), ``False`` :class:`sklearn.decomposition.TruncatedSVD`
    (``algorithm="arpack"``; scanpy's default there would be randomized).
    Stores ``obsm["X_pca"]`` (float32), ``varm["PCs"]`` (zeros outside
    ``mask_var``) and ``uns["pca"]`` (``params``, ``variance``,
    ``variance_ratio``) as scanpy does.
    """
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.utils import check_random_state

    if svd_solver not in (None, "arpack"):
        raise NotImplementedError("only svd_solver='arpack' is ported")
    if mask_var is _EMPTY:
        mask_var = "highly_variable" if "highly_variable" in adata.var else None
    mask = None if mask_var is None else _check_mask(adata, mask_var)
    x = _get_rep(adata, layer)
    if mask is not None:
        x = x[:, mask]
    if n_comps is None:
        min_dim = min(x.shape)
        n_comps = min_dim - 1 if min_dim <= 50 else 50  # scanpy settings.N_PCS
    random_state = check_random_state(random_state)
    if zero_center:
        pca_ = PCA(n_components=n_comps, svd_solver="arpack", random_state=random_state)
    else:
        pca_ = TruncatedSVD(
            n_components=n_comps, random_state=random_state, algorithm="arpack"
        )
    x_pca = pca_.fit_transform(x)
    if x_pca.dtype.descr != np.dtype("float32").descr:
        x_pca = x_pca.astype("float32")

    adata.obsm["X_pca"] = x_pca
    if mask is not None:
        pcs = np.zeros(shape=(adata.n_vars, n_comps))
        pcs[mask] = pca_.components_.T
        adata.varm["PCs"] = pcs
    else:
        adata.varm["PCs"] = pca_.components_.T
    adata.uns["pca"] = dict(
        params=dict(
            zero_center=zero_center,
            use_highly_variable=isinstance(mask_var, str)
            and mask_var == "highly_variable",
            mask_var=mask_var,
            **(dict(layer=layer) if layer is not None else {}),
        ),
        variance=pca_.explained_variance_,
        variance_ratio=pca_.explained_variance_ratio_,
    )


def _check_mask(adata: AnnData, mask) -> np.ndarray:
    if isinstance(mask, str):
        if mask not in adata.var:
            raise ValueError(f"Did not find `adata.var[{mask!r}]`.")
        mask = adata.var[mask].to_numpy()
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (adata.n_vars,):
        raise ValueError("mask_var must be a boolean column or array, one per gene")
    return mask


# --- neighbors: neighbors/__init__.py, neighbors/_common.py, _connectivity.py ---


def neighbors(
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    *,
    use_rep: str,
    metric: str = "euclidean",
    random_state=0,
) -> None:
    """``scanpy.pp.neighbors`` on ``adata.obsm[use_rep]`` (its first ``n_pcs``
    columns), with ``method="umap"`` and ``knn=True``.

    The kNN search is exact (scikit-learn, brute force), as scanpy's for fewer
    than 4096 cells; scanpy searches more cells approximately (pynndescent),
    so the graphs differ there. The connectivities are umap-learn's fuzzy
    simplicial set (:mod:`._fuzzy_simplicial_set`). ``random_state`` is only
    recorded: the exact search needs no seed. Stores ``obsp["distances"]``,
    ``obsp["connectivities"]`` and ``uns["neighbors"]`` as scanpy does.
    """
    x = _choose_rep(adata, use_rep, n_pcs)
    if n_neighbors > adata.n_obs:  # very small datasets
        n_neighbors = 1 + int(0.5 * adata.n_obs)
    distances, connectivities = knn_graph(x, n_neighbors, metric)
    adata.uns["neighbors"] = dict(
        connectivities_key="connectivities",
        distances_key="distances",
        params=dict(
            n_neighbors=n_neighbors,
            method="umap",
            random_state=random_state,
            metric=metric,
            use_rep=use_rep,
            **({} if n_pcs is None else dict(n_pcs=n_pcs)),
        ),
    )
    adata.obsp["distances"] = distances
    adata.obsp["connectivities"] = connectivities


def _choose_rep(adata: AnnData, use_rep: str, n_pcs: int | None):
    """scanpy's ``_choose_representation`` for a ``use_rep`` in ``obsm``."""
    if use_rep not in adata.obsm:
        raise ValueError(f"Did not find {use_rep} in `.obsm.keys()`.")
    x = adata.obsm[use_rep]
    if n_pcs is None:
        return x
    if n_pcs > x.shape[1]:
        raise ValueError(f"{use_rep} has fewer than n_pcs={n_pcs} dimensions")
    return x[:, :n_pcs]


def knn_graph(x, n_neighbors: int, metric: str = "euclidean"):
    """The ``(distances, connectivities)`` CSR matrices of
    :func:`neighbors`: exact kNN of each row of ``x`` (itself included, as the
    first of ``n_neighbors``), then the fuzzy simplicial set."""
    from sklearn.neighbors import NearestNeighbors

    from ._fuzzy_simplicial_set import fuzzy_simplicial_set

    n = x.shape[0]
    # scanpy's KNeighborsTransformer(algorithm="brute").fit_transform(x): each
    # cell is its own extra neighbor
    k = min(n - 1, n_neighbors)
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric).fit(x)
    dist, ind = nn.kneighbors(x, n_neighbors=k + 1)
    dist[ind == np.arange(n)[:, None]] = 0  # scanpy zeroes the diagonal
    # shortcut: assumes the cell is its own first neighbor (scanpy prepends it
    # when no cell is); true for an exact search unless every cell has duplicates
    ind, dist = ind[:, :n_neighbors], dist[:, :n_neighbors]
    m = n_neighbors - 1
    distances = sparse.csr_matrix(
        (dist[:, 1:].ravel(), ind[:, 1:].ravel(), np.arange(0, n * m + 1, m)),
        shape=(n, n),
    )
    return distances, fuzzy_simplicial_set(ind, dist, n, n_neighbors)
