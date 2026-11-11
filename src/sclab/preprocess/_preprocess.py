from collections.abc import Sequence
from typing import Literal

import numpy as np
from anndata import AnnData
from tqdm.auto import tqdm

from .._citations import record_citations
from ._hvg import highly_variable_genes, hvg_citations
from ._scanpy_port import get_pp


def preprocess(
    adata: AnnData,
    counts_layer: str = "counts",
    group_by: str | None = None,
    min_cells: int = 5,
    min_genes: int = 5,
    compute_hvg: bool = True,
    hvg_methods: str | Sequence[str] = ("seurat", "seurat_v3_paper"),
    hvg_combine: Literal["union", "intersection"] = "union",
    regress_total_counts: bool = False,
    regress_n_genes: bool = False,
    normalization_method: Literal["library", "weighted", "none"] = "library",
    target_scale: float = 1e4,
    log1p: bool = True,
    scale: bool = True,
    backend: str = "auto",
    n_top_genes: int | None = None,
):
    """Normalize, transform, and scale single-cell RNA-seq count data.

    Applies a configurable preprocessing pipeline: optional filtering,
    highly-variable gene selection, normalization, log1p transformation,
    optional covariate regression, and per-group scaling. The resulting
    processed matrix is stored in a new named layer whose suffix encodes
    the applied steps (e.g. ``counts_normt_log1p_scale``).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Modified in-place.
    counts_layer : str, optional
        Layer containing raw counts. Default is ``"counts"``.
    group_by : str or None, optional
        Column in ``adata.obs`` for per-group HVG selection, normalization,
        and scaling. When set, batch-aware processing is applied. Default
        is None.
    min_cells : int, optional
        Minimum number of cells a gene must be detected in to be retained.
        Default is 5.
    min_genes : int, optional
        Minimum number of genes detected per cell to be retained. Default
        is 5.
    compute_hvg : bool, optional
        If True, compute highly variable genes with
        :func:`sclab.preprocess.highly_variable_genes` (``hvg_methods``,
        ``hvg_combine``; the dispersion flavors on ``log1p`` of the counts)
        and store the result in ``adata.var["highly_variable"]``, with one
        ``highly_variable_<method>`` column per method. With ``group_by``,
        genes are selected per group (``highly_variable_<group>``) and the
        groups' selections are merged by union; a group named like one of
        ``hvg_methods`` is an error (its column would clash). Default is True.
    hvg_methods : str or sequence of str, optional
        HVG flavors: ``"seurat"``, ``"cell_ranger"``, ``"seurat_v3"``,
        ``"seurat_v3_paper"``. Default ``("seurat", "seurat_v3_paper")``
        (``seurat_v3_paper`` selects as many genes as ``seurat``).
    hvg_combine : {"union", "intersection"}, optional
        How the methods' selections are combined. Default ``"union"``.
    n_top_genes : int or None, optional
        Genes each HVG method selects (see
        :func:`sclab.preprocess.highly_variable_genes`). If None (default),
        the dispersion methods use scanpy's default cutoffs and the seurat_v3
        methods select as many genes as the first method.
    regress_total_counts : bool, optional
        If True, regress out total counts (or log1p total counts if
        ``log1p=True``) per cell. Default is False.
    regress_n_genes : bool, optional
        If True, regress out the number of detected genes per cell.
        Default is False.
    normalization_method : {"library", "weighted", "none"}, optional
        Normalization strategy. ``"library"`` applies library-size
        normalization to ``target_scale`` counts; ``"weighted"`` applies
        entropy-weighted normalization; ``"none"`` skips normalization.
        Default is ``"library"``.
    target_scale : float, optional
        Target sum for library-size normalization (counts per cell after
        normalization). Default is 1e4.
    log1p : bool, optional
        If True, apply log(x + 1) transformation after normalization.
        Default is True.
    scale : bool, optional
        If True, scale each gene to unit variance (zero-center disabled).
        Applied per group when ``group_by`` is set. Default is True.
    backend : {"auto", "native", "scanpy"}, optional
        ``"scanpy"`` runs the scanpy functions (``calculate_qc_metrics``,
        ``filter_cells``, ``filter_genes``, ``highly_variable_genes``,
        ``normalize_total``, ``log1p``, ``regress_out``, ``scale``),
        ``"native"`` sclab's port of them (numpy/scipy; same results within
        float rounding), ``"auto"`` (default) scanpy if it is installed.

    Returns
    -------
    None
        Modifies ``adata`` in-place. Stores the processed matrix in a new
        layer and updates ``adata.X``.

    References
    ----------
    Wolf et al. (2018). SCANPY: large-scale single-cell gene expression
    data analysis. *Genome Biology*, 19, 15.
    https://doi.org/10.1186/s13059-017-1382-0

    Satija et al. (2015). Spatial reconstruction of single-cell gene
    expression data. *Nature Biotechnology*, 33, 495–502.
    https://doi.org/10.1038/nbt.3192

    Stuart et al. (2019). Comprehensive Integration of Single-Cell Data.
    *Cell*, 177, 1888–1902.e21.
    https://doi.org/10.1016/j.cell.2019.05.031

    Zheng et al. (2017). Massively parallel digital transcriptional profiling
    of single cells. *Nature Communications*, 8, 14049.
    https://doi.org/10.1038/ncomms14049
    """
    pp = get_pp(backend)
    if compute_hvg and group_by is not None:  # before adata is changed
        methods = {hvg_methods} if isinstance(hvg_methods, str) else set(hvg_methods)
        if clash := sorted(methods & set(map(str, adata.obs[group_by].unique()))):
            raise ValueError(
                f"group_by={group_by!r} has groups named like HVG methods "
                f"({', '.join(clash)}): their highly_variable_<name> columns "
                "would clash; rename the groups"
            )

    from ._normalize_weighted import apply_by_rows, normalize_weighted

    with tqdm(total=100, bar_format="{percentage:3.0f}%|{bar}|") as pbar:
        if counts_layer not in adata.layers:
            adata.layers[counts_layer] = adata.X.copy()

        if f"{counts_layer}_log1p" not in adata.layers:
            adata.layers[f"{counts_layer}_log1p"] = pp.log1p(
                adata.layers[counts_layer].copy()
            )
        pbar.update(10)

        adata.X = adata.layers[counts_layer].copy()
        pp.calculate_qc_metrics(
            adata,
            percent_top=None,
            log1p=False,
            inplace=True,
        )
        pp.filter_cells(adata, min_genes=min_genes)
        pp.filter_genes(adata, min_cells=min_cells)
        pbar.update(10)

        pp.calculate_qc_metrics(
            adata,
            percent_top=None,
            log1p=False,
            inplace=True,
        )
        pbar.update(10)

        if compute_hvg:
            hvg_kwargs = dict(
                methods=hvg_methods,
                combine=hvg_combine,
                counts_layer=counts_layer,
                n_top_genes=n_top_genes,
                backend=backend,
            )
            if group_by is not None:
                adata.var["highly_variable"] = False
                merged = None
                for name, idx in adata.obs.groupby(
                    group_by, observed=True
                ).groups.items():
                    df = highly_variable_genes(adata[idx], inplace=False, **hvg_kwargs)
                    adata.var[f"highly_variable_{name}"] = df["highly_variable"]
                    merged = df if merged is None else merged | df
                for column in merged:  # per method, and combined: union of groups
                    adata.var[column] = merged[column]
            else:
                highly_variable_genes(adata, **hvg_kwargs)

        pbar.update(10)
        pbar.update(10)

        new_layer = counts_layer
        if normalization_method == "library":
            new_layer += "_normt"
            pp.normalize_total(adata, target_sum=target_scale)
        elif normalization_method == "weighted":
            new_layer += "_normw"
            normalize_weighted(
                adata,
                target_scale=target_scale,
                batch_key=group_by,
            )

        pbar.update(10)
        pbar.update(10)

        if log1p:
            new_layer += "_log1p"
            adata.uns.pop("log1p", None)
            pp.log1p(adata)
        pbar.update(10)

        vars_to_regress = []
        if regress_n_genes:
            vars_to_regress.append("n_genes_by_counts")

        if regress_total_counts and log1p:
            adata.obs["log1p_total_counts"] = np.log1p(adata.obs["total_counts"])
            vars_to_regress.append("log1p_total_counts")
        elif regress_total_counts:
            vars_to_regress.append("total_counts")

        if vars_to_regress:
            new_layer += "_regr"
            pp.regress_out(adata, keys=vars_to_regress, n_jobs=1)
        pbar.update(10)

        if scale:
            new_layer += "_scale"
            if group_by is not None:
                group_col = adata.obs[group_by].astype("str").fillna("_unassigned")
                groups = group_col.groupby(group_col, observed=True).groups.values()
                adata.X = apply_by_rows(
                    adata.X,
                    [adata.obs.index.isin(idx) for idx in groups],
                    lambda x: pp.scale(x, zero_center=False),
                )
            else:
                pp.scale(adata, zero_center=False)

        adata.layers[new_layer] = adata.X.copy()

        pbar.update(10)

        adata.X = adata.X.astype(np.float32)

    hvg_keys = hvg_citations(hvg_methods) if compute_hvg else ()
    record_citations(adata, "scanpy", *hvg_keys)
