import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse

from ..._citations import cites
from ..._compat import requirement
from ..utils import aggregate_and_filter
from ._edgepython_numba_free import import_edgepython
from ._pseudobulk_helpers import pseudobulk_contrasts, validate_counts_layer
from ._pseudobulk_pylimma import edgepython_is_available


@cites(
    "edger",
    "edger_2010",
    "edger_glm",
    "edger_ql_lun2016",
    "edger_ql",
    "tmm",
    "edgepython",
)
def pseudobulk_edgepython(
    adata: AnnData,
    group_key: str,
    condition_group: str | list[str] | None = None,
    reference_group: str | None = None,
    cell_identity_key: str | None = None,
    layer: str = "counts",
    replicas_per_group: int = 3,
    min_cells_per_group: int = 30,
    bootstrap_sampling: bool = False,
    use_cells: dict[str, list[str]] | None = None,
    aggregate: bool = True,
    verbosity: int = 0,
    check_counts: bool = True,
    sample_key: str | None = None,
    random_state: int | None = 0,
) -> dict[str, pd.DataFrame]:
    """Pseudobulk differential expression with the edgeR quasi-likelihood pipeline.

    Pure-Python counterpart of :func:`pseudobulk_edger` (no R required), using
    ``edgepython``. Aggregates single cells into pseudobulk samples, then runs
    ``filterByExpr``, TMM normalization, ``estimateDisp``, ``glmQLFit`` and a
    ``glmQLFTest`` for each requested contrast.

    Requires the Python package ``edgepython`` (``pip install sclab[de-edger]``).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    group_key : str
        Column in ``adata.obs`` defining the experimental groups.
    condition_group : str or list of str or None, optional
        Group(s) to test against ``reference_group``. If None, every group is
        tested. Default is None.
    reference_group : str or None, optional
        Reference group for contrasts. If None, each condition group is
        contrasted with all remaining cells. Default is None.
    cell_identity_key : str or None, optional
        Column in ``adata.obs`` for stratifying contrasts by cell type or
        identity. Separate DE results are returned per identity.
        Default is None.
    layer : str, optional
        Layer containing the raw counts to aggregate. ``"X"`` uses
        ``adata.X``. Passing ``None`` raises a ``ValueError``. Ignored when
        ``aggregate=False``. Default is ``"counts"``.
    replicas_per_group : int, optional
        Number of pseudobulk replicas to create per group. Default is 3.
    min_cells_per_group : int, optional
        Minimum number of cells required for a group to be included.
        Default is 30.
    bootstrap_sampling : bool, optional
        If True, use bootstrap sampling when creating pseudobulk replicas.
        Default is False.
    use_cells : dict or None, optional
        Restrict analysis to specific cell subsets. Keys are ``adata.obs``
        columns and values are lists of categories to include. Default is
        None.
    aggregate : bool, optional
        If True, aggregate cells into pseudobulk samples before fitting.
        Default is True.
    verbosity : int, optional
        Verbosity level (0 = silent). Default is 0.
    check_counts : bool, optional
        If True, raise a ``ValueError`` when the selected layer does not look
        like raw counts (non-negative integers). Default is True.
    sample_key : str or None, optional
        Column in ``adata.obs`` with the biological samples. See
        :func:`pseudobulk_pylimma`. Default is None (random pseudo-replicas).
    random_state : int or None, optional
        Seed for the random pseudo-replica assignment. None uses the global
        random state. Default is 0.

    Returns
    -------
    dict of str to pd.DataFrame
        One DataFrame per contrast (keyed by contrast label, as in
        :func:`pseudobulk_pylimma`), indexed by ``gene_ids`` and sorted by
        p-value, with the same columns as :func:`pseudobulk_edger`:

        - ``logFC`` — log2 fold change.
        - ``logCPM`` — average log2 CPM of the gene over all samples.
        - ``F`` — quasi-likelihood F-statistic.
        - ``PValue`` — raw p-value.
        - ``FDR`` — Benjamini-Hochberg adjusted p-value.
        - ``pct_expr_cnd`` / ``pct_expr_ref`` — fraction of expressing
          cells in condition/reference group.
        - ``logCPM_cnd`` / ``logCPM_ref`` — mean log2-CPM of the
          condition/reference pseudobulk samples.

    Notes
    -----
    The design is ``~ 0 + group + replica``; the ``replica`` term is dropped
    when it is not estimable (see :func:`pseudobulk_pylimma`).

    The pipeline follows R edgeR 4.x, as used by :func:`pseudobulk_edger`:
    ``estimateDisp`` with the 4.x trend span, and ``glmQLFit`` with the mean
    trended dispersion of the 10% most abundant genes. On the same pseudobulk
    counts it reproduces R edgeR 4.4 to numerical precision. edgeR 5 instead
    estimates the dispersion inside ``glmQLFit``, which shifts F-statistics
    slightly.

    Without numba (e.g. in Pyodide) edgepython's compiled kernels are replaced
    by sclab's vectorised NumPy ports of them (same results to rounding error,
    about 10 to 15 times slower than numba once compiled).

    References
    ----------
    Chen et al. (2025). edgeR v4: powerful differential analysis of
    sequencing data with expanded functionality and improved support for
    small counts and larger datasets. *Nucleic Acids Research*, 53,
    gkaf018.
    https://doi.org/10.1093/nar/gkaf018

    Robinson, McCarthy & Smyth (2010). edgeR: a Bioconductor package for
    differential expression analysis of digital gene expression data.
    *Bioinformatics*, 26, 139-140.
    https://doi.org/10.1093/bioinformatics/btp616

    McCarthy, Chen & Smyth (2012). Differential expression analysis of
    multifactor RNA-Seq experiments with respect to biological variation.
    *Nucleic Acids Research*, 40, 4288-4297.
    https://doi.org/10.1093/nar/gks042

    Lun, Chen & Smyth (2016). It's DE-licious: a recipe for differential
    expression analyses of RNA-seq experiments using quasi-likelihood methods
    in edgeR. *Methods in Molecular Biology*, 1418, 391-416.
    https://doi.org/10.1007/978-1-4939-3578-9_19

    Chen et al. (2016). From reads to genes to pathways: differential
    expression analysis of RNA-Seq experiments using Rsubread and the
    edgeR quasi-likelihood pipeline. *F1000Research*, 5, 1438.
    https://doi.org/10.12688/f1000research.8987.2

    Robinson & Oshlack (2010). A scaling normalization method for
    differential expression analysis of RNA-seq data. *Genome Biology*,
    11, R25.
    https://doi.org/10.1186/gb-2010-11-3-r25

    Pachter (2026). Differential analysis of genomics count data with
    edge*. *bioRxiv*.
    https://doi.org/10.64898/2026.02.16.706223
    """
    if not edgepython_is_available():
        raise ImportError(f"pseudobulk_edgepython needs {requirement('edgepython')}")

    if aggregate:
        validate_counts_layer(adata, layer, check_counts)
        aggr_adata = aggregate_and_filter(
            adata,
            group_key,
            cell_identity_key,
            layer,
            replicas_per_group,
            min_cells_per_group,
            bootstrap_sampling,
            use_cells,
            sample_key=sample_key,
            random_state=random_state,
        )
    else:
        aggr_adata = adata.copy()

    return pseudobulk_contrasts(
        aggr_adata,
        group_key,
        condition_group,
        reference_group,
        cell_identity_key,
        fit=lambda aggr, design: _fit_ql_model(aggr, design, verbosity),
        test=_ql_ftest,
        sample_key=sample_key if aggregate else None,
        verbosity=verbosity,
    )


def _fit_ql_model(aggr_adata: AnnData, design: pd.DataFrame, verbosity: int):
    edgepython = import_edgepython()

    X = aggr_adata.X.toarray() if issparse(aggr_adata.X) else aggr_adata.X
    counts = np.asarray(X, dtype=float).T  # genes x samples
    genes = aggr_adata.var_names.to_numpy()

    if verbosity > 1:
        print("Dimensions before subsetting:", counts.shape)

    keep = np.asarray(edgepython.filter_by_expr(counts, design=design.values))
    counts, genes = counts[keep], genes[keep]

    if verbosity > 1:
        print("Dimensions after subsetting:", counts.shape)

    # library sizes recomputed after filtering, as keep.lib.sizes=FALSE
    y = edgepython.make_dgelist(counts)
    y = edgepython.calc_norm_factors(y)
    # edgeR 4.x estimateDisp span; edgeR 4.x glmQLFit.DGEList then uses the
    # mean trended dispersion of the top 10% genes (edgeR 5 estimates its own)
    y = edgepython.estimate_disp(y, design=design.values, legacy_span=True)
    ntop = int(np.ceil(0.1 * counts.shape[0]))
    top = np.argsort(-np.asarray(y["AveLogCPM"]), kind="stable")[:ntop]
    dispersion = float(np.mean(np.asarray(y["trended.dispersion"])[top]))

    fit = edgepython.glm_ql_fit(y, design=design.values, dispersion=dispersion)
    return fit, genes


def _ql_ftest(fit, design: pd.DataFrame, cnd: str, ref: str, genes: np.ndarray):
    edgepython = import_edgepython()

    contrast = np.zeros(design.shape[1])
    contrast[design.columns.get_loc(cnd)] = 1.0
    contrast[design.columns.get_loc(ref)] = -1.0

    qlf = edgepython.glm_ql_ftest(fit, contrast=contrast)
    tt = edgepython.top_tags(qlf, n=np.inf)["table"]
    tt.index = pd.Index(genes[tt.index.to_numpy()], name="gene_ids")
    return tt
