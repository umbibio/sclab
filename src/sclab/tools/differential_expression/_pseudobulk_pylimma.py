from importlib.util import find_spec

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse

from ..._citations import record_citations
from ..utils import aggregate_and_filter
from ._pseudobulk_helpers import pseudobulk_contrasts, validate_counts_layer

_TOP_TABLE_COLUMNS = {
    "log_fc": "logFC",
    "ave_expr": "AveExpr",
    "t": "t",
    "p_value": "P.Value",
    "adj_p_value": "adj.P.Val",
    "b": "B",
}


def pylimma_is_available() -> bool:
    return find_spec("pylimma") is not None


def edgepython_is_available() -> bool:
    return find_spec("edgepython") is not None


def pseudobulk_pylimma(
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
    edger_preprocessing: bool = True,
    verbosity: int = 0,
    check_counts: bool = True,
    sample_key: str | None = None,
    random_state: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Pseudobulk differential expression analysis using limma-voom in Python.

    Pure-Python counterpart of :func:`pseudobulk_limma` (no R required).
    Aggregates single cells into pseudobulk samples, fits a voom-weighted
    linear model with ``pylimma`` and computes top-table statistics for each
    requested contrast.

    Requires the Python package ``pylimma``. With ``edger_preprocessing=True``
    (default) it also requires ``edgepython``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    group_key : str
        Column in ``adata.obs`` defining the experimental groups.
    condition_group : str or list of str or None, optional
        Group(s) to test against ``reference_group``. If None, each group
        is contrasted with the corresponding reference. Default is None.
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
    edger_preprocessing : bool, optional
        If True, filter genes with edgeR's ``filterByExpr`` and normalize
        library sizes with TMM (via ``edgepython``), as in
        :func:`pseudobulk_limma`. If False, only genes with zero total
        counts are removed and raw library sizes are used, so ``edgepython``
        is not needed. Default is True.
    verbosity : int, optional
        Verbosity level (0 = silent). Default is 0.
    check_counts : bool, optional
        If True, raise a ``ValueError`` when the selected layer does not look
        like raw counts (non-negative integers). Set to False for
        non-integer corrected counts. Default is True.
    sample_key : str or None, optional
        Column in ``adata.obs`` with the biological samples. If given, cells
        are aggregated per sample within each group (and identity) and these
        samples are the replicates; ``replicas_per_group`` and
        ``bootstrap_sampling`` are ignored. Each side of every contrast needs
        at least 2 samples, otherwise a ``ValueError`` is raised. A sample
        with cells in several groups gives one pseudobulk sample per group and
        enters the design as a blocking (paired) term; samples nested in
        groups do not. Default is None (random pseudo-replicas).
    random_state : int or None, optional
        Seed for the random pseudo-replica assignment. None uses the global
        random state. Default is None.

    Returns
    -------
    dict of str to pd.DataFrame
        One DataFrame per contrast (keyed by contrast label), with the same
        columns as :func:`pseudobulk_limma`:

        - ``logFC`` — log2 fold change.
        - ``AveExpr`` — average log2 expression.
        - ``t`` — moderated t-statistic.
        - ``P.Value`` — raw p-value.
        - ``adj.P.Val`` — Benjamini-Hochberg adjusted p-value.
        - ``B`` — log-odds of differential expression.
        - ``pct_expr_cnd`` / ``pct_expr_ref`` — fraction of expressing
          cells in condition/reference group.
        - ``logCPM_cnd`` / ``logCPM_ref`` — mean log2-CPM (voom's
          ``log2((count + 0.5) / (library size + 1) * 1e6)``) of the
          condition/reference pseudobulk samples.

    Notes
    -----
    The design is ``~ 0 + group + replica``. The ``replica`` term is dropped
    when it is not estimable: a single replica per group, or replicas
    (samples) nested in groups.

    References
    ----------
    Ritchie et al. (2015). limma powers differential expression analyses
    for RNA-sequencing and microarray studies. *Nucleic Acids Research*,
    43, e47.
    https://doi.org/10.1093/nar/gkv007

    Law et al. (2014). voom: precision weights unlock linear model
    analysis tools for RNA-seq read counts. *Genome Biology*, 15, R29.
    https://doi.org/10.1186/gb-2014-15-2-r29

    Mulvey (2026). pylimma: a faithful, AnnData-native Python port of R
    limma for differential expression analysis. *bioRxiv*.
    https://doi.org/10.64898/2026.07.06.736732

    Chen et al. (2025). edgeR v4: powerful differential analysis of
    sequencing data with expanded functionality and improved support for
    small counts and larger datasets. *Nucleic Acids Research*, 53,
    gkaf018.
    https://doi.org/10.1093/nar/gkaf018

    Robinson & Oshlack (2010). A scaling normalization method for
    differential expression analysis of RNA-seq data. *Genome Biology*,
    11, R25.
    https://doi.org/10.1186/gb-2010-11-3-r25

    Pachter (2026). Differential analysis of genomics count data with
    edge*. *bioRxiv*.
    https://doi.org/10.64898/2026.02.16.706223
    """
    if not pylimma_is_available():
        raise ImportError(
            "pseudobulk_pylimma requires pylimma. Install with `pip install sclab[de]`."
        )
    if edger_preprocessing and not edgepython_is_available():
        raise ImportError(
            "edger_preprocessing=True requires edgepython. Install with "
            "`pip install sclab[de]` or pass edger_preprocessing=False."
        )

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

    results = pseudobulk_contrasts(
        aggr_adata,
        group_key,
        condition_group,
        reference_group,
        cell_identity_key,
        fit=lambda aggr, design: _fit_voom_model(
            aggr, design, edger_preprocessing, verbosity
        ),
        test=_top_table,
        sample_key=sample_key if aggregate else None,
        verbosity=verbosity,
    )
    edger_keys = ("edger", "tmm", "edgepython") if edger_preprocessing else ()
    record_citations(adata, "limma", "voom", "pylimma", *edger_keys)
    return results


def _fit_voom_model(
    aggr_adata: AnnData,
    design: pd.DataFrame,
    edger_preprocessing: bool,
    verbosity: int,
):
    import pylimma

    X = aggr_adata.X.toarray() if issparse(aggr_adata.X) else aggr_adata.X
    counts = np.asarray(X, dtype=float).T  # genes x samples
    genes = aggr_adata.var_names.to_numpy()

    if verbosity > 1:
        print("Dimensions before subsetting:", counts.shape)

    if edger_preprocessing:
        import edgepython

        keep = np.asarray(edgepython.filter_by_expr(counts, design=design.values))
    else:
        keep = counts.sum(axis=1) > 0
    counts, genes = counts[keep], genes[keep]

    if verbosity > 1:
        print("Dimensions after subsetting:", counts.shape)

    lib_size = counts.sum(axis=0)
    if edger_preprocessing:
        lib_size = lib_size * edgepython.calc_norm_factors(counts)

    # fixed lowess span, as R limma's voom default; matches pseudobulk_limma
    v = pylimma.voom(counts, design.values, lib_size=lib_size, adaptive_span=False)
    fit = pylimma.lm_fit(v, design.values)
    ne = pylimma.non_estimable(design.values, list(design.columns))
    if ne is not None and verbosity > 0:
        print("Non-estimable:", ne)

    return fit, genes


def _top_table(fit, design: pd.DataFrame, cnd: str, ref: str, genes: np.ndarray):
    import pylimma

    contrast = np.zeros((design.shape[1], 1))
    contrast[design.columns.get_loc(cnd), 0] = 1.0
    contrast[design.columns.get_loc(ref), 0] = -1.0

    fit2 = pylimma.e_bayes(pylimma.contrasts_fit(fit, contrasts=contrast))
    tt = pylimma.top_table(
        fit2, coef=0, number=np.inf, genelist=pd.DataFrame({"gene_ids": genes})
    )
    return tt.set_index("gene_ids").rename(columns=_TOP_TABLE_COLUMNS)
