from importlib.util import find_spec

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse

from ..utils import aggregate_and_filter

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
    layer: str | None = None,
    replicas_per_group: int = 5,
    min_cells_per_group: int = 30,
    bootstrap_sampling: bool = False,
    use_cells: dict[str, list[str]] | None = None,
    aggregate: bool = True,
    edger_preprocessing: bool = True,
    verbosity: int = 0,
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
    layer : str or None, optional
        Layer containing raw counts. Uses ``adata.X`` if None. Default is None.
    replicas_per_group : int, optional
        Number of pseudobulk replicas to create per group. Default is 5.
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

    Notes
    -----
    The design is ``~ 0 + group + replica``. When the aggregated data has a
    single replica per group, the ``replica`` term is dropped because it
    would not be estimable.
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
        aggr_adata = aggregate_and_filter(
            adata,
            group_key,
            cell_identity_key,
            layer,
            replicas_per_group,
            min_cells_per_group,
            bootstrap_sampling,
            use_cells,
        )
    else:
        aggr_adata = adata.copy()

    if condition_group is None:
        condition_group_list = aggr_adata.obs[group_key].unique()
    elif isinstance(condition_group, str):
        condition_group_list = [condition_group]
    else:
        condition_group_list = condition_group

    if cell_identity_key is not None:
        cids = aggr_adata.obs[cell_identity_key].unique()
    else:
        cids = [""]

    def level(name: str, cid: str) -> str:
        return f"{name}_{cid}" if cid else name

    tt_dict = {}
    for condition_group in condition_group_list:
        if reference_group is not None and condition_group == reference_group:
            continue

        if verbosity > 0:
            print(f"Fitting model for {condition_group}...")

        if reference_group is not None:
            gk = group_key
        else:
            gk = f"{group_key}_{condition_group}"

        groups = aggr_adata.obs[gk].astype(str)
        if cell_identity_key is not None:
            groups = groups + "_" + aggr_adata.obs[cell_identity_key].astype(str)

        try:
            fit, design, genes = _fit_voom_model(
                aggr_adata, groups, edger_preprocessing, verbosity
            )
        except (ValueError, np.linalg.LinAlgError) as e:
            print("Error fitting model for", condition_group)
            print("Error:", e)
            print("Skipping...", flush=True)
            continue

        for cid in cids:
            if reference_group is None:
                cnd = level(condition_group, cid)
                ref = level(f"not{condition_group}", cid)
                contrast_key = f"{condition_group}_vs_rest"
            else:
                cnd = level(condition_group, cid)
                ref = level(reference_group, cid)
                contrast_key = f"{condition_group}_vs_{reference_group}"

            if cid:
                contrast_key = f"{cell_identity_key}:{cid}|{contrast_key}"

            if cnd not in design.columns or ref not in design.columns:
                if verbosity > 0:
                    print(f"Skipping {contrast_key}: group not present after filtering")
                continue

            if verbosity > 0:
                print(f"Computing contrast: {contrast_key}... ({cnd} - {ref})")

            try:
                tt = _top_table(fit, design, cnd, ref, genes)
            except ValueError as e:  # e.g. no residual degrees of freedom
                print("Error computing contrast", contrast_key)
                print("Error:", e)
                print("Skipping...", flush=True)
                continue

            tt["pct_expr_cnd"] = aggr_adata.var[f"pct_expr_{cnd}"].loc[tt.index]
            tt["pct_expr_ref"] = aggr_adata.var[f"pct_expr_{ref}"].loc[tt.index]
            tt["num_expr_cnd"] = aggr_adata.var[f"num_expr_{cnd}"].loc[tt.index]
            tt["num_expr_ref"] = aggr_adata.var[f"num_expr_{ref}"].loc[tt.index]
            tt["tot_expr_cnd"] = aggr_adata.var[f"tot_expr_{cnd}"].loc[tt.index]
            tt["tot_expr_ref"] = aggr_adata.var[f"tot_expr_{ref}"].loc[tt.index]
            tt["mean_cnd"] = tt["tot_expr_cnd"] / tt["num_expr_cnd"]
            tt["mean_ref"] = tt["tot_expr_ref"] / tt["num_expr_ref"]
            tt_dict[contrast_key] = tt

    return tt_dict


def _fit_voom_model(
    aggr_adata: AnnData,
    groups: pd.Series,
    edger_preprocessing: bool,
    verbosity: int,
):
    import pylimma

    design = pd.get_dummies(groups).astype(float)
    if "replica" in aggr_adata.obs and aggr_adata.obs["replica"].nunique() > 1:
        replica = aggr_adata.obs["replica"].astype(str)
        # treatment coding, as R's model.matrix(~ 0 + group + replica)
        replica_dummies = pd.get_dummies(replica, prefix="replica", drop_first=True)
        design = design.join(replica_dummies.astype(float))
    elif verbosity > 0:
        print("Single replicate detected — dropping replica term from design.")

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

    return fit, design, genes


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
