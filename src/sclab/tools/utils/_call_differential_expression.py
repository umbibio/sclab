import numpy as np
from pandas import DataFrame, Series
from scipy.stats import median_abs_deviation


def _robust_z(x: Series):
    # DOI: 10.1080/01621459.1993.10476408
    mad: float = median_abs_deviation(x)
    Sr = 1.4826 * mad
    Zr = (x - x.median()) / Sr

    return Zr


def call_differential_expression(
    table: DataFrame,
    pvalue_col: str,
    logfc_col: str,
    pct_col_prefix: str = "pct_",
    max_pval: float = 0.05,
    min_robust_z_level: float = 2.5,
    min_pct: float = 0.05,
    contrast_key: str | None = None,
    copy: bool = False,
):
    """Call differentially expressed genes from robust Z-scores of log-fold-change.

    Flags each row in `table` as up- or down-regulated by combining a p-value
    cutoff, a robust Z-score threshold on `logfc_col` (median-absolute-deviation
    based; DOI: 10.1080/01621459.1993.10476408), and a minimum expression
    percentage across any `pct_*`-prefixed column.

    Parameters
    ----------
    table : DataFrame
        Differential expression results, one row per gene/feature.
    pvalue_col : str
        Column in `table` with (adjusted) p-values.
    logfc_col : str
        Column in `table` with log-fold-changes.
    pct_col_prefix : str, optional
        Prefix of columns giving percent-expressed values; the max across all
        matching columns is used as the expression filter. Defaults to "pct_".
    max_pval : float, optional
        Maximum p-value for a gene to be called DE. Defaults to 0.05.
    min_robust_z_level : float, optional
        Minimum absolute robust Z-score of `logfc_col` for a gene to be called
        DE. Defaults to 2.5.
    min_pct : float, optional
        Minimum percent-expressed value for a gene to be called DE. Defaults
        to 0.05.
    contrast_key : str or None, optional
        If provided, robust Z-scores are computed independently within each
        `table.groupby(contrast_key)` group rather than across the whole
        table. Defaults to None.
    copy : bool, optional
        If True, operate on and return a copy of `table` instead of modifying
        it in place. Defaults to False.

    Returns
    -------
    DataFrame or None
        `table` with two new columns: `robust_Z` (the computed robust
        Z-score) and `DE` (1 up-regulated, -1 down-regulated, 0 not
        significant). Returned only if `copy=True`; otherwise `table` is
        modified in place and `None` is returned.
    """
    if copy:
        table = table.copy()

    pct_expr_high = table.filter(regex=f"^{pct_col_prefix}").max(axis=1)

    if contrast_key is not None:
        table["robust_Z"] = (
            table.groupby(contrast_key)[logfc_col].apply(_robust_z).droplevel(0)
        )
    else:
        table["robust_Z"] = _robust_z(table[logfc_col])

    mask = table[pvalue_col] < max_pval
    mask &= table["robust_Z"].abs() > min_robust_z_level
    mask &= pct_expr_high > min_pct

    table["DE"] = 0
    table.loc[mask, "DE"] = table["robust_Z"].map(np.sign).astype(int)

    if copy:
        return table
