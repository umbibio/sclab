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
