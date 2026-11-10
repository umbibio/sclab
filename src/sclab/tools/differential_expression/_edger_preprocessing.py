# Ported from edgePython 0.5.1 (https://github.com/pachterlab/edgePython, by
# Lior Pachter; GPL-3.0-or-later): edgepython/filtering.py (filter_by_expr) and
# normalization.py (calc_norm_factors, TMM), themselves ports of edgeR 5.0's
# filterByExpr() and calcNormFactors() (Bioconductor, GPL >= 2). Reduced by
# sclab to the defaults sclab uses (design matrix, TMM, no DGEList).
"""edgeR's ``filterByExpr`` and TMM normalisation, for limma-voom without edgePython.

:func:`pseudobulk_pylimma` uses edgePython's own functions when it is
installed, and these otherwise; they give the same results.

References: filterByExpr, Chen, Lun & Smyth (2016) F1000Research 5:1438,
https://doi.org/10.12688/f1000research.8987.2; TMM, Robinson & Oshlack (2010)
Genome Biology 11:R25, https://doi.org/10.1186/gb-2010-11-3-r25; edgeR, Chen
et al. (2025) NAR 53:gkaf018, https://doi.org/10.1093/nar/gkaf018.
"""

import numpy as np
from scipy.stats import rankdata

from ._edgepython_kernels import _hat_values


def filter_by_expr(
    counts,
    design=None,
    min_count=10,
    min_total_count=15,
    large_n=10,
    min_prop=0.7,
    max_n_samples=40,
):
    """Genes (rows of ``counts``) with enough reads to keep (edgeR's filterByExpr).

    A gene is kept with at least ``min_total_count`` reads and a CPM of at least
    ``min_count`` / (median library size) * 1e6 in at least n samples: the
    number of samples, or 1 / the largest leverage of ``design`` (with an
    intercept, as R's ``hat()``), reduced above ``large_n`` by ``min_prop`` and
    capped at ``max_n_samples``.
    """
    counts = np.asarray(counts, dtype=np.float64)
    lib_size = counts.sum(axis=0)
    if design is None:
        n = counts.shape[1]
    else:
        design = np.asarray(design, dtype=np.float64)
        x = np.column_stack([np.ones(design.shape[0]), design])
        n = 1.0 / np.max(_hat_values(x, np.ones(design.shape[0])))
    if n > large_n:
        n = large_n + (n - large_n) * min_prop
    n = min(n, max_n_samples)
    cutoff = min_count / np.median(lib_size) * 1e6
    keep_cpm = np.sum(counts * 1e6 / lib_size >= cutoff, axis=1) >= n - 1e-14
    return keep_cpm & (counts.sum(axis=1) >= min_total_count - 1e-14)


def _tmm_factor(obs, ref, n_obs, n_ref, logratio_trim=0.3, sum_trim=0.05):
    """TMM factor of library ``obs`` against ``ref`` (edgeR's .calcFactorTMM)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        # one quotient, as edgeR: log2(a) - log2(b) ranks ties differently
        log_r = np.log2((obs / n_obs) / (ref / n_ref))
        abs_e = (np.log2(obs / n_obs) + np.log2(ref / n_ref)) / 2
        v = (n_obs - obs) / n_obs / obs + (n_ref - ref) / n_ref / ref
    fin = np.isfinite(log_r) & np.isfinite(abs_e)
    log_r, abs_e, v = log_r[fin], abs_e[fin], v[fin]
    if log_r.size == 0 or np.max(np.abs(log_r)) < 1e-6:
        return 1.0
    m = log_r.size
    lo_l = int(m * logratio_trim) + 1
    lo_s = int(m * sum_trim) + 1
    rank_r, rank_e = rankdata(log_r), rankdata(abs_e)
    keep = (rank_r >= lo_l) & (rank_r <= m + 1 - lo_l)
    keep &= (rank_e >= lo_s) & (rank_e <= m + 1 - lo_s)
    denom = np.sum(1 / v[keep])
    if denom > 0 and np.isfinite(denom):
        f = np.sum(log_r[keep] / v[keep]) / denom
    else:
        f = np.nanmean(log_r[keep])
    return 2 ** (0.0 if np.isnan(f) else f)


def calc_norm_factors(counts):
    """TMM normalisation factors of the columns of ``counts`` (edgeR's normLibSizes).

    The reference library is the one whose upper quartile is closest to the
    mean upper quartile; the factors are scaled to multiply to one.
    """
    x = np.asarray(counts, dtype=np.float64)
    lib_size = x.sum(axis=0)
    x = x[(x > 0).any(axis=1)]
    if x.shape[0] == 0 or x.shape[1] == 1:
        return np.ones(x.shape[1])
    f75 = np.quantile(x, 0.75, axis=0) / lib_size
    if np.median(f75) < 1e-20:
        ref = np.argmax(np.sqrt(x).sum(axis=0))
    else:
        ref = np.argmin(np.abs(f75 - f75.mean()))
    f = np.array(
        [
            _tmm_factor(x[:, i], x[:, ref], lib_size[i], lib_size[ref])
            for i in range(x.shape[1])
        ]
    )
    return f / np.exp(np.mean(np.log(f)))
