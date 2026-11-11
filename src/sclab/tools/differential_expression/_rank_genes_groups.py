"""numpy/scipy/scikit-learn port of :func:`scanpy.tl.rank_genes_groups`, for
environments without scanpy (e.g. Pyodide).

Ported from scanpy 1.12.5 (https://github.com/scverse/scanpy, BSD-3-Clause,
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab; see
THIRD_PARTY_NOTICES.md), file ``src/scanpy/tools/_rank_genes_groups.py``
(``rank_genes_groups``, ``_RankGenes``, ``_select_top_n``) and
``src/scanpy/get/_aggregated.py``, ``_kernels.py`` (the group means and
variances scanpy computes with ``scanpy.get.aggregate``). The
Benjamini-Hochberg adjusted p-values are computed with the same arithmetic as
``statsmodels.stats.multitest.multipletests(method="fdr_bh")``, which scanpy
calls.

Methods ``t-test``, ``t-test_overestim_var`` (Welch's t-test from per-group
means and variances, :func:`scipy.stats.ttest_ind_from_stats`), ``wilcoxon``
(Wilcoxon rank-sum / Mann-Whitney U z-score, normal approximation, optional
tie correction) and ``logreg`` (:class:`sklearn.linear_model.LogisticRegression`
coefficients); ``reference="rest"`` or a group; ``corr_method``
``"benjamini-hochberg"`` or ``"bonferroni"``; ``n_genes``, ``rankby_abs``.
Not ported: ``pts``, ``mask_var``, ``copy``.

Sparse input stays sparse. The numba kernels are replaced by numpy code with
the same arithmetic: Welford means and variances over the stored values of
each row (a Python loop over rows, vectorized over genes), merged with the
zeros; Wilcoxon ranks from sorting the stored values of a block of genes at a
time, every zero of a gene sharing one tied rank (scanpy densifies blocks of
genes and ranks every value). The results are bit-identical to scanpy's for
sparse and dense data (see
``tests/sclab/tools/differential_expression/test_rank_genes_groups.py``).

References: Wolf, F. A., Angerer, P., & Theis, F. J. (2018). SCANPY:
large-scale single-cell gene expression data analysis. Genome Biology, 19, 15.
https://doi.org/10.1186/s13059-017-1382-0 -- and, for the methods, Welch
(1947) (t-test), Wilcoxon (1945) and Mann & Whitney (1947) (rank-sum test),
Ntranos et al. (2019) (logreg), Benjamini & Hochberg (1995) (FDR); see
:mod:`sclab._citations` and :func:`citation_keys`.
"""

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse, stats

METHODS = ("logreg", "t-test", "wilcoxon", "t-test_overestim_var")
_NNZ_BLOCK = 1_000_000  # stored values ranked at a time: bounds the sort buffers


def citation_keys(method: str, corr_method: str = "benjamini-hochberg") -> tuple:
    """The :mod:`sclab._citations` keys of ``method`` (besides scanpy's)."""
    if method == "logreg":
        return ("ntranos2019", "scikit-learn")
    test = ("wilcoxon1945", "mann1947") if method == "wilcoxon" else ("welch1947",)
    return test + (("benjamini1995",) if corr_method == "benjamini-hochberg" else ())


def rank_genes_groups(
    adata: AnnData,
    groupby: str,
    *,
    use_raw: bool | None = None,
    groups="all",
    reference: str = "rest",
    n_genes: int | None = None,
    rankby_abs: bool = False,
    key_added: str | None = None,
    method: str | None = None,
    corr_method: str = "benjamini-hochberg",
    tie_correct: bool = False,
    layer: str | None = None,
    **kwds,
) -> None:
    """``scanpy.tl.rank_genes_groups`` with the same arguments and output
    (``adata.uns[key_added]``: ``params`` and the ``names``, ``scores``,
    ``pvals``, ``pvals_adj``, ``logfoldchanges`` record arrays), so
    :func:`scanpy.get.rank_genes_groups_df` reads it. ``kwds`` go to
    :class:`~sklearn.linear_model.LogisticRegression` (``method="logreg"``).
    """
    method = "t-test" if method is None else method
    if method not in METHODS:
        raise ValueError(f"Method must be one of {set(METHODS)}.")
    if corr_method not in ("benjamini-hochberg", "bonferroni"):
        raise ValueError(
            "Correction method must be one of {'benjamini-hochberg', 'bonferroni'}."
        )
    if kwds and method != "logreg":  # scanpy ignores them; pts, mask_var: not ported
        raise TypeError(f"unsupported arguments for method={method!r}: {list(kwds)}")
    if use_raw is None:
        use_raw = adata.raw is not None
    elif use_raw and adata.raw is None:
        raise ValueError("Received `use_raw=True`, but `adata.raw` is empty.")

    labels = adata.obs[groupby].astype("category")  # scanpy sanitizes adata.obs
    categories = labels.cat.categories
    if groups == "all":
        groups_order = list(categories)
    elif isinstance(groups, str | int):
        raise ValueError("Specify a sequence of groups")
    else:
        groups_order = [str(g) if isinstance(g, int) else g for g in groups]
        if reference != "rest" and reference not in groups_order:
            groups_order.append(reference)
    if reference != "rest" and reference not in categories:
        raise ValueError(
            f"reference = {reference} needs to be one of groupby = {categories.tolist()}."
        )
    counts = labels.value_counts()
    if singlets := [g for g in groups_order if counts[g] < 2]:
        raise ValueError(
            f"Could not calculate statistics for groups {', '.join(map(str, singlets))} "
            "since they only contain one sample."
        )

    if layer is not None:
        if use_raw:
            raise ValueError("Cannot specify `layer` and have `use_raw=True`.")
        X, var_names = adata.layers[layer], adata.var_names
    elif use_raw:
        X, var_names = adata.raw.X, adata.raw.var_names
    else:
        X, var_names = adata.X, adata.var_names
    masks = np.stack([(labels == g).to_numpy() for g in groups_order])
    iref = None if reference == "rest" else groups_order.index(reference)

    if key_added is None:
        key_added = "rank_genes_groups"
    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = dict(
        groupby=groupby,
        reference=reference,
        method=method,
        use_raw=use_raw,
        layer=layer,
        corr_method=corr_method,
    )

    if (base := adata.uns.get("log1p", {}).get("base")) is not None:
        expm1 = lambda x: np.expm1(x * np.log(base))  # noqa: E731
    else:
        expm1 = np.expm1

    n_vars = X.shape[1]
    n_top = n_vars if n_genes is None or n_genes > n_vars else n_genes
    fields: dict[str, dict] = {}
    for gi, scores, pvals, means in _tests(
        X, masks, iref, method, tie_correct, labels, groups_order, kwds
    ):
        group = str(groups_order[gi])
        top = _select_top_n(np.abs(scores) if rankby_abs else scores, n_top)
        out = {"names": var_names[top], "scores": scores[top]}
        if pvals is not None:
            out["pvals"] = pvals[top]
            if corr_method == "benjamini-hochberg":
                pvals[np.isnan(pvals)] = 1
                out["pvals_adj"] = _fdr_bh(pvals)[top]
            else:
                out["pvals_adj"] = np.minimum(pvals * n_vars, 1.0)[top]
        if means is not None:  # (mean of the group, mean of its reference)
            fc = (expm1(means[0]) + 1e-9) / (expm1(means[1]) + 1e-9)
            out["logfoldchanges"] = np.log2(fc[top])
        for field, values in out.items():
            fields.setdefault(field, {})[group] = values

    dtypes = {
        "names": "O",
        "scores": "float32",
        "logfoldchanges": "float32",
        "pvals": "float64",
        "pvals_adj": "float64",
    }
    for field, columns in fields.items():
        adata.uns[key_added][field] = pd.DataFrame(columns).to_records(
            index=False, column_dtypes=dtypes[field]
        )


def _tests(X, masks, iref, method, tie_correct, labels, groups_order, kwds):
    """Yield ``(group index, scores, pvals or None, (means, means_ref) or None)``
    per tested group, as ``_RankGenes.t_test`` / ``wilcoxon`` / ``logreg``."""
    n_cells = X.shape[0]
    if method == "logreg":
        from sklearn.linear_model import LogisticRegression

        if len(groups_order) == 1:
            raise ValueError("Cannot perform logistic regression on a single cluster.")
        in_groups = labels.isin(groups_order).to_numpy()
        codes = labels.cat.codes.to_numpy()[in_groups]
        coef = LogisticRegression(**kwds).fit(X[in_groups], codes).coef_
        existing = np.unique(codes)
        for gi, group in enumerate(groups_order):
            if len(groups_order) <= 2:  # binary: one coefficient row
                yield gi, coef[0], None, None
                return
            row = np.argmax(existing == labels.cat.categories.get_loc(group))
            yield gi, coef[row], None, None
        return

    t_test = method != "wilcoxon"
    k = len(masks)
    if iref is None:  # every cell is in one tested group, or the remainder (code k)
        codes = np.full(n_cells, k)
        for gi, mask in enumerate(masks):
            codes[mask] = gi
        counts = np.bincount(codes, minlength=k + 1)
        mean, var = _group_mean_var(X, codes, k + 1, t_test)
        # scanpy's _derive_rest_stats: exact for the means, Chan for the variances
        n_rest = n_cells - counts[:k]
        total = (counts[:, None] * mean).sum(axis=0)
        mean_rest = (total - counts[:k, None] * mean[:k]) / n_rest[:, None]
        var_rest = _vars_rest(counts, mean, var, k) if t_test else None
        if not t_test:  # all cells ranked together, once
            rank_sums, tc_all = _rank_sums(X, codes, k + 1, tie_correct)
    else:
        codes = np.where(masks.any(axis=0), masks.argmax(axis=0), -1)
        counts = masks.sum(axis=1)
        mean, var = _group_mean_var(X, codes, k, t_test)

    for gi in range(k):
        if gi == iref:
            continue
        n_group = counts[gi]
        if iref is None:
            m_rest, n_other = mean_rest[gi], n_rest[gi]
            v_rest = var_rest[gi] if t_test else None
        else:
            m_rest, n_other = mean[iref], counts[iref]
            v_rest = var[iref] if t_test else None

        if t_test:
            # "t-test_overestim_var": scanpy's hack overestimating the variance
            # of small groups
            nobs2 = n_other if method == "t-test" else n_group
            with np.errstate(invalid="ignore"):
                scores, pvals = stats.ttest_ind_from_stats(
                    mean1=mean[gi],
                    std1=np.sqrt(var[gi]),
                    nobs1=n_group,
                    mean2=m_rest,
                    std2=np.sqrt(v_rest),
                    nobs2=nobs2,
                    equal_var=False,  # Welch's
                )
            scores[np.isnan(scores)] = 0
            pvals[np.isnan(pvals)] = 1
        else:
            n = n_group + n_other
            if iref is None:
                rank_sum, tc = rank_sums[gi], tc_all
            else:  # the group's and the reference's cells ranked together
                both = masks[gi] | masks[iref]
                sums, tc = _rank_sums(X[both], masks[gi][both], 2, tie_correct)
                rank_sum = sums[1]
            std_dev = np.sqrt(tc * n_group * (n - n_group) * (n + 1) / 12.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                scores = (rank_sum - n_group * ((n + 1) / 2.0)) / std_dev
            scores[np.isnan(scores)] = 0
            pvals = 2 * stats.distributions.norm.sf(np.abs(scores))
        yield gi, scores, pvals, (mean[gi], m_rest)


def _group_mean_var(X, codes, n_codes: int, need_var: bool):
    """Per group (row code; rows coded -1 are skipped) and gene: mean and
    variance (ddof=1; None if not ``need_var``) in float64, as
    ``scanpy.get.aggregate(func=["mean", "var"])``.

    The means alone are sums over counts, accumulated row after row as
    scanpy's ``agg_sum_csr``; with the variances, both come from Welford's
    online algorithm over the rows in order, for sparse X over the stored
    values only, then merged with the gene's zeros (scanpy's ``mean_var_csr``
    / ``mean_var_dense`` kernels, vectorized over genes).
    """
    n_cells, m = X.shape
    codes = np.asarray(codes, dtype=np.intp)
    counts = np.bincount(codes[codes >= 0], minlength=n_codes)
    if issparse := sparse.issparse(X):
        X = sparse.csr_matrix(X)
        if (X.data == 0).any():  # scanpy eliminates explicit zeros first
            X = X.copy()
            X.eliminate_zeros()
    if not need_var:
        if issparse:
            sums = np.zeros(n_codes * m)
            for top, bottom in _blocks(X.indptr):  # rows; add.at adds in order
                start, end = X.indptr[top], X.indptr[bottom]
                row_codes = np.repeat(
                    codes[top:bottom], np.diff(X.indptr[top : bottom + 1])
                )
                keep = row_codes >= 0
                flat = row_codes[keep] * m + X.indices[start:end][keep]
                np.add.at(sums, flat, X.data[start:end][keep].astype(np.float64))
            sums = sums.reshape(n_codes, m)
        else:
            sums = np.stack(
                [X[codes == c].sum(axis=0, dtype=np.float64) for c in range(n_codes)]
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            return sums / counts[:, None], None

    mean = np.zeros((n_codes, m))
    m2 = np.zeros((n_codes, m))
    if issparse:
        nnz = np.zeros((n_codes, m), dtype=np.int64)
        indptr, indices, data = X.indptr, X.indices, X.data
        for r in range(n_cells):
            c, start, end = codes[r], indptr[r], indptr[r + 1]
            if start == end or c < 0:
                continue
            cols, value = indices[start:end], data[start:end].astype(np.float64)
            n = nnz[c, cols] + 1
            nnz[c, cols] = n
            mu = mean[c, cols]
            delta = value - mu
            mu = mu + delta / n
            mean[c, cols] = mu
            m2[c, cols] += delta * (value - mu)
        zeros = counts[:, None] - nnz
        merge = (zeros > 0) & (nnz > 0)  # mean_A, M2_A of the nonzeros + the zeros
        mean_a = mean
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(merge, mean_a * nnz / counts[:, None], mean_a)
            m2 = np.where(
                merge, m2 + mean_a * mean_a * nnz * zeros / counts[:, None], m2
            )
    else:
        seen = np.zeros(n_codes, dtype=np.int64)
        for r in range(n_cells):
            c = codes[r]
            if c < 0:
                continue
            value = np.asarray(X[r], dtype=np.float64).ravel()
            seen[c] += 1
            delta = value - mean[c]
            mean[c] += delta / seen[c]
            m2[c] += delta * (value - mean[c])
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.where(counts[:, None] > 0, m2 / counts[:, None], 0.0)
        denom = np.where(counts > 1, counts - 1, np.nan)  # ddof=1
        var *= (counts / denom)[:, None]
    return mean, var


def _chan(n_a, mean_a, m2_a, n_b, mean_b, m2_b):
    """Combine two ``(count, mean, M2)`` groups (scanpy's ``_chan_combine``)."""
    if n_a == 0.0:
        return n_b, mean_b, m2_b
    if n_b == 0.0:
        return n_a, mean_a, m2_a
    n = n_a + n_b
    delta = mean_b - mean_a
    return (
        n,
        (n_a * mean_a + n_b * mean_b) / n,
        m2_a + m2_b + delta * delta * n_a * n_b / n,
    )


def _vars_rest(counts, mean, var, k: int) -> np.ndarray:
    """Variance (ddof=1) of the cells outside each of the first ``k`` groups:
    the groups before it pooled with those after it (scanpy's ``_vars_rest``,
    Chan's cancellation-free combine, vectorized over genes)."""
    with np.errstate(invalid="ignore"):
        m2 = var * (counts[:, None] - 1)
    m2[counts <= 1] = 0.0
    n_groups, counts = len(counts), counts.astype(np.float64)
    up, down = [None] * n_groups, [None] * n_groups
    acc = (0.0, 0.0, 0.0)
    for i in range(n_groups):
        up[i] = acc = _chan(*acc, counts[i], mean[i], m2[i])
    acc = (0.0, 0.0, 0.0)
    for i in range(n_groups - 1, -1, -1):
        down[i] = acc = _chan(*acc, counts[i], mean[i], m2[i])
    out = np.zeros((k, mean.shape[1]))
    for g in range(k):
        after = down[g + 1]
        n_r, _, m2_r = _chan(*up[g - 1], *after) if g >= 1 else after
        out[g] = m2_r / (n_r - 1.0)
    return out


def _rank_sums(X, codes, n_codes: int, tie_correct: bool):
    """Per group (row code) and gene: the sum of the ranks of the group's rows
    among all rows of X (average ranks of ties, as :func:`scipy.stats.rankdata`);
    and per gene the tie correction factor (:func:`scipy.stats.tiecorrect`; 1
    if not ``tie_correct``).

    Ranks only the stored nonzeros, sorted per gene; the zeros of a gene are
    one tie block, between its negative and positive values.
    """
    n, m = X.shape
    codes = np.asarray(codes, dtype=np.intp)
    per_code = np.bincount(codes, minlength=n_codes)
    rank_sums, tc = np.zeros((n_codes, m)), np.ones(m)
    if sparse.issparse(X):
        X = sparse.csc_matrix(X)
        bounds = _blocks(X.indptr)
    else:
        step = max(_NNZ_BLOCK // max(n, 1), 1)
        bounds = [(i, min(i + step, m)) for i in range(0, m, step)]

    for left, right in bounds:
        block = sparse.csc_matrix(X[:, left:right])
        w = right - left
        col = np.repeat(np.arange(w), np.diff(block.indptr))
        val, row = block.data, block.indices
        if not (keep := val != 0).all():  # explicit zeros
            col, val, row = col[keep], val[keep], row[keep]
        order = np.lexsort((val, col))
        col, val, row = col[order], val[order], row[order]

        nnz = np.bincount(col, minlength=w)
        col_start = np.cumsum(nnz) - nnz  # first position of each gene
        new = np.ones(len(val), dtype=bool)  # starts a tie block
        new[1:] = (col[1:] != col[:-1]) | (val[1:] != val[:-1])
        starts = np.flatnonzero(new)
        lengths = np.diff(np.append(starts, len(val)))
        block_id = np.cumsum(new) - 1
        # average rank among the gene's nonzeros, shifted past its zeros if > 0
        zeros = n - nnz
        rank = (starts - col_start[col[starts]] + (lengths + 1) / 2)[block_id]
        rank += np.where(val > 0, zeros[col], 0)
        zero_rank = np.bincount(col, weights=val < 0, minlength=w) + (zeros + 1) / 2

        flat, size = codes[row] * w + col, n_codes * w
        nnz_code = np.bincount(flat, minlength=size).reshape(n_codes, w)
        sums = np.bincount(flat, weights=rank, minlength=size).reshape(n_codes, w)
        rank_sums[:, left:right] = sums + (per_code[:, None] - nnz_code) * zero_rank
        if tie_correct and n >= 2:
            ties = lengths.astype(np.float64)
            t = np.bincount(col[starts], weights=ties**3 - ties, minlength=w)
            z = zeros.astype(np.float64)
            tc[left:right] = 1.0 - (t + z**3 - z) / (float(n) ** 3 - n)
    return rank_sums, tc


def _blocks(indptr: np.ndarray) -> list[tuple[int, int]]:
    """Consecutive (first, end) rows (CSR) or columns (CSC) holding about
    _NNZ_BLOCK stored values each (at least one row or column)."""
    bounds, left, n = [], 0, len(indptr) - 1
    while left < n:
        right = np.searchsorted(indptr, indptr[left] + _NNZ_BLOCK, "right") - 1
        bounds.append((left, right := min(max(right, left + 1), n)))
        left = right
    return bounds


def _select_top_n(scores: np.ndarray, n_top: int) -> np.ndarray:
    # scanpy's: the same argpartition/argsort, so ties keep scanpy's order
    partition = np.argpartition(scores, -n_top)[-n_top:]
    return np.arange(scores.shape[0])[partition][np.argsort(scores[partition])[::-1]]


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, as
    ``statsmodels.stats.multitest.multipletests(pvals, method="fdr_bh")``."""
    order = np.argsort(pvals)
    p = pvals[order]
    adj = p / (np.arange(1, len(p) + 1) / float(len(p)))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj[adj > 1] = 1
    out = np.empty_like(adj)
    out[order] = adj
    return out
