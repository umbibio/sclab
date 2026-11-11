import warnings
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
from anndata import AnnData

from sclab.utils import layer_scaled, looks_like_raw_counts, unscaled_layer

from ..._citations import record_citations
from ...preprocess._scanpy_port import resolve_backend
from ._pseudobulk_edgepython import pseudobulk_edgepython
from ._pseudobulk_pylimma import pseudobulk_pylimma
from ._rank_genes_groups import citation_keys

SCANPY_METHODS = ("wilcoxon", "t-test", "logreg")

# method -> (backend, {standard column: backend column}, native extras)
PSEUDOBULK_METHODS = {
    "pseudobulk_limma": (
        pseudobulk_pylimma,
        {"score": "t", "pval": "P.Value", "padj": "adj.P.Val"},
        ["AveExpr", "t", "B"],
    ),
    "pseudobulk_edger": (
        pseudobulk_edgepython,
        {"score": "F", "pval": "PValue", "padj": "FDR"},
        ["logCPM", "F"],
    ),
}
PSEUDOBULK_COLUMNS = {
    "logFC": "logFC",
    "mean_cnd": "logCPM_cnd",
    "mean_ref": "logCPM_ref",
    "pct_cnd": "pct_expr_cnd",
    "pct_ref": "pct_expr_ref",
}

COLUMNS = [
    "contrast",
    "cell_identity",
    "gene",
    "logFC",
    "score",
    "pval",
    "padj",
    "mean_cnd",
    "mean_ref",
    "pct_cnd",
    "pct_ref",
]


def scaling_warning(adata: AnnData, method: str, layer: str) -> str | None:
    """Why a cell-level ``method`` on ``layer`` (``"X"``: ``adata.X``) gives
    misleading results because the data are scaled (see
    :func:`sclab.utils.layer_scaled`), or None."""
    if method not in SCANPY_METHODS or not (scaled := layer_scaled(adata, layer)):
        return None
    what = "adata.X" if layer == "X" else f"layer {layer!r}"
    lfc = "fold changes are not meaningful on scaled data."
    if method == "logreg":
        text = f"{what} is scaled: logreg results depend on the scaling, and {lfc}"
    elif isinstance(scaled, str):
        text = (
            f"{what} was scaled per group of {scaled!r}: groups of cells were "
            f"scaled differently, so scores and p-values are affected, and {lfc}"
        )
    else:
        text = f"{what} is scaled: {lfc}"
    alt = unscaled_layer(adata)
    if not layer_scaled(adata, alt):
        return f"{text} Use {'adata.X' if alt == 'X' else f'layer {alt!r}'} (unscaled)."
    return f"{text} Rerun Preprocess to keep the unscaled data."


def differential_expression(
    adata: AnnData,
    group_key: str,
    *,
    method: str = "wilcoxon",
    condition_group: str | list[str] | None = None,
    reference_group: str | None = None,
    cell_identity_key: str | None = None,
    layer: str | None = None,
    sample_key: str | None = None,
    key_added: str = "differential_expression",
    random_state: int = 0,
    backend: str = "auto",
    **method_kwargs,
) -> pd.DataFrame:
    """Differential expression between groups of cells, with one output format.

    Runs a cell-level test (scanpy) or a pseudobulk test (pure Python limma or
    edgeR) for each contrast ``condition vs reference``, and returns all
    results as one long table, also stored in ``adata.uns[key_added]``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    group_key : str
        Column in ``adata.obs`` with the groups to compare.
    method : str, optional
        Test to run. Default is ``"wilcoxon"``.

        - ``"wilcoxon"``, ``"t-test"``, ``"logreg"``: cell-level tests of
          :func:`scanpy.tl.rank_genes_groups` (methods ``"wilcoxon"``,
          ``"t-test"`` (Welch's t-test, not ``"t-test_overestim_var"``) and
          ``"logreg"``), run by scanpy or sclab's port of it (see
          ``backend``).
        - ``"pseudobulk_limma"``: limma-voom on pseudobulk samples,
          :func:`pseudobulk_pylimma`. Requires ``pylimma``.
        - ``"pseudobulk_edger"``: edgeR quasi-likelihood F-test on pseudobulk
          samples, :func:`pseudobulk_edgepython`. Requires ``edgepython``.
    condition_group : str or list of str or None, optional
        Group(s) to test. None tests every group. Default is None.
    reference_group : str or None, optional
        Group to compare against. None compares each condition group with
        the rest, i.e. all other cells (for pseudobulk methods the rest is
        pooled into one reference level). Default is None.
    cell_identity_key : str or None, optional
        Column in ``adata.obs`` with cell identities (e.g. cell types). If
        given, every contrast is run separately within each identity, on the
        cells of that identity only. Default is None.
    layer : str or None, optional
        Expression values to test; ``"X"`` means ``adata.X``. An explicit
        value is always used as given. None picks the method's default: for
        the scanpy methods the log-normalized, unscaled data
        (:func:`sclab.utils.unscaled_layer`: ``adata.X``, unless X is known to
        be scaled and :func:`sclab.preprocess.preprocess` kept the unscaled
        data in a layer, which is then used), for the pseudobulk methods
        ``"counts"`` (raw counts). The layer used is recorded in
        ``adata.uns[key_added]["params"]["layer"]``. The scanpy methods raise
        a ``ValueError`` when the data look like raw counts (non-negative
        integers); the pseudobulk methods raise one when they do not (pass
        ``check_counts=False`` to skip that check). The scanpy methods expect
        log-normalized, unscaled data: they warn when the layer used is known
        to be scaled, e.g. ``adata.X`` after preprocessing with
        ``scale=True`` when the unscaled layer is missing.
    sample_key : str or None, optional
        Pseudobulk methods only. Column in ``adata.obs`` with the biological
        samples: cells are summed per sample within each group (and
        identity), and these sums are the replicates. Each side of every
        contrast needs at least 2 samples, otherwise a ``ValueError`` is
        raised. A sample with cells in several groups yields one pseudobulk
        sample per group and enters the design as a blocking (paired) term;
        samples nested in groups do not. None (default) splits each group at
        random into ``replicas_per_group`` pseudo-replicas (default 3),
        seeded by ``random_state``.
    key_added : str, optional
        Key in ``adata.uns`` for the results. Default is
        ``"differential_expression"``.
    random_state : int, optional
        Seed for the pseudo-replica assignment (pseudobulk) and for
        ``logreg``. Default is 0.
    backend : {"auto", "native", "scanpy"}, optional
        Cell-level methods only. ``"scanpy"`` runs
        :func:`scanpy.tl.rank_genes_groups`, ``"native"`` sclab's port of it
        (numpy, scipy and scikit-learn; sparse data stay sparse), ``"auto"``
        (default) scanpy if it is installed. The results are the same.
    **method_kwargs
        Passed to the backend: :func:`scanpy.tl.rank_genes_groups` (e.g.
        ``tie_correct``, ``corr_method``; for ``logreg`` also
        :class:`sklearn.linear_model.LogisticRegression` arguments), or
        :func:`pseudobulk_pylimma` / :func:`pseudobulk_edgepython` (e.g.
        ``replicas_per_group``, ``min_cells_per_group``, ``use_cells``,
        ``check_counts``).

    Returns
    -------
    pd.DataFrame
        One row per gene per contrast (and per identity), with columns:

        - ``contrast`` — ``"{condition}_vs_{reference}"``, with
          ``reference`` the reference group or ``"rest"``, e.g.
          ``"B_vs_rest"`` or ``"B_vs_A"``.
        - ``cell_identity`` — the identity, only when ``cell_identity_key``
          is set.
        - ``gene`` — gene name (``adata.var_names``).
        - ``logFC`` — log2 fold change, condition over reference. scanpy
          methods: ``log2(expm1(mean_cnd) / expm1(mean_ref))`` (assumes
          ``log1p`` data; computed the same way for ``logreg``). Pseudobulk:
          the limma/edgeR model estimate.
        - ``score`` — test statistic: Wilcoxon z-score, t-test t, logreg
          coefficient, limma moderated t, edgeR quasi-likelihood F.
        - ``pval`` / ``padj`` — p-value and Benjamini-Hochberg adjusted
          p-value (within the contrast). NaN for ``logreg``, which gives no
          p-values.
        - ``mean_cnd`` / ``mean_ref`` — mean expression of the condition and
          reference cells. scanpy methods: mean of the tested values (e.g.
          log-normalized expression). Pseudobulk: mean log2-CPM of the
          side's pseudobulk samples,
          ``log2((count + 0.5) / (library size + 1) * 1e6)``.
        - ``pct_cnd`` / ``pct_ref`` — fraction of condition and reference
          cells with nonzero expression in the tested layer.

        followed by native columns of the method: ``AveExpr``, ``t``, ``B``
        (``pseudobulk_limma``); ``logCPM``, ``F`` (``pseudobulk_edger``).
        Rows keep the method's own order within each contrast (by score for
        scanpy, by p-value for pseudobulk). Pseudobulk methods only report
        genes kept by their expression filter, and skip contrasts whose
        groups have too few cells (``min_cells_per_group``).

    Notes
    -----
    Results are stored as ``adata.uns[key_added] = {"params": ..., "table":
    df}``, where ``params`` holds the resolved arguments (``method``,
    ``layer``, ``reference_group``, ...) and the package versions used. The
    table is the returned DataFrame and survives ``write_h5ad``.

    Cell-level tests treat cells as independent replicates, which inflates
    p-values; prefer a pseudobulk method with ``sample_key`` for comparisons
    between conditions.

    References
    ----------
    Wolf et al. (2018). SCANPY: large-scale single-cell gene expression
    data analysis. *Genome Biology*, 19, 15.
    https://doi.org/10.1186/s13059-017-1382-0

    Ntranos et al. (2019). A discriminative learning approach to
    differential expression analysis for single-cell RNA-seq. *Nature
    Methods*, 16, 163–166.
    https://doi.org/10.1038/s41592-018-0303-9

    Examples
    --------
    >>> import sclab
    >>> df = sclab.tools.de(adata, "leiden")  # each cluster vs the rest
    >>> df = sclab.tools.de(
    ...     adata,
    ...     "condition",
    ...     method="pseudobulk_edger",
    ...     reference_group="control",
    ...     cell_identity_key="cell_type",
    ...     sample_key="donor",
    ... )
    >>> adata.uns["differential_expression"]["table"]  # the same table
    """
    group_values = adata.obs[group_key]
    present = set(group_values.dropna())
    all_groups = [
        g for g in group_values.astype("category").cat.categories if g in present
    ]
    if condition_group is None:
        cnd_list = all_groups
    elif isinstance(condition_group, str):
        cnd_list = [condition_group]
    else:
        cnd_list = list(condition_group)
    cnd_list = [g for g in cnd_list if g != reference_group]
    unknown = [
        g for g in [*cnd_list, reference_group] if g is not None and g not in present
    ]
    if unknown:
        raise ValueError(f"group(s) {unknown} not found in adata.obs[{group_key!r}]")

    if cell_identity_key is None:
        cids = [None]
    else:
        cids = list(adata.obs[cell_identity_key].astype("category").cat.categories)

    if method in SCANPY_METHODS:
        if sample_key is not None:
            raise ValueError("sample_key is only used by the pseudobulk methods")
        layer = unscaled_layer(adata) if layer is None else layer
        if (caveat := scaling_warning(adata, method, layer)) is not None:
            warnings.warn(caveat, UserWarning, stacklevel=2)
        backend = resolve_backend(backend)
        table = _scanpy_de(
            adata,
            group_key,
            method,
            cnd_list,
            reference_group,
            cell_identity_key,
            cids,
            layer,
            {"random_state": random_state, **method_kwargs}
            if method == "logreg"
            else method_kwargs,
            backend,
        )
        if backend == "scanpy":
            backend = f"scanpy.tl.rank_genes_groups(method={method!r})"
            packages = ["scanpy"]
        else:
            backend = f"sclab port of scanpy.tl.rank_genes_groups(method={method!r})"
            packages = ["scipy", "scikit-learn"]
    elif method in PSEUDOBULK_METHODS:
        layer = "counts" if layer is None else layer
        table = _pseudobulk_de(
            adata,
            group_key,
            method,
            cnd_list,
            reference_group,
            cell_identity_key,
            cids,
            layer,
            sample_key,
            random_state,
            method_kwargs,
        )
        backend = PSEUDOBULK_METHODS[method][0].__name__
        packages = ["pylimma", "edgepython"]
    else:
        valid = [*SCANPY_METHODS, *PSEUDOBULK_METHODS]
        raise ValueError(f"method must be one of {valid}, got {method!r}")

    if cell_identity_key is None:
        table = table.drop(columns="cell_identity")

    adata.uns[key_added] = {
        "params": {
            "method": method,
            "backend": backend,
            "group_key": group_key,
            "condition_group": cnd_list,
            "reference_group": "rest" if reference_group is None else reference_group,
            "cell_identity_key": cell_identity_key,
            "layer": layer,
            "sample_key": sample_key,
            "random_state": random_state,
            "method_kwargs": method_kwargs,
            "versions": _versions(["sclab", "anndata", *packages]),
        },
        "table": table,
    }
    if method in SCANPY_METHODS:  # pseudobulk backends record their own
        corr_method = method_kwargs.get("corr_method", "benjamini-hochberg")
        record_citations(adata, "scanpy", *citation_keys(method, corr_method))
    return table


def _scanpy_de(
    adata: AnnData,
    group_key: str,
    method: str,
    cnd_list: list[str],
    reference_group: str | None,
    cell_identity_key: str | None,
    cids: list,
    layer: str,
    method_kwargs: dict,
    backend: str,
) -> pd.DataFrame:
    if backend == "scanpy":
        import scanpy as sc

        rank_genes_groups = sc.tl.rank_genes_groups
    else:
        from ._rank_genes_groups import rank_genes_groups

    if layer == "X":
        X = adata.X
    elif layer in adata.layers:
        X = adata.layers[layer]
    else:
        raise ValueError(f"layer {layer!r} not found in adata.layers")
    if looks_like_raw_counts(X):
        raise ValueError(
            f"layer {layer!r} looks like raw counts (non-negative integers), but "
            f"method={method!r} expects log-normalized expression. Pass a "
            "log-normalized layer=, or use a pseudobulk method for raw counts."
        )

    # a private shell so nothing is written to adata.obs / adata.uns
    shell = AnnData(
        X=X,
        obs=pd.DataFrame(index=adata.obs_names),
        var=pd.DataFrame(index=adata.var_names),
    )
    groups = adata.obs[group_key]
    frames, skipped = [], []
    for cid in cids:
        in_identity = (
            np.ones(adata.n_obs, dtype=bool)
            if cid is None
            else (adata.obs[cell_identity_key] == cid).to_numpy()
        )
        for cnd in cnd_list:
            contrast = (
                f"{cnd}_vs_{'rest' if reference_group is None else reference_group}"
            )
            is_cnd = in_identity & (groups == cnd).to_numpy()
            if reference_group is None:
                is_ref = in_identity & groups.notna().to_numpy() & ~is_cnd
            else:
                is_ref = in_identity & (groups == reference_group).to_numpy()
            if is_cnd.sum() < 2 or is_ref.sum() < 2:
                skipped.append(contrast if cid is None else f"{cid}: {contrast}")
                continue

            codes = np.full(adata.n_obs, -1)
            codes[is_ref], codes[is_cnd] = 0, 1
            # "cnd" has the higher code: logreg's binary coefficient is for it
            shell.obs["_de"] = pd.Categorical.from_codes(codes, ["ref", "cnd"])
            rank_genes_groups(
                shell,
                "_de",
                groups=["cnd"],
                reference="ref",
                method=method,
                use_raw=False,
                key_added="_de",
                **method_kwargs,
            )
            res = shell.uns.pop("_de")
            genes = np.asarray(res["names"]["cnd"]).astype(str)
            idx = adata.var_names.get_indexer(genes)
            mean_cnd, pct_cnd = _mean_pct(X, is_cnd)
            mean_ref, pct_ref = _mean_pct(X, is_ref)
            mean_cnd, pct_cnd, mean_ref, pct_ref = (
                a[idx] for a in (mean_cnd, pct_cnd, mean_ref, pct_ref)
            )
            if "logfoldchanges" in res:
                logfc = res["logfoldchanges"]["cnd"]
            else:  # logreg: same formula as scanpy uses for the other tests
                logfc = np.log2(
                    (np.expm1(mean_cnd) + 1e-9) / (np.expm1(mean_ref) + 1e-9)
                )
            nan = np.full(len(genes), np.nan)
            frames.append(
                pd.DataFrame(
                    {
                        "contrast": contrast,
                        "cell_identity": cid,
                        "gene": genes,
                        "logFC": logfc,
                        "score": res["scores"]["cnd"],
                        "pval": res["pvals"]["cnd"] if "pvals" in res else nan,
                        "padj": res["pvals_adj"]["cnd"] if "pvals_adj" in res else nan,
                        "mean_cnd": mean_cnd,
                        "mean_ref": mean_ref,
                        "pct_cnd": pct_cnd,
                        "pct_ref": pct_ref,
                    }
                )
            )

    if skipped:
        warnings.warn(
            f"Skipped contrasts with fewer than 2 cells on a side: {skipped}",
            stacklevel=3,
        )
    return _concat(frames, [])


def _pseudobulk_de(
    adata: AnnData,
    group_key: str,
    method: str,
    cnd_list: list[str],
    reference_group: str | None,
    cell_identity_key: str | None,
    cids: list,
    layer: str,
    sample_key: str | None,
    random_state: int,
    method_kwargs: dict,
) -> pd.DataFrame:
    backend, renames, extras = PSEUDOBULK_METHODS[method]
    results = backend(
        adata,
        group_key,
        condition_group=cnd_list,
        reference_group=reference_group,
        cell_identity_key=cell_identity_key,
        layer=layer,
        sample_key=sample_key,
        random_state=random_state,
        **method_kwargs,
    )

    frames = []
    for cid in cids:
        for cnd in cnd_list:
            contrast = (
                f"{cnd}_vs_{'rest' if reference_group is None else reference_group}"
            )
            key = contrast if cid is None else f"{cell_identity_key}:{cid}|{contrast}"
            if key not in results:
                continue
            tt = results[key]
            columns = {**PSEUDOBULK_COLUMNS, **renames, **{c: c for c in extras}}
            df = pd.DataFrame({new: tt[old].to_numpy() for new, old in columns.items()})
            df.insert(0, "contrast", contrast)
            df.insert(1, "cell_identity", cid)
            df.insert(2, "gene", tt.index.astype(str))
            frames.append(df)

    return _concat(frames, extras)


def _concat(frames: list[pd.DataFrame], extras: list[str]) -> pd.DataFrame:
    columns = COLUMNS + extras
    if not frames:
        return pd.DataFrame(columns=columns)
    table = pd.concat([f[columns] for f in frames], ignore_index=True)
    numeric = [c for c in columns if c not in ("contrast", "cell_identity", "gene")]
    table[numeric] = table[numeric].astype(float)
    table["contrast"] = table["contrast"].astype(str)
    table["gene"] = table["gene"].astype(str)
    if table["cell_identity"].notna().any():
        table["cell_identity"] = table["cell_identity"].astype(str)
    return table


def _mean_pct(X, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sub = X[mask]
    mean = np.asarray(sub.mean(axis=0), dtype=float).ravel()
    pct = np.asarray((sub != 0).sum(axis=0), dtype=float).ravel() / mask.sum()
    return mean, pct


def _versions(packages: list[str]) -> dict[str, str]:
    out = {}
    for package in packages:
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            pass
    return out
