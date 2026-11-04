"""Merge datasets along obs: a pre-merge check, the merge and depth thinning.

The merge keeps the cells of every dataset, one gene set (``join``) and the raw
counts: ``X`` and the counts layer of the result are both the parents' counts
layers concatenated. Other layers, obsm, obsp, varm and uns are dropped; the
result's step history starts with a ``"merge"`` entry naming its parents.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from anndata import AnnData, concat
from scipy.sparse import issparse

from ..dataset.processor import _history
from ..utils import check_raw_counts, looks_like_raw_counts

INDEX_UNIQUE = "-"


def _id_kind(var_names: pd.Index) -> str:
    ensembl = pd.Index(var_names).astype(str).str.match(r"^ENS").mean()
    return "ensembl" if ensembl > 0.5 else "symbol"


def _depth(X) -> tuple[float, float]:
    """Median total counts and median detected genes per cell."""
    if X.shape[0] == 0:
        return 0.0, 0.0
    totals = np.asarray(X.sum(axis=1)).ravel()
    genes = np.asarray((X > 0).sum(axis=1)).ravel()
    return float(np.median(totals)), float(np.median(genes))


def _obs_conflicts(adatas: Mapping[str, AnnData]) -> dict[str, str]:
    """obs columns shared by datasets whose dtype or category set differs."""
    dtypes: dict[str, dict[str, object]] = {}
    for name, adata in adatas.items():
        for col, dtype in adata.obs.dtypes.items():
            dtypes.setdefault(col, {})[name] = dtype
    conflicts = {}
    for col, by_name in dtypes.items():
        if len(by_name) < 2:
            continue
        cats = [isinstance(d, pd.CategoricalDtype) for d in by_name.values()]
        if all(cats):
            sets = {frozenset(d.categories) for d in by_name.values()}
            if len(sets) > 1:
                conflicts[col] = "categories differ"
        elif len({str(d) for d in by_name.values()}) > 1:
            conflicts[col] = ", ".join(f"{n}: {d}" for n, d in by_name.items())
    return conflicts


def merge_preview(
    adatas: Mapping[str, AnnData],
    *,
    join: str = "inner",
    label: str = "sample",
    counts_layer: str = "counts",
    equalize_depth: bool = False,
) -> dict:
    """What :func:`merge_datasets` would do with ``adatas``, and what blocks it.

    Returns a dict with ``n_obs`` / ``n_vars`` of the result, ``shared_genes``
    (genes in every dataset), per-dataset ``datasets[name]`` stats (``n_obs``,
    ``n_vars``, ``id_kind`` "ensembl" or "symbol", ``genes_dropped`` by the
    join, ``genes_filled`` with zeros by an outer join, ``counts`` "ok",
    "missing" or "not integer", ``median_counts`` and ``median_genes`` per
    cell over the genes kept, ``depth_ratio`` to the shallowest dataset),
    ``obs_conflicts`` (column -> how it differs), and the ``errors`` that
    block the merge and ``warnings`` that do not.
    """
    if join not in ("inner", "outer"):
        raise ValueError(f"join must be 'inner' or 'outer', not {join!r}")
    errors, warnings = [], []
    names = list(adatas)
    if len(names) < 2:
        errors.append("Select at least two datasets to merge.")
    if not label:
        errors.append("The label column name is empty.")

    var_sets = [pd.Index(a.var_names) for a in adatas.values()]
    shared = var_sets[0] if var_sets else pd.Index([])
    union = shared
    for v in var_sets[1:]:
        shared = shared.intersection(v, sort=False)
        union = union.union(v, sort=False)
    kept = shared if join == "inner" else union
    if names and join == "inner" and len(shared) == 0:
        errors.append("The datasets share no genes: an inner join keeps none.")

    datasets = {}
    for name, adata in adatas.items():
        info = {"n_obs": adata.n_obs, "n_vars": adata.n_vars}
        info["id_kind"] = _id_kind(adata.var_names)
        info["genes_dropped"] = int(adata.n_vars - adata.var_names.isin(kept).sum())
        info["genes_filled"] = int(len(kept) - kept.isin(adata.var_names).sum())
        if not adata.var_names.is_unique:
            errors.append(
                f"{name}: gene names are not unique; "
                "call adata.var_names_make_unique() first."
            )
        if counts_layer not in adata.layers:
            info["counts"] = "missing"
            errors.append(f"{name}: no {counts_layer!r} layer with raw counts.")
        else:
            counts = adata.layers[counts_layer]
            ok = looks_like_raw_counts(counts)
            info["counts"] = "ok" if ok else "not integer"
            if not ok:
                (errors if equalize_depth else warnings).append(
                    f"{name}: the {counts_layer!r} layer is not integer counts"
                    + (", so its depth cannot be equalized." if equalize_depth else ".")
                )
            if adata.var_names.is_unique:
                cols = adata.var_names.get_indexer(adata.var_names.intersection(kept))
                info["median_counts"], info["median_genes"] = _depth(
                    counts[:, np.sort(cols)]
                )
        if label and label in adata.obs:
            errors.append(
                f"{name}: obs already has a {label!r} column; "
                "choose another label or rename it."
            )
        datasets[name] = info

    depths = [i["median_counts"] for i in datasets.values() if "median_counts" in i]
    shallowest = min(depths) if depths else 0
    for info in datasets.values():
        if "median_counts" in info and shallowest > 0:
            info["depth_ratio"] = info["median_counts"] / shallowest
    if equalize_depth and depths and shallowest == 0:
        errors.append("A dataset has a median of 0 counts per cell.")

    kinds = {i["id_kind"] for i in datasets.values()}
    if len(kinds) > 1:
        by_kind = {
            k: ", ".join(n for n, i in datasets.items() if i["id_kind"] == k)
            for k in sorted(kinds)
        }
        message = (
            f"Datasets mix Ensembl IDs ({by_kind['ensembl']}) and gene symbols "
            f"({by_kind['symbol']}): their genes would not match."
        )
        with_ids = [n for n, a in adatas.items() if "gene_ids" in a.var]
        if with_ids:
            message += (
                f" {', '.join(with_ids)} has a var 'gene_ids' column: use it as the "
                "gene names before merging, e.g. adata.var_names = adata.var['gene_ids']."
            )
        errors.append(message)

    for name, info in datasets.items():
        if (
            join == "inner"
            and info["n_vars"]
            and info["genes_dropped"] > info["n_vars"] / 2
        ):
            warnings.append(
                f"{name}: the inner join drops {info['genes_dropped']} of its "
                f"{info['n_vars']} genes."
            )
    conflicts = _obs_conflicts(adatas)
    for col, how in conflicts.items():
        warnings.append(f"obs column {col!r} differs between datasets ({how}).")

    return {
        "join": join,
        "label": label,
        "counts_layer": counts_layer,
        "n_obs": sum(a.n_obs for a in adatas.values()),
        "n_vars": len(kept),
        "shared_genes": len(shared),
        "datasets": datasets,
        "obs_conflicts": conflicts,
        "errors": errors,
        "warnings": warnings,
    }


def downsample_counts(
    adata: AnnData,
    fraction: float | Mapping[str, float],
    *,
    by: str | None = None,
    layer: str | None = None,
    random_state: int = 0,
) -> None:
    """Binomial thinning of raw counts, in place.

    Each count is kept with probability ``fraction``, as DropletUtils'
    ``downsampleMatrix`` / ``downsampleBatches`` do, so the result stays
    integer counts with its expected depth scaled by ``fraction``.

    Parameters
    ----------
    adata
        Annotated data matrix; ``X`` (or ``layer``) must hold raw counts.
    fraction
        Probability of keeping each count, or with ``by``, a mapping of
        ``adata.obs[by]`` values to probabilities (absent values: 1, kept).
    by
        obs column grouping the cells for per-group fractions.
    layer
        Layer to thin instead of ``X``.
    random_state
        Seed of the random number generator.
    """
    X = adata.X if layer is None else adata.layers[layer]
    check_raw_counts(X, name=f"layer {layer!r}" if layer else "X")
    if by is None:
        p = np.full(adata.n_obs, float(fraction))
    else:
        groups = adata.obs[by].astype(object)
        p = groups.map(dict(fraction)).astype(float).fillna(1.0).to_numpy()
    if np.any((p < 0) | (p > 1)):
        raise ValueError("fractions must be between 0 and 1")

    rng = np.random.default_rng(random_state)
    if issparse(X):
        if X.format not in ("csr", "csc"):
            X = X.tocsr()
        rows = (
            X.indices
            if X.format == "csc"
            else np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
        )
        X.data = rng.binomial(X.data.astype(np.int64), p[rows]).astype(X.dtype)
        X.eliminate_zeros()
    else:
        X = np.asarray(X)
        X = rng.binomial(X.astype(np.int64), p[:, None]).astype(X.dtype)
    if layer is None:
        adata.X = X
    else:
        adata.layers[layer] = X


def _parent_source(adata: AnnData) -> str | None:
    """Where ``adata`` was read from, per the ``"load"`` entry of its history."""
    history = _history.loads_uns(adata)
    if history and history[0]["step"] == "load":
        return history[0]["params"].get("source")
    return None


def merge_datasets(
    adatas: Mapping[str, AnnData],
    *,
    join: str = "inner",
    label: str = "sample",
    counts_layer: str = "counts",
    equalize_depth: bool = False,
    random_state: int = 0,
) -> AnnData:
    """Concatenate datasets' cells into a new AnnData of raw counts.

    ``anndata.concat`` along obs with ``keys`` = the names in ``adatas``,
    ``label`` naming the obs column of each cell's dataset, ``index_unique="-"``
    and ``merge="same"``; genes missing from a dataset in an outer join are
    filled with zeros. ``X`` and the counts layer of the result are the
    parents' counts layers; the parents are not modified.

    Parameters
    ----------
    adatas
        Datasets by name; each needs raw counts in ``counts_layer``.
    join
        ``"inner"`` keeps the genes in every dataset, ``"outer"`` all genes.
    label
        obs column of the result naming each cell's dataset (its batch key).
    counts_layer
        Layer with the raw counts, in the parents and the result.
    equalize_depth
        Thin the counts of each deeper dataset (binomially, see
        :func:`downsample_counts`) by the ratio of the shallowest dataset's
        median total counts per cell to its own, over the genes kept.
    random_state
        Seed of the thinning.

    Raises
    ------
    ValueError
        If :func:`merge_preview` reports blocking errors.
    """
    preview = merge_preview(
        adatas,
        join=join,
        label=label,
        counts_layer=counts_layer,
        equalize_depth=equalize_depth,
    )
    if preview["errors"]:
        raise ValueError("Cannot merge: " + " ".join(preview["errors"]))
    names = list(adatas)
    parts = [
        AnnData(X=a.layers[counts_layer], obs=a.obs, var=a.var) for a in adatas.values()
    ]
    merged = concat(
        parts,
        axis=0,
        join=join,
        label=label,
        keys=names,
        index_unique=INDEX_UNIQUE,
        merge="same",
        uns_merge=None,
        fill_value=0,
    )
    if equalize_depth:
        totals = pd.Series(np.asarray(merged.X.sum(axis=1)).ravel())
        medians = totals.groupby(merged.obs[label].to_numpy()).median()
        fractions = (medians.min() / medians).to_dict()
        downsample_counts(merged, fractions, by=label, random_state=random_state)
    merged.layers[counts_layer] = merged.X.copy()

    parents = [
        {
            "name": name,
            "history_hash": _history.history_hash(_history.loads_uns(adata)),
            "source": _parent_source(adata),
        }
        for name, adata in adatas.items()
    ]
    params = {
        "parents": parents,
        "join": join,
        "label": label,
        "keys": names,
        "index_unique": INDEX_UNIQUE,
        "counts_layer": counts_layer,
        "equalize_depth": equalize_depth,
        "random_state": random_state,
    }
    entry = _history.make_entry("merge", params, "success", None, None, merged.shape)
    merged.uns[_history.UNS_KEY] = {"history": _history.dumps_uns([entry])}
    return merged
