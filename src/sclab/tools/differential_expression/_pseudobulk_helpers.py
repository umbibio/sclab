from collections.abc import Callable

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse

from sclab.utils import check_raw_counts

from ..utils import aggregate_and_filter

__all__ = [
    "aggregate_and_filter",
    "pseudobulk_contrasts",
    "pseudobulk_design",
    "validate_counts_layer",
]


def validate_counts_layer(adata: AnnData, layer: str, check_counts: bool) -> None:
    """Raise ``ValueError`` for a missing layer or (optionally) non-count data.

    ``layer="X"`` selects ``adata.X``.
    """
    if layer is None:
        raise ValueError(
            "layer=None is ambiguous; pass layer='counts' for raw counts "
            "or layer='X' to use adata.X"
        )
    if layer == "X":
        X = adata.X
    elif layer in adata.layers:
        X = adata.layers[layer]
    else:
        raise ValueError(
            f"layer '{layer}' not found; pass layer='X' if adata.X holds raw counts"
        )

    if check_counts:
        check_raw_counts(
            X,
            name=f"layer '{layer}'",
            hint="Pseudobulk DE needs raw counts; use layer='counts' "
            "(or check_counts=False for non-integer corrected counts).",
        )


def pseudobulk_design(
    aggr_adata: AnnData, groups: pd.Series, verbosity: int = 0
) -> pd.DataFrame:
    """Design matrix ``~ 0 + group + replica`` for pseudobulk samples.

    The ``replica`` term is dropped when there is a single replica, or when
    it is not estimable (replicas nested in groups, e.g. real samples that
    each belong to one group).
    """
    design = pd.get_dummies(groups).astype(float)
    if "replica" in aggr_adata.obs and aggr_adata.obs["replica"].nunique() > 1:
        replica = aggr_adata.obs["replica"].astype(str)
        # treatment coding, as R's model.matrix(~ 0 + group + replica)
        replica_dummies = pd.get_dummies(replica, prefix="replica", drop_first=True)
        full = design.join(replica_dummies.astype(float))
        if np.linalg.matrix_rank(full.values) == full.shape[1]:
            return full
        if verbosity > 0:
            print("Replicas nested in groups — dropping replica term from design.")
    elif verbosity > 0:
        print("Single replicate detected — dropping replica term from design.")
    return design


def pseudobulk_contrasts(
    aggr_adata: AnnData,
    group_key: str,
    condition_group: str | list[str] | None,
    reference_group: str | None,
    cell_identity_key: str | None,
    fit: Callable[[AnnData, pd.DataFrame], tuple[object, np.ndarray]],
    test: Callable[[object, pd.DataFrame, str, str, np.ndarray], pd.DataFrame],
    sample_key: str | None = None,
    verbosity: int = 0,
) -> dict[str, pd.DataFrame]:
    """Fit one model per condition group and test each requested contrast.

    ``fit(aggr_adata, design)`` returns ``(model, genes)``; ``test(model,
    design, cnd, ref, genes)`` returns a table indexed by gene. Contrasts are
    keyed ``"B_vs_rest"`` / ``"B_vs_A"``, prefixed with
    ``"{cell_identity_key}:{identity}|"`` when stratified. Adds the
    ``pct_expr_*``, ``num_expr_*``, ``tot_expr_*``, ``mean_*`` (mean count in
    expressing cells) and ``logCPM_*`` (mean log2-CPM of the side's pseudobulk
    samples) columns for the condition (``cnd``) and reference (``ref``) sides.
    """
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

    log_cpm = _log_cpm(aggr_adata)

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

        design = pseudobulk_design(aggr_adata, groups, verbosity)
        try:
            model, genes = fit(aggr_adata, design)
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

            if sample_key is not None:
                n_cnd, n_ref = (groups == cnd).sum(), (groups == ref).sum()
                if min(n_cnd, n_ref) < 2:
                    raise ValueError(
                        f"Contrast {contrast_key!r} has {n_cnd} condition and "
                        f"{n_ref} reference sample(s) from sample_key="
                        f"{sample_key!r}; each side needs at least 2 samples. "
                        "Drop the contrast (condition_group/use_cells) or use "
                        "sample_key=None for random pseudo-replicas."
                    )

            if verbosity > 0:
                print(f"Computing contrast: {contrast_key}... ({cnd} - {ref})")

            try:
                tt = test(model, design, cnd, ref, genes)
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
            tt["logCPM_cnd"] = log_cpm[(groups == cnd).values].mean().loc[tt.index]
            tt["logCPM_ref"] = log_cpm[(groups == ref).values].mean().loc[tt.index]
            tt_dict[contrast_key] = tt

    return tt_dict


def _log_cpm(aggr_adata: AnnData) -> pd.DataFrame:
    """Samples x genes log2-CPM, as limma's voom: log2((y + 0.5) / (N + 1) * 1e6)."""
    X = aggr_adata.X.toarray() if issparse(aggr_adata.X) else aggr_adata.X
    X = np.asarray(X, dtype=float)
    lib_size = X.sum(axis=1, keepdims=True)
    values = np.log2((X + 0.5) / (lib_size + 1) * 1e6)
    return pd.DataFrame(
        values, index=aggr_adata.obs_names, columns=aggr_adata.var_names
    )
