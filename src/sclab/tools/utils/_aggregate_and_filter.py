import random

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy import ndarray
from scipy.sparse import csr_matrix, issparse


# code inspired from
# https://www.sc-best-practices.org/conditions/differential_gene_expression.html
def aggregate_and_filter(
    adata: AnnData,
    group_key: str = "batch",
    cell_identity_key: str | None = None,
    layer: str | None = None,
    replicas_per_group: int = 3,
    min_cells_per_group: int = 30,
    bootstrap_sampling: bool = False,
    use_cells: dict[str, list[str]] | None = None,
    make_stats: bool = True,
    make_dummies: bool = True,
) -> AnnData:
    """
    Aggregate and filter cells in an AnnData object into cell populations.

    Parameters
    ----------
    adata : AnnData
        AnnData object to aggregate and filter.
    group_key : str, optional
        Key to group cells by. Defaults to 'batch'.
    cell_identity_key : str, optional
        Key to use to identify cell identities. Defaults to None.
    layer : str, optional
        Layer in AnnData object to use for aggregation. Defaults to None.
    replicas_per_group : int, optional
        Number of replicas to create for each group. Defaults to 3.
    min_cells_per_group : int, optional
        Minimum number of cells required for a group to be included. Defaults to 30.
    bootstrap_sampling : bool, optional
        Whether to use bootstrap sampling to create replicas. Defaults to False.
    use_cells : dict[str, list[str]], optional
        If not None, only use the specified cells. Defaults to None.
    make_stats : bool, optional
        Whether to create expression statistics for each group. Defaults to True.
    make_dummies : bool, optional
        Whether to make categorical columns into dummies. Defaults to True.

    Returns
    -------
    AnnData
        AnnData object with aggregated and filtered cells.
    """
    adata = _prepare_dataset(adata, use_cells)

    grouping_keys = [group_key]
    if cell_identity_key is not None:
        grouping_keys.append(cell_identity_key)

    groups_to_drop = _get_groups_to_drop(adata, grouping_keys, min_cells_per_group)

    _prepare_categorical_column(adata, group_key)
    group_dtype = adata.obs[group_key].dtype

    if cell_identity_key is not None:
        _prepare_categorical_column(adata, cell_identity_key)
        cell_identity_dtype = adata.obs[cell_identity_key].dtype

    if make_stats:
        var_dataframe = _create_var_dataframe(
            adata, layer, grouping_keys, groups_to_drop
        )
    else:
        var_dataframe = pd.DataFrame(index=adata.var_names)

    data = {}
    meta = {}
    groups = adata.obs.groupby(grouping_keys, observed=True).groups
    for group, group_idxs in groups.items():
        if not isinstance(group, tuple):
            group = (group,)

        if not _including(group, groups_to_drop):
            continue

        sample_id = "_".join(group)
        match group:
            case (gid, cid):
                group_metadata = {group_key: gid, cell_identity_key: cid}
            case (gid,):
                group_metadata = {group_key: gid}

        adata_group = adata[group_idxs]
        indices = _get_replica_idxs(adata_group, replicas_per_group, bootstrap_sampling)
        for i, rep_idx in enumerate(indices):
            replica_number = i + 1
            replica_size = len(rep_idx)
            replica_sample_id = f"{sample_id}_rep{replica_number}"

            adata_group_replica = adata_group[rep_idx]
            X = _get_layer(adata_group_replica, layer)

            data[replica_sample_id] = np.array(X.sum(axis=0)).flatten()
            meta[replica_sample_id] = {
                **group_metadata,
                "replica": str(replica_number),
                "replica_size": replica_size,
            }

    data = pd.DataFrame(data).T
    meta = pd.DataFrame(meta).T
    meta["replica"] = meta["replica"].astype("category")
    meta["replica_size"] = meta["replica_size"].astype(int)
    meta[group_key] = meta[group_key].astype(group_dtype)
    if cell_identity_key is not None:
        meta[cell_identity_key] = meta[cell_identity_key].astype(cell_identity_dtype)

    aggr_adata = AnnData(
        data.values,
        obs=meta,
        var=var_dataframe,
    )

    if make_dummies:
        _join_dummies(aggr_adata, group_key)

    return aggr_adata


def _prepare_dataset(
    adata: AnnData,
    use_cells: dict[str, list[str]] | None,
) -> AnnData:
    if use_cells is not None:
        for key, value in use_cells.items():
            adata = adata[adata.obs[key].isin(value)]

    return adata.copy()


def _get_groups_to_drop(
    adata: AnnData,
    grouping_keys: str | list[str],
    min_cells_per_group: int,
):
    group_sizes = adata.obs.groupby(grouping_keys, observed=True).size()
    groups_to_drop = group_sizes[group_sizes < min_cells_per_group].index.to_list()

    if len(groups_to_drop) > 0:
        print("Dropping the following samples:")
        print(groups_to_drop)

    groups_to_drop = groups_to_drop + [
        (g,) for g in groups_to_drop if not isinstance(g, tuple)
    ]

    return groups_to_drop


def _prepare_categorical_column(adata: AnnData, column: str) -> None:
    if not isinstance(adata.obs[column].dtype, pd.CategoricalDtype):
        adata.obs[column] = adata.obs[column].astype("category")


def _create_var_dataframe(
    adata: AnnData,
    layer: str,
    grouping_keys: list[str],
    groups_to_drop: list[str],
):
    columns = _get_var_dataframe_columns(adata, grouping_keys, groups_to_drop)
    var_dataframe = pd.DataFrame(index=adata.var_names, columns=columns, dtype=float)

    groups = adata.obs.groupby(grouping_keys, observed=True).groups
    for group, idx in groups.items():
        if not isinstance(group, tuple):
            group = (group,)

        if not _including(group, groups_to_drop):
            continue

        sample_id = "_".join(group)
        rest_id = f"not{sample_id}"

        adata_subset = adata[idx]
        rest_subset = adata[~adata.obs_names.isin(idx)]

        X = _get_layer(adata_subset, layer, dense=True)
        Y = _get_layer(rest_subset, layer, dense=True)

        var_dataframe[f"pct_expr_{sample_id}"] = (X > 0).mean(axis=0)
        var_dataframe[f"pct_expr_{rest_id}"] = (Y > 0).mean(axis=0)
        var_dataframe[f"num_expr_{sample_id}"] = (X > 0).sum(axis=0)
        var_dataframe[f"num_expr_{rest_id}"] = (Y > 0).sum(axis=0)
        var_dataframe[f"tot_expr_{sample_id}"] = X.sum(axis=0)
        var_dataframe[f"tot_expr_{rest_id}"] = Y.sum(axis=0)

    return var_dataframe


def _get_var_dataframe_columns(
    adata: AnnData, grouping_keys: list[str], groups_to_drop: list[str]
) -> list[str]:
    columns = []

    groups = adata.obs.groupby(grouping_keys, observed=True).groups
    for group, _ in groups.items():
        if not isinstance(group, tuple):
            group = (group,)

        if not _including(group, groups_to_drop):
            continue

        sample_id = "_".join(group)
        rest_id = f"not{sample_id}"

        columns.extend(
            [
                f"pct_expr_{sample_id}",
                f"pct_expr_{rest_id}",
                f"num_expr_{sample_id}",
                f"num_expr_{rest_id}",
                f"tot_expr_{sample_id}",
                f"tot_expr_{rest_id}",
            ]
        )

    return columns


def _including(group: tuple | str, groups_to_drop: list[str]) -> bool:
    match group:
        case (gid, cid):
            if isinstance(cid, float) and np.isnan(cid):
                return False

        case (gid,) | gid:
            ...

    if group in groups_to_drop:
        return False

    if gid in groups_to_drop:
        return False

    return True


def _get_replica_idxs(
    adata_group: AnnData,
    replicas_per_group: int,
    bootstrap_sampling: bool,
):
    group_size = adata_group.n_obs
    indices = list(adata_group.obs_names)
    if bootstrap_sampling:
        indices = np.array(
            [
                np.random.choice(indices, size=group_size, replace=True)
                for _ in range(replicas_per_group)
            ]
        )

    else:
        random.shuffle(indices)
        indices = np.array_split(np.array(indices), replicas_per_group)

    return indices


def _get_layer(adata: AnnData, layer: str | None, dense: bool = False):
    X: ndarray | csr_matrix

    if layer is None or layer == "X":
        X = adata.X
    else:
        X = adata.layers[layer]

    if dense:
        if issparse(X):
            X = np.asarray(X.todense())
        else:
            X = np.asarray(X)

    return X


def _join_dummies(aggr_adata: AnnData, group_key: str) -> None:
    dummies = pd.get_dummies(aggr_adata.obs[group_key], prefix=group_key).astype(str)
    dummies = dummies.astype(str).apply(lambda s: s.map({"True": "", "False": "not"}))
    dummies = dummies + aggr_adata.obs[group_key].cat.categories

    aggr_adata.obs = aggr_adata.obs.join(dummies)
