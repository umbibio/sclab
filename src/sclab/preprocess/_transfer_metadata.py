from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm


def transfer_metadata(
    adata: AnnData,
    group_key: str,
    source_group: str,
    column: str,
    periodic: bool = False,
    vmin: float = 0,
    vmax: float = 1,
):
    D: csr_matrix = adata.obsp["distances"]
    C: csr_matrix = adata.obsp["connectivities"]
    D = D.tocsr()
    C = C.tocsr()

    meta_values: pd.Series
    new_values: pd.Series

    series = adata.obs[column]
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype) or is_bool_dtype(dtype):
        assign_value_fn = _assign_categorical
    elif is_numeric_dtype(dtype) and periodic:
        assign_value_fn = partial(_assign_numerical_periodic, vmin=vmin, vmax=vmax)
    elif is_numeric_dtype(dtype):
        assign_value_fn = _assign_numerical
    else:
        raise ValueError(f"Unsupported dtype {dtype} for column {column}")

    meta_values = adata.obs[column].copy()
    meta_values[adata.obs[group_key] != source_group] = np.nan
    new_values = meta_values.copy()

    for i, (d, c) in tqdm(enumerate(zip(D, C)), total=D.shape[0]):
        if not pd.isna(meta_values.iloc[i]):
            continue

        d = d.tocoo()
        c = c.toarray().ravel()
        neighs = d.coords[1]

        values: pd.Series = meta_values.iloc[neighs]
        msk = pd.notna(values)
        if msk.sum() < 2:
            continue

        values = values.loc[msk]
        weights = c[neighs][msk]

        if np.allclose(weights, 0):
            continue

        new_values.iloc[i] = assign_value_fn(values, weights)

    adata.obs[f"transferred_{column}"] = new_values.copy()


def _assign_categorical(values: pd.Series, weights: NDArray):
    # weighted majority
    tally = Counter(dict(zip(values, weights))).most_common()
    return tally[0][0]


def _assign_numerical(values: pd.Series, weights: NDArray):
    # weighted average
    return np.average(values, weights=weights)


def _assign_numerical_periodic(
    values: pd.Series, weights: NDArray, vmin: float, vmax: float
):
    vspan = vmax - vmin

    values = values - vmin
    offset = np.median(values)
    values = values - offset + vspan / 2
    values = values % vspan
    assigned_value = np.average(values, weights=weights)
    assigned_value = assigned_value + offset - vspan / 2
    assigned_value = assigned_value % vspan
    assigned_value = assigned_value + vmin

    return assigned_value
