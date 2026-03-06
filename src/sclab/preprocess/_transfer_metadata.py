from collections import Counter
from functools import partial
from typing import Callable, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy.sparse import csr_matrix
from scipy.special import gamma
from tqdm.auto import tqdm


def transfer_metadata(
    adata: AnnData,
    group_key: str,
    source_group: str,
    column: str,
    periodic: bool = False,
    vmin: float = 0,
    vmax: float = 1,
    min_neighs: int = 5,
    weight_by: Literal["connectivity", "distance", "constant"] = "connectivity",
):
    """Transfer a metadata column from a source group to the rest of the cells.

    Uses the k-nearest-neighbor graph (``adata.obsp["connectivities"]`` and
    ``adata.obsp["distances"]``) to propagate values from labeled cells
    (``source_group``) to unlabeled cells. Results are stored as new columns
    with the ``transferred_`` prefix.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with a computed neighbor graph. Modified
        in-place.
    group_key : str
        Column in ``adata.obs`` identifying the groups (e.g. ``"batch"``).
    source_group : str
        Value in ``adata.obs[group_key]`` whose cells serve as the labeled
        source. Cells in all other groups receive transferred values.
    column : str
        Column in ``adata.obs`` containing the values to transfer (numeric,
        categorical, or boolean).
    periodic : bool, optional
        If True, treat ``column`` as a periodic variable (e.g. cell-cycle
        phase in [vmin, vmax]). Default is False.
    vmin : float, optional
        Minimum value for periodic wrapping. Default is 0.
    vmax : float, optional
        Maximum value for periodic wrapping. Default is 1.
    min_neighs : int, optional
        Minimum number of labeled neighbors required to assign a value.
        Cells with fewer labeled neighbors are left as NaN. Default is 5.
    weight_by : {"connectivity", "distance", "constant"}, optional
        How to weight neighbors when aggregating values. ``"connectivity"``
        uses the connectivity matrix; ``"distance"`` uses inverse distances;
        ``"constant"`` gives equal weight to all neighbors. Default is
        ``"connectivity"``.

    Returns
    -------
    None
        Adds ``transferred_{column}`` and ``transferred_{column}_error``
        (or ``transferred_{column}_proportion`` for categorical columns)
        to ``adata.obs``.
    """
    new_values, new_values_err = _propagate_metadata(
        adata,
        column=column,
        periodic=periodic,
        vmin=vmin,
        vmax=vmax,
        min_neighs=min_neighs,
        weight_by=weight_by,
        mask=adata.obs[group_key] != source_group,
    )

    adata.obs[f"transferred_{new_values.name}"] = new_values
    adata.obs[f"transferred_{new_values_err.name}"] = new_values_err


def propagate_metadata(
    adata: AnnData,
    column: str,
    periodic: bool = False,
    vmin: float = 0,
    vmax: float = 1,
    min_neighs: int = 5,
    weight_by: Literal["connectivity", "distance", "constant"] = "connectivity",
):
    """Fill missing values in a metadata column by propagation through the neighbor graph.

    Cells that already have a value in ``column`` are used as anchors; NaN
    cells receive an estimated value from their labeled neighbors. Useful
    for imputing partially annotated metadata (e.g. pseudotime or cell-type
    labels) based on the k-NN graph structure.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with a computed neighbor graph. Modified
        in-place.
    column : str
        Column in ``adata.obs`` with partial values (NaNs to be filled).
    periodic : bool, optional
        If True, treat the variable as periodic (circular). Default is False.
    vmin : float, optional
        Minimum value for periodic wrapping. Default is 0.
    vmax : float, optional
        Maximum value for periodic wrapping. Default is 1.
    min_neighs : int, optional
        Minimum number of labeled neighbors required to assign a value.
        Default is 5.
    weight_by : {"connectivity", "distance", "constant"}, optional
        Neighbor weighting scheme. Default is ``"connectivity"``.

    Returns
    -------
    None
        Fills NaN entries in ``adata.obs[column]`` in-place and adds an
        error/proportion column (``{column}_error`` or
        ``{column}_proportion``).
    """
    new_values, new_values_err = _propagate_metadata(
        adata,
        column=column,
        periodic=periodic,
        vmin=vmin,
        vmax=vmax,
        min_neighs=min_neighs,
        weight_by=weight_by,
    )

    mask = adata.obs[column].isna()
    adata.obs.loc[mask, column] = new_values.loc[mask]
    adata.obs.loc[mask, new_values_err.name] = new_values_err.loc[mask]


def _propagate_metadata(
    adata: AnnData,
    column: str,
    periodic: bool = False,
    vmin: float = 0,
    vmax: float = 1,
    min_neighs: int = 5,
    weight_by: Literal["connectivity", "distance", "constant"] = "connectivity",
    mask: np.ndarray | pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    D, W = _get_neighbors_and_weights(adata, weight_by=weight_by)

    assign_value_fn: Callable
    series = adata.obs[column]
    if isinstance(series.dtype, pd.CategoricalDtype) or is_bool_dtype(series.dtype):
        assign_value_fn = _assign_categorical
    elif is_numeric_dtype(series.dtype) and periodic:
        assign_value_fn = partial(_assign_numerical_periodic, vmin=vmin, vmax=vmax)
    elif is_numeric_dtype(series.dtype):
        assign_value_fn = _assign_numerical
    else:
        raise ValueError(f"Unsupported dtype {series.dtype} for column {column}")

    if isinstance(series.dtype, pd.CategoricalDtype) or is_bool_dtype(series.dtype):
        column_err = f"{column}_proportion"
    else:
        column_err = f"{column}_error"

    meta_values: pd.Series = series.copy()
    if mask is not None:
        meta_values[mask] = pd.NA

    new_values = pd.Series(index=series.index, dtype=series.dtype, name=column)
    new_values_err = pd.Series(index=series.index, dtype=float, name=column_err)

    for i, (d, w) in tqdm(enumerate(zip(D, W)), total=D.shape[0]):
        if not pd.isna(meta_values.iloc[i]):
            continue

        d = d.tocoo()
        w = w.toarray().ravel()
        neighs = d.coords[1]

        values: pd.Series = meta_values.iloc[neighs]
        msk = pd.notna(values)
        if msk.sum() < min_neighs:
            continue

        values = values.loc[msk]
        weights = w[neighs][msk]

        if np.allclose(weights, 0):
            continue

        assigned_value, assigned_value_err = assign_value_fn(values, weights)
        new_values.iloc[i] = assigned_value
        new_values_err.iloc[i] = assigned_value_err

    new_values = pd.concat([new_values, meta_values], axis=1).bfill(axis=1).iloc[:, 0]

    return new_values, new_values_err


def _get_neighbors_and_weights(
    adata: AnnData,
    weight_by: Literal["connectivity", "distance", "constant"] = "connectivity",
):
    D: csr_matrix = adata.obsp["distances"].copy()
    C: csr_matrix = adata.obsp["connectivities"].copy()
    D = D.tocsr()
    W: csr_matrix

    match weight_by:
        case "connectivity":
            W = C.tocsr().copy()
        case "distance":
            W = D.tocsr().copy()
            W.data = 1.0 / W.data
        case "constant":
            W = D.tocsr().copy()
            W.data[:] = 1.0
        case _:
            raise ValueError(f"Unsupported weight_by {weight_by}")

    return D, W


def _assign_categorical(values: pd.Series, weights: NDArray):
    # weighted majority and proportion of votes
    tally = Counter()
    for v, w in zip(values, weights):
        tally[v] += w

    winner, shares = tally.most_common()[0]
    return winner, shares / weights.sum()


def _assign_numerical(values: pd.Series, weights: NDArray):
    # weighted mean and standard error
    sum_w: float = weights.sum()
    sum2_w: float = weights.sum() ** 2
    sum_w2: float = (weights**2).sum()
    n_eff: float = sum2_w / sum_w2

    mean_x: float = (values * weights).sum() / sum_w
    var_x: float = ((values - mean_x) ** 2 * weights).sum() * sum_w / (sum2_w - sum_w2)
    err_x: float = np.sqrt(var_x / n_eff)

    return mean_x, err_x


def _assign_numerical_periodic(
    values: pd.Series, weights: NDArray, vmin: float, vmax: float
):
    vspan = vmax - vmin

    values = values - vmin
    offset = np.median(values)
    values = values - offset + vspan / 2
    values = values % vspan
    assigned_value, assigned_value_err = _assign_numerical(values, weights)
    assigned_value = assigned_value + offset - vspan / 2
    assigned_value = assigned_value % vspan
    assigned_value = assigned_value + vmin

    return assigned_value, assigned_value_err


def _c4(n: float):
    # correct for bias
    nm1 = n - 1
    return np.sqrt(2 / nm1) * gamma(n / 2) / gamma(nm1 / 2)
