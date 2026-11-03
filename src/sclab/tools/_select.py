"""Select cells by a box or a polygon drawn on an embedding or two obs columns."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.path import Path


def _coordinates(adata: AnnData, basis: str | None, dims: Sequence) -> np.ndarray:
    if basis is None:
        columns = [adata.obs[d].to_numpy(dtype=float, na_value=np.nan) for d in dims]
        return np.column_stack(columns)
    X = adata.obsm[basis]
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    return X[:, list(dims)].astype(float)


def select_box(
    adata: AnnData,
    basis: str | None,
    x: tuple[float, float] | None = None,
    y: tuple[float, float] | None = None,
    dims: Sequence[int] | Sequence[str] = (0, 1),
) -> np.ndarray:
    """Cells whose coordinates fall inside an axis-aligned box.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    basis : str or None
        Key of ``adata.obsm`` holding the coordinates (e.g. ``"X_umap"``). If
        None, ``dims`` are ``adata.obs`` column names.
    x, y : tuple of float or None, optional
        ``(min, max)`` range on the first and second dimension; the bounds may
        be given in either order. None means no constraint on that axis, so
        ``y=None`` selects a range on one axis (e.g. from a 1D plot).
    dims : sequence of int or str, optional
        Columns of ``adata.obsm[basis]`` (positions) or obs column names
        (if ``basis`` is None) used as the x and y axes. Default ``(0, 1)``.

    Returns
    -------
    numpy.ndarray
        Boolean mask of length ``adata.n_obs``.

    Notes
    -----
    Bounds are inclusive on every edge, as in plotly's box select. Cells with
    a NaN coordinate on a constrained axis are never selected.
    """
    if len(dims) == 1 and y is not None:
        raise ValueError("a y range needs two dims")
    coords = _coordinates(adata, basis, dims)
    mask = np.ones(adata.n_obs, dtype=bool)
    for values, bounds in zip(coords.T, (x, y)):
        if bounds is not None:
            lo, hi = sorted(bounds)
            mask &= (values >= lo) & (values <= hi)
    return mask


def select_polygon(
    adata: AnnData,
    basis: str | None,
    polygon: Sequence[tuple[float, float]] | np.ndarray,
    dims: Sequence[int] | Sequence[str] = (0, 1),
) -> np.ndarray:
    """Cells whose coordinates fall inside a polygon (e.g. a lasso).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    basis : str or None
        Key of ``adata.obsm`` holding the coordinates (e.g. ``"X_umap"``). If
        None, ``dims`` are ``adata.obs`` column names.
    polygon : sequence of (x, y) or numpy.ndarray of shape (n, 2)
        Polygon vertices, at least three. The polygon is closed implicitly.
    dims : sequence of int or str, optional
        The two columns of ``adata.obsm[basis]`` (positions) or obs column
        names (if ``basis`` is None) used as the x and y axes. Default
        ``(0, 1)``.

    Returns
    -------
    numpy.ndarray
        Boolean mask of length ``adata.n_obs``.

    Notes
    -----
    Uses :meth:`matplotlib.path.Path.contains_points`, which, like plotly's
    lasso, applies the even-odd rule to self-intersecting polygons. Plotly
    counts points lying exactly on an edge as inside; here such points may go
    either way. Cells with a NaN coordinate are never selected.
    """
    polygon = np.asarray(polygon, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("polygon must be at least three (x, y) vertices")
    if len(dims) != 2:
        raise ValueError("a polygon needs two dims")
    coords = _coordinates(adata, basis, dims)
    inside = Path(polygon).contains_points(np.nan_to_num(coords))
    return inside & ~np.isnan(coords).any(axis=1)
