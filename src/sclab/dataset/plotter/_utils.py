from functools import lru_cache
from itertools import product
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy import float64
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from scipy.interpolate import BSpline, make_smoothing_spline
from sklearn.neighbors import KernelDensity


def make_periodic_smoothing_spline(
    x: NDArray[float64],
    y: NDArray[float64],
    t_range: tuple[float, float],
    w: NDArray[float64] | None = None,
    lam: float | None = None,
    n_reps: int = 5,
) -> BSpline:
    assert n_reps % 2 == 1

    o = np.argsort(x)
    x, y = x[o], y[o]

    tmin, tmax = t_range
    tspan = tmax - tmin

    mask = np.logical_and((x >= tmin), (x < tmax))
    x, y = x[mask], y[mask]
    n = x.size

    xx = np.concatenate([x + i * tspan for i in range(n_reps)])
    yy = np.tile(y, n_reps)
    ww = np.tile(w, n_reps) if w is not None else None
    bspl = make_smoothing_spline(xx, yy, ww, lam)
    t, c, k = bspl.tck

    N = n_reps // 2
    t = t - tspan * N
    t = t[n * N : -n * N + 1]
    c = c[n * N : -n * N + 1]

    return BSpline(t, c, k)


class DensityResult(NamedTuple):
    kde: KernelDensity
    grid_size: tuple
    bounds: tuple
    grid: NDArray
    density: NDArray
    scale: float


def _density_nd(
    data: NDArray,
    bandwidth: float | Literal["scott", "silverman"] | None = None,
    bandwidth_factor: float = 1,
    algorithm: Literal["kd_tree", "ball_tree", "auto"] = "auto",
    kernel: str = "gaussian",
    metric: str = "euclidean",
    grid_size: tuple | None = None,
    max_grid_size: int = 2**5 + 1,
    periodic: bool = False,
    bounds: tuple[tuple[float, float]] | None = None,
    normalize: bool = False,
) -> DensityResult:
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    nsamples, ndims = data.shape
    if bounds is None:
        assert not periodic, "bounds must be specified if periodic=True"
        lower, upper = data.min(axis=0), data.max(axis=0)
        span = upper - lower
        margins = span / 10
        bounds = tuple(zip(lower - margins, upper + margins))
    assert len(bounds) == ndims, "must provide bounds for each dimension"
    bounds = np.array(bounds)

    if periodic:
        offsets = np.array(list(product([-1, 0, 1], repeat=ndims)))
        offsets = offsets * np.diff(bounds).T
        dat = np.empty((nsamples * 3**ndims, ndims))
        for i, offset in enumerate(offsets):
            dat[i * nsamples : (i + 1) * nsamples] = data + offset[None, :]
    else:
        dat = data
    dat = (dat - bounds.min(axis=1)) / (bounds.max(axis=1) - bounds.min(axis=1))

    if bandwidth is None:
        bandwidth = bandwidth_factor

    kde = KernelDensity(
        bandwidth=bandwidth,
        algorithm=algorithm,
        kernel=kernel,
        metric=metric,
    )
    kde.fit(dat)

    if grid_size is None:
        grid_size = (max_grid_size, max_grid_size)

    grid = np.meshgrid(*[np.linspace(0, 1, n) for n in grid_size], indexing="ij")
    grid = np.vstack([x.ravel() for x in grid]).T
    d = np.exp(kde.score_samples(grid))

    if normalize and ndims == 1:
        scale = trapezoid(d, grid.reshape(-1))
    elif normalize:
        # perform simple Riemmann sum for higher dimensions
        deltas = np.diff(bounds).T / (np.array(grid_size) - 1)
        tmp = d.reshape(grid_size).copy()
        for i, s in enumerate(grid_size):
            # take left corners for the sum
            tmp = tmp.take(np.arange(s - 1), axis=i)
        scale = tmp.sum() * np.prod(deltas)
    else:
        scale = 1

    d /= scale

    grid = (grid * (bounds.max(axis=1) - bounds.min(axis=1))) + bounds.min(axis=1)

    return DensityResult(kde, grid_size, bounds, grid, d, scale)


@lru_cache
def _make_density_heatmap(
    data: tuple[tuple[float, float]],
    bandwidth_factor: float,
    grid_resolution: int,
    line_smoothing: float,
    contours: int,
    color: str = "orange",
):
    data = np.array(data)
    result = _density_nd(
        data,
        bandwidth_factor=bandwidth_factor,
        max_grid_size=2**grid_resolution + 1,
    )
    nx, ny = result.grid_size
    X: NDArray = result.grid.reshape(result.grid_size + (2,))
    D: NDArray = result.density.reshape(result.grid_size)
    x = X[:nx, 0, 0]
    y = X[0, :ny, 1]
    z = D.clip(min=0).T

    start = z.min() + 1e-9
    end = z.max() + 1e-9
    size = (end - start) / contours
    contours = dict(start=start, end=end, size=size)

    return go.Contour(
        z=z,
        x=x,
        y=y,
        showscale=False,
        colorscale=["white", color],
        zmin=0,
        line_smoothing=line_smoothing,
        contours=contours,
    )


def _get_color_sequence():
    """Get a list of color names that are distinguishable by redmean distance."""

    import plotly.colors as pc
    from scipy.spatial.distance import pdist, squareform

    color_ids = []
    color_sequence = []
    for scale in [
        "D3",
        "Plotly",
        "G10",
        "T10",
        "Alphabet",
        "Dark24",
        "Light24",
        "Set1",
        "Pastel1",
        "Dark2",
        "Set2",
        "Pastel2",
        "Set3",
        "Antique",
        "Bold",
        "Pastel",
        "Prism",
        "Safe",
        "Vivid",
    ]:
        colors = getattr(pc.qualitative, scale)
        color_ids.extend([f"{scale}_{i}" for i in range(len(colors))])
        color_sequence.extend(colors)
    banned = [
        (211, 211, 211),  # lightgray - used for missing values
    ]
    color_ids = np.array(color_ids)
    X = np.array(
        [
            pc.hex_to_rgb(c) if c.startswith("#") else pc.unlabel_rgb(c)
            for c in color_sequence
        ],
        dtype=int,
    )
    color_sequence = np.array([pc.label_rgb(c) for c in X])

    def redmean(c1, c2):
        # https://en.wikipedia.org/wiki/Color_difference#sRGB
        r1, g1, b1 = c1
        r2, g2, b2 = c2
        rm = (r1 + r2) / 2
        dr, dg, db = r1 - r2, g1 - g2, b1 - b2
        return np.sqrt(
            (2 + rm / 256) * dr**2 + 4 * dg**2 + (2 + (255 - rm) / 256) * db**2
        )

    D = squareform(pdist(X, redmean))
    np.fill_diagonal(D, np.inf)

    mindist = 65

    mask = (np.array([[redmean(b, c) for b in banned] for c in X]) > mindist).all(
        axis=1
    )
    for i, d in enumerate(D):
        mask[i] *= (d[: i + 1][mask[: i + 1]] > mindist).all()

    return color_sequence[mask].tolist()


COLOR_DISCRETE_SEQUENCE = _get_color_sequence()


def Rx(degs: float):
    """
    Rotate a 3D coordinate system around its x-axis by the given angle (in degrees).
    The rotation is counter-clockwise when viewed from the positive x-axis.
    The returned matrix is a 3x3 numpy array, which can be used to transform
    3-element numpy vectors or arrays.

    Parameters
    ----------
    degs : float
        Angle of rotation in degrees.

    Returns
    -------
    NDArray
        3x3 rotation matrix as a right operating matrix.

    Examples
    --------
    >>> rotated_X = X @ Rx(45)
    """
    rads = np.pi * degs / 180
    c, s = np.cos(rads), np.sin(rads)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]).T


def Ry(degs: float):
    """
    Rotate a 3D coordinate system around its y-axis by the given angle (in degrees).
    The rotation is counter-clockwise when viewed from the positive y-axis.
    The returned matrix is a 3x3 numpy array, which can be used to transform
    3-element numpy vectors or arrays.

    Parameters
    ----------
    degs : float
        Angle of rotation in degrees.

    Returns
    -------
    NDArray
        3x3 rotation matrix as a right operating matrix.

    Examples
    --------
    >>> rotated_X = X @ Ry(45)
    """
    rads = np.pi * degs / 180
    c, s = np.cos(rads), np.sin(rads)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]).T


def Rz(degs: float):
    """
    Rotate a 3D coordinate system around its z-axis by the given angle (in degrees).
    The rotation is counter-clockwise when viewed from the positive z-axis.
    The returned matrix is a 3x3 numpy array, which can be used to transform
    3-element numpy vectors or arrays.

    Parameters
    ----------
    degs : float
        Angle of rotation in degrees.

    Returns
    -------
    NDArray
        3x3 rotation matrix as a right operating matrix.

    Examples
    --------
    >>> rotated_X = X @ Rz(45)
    """

    rads = np.pi * degs / 180
    c, s = np.cos(rads), np.sin(rads)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T


def Rxyz(alpha, beta, gamma):
    return Rz(gamma) @ Ry(beta) @ Rx(alpha)


def rotate_xyz(
    X: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
):
    """
    Rotate the data in X by alpha, beta, and gamma degrees around the x, y, and z axes,
    respectively, and return the rotated data.

    Parameters
    ----------
    X : pd.DataFrame
        3D data to be rotated
    alpha : float
        angle in degrees to rotate around the x axis
    beta : float
        angle in degrees to rotate around the y axis
    gamma : float
        angle in degrees to rotate around the z axis

    Returns
    -------
    pd.DataFrame
        rotated data
    """
    #
    colnames = X.columns
    X = X @ Rxyz(alpha, beta, gamma)
    X.columns = colnames

    return X


def rotate_multiple_steps(
    X: pd.DataFrame,
    steps: str,
):
    """
    Rotate the data in X according to a sequence of steps.

    Parameters
    ----------
    X : pd.DataFrame
        3D data to be rotated
    steps : str
        string of comma-separated "axis:angle" pairs, where axis is in {"x", "y", "z"}
        and angle is in degrees

    Returns
    -------
    pd.DataFrame
        rotated data
    """
    colnames = X.columns

    # remove spaces
    steps = steps.replace(" ", "")

    # replace all separators with newlines
    for sep in ",;":
        steps = steps.replace(sep, "\n")

    # remove key:value assignment tokens
    for chr in ":=":
        steps = steps.replace(chr, "")

    # get the list of steps
    steps_list = steps.split("\n")

    for step in steps_list:
        if len(step) < 2:
            continue

        step = step.strip()
        axis = step[0].lower()

        if axis not in {"x", "y", "z"}:
            break

        try:
            angle = float(step[1:])
        except ValueError:
            break

        match axis:
            case "x":
                X = X @ Rx(angle)
            case "y":
                X = X @ Ry(angle)
            case "z":
                X = X @ Rz(angle)
            case _:
                break

    X.columns = colnames
    return X


def rotate_and_project_traces(
    X: pd.DataFrame,
    figure_data: list[go.Scatter | go.Scattergl | go.Scatter3d],
    alpha: float,
    beta: float,
    gamma: float,
):
    X = (Rxyz(alpha, beta, gamma) @ X.T).T
    for trace in figure_data:
        marker_ids = trace.hovertext
        if not isinstance(marker_ids, NDArray | list):
            continue
        trace.x, trace.y = X.loc[marker_ids].values[:, :2].T
