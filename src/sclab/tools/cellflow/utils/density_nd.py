from itertools import product
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from scipy.interpolate import BSpline
from sklearn.neighbors import KernelDensity

from .interpolate import fit_smoothing_spline


class DensityResult(NamedTuple):
    kde: KernelDensity
    grid_size: int
    bounds: tuple[tuple[float, float], ...]
    grid: NDArray
    density: NDArray
    scale: float
    periodic: bool


def density_nd(
    data: NDArray,
    bandwidth: float | Literal["scott", "silverman"] | None = None,
    algorithm: Literal["kd_tree", "ball_tree", "auto"] = "auto",
    kernel: str = "gaussian",
    metric: str = "euclidean",
    grid_size: tuple | None = None,
    max_grid_size: int = 2**5 + 1,
    periodic: bool = False,
    bounds: tuple[tuple[float, float], ...] | None = None,
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

    if periodic:
        offsets = np.array(list(product([-1, 0, 1], repeat=ndims)))
        offsets = offsets * np.diff(bounds).T
        dat = np.empty((nsamples * 3**ndims, ndims))
        for i, offset in enumerate(offsets):
            dat[i * nsamples : (i + 1) * nsamples] = data + offset[None, :]
    else:
        dat = data

    if bandwidth is None:
        bandwidth = np.diff(bounds).max() / 64

    kde = KernelDensity(
        bandwidth=bandwidth,
        algorithm=algorithm,
        kernel=kernel,
        metric=metric,
    )
    kde.fit(dat)

    if grid_size is None:
        max_span = np.diff(bounds).max()
        rel_span = np.diff(bounds).flatten() / max_span
        grid_size = tuple((rel_span * max_grid_size).astype(int))

    grid = np.meshgrid(
        *[np.linspace(*b, n) for b, n in zip(bounds, grid_size)], indexing="ij"
    )
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

    return DensityResult(kde, grid_size, bounds, grid, d, scale, periodic)


def fit_density_1d(
    times: NDArray[np.floating],
    t_range: tuple[float, float],
    periodic: bool,
    bandwidth: float | None = None,
    algorithm: str = "auto",
    kernel: str = "gaussian",
    metric: str = "euclidean",
    max_grid_size: int = 2**8 + 1,
    lam: float = 1e-5,
) -> tuple[DensityResult, BSpline]:
    tmin, tmax = t_range
    tspan = tmax - tmin

    times_mask = (tmin <= times) * (times <= tmax)
    times = times[times_mask]

    if bandwidth is None:
        bandwidth = tspan / 64

    rslt = density_nd(
        times.reshape(-1, 1),
        bandwidth=bandwidth,
        algorithm=algorithm,
        kernel=kernel,
        metric=metric,
        max_grid_size=max_grid_size,
        periodic=periodic,
        bounds=(t_range,),
        normalize=True,
    )

    bspl = fit_smoothing_spline(
        rslt.grid[:, 0],
        rslt.density,
        t_range,
        lam=lam,
        periodic=periodic,
    )

    return rslt, bspl
