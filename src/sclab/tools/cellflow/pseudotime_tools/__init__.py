from typing import Literal, NamedTuple

import numpy as np
from numpy import bool_, floating
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from tqdm.auto import tqdm

from .._pseudotime._timeseries import periodic_sliding_window
from ..utils.interpolate import NDBSpline, NDFourier

_2PI = 2 * np.pi


class PseudotimeResult(NamedTuple):
    pseudotime: NDArray[floating]
    residues: NDArray[floating]
    phi: NDArray[floating]
    F: NDFourier | NDBSpline
    SNR: NDArray[floating]
    snr_mask: NDArray[bool_]
    t_mask: NDArray[bool_]
    fp_resolution: float


def periodic_parameter(data: NDArray[floating]) -> NDArray[floating]:
    x, y = data.T.astype(float)
    return np.arctan2(y, x) % _2PI


def _compute_pseudotime(
    t: NDArray[floating],
    X: NDArray[floating],
    t_range: tuple[float, float],
    n_dims: int = 10,
    min_snr: float = 0.25,
    periodic: bool = False,
    method: Literal["fourier", "splines"] = "splines",
    largest_harmonic: int = 5,
    roughness: float | None = None,
    progress: bool = True,
) -> PseudotimeResult:
    if not periodic:
        assert method == "splines"

    tmin, tmax = t_range
    tspan = tmax - tmin

    if periodic:
        assert tmin == 0.0

    match method:
        case "fourier":
            F = NDFourier(t_range=t_range, largest_harmonic=largest_harmonic)
        case "splines":
            F = NDBSpline(t_range=t_range, periodic=periodic, roughness=roughness)
        case _:
            raise ValueError(
                f'{method} is not a valid fitting method. Choose one of: "fourier", "splines"'
            )

    t_mask = (tmin <= t) * (t <= tmax)
    t = t[t_mask]
    X = X[t_mask]

    if periodic:
        M = periodic_sliding_window(X, t, 50, np.median)
    else:
        M = X

    # we fit an n-dimensional curve to the data
    F.fit(t, M)

    # we use the signal-to-noise ratio to assess which dimensions show a strong signal
    # we only keep dimensions with some signal through the initial ordering t
    SNR: NDArray[floating] = F(t).var(axis=0) / X.var(axis=0)
    SNR = SNR / SNR.max()
    snr_mask = SNR > min_snr

    dim_mask = np.arange(X.shape[1]) < n_dims

    # we remove noisy dimensions
    X = X[:, snr_mask & dim_mask]
    # `NDFourier` and `NDBSpline` objects can be sliced like so
    full_F = F
    F = F[snr_mask & dim_mask]

    # we will find the closest points in the curve for each data point in X
    # we do this in stages using euclidean distance
    # after each stage we increase the numeric precision
    n = 100
    m = 10
    k = 10

    # T is a matrix of timepoints
    # dim 0 has resolution 0.01
    # dim 1 has resolution 0.0001
    T = (
        np.linspace(tmin, tmax, n + 1)[:-1, None]
        + np.linspace(0, tspan / n, m + 1)[None]
    )
    # evaluate the curve points
    Y = F(T)

    # for each point, we find which row in T has the closest point to the curve
    closest_order_1 = np.argmin(
        np.linalg.norm(
            X[None] - Y[:, [m // 2]],
            axis=2,
        ),
        axis=0,
    )

    # for each point, we find which column in T has the closest point to the curve
    closest_order_2 = np.argmin(
        np.linalg.norm(
            X[:, None] - Y[closest_order_1],
            axis=2,
        ),
        axis=1,
    )

    # we obtain the corresponding pseudotime ordering
    phi = T[closest_order_1, closest_order_2]

    # so far our pseudotime estimation has resolution 0.0001
    # we can refine it to match the floating point resolution of the data's dtype
    fp_res = np.finfo(X.dtype).resolution
    res = 1 / n / m

    n_iters = int(np.floor(np.log10(res) - np.log10(fp_res)))
    range_obj = range(n_iters)
    if progress:
        range_obj = tqdm(range_obj, bar_format="{percentage:3.0f}%|{bar}|")

    for _ in range_obj:
        # we create a new matrix of timepoints
        T = phi[:, None] + np.linspace(-tspan * res / 2, tspan * res / 2, k + 1)

        # make sure we didn't go over the range
        T = T.clip(*t_range)

        # and evaluate the curve points
        Y = F(T)

        # for each point, we find which column in T has the closest point to the curve
        closest_order_3 = np.argmin(
            np.linalg.norm(
                X[:, None] - Y,
                axis=2,
            ),
            axis=1,
        )

        # we obtain the corresponding pseudotime ordering with the current resolution
        phi = T[np.arange(X.shape[0]), closest_order_3]
        # update the current resolution
        res = res / k

    # # converts to unit vectors. returns an array of shape (n_points, n_dims)
    # def unit(v):
    #     return v / np.linalg.norm(v, axis=-1, keepdims=True)

    # # cosine of the angle between the vector from the curve to the data point
    # # and the tangent vector to the curve at the closest point
    # def rv_cosine(p):
    #     R = unit(X - F(p))
    #     V = unit(F(p, d=1))
    #     C = (V * R).sum(axis=-1)
    #     return C

    # cosine_mask = np.abs(rv_cosine(phi)) < 0.01
    # t_mask[t_mask] = cosine_mask
    # phi = phi[cosine_mask]
    # X = X[cosine_mask]

    if periodic:
        phi = phi % tspan

    residues = np.linalg.norm(X - F(phi), axis=-1)

    # speed returns an array of shape (n_points,)
    def speed(t):
        return np.linalg.norm(F(t, d=1), axis=-1)

    # # arclen returns a scalar
    # def arclen(t):
    #     return quad(speed, tmin, t, limit=500, epsrel=1.49e-6)[0]

    # we will use cumulative_trapezoid to calculate the integral
    # we should make sure that we have enough points to get a good approximation
    # we will use 1000 extra points evenly distributed between 0 and 1 to
    # fill in the gaps, and make sure to remove them after the calculation
    n = 1_000
    x = np.concatenate([phi, np.linspace(tmin + 1 / n, tmax, n)])

    o = np.argsort(x)
    oo = np.argsort(o)
    x = np.insert(x[o], 0, tmin)
    integral = cumulative_trapezoid(speed(x), x=x)
    pseudotime: NDArray[floating] = integral[oo][:-n] / integral.max()

    return PseudotimeResult(
        pseudotime,
        residues,
        phi,
        full_F,
        SNR,
        snr_mask,
        t_mask,
        fp_res,
    )
