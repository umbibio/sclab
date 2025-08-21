import logging
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from numpy import floating
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import BSpline, interp1d
from scipy.signal import find_peaks

from ..utils.density_nd import fit_density_1d
from ..utils.times import guess_trange

logger = logging.getLogger(__name__)


def density(
    adata: AnnData,
    time_key: str = "pseudotime",
    t_range: tuple[float, float] | None = None,
    periodic: bool | None = None,
    bandwidth: float = 1 / 64,
    algorithm: str = "auto",
    kernel: str = "gaussian",
    metric: str = "euclidean",
    max_grid_size: int = 2**8 + 1,
    plot_density: bool = False,
    plot_density_fit: bool = False,
    plot_density_fit_derivative: bool = False,
    plot_histogram: bool = False,
    histogram_nbins: int = 50,
):
    if t_range is None and time_key in adata.uns:
        # using stored t_range
        t_range = adata.uns[time_key]["t_range"]
    else:
        # guessing t_range
        pts = adata.obs[time_key].values
        pts = pts[np.isfinite(pts)]
        t_range = guess_trange(pts)
        if pts.size < 500:
            logger.warning(
                "Guessing t_range may not be accurate for fewer than 500 points."
                " Consider setting the pseudotime_t_range parameter instead."
            )

    if periodic is None and time_key in adata.uns and "periodic" in adata.uns[time_key]:
        periodic = adata.uns[time_key]["periodic"]
    else:
        periodic = False

    times = adata.obs[time_key].values
    lam = 1 / max_grid_size / 1e4
    rslt, bspl = fit_density_1d(
        times=times,
        t_range=t_range,
        periodic=periodic,
        bandwidth=bandwidth,
        algorithm=algorithm,
        kernel=kernel,
        metric=metric,
        max_grid_size=max_grid_size,
        lam=lam,
    )

    if time_key not in adata.uns:
        adata.uns[time_key] = {
            "t_range": list(t_range),
            "periodic": periodic,
        }

    t, c, k = bspl.tck
    density_bspline_tck = dict(t=t.tolist(), c=c.tolist(), k=k)
    adata.uns[time_key].update(
        {
            "density": {
                "params": {
                    "bandwidth": bandwidth,
                    "algorithm": algorithm,
                    "kernel": kernel,
                    "metric": metric,
                    "max_grid_size": max_grid_size,
                },
                "density_bspline_tck": density_bspline_tck,
            }
        }
    )

    if plot_density | plot_density_fit | plot_density_fit_derivative | plot_histogram:
        from ..utils import plot

        plot.density_result_1d(
            rslt,
            data=times[~np.isnan(times)],
            density_fit_lam=lam,
            plot_density=plot_density,
            plot_density_fit=plot_density_fit,
            plot_density_fit_derivative=plot_density_fit_derivative,
            plot_histogram=plot_histogram,
            histogram_nbins=histogram_nbins,
            show=True,
        )


def density_dynamics(
    adata: AnnData,
    time_key: str = "pseudotime",
    t_range: tuple[float, float] | None = None,
    periodic: bool | None = None,
    bandwidth: float = 1 / 64,
    algorithm: str = "auto",
    kernel: str = "gaussian",
    metric: str = "euclidean",
    max_grid_size: int = 2**8 + 1,
    derivative: int = 0,
    mode: Literal["peaks", "valleys"] = "peaks",
    find_peaks_kwargs: dict = {},
    plot_density: bool = False,
    plot_density_fit: bool = False,
    plot_density_fit_derivative: bool = False,
    plot_histogram: bool = False,
    histogram_nbins: int = 50,
):
    if t_range is None:
        test = time_key in adata.uns and "t_range" in adata.uns[time_key]
        assert test, f"t_range must be provided for time_key: {time_key}"
        t_range = adata.uns[time_key]["t_range"]

    if periodic is None:
        if time_key in adata.uns and "periodic" in adata.uns[time_key]:
            periodic = adata.uns[time_key]["periodic"]
        else:
            periodic = False

    times = adata.obs[time_key].values
    lam = 1 / max_grid_size / 1e4
    rslt, bspl = fit_density_1d(
        times=times,
        t_range=t_range,
        periodic=periodic,
        bandwidth=bandwidth,
        algorithm=algorithm,
        kernel=kernel,
        metric=metric,
        max_grid_size=max_grid_size,
        lam=lam,
    )

    t = np.linspace(*t_range, 2**16 + 1)
    y = bspl.derivative(derivative)(t)
    if mode == "peaks":
        pass
    elif mode == "valleys":
        y = -y

    tmin, tmax = t_range
    tspan = tmax - tmin
    if periodic:
        tt = np.concatenate([t[:-1] + i * tspan for i in range(3)]) - tspan
        yy = np.tile(y[:-1], 3)
    else:
        tt = t
        yy = y

    peak_height = find_peaks_kwargs.pop("height", 0.0)
    peak_height = peak_height * y.max()

    peaks, _ = find_peaks(yy, height=peak_height, **find_peaks_kwargs)
    peak_times = tt[peaks]
    peak_heights = yy[peaks]

    peaks_mask = np.logical_and(peak_times >= tmin, peak_times < tmax)
    peak_times = peak_times[peaks_mask]
    peak_heights = peak_heights[peaks_mask]

    timepoints = peak_times - tmin
    if periodic:
        deltas = (timepoints - np.roll(timepoints, 1)) % tspan
    else:
        timepoints = np.insert(timepoints, 0, 0)
        deltas = timepoints[1:] - timepoints[:-1]

    if time_key not in adata.uns:
        adata.uns[time_key] = {}

    t, c, k = bspl.tck
    density_bspline_tck = dict(t=t.tolist(), c=c.tolist(), k=k)
    adata.uns[time_key].update(
        {
            f"density_dynamics_d{derivative}_{mode}": {
                "times": peak_times,
                "deltas": deltas,
                "heights": peak_heights,
                "params": {
                    "bandwidth": bandwidth,
                    "algorithm": algorithm,
                    "kernel": kernel,
                    "metric": metric,
                    "max_grid_size": max_grid_size,
                    "find_peaks_kwargs": {"height": peak_height, **find_peaks_kwargs},
                },
                "density_bspline_tck": density_bspline_tck,
            }
        }
    )

    if plot_density | plot_density_fit | plot_density_fit_derivative | plot_histogram:
        from ..utils import plot

        ax = plot.density_result_1d(
            rslt,
            data=times[~np.isnan(times)],
            density_fit_lam=lam,
            plot_density=plot_density,
            plot_density_fit=plot_density_fit,
            plot_density_fit_derivative=plot_density_fit_derivative,
            plot_histogram=plot_histogram,
            histogram_nbins=histogram_nbins,
            show=False,
        )
        for t in peak_times:
            ax.axvline(t, color="k", linestyle="--")
        plt.show()


def real_time(
    adata: AnnData,
    pseudotime_key: str = "pseudotime",
    pseudotime_t_range: tuple[float, float] | None = None,
    periodic: bool | None = None,
    key_added: str = "real_time",
    tmax: float = 100,
    units: Literal["minutes", "hours", "days", "percent"] = "percent",
    bandwidth: float = 1 / 64,
    algorithm: str = "auto",
    kernel: str = "gaussian",
    metric: str = "euclidean",
    max_grid_size: int = 2**8 + 1,
    plot_density: bool = False,
    plot_density_fit: bool = False,
    plot_density_fit_derivative: bool = False,
    plot_histogram: bool = False,
    histogram_nbins: int = 50,
):
    density(
        adata,
        time_key=pseudotime_key,
        t_range=pseudotime_t_range,
        periodic=periodic,
        bandwidth=bandwidth,
        algorithm=algorithm,
        kernel=kernel,
        metric=metric,
        max_grid_size=max_grid_size,
        plot_density=plot_density,
        plot_density_fit=plot_density_fit,
        plot_density_fit_derivative=plot_density_fit_derivative,
        plot_histogram=plot_histogram,
        histogram_nbins=histogram_nbins,
    )

    time_key_uns = adata.uns[pseudotime_key]
    # density function sets appropriate t_range and periodic parameters if missing
    pseudotime_t_range = time_key_uns["t_range"]
    periodic = time_key_uns["periodic"]
    # density_bspline_tck is computed in density function
    density_bspline_tck = time_key_uns["density"]["density_bspline_tck"]

    pt_min, pt_tmax = pseudotime_t_range
    pseudotimes = adata.obs[pseudotime_key].values
    pt_mask = (pt_min <= pseudotimes) * (pseudotimes <= pt_tmax)
    pseudotimes = pseudotimes[pt_mask]

    rt = _area_under_curve(pseudotimes, tmax, density_bspline_tck)

    adata.obs[key_added] = np.nan
    adata.obs.loc[pt_mask, key_added] = rt

    adata.uns[key_added] = {
        "params": {
            "pseudotime_key": pseudotime_key,
            "pseudotime_t_range": pseudotime_t_range,
            "tmax": tmax,
            "units": units,
            "bandwidth": bandwidth,
            "algorithm": algorithm,
            "kernel": kernel,
            "metric": metric,
            "max_grid_size": max_grid_size,
        },
        "density_bspline_tck": density_bspline_tck,
        "tmax": tmax,
        "t_range": [0.0, tmax],
        "t_units": units,
        "periodic": periodic,
    }


def _area_under_curve(
    pseudotimes: NDArray[floating], tmax: float, tck_dict: dict[str, list[float] | int]
):
    bspl = BSpline(**tck_dict)

    # the normalized flux should be 1 / tmax
    q = 1.0 / tmax

    # we will use cumulative_trapezoid to calculate the integral
    # we should make sure that we have enough points to get a good approximation
    # we will use 1000 extra points evenly distributed between 0 and 1 to
    # fill the gaps, and make sure to remove them after the calculation
    n = 1000
    x = np.concatenate([pseudotimes, np.linspace(1 / n, 1, n)])

    # cumulative_trapezoid requires the x values to be sorted
    o = np.argsort(x)
    # we will need to sort the result back to the original order
    oo = np.argsort(o)

    # we need to insert 0 at the beginning, this defines the starting point
    # of the integral
    x = np.insert(x[o], 0, 0)
    d = bspl(x)

    return cumulative_trapezoid(d, x)[oo][:-n] / q


def get_realtimes(
    pseudotimes: NDArray[floating], adata: AnnData, realtime_key: str = "real_time"
):
    tmax = adata.uns[realtime_key]["tmax"]
    tck_dict = adata.uns[realtime_key]["density_bspline_tck"]

    return _area_under_curve(pseudotimes, tmax, tck_dict)


def get_pseudotimes(
    realtimes: NDArray[floating], adata: AnnData, realtime_key: str = "real_time"
):
    tmax = adata.uns[realtime_key]["tmax"]
    tck_dict = adata.uns[realtime_key]["density_bspline_tck"]
    pseudotime_t_range = adata.uns[realtime_key]["params"]["pseudotime_t_range"]

    x = np.linspace(*pseudotime_t_range, 1001)
    y = _area_under_curve(x, tmax, tck_dict)
    interpolator = interp1d(y, x, kind="cubic")

    return interpolator(realtimes)
