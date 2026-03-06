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

from .utils.density_nd import fit_density_1d
from .utils.times import guess_trange

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
    """Estimate and store a 1-D cell-density profile along pseudotime.

    Fits a kernel density estimate (KDE) to the pseudotime values in
    ``adata.obs[time_key]`` and then smooths the result with a B-spline.
    The fitted spline is stored in ``adata.uns[time_key]['density']`` for
    downstream use by :func:`density_dynamics` and :func:`real_time`.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must contain pseudotime values in
        ``adata.obs[time_key]``.
    time_key : str, optional
        Column in ``adata.obs`` that holds pseudotime values and key under
        which results are stored in ``adata.uns``. Default is ``"pseudotime"``.
    t_range : tuple of float, optional
        ``(t_min, t_max)`` domain for the density estimate. When *None* and
        ``adata.uns[time_key]['t_range']`` exists that value is used;
        otherwise it is guessed from the data. Default is *None*.
    periodic : bool, optional
        Whether pseudotime is periodic (e.g. cell cycle). When *None*,
        inferred from ``adata.uns[time_key]['periodic']`` if available,
        otherwise defaults to *False*. Default is *None*.
    bandwidth : float, optional
        Bandwidth for the KDE, expressed as a fraction of the time range.
        Default is ``1/64``.
    algorithm : str, optional
        Algorithm passed to the KDE back-end. Default is ``"auto"``.
    kernel : str, optional
        Kernel function used for the KDE. Default is ``"gaussian"``.
    metric : str, optional
        Distance metric used for the KDE. Default is ``"euclidean"``.
    max_grid_size : int, optional
        Number of grid points used when evaluating the KDE.
        Default is ``2**8 + 1``.
    plot_density : bool, optional
        If *True*, plot the raw KDE. Default is *False*.
    plot_density_fit : bool, optional
        If *True*, plot the smoothed B-spline fit. Default is *False*.
    plot_density_fit_derivative : bool, optional
        If *True*, plot the derivative of the B-spline fit. Default is *False*.
    plot_histogram : bool, optional
        If *True*, overlay a histogram of the pseudotime values on the plot.
        Default is *False*.
    histogram_nbins : int, optional
        Number of bins used for the histogram. Default is ``50``.

    Returns
    -------
    None
        Modifies *adata* in-place. Results are stored under
        ``adata.uns[time_key]['density']`` as a dict with keys:

        * ``'params'`` — KDE hyper-parameters used for this run.
        * ``'density_bspline_tck'`` — B-spline knots, coefficients, and
          degree (``t``, ``c``, ``k``) as plain Python lists/ints.
    """
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
        from ..utils.density_nd import density_result_1d

        density_result_1d(
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
    """Detect density peaks or valleys along pseudotime via B-spline fitting.

    Fits a KDE to the pseudotime distribution, smooths it with a B-spline,
    optionally takes a derivative of the spline, and identifies peaks (or
    valleys) using :func:`scipy.signal.find_peaks`. Detected peak times,
    heights, and inter-peak durations are stored in ``adata.uns``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must contain pseudotime values in
        ``adata.obs[time_key]`` and ``adata.uns[time_key]['t_range']``.
    time_key : str, optional
        Column in ``adata.obs`` that holds pseudotime values and key under
        which results are stored in ``adata.uns``. Default is ``"pseudotime"``.
    t_range : tuple of float, optional
        ``(t_min, t_max)`` domain for the density estimate. When *None*,
        the value stored in ``adata.uns[time_key]['t_range']`` is used
        (an ``AssertionError`` is raised if that key is absent).
        Default is *None*.
    periodic : bool, optional
        Whether pseudotime is periodic. When *None*, inferred from
        ``adata.uns[time_key]['periodic']`` if available, otherwise *False*.
        Default is *None*.
    bandwidth : float, optional
        Bandwidth for the KDE. Default is ``1/64``.
    algorithm : str, optional
        Algorithm passed to the KDE back-end. Default is ``"auto"``.
    kernel : str, optional
        Kernel function for the KDE. Default is ``"gaussian"``.
    metric : str, optional
        Distance metric for the KDE. Default is ``"euclidean"``.
    max_grid_size : int, optional
        Number of grid points for KDE evaluation. Default is ``2**8 + 1``.
    derivative : int, optional
        Order of the B-spline derivative to analyse. ``0`` analyses the
        density itself; ``1`` analyses its rate of change, etc.
        Default is ``0``.
    mode : {"peaks", "valleys"}, optional
        Whether to detect peaks or valleys in the (derivative of the)
        density. Default is ``"peaks"``.
    find_peaks_kwargs : dict, optional
        Extra keyword arguments forwarded to
        :func:`scipy.signal.find_peaks`. The ``'height'`` key, if present,
        is treated as a *fraction* of the global maximum and rescaled
        accordingly. Default is ``{}``.
    plot_density : bool, optional
        If *True*, plot the raw KDE. Default is *False*.
    plot_density_fit : bool, optional
        If *True*, plot the smoothed B-spline fit. Default is *False*.
    plot_density_fit_derivative : bool, optional
        If *True*, plot the derivative of the B-spline. Default is *False*.
    plot_histogram : bool, optional
        If *True*, overlay a histogram on the plot. Default is *False*.
    histogram_nbins : int, optional
        Number of histogram bins. Default is ``50``.

    Returns
    -------
    None
        Modifies *adata* in-place. Results are stored under
        ``adata.uns[time_key][f'density_dynamics_d{derivative}_{mode}']``
        as a dict with keys:

        * ``'times'`` — pseudotime positions of detected peaks.
        * ``'deltas'`` — inter-peak durations (or phase durations for
          periodic data).
        * ``'heights'`` — density (or derivative) values at each peak.
        * ``'params'`` — KDE and peak-finding hyper-parameters.
        * ``'density_bspline_tck'`` — B-spline representation of the
          fitted density.
    """
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
        from .utils.density_nd import density_result_1d

        ax = density_result_1d(
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
    """Convert pseudotime to real time by normalising for cell-cycle density.

    Fits a density profile along pseudotime (via :func:`density`) and then
    maps each cell's pseudotime to a real-time value by integrating the
    inverse of the density curve (area-under-curve normalisation). This
    corrects for non-uniform sampling across the trajectory so that equal
    real-time intervals contain proportionally equal numbers of cells.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must contain pseudotime values in
        ``adata.obs[pseudotime_key]``.
    pseudotime_key : str, optional
        Column in ``adata.obs`` with pseudotime values. Default is
        ``"pseudotime"``.
    pseudotime_t_range : tuple of float, optional
        ``(t_min, t_max)`` domain of the pseudotime axis. When *None*,
        inferred from the data via :func:`density`. Default is *None*.
    periodic : bool, optional
        Whether pseudotime is periodic. When *None*, inferred from
        ``adata.uns[pseudotime_key]['periodic']`` if available, otherwise
        *False*. Default is *None*.
    key_added : str, optional
        Column in ``adata.obs`` and key in ``adata.uns`` under which the
        real-time values and metadata are stored. Default is ``"real_time"``.
    tmax : float, optional
        Maximum real-time value (upper bound of the output axis). Cells at
        the very end of the trajectory are mapped to this value.
        Default is ``100``.
    units : {"minutes", "hours", "days", "percent"}, optional
        Interpretive label for the real-time axis; stored in
        ``adata.uns[key_added]['t_units']`` but does not affect the
        computation. Default is ``"percent"``.
    bandwidth : float, optional
        Bandwidth for the KDE. Default is ``1/64``.
    algorithm : str, optional
        Algorithm passed to the KDE back-end. Default is ``"auto"``.
    kernel : str, optional
        Kernel function for the KDE. Default is ``"gaussian"``.
    metric : str, optional
        Distance metric for the KDE. Default is ``"euclidean"``.
    max_grid_size : int, optional
        Number of grid points for KDE evaluation. Default is ``2**8 + 1``.
    plot_density : bool, optional
        If *True*, plot the raw KDE. Default is *False*.
    plot_density_fit : bool, optional
        If *True*, plot the smoothed B-spline fit. Default is *False*.
    plot_density_fit_derivative : bool, optional
        If *True*, plot the derivative of the B-spline. Default is *False*.
    plot_histogram : bool, optional
        If *True*, overlay a histogram on the plot. Default is *False*.
    histogram_nbins : int, optional
        Number of histogram bins. Default is ``50``.

    Returns
    -------
    None
        Modifies *adata* in-place:

        * ``adata.obs[key_added]`` — real-time values for each cell. Cells
          outside ``pseudotime_t_range`` are assigned *NaN*.
        * ``adata.uns[key_added]`` — dict containing fitting parameters,
          the B-spline TCK representation, ``'tmax'``, ``'t_range'``,
          ``'t_units'``, and ``'periodic'``.
    """
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
    """Compute cumulative area under a density B-spline up to each pseudotime.

    Integrates the density B-spline from the origin to each pseudotime value
    using :func:`scipy.integrate.cumulative_trapezoid`, then normalises so
    that the total integral equals *tmax*. This maps pseudotime to real time.

    Parameters
    ----------
    pseudotimes : NDArray[floating]
        Pseudotime values at which the cumulative integral is evaluated.
    tmax : float
        Total real time; the integral is normalised to this value so the
        output lies in ``[0, tmax]``.
    tck_dict : dict
        B-spline representation of the density curve as returned by
        :func:`density`. Must contain keys ``'t'`` (knots), ``'c'``
        (coefficients), and ``'k'`` (degree).

    Returns
    -------
    NDArray[floating]
        Real-time values corresponding to *pseudotimes*, shape
        ``(len(pseudotimes),)``.
    """
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
    """Convert pseudotime values to real time using a pre-fitted density model.

    Applies the same area-under-curve mapping used by :func:`real_time` to
    an arbitrary array of pseudotime values, using the B-spline density model
    already stored in ``adata.uns[realtime_key]``.

    Parameters
    ----------
    pseudotimes : NDArray[floating]
        Pseudotime values to convert.
    adata : AnnData
        Annotated data matrix containing a previously computed real-time
        model in ``adata.uns[realtime_key]``.
    realtime_key : str, optional
        Key in ``adata.uns`` where the fitted density model is stored.
        Default is ``"real_time"``.

    Returns
    -------
    NDArray[floating]
        Real-time values corresponding to *pseudotimes*, shape
        ``(len(pseudotimes),)``.
    """
    tmax = adata.uns[realtime_key]["tmax"]
    tck_dict = adata.uns[realtime_key]["density_bspline_tck"]

    return _area_under_curve(pseudotimes, tmax, tck_dict)


def get_pseudotimes(
    realtimes: NDArray[floating], adata: AnnData, realtime_key: str = "real_time"
):
    """Convert real-time values back to pseudotime using a pre-fitted density model.

    Inverts the real-time mapping by constructing a cubic interpolator over
    the pseudotime-to-real-time curve stored in ``adata.uns[realtime_key]``
    and evaluating it at each requested real-time value.

    Parameters
    ----------
    realtimes : NDArray[floating]
        Real-time values to convert back to pseudotime.
    adata : AnnData
        Annotated data matrix containing a previously computed real-time
        model in ``adata.uns[realtime_key]``.
    realtime_key : str, optional
        Key in ``adata.uns`` where the fitted density model is stored.
        Default is ``"real_time"``.

    Returns
    -------
    NDArray[floating]
        Pseudotime values corresponding to *realtimes*, shape
        ``(len(realtimes),)``.
    """
    tmax = adata.uns[realtime_key]["tmax"]
    tck_dict = adata.uns[realtime_key]["density_bspline_tck"]
    pseudotime_t_range = adata.uns[realtime_key]["params"]["pseudotime_t_range"]

    x = np.linspace(*pseudotime_t_range, 1001)
    y = _area_under_curve(x, tmax, tck_dict)
    interpolator = interp1d(y, x, kind="cubic")

    return interpolator(realtimes)
