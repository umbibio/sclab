import logging
from typing import Callable, NamedTuple, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix, issparse
from tqdm.auto import tqdm

from ..utils.interpolate import NDBSpline

logger = logging.getLogger(__name__)


def periodic_sliding_window(
    data: NDArray, t: NDArray, window_size: int, fn: Callable[[NDArray], NDArray]
) -> NDArray:
    ws = window_size + ((window_size - 1) % 2)
    window_shape = (ws,) + (1,) * (data.ndim - 1)

    o = np.argsort(t)
    oo = np.argsort(o)

    d = data[o]
    dd = [*d[-ws // 2 + 1 :], *d, *d[: ws // 2]]

    windows = sliding_window_view(dd, window_shape=window_shape).squeeze()
    return fn(windows, axis=-1)[oo]


def equalization(
    times: NDArray,
    t_range: tuple[float, float],
    max_bins: int = 200,
    iterations: int = 1e4,
    tolerance: float = 0.02,
) -> NDArray:
    if not isinstance(times, np.ndarray):
        raise TypeError("times must be a numpy array")

    if times.ndim != 1:
        raise ValueError("times must be a 1D array")

    t_min, t_max = t_range
    t_span = t_max - t_min

    # for sorting the values
    o = np.argsort(times)
    # and recovering the original order
    oo = np.argsort(o)

    alpha = 0.1
    scale_offset = 1

    rng = np.random.default_rng()
    scaled_times = times.copy()

    for n_bins in tqdm(np.arange(25, max_bins + 1, 25)):
        for it in range(int(iterations)):
            bins = np.linspace(t_min, t_max, n_bins + 1)
            bins[1:-1] += rng.normal(0, t_span / n_bins / 100, bins[1:-1].size)
            counts, _ = np.histogram(scaled_times, bins=bins)
            tmp: NDArray = counts / counts.max()
            rms = np.sqrt(np.mean((tmp - tmp.mean()) ** 2))
            if rms < tolerance:
                break

            scales = counts / counts.max() * alpha + scale_offset

            t = scaled_times[o]
            tt = []
            i = 0
            timepoint = 0.0
            for start, end, scale in zip(bins[:-1], bins[1:], scales):
                bin_size = end - start
                new_size = bin_size * scale
                while i < t.size and t[i] < end:
                    new_t = (t[i] - start) * scale + timepoint
                    tt.append(new_t)
                    i += 1
                timepoint += new_size

            tt = np.array(tt)
            scaled_times = tt[oo] / timepoint * t_span + t_min

        else:
            cnts_mean, cnts_max, cnts_min = counts.mean(), counts.max(), counts.min()
            print(
                f"Failed to converge. RMS: {rms}. "
                + f"({cnts_mean=:.2f}, {cnts_max=:.2f}, {cnts_min=:.2f})"
            )

    return scaled_times


def fit_trends(
    X: NDArray | csr_matrix,
    times: NDArray,
    t_range: tuple[float, float],
    periodic: bool,
    grid_size: int = 128,
    roughness: float | None = None,
    zero_weight: float = 0.5,
    window_width: float | None = None,
    n_timesteps: int | None = None,
    timestep_delta: float | None = None,
    progress: bool = True,
) -> None:
    if issparse(X):
        X = np.ascontiguousarray(X.todense())

    tmin, tmax = t_range

    mask = ~np.isnan(times)
    t = times[mask]
    X = X[mask]

    F = NDBSpline(
        grid_size=grid_size,
        t_range=t_range,
        periodic=periodic,
        zero_weight=zero_weight,
        roughness=roughness,
        window_width=window_width,
    )
    F.fit(t, X, progress=progress)

    eps = np.finfo(float).eps
    SNR: NDArray
    SNR = F(t).var(axis=0) / (X.var(axis=0) + eps)
    SNR = SNR / SNR.max()

    # x = np.linspace(*t_range, 10001)[:-1]
    # peak_time = x[np.argmax(F(x), axis=0)]

    if n_timesteps is not None and timestep_delta is not None:
        raise ValueError("Cannot specify both n_timesteps and timestep_delta")
    elif n_timesteps is None and timestep_delta is None:
        # default
        x = np.linspace(*t_range, 101)
    elif n_timesteps is not None:
        x = np.linspace(*t_range, n_timesteps)
    elif timestep_delta is not None:
        x = np.arange(tmin, tmax + timestep_delta, timestep_delta)

    Y = F(x)

    return x, Y


class SinglePeakResult(NamedTuple):
    times: NDArray
    heights: NDArray
    scores: NDArray
    info: NDArray


def find_single_peaks(
    X: NDArray,
    t: NDArray,
    t_range: tuple[float, float] = (0, 1),
    grid_size: int = 512,
    periodic: bool = True,
    zero_weight: float = 0.2,
    roughness: float = 2,
    n_timesteps: int = 201,
    width_range: tuple[float, float] = (0, 100),
    score_threshold: float = 2.5,
    progress: bool = True,
) -> tuple[NDArray, NDArray]:
    X = X / np.percentile(X + 1, 99, axis=0, keepdims=True)
    x, Y = fit_trends(
        X,
        t,
        t_range=t_range,
        periodic=periodic,
        grid_size=grid_size,
        zero_weight=zero_weight,
        roughness=roughness,
        n_timesteps=n_timesteps,
        progress=progress,
    )

    peak_times = np.full(X.shape[1], np.nan)
    peak_heights = np.full(X.shape[1], np.nan)
    peak_scores = np.full(X.shape[1], np.nan)
    peak_info_data = [{}] * X.shape[1]

    idx_sequence = range(X.shape[1])
    if progress:
        idx_sequence = tqdm(idx_sequence)

    for i in idx_sequence:
        y = Y[:, i]
        k, info = find_peaks(y, prominence=0.05, width=width_range, height=0)
        m = np.median(y)
        s = y[k] / m
        k = k[s > score_threshold]
        if len(k) == 1:
            peak_times[i] = x[k]
            peak_heights[i] = y[k]
            peak_scores[i] = np.log2(s[0])
            peak_info_data[i] = info

    return SinglePeakResult(peak_times, peak_heights, peak_scores, peak_info_data)


def piecewise_scaling(
    times: NDArray,
    t_range: tuple[float, float],
    start: float,
    end: float,
    new_end: float,
) -> NDArray:
    tmin, tmax = t_range

    times_pws = np.full(times.shape, np.nan)

    mask = (times >= tmin) & (times < start)
    times_pws[mask] = times[mask]

    mask = (times >= start) & (times < end)
    times_pws[mask] = (times[mask] - start) / (end - start) * (new_end - start) + start

    mask = (times >= end) & (times < tmax)
    times_pws[mask] = (times[mask] - end) / (tmax - end) * (tmax - new_end) + new_end

    return times_pws


def rescale_pseudotime(
    times: Sequence | NDArray,
    transitions: Sequence | NDArray,
    durations: Sequence | NDArray,
    t_range: tuple[float, float] | None = None,
    periodic: bool = False,
) -> NDArray:
    """Rescale pseudotime to real-time based on known durations.

    Parameters
    ----------
    times
        Pseudotime values to rescale.
    transitions
        Transition points between categories.
    durations
        Durations of each category.
    t_range
        Range of pseudotime (e.g., (0, 1)). If None, inferred from times.
    periodic
        Whether the trajectory is periodic.

    Returns
    -------
    Rescaled real-time values.
    """
    t = np.asarray(times).copy()
    trans = np.sort(np.asarray(transitions))
    durs = np.asarray(durations)

    if t_range is None:
        tmin, tmax = np.nanmin(t), np.nanmax(t)
        inferred_tmax = True
    else:
        tmin, tmax = t_range
        inferred_tmax = False

    if periodic:
        if inferred_tmax:
            raise ValueError("tmax must be specified (via t_range) for periodic scaling.")
        if len(durs) != len(trans):
            raise ValueError(
                f"Number of durations must be {len(trans)} for periodic scaling "
                f"(one for each interval defined by N transitions)."
            )

        # For periodic, we start the scale at trans[-1]
        # Shift all times so trans[-1] is at 0
        shift = trans[-1]
        t_s = (t - shift) % tmax
        trans_s = (trans - shift) % tmax
        # trans_s will be [trans[0]-shift, trans[1]-shift, ..., 0]
        # Sorted it looks like: [0, t0, t1, ..., t_{n-2}]
        trans_s = np.sort(trans_s)

        rescaled = np.full_like(t, np.nan)
        current_time = 0.0

        # Intervals are [trans_s[i], trans_s[i+1]]
        # The first interval [0, trans_s[1]] corresponds to durs[0]
        # Wait, if we shift by trans[-1], the last interval was [trans[-1], trans[0]] wrapping around.
        # Let's be more precise. If trans = [0.2, 0.5, 0.8]
        # Interval 0: [0.8, 0.2] -> duration 0
        # Interval 1: [0.2, 0.5] -> duration 1
        # Interval 2: [0.5, 0.8] -> duration 2

        # Shift by trans[last] (0.8):
        # 0.8 -> 0
        # 0.2 -> 0.4
        # 0.5 -> 0.7
        # trans_s = [0, 0.4, 0.7]

        for i in range(len(trans_s)):
            t_start = trans_s[i]
            t_end = trans_s[i + 1] if i + 1 < len(trans_s) else tmax
            duration = durs[i]

            mask = (t_s >= t_start) & (t_s < t_end)
            rescaled[mask] = (t_s[mask] - t_start) / (t_end - t_start) * duration + current_time
            current_time += duration

        return rescaled

    else:
        if len(durs) != len(trans) + 1:
            raise ValueError(
                f"Number of durations must be {len(trans) + 1} for sequential scaling "
                f"(one for each of the N+1 intervals defined by N transitions)."
            )

        rescaled = np.full_like(t, np.nan)
        current_time = 0.0

        # Boundary points for intervals
        boundaries = [tmin] + list(trans) + [tmax]

        for i in range(len(boundaries) - 1):
            t_start = boundaries[i]
            t_end = boundaries[i + 1]
            duration = durs[i]

            mask = (t >= t_start) & (t <= t_end)
            if t_end > t_start:
                rescaled[mask] = (t[mask] - t_start) / (t_end - t_start) * duration + current_time
            else:
                rescaled[mask] = current_time
            current_time += duration

        return rescaled


def find_category_transitions(
    times: Sequence,
    labels: Sequence,
    categories: Sequence,
    periodic: bool = False,
    tmax: float | None = None,
) -> NDArray:
    """Find transition points between categories in pseudotime.

    Parameters
    ----------
    times
        Pseudotime values for each sample.
    labels
        Category labels for each sample.
    categories
        Ordered list of categories defining the trajectory.
    periodic
        Whether the trajectory is periodic.
    tmax
        Maximum pseudotime value. Required if periodic=True.

    Returns
    -------
    Array of transition points.
    """
    if periodic and tmax is None:
        raise ValueError("tmax must be specified for periodic trajectories.")

    t = np.asarray(times).copy()
    labels = np.asarray(labels)
    categories = np.asarray(categories)

    # Use 1.0 as default tmax if not provided for non-periodic
    if tmax is None:
        tmax = np.nanmax(t)

    transitions = []

    # Number of transitions to find
    n_transitions = len(categories) if periodic else len(categories) - 1

    for i in range(n_transitions):
        cat1 = categories[i]
        cat2 = categories[(i + 1) % len(categories)]

        mask1 = labels == cat1
        mask2 = labels == cat2

        if not np.any(mask1) or not np.any(mask2):
            logger.warning(f"Category {cat1} or {cat2} has no samples. Skipping.")
            transitions.append(np.nan)
            continue

        t1 = t[mask1]
        t2 = t[mask2]

        if periodic:

            def loss(x, t1_shifted, t2_shifted):
                return np.abs(np.quantile(t2_shifted, x) - np.quantile(t1_shifted, 1 - x))

            # Shift time so that the transition is away from the boundary.
            # We use the midpoint between the medians of cat1 and cat2.
            m1 = np.quantile(t1, 0.5)
            m2 = np.quantile(t2, 0.5)

            # Periodic distance
            d = (m2 - m1) % tmax
            # Midpoint in periodic space
            mid = (m1 + d / 2) % tmax
            shift = mid - tmax / 2

            t1_s = (t1 - shift) % tmax
            t2_s = (t2 - shift) % tmax

            res = minimize_scalar(
                loss, bounds=(0, 0.5), method="bounded", args=(t1_s, t2_s)
            )
            ti = (np.quantile(t1_s, 1 - res.x) + np.quantile(t2_s, res.x)) / 2
            ti = (ti + shift) % tmax
        else:
            # Sequential case
            if np.nanmax(t1) <= np.nanmin(t2):
                ti = (np.nanmax(t1) + np.nanmin(t2)) / 2
            else:

                def loss(x):
                    return np.abs(np.quantile(t2, x) - np.quantile(t1, 1 - x))

                res = minimize_scalar(loss, bounds=(0, 0.5), method="bounded")
                ti = (np.quantile(t1, 1 - res.x) + np.quantile(t2, res.x)) / 2

        transitions.append(ti)

    return np.asarray(transitions)


def piecewise_rescale(
    adata: "AnnData",
    time_key: str,
    groupby: str,
    durations: list[float] | dict[str, float],
    new_key: str = "real_time",
    periodic: bool = False,
    t_range: tuple[float, float] | None = None,
) -> None:
    """Rescale pseudotime to real-time using piecewise linear mapping.

    Parameters
    ----------
    adata
        Annotated data matrix.
    time_key
        Key in `adata.obs` for pseudotime.
    groupby
        Key in `adata.obs` for categorical labels used to define intervals.
    durations
        Durations for each interval. If a list, must match number of intervals.
        If a dictionary, must map category labels to durations.
    new_key
        Key in `adata.obs` to store the rescaled real-time values.
    periodic
        Whether the trajectory is periodic.
    t_range
        Range of pseudotime. If None, inferred from `adata.obs[time_key]`.
    """
    times = adata.obs[time_key].values
    labels = adata.obs[groupby]

    if not isinstance(labels.dtype, pd.CategoricalDtype):
        raise TypeError(f"Column '{groupby}' must be categorical.")

    categories = labels.cat.categories

    # Convert durations to a list in the order of categories
    if isinstance(durations, dict):
        durs_list = [durations[cat] for cat in categories]
    else:
        durs_list = durations

    # Estimate transition points
    transitions = find_category_transitions(
        times=times,
        labels=labels.values,
        categories=categories,
        periodic=periodic,
        tmax=t_range[1] if t_range else None,
    )

    # Perform rescaling
    real_times = rescale_pseudotime(
        times=times,
        transitions=transitions,
        durations=durs_list,
        t_range=t_range,
        periodic=periodic,
    )

    adata.obs[new_key] = real_times
