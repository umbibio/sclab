from typing import Callable, NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix, issparse
from tqdm.auto import tqdm

from ..utils.interpolate import NDBSpline


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
