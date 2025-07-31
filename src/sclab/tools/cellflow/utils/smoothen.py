import logging
from typing import Callable

import numpy as np
from numpy import bool_, floating, integer
from numpy.typing import NDArray
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
PIX2 = 2 * np.pi


def count_empty_intervals(t: NDArray[floating], t_grid: NDArray[floating]) -> int:
    n_data_in_intervals = count_data_in_intervals(t, t_grid)
    empty_intervals_count = np.sum(n_data_in_intervals == 0)
    return empty_intervals_count


def count_data_in_intervals(
    t: NDArray[floating], t_grid: NDArray[floating]
) -> NDArray[integer]:
    t = t.reshape(-1, 1)
    return np.logical_and(t_grid[:-1] <= t, t <= t_grid[1:]).sum(axis=0)


def choose_grid_size(t: NDArray[floating], t_range: tuple[float, float]) -> int:
    grid_size = 2**10
    for _ in range(10):
        t_grid: NDArray[floating] = np.linspace(*t_range, grid_size + 1)
        empty_intervals = count_empty_intervals(t, t_grid)
        if empty_intervals == 0:
            break
        grid_size //= 2
    else:
        raise ValueError("Could not find a suitable grid size")

    return grid_size


def smoothen_data(
    t: NDArray[floating],
    X: NDArray[floating],
    t_range: tuple[float, float] | None = None,
    t_grid: NDArray[floating] | None = None,
    fn: Callable[[NDArray[floating]], NDArray[floating]] = np.average,
    window_width: float | None = None,
    weights: NDArray[floating] | None = None,
    zero_weight: float = 1,
    periodic: bool = False,
    quiet: bool = False,
    progress: bool = False,
) -> NDArray[floating]:
    if t_grid is None:
        # no grid provided. We will have one output point for each input point
        t_grid = t
        is_grid = False
    else:
        # grid is provided
        is_grid = True
        empty_intervals = count_empty_intervals(t, t_grid)
        if empty_intervals > 0 and not quiet:
            logger.warning(f"Provided grid has {empty_intervals} empty intervals")

    if t_range is not None:
        # we used a specific t values range
        tmin, tmax = t_range
    else:
        tmin, tmax = t_grid.min(), t_grid.max()

    # full time window size
    tspan = tmax - tmin

    if window_width is None and not is_grid:
        window_width = tspan * 0.05
    elif window_width is None and is_grid:
        window_width = tspan / (t_grid.size - 1) * 2

    # initialize the output matrix with NaNs
    X_smooth: NDArray[floating] = np.full((t_grid.size,) + X.shape[1:], np.nan)

    generator = enumerate(t_grid)
    if progress:
        generator = tqdm(
            generator,
            total=t_grid.size,
            bar_format="{desc} {percentage:3.0f}%|{bar}|",
            desc="Smoothing data",
        )

    X = X.astype(float)
    eps = np.finfo(float).eps
    for i, m in generator:
        low = m - window_width / 2
        hig = m + window_width / 2

        mask: NDArray[bool_] = (t >= low) & (t <= hig)
        if periodic:
            # include points beyond the periodic boundaries
            mask = (
                mask
                | (t >= low + tspan) & (t <= hig + tspan)
                | (t >= low - tspan) & (t <= hig - tspan)
            )

        if mask.sum() == 0:
            continue

        x = X[mask] + eps
        if fn == np.average and weights is not None:
            w = weights[mask]
            X_smooth[i] = np.average(x, axis=0, weights=w)

        elif fn == np.average and zero_weight == 1:
            X_smooth[i] = np.mean(x, axis=0)

        elif fn == np.average and zero_weight != 1:
            w = np.ones_like(x)
            w[x == eps] = zero_weight + eps
            X_smooth[i] = fn(x, axis=0, weights=w)

        else:
            X_smooth[i] = fn(x, axis=0)

    return X_smooth - eps
