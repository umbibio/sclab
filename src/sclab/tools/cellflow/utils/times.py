import itertools

import numpy as np
from numpy import floating
from numpy.typing import NDArray


def guess_trange(
    times: NDArray[floating], verbose: bool = False
) -> tuple[float, float]:
    tmin, tmax = times.min(), times.max()
    tspan = tmax - tmin

    scale = 10.0 ** np.ceil(np.log10(tspan)) / 100
    tspan = np.ceil(tspan / scale) * scale

    scale = 10.0 ** np.ceil(np.log10(tspan)) / 100
    g_tmin = np.floor(tmin / scale) * scale
    g_tmax = np.ceil(tmax / scale) * scale

    g_tmin = 0.0 if g_tmin == -0.0 else g_tmin
    g_tmax = 0.0 if g_tmax == -0.0 else g_tmax

    if verbose:
        print(
            f"tspan: {tspan:10.4f}    min-max: {tmin:10.4f} - {tmax:10.4f} | {g_tmin:>8} - {g_tmax:>8}"
        )

    return g_tmin, g_tmax


def test_guess_trange(N: int = 1000, verbose: bool = False) -> None:
    def _test1(trange: tuple[float, float]) -> bool:
        tmin, tmax = trange
        tspan = tmax - tmin
        g_tmin, g_tmax = guess_trange(np.random.uniform(*trange, N))
        err_min = np.abs(g_tmin - tmin) / tspan
        err_max = np.abs(g_tmax - tmax) / tspan
        return err_min <= 0.01 and err_max <= 0.01

    scales1 = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    scales2 = [1, 2, 3, 5, 7]
    for s1, s2 in itertools.product(scales1, scales2):
        scale = s1 * s2
        for lw, hg in [(-2, -1), (-1 / 2, 1 / 2), (1, 2)]:
            trange = lw * scale, hg * scale
            acc1 = np.mean([_test1(trange) for _ in range(500)])
            if verbose:
                print(
                    f"scale: {scale: 9.3f} | lw-hg: {lw: 5.1f} - {hg: 5.1f} | {acc1: 8.2%}"
                )
            else:
                assert acc1 > 0.95, (
                    f"scale: {scale: 9.3f} | lw-hg: {lw: 5.1f} - {hg: 5.1f} | {acc1: 8.2%}"
                )
