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
