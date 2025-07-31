import logging
from typing import Callable

import numpy as np
from numpy import asarray, ascontiguousarray, floating, prod
from numpy import empty as np_empty
from numpy import float64 as np_float64
from numpy.typing import NDArray
from scipy.fft import fft, fftfreq
from scipy.interpolate import BSpline, _fitpack_impl, make_smoothing_spline
from tqdm.auto import tqdm

from .smoothen import choose_grid_size, count_data_in_intervals, smoothen_data

try:
    from scipy.interpolate._dierckx import evaluate_spline
except ImportError:
    from scipy.interpolate._bspl import evaluate_spline


logger = logging.getLogger(__name__)
PIX2 = 2 * np.pi


def fit_smoothing_spline(
    x: NDArray[floating],
    y: NDArray[floating],
    t_range: tuple[float, float],
    w: NDArray[floating] | None = None,
    lam: float | None = None,
    periodic: bool = False,
    n_reps: int = 3,
) -> BSpline:
    if periodic:
        assert n_reps % 2 == 1

    o = np.argsort(x)
    x, y = x[o], y[o]
    if w is not None:
        w = w[o]

    tmin, tmax = t_range
    tspan = tmax - tmin

    if periodic:
        mask = np.logical_and((x >= tmin), (x < tmax))
    else:
        mask = np.logical_and((x >= tmin), (x <= tmax))

    x, y = x[mask], y[mask]
    if w is not None:
        w = w[mask]
    n = x.size

    if periodic:
        xx = np.concatenate([x + i * tspan for i in range(n_reps)])
        yy = np.tile(y, n_reps)
        ww = np.tile(w, n_reps) if w is not None else None
    else:
        xx = x
        yy = y
        ww = w

    bspl = make_smoothing_spline(xx, yy, ww, lam)
    t, c, k = bspl.tck

    if periodic:
        N = n_reps // 2
        t = t - tspan * N
        t = t[n * N : -n * N + 1]
        c = c[n * N : -n * N + 1]

    return BSpline(t, c, k)


class NDFourier:
    def __init__(
        self,
        xh: NDArray[floating] | None = None,
        freq: NDArray[floating] | None = None,
        t_range: tuple[float, float] | None = None,
        grid_size: int | None = None,
        periodic: bool = True,
        largest_harmonic: int = 5,
        d: int = 0,
        zero_weight: float = 1.0,
        smoothing_fn: Callable = np.average,
    ) -> None:
        assert periodic
        assert t_range is not None
        assert t_range[0] == 0

        self.tmin, self.tmax = self.t_range = t_range
        self.tscale = PIX2 / self.tmax

        if xh is not None:
            assert freq is not None
            self.n = grid_size + 1
            self.xh = xh.reshape((xh.shape[0], -1, 1)).copy()
            self.freq = freq.reshape((freq.shape[0], -1, 1)).copy()
            self.scaled_freq = 1j * self.freq * self.tscale

        self.grid_size = grid_size
        self.periodic = periodic
        self.largest_harmonic = largest_harmonic
        self.d = d
        self.zero_weight = zero_weight
        self.smoothing_fn = smoothing_fn

    def fit(
        self,
        t: NDArray[floating],
        X: NDArray[floating],
    ) -> "NDFourier":
        if self.grid_size is None:
            self.grid_size = choose_grid_size(t, self.t_range)

        t_grid = np.linspace(*self.t_range, self.grid_size + 1)
        self.X_smooth = smoothen_data(
            t,
            X,
            t_range=self.t_range,
            t_grid=t_grid,
            periodic=self.periodic,
            zero_weight=self.zero_weight,
            fn=self.smoothing_fn,
        )

        self.n = n = self.X_smooth.shape[0]
        self.X_smooth = self.X_smooth.reshape((n, -1))

        xh: NDArray[floating] = fft(self.X_smooth, axis=0)
        freq: NDArray[floating] = fftfreq(n, d=1 / n)

        mask = np.abs(freq) <= self.largest_harmonic
        xh = xh[mask]
        freq = freq[mask]

        self.xh = xh.reshape((xh.shape[0], -1, 1))
        self.freq = freq.reshape((freq.shape[0], -1, 1))
        self.scaled_freq = 1j * self.freq * self.tscale

        return self

    def derivative(self, d=1) -> "NDFourier":
        return NDFourier(
            self.xh,
            self.freq,
            self.t_range,
            self.grid_size,
            self.periodic,
            self.largest_harmonic,
            d + self.d,
        )

    def __getitem__(self, key) -> "NDFourier":
        return NDFourier(
            self.xh[:, key],
            self.freq,
            self.t_range,
            self.grid_size,
            self.periodic,
            self.largest_harmonic,
            self.d,
        )

    def __call__(self, x: NDArray[floating], d=0) -> NDArray[floating]:
        x = asarray(x)
        x_shape = x.shape

        x = ascontiguousarray(x.ravel(), dtype=np_float64)

        d = d + self.d
        out: NDArray[floating] = np.real(
            (self.xh * self.scaled_freq**d * np.exp(self.scaled_freq * x)).sum(axis=0)
            / self.n
        )
        out = out.T
        out = out.reshape(x_shape + (self.xh.shape[1],))

        return out


class NDBSpline:
    def __init__(
        self,
        t: NDArray[floating] | None = None,
        C: NDArray[floating] | None = None,
        k: int | None = None,
        t_range: tuple[float, float] | None = None,
        grid_size: int | None = None,
        periodic: bool = False,
        roughness: float | None = None,
        zero_weight: float = 1.0,
        window_width: float | None = None,
        use_grid: bool = True,
        weight_grid: bool = False,
        smoothing_fn: Callable = np.average,
    ) -> None:
        if periodic:
            assert t_range is not None
            assert t_range[0] == 0

        if t is not None or C is not None or k is not None:
            assert t is not None
            assert C is not None
            assert k is not None
            self.t = t.copy()
            self.C = C.reshape((C.shape[0], -1)).copy()
            self.k = k

        if t_range is not None:
            self.tmin, self.tmax = self.t_range = t_range

        self.grid_size = grid_size
        self.periodic = periodic
        self.window_width = window_width
        self.use_grid = use_grid
        self.weight_grid = weight_grid
        self.roughness = roughness
        self.zero_weight = zero_weight
        self.smoothing_fn = smoothing_fn

    def fit(
        self,
        t: NDArray[floating],
        X: NDArray[floating],
        progress: bool = False,
    ) -> "NDBSpline":
        X = X.reshape((X.shape[0], -1))
        if self.t_range is None:
            self.tmin, self.tmax = self.t_range = t.min(), t.max()

        if self.grid_size is None:
            self.grid_size = choose_grid_size(t, self.t_range)

        if self.roughness is None:
            self.roughness = 1

        if self.use_grid:
            t_grid: NDArray[floating] = np.linspace(*self.t_range, self.grid_size + 1)
            self.lam = 1 / self.grid_size / 10**self.roughness
        else:
            t_grid = None
            self.lam = 1 / 10**self.roughness
        self.X_smooth = smoothen_data(
            t,
            X,
            t_range=self.t_range,
            t_grid=t_grid,
            periodic=self.periodic,
            window_width=self.window_width,
            zero_weight=self.zero_weight,
            progress=progress,
            fn=self.smoothing_fn,
        )

        if t_grid is not None and self.weight_grid:
            w = np.zeros(self.X_smooth.shape[0], dtype=float)
            n = count_data_in_intervals(t, t_grid) + 1
            if self.periodic:
                n = np.append(n, n[0])
            else:
                n = np.append(n, n[-1])
            w[n > 1] = 1 / np.log1p(n[n > 1])
        else:
            w = None

        iterator = self.X_smooth.T
        if progress:
            iterator = tqdm(
                iterator,
                bar_format="{desc} {percentage:3.0f}%|{bar}|",
                desc="Fitting bsplines",
            )

        fit_t_range = (0, 1)
        fit_t_grid = np.linspace(0, 1, self.grid_size + 1)
        fit_t = (t - self.tmin) / (self.tmax - self.tmin)
        C = []
        for x in iterator:
            f = fit_smoothing_spline(
                fit_t_grid if self.use_grid else fit_t,
                x,
                t_range=fit_t_range,
                w=w,
                lam=self.lam,
                periodic=self.periodic,
            )

            C.append(f.c)

        self.t = f.t.copy()
        self.t *= self.tmax - self.tmin
        self.t += self.tmin
        self.C = np.array(C).T.copy()
        self.k = 3

        return self

    def derivative(self, d: int = 1) -> "NDBSpline":
        # pad the c array if needed
        ct = len(self.t) - len(self.C)
        if ct > 0:
            self.C = np.r_[self.C, np.zeros((ct, self.C.shape[1]))]
        t, C, k = _fitpack_impl.splder((self.t, self.C, self.k), d)
        return NDBSpline(t, C, k, self.t_range, self.grid_size, self.periodic)

    def __getitem__(self, key) -> "NDBSpline":
        t = self.t
        C = self.C[:, key]
        k = self.k
        return NDBSpline(t, C, k, self.t_range, self.grid_size, self.periodic)

    def __call__(self, x: NDArray[floating], d: int = 0) -> NDArray[floating]:
        x = asarray(x)
        x_shape = x.shape

        x = ascontiguousarray(x.ravel(), dtype=np_float64)
        if self.periodic:
            n = self.t.size - self.k - 1
            x = self.t[self.k] + (x - self.t[self.k]) % (self.t[n] - self.t[self.k])

        out = np_empty((len(x), prod(self.C.shape[1:])), dtype=self.C.dtype)

        if not self.t.flags.c_contiguous:
            self.t = self.t.copy()
        if not self.C.flags.c_contiguous:
            self.C = self.C.copy()

        evaluate_spline(self.t, self.C, self.k, x, d, False, out)
        out = out.reshape(x_shape + (self.C.shape[1],))

        return out
