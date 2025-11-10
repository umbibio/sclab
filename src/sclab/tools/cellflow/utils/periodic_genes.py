import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from scipy.signal import get_window, periodogram
from scipy.sparse import spmatrix

from sclab.tools.utils import aggregate_and_filter


def periodic_genes(
    adata: AnnData,
    time_key: str,
    tmin: float,
    tmax: float,
    period: float,
    n: int,
    min_pct_power_below: float = 0.75,
    layer: str | None = None,
):
    times = adata.obs[time_key].values.copy()
    if layer is None or layer == "X":
        X = adata.X
    else:
        X = adata.layers[layer]

    _assert_integer_counts(X)

    tmp_adata = AnnData(X, obs=adata.obs[[time_key]], var=adata.var[[]])

    w = (tmax - tmin) / n
    bins = np.arange(-w / 2 + tmin, tmax, w)
    labels = list(map(lambda x: f"{x:.2f}", bins[:-1] + w / 2))

    times[times >= bins.max()] = times[times >= bins.max()] - tmax
    tmp_adata.obs["timepoint"] = pd.cut(times, bins=bins, labels=labels)
    aggregated = aggregate_and_filter(
        tmp_adata,
        "timepoint",
        replicas_per_group=1,
        make_stats=False,
        make_dummies=False,
    )
    log_cnts = np.log1p(aggregated.X)
    profiles = pd.DataFrame(log_cnts, index=labels, columns=aggregated.var_names)
    ps = power_spectrum_df(profiles)
    pp = pct_power_below(ps, 1 / period)

    adata.varm["profile"] = profiles.T
    adata.varm["periodogram"] = ps.T
    adata.var["pct_power_below"] = pp
    adata.var["periodic"] = pp > min_pct_power_below


def _assert_integer_counts(X: spmatrix | NDArray):
    message = "Periodic genes requires raw integer counts. E.g. `layer = 'counts'`."
    if isinstance(X, spmatrix):
        assert all(X.data % 1 == 0), message
    else:
        assert all(X % 1 == 0), message


def infer_dt_from_index(idx: pd.Index) -> float:
    # Works for numeric or datetime indexes
    if isinstance(idx, pd.DatetimeIndex):
        dt = np.median(np.diff(idx.view("i8"))) / 1e9  # seconds
    else:
        dt = float(np.median(np.diff(idx.values.astype(float))))
    return dt


def power_spectrum_df(X: pd.DataFrame, window: str = "hann", detrend: str = "constant"):
    # X: rows=timepoints, columns=variables
    Xd = X - X.mean()  # remove DC so percent computations are stable
    dt = infer_dt_from_index(X.index) if X.index.size > 1 else 1.0
    fs = 1.0 / dt
    win = get_window(window, X.shape[0], fftbins=True)

    # Build a tidy dataframe of periodograms for all columns
    out = {}
    for c in Xd.columns:
        f, Pxx = periodogram(
            Xd[c].values,
            fs=fs,
            window=win,
            detrend=detrend,
            scaling="spectrum",  # integrates to variance
            return_onesided=True,
        )
        out[c] = Pxx
    ps = pd.DataFrame(out, index=pd.Index(f, name="frequency"))
    return ps  # units: (data units)^2, integrates (sum * df) to variance per column


def pct_power_below(ps: pd.DataFrame, max_freq: float) -> pd.Series:
    # ps is spectrum from power_spectrum_df (one-sided, DC included but we demeaned)
    # Compute integrals via the rectangle rule: sum * df (df = freq spacing)
    if len(ps.index) < 2:
        return pd.Series({c: np.nan for c in ps.columns}, name="pct_power_at_low_freq")
    df = ps.index[1] - ps.index[0]
    mask_low = ps.index <= max_freq
    num: pd.Series = ps.loc[mask_low].sum() * df
    den: pd.Series = ps.sum() * df
    s = num / den
    s.name = "pct_power_at_low_freq"
    return s
