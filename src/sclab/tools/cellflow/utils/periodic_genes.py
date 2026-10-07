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
    import scanpy as sc

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
    sc.pp.normalize_total(aggregated, target_sum=1e4)
    log_cnts = np.log1p(aggregated.X)
    profiles = pd.DataFrame(log_cnts, index=labels, columns=aggregated.var_names)
    ps = power_spectrum_df(profiles, window="boxcar", detrend="constant")
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


def power_spectrum_df(
    X: pd.DataFrame,
    window: str = "boxcar",
    detrend: str = "constant",
):
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


def periodic_genes_cosinor(
    adata: AnnData,
    time_key: str,
    tmin: float | None = None,
    tmax: float | None = None,
    n: int = 10,
    period: float | None = None,
    n_harmonics: int = 1,
    alpha: float = 0.05,
    layer: str | None = None,
):
    """
    Detect periodic genes via cosinor regression on pseudobulked profiles.

    Model per gene:
        y(t) = c + sum_{k=1..K} [ A_k cos(k*omega*t) + B_k sin(k*omega*t) ] + eps
    where omega = 2*pi / period. K = n_harmonics. Default period = tmax - tmin.
    """
    import scanpy as sc
    from scipy.stats import f as f_dist
    from scipy.stats import false_discovery_control

    if tmin is None:
        tmin = adata.obs[time_key].min().round(2)

    if tmax is None:
        tmax = adata.obs[time_key].max().round(2)

    if period is None:
        period = tmax - tmin

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
    sc.pp.normalize_total(aggregated, target_sum=1e4)
    log_cnts = np.log1p(aggregated.X)
    profiles = pd.DataFrame(log_cnts, index=labels, columns=aggregated.var_names)

    # Bin centers as floats; survives bin dropouts from aggregate_and_filter
    t = profiles.index.astype(float).values  # (n_bins,)
    Y = profiles.values  # (n_bins, n_genes)
    n_bins, n_genes = Y.shape

    # Design matrix: intercept + (cos, sin) per harmonic — shared across genes
    omega = 2 * np.pi / period
    cols = [np.ones(n_bins)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(k * omega * t))
        cols.append(np.sin(k * omega * t))
    Xd = np.column_stack(cols)  # (n_bins, 1 + 2K)

    # Vectorized OLS: one solve, all genes at once
    beta = np.linalg.solve(Xd.T @ Xd, Xd.T @ Y)  # (1 + 2K, n_genes)
    Y_hat = Xd @ beta  # (n_bins, n_genes)

    # Variance decomposition
    RSS = ((Y - Y_hat) ** 2).sum(axis=0)
    TSS = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        R2 = np.where(TSS > 0, 1.0 - RSS / TSS, 0.0)

    # F-test of the joint null "all harmonic coefficients = 0"
    df_num = 2 * n_harmonics
    df_den = n_bins - (1 + 2 * n_harmonics)
    if df_den <= 0:
        raise ValueError(
            f"Not enough bins (n={n_bins}) for {n_harmonics} harmonic(s); "
            f"need at least {2 * n_harmonics + 2}."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (R2 / df_num) / ((1.0 - R2) / df_den)
        F = np.where(np.isfinite(F), F, 0.0)
    pvalue = f_dist.sf(F, df_num, df_den)
    padj = false_discovery_control(pvalue)

    # Fundamental harmonic: amplitude and acrophase
    A1, B1 = beta[1], beta[2]
    amplitude = np.hypot(A1, B1)
    phase = np.arctan2(B1, A1) % (2 * np.pi)  # peak phase in radians
    peak_time = ((phase / omega) - tmin) % period + tmin

    # Per-harmonic amplitudes (useful when n_harmonics > 1)
    harmonic_amps = np.empty((n_harmonics, n_genes))
    for k in range(n_harmonics):
        harmonic_amps[k] = np.hypot(beta[1 + 2 * k], beta[2 + 2 * k])

    # Store results
    adata.varm["profile"] = profiles.T
    adata.varm["cosinor_fit"] = pd.DataFrame(
        Y_hat, index=labels, columns=aggregated.var_names
    ).T
    adata.varm["harmonic_amplitudes"] = pd.DataFrame(
        harmonic_amps.T,
        index=aggregated.var_names,
        columns=[f"h{k + 1}" for k in range(n_harmonics)],
    )
    adata.var["cosinor_intercept"] = beta[0]
    adata.var["cosinor_amplitude"] = amplitude
    adata.var["cosinor_phase"] = phase
    adata.var["cosinor_peak_time"] = peak_time
    adata.var["cosinor_r_squared"] = R2
    adata.var["cosinor_pvalue"] = pvalue
    adata.var["cosinor_padj"] = padj
    adata.var["periodic"] = padj <= alpha
