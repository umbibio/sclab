import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.signal import get_window, periodogram

from sclab.tools.utils import aggregate_and_filter
from sclab.utils import check_raw_counts

MIN_CELLS_PER_BIN = 30


def periodic_genes(
    adata: AnnData,
    time_key: str,
    tmin: float,
    tmax: float,
    period: float,
    n: int,
    min_pct_power_below: float = 0.75,
    layer: str = "counts",
):
    """
    Detect periodic genes via the periodogram of pseudobulked profiles.

    Cells are binned by ``time_key`` into ``n`` equally spaced bins covering
    ``[tmin, tmax)`` (wrapping around), pseudobulked, normalized and
    log-transformed. Requires raw counts: ``layer`` must hold non-negative
    integers (use ``layer="X"`` for ``adata.X``). Bins with fewer than 30 cells
    are dropped; since the periodogram assumes equally spaced samples, this
    raises a ``ValueError`` suggesting a smaller ``n``.
    """
    profiles, all_labels = _binned_log_profiles(adata, time_key, tmin, tmax, n, layer)

    present = set(profiles.index)
    missing = [label for label in all_labels if label not in present]
    if missing:
        raise ValueError(
            f"Bins {missing} have fewer than {MIN_CELLS_PER_BIN} cells and were "
            "dropped, but the periodogram requires equally spaced bins. "
            "Try a smaller `n`."
        )

    ps = power_spectrum_df(profiles, window="boxcar", detrend="constant")
    pp = pct_power_below(ps, 1 / period)

    adata.varm["profile"] = profiles.T
    adata.varm["periodogram"] = ps.T
    adata.var["pct_power_below"] = pp
    adata.var["periodic"] = pp > min_pct_power_below


def _binned_log_profiles(
    adata: AnnData,
    time_key: str,
    tmin: float,
    tmax: float,
    n: int,
    layer: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Pseudobulk cells into ``n`` circular time bins and log-normalize.

    Returns the profiles (surviving bins x genes, in bin order, indexed by
    bin-center label) and the labels of all ``n`` bins.
    """
    import scanpy as sc

    if not tmax > tmin:
        raise ValueError(f"tmax ({tmax}) must be greater than tmin ({tmin}).")

    if layer == "X":
        X = adata.X
    elif layer in adata.layers:
        X = adata.layers[layer]
    else:
        raise ValueError(
            f"layer '{layer}' not found; pass layer='X' if adata.X holds raw counts"
        )

    check_raw_counts(
        X,
        name=f"layer '{layer}'",
        hint="Periodic gene detection needs raw counts, e.g. layer='counts'.",
    )

    period = tmax - tmin
    w = period / n
    edges = np.linspace(tmin - w / 2, tmax - w / 2, n + 1)
    all_labels = [f"{x:.2f}" for x in edges[:-1] + w / 2]

    times = np.asarray(adata.obs[time_key].values, dtype=float)
    finite = np.isfinite(times)
    t = times[finite]
    t = tmin + (t - tmin) % period
    t = np.where(t >= tmax - w / 2, t - period, t)

    # right-closed bins (edges[i], edges[i + 1]]; clip guards float round-off
    bin_idx = np.clip(np.searchsorted(edges, t, side="left") - 1, 0, n - 1)

    counts = np.bincount(bin_idx, minlength=n)
    if not (counts >= MIN_CELLS_PER_BIN).any():
        raise ValueError(
            f"No time bin has at least {MIN_CELLS_PER_BIN} cells; "
            "try a smaller `n` or check `time_key`."
        )

    tmp_adata = AnnData(
        X[finite],
        obs=adata.obs.loc[finite, [time_key]].copy(),
        var=adata.var[[]].copy(),
    )
    tmp_adata.obs["timepoint"] = pd.Categorical(
        np.asarray(all_labels)[bin_idx], categories=all_labels
    )

    aggregated = aggregate_and_filter(
        tmp_adata,
        "timepoint",
        replicas_per_group=1,
        min_cells_per_group=MIN_CELLS_PER_BIN,
        make_stats=False,
        make_dummies=False,
    )
    sc.pp.normalize_total(aggregated, target_sum=1e4)
    log_cnts = np.log1p(aggregated.X)

    surviving = aggregated.obs["timepoint"].astype(str).tolist()
    profiles = pd.DataFrame(log_cnts, index=surviving, columns=aggregated.var_names)
    profiles = profiles.loc[[label for label in all_labels if label in surviving]]
    return profiles, all_labels


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
    layer: str = "counts",
):
    """
    Detect periodic genes via cosinor regression on pseudobulked profiles.

    Model per gene:
        y(t) = c + sum_{k=1..K} [ A_k cos(k*omega*t) + B_k sin(k*omega*t) ] + eps
    where omega = 2*pi / period. K = n_harmonics. Default period = tmax - tmin.

    Requires raw counts: ``layer`` must hold non-negative integers (use
    ``layer="X"`` for ``adata.X``). Bins with fewer than 30 cells are dropped
    and the regression is done on the remaining bin centers.
    """
    from scipy.stats import f as f_dist
    from scipy.stats import false_discovery_control

    if tmin is None:
        tmin = adata.obs[time_key].min().round(2)

    if tmax is None:
        tmax = adata.obs[time_key].max().round(2)

    if period is None:
        period = tmax - tmin

    profiles, _ = _binned_log_profiles(adata, time_key, tmin, tmax, n, layer)
    labels = list(profiles.index)

    # Bin centers as floats; only surviving bins, so gaps are handled
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
        Y_hat, index=labels, columns=profiles.columns
    ).T
    adata.varm["harmonic_amplitudes"] = pd.DataFrame(
        harmonic_amps.T,
        index=profiles.columns,
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
