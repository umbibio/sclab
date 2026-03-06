from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from ..pseudotime_tools import PseudotimeResult, _compute_pseudotime
from ..utils.density_nd import density_nd
from ..utils.interpolate import fit_smoothing_spline


def pseudotime(
    adata: AnnData,
    use_rep: str,
    t_key: str,
    t_range: tuple[float, float],
    n_dims: int = 10,
    min_snr: float = 0.25,
    periodic: bool = False,
    method: Literal["fourier", "splines"] = "splines",
    largest_harmonic: int = 5,
    roughness: float | None = None,
    key_added="pseudotime",
) -> PseudotimeResult:
    """Compute pseudotime ordering for cells by fitting a curve through a low-dimensional embedding.

    Fits either a Fourier series or smoothing spline to a reduced-dimensional
    representation of the data, then projects each cell onto the nearest point
    along the fitted curve. The arc-length along that curve is used as the
    pseudotime coordinate, normalised to the range [0, 1].

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must contain ``adata.obsm[use_rep]`` and
        ``adata.obs[t_key]``.
    use_rep : str
        Key in ``adata.obsm`` containing the low-dimensional embedding
        (e.g. ``"X_pca"``) used to fit the pseudotime curve.
    t_key : str
        Key in ``adata.obs`` that holds an initial continuous ordering of
        cells (e.g. a coarse time label or an existing pseudotime estimate)
        used to initialise the curve fit.
    t_range : tuple[float, float]
        ``(t_min, t_max)`` interval of ``t_key`` values to consider. Cells
        outside this range are excluded from fitting and their pseudotime is
        set to ``NaN``.
    n_dims : int, optional
        Maximum number of embedding dimensions to use for the curve fit.
        Default is 10.
    min_snr : float, optional
        Minimum signal-to-noise ratio (relative to the dimension with the
        highest SNR) required to include a dimension in the fit. Dimensions
        below this threshold are discarded. Default is 0.25.
    periodic : bool, optional
        If ``True``, treat the trajectory as periodic (cyclic). Requires
        ``t_range[0] == 0.0`` and ``method="fourier"`` or
        ``method="splines"`` with periodic boundary conditions. Default is
        ``False``.
    method : {"splines", "fourier"}, optional
        Curve-fitting method. ``"splines"`` fits an N-D smoothing spline;
        ``"fourier"`` fits an N-D Fourier series (only valid when
        ``periodic=True``). Default is ``"splines"``.
    largest_harmonic : int, optional
        Highest harmonic to include when ``method="fourier"``. Ignored for
        ``method="splines"``. Default is 5.
    roughness : float or None, optional
        Roughness penalty for the smoothing spline when ``method="splines"``.
        If ``None``, an automatic penalty is chosen. Default is ``None``.
    key_added : str, optional
        Base key under which results are stored. Default is ``"pseudotime"``.
        The following entries are written to ``adata``:

        - ``adata.obs[key_added]`` -- arc-length pseudotime in [0, 1].
        - ``adata.obs[key_added + "_path_residue"]`` -- Euclidean distance
          from each cell to its nearest point on the fitted curve.
        - ``adata.obsm[key_added + "_path"]`` -- fitted curve coordinates
          evaluated at each cell's projected pseudotime.
        - ``adata.obsm[key_added + "_path_derivative"]`` -- first derivative
          of the fitted curve at each cell's projected pseudotime.
        - ``adata.uns[key_added]`` -- dictionary of run parameters and SNR
          values.

    Returns
    -------
    PseudotimeResult
        A named tuple with the following fields:

        - ``pseudotime`` -- arc-length pseudotime values for cells within
          ``t_range``, normalised to [0, 1].
        - ``residues`` -- Euclidean residuals between each cell and its
          nearest curve point.
        - ``phi`` -- raw parameter values (in the original ``t_key`` units)
          of the nearest curve point for each cell.
        - ``F`` -- fitted curve object (``NDBSpline`` or ``NDFourier``)
          defined over the full embedding dimensionality.
        - ``SNR`` -- per-dimension signal-to-noise ratios, normalised so
          the maximum is 1.
        - ``snr_mask`` -- boolean mask indicating which dimensions passed
          the ``min_snr`` threshold.
        - ``t_mask`` -- boolean mask indicating which cells fall within
          ``t_range``.
        - ``fp_resolution`` -- floating-point resolution used during the
          final pseudotime refinement stage.

    Notes
    -----
    Results for cells outside ``t_range`` are stored as ``NaN`` in
    ``adata.obs``. The curve is fitted only on cells whose ``t_key`` value
    lies within ``[t_min, t_max]``.
    """
    X = adata.obsm[use_rep].copy().astype(float)
    X_path = np.zeros_like(X)
    X_path_derivative = np.zeros_like(X)
    X_path_derivative_norm = np.zeros((adata.n_obs,))

    t = adata.obs[t_key].values

    result = _compute_pseudotime(
        t, X, t_range, n_dims, min_snr, periodic, method, largest_harmonic, roughness
    )

    t_mask = result.t_mask
    pcs_mask = result.snr_mask
    mask = t_mask[:, None] * pcs_mask

    X_path[mask] = result.F[pcs_mask](result.phi).flatten()
    X_path_derivative[mask] = result.F[pcs_mask](result.phi, d=1).flatten()
    X_path_derivative_norm[t_mask] = np.linalg.norm(X_path_derivative[t_mask], axis=1)

    adata.obs[key_added] = np.nan
    adata.obs[key_added + "_path_residue"] = np.nan

    adata.obs.loc[t_mask, key_added] = result.pseudotime
    adata.obs.loc[t_mask, key_added + "_path_residue"] = result.residues
    # adata.obs[key_added + "_path_derivative_norm"] = X_path_derivative_norm
    adata.obsm[key_added + "_path"] = X_path
    adata.obsm[key_added + "_path_derivative"] = X_path_derivative
    adata.uns[key_added] = {
        "params": {
            "use_rep": use_rep,
            "t_key": t_key,
            "t_range": list(t_range),
            "min_snr": min_snr,
            "periodic": periodic,
            "method": method,
            "largest_harmonic": largest_harmonic,
            "roughness": roughness,
        },
        "snr": result.SNR.tolist(),
        "t_range": [0, 1],
        "periodic": periodic,
    }

    return result


def estimate_periodic_pseudotime_start(
    adata: AnnData,
    time_key: str = "pseudotime",
    bandwidth: float = 1 / 64,
    show_plot: bool = False,
    nth_root: int = 1,
):
    """Re-align a periodic pseudotime so that its zero corresponds to a density minimum.

    For a periodic (cyclic) trajectory, the choice of where pseudotime "starts"
    (i.e. where 0 and 1 meet) is arbitrary. This function estimates a
    principled start point by locating a trough in the cell-density
    distribution along the pseudotime axis. Concretely it:

    1. Estimates the 1-D kernel-density of pseudotime values on the circle.
    2. Fits the reciprocal density (sparsity) with a periodic smoothing spline.
    3. Finds inflection points of that spline and selects the ``nth_root``-th
       one corresponding to a local maximum of the sparsity derivative, i.e. a
       region of rapidly increasing cell sparsity.
    4. Shifts the pseudotime axis so that this point maps to 0, wrapping values
       modulo 1.

    The direction of the pseudotime axis is also checked and flipped if
    necessary so that the sparsity is increasing (positive derivative) at the
    selected start point.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must contain ``adata.obs[time_key]`` with
        periodic pseudotime values in [0, 1). Cells with ``NaN`` pseudotime
        are ignored.
    time_key : str, optional
        Key in ``adata.obs`` that holds the periodic pseudotime to be
        realigned. Modified in-place. Default is ``"pseudotime"``.
    bandwidth : float, optional
        Kernel bandwidth (as a fraction of the [0, 1] period) for the
        kernel-density estimate of the pseudotime distribution. Smaller
        values yield a finer-grained density estimate. Default is ``1/64``.
    show_plot : bool, optional
        If ``True``, display a diagnostic plot showing the pseudotime
        histogram, the KDE, the normalised sparsity, its derivative, and the
        selected start point. Default is ``False``.
    nth_root : int, optional
        Which inflection point (ranked by ascending sparsity-derivative
        height) to use as the start. ``1`` selects the inflection point with
        the smallest positive derivative, i.e. the gentlest transition out of
        the densest region. Default is ``1``.

    Returns
    -------
    None
        Modifies ``adata`` in-place. The realigned pseudotime values
        (shifted and wrapped to [0, 1)) are written back to
        ``adata.obs[time_key]``.

    Notes
    -----
    This function is intended for use after :func:`pseudotime` when
    ``periodic=True``. The implementation is experimental and has not yet
    been fully validated across all dataset types (see inline ``TODO``
    comment).
    """
    # TODO: Test implementation
    pseudotime = adata.obs[time_key].values.copy()
    t_mask = ~np.isnan(pseudotime)
    for _ in range(2):
        rslt = density_nd(
            pseudotime[t_mask].reshape(-1, 1),
            bandwidth,
            max_grid_size=2**10 + 1,
            periodic=True,
            bounds=((0, 1),),
            normalize=True,
        )
        bspl = fit_smoothing_spline(
            rslt.grid[:, 0],
            1 / rslt.density,
            t_range=(0, 1),
            lam=1e-5,
            periodic=True,
        )
        x = np.linspace(0, 1, 10001)
        y = bspl.derivative(0)(x)
        yp = bspl.derivative(1)(x)
        ypp = bspl.derivative(2)(x)

        if yp[np.argmax(np.abs(yp))] < 0:
            break

        pseudotime = -pseudotime % 1
    else:
        print("Warning: could not check direction for the pseudotime")

    idx = np.argwhere(np.sign(ypp[:-1]) < np.sign(ypp[1:])).flatten()
    roots = (x[idx] + x[1:][idx]) / 2
    heights = yp[idx]

    roots = roots[heights.argsort()]
    heights = heights[heights.argsort()]

    max_peak_x = roots[nth_root - 1]

    if show_plot:
        plt.hist(
            pseudotime, bins=100, density=True, fill=False, linewidth=0.5, alpha=0.5
        )
        plt.plot(rslt.grid[:-1, 0], rslt.density[:-1], color="k")
        plt.plot(x, y / np.abs(y).max())
        plt.plot(x, yp / np.abs(yp).max())
        plt.axvline(max_peak_x, color="k", linestyle="--")
        plt.show()

    pseudotime = (pseudotime - max_peak_x) % 1
    adata.obs[time_key] = pseudotime
