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
