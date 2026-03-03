import numpy as np
from anndata import AnnData
from pandas.api.types import is_bool_dtype
from scipy.sparse import issparse

from ..utils.interpolate import NDBSpline


def expression_dynamics(
    adata: AnnData,
    time_key: str,
    t_range: tuple[float, float] | None = None,
    periodic: bool | None = None,
    layer: str | None = None,
    gene_mask: str | None = None,
    n_grid: int = 1001,
    progress: bool = False,
):
    """Compute per-cell gene turnover from expression dynamics over pseudotime.

    Fits a smooth B-spline to the expression matrix over pseudotime, takes
    the analytical derivative (dX/dt), then counts the number of genes with
    high activation (rate > median of positives) and high repression
    (rate < median of negatives) for each cell.

    Additionally computes per-gene timing summaries (pseudotime of peak
    activation, peak repression, acceleration onset, and deceleration onset)
    and a per-cell transcriptional flux (total absolute velocity across
    genes).

    Parameters
    ----------
    adata
        Annotated data matrix. Must contain pseudotime values in
        ``adata.obs[time_key]``.
    time_key
        Column in ``adata.obs`` with pseudotime values. If
        ``adata.uns[time_key]`` exists, ``t_range`` and ``periodic`` are
        read from it when not explicitly provided.
    t_range
        Min and max pseudotime for the spline domain. Inferred from
        ``adata.uns[time_key]['t_range']`` or the data range if *None*.
    periodic
        Whether pseudotime is periodic (e.g. cell cycle). Inferred from
        ``adata.uns[time_key]['periodic']`` or defaults to *False*.
    layer
        Layer in ``adata.layers`` to use as expression matrix. Uses
        ``adata.X`` when *None*.
    gene_mask
        Boolean column in ``adata.var`` to subset genes before fitting.
        When provided, output columns are prefixed with ``{gene_mask}_``
        instead of the defaults.
    n_grid
        Number of evenly spaced points over ``t_range`` used to locate
        per-gene derivative extrema. Higher values give more precise
        timing estimates at modest computational cost.
    progress
        Show a progress bar during spline fitting.

    Returns
    -------
    None
        Modifies ``adata`` in-place.

        **obs columns** (per-cell):

        - ``n_activation`` / ``{gene_mask}_up`` — number of genes with
          velocity above the median of all positive velocities.
        - ``n_repression`` / ``{gene_mask}_dw`` — number of genes with
          velocity below the median of all negative velocities.
        - ``transcriptional_flux`` / ``{gene_mask}_flux`` — sum of
          absolute velocities across genes.

        **var columns** (per-gene, restricted to *gene_mask* rows when
        provided):

        - ``peak_activation_t`` / ``{gene_mask}_peak_activation_t`` —
          pseudotime of maximum first derivative.
        - ``peak_repression_t`` / ``{gene_mask}_peak_repression_t`` —
          pseudotime of minimum first derivative.
        - ``acceleration_onset_t`` / ``{gene_mask}_acceleration_onset_t``
          — pseudotime of maximum second derivative.
        - ``deceleration_onset_t`` / ``{gene_mask}_deceleration_onset_t``
          — pseudotime of minimum second derivative.
    """
    if gene_mask is not None:
        assert gene_mask in adata.var, "gene_mask must be a column name"
        assert is_bool_dtype(adata.var[gene_mask]), "gene_mask must be a boolean column"

    if t_range is None and time_key in adata.uns and "t_range" in adata.uns[time_key]:
        t_range = tuple(adata.uns[time_key]["t_range"])
    elif t_range is None:
        t_range = adata.obs[time_key].min(), adata.obs[time_key].max()

    if periodic is None and time_key in adata.uns and "periodic" in adata.uns[time_key]:
        periodic = adata.uns[time_key]["periodic"]
    elif periodic is None:
        periodic = False

    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]

    if issparse(X):
        X = np.ascontiguousarray(X.todense("C"), dtype=np.float32)

    if gene_mask is not None:
        X = X[:, adata.var[gene_mask]]

    t = adata.obs[time_key].values
    F = NDBSpline(t_range=t_range, periodic=periodic).fit(t, X, progress=progress)
    D = F.derivative()
    V = D(t)

    # -- per-cell turnover counts ------------------------------------------
    hiv = np.median(V[V > 0])
    lov = np.median(V[V < 0])

    n_up = (V > hiv).sum(axis=1, keepdims=True)
    n_dw = (V < lov).sum(axis=1, keepdims=True)
    flux = np.abs(V).sum(axis=1, keepdims=True)

    if gene_mask is not None:
        adata.obs[[f"{gene_mask}_up", f"{gene_mask}_dw"]] = np.hstack([n_up, n_dw])
        adata.obs[f"{gene_mask}_flux"] = flux
    else:
        adata.obs[["n_activation", "n_repression"]] = np.hstack([n_up, n_dw])
        adata.obs["transcriptional_flux"] = flux

    # -- per-gene timing summaries -----------------------------------------
    t_grid = np.linspace(t_range[0], t_range[1], n_grid)

    V_grid = D(t_grid)  # (n_grid, n_genes) first derivative
    D2 = D.derivative()
    A_grid = D2(t_grid)  # (n_grid, n_genes) second derivative

    peak_activation_t = t_grid[np.argmax(V_grid, axis=0)]
    peak_repression_t = t_grid[np.argmin(V_grid, axis=0)]
    acceleration_onset_t = t_grid[np.argmax(A_grid, axis=0)]
    deceleration_onset_t = t_grid[np.argmin(A_grid, axis=0)]

    var_idx = adata.var[gene_mask] if gene_mask is not None else slice(None)
    prefix = f"{gene_mask}_" if gene_mask is not None else ""

    adata.var.loc[var_idx, f"{prefix}peak_activation_t"] = peak_activation_t
    adata.var.loc[var_idx, f"{prefix}peak_repression_t"] = peak_repression_t
    adata.var.loc[var_idx, f"{prefix}acceleration_onset_t"] = acceleration_onset_t
    adata.var.loc[var_idx, f"{prefix}deceleration_onset_t"] = deceleration_onset_t
