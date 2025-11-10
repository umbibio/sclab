import gc

import numpy as np
from numpy import float32
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, csr_matrix, issparse


def _alra_on_ndarray(
    data: NDArray | csr_matrix,
) -> tuple[NDArray[float32], NDArray[float32]]:
    """
    Run ALRA on the given data.

    Parameters
    ----------
    data : NDArray | csr_matrix
        Input data to impute.

    Returns
    -------
    data_aprx : NDArray
        Approximated data.
    data_alra : NDArray
        Imputed data.
    """
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri
    from rpy2.robjects.packages import importr

    rpy2.robjects.numpy2ri.activate()
    R = robjects.r
    alra = importr("ALRA")

    if issparse(data):
        data = np.ascontiguousarray(data.todense("C"), dtype=np.float32)

    # convert to R object
    r_X = R.matrix(data, nrow=data.shape[0], ncol=data.shape[1])
    # run ALRA
    r_res = alra.alra(r_X, 0, 10, 0.001)
    # retrieve imputed data
    r_K = r_res[0]  # rank k
    r_T = r_res[1]  # rank k thresholded
    r_S = r_res[2]  # rank k thresholded scaled
    # convert back to numpy array
    data_aprx = np.array(r_K, dtype=float32)
    data_thrs = np.array(r_T, dtype=float32)
    data_alra = np.array(r_S, dtype=float32)

    # clean up
    del (r_X, r_res, r_K, r_T, r_S)
    R("gc()")
    gc.collect()

    return data_aprx, data_thrs, data_alra


def _fix_alra_scale(
    input_data: NDArray | csr_matrix | csc_matrix,
    thrs_data: NDArray,
    target_data: NDArray,
) -> NDArray:
    # Convert sparse -> dense
    if issparse(input_data):
        input_data = input_data.toarray("C")
        input_data = input_data.astype(np.float32)
        input_data = np.ascontiguousarray(input_data, dtype=np.float32)

    n_cells, n_genes = input_data.shape

    # per-gene nonzero means/sds (match R: sample sd ddof=1)
    input_means = np.full(n_genes, fill_value=np.nan)
    input_stds = np.full(n_genes, fill_value=np.nan)
    thrs_means = np.full(n_genes, fill_value=np.nan)
    thrs_stds = np.full(n_genes, fill_value=np.nan)
    v: NDArray

    for i, e in enumerate(input_data.T):
        v = e[e > 0]

        if v.size == 0:
            continue
        input_means[i] = v.mean()

        if v.size == 1:
            continue
        input_stds[i] = v.std(ddof=1)

    for i, e in enumerate(thrs_data.T):
        v = e[e > 0]

        if v.size == 0:
            continue
        thrs_means[i] = v.mean()

        if v.size == 1:
            continue
        thrs_stds[i] = v.std(ddof=1)

    # columns to scale (mirror R's toscale)
    toscale = (
        ~np.isnan(thrs_stds)
        & ~np.isnan(input_stds)
        & ~((thrs_stds == 0) & (input_stds == 0))
        & ~(thrs_stds == 0)
    )

    # affine params
    a = np.full(n_genes, fill_value=1.0)
    b = np.full(n_genes, fill_value=0.0)
    a[toscale] = input_stds[toscale] / thrs_stds[toscale]
    b[toscale] = input_means[toscale] - a[toscale] * thrs_means[toscale]

    # apply to target matrix (only columns in toscale)
    out = target_data.copy()
    out[:, toscale] = out[:, toscale] * a[toscale] + b[toscale]

    # keep zeros as zeros
    out[thrs_data == 0] = 0

    # clip negatives to zero
    out[out < 0] = 0

    # restore originally observed positives that became zero
    mask = (input_data > 0) & (out == 0)
    out[mask] = input_data[mask]

    return out


__all__ = [
    "_alra_on_ndarray",
    "_fix_alra_scale",
]
