# Replaces pylimma's per-gene weighted least-squares loop, _lm_series_slow in
# pylimma/lmfit.py (https://github.com/john-mulvey/pylimma, by John Mulvey;
# GPL-3.0-or-later; checked against pylimma 0.2.0 and 0.3.0), itself a port of
# lm.series in R limma's lmfit.R (Gordon Smyth; GPL >= 2). Modified by sclab:
# the genes with all samples observed and a full-rank weighted design are
# fitted together with NumPy instead of one QR per gene; the other genes go to
# pylimma's own loop. Same results to rounding error.
"""Batched weighted least squares for ``pylimma.lm_fit`` with voom weights.

voom gives every gene its own precision weights, so ``pylimma.lm_fit`` takes
its per-gene path (``pylimma.lmfit._lm_series_slow``): a Python loop with a
pivoted QR per gene, ~90% of a pseudobulk limma-voom run (about 18 s for 20k
genes x 24 samples). :func:`fast_lm_series` swaps in :func:`lm_series_batched`,
which fits all genes at once with stacked QR factorisations, for the duration
of a ``with`` block and only on pylimma versions whose ``_lm_series_slow`` it
was checked against. On other versions pylimma runs unchanged.

limma: Ritchie et al. (2015), Nucleic Acids Res 43, e47,
https://doi.org/10.1093/nar/gkv007. voom: Law et al. (2014), Genome Biol 15,
R29, https://doi.org/10.1186/gb-2014-15-2-r29.
"""

import threading
from contextlib import contextmanager
from functools import partial

import numpy as np

# pylimma versions whose _lm_series_slow (M, design, weights[, coef_names]) the
# batched fit reproduces; the loop body is the same in both
TESTED_VERSIONS = ("0.2.0", "0.3.0")
CHUNK = 4096  # genes per stacked QR, bounds memory at chunk x samples x coefs
# R's dqrdc2 drops a column whose reduced norm falls below 1e-7 of its norm;
# genes within 1000x of that go to pylimma, which pivots as R does
RANK_TOL = 1e-4

_lock = threading.RLock()


def lm_series_batched(slow, M, design, weights, *args):
    """``_lm_series_slow(M, design, weights, *args)``, with most genes batched.

    A gene is fitted here when all its values and weights are finite and
    positive (``lm_series`` has already set non-positive weights to NaN and
    the matching values to NaN) and each column of its weighted design keeps
    at least ``RANK_TOL`` of its norm after projecting out the earlier
    columns, so R's ``dqrdc2`` would not pivot it. Every other gene (missing
    values, zero weights, rank deficiency, no weights, fewer samples than
    coefficients) is passed to pylimma's ``slow``, which also gives the
    unweighted ``cov_coefficients``, ``pivot`` and ``rank``.
    """
    n_genes, n = M.shape
    p = design.shape[1]
    coef = np.full((n_genes, p), np.nan)
    stdev = np.full((n_genes, p), np.nan)
    sigma = np.full(n_genes, np.nan)
    df = np.zeros(n_genes)

    good = np.zeros(n_genes, dtype=bool)
    if weights is not None and n >= p:
        good = np.isfinite(M).all(axis=1) & np.isfinite(weights).all(axis=1)

    for start in range(0, n_genes, CHUNK):
        idx = start + np.flatnonzero(good[start : start + CHUNK])
        if not idx.size:
            continue
        sw = np.sqrt(weights[idx])
        xw = design[None, :, :] * sw[:, :, None]  # genes x samples x coefs
        q, r = np.linalg.qr(xw)
        norm = np.sqrt((xw * xw).sum(axis=1))
        ok = (np.abs(np.diagonal(r, axis1=1, axis2=2)) >= RANK_TOL * norm).all(axis=1)
        ok &= (norm > 0).all(axis=1)
        good[idx[~ok]] = False
        idx, q, r = idx[ok], q[ok], r[ok]
        if not idx.size:
            continue

        yw = (M[idx] * sw[ok])[:, :, None]
        qty = q.transpose(0, 2, 1) @ yw
        coef[idx] = np.linalg.solve(r, qty)[:, :, 0]
        rinv = np.linalg.inv(r)
        stdev[idx] = np.sqrt((rinv * rinv).sum(axis=2))  # sqrt(diag(R^-1 R^-T))
        df[idx] = n - p
        if n > p:
            resid = yw - q @ qty
            sigma[idx] = np.sqrt((resid * resid).sum(axis=(1, 2)) / (n - p))

    bad = ~good
    out = slow(M[bad], design, None if weights is None else weights[bad], *args)
    coef[bad] = out["coefficients"]
    stdev[bad] = out["stdev_unscaled"]
    sigma[bad] = out["sigma"]
    df[bad] = out["df_residual"]
    return {
        **out,
        "coefficients": coef,
        "stdev_unscaled": stdev,
        "sigma": sigma,
        "df_residual": df,
    }


@contextmanager
def fast_lm_series():
    """Within the block, ``pylimma.lm_fit`` fits per-gene weights batched.

    Rebinds ``pylimma.lmfit._lm_series_slow`` (which ``lm_series`` looks up at
    call time) and restores it on exit. Does nothing on pylimma versions not
    in ``TESTED_VERSIONS``. The lock keeps concurrent blocks from restoring
    each other's binding; any other pylimma call made meanwhile gets the same
    results from the batched fit.
    """
    import pylimma
    import pylimma.lmfit as lmfit

    slow = getattr(lmfit, "_lm_series_slow", None)
    if pylimma.__version__ not in TESTED_VERSIONS or slow is None:
        yield
        return
    with _lock:
        slow = lmfit._lm_series_slow
        lmfit._lm_series_slow = partial(lm_series_batched, slow)
        try:
            yield
        finally:
            lmfit._lm_series_slow = slow
