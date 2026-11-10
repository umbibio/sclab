# Ported from edgePython 0.5.1 (https://github.com/pachterlab/edgePython, by
# Lior Pachter; GPL-3.0-or-later): edgepython/glm_kernels.py, ql_weights.py,
# dispersion_lowlevel.py and weighted_lowess.py. Those are numba ports of the C
# code of edgeR 5.0 (Bioconductor, GPL >= 2: glm.c, compute_nbdev.c,
# ql_weights.c, ql_glm.c, compute_adj_profile_ll, fmm_spline from R's
# splines.c) and of limma's weighted_lowess.c (GPL >= 2). Modified by sclab:
# vectorised over genes with NumPy instead of numba loops, same results to
# rounding error.
"""NumPy versions of the numba kernels of edgePython that pseudobulk edgeR reaches.

Drop-in replacements, with the same signatures and return values, for the
``@njit`` kernels that :func:`sclab.tools.de` (``method="pseudobulk_edger"``)
runs through ``filter_by_expr``, ``estimate_disp``, ``glm_ql_fit`` and
``glm_ql_ftest``. Instead of a compiled loop over genes, each kernel runs its
per-gene recursion on all genes at once with NumPy array operations, so it is
fast without numba (Pyodide / WebAssembly). Sums are accumulated in the same
order as the loops they replace, except the X'WX matrix products (one matrix
multiplication for all genes), so results agree with numba's to rounding
error. :mod:`._edgepython_numba_free` swaps them in
only when numba is not installed.

The kernels follow, loop for loop, the edgeR C code that edgePython ports:

- Negative binomial GLM fits (Levenberg-Marquardt, one-group and one-way
  Newton-Raphson) and deviances: McCarthy, Chen & Smyth (2012),
  https://doi.org/10.1093/nar/gks042; Robinson, McCarthy & Smyth (2010),
  https://doi.org/10.1093/bioinformatics/btp616.
- Cox-Reid adjusted profile likelihood and its spline maximisation
  (``estimateDisp``): McCarthy, Chen & Smyth (2012).
- Quasi-likelihood adjusted deviances and degrees of freedom (``glmQLFit``,
  edgeR v4): Chen, Chen, Lun, Baldoni & Smyth (2025),
  https://doi.org/10.1093/nar/gkaf018; Lun, Chen & Smyth (2016),
  https://doi.org/10.1007/978-1-4939-3578-9_19; Chen, Lun & Smyth (2016),
  https://doi.org/10.12688/f1000research.8987.2.
- Weighted lowess (limma's ``weightedLowess``, used by the QL prior trend).
- edgePython: Pachter (2026), https://doi.org/10.64898/2026.02.16.706223.

They use only NumPy and SciPy, and edgePython's own Chebyshev tables and scalar
Bunch-Kaufman factorisation, so they can be offered to edgePython as they are.
"""

import math

import numpy as np
from scipy.special import gammaln

LOW_BOUND = 1e-32  # ql_weights.c: mu below this has zero weight
THRESHOLD_ZERO = 1e-4  # ql_glm.c: leverage-free observations


def _rowsum(a):
    """Row sums accumulated column by column, as the C loops (not pairwise)."""
    s = np.zeros(a.shape[0])
    for j in range(a.shape[1]):
        s += a[:, j]
    return s


def unit_nb_deviance(y, mu, phi):
    """Unit NB deviance, elementwise (compute_nbdev.c compute_unit_nb_deviance).

    Port of edgepython.glm_kernels.unit_nb_deviance: Taylor expansion for
    phi < 1e-4, gamma approximation for mu * phi > 1e6.
    """
    y = y + 1e-8
    mu = mu + 1e-8
    phi = np.broadcast_to(phi, y.shape)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        logyu = np.log(y / mu)
        product = mu * phi
        invphi = 1.0 / phi
        out = 2.0 * (y * logyu + (y + invphi) * np.log((mu + invphi) / (y + invphi)))
        small = phi < 1e-4  # Taylor expansion (also the Poisson case)
        if small.any():
            ys, ms, ps, ls = y[small], mu[small], phi[small], logyu[small]
            r = ys - ms
            out[small] = 2.0 * (
                ys * ls - r - 0.5 * r * r * ps * (1.0 + ps * (2.0 / 3.0 * r - ys))
            )
        large = ~small & (product > 1e6)  # gamma approximation
        if large.any():
            ys, ms, ls = y[large], mu[large], logyu[large]
            out[large] = 2.0 * ((ys - ms) / ms - ls) * ms / (1.0 + product[large])
    return np.where(out < 0.0, 0.0, out)


def nb_deviance_sum_mat(y, mu, disp, w):
    """Genewise weighted sum of unit NB deviances (compute_nbdev_sum)."""
    return _rowsum(w * unit_nb_deviance(y, mu, disp))


def fit_one_group_mat(y, offset, disp, w, maxit, tol, beta_start):
    """One-group NB fits by Newton-Raphson (glm.c glm_one_group_vec), all genes.

    beta_start NaN means no start value. All-zero rows give -1e8, as edgeR.
    Returns (beta, converged).
    """
    y = np.asarray(y, dtype=np.float64)
    beta = np.array(beta_start, dtype=np.float64)
    pos = y > 1e-10
    allzero = ~pos.any(axis=1)
    start = np.isnan(beta) & ~allzero
    if start.any():
        ys, os_, ws = y[start], offset[start], w[start]
        num = _rowsum(np.where(pos[start], ys / np.exp(os_) * ws, 0.0))
        beta[start] = np.log(num / _rowsum(ws))
    beta[allzero] = -1e8
    conv = allzero.copy()
    act = np.flatnonzero(~allzero)
    for _ in range(maxit):
        if act.size == 0:
            break
        b = beta[act]
        mu = np.exp(b[:, None] + offset[act])
        denom = 1.0 + mu * disp[act]
        dl = _rowsum((y[act] - mu) / denom * w[act])
        info = _rowsum(mu / denom * w[act])
        step = dl / info
        beta[act] = b + step
        done = np.abs(step) < tol
        conv[act[done]] = True
        act = act[~done]
    return beta, conv


def fit_one_way_mat(y, offset, disp, w, group, ngroups, maxit, tol, beta_start):
    """Groupwise one-group fits (fit_one_way_mat); one beta column per group."""
    ngenes = y.shape[0]
    beta = np.empty((ngenes, ngroups))
    conv = np.empty((ngenes, ngroups), dtype=bool)
    for k in range(ngroups):
        s = np.asarray(group) == k
        beta[:, k], conv[:, k] = fit_one_group_mat(
            y[:, s], offset[:, s], disp[:, s], w[:, s], maxit, tol, beta_start[:, k]
        )
    return beta, conv


def _autofill(beta, offset, design):
    """exp(design @ beta + offset), summed in dgemv's column order."""
    eta = np.array(offset, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):  # all-zero genes: -inf
        for c in range(design.shape[1]):
            eta += design[:, c] * beta[:, c, None]
        return np.exp(eta)


def _cholesky_upper(a):
    """Upper Cholesky factors U'U = a of a stack (LAPACK dpotf2, uplo = 'U').

    Returns (U, ok); ok is False where a is not positive definite.
    """
    n, p = a.shape[0], a.shape[1]
    u = np.zeros_like(a)
    ok = np.ones(n, dtype=bool)
    for j in range(p):
        s = a[:, j, j].copy()
        t = a[:, j, j + 1 :].copy()
        for k in range(j):
            s -= u[:, k, j] * u[:, k, j]
            t -= u[:, k, j, None] * u[:, k, j + 1 :]
        ok &= s > 0.0
        ujj = np.sqrt(np.where(ok, s, 1.0))
        u[:, j, j] = ujj
        u[:, j, j + 1 :] = t / ujj[:, None]
    return u, ok


def _cholesky_solve_upper(u, b):
    """Solve U'U x = b for a stack, given the upper factors (LAPACK dpotrs)."""
    p = u.shape[1]
    x = np.empty_like(b)
    for i in range(p):
        t = b[:, i].copy()
        for k in range(i):
            t -= u[:, k, i] * x[:, k]
        x[:, i] = t / u[:, i, i]
    for i in range(p - 1, -1, -1):
        t = x[:, i].copy()
        for k in range(i + 1, p):
            t -= u[:, i, k] * x[:, k]
        x[:, i] = t / u[:, i, i]
    return x


def fit_leven_mat(y, offset, disp, w, design, beta_start, maxit, tol):
    """Levenberg-Marquardt NB fits for every gene (glm.c fit_leven).

    Returns (beta, mu, deviance, iterations, failed), as
    edgepython.glm_kernels.fit_leven_mat. The outer (Fisher scoring) and inner
    (damping) loops run on the genes still iterating.
    """
    ngenes, nlibs = y.shape
    p = design.shape[1]
    beta = np.array(beta_start, dtype=np.float64)
    ymax = y.max(axis=1)
    mu = _autofill(beta, offset, design)
    dev = nb_deviance_sum_mat(y, mu, disp, w)
    iters = np.zeros(ngenes, dtype=np.int64)
    failed = np.zeros(ngenes, dtype=bool)
    max_info = np.full(ngenes, -1.0)
    lam = np.zeros(ngenes)

    act = ymax >= 1e-10
    beta[~act] = np.nan
    mu[~act] = 0.0
    dev[~act] = 0.0
    A = np.flatnonzero(act)
    xx = (design[:, :, None] * design[:, None, :]).reshape(nlibs, p * p)
    while A.size:
        iters[A] += 1
        A = A[iters[A] <= maxit]
        if A.size == 0:
            break
        m, d, ww = mu[A], disp[A], w[A]
        denom = 1.0 + m * d
        zw = m / denom * ww
        dr = (y[A] - m) / denom * ww
        xtwx = (zw @ xx).reshape(-1, p, p)  # a GEMM, not edgeR's summation order
        dl = dr @ design
        mi = max_info[A]
        for c in range(p):
            mi = np.where(xtwx[:, c, c] > mi, xtwx[:, c, c], mi)
        max_info[A] = mi
        first = iters[A] == 1
        lam[A[first]] = np.maximum(mi[first] * 1e-6, 1e-13)

        # inner loop: damping search, on the genes not yet accepted (S indexes A)
        lev = np.zeros(A.size, dtype=np.int64)
        low_dev = np.zeros(A.size, dtype=bool)
        fail = np.zeros(A.size, dtype=bool)
        db = np.zeros((A.size, p))
        S = np.arange(A.size)
        eye = np.eye(p)
        while S.size:
            lev[S] += 1
            U = np.empty((S.size, p, p))
            todo = np.arange(S.size)  # increase damping until factorisable
            while todo.size:
                g = A[S[todo]]
                u, ok = _cholesky_upper(xtwx[S[todo]] + lam[g, None, None] * eye)
                U[todo[ok]] = u[ok]
                todo = todo[~ok]
                g = A[S[todo]]
                lam[g] *= 10.0
                lam[g] = np.where(lam[g] <= 0.0, 1e-100, lam[g])
            db[S] = _cholesky_solve_upper(U, dl[S])
            g = A[S]
            nbeta = beta[g] + db[S]
            nmu = _autofill(nbeta, offset[g], design)
            ndev = nb_deviance_sum_mat(y[g], nmu, disp[g], w[g])
            low = ndev / ymax[g] < 1e-13
            low_dev[S] = low
            acc = (ndev <= dev[g]) | low
            ga = g[acc]
            beta[ga], mu[ga], dev[ga] = nbeta[acc], nmu[acc], ndev[acc]
            R = S[~acc]
            gr = A[R]
            lam[gr] *= 2.0
            lam[gr] = np.where(lam[gr] <= 0.0, 1e-100, lam[gr])
            f = lam[gr] / max_info[gr] > 1e13
            fail[R[f]] = True
            S = R[~f]
        failed[A] |= fail
        divergence = _rowsum(dl * db)
        done = fail | low_dev | (divergence < tol)
        cont = ~done & (lev == 1)
        lam[A[cont]] /= 10.0
        A = A[~done]
    return beta, mu, dev, iters, failed


def _dsytf2_diagonal(a):
    """Diagonal of the Bunch-Kaufman factor D of each upper-stored matrix.

    LAPACK dsytf2 (uplo = 'U') on a stack, in place. All genes take the
    unpivoted 1x1 path together; a gene needing an interchange or a 2x2 pivot
    is refactored by edgePython's scalar port (glm_kernels._dsytf2_upper).
    """
    from edgepython.glm_kernels import _dsytf2_upper

    orig = a.copy()
    n, p = a.shape[0], a.shape[1]
    alpha = (1.0 + math.sqrt(17.0)) / 8.0
    bad = np.zeros(n, dtype=bool)
    for k in range(p - 1, -1, -1):
        akk = a[:, k, k]
        absakk = np.abs(akk)
        col = a[:, :k, k]
        colmax = np.fmax.reduce(np.abs(col), axis=1) if k else np.zeros(n)
        skip = (np.fmax(absakk, colmax) == 0.0) | np.isnan(absakk)
        bad |= ~skip & ~(absakk >= alpha * colmax)
        upd = ~skip & ~bad
        if k == 0 or not upd.any():
            continue
        r1 = np.where(upd, 1.0 / np.where(upd, akk, 1.0), 0.0)[:, None]
        with np.errstate(invalid="ignore"):
            temp = -r1 * col
        # rank-1 update of the leading block (the lower triangle is never read)
        nz = upd[:, None, None] & (col[:, None, :] != 0.0)
        with np.errstate(invalid="ignore", over="ignore"):
            a[:, :k, :k] += np.where(nz, col[:, :, None] * temp[:, None, :], 0.0)
        a[:, :k, k] = np.where(upd[:, None], col * r1, col)
    for g in np.flatnonzero(bad):
        a[g] = orig[g]
        _dsytf2_upper(a[g])
    return np.diagonal(a, axis1=1, axis2=2)


def adj_profile_ll_mat(y, mu, disp, w, design, adjust):
    """Cox-Reid adjusted profile log-likelihoods (compute_adj_profile_ll).

    As edgepython.glm_kernels.adj_profile_ll_mat: libraries with mu = 0 are
    skipped; the adjustment is half the sum of the logs of the Bunch-Kaufman
    diagonal of X'WX (log(1e-10) for entries below 1e-10).
    """
    ngenes, nlibs = y.shape
    nvar = design.shape[1]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        curu = mu * w
        cury = y * w
        curd = disp / w
        r = 1.0 / curd
        logmur = np.log(curu + r)
        nb = (
            cury * np.log(curu)
            - cury * logmur
            + r * np.log(r)
            - r * logmur
            + gammaln(cury + r)
            - gammaln(cury + 1.0)
            - gammaln(r)
        )
        pois = cury * np.log(curu) - curu - gammaln(cury + 1.0)
        zw = curu / (1.0 + curd * curu)
    skip = mu == 0.0
    s = _rowsum(np.where(skip, 0.0, np.where(curd > 0.0, nb, pois)))
    if not adjust:
        return s
    zw = np.where(skip, 0.0, zw)
    if nvar == 1:
        with np.errstate(divide="ignore"):
            return s - np.log(np.abs(_rowsum(zw))) / 2.0
    xx = (design[:, :, None] * design[:, None, :]).reshape(nlibs, nvar * nvar)
    xtwx = (zw @ xx).reshape(ngenes, nvar, nvar)
    diag = _dsytf2_diagonal(xtwx)
    adj = np.zeros(ngenes)
    log_low = math.log(1e-10)
    for i in range(nvar):
        di = diag[:, i]
        with np.errstate(divide="ignore", invalid="ignore"):
            adj += np.where(di < 1e-10, log_low, np.log(di) * 0.5)
    return s - adj


def _fmm_spline(x, y):
    """Forsythe-Malcolm-Moler cubic splines through the rows of y (splines.c).

    Returns (b, c, d) with S(t) = y[i] + b[i] t + c[i] t^2 + d[i] t^3 on
    segment i, as edgepython.dispersion_lowlevel._fmm_spline per row.
    """
    ngenes, n = y.shape
    b = np.zeros((ngenes, n))
    c = np.zeros((ngenes, n))
    d = np.zeros((ngenes, n))
    if n < 2:
        return b, c, d
    if n < 3:
        t = (y[:, 1] - y[:, 0]) / (x[1] - x[0])
        b[:, 0] = t
        b[:, 1] = t
        return b, c, d
    nm1 = n - 1
    d[:, 0] = x[1] - x[0]
    c[:, 1] = (y[:, 1] - y[:, 0]) / d[:, 0]
    for i in range(1, nm1):
        d[:, i] = x[i + 1] - x[i]
        b[:, i] = 2.0 * (d[:, i - 1] + d[:, i])
        c[:, i + 1] = (y[:, i + 1] - y[:, i]) / d[:, i]
        c[:, i] = c[:, i + 1] - c[:, i]
    b[:, 0] = -d[:, 0]
    b[:, nm1] = -d[:, nm1 - 1]
    c[:, 0] = 0.0
    c[:, nm1] = 0.0
    if n > 3:
        c[:, 0] = c[:, 2] / (x[3] - x[1]) - c[:, 1] / (x[2] - x[0])
        c[:, nm1] = c[:, nm1 - 1] / (x[nm1] - x[nm1 - 2]) - c[:, nm1 - 2] / (
            x[nm1 - 1] - x[nm1 - 3]
        )
        c[:, 0] = c[:, 0] * d[:, 0] * d[:, 0] / (x[3] - x[0])
        c[:, nm1] = -c[:, nm1] * d[:, nm1 - 1] * d[:, nm1 - 1] / (x[nm1] - x[nm1 - 3])
    for i in range(1, n):
        t = d[:, i - 1] / b[:, i - 1]
        b[:, i] = b[:, i] - t * d[:, i - 1]
        c[:, i] = c[:, i] - t * c[:, i - 1]
    c[:, nm1] = c[:, nm1] / b[:, nm1]
    for i in range(nm1 - 1, -1, -1):
        c[:, i] = (c[:, i] - d[:, i] * c[:, i + 1]) / b[:, i]
    b[:, nm1] = (y[:, nm1] - y[:, nm1 - 1]) / d[:, nm1 - 1] + d[:, nm1 - 1] * (
        c[:, nm1 - 1] + 2.0 * c[:, nm1]
    )
    for i in range(nm1):
        b[:, i] = (y[:, i + 1] - y[:, i]) / d[:, i] - d[:, i] * (
            c[:, i + 1] + 2.0 * c[:, i]
        )
        d[:, i] = (c[:, i + 1] - c[:, i]) / d[:, i]
        c[:, i] = 3.0 * c[:, i]
    c[:, nm1] = 3.0 * c[:, nm1]
    d[:, nm1] = d[:, nm1 - 1]
    return b, c, d


def _maximize_interpolant_kernel(x, y_mat, ngenes, npts, result):
    """Spline maximum of each row of y_mat (edgeR's maximizeInterpolant), into result.

    As edgepython.dispersion_lowlevel._maximize_interpolant_kernel: the grid
    maximum, refined by the stationary points of the spline on the two
    neighbouring segments.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y_mat, dtype=np.float64)[:ngenes, :npts]
    rows = np.arange(ngenes)
    # first maximum, skipping NaN (comparisons with NaN are False in the loop)
    at = np.argmax(np.where(np.isnan(y), -np.inf, y), axis=1)
    at[np.isnan(y[:, 0])] = 0
    maxed = y[rows, at]
    x_max = x[at]
    b, c, d = _fmm_spline(x, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        for seg, ok in ((at - 1, at > 0), (at, at < npts - 1)):
            seg = np.clip(seg, 0, npts - 2)
            sb, sc, sd = b[rows, seg], c[rows, seg], d[rows, seg]
            delta = sc * sc - 3.0 * sd * sb
            sol = np.where(sd != 0.0, (-sc - np.sqrt(delta)) / (3.0 * sd), 0.0)
            width = x[seg + 1] - x[seg]
            temp = ((sd * sol + sc) * sol + sb) * sol + y[rows, seg]
            better = ok & (delta >= 0.0) & (sol > 0.0) & (sol < width) & (temp > maxed)
            maxed = np.where(better, temp, maxed)
            x_max = np.where(better, sol + x[seg], x_max)
    result[:ngenes] = x_max


def _hat_values(design, zwpt):
    """Leverages of diag(zwpt) @ design for each row of zwpt (edgeR's qr_hat).

    As edgepython.ql_weights._hat_values per row: Gram-Schmidt with
    reorthogonalisation, setting aside a column whose remaining norm is below
    1e-7 of its original norm (LINPACK dqrdc2's rule). zwpt is (ngenes, nlib)
    or (nlib,); returns the same shape.
    """
    zwpt = np.asarray(zwpt, dtype=np.float64)
    single = zwpt.ndim == 1
    zwpt = np.atleast_2d(zwpt)
    ngenes, nlib = zwpt.shape
    basis = np.zeros((design.shape[1], ngenes, nlib))  # zero for set-aside columns
    k = np.zeros(ngenes, dtype=np.int64)
    for c in range(design.shape[1]):
        v = design[:, c] * zwpt
        orig = np.sqrt(_rowsum(v * v))
        for _ in range(2):
            for q in range(c):
                v = v - _rowsum(basis[q] * v)[:, None] * basis[q]
        nv = np.sqrt(_rowsum(v * v))
        keep = (orig > 0.0) & (nv >= 1e-7 * orig) & (k < nlib)
        basis[c, keep] = v[keep] / nv[keep, None]
        k += keep
    hat = np.zeros((ngenes, nlib))
    for q in range(design.shape[1]):
        hat += basis[q] * basis[q]
    return hat[0] if single else hat


# Chebyshev approximations of the QL weights (ql_weights.c). The coefficient
# tables are edgePython's; ``it`` is the offset of each element's coefficients.


def _cheb_terms(x, n):
    t = [np.ones_like(x), x]
    for _ in range(2, n):
        t.append(2.0 * x * t[-1] - t[-2])
    return t


def _cheb(weights, x, n, it):
    """sum_i weights[it + i] T_i(x), elementwise, in _cheb_eval's order."""
    out = 0.0
    for i, t in enumerate(_cheb_terms(x, n)):
        out = out + weights[it + i] * t
    return out


def _cheb_2d(weights, x, y, nx, ny, it):
    """2-D Chebyshev sum, elementwise, in _cheb_eval_2d's order."""
    tx, ty = _cheb_terms(x, nx), _cheb_terms(y, ny)
    out = 0.0
    for i in range(ny):
        for j in range(nx):
            out = out + weights[it + i * nx + j] * tx[j] * ty[i]
    return out


def _pieces(mu, bounds, xs, its):
    """x and table offset of the first interval mu < bound (else the last)."""
    conds = [mu < b for b in bounds]
    x = np.select(conds, [f(mu) for f in xs[:-1]], xs[-1](mu))
    return x, np.select(conds, its[:-1], its[-1])


def _pois_alpha(mu):
    from edgepython.ql_weights import _pois_alpha_weights as w

    with np.errstate(divide="ignore", invalid="ignore"):
        x, it = _pieces(
            mu,
            [0.02, 0.4249, 1.5, 3.544],
            [
                lambda m: 2.0 * m / 0.02 - 1.0,
                lambda m: (2.0 * m - 0.4449) / 0.4049,
                lambda m: (2.0 * m - 1.9249) / 1.0751,
                lambda m: (2.0 * m - 5.0440) / 2.0440,
                lambda m: (2.0 * m - 23.544) / 16.456,
            ],
            [0, 10, 20, 30, 40],
        )
        out = _cheb(w, x, 10, it)
        logmu = np.log(mu)
        out = np.where(mu < 0.02, -out * logmu / ((1.0 + logmu) * (1.0 + logmu)), out)
        out = np.where(mu < 20.0, out, 1.0 - 1.0 / (6.0 * mu) - 1.0 / (2.0 * mu * mu))
    return np.where(mu < LOW_BOUND, 0.0, out)


def _pois_kappa(mu):
    from edgepython.ql_weights import _pois_kappa_weights as w

    with np.errstate(divide="ignore", invalid="ignore"):
        x, it = _pieces(
            mu,
            [0.02, 0.4966, 1.5, 4.2714],
            [
                lambda m: 2.0 * m / 0.02 - 1.0,
                lambda m: (2.0 * m - 0.5166) / 0.4766,
                lambda m: (2.0 * m - 1.9966) / 1.0034,
                lambda m: (2.0 * m - 5.7714) / 2.7714,
                lambda m: (2.0 * m - 24.2714) / 15.7286,
            ],
            [0, 10, 20, 30, 40],
        )
        out = _cheb(w, x, 10, it)
        logmu = np.log(mu) / (1.0 + np.log(mu))
        out = np.where(mu < 0.02, out * mu * logmu * logmu, out)
        out = np.where(mu < 20.0, out, 1.0 - 1.0 / (2.5 * mu * mu))
    return np.where(mu < LOW_BOUND, 0.0, out)


def _nb_weights_1(mu, phi, kappa):
    """anbinomdevc_1 (kappa=False) or knbinomdevc_1 (kappa=True): phi < 0.736."""
    from edgepython import ql_weights as q

    w1, w2, w3 = (
        (q._nb_k_1_1, q._nb_k_1_2, q._nb_k_1_3)
        if kappa
        else (
            q._nb_a_1_1,
            q._nb_a_1_2,
            q._nb_a_1_3,
        )
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        y = phi / 0.368 - 1.0
        tail = (
            1.0 - 1.0 / (2.5 * mu * mu)
            if kappa
            else (1.0 - 1.0 / (6.0 * mu) - 1.0 / (2.0 * mu * mu))
        )
        out = np.empty_like(mu)
        hi = mu > 60
        if kappa:
            it = np.select([mu > 250, mu > 120, mu > 80], [18, 12, 6], 0)
        else:
            it = np.select([mu > 120, mu > 80], [12, 6], 0)
        out[hi] = _cheb(w3, y[hi], 6, it[hi]) * tail[hi]
        mid = ~hi & (mu > 20)
        m = mu[mid]
        x, it = _pieces(
            m,
            [25.0, 30.0, 40.0],
            [
                lambda m: (2.0 * m - 45.0) / 5.0,
                lambda m: (2.0 * m - 55.0) / 5.0,
                lambda m: (m - 35.0) / 5.0,
                lambda m: (m - 50.0) / 10.0,
            ],
            [0, 7, 14, 21],
        )
        a = _cheb(w2, y[mid], 7, it)
        b = _cheb(w2, y[mid], 7, it + 7)
        out[mid] = (b + (a - b) * _cheb(w2, x, 7, it + 35)) * tail[mid]
        lo = ~hi & ~mid
        m = mu[lo]
        b3, x3 = (
            (1.30, (lambda m: (2.0 * m - 1.63) / 0.97))
            if kappa
            else (
                1.77,
                (lambda m: (2.0 * m - 2.10) / 1.44),
            )
        )
        x4 = (
            (lambda m: (2.0 * m - 5.30) / 2.70)
            if kappa
            else (lambda m: (2.0 * m - 5.77) / 2.23)
        )
        x, it = _pieces(
            m,
            [0.01, 0.33, b3, 4.0, 10.0],
            [
                lambda m: 200.0 * m - 1.0,
                lambda m: (2.0 * m - 0.34) / 0.32,
                x3,
                x4,
                lambda m: (m - 7.0) / 3.0,
                lambda m: (m - 15.0) / 5.0,
            ],
            [0, 49, 98, 147, 196, 245],
        )
        pois = _pois_kappa(m) if kappa else _pois_alpha(m)
        out[lo] = _cheb_2d(w1, x, y[lo], 7, 7, it) * pois
    return np.where(mu < LOW_BOUND, 0.0, out)


def _nb_weights_2(mu, phi, kappa):
    """anbinomdevc_2 (kappa=False) or knbinomdevc_2 (kappa=True): phi >= 0.736."""
    from edgepython import ql_weights as q

    w1, w2 = (q._nb_k_2_1, q._nb_k_2_2) if kappa else (q._nb_a_2_1, q._nb_a_2_2)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = phi / 2.0 - 1.0
        out = np.empty_like(mu)
        lo = mu < 50
        m = mu[lo]
        bounds = (
            [0.01, 0.50, 3.88, 10.0, 30.0] if kappa else [0.01, 0.43, 3.62, 10.0, 30.0]
        )
        x2, x3, x4 = (
            (
                lambda m: (2.0 * m - 0.51) / 0.49,
                lambda m: (2.0 * m - 4.38) / 3.38,
                lambda m: (2.0 * m - 13.88) / 6.12,
            )
            if kappa
            else (
                lambda m: (m - 0.22) / 0.21,
                lambda m: (2.0 * m - 4.05) / 3.19,
                lambda m: (2.0 * m - 13.62) / 6.38,
            )
        )
        x, it = _pieces(
            m,
            bounds,
            [
                lambda m: m / 0.01 - 1.0,
                x2,
                x3,
                x4,
                lambda m: m / 10.0 - 2.0,
                lambda m: m / 10.0 - 4.0,
            ],
            [0, 100, 200, 300, 400, 500],
        )
        o = _cheb_2d(w1, x, y[lo], 10, 10, it)
        if kappa:
            logmu = np.log(m) / (1.0 + np.log(m))
            o = np.where(m < 0.01, o * m * logmu * logmu, o)
        else:
            logmu = np.log(m)
            o = np.where(m < 0.01, o * (logmu / ((1.0 + logmu) * (1.0 + logmu))), o)
        out[lo] = o
        mid = ~lo & (mu < 5000)
        x, it = _pieces(
            mu[mid],
            [100.0, 1000.0],
            [
                lambda m: 200.0 / m - 3.0,
                lambda m: (2000.0 / m - 11.0) / 9.0,
                lambda m: 2500.0 / m - 1.5,
            ],
            [0, 10, 20],
        )
        a = _cheb(w2, y[mid], 10, it)
        b = _cheb(w2, y[mid], 10, it + 10)
        out[mid] = a + (b - a) * _cheb(w2, x, 10, it + 40)
        hi = ~lo & ~mid
        out[hi] = _cheb(w2, y[hi], 10, 30)
    return np.where(mu < LOW_BOUND, 0.0, out)


def compute_weight(u, phi, prior):
    """QL weights (alpha, kappa), elementwise (ql_weights.c compute_weight_negbin).

    As edgepython.ql_weights.compute_weight: Chebyshev approximations, the
    dispersion capped at 4 in the second case.
    """
    mu, phi = np.broadcast_arrays(np.asarray(u / prior, float), np.asarray(phi, float))
    alpha, kappa = np.empty(mu.shape), np.empty(mu.shape)
    c1 = phi < 0.736
    for m, nb in ((c1, _nb_weights_1), (~c1, _nb_weights_2)):
        ph = phi[m] if m is c1 else np.minimum(phi[m], 4.0)
        alpha[m] = nb(mu[m], ph, kappa=False)
        kappa[m] = nb(mu[m], ph, kappa=True)
    return alpha, kappa


def _compute_adjust_inner(
    y, mu, design, disp_mat, w_mat, prior, dev_out, df_out, s2_out
):
    """QL adjusted deviance, df and s2 per gene (ql_glm.c compute_adjust_vec).

    As edgepython.ql_weights._compute_adjust_inner, writing into dev_out,
    df_out and s2_out.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        zwpt = np.sqrt(mu * w_mat / (1.0 + mu * disp_mat / prior))
    hat = _hat_values(design, zwpt)
    alpha, kappa = compute_weight(mu, disp_mat, prior / w_mat)
    udp = unit_nb_deviance(y, mu, disp_mat * w_mat / prior)
    hdp = 1.0 - hat
    zero = hdp < THRESHOLD_ZERO
    udp = np.where(zero, 0.0, udp)
    hdp = np.where(zero, 0.0, hdp)
    dev_out[:] = _rowsum(udp * alpha * w_mat)
    df_out[:] = _rowsum(hdp * kappa)
    with np.errstate(divide="ignore", invalid="ignore"):
        s2_out[:] = np.where(df_out < THRESHOLD_ZERO, 0.0, dev_out / df_out)


# limma's weighted lowess (weighted_lowess.c), used by squeezeVar's robust trend


def _seqsum(a):
    """Sum accumulated left to right, as the C loops (cumsum is sequential)."""
    return np.cumsum(a)[-1] if a.size else 0.0


def _find_limits(seed_idx, nseeds, xs, ws, n, span_weight):
    """Span window and its half-width around each seed (weighted_lowess.c find_limits).

    As edgepython.weighted_lowess._find_limits: the window grows by the nearer
    neighbour (the right one on ties) until it holds span_weight, then extends
    to ties. The growth order is a stable sort of the right and left distances.
    """
    frame_start = np.empty(nseeds, dtype=np.intp)
    frame_end = np.empty(nseeds, dtype=np.intp)
    max_dist = np.empty(nseeds, dtype=np.float64)
    for s in range(nseeds):
        c = int(seed_idx[s])
        dist = np.concatenate(
            [xs[c + 1 : n] - xs[c], xs[c] - xs[c - 1 :: -1] if c else []]
        )
        wts = np.concatenate([ws[c + 1 : n], ws[c - 1 :: -1] if c else []])
        order = np.argsort(dist, kind="stable")
        cum = np.cumsum(np.concatenate([[ws[c]], wts[order]]))
        reached = cum >= span_weight
        k = int(np.argmax(reached)) if reached.any() else order.size
        nright = int(np.count_nonzero(order[:k] < n - 1 - c))
        left, right = c - (k - nright), c + nright
        while left > 0 and xs[left] == xs[left - 1]:
            left -= 1
        while right < n - 1 and xs[right] == xs[right + 1]:
            right += 1
        frame_start[s], frame_end[s] = left, right
        max_dist[s] = max(0.0, dist[order[k - 1]]) if k else 0.0
    return frame_start, frame_end, max_dist


def _lowess_fit(xs, ys, ws, rw, curpt, left, right, dist):
    """Local weighted linear fit at xs[curpt] (weighted_lowess.c lowess_fit).

    As edgepython.weighted_lowess._lowess_fit: tricube weights times prior and
    robustness weights over the window [left, right].
    """
    x, y = xs[left : right + 1], ys[left : right + 1]
    w = ws[left : right + 1] * rw[left : right + 1]
    if dist < 1e-7:
        allweight = _seqsum(w)
        return 0.0 if allweight == 0.0 else _seqsum(y * w) / allweight
    u = np.abs(xs[curpt] - x) / dist
    tricube = 1.0 - u * u * u
    w = tricube * tricube * tricube * w
    allweight = _seqsum(w)
    if allweight == 0.0:
        return 0.0
    xmean = _seqsum(w * x) / allweight
    ymean = _seqsum(w * y) / allweight
    temp = x - xmean
    var = _seqsum(temp * temp * w)
    covar = _seqsum(temp * (y - ymean) * w)
    if var < 1e-7:
        return ymean
    slope = covar / var
    return slope * xs[curpt] + ymean - slope * xmean
