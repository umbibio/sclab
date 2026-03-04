import numpy as np
import pandas as pd


def simple_loop_adata(n_obs: int = 600):
    from anndata import AnnData
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(42)

    def f(x, t0, r0, t1, r1):
        out = 1 / (1 + np.exp(-(x - t0) * r0)) - 1 / (1 + np.exp(-(x - t1) * r1))
        out += 1 / (1 + np.exp(-(x - t0 + (np.pi * 2)) * r0)) - 1 / (
            1 + np.exp(-(x - t1 + (np.pi * 2)) * r1)
        )
        out += 1 / (1 + np.exp(-(x - t0 - (np.pi * 2)) * r0)) - 1 / (
            1 + np.exp(-(x - t1 - (np.pi * 2)) * r1)
        )
        return out

    # TODO: simulate batches by changing sparsity
    # sparsities = [0.9, 0.99, 0.999]
    # n_batches = len(sparsities)
    sparsity = 0.99
    n = n_obs
    m = n_vars = 200
    # obs time order
    # t = rng.uniform(0, np.pi*2, size=n_obs)

    # generate a density profile using a stochastic model
    checkpoint_strenght = 0.3
    n_phases, n_stages_per_phase = 4, 5
    rates = np.array(
        [
            2
            ** (
                i // n_stages_per_phase
                - checkpoint_strenght
                * (i % n_stages_per_phase > (n_stages_per_phase - 3))
            )
            for i in range(n_phases * n_stages_per_phase)
        ]
    )

    density = 1.0 / rates
    density /= density.sum()

    bins = np.linspace(0, 2 * np.pi, rates.size + 1)[:-1]
    t = rng.choice(bins, p=density, size=n)
    offset = rng.uniform(high=2 * np.pi / rates.size, size=n)
    t = t + offset

    # var risetime
    # tt = (np.cos(np.linspace(0, np.pi, n_vars)) + 1) * np.pi
    tt = rng.uniform(0, np.pi * 2, size=n_vars)
    rng.shuffle(tt)

    X = np.array([f(t, t0, 10, t0 + 1, 10) for t0 in tt]).T
    # X[X < 0.05] = 0.
    X = rng.poisson(X * 100, size=X.shape)

    X.flat[rng.integers(X.size, size=int(X.size * sparsity))] = 0.0
    X = csr_matrix(X)

    s = 0.02
    x = np.sin(t + rng.normal(0.0, s, n)) + rng.normal(0.0, s, n)
    y = np.cos(t + rng.normal(0.0, s, n)) + rng.normal(0.0, s, n)

    obsm = {}
    for m in [2, 4, 6, 8]:
        z = np.sin((t + rng.normal(0.0, s, n)) * m) / m + rng.normal(0.0, s, n)
        if m == 2:
            Y = np.array([x, y]).T
        else:
            Y = np.array([x, y, z]).T

        obsm[f"Y_array{m}"] = Y

    obs = pd.DataFrame(index=t)
    obs["quadrant"] = pd.NA
    obs["time"] = t
    obs["poisson"] = rng.poisson(5, t.size)
    obs.loc[obs.index >= 0, "quadrant"] = "I"
    obs.loc[obs.index >= np.pi / 2, "quadrant"] = "II"
    obs.loc[obs.index >= np.pi, "quadrant"] = "III"
    obs.loc[obs.index >= 3 * np.pi / 2, "quadrant"] = "IV"
    obs.loc[obs.sample(frac=0.5).index, "quadrant"] = pd.NA
    obs["quadrant"] = obs["quadrant"].astype(
        pd.CategoricalDtype(["I", "II", "III", "IV"], ordered=True)
    )
    obs.index = obs.index.astype(str)

    var = pd.DataFrame(index=tt)
    var.index = pd.Index(np.arange(1, tt.size + 1), dtype=int).map("var_{:04d}".format)

    return AnnData(X, obs=obs, obsm=obsm, var=var)
