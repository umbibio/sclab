import numpy as np
import pandas as pd


def _sigmoid_wave(x, t0, r0, t1, r1):
    """Compute a periodic sigmoid-based wave with 2π wrapping."""
    result = np.zeros_like(x, dtype=float)
    for shift in [0, 2 * np.pi, -2 * np.pi]:
        result += 1 / (1 + np.exp(-(x - t0 + shift) * r0)) - 1 / (
            1 + np.exp(-(x - t1 + shift) * r1)
        )
    return result


def _sample_cell_times(
    rng, n_obs, n_phases=4, n_stages_per_phase=5, checkpoint_strength=0.3
):
    """Sample observation times from a cell-cycle-like density profile."""
    rates = np.array(
        [
            2
            ** (
                i // n_stages_per_phase
                - checkpoint_strength
                * (i % n_stages_per_phase > (n_stages_per_phase - 3))
            )
            for i in range(n_phases * n_stages_per_phase)
        ]
    )

    density = 1.0 / rates
    density /= density.sum()

    bins = np.linspace(0, 2 * np.pi, rates.size + 1)[:-1]
    t = rng.choice(bins, p=density, size=n_obs)
    offset = rng.uniform(high=2 * np.pi / rates.size, size=n_obs)
    return t + offset


def _generate_expression_matrix(rng, t, n_vars, sparsity=0.99):
    """Build a sparse count matrix from sigmoid waves with Poisson sampling."""
    from scipy.sparse import csr_matrix

    gene_onsets = rng.uniform(0, 2 * np.pi, size=n_vars)
    rng.shuffle(gene_onsets)

    X = np.array([_sigmoid_wave(t, t0, 10, t0 + 1, 10) for t0 in gene_onsets]).T
    X = rng.poisson(X * 100, size=X.shape).astype(float)

    # Apply dropout sparsity
    zero_idx = rng.integers(X.size, size=int(X.size * sparsity))
    X.flat[zero_idx] = 0.0

    return csr_matrix(X), gene_onsets


def _build_obsm(rng, t, noise_scale=0.02):
    """Create noisy circular embeddings at multiple harmonic frequencies."""
    s = noise_scale
    n = t.size
    x = np.sin(t + rng.normal(0, s, n)) + rng.normal(0, s, n)
    y = np.cos(t + rng.normal(0, s, n)) + rng.normal(0, s, n)

    obsm = {}
    for m in [2, 4, 6, 8]:
        z = np.sin((t + rng.normal(0, s, n)) * m) / m + rng.normal(0, s, n)
        Y = np.column_stack([x, y]) if m == 2 else np.column_stack([x, y, z])
        obsm[f"Y_array{m}"] = Y

    return obsm


def _build_obs(rng, t):
    """Construct the observation DataFrame with quadrant labels and metadata."""
    obs = pd.DataFrame(index=t)
    obs["quadrant"] = pd.NA
    obs["time"] = t
    obs["poisson"] = rng.poisson(5, t.size)

    # Assign quadrant based on angular position
    boundaries = [(0, "I"), (np.pi / 2, "II"), (np.pi, "III"), (3 * np.pi / 2, "IV")]
    for threshold, label in boundaries:
        obs.loc[obs.index >= threshold, "quadrant"] = label

    # Randomly mask half the quadrant labels
    obs.loc[obs.sample(frac=0.5, random_state=rng).index, "quadrant"] = pd.NA
    obs["quadrant"] = obs["quadrant"].astype(
        pd.CategoricalDtype(["I", "II", "III", "IV"], ordered=True)
    )
    obs.index = obs.index.astype(str)
    return obs


def _build_var(gene_onsets):
    """Construct the variable DataFrame with formatted gene names."""
    var = pd.DataFrame(index=gene_onsets)
    var.index = pd.Index(np.arange(1, gene_onsets.size + 1), dtype=int).map(
        "var_{:04d}".format
    )
    return var


def simple_loop_adata(n_obs: int = 600, n_vars: int = 200):
    from anndata import AnnData

    rng = np.random.default_rng(42)

    t = _sample_cell_times(rng, n_obs)
    X, gene_onsets = _generate_expression_matrix(rng, t, n_vars)
    obsm = _build_obsm(rng, t)
    obs = _build_obs(rng, t)
    var = _build_var(gene_onsets)

    adata = AnnData(X, obs=obs, obsm=obsm, var=var)

    return adata
