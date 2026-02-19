import string

import numpy as np
from scipy.stats import norm

from sclab.tools.cellflow.pseudotime.timeseries import find_category_transitions


def test_find_category_transitions():
    for periodic in [True, False]:
        for tmax in [1, 5]:
            for n in [3, 4, 5]:
                step = tmax / n

                rng = np.random.default_rng(20)

                chars = np.array(list(string.ascii_letters))
                categories = np.array(
                    ["".join(row) for row in rng.choice(chars, size=(n, 5))]
                )

                x = rng.uniform(0, tmax, size=10000)

                # Use n means, one for each category
                mu = (np.arange(n) * step)[:, None]
                ps = norm.pdf(x, loc=mu, scale=step / 4)
                if periodic:
                    # add wrapping contribution
                    ps += norm.pdf(x, loc=mu + tmax, scale=step / 4)
                    ps += norm.pdf(x, loc=mu - tmax, scale=step / 4)

                ps /= ps.sum(axis=0, keepdims=True)

                labels = np.array([rng.choice(n, p=p) for p in ps.T])
                labels = categories[labels]

                if periodic:
                    offsets = [-step / 3, 0, step / 3]
                else:
                    offsets = [0]

                for offset in offsets:
                    pseudotime = (x - offset) % tmax
                    ground_truth = np.sort((mu.ravel() + step / 2 - offset) % tmax)
                    if not periodic:
                        # Non-periodic case has n-1 transitions between n categories
                        ground_truth = ground_truth[:-1]

                    estimated_transitions = find_category_transitions(
                        times=pseudotime,
                        labels=labels,
                        categories=categories,
                        periodic=periodic,
                        tmax=tmax,
                    )

                    assert estimated_transitions is not None, (
                        f"No result returned for periodic={periodic}, tmax={tmax}, n={n}, offset={offset:.3f}"
                    )

                    estimated = np.sort(np.asarray(estimated_transitions, dtype=float))

                    assert len(estimated) == len(ground_truth), (
                        f"Expected {len(ground_truth)} transitions, got {len(estimated)} "
                        f"for periodic={periodic}, tmax={tmax}, n={n}, offset={offset:.3f}"
                    )

                    # Match sorted estimated to sorted ground truth.
                    # For the periodic case the closest match may wrap around,
                    # so we compute the periodic distance for each pair.
                    if periodic:
                        diff = np.abs(estimated - ground_truth)
                        diff = np.minimum(diff, tmax - diff)
                    else:
                        diff = np.abs(estimated - ground_truth)

                    tolerance = (
                        step / 4
                    )  # generous but meaningful: quarter of one interval width
                    max_err = diff.max()

                    assert max_err < tolerance, (
                        f"Max transition error {max_err:.4f} exceeds tolerance {tolerance:.4f} "
                        f"for periodic={periodic}, tmax={tmax}, n={n}, offset={offset:.3f}.\n"
                        f"  ground_truth: {np.round(ground_truth, 4)}\n"
                        f"  estimated:    {np.round(estimated, 4)}\n"
                        f"  diffs:        {np.round(diff, 4)}"
                    )
