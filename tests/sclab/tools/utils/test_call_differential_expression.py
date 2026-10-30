import numpy as np
import pandas as pd
import pytest

from sclab.tools.utils import call_differential_expression
from sclab.tools.utils._call_differential_expression import _robust_z


@pytest.fixture
def table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    lfc = rng.normal(0, 0.3, n)
    lfc[:5] = 3.0  # strongly up
    lfc[5:10] = -3.0  # strongly down
    df = pd.DataFrame(
        {
            "lfc": lfc,
            "padj": np.full(n, 0.001),
            "pct_a": 0.5,
            "pct_b": 0.01,
        },
        index=[f"g{i}" for i in range(n)],
    )
    df.loc["g0", "padj"] = 0.5  # not significant
    df.loc["g1", ["pct_a", "pct_b"]] = 0.01  # lowly expressed
    return df


class TestRobustZ:
    def test_median_is_zero_and_scale_is_mad_based(self):
        x = pd.Series(np.arange(1.0, 10.0))

        z = _robust_z(x)

        assert z.median() == 0
        # MAD of 1..9 is 2 -> sigma_r = 1.4826 * 2
        np.testing.assert_allclose(z, (x - 5) / (1.4826 * 2))

    def test_robust_to_outliers(self):
        x = pd.Series(np.r_[np.random.default_rng(0).normal(size=100), 1e6])

        z = _robust_z(x)

        assert abs(z.iloc[:-1]).max() < 5
        assert z.iloc[-1] > 1e5

    def test_constant_input_gives_non_finite(self):
        z = _robust_z(pd.Series([1.0, 1.0, 1.0]))

        assert not np.isfinite(z).any()


class TestCall:
    def test_calls_up_and_down(self, table):
        res = call_differential_expression(table, "padj", "lfc", copy=True)

        assert res.loc[["g2", "g3", "g4"], "DE"].tolist() == [1, 1, 1]
        assert res.loc[[f"g{i}" for i in range(5, 10)], "DE"].tolist() == [-1] * 5

    def test_each_filter_is_applied(self, table):
        res = call_differential_expression(table, "padj", "lfc", copy=True)

        assert res.loc["g0", "DE"] == 0  # p-value filter
        assert res.loc["g1", "DE"] == 0  # expression filter
        assert res.loc["g0", "robust_Z"] > 2.5 and res.loc["g1", "robust_Z"] > 2.5
        assert (res.loc[res["robust_Z"].abs() <= 2.5, "DE"] == 0).all()  # effect size
        assert set(res["DE"].unique()) <= {-1, 0, 1}

    def test_copy_true_leaves_input_untouched(self, table):
        cols = list(table.columns)

        res = call_differential_expression(table, "padj", "lfc", copy=True)

        assert list(table.columns) == cols
        assert {"robust_Z", "DE"} <= set(res.columns)
        assert res is not table

    def test_copy_false_modifies_in_place_and_returns_none(self, table):
        out = call_differential_expression(table, "padj", "lfc")

        assert out is None
        assert {"robust_Z", "DE"} <= set(table.columns)

    def test_de_column_is_integer(self, table):
        res = call_differential_expression(table, "padj", "lfc", copy=True)

        assert pd.api.types.is_integer_dtype(res["DE"])

    def test_robust_z_values(self, table):
        res = call_differential_expression(table, "padj", "lfc", copy=True)

        np.testing.assert_allclose(res["robust_Z"], _robust_z(table["lfc"]))

    def test_thresholds_are_configurable(self, table):
        strict = call_differential_expression(
            table, "padj", "lfc", max_pval=1e-5, copy=True
        )
        assert (strict["DE"] == 0).all()

        high_z = call_differential_expression(
            table, "padj", "lfc", min_robust_z_level=1000, copy=True
        )
        assert (high_z["DE"] == 0).all()

        loose_pct = call_differential_expression(
            table, "padj", "lfc", min_pct=0.0, copy=True
        )
        assert loose_pct.loc["g1", "DE"] == 1

    def test_pvalue_threshold_is_strict(self, table):
        table.loc["g2", "padj"] = 0.05

        res = call_differential_expression(table, "padj", "lfc", copy=True)

        assert res.loc["g2", "DE"] == 0

    def test_max_over_pct_columns_with_custom_prefix(self, table):
        table = table.rename(columns={"pct_a": "frac_a", "pct_b": "frac_b"})
        table.loc["g1", "frac_a"] = 0.9

        res = call_differential_expression(
            table, "padj", "lfc", pct_col_prefix="frac_", copy=True
        )

        # any column above min_pct is enough
        assert res.loc["g1", "DE"] == 1

    def test_pct_prefix_matches_only_start_of_column_name(self, table):
        table["other_pct_x"] = 0.99  # must not count as a pct_ column

        res = call_differential_expression(table, "padj", "lfc", copy=True)

        assert res.loc["g1", "DE"] == 0

    def test_without_pct_columns_nothing_is_called(self, table):
        table = table[["lfc", "padj"]]

        res = call_differential_expression(table, "padj", "lfc", copy=True)

        assert (res["DE"] == 0).all()


class TestContrastKey:
    @pytest.fixture
    def two_contrasts(self, table) -> pd.DataFrame:
        # contrast v has a much wider lfc distribution than u
        rng = np.random.default_rng(1)
        table["contrast"] = np.repeat(["u", "v"], len(table) // 2)
        table["lfc"] = np.where(
            table["contrast"] == "u",
            rng.normal(0, 0.1, len(table)),
            rng.normal(0, 1.0, len(table)),
        )
        table.loc["g50", "lfc"] = 0.8  # outlier in u, unremarkable in v's scale
        table.loc["g150", "lfc"] = 0.8
        return table

    def test_z_scores_are_computed_within_groups(self, two_contrasts):
        res = call_differential_expression(
            two_contrasts, "padj", "lfc", contrast_key="contrast", copy=True
        )

        for _, grp in res.groupby("contrast"):
            np.testing.assert_allclose(grp["robust_Z"], _robust_z(grp["lfc"]))
        assert res.loc["g50", "robust_Z"] > 2.5 > res.loc["g150", "robust_Z"]
        assert res.loc["g50", "DE"] == 1
        assert res.loc["g150", "DE"] == 0

    def test_global_scoring_differs_from_grouped(self, two_contrasts):
        grouped = call_differential_expression(
            two_contrasts, "padj", "lfc", contrast_key="contrast", copy=True
        )
        pooled = call_differential_expression(two_contrasts, "padj", "lfc", copy=True)

        assert not np.allclose(grouped["robust_Z"], pooled["robust_Z"])

    def test_row_order_is_preserved(self, two_contrasts):
        shuffled = two_contrasts.sample(frac=1.0, random_state=0)

        a = call_differential_expression(
            two_contrasts, "padj", "lfc", contrast_key="contrast", copy=True
        )
        b = call_differential_expression(
            shuffled, "padj", "lfc", contrast_key="contrast", copy=True
        )

        assert list(b.index) == list(shuffled.index)
        pd.testing.assert_series_equal(
            b["robust_Z"].loc[a.index], a["robust_Z"], check_names=False
        )
