import sys

import numpy as np
import pandas as pd
import pytest

from sclab.tools.differential_expression import (
    _pseudobulk_edger as edger_mod,
)
from sclab.tools.differential_expression import edger_is_available, pseudobulk_edger

from ._fake_r import install_fake_r

TT_COLUMNS = ["logFC", "logCPM", "F", "PValue", "FDR"]
needs_r = pytest.mark.skipif(
    not edger_is_available(), reason="requires R with edgeR/MAST/SingleCellExperiment"
)


class TestAvailability:
    def test_returns_bool(self):
        assert isinstance(edger_is_available(), bool)

    def test_python_deps_check_uses_find_spec(self, monkeypatch):
        monkeypatch.setattr(edger_mod, "find_spec", lambda name: object())
        assert edger_mod._python_deps_available() is True

        monkeypatch.setattr(
            edger_mod,
            "find_spec",
            lambda name: None if name == "anndata2ri" else object(),
        )
        assert edger_mod._python_deps_available() is False

    def test_unavailable_without_python_deps(self, monkeypatch):
        monkeypatch.setattr(edger_mod, "find_spec", lambda name: None)

        assert edger_mod._python_deps_available() is False
        assert edger_mod._r_deps_available() is False
        assert edger_is_available() is False

    def test_declared_dependencies(self):
        assert edger_mod.PYTHON_DEPENDENCIES == ["rpy2", "anndata2ri"]
        assert {"edgeR", "MAST", "SingleCellExperiment"} == set(
            edger_mod.R_DEPENDENCIES
        )

    def test_batch_key_is_not_a_parameter(self):
        import inspect

        assert "batch_key" not in inspect.signature(pseudobulk_edger).parameters


def test_missing_rpy2_raises_helpful_error(monkeypatch, counts_adata):
    monkeypatch.setitem(sys.modules, "rpy2", None)  # import rpy2 raises

    with pytest.raises(
        ImportError, match=r"Needs rpy2, anndata2ri: pip install 'sclab\[r\]'"
    ):
        pseudobulk_edger(counts_adata, "cond", replicas_per_group=2)


@pytest.mark.skipif(
    edger_is_available(), reason="R packages are installed; error path not reachable"
)
def test_missing_r_packages_raises_when_unavailable(counts_adata):
    pytest.importorskip("rpy2")
    pytest.importorskip("anndata2ri")

    with pytest.raises((ImportError, ModuleNotFoundError)):
        pseudobulk_edger(counts_adata, "cond", replicas_per_group=2)


class TestOrchestrationWithFakeR:
    """Python-side logic, with the R layer replaced by a recording fake."""

    @pytest.fixture(autouse=True)
    def _need_rpy2(self):
        pytest.importorskip("rpy2")
        pytest.importorskip("anndata2ri")

    def test_condition_vs_reference(self, monkeypatch, counts_adata):
        fake = install_fake_r(monkeypatch, edger_mod, TT_COLUMNS)

        res = pseudobulk_edger(
            counts_adata,
            "cond",
            condition_group="A",
            reference_group="B",
            replicas_per_group=2,
        )

        assert list(res) == ["A_vs_B"]
        assert fake.contrasts() == ["groupA-groupB"]
        assert any(
            'fit_edger_model(aggr_adata, "cond", "None"' in c for c in fake.calls
        )
        tt = res["A_vs_B"]
        assert tt.index.name == "gene_ids"
        assert set(TT_COLUMNS) <= set(tt.columns)
        for col in [
            "pct_expr_cnd",
            "pct_expr_ref",
            "num_expr_cnd",
            "num_expr_ref",
            "tot_expr_cnd",
            "tot_expr_ref",
            "mean_cnd",
            "mean_ref",
        ]:
            assert col in tt.columns

    def test_annotation_matches_cell_level_statistics(self, monkeypatch, counts_adata):
        install_fake_r(monkeypatch, edger_mod, TT_COLUMNS)

        res = pseudobulk_edger(
            counts_adata,
            "cond",
            condition_group="A",
            reference_group="B",
            replicas_per_group=2,
        )

        tt = res["A_vs_B"]
        X = counts_adata.X.toarray()
        is_a = (counts_adata.obs["cond"] == "A").values
        for j, gene in enumerate(counts_adata.var_names):
            assert tt.loc[gene, "tot_expr_cnd"] == X[is_a, j].sum()
            assert tt.loc[gene, "tot_expr_ref"] == X[~is_a, j].sum()
            assert tt.loc[gene, "num_expr_cnd"] == (X[is_a, j] > 0).sum()
            assert tt.loc[gene, "pct_expr_cnd"] == pytest.approx(
                (X[is_a, j] > 0).mean()
            )
            assert tt.loc[gene, "mean_ref"] == pytest.approx(
                X[~is_a, j].sum() / (X[~is_a, j] > 0).sum()
            )

    def test_default_contrasts_each_group_vs_rest(self, monkeypatch, counts_adata):
        fake = install_fake_r(monkeypatch, edger_mod, TT_COLUMNS)

        res = pseudobulk_edger(counts_adata, "cond", replicas_per_group=2)

        assert set(res) == {"A_vs_rest", "B_vs_rest"}
        assert set(fake.contrasts()) == {"groupA-groupnotA", "groupB-groupnotB"}
        # one model is fit per condition group on its dummy-coded column
        assert any('"cond_A"' in c for c in fake.calls)
        assert any('"cond_B"' in c for c in fake.calls)

    def test_condition_group_list_skips_reference(self, monkeypatch, counts_adata):
        install_fake_r(monkeypatch, edger_mod, TT_COLUMNS)

        res = pseudobulk_edger(
            counts_adata,
            "cond",
            condition_group=["A", "B"],
            reference_group="B",
            replicas_per_group=2,
        )

        assert list(res) == ["A_vs_B"]

    def test_cell_identity_makes_one_contrast_per_identity(
        self, monkeypatch, counts_adata
    ):
        fake = install_fake_r(monkeypatch, edger_mod, TT_COLUMNS)

        res = pseudobulk_edger(
            counts_adata,
            "cond",
            condition_group="A",
            reference_group="B",
            cell_identity_key="ctype",
            replicas_per_group=2,
        )

        assert set(res) == {"ctype:x|A_vs_B", "ctype:y|A_vs_B"}
        assert set(fake.contrasts()) == {"groupA_x-groupB_x", "groupA_y-groupB_y"}
        assert any('"ctype"' in c for c in fake.calls if "fit_edger_model(" in c)
        # statistics are specific to the identity
        x = counts_adata.obs["ctype"] == "x"
        a = counts_adata.obs["cond"] == "A"
        X = counts_adata.X.toarray()
        tt = res["ctype:x|A_vs_B"]
        assert tt.loc["gene0", "tot_expr_cnd"] == X[(a & x).values, 0].sum()

    def test_use_cells_restricts_aggregation(self, monkeypatch, counts_adata):
        install_fake_r(monkeypatch, edger_mod, TT_COLUMNS)

        res = pseudobulk_edger(
            counts_adata,
            "cond",
            condition_group="A",
            reference_group="B",
            use_cells={"ctype": ["x"]},
            replicas_per_group=2,
        )

        X = counts_adata.X.toarray()
        mask = (
            (counts_adata.obs["cond"] == "A") & (counts_adata.obs["ctype"] == "x")
        ).values
        assert res["A_vs_B"].loc["gene0", "tot_expr_cnd"] == X[mask, 0].sum()

    def test_aggregated_object_has_requested_replicas(self, monkeypatch, counts_adata):
        fake = install_fake_r(monkeypatch, edger_mod, TT_COLUMNS)

        pseudobulk_edger(
            counts_adata,
            "cond",
            condition_group="A",
            reference_group="B",
            replicas_per_group=3,
        )

        aggr = fake.assigned["aggr_adata"]
        assert aggr.n_obs == 6
        assert set(aggr.obs["replica"]) == {"1", "2", "3"}
        assert aggr.obs["replica_size"].sum() == counts_adata.n_obs

    def test_r_failure_skips_condition_and_continues(
        self, monkeypatch, counts_adata, capsys
    ):
        # fail only for the model of condition A
        install_fake_r(monkeypatch, edger_mod, TT_COLUMNS, fail_on='"cond_A"')

        res = pseudobulk_edger(counts_adata, "cond", replicas_per_group=2)

        assert set(res) == {"B_vs_rest"}
        out = capsys.readouterr().out
        assert "Error fitting model for A" in out and "Skipping" in out

    def test_verbose_progress_messages(self, monkeypatch, counts_adata, capsys):
        install_fake_r(monkeypatch, edger_mod, TT_COLUMNS)

        pseudobulk_edger(
            counts_adata,
            "cond",
            condition_group="A",
            reference_group="B",
            replicas_per_group=2,
            verbosity=1,
        )

        out = capsys.readouterr().out
        assert "Fitting model for A" in out
        assert "Computing contrast: A_vs_B" in out


@needs_r
class TestWithR:
    def test_reproducible_by_default(self, r_counts_adata):
        kw = dict(condition_group="A", replicas_per_group=3, min_cells_per_group=10)
        a = pseudobulk_edger(r_counts_adata, "cond", **kw)["A_vs_rest"]
        b = pseudobulk_edger(r_counts_adata, "cond", **kw)["A_vs_rest"]
        pd.testing.assert_frame_equal(a, b)

    def test_detects_upregulated_genes(self, r_counts_adata):
        res = pseudobulk_edger(
            r_counts_adata,
            "cond",
            condition_group="A",
            reference_group="B",
            replicas_per_group=3,
            min_cells_per_group=10,
        )

        tt = res["A_vs_B"]
        assert {"logFC", "PValue", "FDR", "pct_expr_cnd", "pct_expr_ref"} <= set(
            tt.columns
        )
        top = tt.sort_values("PValue").index[:2]
        assert set(top) == {"gene0", "gene1"}
        assert (tt.loc[["gene0", "gene1"], "logFC"] > 0.5).all()
        assert (tt.loc[["gene0", "gene1"], "FDR"] < 0.05).all()

    def test_vs_rest_and_cell_identity(self, r_counts_adata):
        res = pseudobulk_edger(
            r_counts_adata,
            "cond",
            cell_identity_key="ctype",
            replicas_per_group=3,
            min_cells_per_group=10,
        )

        assert "ctype:x|A_vs_rest" in res
        tt = res["ctype:x|A_vs_rest"]
        assert tt.loc["gene0", "logFC"] > 0

    def test_single_replicate_drops_replica_term(self, r_three_group_adata):
        res = pseudobulk_edger(
            r_three_group_adata,
            "cond",
            condition_group="A",
            replicas_per_group=1,
            min_cells_per_group=10,
            verbosity=2,  # also exercises the R-side cat() calls
        )

        tt = res["A_vs_rest"]
        assert np.isfinite(tt["PValue"]).all()
        assert set(tt.sort_values("PValue").index[:2]) == {"gene0", "gene1"}
