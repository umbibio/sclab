import pytest
from anndata import AnnData

from sclab import SCLabDashboard, active_citations
from sclab.dataset.processor._citations_panel import CitationsPanel
from sclab.methods import GuidedPseudotime

from .test_utils import (
    assert_widgets_serializable,
    live_widget_ids,
    run_step,
    simple_loop_adata,
)


@pytest.fixture(scope="class")
def adata() -> AnnData:
    return simple_loop_adata(n_obs=1000)


@pytest.fixture(scope="class")
def preexisting_widgets() -> set[str]:
    return live_widget_ids()


@pytest.fixture(scope="class")
def dashboard(adata: AnnData, preexisting_widgets: set[str]) -> SCLabDashboard:
    return SCLabDashboard(adata)


class TestWorkflow:
    def test_dashboard_builds(self, dashboard: SCLabDashboard):
        assert not dashboard.broker.exceptions_log

    def test_qc(self, dashboard: SCLabDashboard):
        run_step(dashboard, "QC")

    def test_preprocess(self, dashboard: SCLabDashboard):
        run_step(dashboard, "Preprocess")

    def test_pca(self, dashboard: SCLabDashboard):
        run_step(dashboard, "PCA")

    def test_auto_pseudotime(self, dashboard: SCLabDashboard):
        step: GuidedPseudotime = dashboard.processor.steps["Guided Pseudotime"]
        step._automatic_periodic_path_drawing()
        step.variable_controls["roughness"].value = 1
        run_step(dashboard, "Guided Pseudotime")

    def test_neighbors(self, dashboard: SCLabDashboard):
        run_step(dashboard, "Neighbors")

    def test_cluster(self, dashboard: SCLabDashboard):
        run_step(dashboard, "Cluster")

    def test_umap(self, dashboard: SCLabDashboard):
        run_step(dashboard, "UMAP")

    def test_citations(self, dashboard: SCLabDashboard):
        adata = dashboard.dataset.adata
        keys = [e["key"] for e in active_citations(adata, format="dict")]
        # QC/Preprocess/PCA/Neighbors/UMAP via the library, Cluster in the step
        for key in ("sclab", "anndata", "scanpy", "stuart2019", "umap", "leiden"):
            assert key in keys
        (panel,) = [
            c
            for c in dashboard.processor.history_panel.children
            if isinstance(c, CitationsPanel)
        ]
        assert "From Louvain to Leiden" in panel.view.value
        assert panel.actions.text == active_citations(adata)
        assert panel.actions.filename == "sclab_citations.bib"
        assert panel.actions.mime == "application/x-bibtex"
        bib = panel.actions.download_text
        assert bib == active_citations(adata, format="bibtex")
        assert "@article{leiden," in bib
        panel.toggle.value = True
        assert panel.view.layout.display == ""

    def test_widgets_serializable(self, preexisting_widgets: set[str]):
        assert_widgets_serializable(exclude_ids=preexisting_widgets)
