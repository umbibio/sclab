import pytest
from anndata import AnnData

from sclab import SCLabDashboard
from sclab.methods import GuidedPseudotime

from .test_utils import simple_loop_adata


@pytest.fixture(scope="class")
def adata() -> AnnData:
    return simple_loop_adata(n_obs=1000)


@pytest.fixture(scope="class")
def dashboard(adata: AnnData) -> SCLabDashboard:
    return SCLabDashboard(adata)


class TestWorkflow:
    def test_qc(self, dashboard: SCLabDashboard):
        dashboard.processor.steps["QC"].run()

    def test_preprocess(self, dashboard: SCLabDashboard):
        dashboard.processor.steps["Preprocess"].run()

    def test_pca(self, dashboard: SCLabDashboard):
        dashboard.processor.steps["PCA"].run()

    def test_auto_pseudotime(self, dashboard: SCLabDashboard):
        step: GuidedPseudotime = dashboard.processor.steps["Guided Pseudotime"]
        step._automatic_periodic_path_drawing()
        step.variable_controls["roughness"].value = 1
        step.run()

    def test_neighbors(self, dashboard: SCLabDashboard):
        dashboard.processor.steps["Neighbors"].run()

    def test_cluster(self, dashboard: SCLabDashboard):
        dashboard.processor.steps["Cluster"].run()

    def test_umap(self, dashboard: SCLabDashboard):
        dashboard.processor.steps["UMAP"].run()
