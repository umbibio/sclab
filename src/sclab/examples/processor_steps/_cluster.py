from ipywidgets import FloatSlider

from sclab._citations import record_citations
from sclab._compat import missing
from sclab.dataset.processor import Processor
from sclab.dataset.processor._codegen import ADATA, call
from sclab.dataset.processor.step import ProcessorStepBase

from ._neighbors import missing_neighbors


class Cluster(ProcessorStepBase):
    """Leiden graph-based clustering step.

    Partitions cells into clusters using the Leiden algorithm on the
    precomputed neighbor graph. The resulting cluster labels are stored in
    ``adata.obs["leiden"]`` and published to the plotter as a metadata
    change.

    Requires ``scanpy`` (≥1.9 with igraph flavor support) to be installed.
    """

    parent: Processor
    name: str = "cluster"
    description: str = "Cluster"
    background = True
    leiden_kwargs = dict(flavor="igraph", n_iterations=2, directed=False)

    def __init__(self, parent: Processor) -> None:
        variable_controls = dict(
            resolution=FloatSlider(
                value=1.0, min=0.1, max=10.0, step=0.1, description="Resolution"
            )
        )

        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )

    def validate(self, params: dict) -> str | None:
        return missing("scanpy") or missing_neighbors(self.parent.dataset.adata)

    def to_code(self, params: dict) -> str:
        return call(
            "sc.tl.leiden",
            ADATA,
            resolution=params["resolution"],
            **self.leiden_kwargs,
            random_state=params["random_state"],
        )

    def function(self, resolution: float = 1.0, random_state: int = 0):
        import scanpy as sc

        dataset = self.parent.dataset
        yield from self.compute(
            sc.tl.leiden,
            resolution=resolution,
            **self.leiden_kwargs,
            random_state=random_state,
        )
        record_citations(dataset.adata, "scanpy", "leiden", "igraph")

        self.broker.publish("dset_metadata_change", dataset.metadata, "leiden")
