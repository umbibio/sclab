from ipywidgets import FloatSlider

from sclab.dataset.processor import Processor
from sclab.dataset.processor._codegen import ADATA, call
from sclab.dataset.processor.step import ProcessorStepBase
from sclab.preprocess._scanpy_port import native_backend, resolve_backend
from sclab.tools.cluster import leiden

from ._neighbors import missing_neighbors


class Cluster(ProcessorStepBase):
    """Leiden graph-based clustering step.

    Runs :func:`sclab.tools.cluster.leiden`: partitions cells into clusters
    using the Leiden algorithm (igraph) on the precomputed neighbor graph.
    The resulting cluster labels are stored in ``adata.obs["leiden"]`` and
    published to the plotter as a metadata change.

    Uses scanpy if it is installed, else sclab's native port of
    :func:`scanpy.tl.leiden` (``backend="auto"``; the same igraph call and
    seeding, so the same labels); the history records which ran.
    """

    parent: Processor
    name: str = "cluster"
    description: str = "Cluster"
    background = True

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
        return missing_neighbors(self.parent.dataset.adata)

    def tool(self, params: dict) -> str:
        if resolve_backend("auto") == "scanpy":
            return "Python · scanpy.tl.leiden (igraph)"
        return "Python · sclab (port of scanpy.tl.leiden, igraph)"

    def resolved_params(self, params: dict) -> dict:
        return {**super().resolved_params(params), "backend": resolve_backend("auto")}

    def to_code(self, params: dict) -> str:
        return call(
            "sclab.tools.cluster.leiden",
            ADATA,
            resolution=params["resolution"],
            random_state=params["random_state"],
            **native_backend(params),
        )

    def function(self, resolution: float = 1.0, random_state: int = 0):
        dataset = self.parent.dataset
        yield from self.compute(
            leiden,
            resolution=resolution,
            random_state=random_state,
        )

        self.broker.publish("dset_metadata_change", dataset.metadata, "leiden")
