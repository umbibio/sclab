from ipywidgets import FloatSlider

from sclab.dataset.processor import Processor
from sclab.dataset.processor.step import ProcessorStepBase


class Cluster(ProcessorStepBase):
    parent: Processor
    name: str = "cluster"
    description: str = "Cluster"

    def __init__(self, parent: Processor) -> None:
        try:
            import scanpy as sc  # noqa: F401
        except ImportError:
            raise ImportError("Please install scanpy: `pip install scanpy`")

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

    def function(self, resolution: float = 1.0):
        import scanpy as sc

        dataset = self.parent.dataset
        adata = self.parent.dataset.adata
        sc.tl.leiden(adata, resolution=resolution)

        self.broker.publish("dset_metadata_change", dataset.metadata, "leiden")
