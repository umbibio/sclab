from ipywidgets import Dropdown

from sclab.dataset.processor import Processor
from sclab.dataset.processor.step import ProcessorStepBase
from sclab.tools.doublet_detection import scrublet

# from sclab.tools.doublet_detection import doubletdetection
# from sclab.tools.doublet_detection import scdblfinder


class DoubletDetection(ProcessorStepBase):
    parent: Processor
    name: str = "doublet_detection"
    description: str = "Doublet Detection"

    def __init__(self, parent: Processor) -> None:
        variable_controls = dict(
            layer=Dropdown(
                options=tuple(parent.dataset.adata.layers.keys()),
                value=None,
                description="Layer",
            ),
            flavor=Dropdown(
                options=[
                    "scrublet",
                    # "doubletdetection",
                    # "scDblFinder",
                ],
                description="Flavor",
            ),
        )

        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )

    def function(self, layer: str, flavor: str):
        adata = self.parent.dataset.adata

        kvargs = {"adata": adata, "layer": layer, "key_added": flavor}

        self.broker.std_output.clear_output(wait=False)
        with self.broker.std_output:
            match flavor:
                # case "scDblFinder":
                #     scdblfinder(**kvargs, clusters_col="leiden")

                # case "doubletdetection":
                #     doubletdetection(
                #         **kvargs,
                #         pseudocount=1,
                #         clustering_algorithm="leiden",
                #         clustering_kwargs=dict(resolution=5.0),
                #     )

                case "scrublet":
                    scrublet(**kvargs)

                case _:
                    raise ValueError(f"Unknown flavor: {flavor}")

        self.broker.publish(
            "dset_metadata_change",
            self.parent.dataset.metadata,
            f"{flavor}_label",
        )
