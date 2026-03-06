from ipywidgets import Dropdown

from sclab.dataset.processor import Processor
from sclab.dataset.processor.step import ProcessorStepBase
from sclab.tools.doublet_detection import (
    doubletdetection,
    scdblfinder,
    scrublet,
)


class DoubletDetection(ProcessorStepBase):
    """Doublet detection step.

    Identifies putative doublet cells using Scrublet. Scores and binary
    labels (``"singlet"`` / ``"doublet"``) are added to ``adata.obs``
    under keys ``{flavor}_score`` and ``{flavor}_label``.

    Requires the ``scrublet`` package to be installed
    (``pip install scrublet``).
    """

    parent: Processor
    name: str = "doublet_detection"
    description: str = "Doublet Detection"

    def __init__(self, parent: Processor) -> None:
        from sclab.tools.doublet_detection import (
            any_doublet_detection_available,
            get_available_doublet_detection_methods,
        )

        if not any_doublet_detection_available():
            raise ImportError

        methods_available = get_available_doublet_detection_methods()

        variable_controls = dict(
            layer=Dropdown(
                options=tuple(parent.dataset.adata.layers.keys()),
                value=None,
                description="Layer",
            ),
            flavor=Dropdown(
                options=methods_available,
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
                case "scDblFinder":
                    scdblfinder(**kvargs, clusters_col="leiden")

                case "doubletdetection":
                    doubletdetection(
                        **kvargs,
                        pseudocount=1,
                        clustering_algorithm="leiden",
                        clustering_kwargs=dict(resolution=5.0),
                    )

                case "scrublet":
                    scrublet(**kvargs)

                case _:
                    raise ValueError(f"Unknown flavor: {flavor}")

        self.broker.publish(
            "dset_metadata_change",
            self.parent.dataset.metadata,
            f"{flavor}_label",
        )
