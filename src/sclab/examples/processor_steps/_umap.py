from ipywidgets import Checkbox, IntText

from sclab._compat import missing
from sclab.dataset.processor import Processor
from sclab.dataset.processor._codegen import ADATA, call
from sclab.dataset.processor.step import ProcessorStepBase

from ._neighbors import missing_neighbors
from sclab.tools.embedding import umap


class UMAP(ProcessorStepBase):
    """UMAP dimensionality reduction step.

    Runs :func:`sclab.tools.embedding.umap`: computes a 2D or 3D UMAP
    embedding using the precomputed neighbor graph and stores it in ``adata.obsm`` under the key ``X_{n_components}Dumap``
    (e.g. ``"X_2Dumap"``). Optionally zero-centers the embedding. The
    plotter is updated to display the new embedding.

    Requires ``scanpy`` to be installed.
    """

    parent: Processor
    name: str = "umap"
    description: str = "UMAP"
    background = True

    def __init__(self, parent: Processor) -> None:
        variable_controls = dict(
            n_components=IntText(value=2, description="N comp."),
            zero_center=Checkbox(value=False, description="Zero center"),
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
            "sclab.tools.embedding.umap",
            ADATA,
            n_components=params["n_components"],
            zero_center=params["zero_center"],
            random_state=params["random_state"],
        )

    def function(
        self,
        n_components: int = 2,
        zero_center: bool = True,
        random_state: int = 0,
    ):
        dataset = self.parent.dataset
        key = yield from self.compute(
            umap,
            n_components=n_components,
            zero_center=zero_center,
            random_state=random_state,
        )

        self.broker.publish("dset_data_dict_change", dataset.data_dict, key)
