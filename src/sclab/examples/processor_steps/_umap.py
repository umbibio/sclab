import numpy as np
from ipywidgets import Checkbox, IntText

from sclab.dataset.processor import Processor
from sclab.dataset.processor.step import ProcessorStepBase


class UMAP(ProcessorStepBase):
    parent: Processor
    name: str = "umap"
    description: str = "UMAP"

    def __init__(self, parent: Processor) -> None:
        try:
            import scanpy as sc  # noqa: F401
        except ImportError:
            raise ImportError("Please install scanpy: `pip install scanpy`")

        variable_controls = dict(
            n_components=IntText(value=2, description="N comp."),
            zero_center=Checkbox(value=False, description="Zero center"),
        )

        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )

    def function(
        self,
        n_components: int = 2,
        zero_center: bool = True,
    ):
        import scanpy as sc

        dataset = self.parent.dataset
        adata = self.parent.dataset.adata

        sc.tl.umap(adata, n_components=n_components)
        X: np.ndarray = adata.obsm.pop("X_umap")
        if zero_center:
            X = X - X.mean(axis=0, keepdims=True)

        key = f"X_{n_components}Dumap"
        adata.obsm[key] = X

        self.broker.publish("dset_data_dict_change", dataset.data_dict, key)
