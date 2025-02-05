from ipywidgets import Dropdown, IntText

from sclab.dataset.processor import Processor
from sclab.dataset.processor.step import ProcessorStepBase


class Neighbors(ProcessorStepBase):
    parent: Processor

    def __init__(self, parent: Processor) -> None:
        try:
            import scanpy as sc  # noqa: F401
        except ImportError:
            raise ImportError("Please install scanpy: `pip install scanpy`")

        variable_controls = dict(
            use_rep=Dropdown(
                options=tuple(parent.dataset.adata.obsm.keys()),
                value=None,
                description="Use rep.",
            ),
            n_neighbors=IntText(value=20, description="N neighbors"),
            n_dims=IntText(value=10, description="N Dims"),
            metric=Dropdown(
                options=["euclidean", "cosine"],
                value="euclidean",
                description="Metric",
            ),
            **parent.make_groupbybatch_checkbox(),
        )

        super().__init__(
            parent=parent,
            name="neighbors",
            description="Neighbors",
            fixed_params={},
            variable_controls=variable_controls,
        )

    def function(
        self,
        n_neighbors: int = 20,
        use_rep: str = "X_pca",
        n_dims: int = 10,
        metric: str = "euclidean",
        group_by_batch: bool = False,
    ):
        import scanpy as sc

        adata = self.parent.dataset.adata

        if group_by_batch and self.parent.batch_key:
            group_by = self.parent.batch_key
            sc.external.pp.bbknn(
                adata,
                batch_key=group_by,
                use_rep=use_rep,
                n_pcs=n_dims,
                use_annoy=False,
                metric=metric,
                pynndescent_n_neighbors=n_neighbors,
            )
        else:
            sc.pp.neighbors(
                adata,
                n_neighbors=n_neighbors,
                use_rep=use_rep,
                n_pcs=n_dims,
                metric=metric,
            )

        self.broker.publish("dset_anndata_neighbors_change")
