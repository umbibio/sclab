from ipywidgets import Dropdown, IntRangeSlider, IntText

from sclab.dataset.processor import Processor
from sclab.dataset.processor.step import ProcessorStepBase


class Neighbors(ProcessorStepBase):
    parent: Processor
    name: str = "neighbors"
    description: str = "Neighbors"

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
            dims=IntRangeSlider(
                min=1,
                max=30,
                value=(1, 10),
                description="Use dims",
            ),
            metric=Dropdown(
                options=["euclidean", "cosine"],
                value="euclidean",
                description="Metric",
            ),
            **parent.make_groupbybatch_checkbox(),
        )

        def update_dims_range(*args, **kwargs):
            adata = self.parent.dataset.adata
            use_rep = variable_controls["use_rep"].value
            max_dim = adata.obsm[use_rep].shape[1]
            variable_controls["dims"].max = max_dim

        variable_controls["use_rep"].observe(update_dims_range, names="value")

        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )

    def function(
        self,
        n_neighbors: int = 20,
        use_rep: str = "X_pca",
        dims: tuple[int, int] = (1, 10),
        metric: str = "euclidean",
        group_by_batch: bool = False,
    ):
        import scanpy as sc

        adata = self.parent.dataset.adata
        min_dim, max_dim = dims
        min_dim = min_dim - 1

        if min_dim > 0:
            adata.obsm[use_rep + "_trimmed"] = adata.obsm[use_rep][:, min_dim:max_dim]
            use_rep = use_rep + "_trimmed"
        n_dims = max_dim - min_dim

        if group_by_batch and self.parent.batch_key:
            group_by = self.parent.batch_key
            sc.external.pp.bbknn(
                adata,
                batch_key=group_by,
                use_rep=use_rep,
                n_pcs=n_dims,
                use_annoy=False,
                metric=metric,
                neighbors_within_batch=n_neighbors,
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
