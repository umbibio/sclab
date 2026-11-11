from ipywidgets import Dropdown, IntRangeSlider, IntText

from sclab.dataset.processor import Processor
from sclab.dataset.processor._codegen import ADATA, call
from sclab.dataset.processor.step import ProcessorStepBase
from sclab.preprocess import neighbors
from sclab.preprocess._scanpy_port import native_backend, resolve_backend


def missing_neighbors(adata) -> str | None:
    """A step's :meth:`validate` reason if ``adata`` has no neighbor graph."""
    if "neighbors" not in adata.uns or "connectivities" not in adata.obsp:
        return "Run the Neighbors step first"
    return None


class Neighbors(ProcessorStepBase):
    """k-nearest-neighbor graph construction step.

    Runs :func:`sclab.preprocess.neighbors`: builds the cell neighborhood
    graph using a selected embedding (``use_rep``) and a specified range of
    dimensions. Supports standard k-NN (via :func:`scanpy.pp.neighbors`) and
    batch-aware BBKNN graph construction when ``group_by_batch`` is enabled
    and the dashboard has a ``batch_key``.

    The standard k-NN graph uses scanpy if it is installed, else sclab's
    native port of :func:`scanpy.pp.neighbors` (``backend="auto"``; exact
    k-NN, umap-learn's connectivities); BBKNN uses bbknn if it and scanpy are
    installed, else sclab's native port of bbknn (exact k-NN per batch). The
    history records which ran.
    """

    parent: Processor
    name: str = "neighbors"
    description: str = "Neighbors"
    background = True

    def __init__(self, parent: Processor) -> None:
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
            if use_rep not in adata.obsm:
                return

            max_dim = adata.obsm[use_rep].shape[1]
            variable_controls["dims"].max = max_dim

        variable_controls["use_rep"].observe(update_dims_range, names="value")

        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )

    def _backend(self, params: dict) -> str:
        backend = params.get("backend", "auto")  # recorded: a replay
        if params.get("group_by_batch") and self.parent.batch_key:
            return resolve_backend(backend, "bbknn")
        return resolve_backend(backend)

    def unavailable(self, params: dict) -> str | None:
        batch = params.get("group_by_batch") and params.get("batch_key")
        return self.scanpy_missing(params, *["bbknn"] * bool(batch))

    def validate(self, params: dict) -> str | None:
        adata = self.parent.dataset.adata
        if params.get("use_rep") not in adata.obsm:
            return "Choose a 'Use rep.' representation (run PCA first)"
        return None

    def tool(self, params: dict) -> str:
        native = self._backend(params) == "native"
        if params.get("group_by_batch") and self.parent.batch_key:
            if native:
                return (
                    "Python · sclab (port of bbknn / "
                    "umap-learn fuzzy_simplicial_set)"
                )
            return "Python · bbknn (scanpy.external.pp.bbknn)"
        if not native:
            return "Python · scanpy.pp.neighbors"
        return (
            "Python · sclab (port of scanpy.pp.neighbors / "
            "umap-learn fuzzy_simplicial_set)"
        )

    def resolved_params(self, params: dict) -> dict:
        return {
            **super().resolved_params(params),
            "batch_key": self.parent.batch_key,
            "backend": self._backend(params),
        }

    def to_code(self, params: dict) -> str:
        return call(
            "sclab.preprocess.neighbors",
            ADATA,
            use_rep=params["use_rep"],
            dims=tuple(params["dims"]),
            n_neighbors=params["n_neighbors"],
            metric=params["metric"],
            batch_key=params["batch_key"] if params["group_by_batch"] else None,
            random_state=params["random_state"],
            **native_backend(params),
        )

    def function(
        self,
        n_neighbors: int = 20,
        use_rep: str = "X_pca",
        dims: tuple[int, int] = (1, 10),
        metric: str = "euclidean",
        group_by_batch: bool = False,
        random_state: int = 0,
        backend: str = "auto",
    ):
        yield from self.compute(
            neighbors,
            use_rep=use_rep,
            dims=dims,
            n_neighbors=n_neighbors,
            metric=metric,
            batch_key=self.parent.batch_key if group_by_batch else None,
            random_state=random_state,
            backend=backend,
        )

        self.broker.publish("dset_anndata_neighbors_change")
