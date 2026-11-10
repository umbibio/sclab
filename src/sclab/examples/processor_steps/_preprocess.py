from ipywidgets import Checkbox, Dropdown

from sclab._compat import missing
from sclab.dataset.processor import Processor
from sclab.dataset.processor._codegen import ADATA, call
from sclab.dataset.processor.step import ProcessorStepBase
from sclab.event._jobs import capture
from sclab.preprocess import preprocess
from sclab.utils import layer_keys, text_or_category_columns

# dropdown label -> ``sclab.preprocess.preprocess(normalization_method=...)``
NORMALIZATION_METHODS = {
    "No normalization": "none",
    "Library size": "library",
    "Weighted library size": "weighted",
}


class Preprocess(ProcessorStepBase):
    """Normalization and scaling step for single-cell RNA-seq data.

    Runs :func:`sclab.preprocess.preprocess`: cell/gene filtering, HVG
    selection (optionally per group), normalization (library size,
    entropy-weighted, or none), log1p transformation, optional covariate
    regression, and (per-group) scaling. The processed matrix is stored in a
    new named layer and the plotter is updated to show ``n_genes_by_counts``
    vs ``total_counts``.

    Requires ``scanpy`` to be installed.
    """

    parent: Processor
    name: str = "preprocess"
    description: str = "Preprocess"

    def __init__(self, parent: Processor) -> None:
        cat_options = {
            "": None,
            **{c: c for c in text_or_category_columns(parent.dataset._metadata)},
        }

        variable_controls = dict(
            layer=Dropdown(
                options=tuple(layer_keys(parent.dataset.adata)),
                value=parent.dataset.counts_layer,
                description="Layer",
            ),
            group_by=Dropdown(
                options=cat_options,
                value=None,
                description="GroupBy",
            ),
            regress_total_counts=Checkbox(description="Regr. out total counts"),
            regress_n_genes=Checkbox(description="Regr. out n genes"),
            normalization_method=Dropdown(
                options=tuple(NORMALIZATION_METHODS),
                value="Library size",
                description="Norm. method",
            ),
            log1p=Checkbox(value=True, description="Log1p"),
            scale=Checkbox(value=True, description="Scale"),
        )

        def filter_layers(change: dict):
            new_options: tuple[str] = change["new"]
            if any(s.endswith("log1p") for s in new_options):
                new_options = tuple(
                    filter(lambda y: not y.endswith("log1p"), new_options)
                )
                variable_controls["layer"].options = new_options

        variable_controls["layer"].observe(filter_layers, "options", "change")

        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )

    def validate(self, params: dict) -> str | None:
        if reason := missing("scanpy"):
            return reason
        dataset = self.parent.dataset
        if (params.get("layer") or dataset.counts_layer) not in layer_keys(
            dataset.adata
        ):
            return "Choose a 'Layer' with raw counts"
        group_by = params.get("group_by")
        if group_by is not None and group_by not in dataset.adata.obs:
            return f"No column {group_by!r} in the data"
        return None

    def dset_counts_layer_change_callback(self, old: str, new: str):
        layer = self.variable_controls["layer"]
        if layer.value == old and new in layer.options:  # else chosen by the user
            layer.value = new

    def resolved_params(self, params: dict) -> dict:
        resolved = super().resolved_params(params)
        if resolved.get("layer") is None:
            resolved["layer"] = self.parent.dataset.counts_layer
        return resolved

    def to_code(self, params: dict) -> str:
        method = NORMALIZATION_METHODS[params["normalization_method"]]
        return call(
            "sclab.preprocess.preprocess",
            ADATA,
            counts_layer=params["layer"],
            group_by=params["group_by"],
            regress_total_counts=params["regress_total_counts"],
            regress_n_genes=params["regress_n_genes"],
            normalization_method=method,
            log1p=params["log1p"],
            scale=params["scale"],
        )

    def function(
        self,
        layer: str | None = None,
        group_by: str | None = None,
        regress_total_counts: bool = False,
        regress_n_genes: bool = False,
        normalization_method: str = "No normalization",
        log1p: bool = True,
        scale: bool = True,
    ):
        dataset = self.parent.dataset
        adata = dataset.adata
        start_n_cells, start_n_genes = adata.shape

        self.output.clear_output(wait=True)
        with capture(self.output):  # the library's progress bar
            preprocess(
                adata,
                counts_layer=layer or dataset.counts_layer,
                group_by=group_by,
                regress_total_counts=regress_total_counts,
                regress_n_genes=regress_n_genes,
                normalization_method=NORMALIZATION_METHODS[normalization_method],
                log1p=log1p,
                scale=scale,
            )

        self.broker.publish("dset_metadata_change", dataset.metadata)
        self.broker.publish("dset_data_dict_change", dataset.data_dict, "metadata")
        self.broker.publish("dset_anndata_layers_change", layer_keys(dataset.adata))
        self.broker.publish(
            "ctrl_value_change_request",
            data_key="metadata",
            selected_axes_1="n_genes_by_counts",
            selected_axes_2="total_counts",
        )
        self.broker.publish("dset_var_dataframe_change", dataset.adata.var)

        end_n_cells, end_n_genes = adata.shape
        if start_n_cells != end_n_cells:
            self.broker.publish("dset_total_rows_change", dataset.metadata)
        if start_n_genes != end_n_genes:
            self.broker.publish("dset_total_vars_change", adata.var)
