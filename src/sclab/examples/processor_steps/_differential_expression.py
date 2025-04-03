from typing import Any, Iterable, Literal

import itables
import numpy as np
import pandas as pd
from anndata import AnnData
from IPython.display import Markdown, display
from ipywidgets import Dropdown, Output, SelectMultiple, Text, ToggleButtons
from ipywidgets.widgets.valuewidget import ValueWidget
from ipywidgets.widgets.widget_box import VBox
from ipywidgets.widgets.widget_description import DescriptionWidget
from packaging.version import Version

from sclab.dataset import SCLabDataset
from sclab.dataset.processor import Processor
from sclab.dataset.processor.step import ProcessorStepBase


class DifferentialExpressionResults(VBox):
    dataset: SCLabDataset
    result_selector: Dropdown
    group_selector: ToggleButtons
    table_output: Output
    namespace: str = "differential_expression"

    def __init__(self, dataset: SCLabDataset):
        self.dataset = dataset
        self.result_selector = Dropdown()
        self.group_selector = ToggleButtons()
        self.table_output = Output()

        self.result_selector.observe(self._update_group_selector, "value")
        self.result_selector.observe(self._update_table, "value")
        self.group_selector.observe(self._update_table, "value")

        super().__init__(
            [
                self.result_selector,
                self.group_selector,
                self.table_output,
            ]
        )

        self.sync_results_list()

    def sync_results_list(self, focus_result: str | None = None):
        adata = self.dataset.adata
        uns: dict[str, Any] = adata.uns
        current_selection = self.result_selector.value
        new_options = tuple(filter(lambda x: x.startswith(self.namespace), uns.keys()))

        if focus_result is not None and focus_result in new_options:
            current_selection = focus_result
        elif current_selection not in new_options:
            current_selection = None

        self.result_selector.options = new_options
        self.result_selector.value = current_selection

    def _update_group_selector(self, *args, **kwargs):
        selected_result = self.result_selector.value
        uns: dict[str, Any] = self.dataset.adata.uns
        gene_names: np.rec.recarray = uns[selected_result]["names"]
        self.group_selector.options = ("all",) + gene_names.dtype.names
        self.group_selector.value = "all"

    def _update_table(self, *args, **kwargs):
        selected_result = self.result_selector.value
        selected_group = self.group_selector.value

        adata = self.dataset.adata
        params = adata.uns[selected_result]["params"]

        groupby = params["groupby"]
        reference = params["reference"]
        table_name = f"{selected_result}_by_{groupby}_{selected_group}_vs_{reference}"

        params_text = "Parameters:\n    "
        params_text += "\n    ".join(f"{k}: {v}" for k, v in params.items())
        params_text = f"```\n{params_text}\n```"

        if "gene_name" in adata.var:
            gene_symbols = "gene_name"
        elif "name" in adata.var:
            gene_symbols = "name"
        elif "gene_symbol" in adata.var:
            gene_symbols = "gene_symbol"
        elif "symbol" in adata.var:
            gene_symbols = "symbol"
        else:
            gene_symbols = None

        group = selected_group if selected_group != "all" else None
        df = _rank_genes_groups_df(
            adata, group=group, key=selected_result, gene_symbols=gene_symbols
        )

        self.table_output.clear_output()
        with self.table_output:
            display(Markdown(f"## {table_name}"))
            itables.show(
                df,
                buttons=[
                    "pageLength",
                    {
                        "extend": "colvis",
                        "collectionLayout": "fixed columns",
                        "popoverTitle": "Column visibility control",
                    },
                    "copyHtml5",
                    {"extend": "csvHtml5", "title": table_name},
                ],
                columnDefs=[
                    {"visible": True, "targets": [0]},
                    {"visible": True, "targets": "_all"},
                ],
                style="width:100%",
                classes="display cell-border",
                stateSave=False,
            )
            display(Markdown(params_text))


class DifferentialExpression(ProcessorStepBase):
    parent: Processor
    results: DifferentialExpressionResults

    def __init__(self, parent: Processor) -> None:
        try:
            import scanpy as sc  # noqa: F401
        except ImportError:
            raise ImportError("Please install scanpy: `pip install scanpy`")

        metadata = parent.dataset._metadata.select_dtypes(
            include=["object", "category"]
        )
        groupby_options = (None,) + tuple(metadata.columns)

        variable_controls: dict[str, DescriptionWidget | ValueWidget]
        variable_controls = dict(
            groupby=Dropdown(options=groupby_options, description="Group by"),
            groups=SelectMultiple(description="Groups"),
            reference=Dropdown(description="Reference"),
            layer=Dropdown(
                options=(None,) + tuple(parent.dataset.adata.layers.keys()),
                value=None,
                description="Layer",
            ),
            name=Text(description="Name", value="", continuous_update=False),
        )

        variable_controls["groupby"].observe(
            self._update_groups_options, "value", "change"
        )
        variable_controls["groupby"].observe(
            self._update_reference_options, "value", "change"
        )

        results = DifferentialExpressionResults(parent.dataset)
        super().__init__(
            parent=parent,
            name="differential_expression",
            description="Differential Expression",
            fixed_params={},
            variable_controls=variable_controls,
            results=results,
        )

    def function(
        self,
        groupby: str,
        groups: Iterable[str] | Literal["all"],
        reference: str,
        layer: str | None,
        name: str | None,
    ):
        import scanpy as sc

        assert groupby

        if not groups:
            groups = "all"

        key_added = "differential_expression"
        if name:
            key_added = f"{key_added}_{name}"

        adata = self.parent.dataset.adata
        uns: dict[str, Any] = adata.uns
        if key_added in adata.uns:
            related_names = list(filter(lambda x: x.startswith(key_added), uns.keys()))
            key_added = f"{key_added}_{len(related_names) + 1}"

        sc.tl.rank_genes_groups(
            adata,
            groupby,
            groups=groups,
            reference=reference,
            layer=layer,
            key_added=key_added,
        )

        self.results.sync_results_list(focus_result=key_added)

    def _update_groups_options(self, *args, **kwargs):
        groupby = self.variable_controls["groupby"].value
        metadata = self.parent.dataset._metadata
        control: Dropdown = self.variable_controls["groups"]

        if groupby is None:
            control.options = ("",)
            return

        options = tuple(metadata[groupby].sort_values().unique())
        control.options = options

    def _update_reference_options(self, *args, **kwargs):
        groupby = self.variable_controls["groupby"].value
        metadata = self.parent.dataset._metadata
        control: Dropdown = self.variable_controls["reference"]

        if groupby is None:
            control.options = ("",)
            control.value = ""
            return

        options = ("rest",)
        options += tuple(metadata[groupby].sort_values().unique())

        current_value = control.value
        control.options = options
        if current_value not in control.options:
            control.value = "rest"
        else:
            control.value = current_value

    def dset_var_dataframe_change_callback(self, *args, **kwargs):
        var_df = self.parent.dataset.adata.var
        df = var_df.select_dtypes(include=["bool"])
        options = {"": None, **{c: c for c in df.columns}}

        control: Dropdown = self.variable_controls["mask_var"]
        current_value = control.value
        control.options = options
        if current_value not in control.options:
            control.value = None
        else:
            control.value = current_value


# from scanpy 1.10.4
# scanpy/src/scanpy/get/get.py
def _rank_genes_groups_df(
    adata: AnnData,
    group: str | Iterable[str] | None,
    *,
    key: str = "rank_genes_groups",
    pval_cutoff: float | None = None,
    log2fc_min: float | None = None,
    log2fc_max: float | None = None,
    gene_symbols: str | None = None,
) -> pd.DataFrame:
    """\
    Params
    ------
    adata
        Object to get results from.
    group
        Which group (as in :func:`scanpy.tl.rank_genes_groups`'s `groupby`
        argument) to return results from. Can be a list. All groups are
        returned if groups is `None`.
    key
        Key differential expression groups were stored under.
    pval_cutoff
        Return only adjusted p-values below the  cutoff.
    log2fc_min
        Minimum logfc to return.
    log2fc_max
        Maximum logfc to return.
    gene_symbols
        Column name in `.var` DataFrame that stores gene symbols. Specifying
        this will add that column to the returned dataframe.

    """
    if isinstance(group, str):
        group = [group]
    if group is None:
        group = list(adata.uns[key]["names"].dtype.names)
    method = adata.uns[key]["params"]["method"]
    if method == "logreg":
        colnames = ["names", "scores"]
    else:
        colnames = ["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]

    d = [pd.DataFrame(adata.uns[key][c])[group] for c in colnames]
    d = pd.concat(d, axis=1, names=[None, "group"], keys=colnames)
    if Version(pd.__version__) >= Version("2.1"):
        d = d.stack(level=1, future_stack=True).reset_index()
    else:
        d = d.stack(level=1).reset_index()
    d["group"] = pd.Categorical(d["group"], categories=group)
    d = d.sort_values(["group", "level_0"]).drop(columns="level_0")

    if method != "logreg":
        if pval_cutoff is not None:
            d = d[d["pvals_adj"] < pval_cutoff]
        if log2fc_min is not None:
            d = d[d["logfoldchanges"] > log2fc_min]
        if log2fc_max is not None:
            d = d[d["logfoldchanges"] < log2fc_max]
    if gene_symbols is not None:
        d = d.join(adata.var[gene_symbols], on="names")

    for pts, name in {"pts": "pct_nz_group", "pts_rest": "pct_nz_reference"}.items():
        if pts in adata.uns[key]:
            pts_df = (
                adata.uns[key][pts][group]
                .rename_axis(index="names")
                .reset_index()
                .melt(id_vars="names", var_name="group", value_name=name)
            )
            d = d.merge(pts_df)

    # remove group column for backward compat if len(group) == 1
    if len(group) == 1:
        d.drop(columns="group", inplace=True)

    return d.reset_index(drop=True)
