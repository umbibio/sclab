import re
import tempfile
from html import escape
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import itables
import pandas as pd
from ipywidgets import (
    BoundedIntText,
    Button,
    Checkbox,
    Dropdown,
    HBox,
    SelectMultiple,
    Text,
)
from ipywidgets.widgets.valuewidget import ValueWidget
from ipywidgets.widgets.widget_box import VBox
from ipywidgets.widgets.widget_description import DescriptionWidget

from sclab import tools
from sclab._compat import requirement
from sclab.dataset import SCLabDataset
from sclab.dataset.processor import Processor
from sclab.dataset.processor._codegen import ADATA, call
from sclab.dataset.processor.step import ProcessorStepBase
from sclab.tools.differential_expression import (
    edgepython_is_available,
    pylimma_is_available,
)
from sclab.tools.differential_expression._differential_expression import (
    PSEUDOBULK_METHODS,
    SCANPY_METHODS,
)
from sclab.utils import (
    LazyOutput,
    layer_keys,
    looks_like_raw_counts,
    set_options_keep_value,
    text_or_category_columns,
    write_excel,
)
from sclab.utils._widgets import Download

METHODS = (*SCANPY_METHODS, *PSEUDOBULK_METHODS)
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _unavailable(method: str) -> str | None:
    """Why ``method`` cannot run here (a missing optional package), or None."""
    if method in SCANPY_METHODS and find_spec("scanpy") is None:
        return f"{method} needs {requirement('scanpy')}"
    if method == "pseudobulk_limma" and not pylimma_is_available():
        return f"pseudobulk_limma needs {requirement('pylimma')}"
    if method == "pseudobulk_edger" and not edgepython_is_available():
        return f"pseudobulk_edger needs {requirement('edgepython')}"
    return None


def is_de_result(value) -> bool:
    """True for an ``adata.uns`` entry written by :func:`sclab.tools.de`."""
    return isinstance(value, dict) and "params" in value and "table" in value


def _sheet_name(name: str, taken: set[str]) -> str:
    """An Excel sheet name: no ``[]:*?/\\``, at most 31 characters, unique."""
    base = re.sub(r"[\[\]:*?/\\]", "_", str(name))[:31] or "sheet"
    out, i = base, 2
    while out.lower() in taken:
        out, i = f"{base[: 30 - len(str(i))]}~{i}", i + 1
    taken.add(out.lower())
    return out


def excel_bytes(entry: dict) -> bytes:
    """``entry`` (see :func:`is_de_result`) as an .xlsx file: a sheet per
    contrast, then a ``params`` sheet."""
    table: pd.DataFrame = entry["table"]
    taken = {"params"}
    sheets = {
        _sheet_name(contrast, taken): df.drop(columns="contrast").set_index("gene")
        for contrast, df in table.groupby("contrast", sort=False)
    }
    params = entry["params"]
    sheets["params"] = pd.DataFrame(
        {"value": [str(v) for v in params.values()]},
        index=pd.Index(list(params), name="param"),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "de.xlsx"
        write_excel(
            sheets,
            path,
            column_formats={"pct_cnd": "percentage", "pct_ref": "percentage"},
        )
        return path.read_bytes()


class DifferentialExpressionResults(VBox):
    """Results panel for the tables of :func:`sclab.tools.de`.

    Lists the ``adata.uns`` entries holding ``{"params", "table"}``. For the
    selected analysis it shows the table filtered by contrast (and cell
    identity, if the analysis has one), its parameters, and exports it to
    Excel: one sheet per contrast plus a params sheet.
    """

    dataset: SCLabDataset
    result_selector: Dropdown
    contrast_selector: Dropdown
    identity_selector: Dropdown
    table_output: LazyOutput
    namespace: str = "differential_expression"

    def __init__(self, dataset: SCLabDataset):
        self.dataset = dataset
        self.result_selector = Dropdown(description="Analysis")
        self.contrast_selector = Dropdown(options=("all",), description="Contrast")
        self.identity_selector = Dropdown(options=("all",), description="Identity")
        self.export_button = Button(description="Export Excel", icon="download")
        self.export_button.on_click(self.export)
        self.download = Download()
        # tens of MB for all contrasts: built only while shown, see show_when
        self.table_output = LazyOutput(self._table)
        self._table_key = None
        self._syncing = False

        # _update_selectors first: _update_table then sees the new options
        self.result_selector.observe(self._update_selectors, "value")
        for selector in (
            self.result_selector,
            self.contrast_selector,
            self.identity_selector,
        ):
            selector.observe(self._update_table, "value")

        super().__init__(
            [
                self.result_selector,
                HBox([self.contrast_selector, self.identity_selector]),
                self.export_button,
                self.download,
                self.table_output,
            ]
        )

        self.sync_results_list()

    def sync_results_list(self, focus_result: str | None = None):
        uns: dict[str, Any] = self.dataset.adata.uns
        current_selection = self.result_selector.value
        new_options = tuple(k for k, v in uns.items() if is_de_result(v))

        if focus_result is not None and focus_result in new_options:
            current_selection = focus_result
        elif current_selection not in new_options:
            current_selection = None

        self.result_selector.options = new_options
        self.result_selector.value = current_selection
        self._update_selectors()

    def show_when(self, *path):
        self.table_output.show_when(*path)

    def _entry(self) -> dict | None:
        uns: dict[str, Any] = self.dataset.adata.uns
        return uns.get(self.result_selector.value)

    def _update_selectors(self, *args, **kwargs):
        entry = self._entry()
        table = entry["table"] if entry is not None else pd.DataFrame()
        has_identity = "cell_identity" in table
        self._syncing = True  # render once, with both selectors updated
        try:
            for selector, column in (
                (self.contrast_selector, "contrast"),
                (self.identity_selector, "cell_identity"),
            ):
                values = table[column].unique() if column in table else ()
                selector.options = ("all", *values)
                selector.value = "all"
        finally:
            self._syncing = False
        self.identity_selector.layout.display = None if has_identity else "none"
        self._update_table()

    def _update_table(self, *args, **kwargs):
        if self._syncing:
            return
        key = (
            self.result_selector.value,
            self.contrast_selector.value,
            self.identity_selector.value,
        )
        if key != self._table_key:
            self._table_key = key
            self.table_output.invalidate()

    def filtered_table(self) -> pd.DataFrame:
        """The selected analysis' table, as shown: gene first, filtered rows."""
        df: pd.DataFrame = self._entry()["table"]
        for column, value in (
            ("contrast", self.contrast_selector.value),
            ("cell_identity", self.identity_selector.value),
        ):
            if column in df and value != "all":
                df = df[df[column] == value].drop(columns=column)
        return df[["gene", *df.columns.drop("gene")]].reset_index(drop=True)

    def export(self, *_):
        """Save the selected analysis as an .xlsx file in the browser."""
        entry = self._entry()
        if entry is not None:
            name = f"{self.result_selector.value}.xlsx"
            self.download.save(excel_bytes(entry), name, type=XLSX)

    def _table(self) -> tuple[dict, ...]:
        entry = self._entry()
        if entry is None:
            return ()

        params = entry["params"]
        df = self.filtered_table()
        parts = [self.result_selector.value, self.contrast_selector.value]
        if self.identity_selector.value != "all":
            parts.append(self.identity_selector.value)
        title = " / ".join(parts)

        bundles = [{"text/markdown": f"## {title}"}]
        if params.get("method") == "logreg":
            bundles.append(
                {
                    "text/markdown": "*logreg gives no p-values: pval and padj are empty.*"
                }
            )

        # floats shown with 4 significant digits (compact or scientific)
        with pd.option_context("display.float_format", "{:.4g}".format):
            html = itables.to_html_datatable(
                df,
                buttons=[
                    "pageLength",
                    {
                        "extend": "colvis",
                        "collectionLayout": "fixed columns",
                        "popoverTitle": "Column visibility control",
                    },
                    "copyHtml5",
                    {"extend": "csvHtml5", "title": title.replace(" / ", "_")},
                ],
                style="width:100%",
                classes="display cell-border",
                stateSave=False,
            )
        bundles.append({"text/html": html})

        lines = "\n".join(f"{k}: {v}" for k, v in params.items())
        bundles.append(
            {
                "text/html": "<details><summary>Parameters</summary>"
                f"<pre>{escape(lines)}</pre></details>"
            }
        )
        return tuple(bundles)


class DifferentialExpression(ProcessorStepBase):
    """Differential gene expression step, running :func:`sclab.tools.de`.

    Compares each condition group with a reference group (or the rest of
    the cells), optionally within each cell identity. Methods: the cell-level
    tests of scanpy (``wilcoxon``, ``t-test``, ``logreg``) on ``adata.X``,
    and the pseudobulk tests (``pseudobulk_limma``, ``pseudobulk_edger``,
    with the ``de`` extra) on the dataset's counts layer; tick "Override
    layer" to pick another layer. Results are stored in ``adata.uns`` and
    shown in a :class:`DifferentialExpressionResults` panel.

    The cell-level tests need ``scanpy``; without it, only the pseudobulk
    methods can run.
    """

    parent: Processor
    results: DifferentialExpressionResults
    name: str = "differential_expression"
    description: str = "Differential Expression"
    background = True

    def __init__(self, parent: Processor) -> None:
        metadata = parent.dataset._metadata
        groupby_options = (None,) + tuple(text_or_category_columns(metadata))
        column_options = self._column_options(metadata)
        self._raw_checked: dict[str, tuple[Any, bool]] = {}

        variable_controls: dict[str, DescriptionWidget | ValueWidget]
        variable_controls = dict(
            method=Dropdown(options=METHODS, value="wilcoxon", description="Method"),
            groupby=Dropdown(options=groupby_options, description="Group by"),
            groups=SelectMultiple(description="Conditions"),
            reference=Dropdown(description="Reference"),
            cell_identity=Dropdown(
                options=column_options, value=None, description="Cell identity"
            ),
            sample_key=Dropdown(
                options=column_options, value=None, description="Sample key"
            ),
            replicas_per_group=BoundedIntText(
                value=3, min=1, max=100, description="Replicas"
            ),
            override_layer=Checkbox(value=False, description="Override layer"),
            layer=Dropdown(
                options=tuple(layer_keys(parent.dataset.adata)),
                value=None,
                description="Layer",
            ),
            name=Text(description="Name", value="", continuous_update=False),
        )

        results = DifferentialExpressionResults(parent.dataset)
        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
            results=results,
        )

        variable_controls["groupby"].observe(
            self.guarded(self._update_groups_options), "value", "change"
        )
        variable_controls["groupby"].observe(
            self.guarded(self._update_reference_options), "value", "change"
        )
        for key in ("method", "override_layer", "sample_key"):
            variable_controls[key].observe(self.guarded(self._update_method), "value")
        self._update_method()

    @staticmethod
    def _column_options(metadata: pd.DataFrame) -> dict:
        return {"": None, **{c: c for c in text_or_category_columns(metadata)}}

    def dset_metadata_change_callback(self, *args, **kwargs):
        options = self._column_options(self.parent.dataset._metadata)
        for key in ("cell_identity", "sample_key"):
            set_options_keep_value(self.variable_controls[key], options)

    def dset_counts_layer_change_callback(self, old: str, new: str):
        self._update_method()  # the auto layer, unless overridden

    def _auto_layer(self, method: str | None) -> str:
        """The layer ``method`` uses unless overridden."""
        return self.parent.dataset.counts_layer if method in PSEUDOBULK_METHODS else "X"

    def _layer(self, params: dict) -> str | None:
        if params.get("override_layer"):
            return params.get("layer")
        return self._auto_layer(params.get("method"))

    def _update_method(self, *args, **kwargs):
        """Show the controls the method uses; the layer follows the method."""
        c = self.variable_controls
        method, override = c["method"].value, c["override_layer"].value
        pseudobulk = method in PSEUDOBULK_METHODS
        auto = self._auto_layer(method)
        c["override_layer"].description = f"Override layer ({auto})"
        if not override:
            c["layer"].value = auto if auto in c["layer"].options else None
        shown = {
            "sample_key": pseudobulk,
            "replicas_per_group": pseudobulk and c["sample_key"].value is None,
            "layer": override,
        }
        for key, visible in shown.items():
            c[key].layout.display = None if visible else "none"

    def _looks_raw(self, layer: str) -> bool:
        adata = self.parent.dataset.adata
        X = adata.X if layer == "X" else adata.layers[layer]
        checked = self._raw_checked.get(layer)
        if checked is None or checked[0] is not X:  # once per matrix: reads all values
            checked = self._raw_checked[layer] = (X, looks_like_raw_counts(X))
        return checked[1]

    def validate(self, params: dict) -> str | None:
        adata = self.parent.dataset.adata
        groupby = params.get("groupby")
        if groupby is None:
            return "Choose a 'Group by' column with at least 2 groups"
        if groupby not in adata.obs:
            return f"No column {groupby!r} in the data"
        if adata.obs[groupby].nunique() < 2:
            return f"{groupby!r} has a single group: comparing needs at least 2"
        groups = params.get("groups")  # empty or "all": all
        if groups and groups != "all" and set(groups) <= {params.get("reference")}:
            return "Choose a group other than the reference"
        method = params.get("method")
        if (reason := _unavailable(method)) is not None:
            return reason
        keys = ["cell_identity"] + ["sample_key"] * (method in PSEUDOBULK_METHODS)
        for key in keys:
            if (column := params.get(key)) is not None and column not in adata.obs:
                return f"No column {column!r} in the data"

        layer = self._layer(params)
        if layer is None:
            return "Choose a layer"
        if layer != "X" and layer not in layer_keys(adata):
            if params.get("override_layer"):
                return f"No layer {layer!r} in the data"
            return f"No counts layer {layer!r}: tick 'Override layer' to choose one"
        what = "adata.X" if layer == "X" else f"Layer {layer!r}"
        if method in SCANPY_METHODS and self._looks_raw(layer):
            return (
                f"{what} looks like raw counts, but {method} needs log-normalized "
                "data: run the Preprocess step, or choose a pseudobulk method"
            )
        if method in PSEUDOBULK_METHODS and not self._looks_raw(layer):
            return f"{what} does not look like raw counts, which {method} needs"
        return None

    def _key_added(self, name: str | None) -> str:
        """``differential_expression[_{name}]``, numbered if already in uns."""
        key_added = "differential_expression"
        if name:
            key_added = f"{key_added}_{name}"

        uns: dict[str, Any] = self.parent.dataset.adata.uns
        if key_added in uns:
            related_names = list(filter(lambda x: x.startswith(key_added), uns.keys()))
            key_added = f"{key_added}_{len(related_names) + 1}"
        return key_added

    def resolved_params(self, params: dict) -> dict:
        resolved = super().resolved_params(params)
        resolved["layer"] = self._layer(resolved)
        resolved["key_added"] = self._key_added(resolved.get("name"))
        # pylimma without edgepython: limma's own filtering, raw library sizes
        if resolved["method"] == "pseudobulk_limma" and not edgepython_is_available():
            resolved["edger_preprocessing"] = False
        return resolved

    @staticmethod
    def _de_kwargs(params: dict) -> dict:
        """The keyword arguments of :func:`sclab.tools.de` for recorded ``params``."""
        groups, reference = params.get("groups"), params.get("reference")
        kwargs = dict(
            method=params["method"],
            condition_group=list(groups) if groups and groups != "all" else None,
            reference_group=None if reference in (None, "", "rest") else reference,
            cell_identity_key=params.get("cell_identity"),
            layer=params["layer"],
        )
        if params["method"] in PSEUDOBULK_METHODS:
            kwargs["sample_key"] = params.get("sample_key")
            if kwargs["sample_key"] is None:
                kwargs["replicas_per_group"] = params["replicas_per_group"]
            if "edger_preprocessing" in params:
                kwargs["edger_preprocessing"] = params["edger_preprocessing"]
        kwargs["key_added"] = params["key_added"]
        return kwargs

    def to_code(self, params: dict) -> str:
        return call(
            "sclab.tools.de", ADATA, params["groupby"], **self._de_kwargs(params)
        )

    def function(self, **params):
        resolved = self.resolved_params(params)
        kwargs = self._de_kwargs(resolved)
        yield from self.compute(tools.de, resolved["groupby"], **kwargs)
        self.results.sync_results_list(focus_result=kwargs["key_added"])

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
