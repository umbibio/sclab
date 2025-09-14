from typing import Literal

import numpy as np
from ipywidgets import (
    Checkbox,
    Dropdown,
    FloatText,
    IntText,
)
from pandas.api.types import is_numeric_dtype

from sclab.dataset.processor import Processor
from sclab.dataset.processor.step import ProcessorStepBase

_2PI = 2 * np.pi


class TransferMetadata(ProcessorStepBase):
    parent: Processor
    name: str = "transfer_metadata"
    description: str = "Transfer Metadata"

    run_button_description = "Transfer Metadata"

    def __init__(self, parent: Processor) -> None:
        variable_controls = dict(
            group_key=Dropdown(
                options=[],
                value=None,
                description="Group Key",
            ),
            source_group=Dropdown(
                options=[],
                value=None,
                description="Source Group",
            ),
            column=Dropdown(
                options=[],
                value=None,
                description="Column",
            ),
            periodic=Checkbox(
                value=False,
                description="Periodic",
            ),
            vmin=FloatText(
                value=0,
                description="Vmin",
                continuous_update=False,
            ),
            vmax=FloatText(
                value=1,
                description="Vmax",
                continuous_update=False,
            ),
            min_neighs=IntText(
                value=5,
                min=3,
                description="Min Neighs",
                continuous_update=False,
            ),
            weight_by=Dropdown(
                options=["connectivity", "distance", "constant"],
                value="connectivity",
                description="Weight By",
            ),
        )

        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )

        self._update_groupby_options()
        self._update_column_options()
        self._update_numeric_column_controls()

        self.variable_controls["group_key"].observe(
            self._update_source_group_options, "value", "change"
        )
        self.variable_controls["column"].observe(
            self._update_numeric_column_controls, "value", "change"
        )
        self.variable_controls["periodic"].observe(
            self._update_vmin_vmax_visibility, "value", "change"
        )

    def _update_groupby_options(self, *args, **kwargs):
        metadata = self.parent.dataset._metadata.select_dtypes(include=["category"])
        options = {"": None, **{c: c for c in metadata.columns}}
        self.variable_controls["group_key"].options = options

    def _update_source_group_options(self, *args, **kwargs):
        group_key = self.variable_controls["group_key"].value
        if group_key is None:
            self.variable_controls["source_group"].options = ("",)
            return

        options = self.parent.dataset._metadata[group_key].sort_values().unique()
        options = {"": None, **{c: c for c in options}}
        self.variable_controls["source_group"].options = options

    def _update_column_options(self, *args, **kwargs):
        metadata = self.parent.dataset._metadata.select_dtypes(
            include=["category", "bool", "number"]
        )
        options = {"": None, **{c: c for c in metadata.columns}}
        self.variable_controls["column"].options = options

    def _update_numeric_column_controls(self, *args, **kwargs):
        column = self.variable_controls["column"].value
        if column is None:
            self._hide_control(self.variable_controls["periodic"])
            self._hide_control(self.variable_controls["vmin"])
            self._hide_control(self.variable_controls["vmax"])
            return

        series = self.parent.dataset._metadata[column]
        periodic = self.variable_controls["periodic"].value

        if is_numeric_dtype(series):
            self._show_control(self.variable_controls["periodic"])
            if periodic:
                self._show_control(self.variable_controls["vmin"])
                self._show_control(self.variable_controls["vmax"])
        else:
            self._hide_control(self.variable_controls["periodic"])
            self._hide_control(self.variable_controls["vmin"])
            self._hide_control(self.variable_controls["vmax"])

    def _update_vmin_vmax_visibility(self, *args, **kwargs):
        periodic = self.variable_controls["periodic"].value

        if periodic:
            self._show_control(self.variable_controls["vmin"])
            self._show_control(self.variable_controls["vmax"])
        else:
            self._hide_control(self.variable_controls["vmin"])
            self._hide_control(self.variable_controls["vmax"])

    def _hide_control(self, control):
        control.layout.visibility = "hidden"
        control.layout.height = "0px"

    def _show_control(self, control):
        control.layout.visibility = "visible"
        control.layout.height = "28px"

    def function(
        self,
        group_key: str,
        source_group: str,
        column: str,
        periodic: bool = False,
        vmin: float = 0,
        vmax: float = 1,
        min_neighs: int = 5,
        weight_by: Literal["connectivity", "distance", "constant"] = "connectivity",
        **kwargs,
    ):
        from ...preprocess._transfer_metadata import transfer_metadata

        self.output.clear_output(wait=True)
        with self.output:
            transfer_metadata(
                self.parent.dataset.adata,
                group_key=group_key,
                source_group=source_group,
                column=column,
                periodic=periodic,
                vmin=vmin,
                vmax=vmax,
                min_neighs=min_neighs,
                weight_by=weight_by,
            )

        new_column = f"transferred_{column}"

        self.broker.publish(
            "dset_metadata_change", self.parent.dataset.metadata, new_column
        )

    def dset_metadata_change_callback(self, *args, **kwargs):
        self._update_groupby_options(*args, **kwargs)
        self._update_column_options(*args, **kwargs)
