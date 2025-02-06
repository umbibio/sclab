from collections.abc import Sequence

import itables
import itables.options
import numpy as np
import pandas as pd
from anndata import AnnData
from ipywidgets import GridBox, Layout, Output
from numpy.typing import NDArray

from ..event import EventBroker, EventClient
from ._exceptions import InvalidRowSubset

itables.options.maxBytes = "50MB"


class SCLabDataset(EventClient):
    adata: AnnData
    name: str
    _data_dict: dict[str, pd.DataFrame]
    _metadata: pd.DataFrame
    _selected_data_key: str | None = None
    events: list[str] = [
        "dset_data_dict_change",
        "dset_data_key_selection_change",
        "dset_metadata_change",
        "dset_selected_rows_change",
        "dset_total_rows_change",
        "dset_anndata_layers_change",
        "dset_anndata_neighbors_change",
        "dset_var_dataframe_change",
        "dset_total_vars_change",
    ]
    preemptions: dict[str, list[str]] = {
        "dset_data_key_selection_change": [
            "ctrl_selected_axes_1_change",
            "ctrl_selected_axes_2_change",
            "ctrl_selected_axes_3_change",
            "ctrl_n_dimensions_change",
        ],
        "dset_metadata_change": [
            "dspr_selection_values_change",
        ],
        "dset_total_rows_change": [
            "dspr_selection_values_change",
        ],
    }
    _selected_rows: pd.Index | None = None

    def __init__(
        self,
        adata: AnnData,
        name: str = "SCLabDataset",
        counts_layer: str = "counts",
        copy: bool = True,
        broker: EventBroker | None = None,
    ):
        if not isinstance(adata, AnnData):
            raise TypeError("adata must be an instance of AnnData")

        self.name = name

        # we keep the original counts layer to be able to reset it
        self.counts_layer = counts_layer

        self.load_adata(adata, copy=copy)

        self.obs_table_output = Output(style={"width": "98%"})
        self.var_table_output = Output(style={"width": "98%"})

        self.obs_table = GridBox(
            [
                self.obs_table_output,
            ],
            layout=Layout(
                width="100%",
                grid_template_columns="auto",
                grid_template_areas=""" "obs_table" """,
                border="0px solid black",
            ),
        )

        self.var_table = GridBox(
            [
                self.var_table_output,
            ],
            layout=Layout(
                width="100%",
                grid_template_columns="auto",
                grid_template_areas=""" "var_table" """,
                border="0px solid black",
            ),
        )

        if broker is None:
            broker = EventBroker()

        super().__init__(broker)

        def update_obs_table(incoming_change: pd.DataFrame | dict, *args, **kvargs):
            if isinstance(incoming_change, dict):
                df = self.adata.obs
            elif isinstance(incoming_change, pd.DataFrame):
                df = incoming_change
            else:
                raise TypeError("incoming_change must be a DataFrame or a dict")

            self.obs_table_output.clear_output(wait=True)
            with self.obs_table_output:
                itables.show(
                    df.reset_index(),
                    tableId=f"singlecell_dataset_obs_itable_{self.uuid}",
                    layout={"top1": "searchBuilder"},
                    buttons=[
                        "pageLength",
                        {
                            "extend": "colvis",
                            "collectionLayout": "fixed columns",
                            "popoverTitle": "Column visibility control",
                        },
                        "copyHtml5",
                        {"extend": "csvHtml5", "title": f"{self.name}_cells"},
                        {"extend": "excelHtml5", "title": f"{self.name}_cells"},
                    ],
                    columnDefs=[
                        {"visible": True, "targets": [0]},
                        {"visible": False, "targets": "_all"},
                    ],
                    style="width:100%",
                    classes="display cell-border",
                    stateSave=True,
                )

        def update_var_table(incoming_change: pd.DataFrame | dict, *args, **kvargs):
            if isinstance(incoming_change, dict):
                df = self.adata.var
            elif isinstance(incoming_change, pd.DataFrame):
                df = incoming_change
            else:
                raise TypeError("incoming_change must be a DataFrame or a dict")

            self.var_table_output.clear_output(wait=True)
            with self.var_table_output:
                itables.show(
                    df.reset_index(),
                    tableId=f"singlecell_dataset_var_itable_{self.uuid}",
                    layout={"top1": "searchBuilder"},
                    buttons=[
                        "pageLength",
                        {
                            "extend": "colvis",
                            "collectionLayout": "fixed columns",
                            "popoverTitle": "Column visibility control",
                        },
                        "copyHtml5",
                        {"extend": "csvHtml5", "title": f"{self.name}_genes"},
                        {"extend": "excelHtml5", "title": f"{self.name}_genes"},
                    ],
                    columnDefs=[
                        {"visible": True, "targets": [0]},
                        {"visible": False, "targets": "_all"},
                    ],
                    style="width:100%",
                    classes="display cell-border",
                    stateSave=True,
                )

        update_obs_table(self.adata.obs)
        update_var_table(self.adata.var)

        broker.subscribe("dset_metadata_change", update_obs_table)
        broker.subscribe("dset_total_rows_change", update_obs_table)

        broker.subscribe("dset_var_dataframe_change", update_var_table)
        broker.subscribe("dset_total_vars_change", update_var_table)

    def load_adata(self, adata: AnnData, copy: bool = True):
        if copy:
            self.adata = adata.copy()
        else:
            self.adata = adata

        if self.counts_layer not in self.adata.layers:
            self.adata.layers[self.counts_layer] = self.adata.X.copy()

    @property
    def data_dict(self) -> dict:
        return {
            "metadata": self.metadata.select_dtypes(include="number"),
            **self._data_dict,
        }

    @data_dict.setter
    def data_dict(self, value: dict[str, pd.DataFrame | NDArray]):
        self._data_dict = self._validate_data_dict(value)
        self.broker.publish("dset_data_dict_change", self.data_dict)

    @property
    def _data_dict(self):
        return self._validate_data_dict(self.adata.obsm._data)

    @property
    def _metadata(self):
        return self.adata.obs

    def update_data_dict(self):
        self.data_dict = self.adata.obsm._data

    def _validate_data_dict(self, value: dict[str, pd.DataFrame | NDArray]) -> dict:
        assert isinstance(value, dict), "data_dict must be a dictionary"

        index = None
        tmp_dict = {}
        for key, val in value.items():
            assert isinstance(key, str), "data_dict keys must be strings"

            val = self._validate_data(key, val)

            if index is None:
                index = val.index
            else:
                # TODO: improve matching of index. We should accept index in different order
                assert val.index.equals(index), "all data must have the same index"

            tmp_dict[key] = val

        return tmp_dict

    @property
    def data(self) -> pd.DataFrame:
        if not self._selected_data_key:
            return pd.DataFrame(index=self.metadata.index)
        return self.data_dict[self._selected_data_key]

    def select_data_key(self, key: str):
        if key not in self.data_dict:
            raise ValueError(f"key '{key}' not found in data_dict")

        self._selected_data_key = key

        self.broker.publish("dset_data_key_selection_change", self.data)

    def reset_data_key(self):
        self._selected_data_key = None

        self.broker.publish("dset_data_key_selection_change", self.data)

    def _validate_data(
        self, dk: str, value: pd.DataFrame | NDArray | None
    ) -> pd.DataFrame:
        if value is None:
            value = pd.DataFrame(index=self.metadata.index)

        elif isinstance(value, np.ndarray):
            assert value.ndim <= 2, "data array must be 1D or 2D"

            if not self.metadata.empty:
                assert value.shape[0] == self._metadata.shape[0], (
                    "data must have same length as metadata"
                )
                value = pd.DataFrame(value, index=self.metadata.index)

            else:
                value = pd.DataFrame(value)
            value.columns = [f"{dk.upper()} {i + 1}" for i in range(value.shape[1])]

        elif isinstance(value, pd.DataFrame):
            if not self.metadata.empty:
                assert value.index.equals(self.metadata.index), (
                    "data must have same index as metadata"
                )

        else:
            raise TypeError("data must be a pandas DataFrame or numpy array")

        return value

    @property
    def metadata(self) -> pd.DataFrame:
        # Retain only numerical, categorical and string columns.
        # If a column has object dtype (string) and there are no more than 10 unique values,
        # convert it to categorical.
        metadata = self._metadata.select_dtypes(
            include=["number", "object", "category", "boolean"]
        ).copy()
        for col in metadata.columns:
            if metadata[col].dtype == "object":
                if metadata[col].nunique() <= 10:
                    metadata[col] = metadata[col].astype("category")
                else:
                    metadata.drop(col, axis=1, inplace=True)

        # is_selected may be a boolean column or a column of NaNs
        # if it is a boolean column, a selection has been defined (possible all False)
        # if it is a column of NaNs, no selection has been defined
        if self._selected_rows is not None:
            metadata["is_selected"] = metadata.index.isin(self._selected_rows)
        else:
            metadata["is_selected"] = pd.NA
            metadata["is_selected"] = metadata["is_selected"].astype("boolean")
        return metadata

    @metadata.setter
    def metadata(self, value: pd.DataFrame | None):
        if value is None:
            value = pd.DataFrame()

        if not isinstance(value, pd.DataFrame):
            raise TypeError("metadata must be a pandas DataFrame")

        self._metadata = value
        self.broker.publish("dset_metadata_change", self.metadata)

    @property
    def row_names(self) -> pd.Index:
        return self.metadata.index

    @property
    def selected_rows(self) -> pd.Index:
        if self._selected_rows is None:
            index = pd.Index([], name="selected_rows")
        else:
            index = self._selected_rows
        return index

    @selected_rows.setter
    def selected_rows(self, value: pd.Index | None):
        if value is None:
            self._selected_rows = None
        else:
            row_names_dtype = self.metadata.index.dtype

            self._selected_rows = value.astype(row_names_dtype)
            self._selected_rows.name = "selected_rows"

        self.broker.publish("dset_selected_rows_change", value)

    @property
    def selected_rows_mask(self) -> NDArray[np.bool]:
        return self.metadata.index.isin(self.selected_rows)

    @property
    def selected_rows_data(self) -> pd.DataFrame:
        return self.data.loc[self.selected_rows]

    @property
    def selected_rows_metadata(self) -> pd.DataFrame:
        return self.metadata.loc[self.selected_rows]

    def select_rows(self, index: pd.Index):
        assert isinstance(index, pd.Index), "index must be a pandas Index"
        assert index.isin(self.metadata.index).all(), "index contains invalid values"
        self.selected_rows = self.selected_rows.union(index)

    def deselect_rows(self, index: pd.Index):
        assert isinstance(index, pd.Index), "index must be a pandas Index"
        assert index.isin(self.metadata.index).all(), "index contains invalid values"
        self.selected_rows = self.selected_rows.difference(index)

    def clear_selected_rows(self):
        self.selected_rows = None

    def filter_rows(self, index: pd.Index | Sequence):
        if not isinstance(index, pd.Index):
            index = pd.Index(index)

        if not index.isin(self.metadata.index).all():
            raise InvalidRowSubset("index contains invalid values")

        self.adata = self.adata[index].copy()

        self.broker.publish("dset_total_rows_change", self.metadata)

    def apply_label(self, index: pd.Index, column: str, label: str):
        if column not in self._metadata.columns:
            dtype = pd.CategoricalDtype([label], ordered=False)
            self._metadata[column] = pd.Series(index=self.row_names, dtype=dtype)

        if label and label not in self._metadata[column].cat.categories:
            self._metadata[column] = self._metadata[column].cat.add_categories(label)
        elif not label:
            label = np.nan

        self._metadata.loc[index, column] = label

        self.broker.publish("dset_metadata_change", self.metadata, column)

    def ctrl_data_key_change_callback(self, new_value: str):
        if new_value is None:
            self.reset_data_key()
        else:
            self.select_data_key(new_value)

    def dplt_selected_points_change_callback(self, new_value: pd.Index):
        self.selected_rows = new_value

    def dspr_clear_selection_click_callback(self):
        self.clear_selected_rows()
