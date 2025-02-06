from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import svgpathtools as spt
from ipywidgets.widgets import (
    Box,
    GridBox,
    Layout,
)
from numpy.typing import NDArray
from pandas.api.types import is_any_real_numeric_dtype, is_bool_dtype
from plotly.graph_objs.scatter import Marker as Marker2D
from plotly.graph_objs.scatter3d import Marker as Marker3D
from scipy.interpolate import make_smoothing_spline

from ...event import EventBroker, EventClient
from .._dataset import SCLabDataset
from ._controls import PlotterControls
from ._utils import (
    COLOR_DISCRETE_SEQUENCE,
    _make_density_heatmap,
    make_periodic_smoothing_spline,
    rotate_multiple_steps,
)


class Plotter(GridBox, EventClient):
    g: go.FigureWidget
    controls: PlotterControls
    dataset: SCLabDataset
    state: dict[str, Any]

    events: list[str] = [
        "dplt_dataset_change",
        "dplt_selected_points_change",
        "dplt_point_click",
        "dplt_layout_shapes_change",
        "dplt_start_drawing_request",
        "dplt_end_drawing_request",
        "dplt_soft_path_computed",
        "dplt_plot_figure_request",
        "dplt_add_trace_request",
        "dplt_add_vline_request",
        "dplt_add_hline_request",
        "dplt_add_data_as_line_trace_request",
    ]
    preemptions: dict[str, list[str]] = {
        "dplt_selected_points_change": [
            "dspr_selection_values_change",
        ],
        "dplt_plot_figure_request": [
            "ctrl_data_key_change",
            "ctrl_color_change",
            "ctrl_sizeby_change",
        ],
    }

    def __init__(
        self,
        dataset: SCLabDataset | None = None,
        broker: EventBroker | None = None,
    ):
        if broker is None:
            assert dataset is not None, "dataset must be provided if broker is None"
            broker = dataset.broker

        EventClient.__init__(self, broker)

        self.modebar = dict(
            add=[],
        )
        self.state = {}

        self._init_figure()
        self._init_controls()
        self.load_dataset(dataset)

        graph_layout = Layout(
            display="block", width="100%", height="600px", border="0px solid red"
        )
        self.graph_box = Box([self.g], layout=graph_layout)

        GridBox.__init__(
            self,
            [self.graph_box, self.controls],
            layout=Layout(
                width="100%",
                grid_template_columns="auto 350px",
                grid_template_areas=""" "graph controls" """,
                border="0px solid black",
            ),
        )
        self.broker.subscribe("ctrl_data_key_change", self.make_new_figure)
        self.broker.subscribe("ctrl_color_change", self.make_new_figure)
        self.broker.subscribe("ctrl_n_dimensions_change", self.make_new_figure)
        self.broker.subscribe("ctrl_selected_axes_1_change", self.make_new_figure)
        self.broker.subscribe("ctrl_selected_axes_2_change", self.make_new_figure)
        self.broker.subscribe("ctrl_selected_axes_3_change", self.make_new_figure)
        self.broker.subscribe("ctrl_log_axes_1_change", self.make_new_figure)
        self.broker.subscribe("ctrl_log_axes_2_change", self.make_new_figure)
        self.broker.subscribe("ctrl_log_axes_3_change", self.make_new_figure)
        self.broker.subscribe("ctrl_rotate_steps_change", self.make_new_figure)
        self.broker.subscribe("ctrl_refresh_button_click", self.make_new_figure)
        self.broker.subscribe("ctrl_histogram_nbins_change", self.make_new_figure)
        self.broker.subscribe("dset_selected_rows_change", self.select_points)
        self.broker.subscribe("dset_total_rows_change", self.make_new_figure)
        self.broker.subscribe("dset_metadata_change", self.make_new_figure)

    def _init_figure(self):
        self.g = go.FigureWidget(
            dict(
                layout=dict(
                    xaxis_title="",
                    yaxis_title="",
                    title="",
                    template="simple_white",
                    height=600,
                    modebar=self.modebar,
                )
            )
        )

        def relayout_publisher(event):
            if not event["new"]:
                return

            relayout_data: dict = event["new"]["relayout_data"]

            if relayout_data.get("selections", None):
                selected_points = self.selected_points
                if selected_points.empty:
                    selected_points = None
                self.broker.publish("dplt_selected_points_change", selected_points)

                with self.g.batch_update():
                    self.g.update_traces(selectedpoints=None)
                    self.g.plotly_relayout({"selections": None})

                return

            if dragmode := relayout_data.get("dragmode", None):
                ndims = self.controls.n_dimensions.value
                self.state[f"{ndims}dragmode"] = dragmode
                return

            if shapes := relayout_data.get("shapes", None):
                self.broker.publish("dplt_layout_shapes_change", shapes=shapes)
                return

        self.g.observe(relayout_publisher, names="_js2py_relayout")

    def _init_controls(self):
        self.controls = PlotterControls(self.broker)

    def load_dataset(self, dataset: SCLabDataset | None):
        if dataset is None:
            dataset = SCLabDataset(self.broker)

        if not isinstance(dataset, SCLabDataset):
            raise TypeError("dataset must be an instance of SCLabDataset")

        if dataset.broker.id != self.broker.id:
            raise ValueError("dataset broker must be the same as the provided broker")

        self.dataset = dataset

        self.broker.publish("dplt_dataset_change", dataset)

    @property
    def selected_points(self):
        selectedpoints = set()
        for data in self.g.data:
            if isinstance(data, go.Contour) or not data.selectedpoints:
                continue
            ids = data.hovertext[list(data.selectedpoints)]
            selectedpoints = selectedpoints.union(ids)
        return pd.Index(selectedpoints)

    def select_points(self, points: pd.Index | None):
        if self.controls.n_dimensions.value != "2D":
            self.make_new_figure()
            return
        else:
            self.update_marker_sizes()
            return

    @property
    def data_for_plot(self):
        # TODO: Needs refactoring. Define clear behavior

        if not self.controls.rotate_steps.value:
            return self.dataset.data

        ndims_avail = self.dataset.data.shape[1]
        col_x: str = self.controls.selected_axes.children[0].value
        col_y: str = self.controls.selected_axes.children[1].value
        col_z: str = self.controls.selected_axes.children[2].value

        if ndims_avail < 3 or not col_z:
            return self.dataset.data

        X = self.dataset.data[[col_x, col_y, col_z]]
        return rotate_multiple_steps(X, self.controls.rotate_steps.value)

    def make_new_figure(
        self,
        metadata: pd.DataFrame | None = None,
        colorby: str | None = None,
        sizeby: str | None = None,
        marker_size_scale: float | None = None,
        new_value: str | None = None,
        *args,
        **figure_kwargs,
    ):
        self.g.layout.annotations = []

        ndims = self.controls.n_dimensions.value
        col_x: str = self.controls.selected_axes.children[0].value
        col_y: str = self.controls.selected_axes.children[1].value
        col_z: str = self.controls.selected_axes.children[2].value

        invalid_axes = col_x == "rank" and not col_y

        if self.dataset.data.empty or invalid_axes:
            layout = dict(
                xaxis_title="", yaxis_title="", title="No Data", template="simple_white"
            )
            self.g.update(dict(layout=layout, data=[]), overwrite=True)
            return

        data = self.data_for_plot
        if metadata is None:
            metadata = self.dataset.metadata

        df = data.join(metadata.loc[:, ~metadata.columns.isin(data.columns)])
        if col_x == "rank":
            df = df.sort_values(col_y, ascending=False, na_position="last")
            df["rank"] = np.arange(df.shape[0]) + 1

        log_x: bool = self.controls.log_axes.children[0].value
        log_y: bool = self.controls.log_axes.children[1].value
        log_z: bool = self.controls.log_axes.children[2].value

        if log_x and ndims in ["1D", "2D", "3D"]:
            df = df[df[col_x] > 0]
            df["log " + col_x] = df[col_x].apply(np.log10)
            col_x = "log " + col_x

        if log_y and ndims in ["2D", "3D"]:
            df = df[df[col_y] > 0]
            df["log " + col_y] = df[col_y].apply(np.log10)
            col_y = "log " + col_y

        if log_z and ndims == "3D":
            df = df[df[col_z] > 0]
            df["log " + col_z] = df[col_z].apply(np.log10)
            col_z = "log " + col_z

        log_x = log_y = log_z = False

        if ndims in ["1D", "2D", "3D"]:
            x = df[col_x]
            dx = x.max() - x.min()

        if ndims in ["2D", "3D"]:
            y = df[col_y]
            dy = y.max() - y.min()

        if ndims == "3D":
            z = df[col_z]
            dz = z.max() - z.min()

        selected = df["is_selected"]
        selection_is_active = not selected.isna().all()
        if selection_is_active and (ndims == "1D" or ndims == "3D"):
            df = df.loc[selected]

        if colorby is None:
            colorby = self.controls.color.value

        if colorby:
            ascending = is_any_real_numeric_dtype(df[colorby])
            df = df.sort_values(colorby, ascending=ascending, na_position="first")

            series: pd.Series = df[colorby]
            if isinstance(series.dtype, pd.CategoricalDtype | bool):
                df[colorby] = series.astype(str).replace("nan", "NA")
            elif ndims == "1D":
                # break into 10 evenly distributed bins
                df[colorby] = pd.cut(series, 10).astype(str).replace("nan", "NA")

        if ndims == "1D":
            fig = px.histogram(
                df,
                x=col_x,
                log_x=log_x,
                color=colorby,
                color_discrete_sequence=COLOR_DISCRETE_SEQUENCE,
                template="simple_white",
                nbins=self.controls.histogram_nbins.value,
                **figure_kwargs,
            )
        elif ndims == "2D":
            fig = px.scatter(
                df,
                x=col_x,
                y=col_y,
                log_x=log_x,
                log_y=log_y,
                color=colorby,
                color_discrete_sequence=COLOR_DISCRETE_SEQUENCE,
                hover_name=df.index,
                template="simple_white",
                render_mode="webgl",
                **figure_kwargs,
            )
        elif ndims == "3D":
            fig = px.scatter_3d(
                df,
                x=col_x,
                y=col_y,
                z=col_z,
                log_x=log_x,
                log_y=log_y,
                log_z=log_z,
                color=colorby,
                color_discrete_sequence=COLOR_DISCRETE_SEQUENCE,
                hover_name=df.index,
                template="simple_white",
                **figure_kwargs,
            )
        else:
            layout = dict(
                xaxis_title="",
                yaxis_title="",
                title="No Data",
                template="simple_white",
                height=self.controls.plot_height.value,
                modebar=self.modebar,
                dragmode=False,
            )
            self.g.update(dict(layout=layout, data=[]), overwrite=True)
            return

        fig.update_layout(legend_title_text="")
        fig.update_layout(coloraxis_colorbar_title_text="")

        fig.update_traces(marker_color="lightgray", selector={"name": "NA"})
        if colorby and is_bool_dtype(series):
            fig.update_traces(marker_color="lightgray", selector={"name": "False"})
            fig.update_traces(
                marker_color=COLOR_DISCRETE_SEQUENCE[0], selector={"name": "True"}
            )
            fig.data = fig.data[::-1]

        if colorby and isinstance(series.dtype, pd.CategoricalDtype):
            color_pallete_size = len(COLOR_DISCRETE_SEQUENCE)
            for i, cat in enumerate(series.cat.categories):
                fig.update_traces(
                    marker_color=COLOR_DISCRETE_SEQUENCE[i % color_pallete_size],
                    selector={"name": cat},
                )

        if ndims == "2D" or ndims == "3D":
            # trace_opacity = self.controls.marker_opacity.value
            # fig.update_traces(opacity=trace_opacity)

            fig.update_layout(legend_traceorder="reversed")
            if col_x == "rank" or not self.controls.enable_hover_info.value:
                fig.update_traces(hoverinfo="skip", hovertemplate=None)

        else:
            fig.data = fig.data[::-1]
            fig.update_layout(legend_traceorder="normal")

        if ndims == "2D" and self.controls.show_density.value:
            # make density plot
            data = df.sort_index()[[col_x, col_y]].values
            data = tuple(tuple(row) for row in data)
            grid_resolution = self.controls.density_grid_resolution.value
            line_smoothing = self.controls.density_line_smoothing.value
            bandwidth_factor = self.controls.density_bandwidth_factor.value
            contours = self.controls.density_contours.value
            trace = _make_density_heatmap(
                data,
                bandwidth_factor,
                grid_resolution,
                line_smoothing,
                contours,
                "rgba(255, 255, 255, 0)",
            )
            fig.add_trace(trace)

        height = self.controls.plot_height.value
        fig.update_layout(height=height)

        if self.controls.aspect_equal.value:
            if ndims == "3D":
                fig.update_layout(scene_aspectmode="data")
            elif ndims == "2D":
                fig.update_xaxes(scaleanchor="y", scaleratio=1)
                fig.update_yaxes(scaleanchor="x", scaleratio=1)

        # fix ranges
        if ndims == "1D":
            fig.update_layout(xaxis_range=[x.min() - dx * 0.1, x.max() + dx * 0.1])

        elif ndims == "2D":
            fig.update_layout(xaxis_range=[x.min() - dx * 0.1, x.max() + dx * 0.1])
            fig.update_layout(yaxis_range=[y.min() - dy * 0.1, y.max() + dy * 0.1])

        elif ndims == "3D":
            fig.update_layout(
                scene_xaxis_range=[x.min() - dx * 0.1, x.max() + dx * 0.1]
            )
            fig.update_layout(
                scene_yaxis_range=[y.min() - dy * 0.1, y.max() + dy * 0.1]
            )
            fig.update_layout(
                scene_zaxis_range=[z.min() - dz * 0.1, z.max() + dz * 0.1]
            )

        # set dragmode
        if not (dragmode := self.get_dragmode()):
            if ndims == "3D":
                dragmode = "turntable"
            elif ndims == "2D":
                dragmode = "lasso"
            else:
                dragmode = False

        fig.update_layout(title=self.dataset._selected_data_key, modebar=self.modebar)
        fig.update_layout(template_layout_shapedefaults_fillcolor="rgba(0, 0, 0, 0)")
        fig.update_layout(legend={"itemsizing": "constant"})

        if colorby:
            fig.update_layout(showlegend=True)
        else:
            fig.update_layout(showlegend=False)

        with self.g.batch_update():
            self.g.update(fig.to_dict(), overwrite=True)
            self.update_marker_sizes(
                metadata=metadata,
                colorby=colorby,
                sizeby=sizeby,
                marker_size_scale=marker_size_scale,
            )
            self.set_dragmode(dragmode)

    def set_dragmode(self, dragmode: str | None = None):
        ndims = self.controls.n_dimensions.value
        if dragmode is None:
            dragmode = self.state.get(f"{ndims}dragmode", False)

        self.g.plotly_relayout({"dragmode": dragmode})
        self.state[f"{ndims}dragmode"] = dragmode

    def get_dragmode(self) -> str:
        ndims = self.controls.n_dimensions.value
        return self.state.get(f"{ndims}dragmode", False)

    def set_shapes(self, shapes: list[dict]):
        self.g.layout.shapes = shapes
        self.g._send_relayout_msg({"shapes": shapes})
        self.broker.publish("dplt_layout_shapes_change", shapes=shapes)

    def clear_shapes(self):
        self.set_shapes([])

    def update_marker_sizes(
        self,
        marker_size_scale: float | None = None,
        sizeby: str | None = None,
        colorby: str | None = None,
        metadata: pd.DataFrame | None = None,
    ):
        if colorby is None:
            colorby = self.controls.color.value

        if marker_size_scale is None:
            marker_size_scale = self.controls.marker_size.value

        if sizeby is None:
            sizeby = self.controls.sizeby.value

        if metadata is None:
            df = self.dataset.metadata
        else:
            df = metadata

        sizeby_series = pd.Series(1.0, index=df.index)
        marker_sizeref = 1.0 / marker_size_scale**2

        # is_selected may be a boolean column or a column of NaNs
        if active_selection := (not (selected := df["is_selected"]).isna().all()):
            # if it is a boolean column, a selection has been defined (possible all False)
            sizeby_series.loc[selected] = 3.0
            sizeby_series.loc[~selected] = 0.5

        elif sizeby is not None:
            sizeby_series = df[sizeby].astype(float)
            sizeby_series.loc[sizeby_series < 0] = 0.0
            sizeby_series = sizeby_series.fillna(0.0)
            marker_sizeref = sizeby_series.max() / marker_size_scale**2

        trace: go.Scatter | go.Scattergl | go.Scatter3d
        for trace in self.g.data:
            marker_ids = trace.hovertext
            if not isinstance(marker_ids, NDArray | list):
                continue

            if df.index.intersection(marker_ids).empty:
                continue

            trace.hovertemplate = self.get_hovertemplate(
                info={colorby: trace.name}, show_size=not active_selection
            )

            marker: Marker2D | Marker3D = trace.marker
            marker.sizemode = "area"
            marker.sizeref = marker_sizeref
            marker.size = sizeby_series.loc[marker_ids].values
            marker.line.width = 0.0

    def get_hovertemplate(self, info: dict = {}, show_size: bool = True) -> str:
        ndims = self.controls.n_dimensions.value
        x_axis = self.controls.selected_axes.children[0].value
        y_axis = self.controls.selected_axes.children[1].value
        z_axis = self.controls.selected_axes.children[2].value
        marker_color: str | None = self.controls.color.value
        marker_size: str | None = self.controls.sizeby.value

        if marker_color is not None:
            series = self.dataset.metadata[marker_color]
            marker_color_is_cat = isinstance(series.dtype, pd.CategoricalDtype)
        else:
            marker_color_is_cat = False

        is_histogram = ndims == "1D"
        is_scatter = ndims in ["2D", "3D"]
        is_3d_scatter = ndims == "3D"
        is_na = ndims == "NA"

        hovertemplate = ""
        if is_scatter | is_na:
            hovertemplate = "<b>%{hovertext}</b><br>"

        if is_scatter and marker_color and not marker_color_is_cat:
            info.pop(marker_color, None)
            hovertemplate += f"{marker_color} = %{{marker.color}}<br>"

        if is_scatter and marker_color and marker_color_is_cat:
            trace_name = info.pop(marker_color, None)
            if trace_name is not None:
                hovertemplate += f"{marker_color} = {trace_name}<br>"

        if is_scatter and marker_size and show_size and marker_size != marker_color:
            hovertemplate += f"{marker_size} = %{{marker.size}}<br>"

        if is_histogram:
            hovertemplate += "count = %{y}<br>"

        hovertemplate = hovertemplate[:-4]
        hovertemplate += "<extra>"

        if not is_na:
            hovertemplate += f"{x_axis} = %{{x}}<br>"
        if is_scatter:
            hovertemplate += f"{y_axis} = %{{y}}<br>"
        if is_3d_scatter:
            hovertemplate += f"{z_axis} = %{{z}}<br>"

        if info:
            hovertemplate += "<br>"

        for key, val in info.items():
            if key or val:
                hovertemplate += f"{key} = {val}<br>"
        else:
            hovertemplate = hovertemplate[:-4]

        hovertemplate += "</extra>"

        return hovertemplate

    def dplt_point_click_callback(self, row_name, device_state, **kwargs):
        ndims = self.controls.n_dimensions.value
        if ndims == "1D":
            return

        for trace in self.g.data:
            if isinstance(trace, go.Contour):
                continue

            marker_size = trace.marker.size.copy()
            # if ndims == "2D":
            #     marker_opacity = trace.marker.opacity.copy()

            if row_name in trace.hovertext:
                idx = list(trace.hovertext).index(row_name)

                default_size = self.controls.marker_size.value
                current_size = marker_size[idx]

                if current_size == default_size:
                    marker_size[idx] = default_size * 3
                    # if ndims == "2D":
                    #     marker_opacity[idx] = 1.0
                else:
                    marker_size[idx] = default_size
                    # if ndims == "2D":
                    #     marker_opacity[idx] = 0.7

                trace.marker.size = marker_size
                # if ndims == "2D":
                #     with self.g.batch_animate():
                #         trace.marker.size = marker_size
                #         trace.marker.opacity = marker_opacity
                # else:
                #     with self.g.batch_update():
                #         trace.marker.size = marker_size

                break

    def ctrl_show_density_change_callback(self, new_value):
        if new_value:
            try:
                trace = next(filter(lambda o: isinstance(o, go.Contour), self.g.data))
                trace.visible = True
            except StopIteration:
                self.make_new_figure()
        else:
            try:
                trace = next(filter(lambda o: isinstance(o, go.Contour), self.g.data))
                trace.visible = False
            except StopIteration:
                pass

    def ctrl_density_line_smoothing_change_callback(self, new_value):
        if self.controls.show_density.value:
            self.g.update_traces(
                selector=dict(type="contour"), line_smoothing=new_value
            )

    def ctrl_density_contours_change_callback(self, new_value):
        if self.controls.show_density.value:
            trace: go.Contour = next(
                self.g.select_traces(selector=dict(type="contour"))
            )

            z_values: NDArray = trace.z
            start = z_values.min() + 0.0001
            end = z_values.max() + 0.0001
            size = (end - start) / new_value
            contours = dict(start=start, end=end, size=size)

            self.g.update_traces(selector=dict(type="contour"), contours=contours)

    def ctrl_density_grid_resolution_change_callback(self, new_value):
        if self.controls.show_density.value:
            self.make_new_figure()

    def ctrl_density_bandwidth_factor_change_callback(self, new_value):
        if self.controls.show_density.value:
            self.make_new_figure()

    def ctrl_marker_size_change_callback(self, new_value):
        ndims = self.controls.n_dimensions.value
        if ndims == "1D":
            return

        with self.g.batch_update():
            self.update_marker_sizes(marker_size_scale=new_value)

    def ctrl_sizeby_change_callback(self, new_value):
        ndims = self.controls.n_dimensions.value
        if ndims == "1D":
            return

        with self.g.batch_update():
            self.update_marker_sizes(sizeby=new_value)

    def ctrl_marker_opacity_change_callback(self, new_value):
        ndims = self.controls.n_dimensions.value
        if ndims == "1D":
            return

        with self.g.batch_update():
            self.g.update_traces(selector=dict(type="scatter"), opacity=new_value)
            self.g.update_traces(selector=dict(type="scattergl"), opacity=new_value)
            self.g.update_traces(selector=dict(type="scatter3d"), opacity=new_value)

    def ctrl_aspect_equal_change_callback(self, new_value):
        ndims = self.controls.n_dimensions.value
        if new_value:
            if ndims == "3D":
                self.g.update_layout(scene_aspectmode="data")
            elif ndims == "2D":
                self.g.update_xaxes(scaleanchor="y", scaleratio=1)
                self.g.update_yaxes(scaleanchor="x", scaleratio=1)
        else:
            if ndims == "3D":
                self.g.update_layout(scene_aspectmode="auto")
            elif ndims == "2D":
                self.g.update_xaxes(scaleanchor=None, scaleratio=None)
                self.g.update_yaxes(scaleanchor=None, scaleratio=None)

    def ctrl_enable_hover_info_change_callback(self, new_value):
        if new_value:
            colorby = self.controls.color.value
            for trace in self.g.data:
                trace.hoverinfo = "all"
                trace.hovertemplate = self.get_hovertemplate(info={colorby: trace.name})
        else:
            self.g.update_traces(hoverinfo="skip", hovertemplate=None)

    def ctrl_plot_width_change_callback(self, new_value):
        if new_value > 0:
            self.graph_box.layout.width = f"{new_value}px"
        else:
            self.graph_box.layout.width = "auto"

    def ctrl_plot_height_change_callback(self, new_value):
        self.graph_box.layout.height = f"{new_value}px"
        self.g.update_layout(height=new_value)

    def ctrl_save_button_click_callback(self):
        dataset_name = self.dataset.name
        data_key = self.dataset._selected_data_key
        if not data_key:
            return

        filename = f"{dataset_name}___saved_figure___data_{data_key}"

        ndims = self.controls.n_dimensions.value
        if ndims == "1D":
            variable = self.controls.selected_axes.children[0].value
            filename += f"___{variable}-histogram"
        elif ndims == "2D":
            x = self.controls.selected_axes.children[0].value
            y = self.controls.selected_axes.children[1].value
            filename += f"___{x}-vs-{y}"
        elif ndims == "3D":
            x = self.controls.selected_axes.children[0].value
            y = self.controls.selected_axes.children[1].value
            z = self.controls.selected_axes.children[2].value
            filename += f"___{x}-vs-{y}-vs-{z}"

        if self.controls.color.value:
            filename += f"___colorBy_{self.controls.color.value}"

        filename += ".html"
        self.g.write_html(filename)

    def dspr_clear_selection_click_callback(self):
        with self.g.batch_update():
            if self.controls.n_dimensions.value == "2D":
                self.g.update_traces(selectedpoints=None)
            self.g.plotly_relayout({"selections": None})

    def dplt_start_drawing_request_callback(self):
        if "2D" not in self.controls.n_dimensions.options:
            return

        self.controls.dict["n_dimensions"].value = "2D"
        self.state["drawing"] = True
        self.state["previous_dragmode"] = self.get_dragmode()
        for control_name in [
            "data_key",
            "color",
            "sizeby",
            "n_dimensions",
            "selected_axes_1",
            "selected_axes_2",
            "selected_axes_3",
            "log_axes_1",
            "log_axes_2",
            "log_axes_3",
        ]:
            self.controls.dict[control_name].disabled = True
        self.set_dragmode("drawopenpath")

    def dplt_end_drawing_request_callback(self):
        if not self.state.get("drawing", False):
            return

        self.state["drawing"] = False
        self.clear_shapes()
        for control_name in [
            "data_key",
            "color",
            "sizeby",
            "n_dimensions",
            "selected_axes_1",
            "selected_axes_2",
            "selected_axes_3",
            "log_axes_1",
            "log_axes_2",
            "log_axes_3",
        ]:
            self.controls.dict[control_name].disabled = False
        self.set_dragmode(self.state["previous_dragmode"])

    def dplt_layout_shapes_change_callback(self, shapes: list[dict]):
        shapes = [s for s in shapes if s["name"] != "smooth path"]

        if not shapes:
            return

        path = spt.Path()
        for shape in shapes:
            p = spt.parse_path(shape["path"])
            if len(path) > 0:
                path.append(spt.Line(path[-1].end, p[0].start))
            path.extend(p)

        pi, pf = path[0].start, path[-1].end
        if np.abs(pf - pi) / path.length() < 0.01:
            path[-1].end = pi

        n = len(path) + 1
        if n < 3:
            return

        p = np.array([path[0].start] + [line.end for line in path])
        path_is_closed = p[-1] == p[0]

        if n >= 5:
            X = np.array([[line.start, line.end] for line in path])
            t = np.abs(np.diff(X))
            t = np.insert(np.cumsum(t / t.sum()), 0, 0)
            if path_is_closed:
                px_bspl = make_periodic_smoothing_spline(
                    t[:-1], p.real[:-1], t_range=(0, 1), lam=1 / 5e3 / n
                )
                py_bspl = make_periodic_smoothing_spline(
                    t[:-1], p.imag[:-1], t_range=(0, 1), lam=1 / 5e3 / n
                )
                t = np.linspace(0, 1, 10 * n) % 1
            else:
                px_bspl = make_smoothing_spline(t, p.real, lam=1 / 1e3 / n)
                py_bspl = make_smoothing_spline(t, p.imag, lam=1 / 1e3 / n)
                t = np.linspace(0, 1, 10 * n)
            x, y = px_bspl(t), py_bspl(t)

            points = x + 1j * y
            spath = spt.Path()
            spath.extend(
                [spt.Line(start, end) for start, end in zip(points, points[1:])]
            )

        else:
            spath = path

        s1 = {
            "editable": False,
            "visible": False,
            "name": "drawn path",
            "showlegend": False,
            "legend": "legend",
            "legendgroup": "",
            "legendgrouptitle": {
                "text": "",
                "font": {"weight": "normal", "style": "normal", "variant": "normal"},
            },
            "legendrank": 1000,
            "label": {"text": "", "texttemplate": ""},
            "xref": "x",
            "yref": "y",
            "layer": "above",
            "opacity": 1,
            "line": {"color": "#444", "width": 4, "dash": "solid"},
            "type": "path",
            "path": path.d(use_closed_attrib=path_is_closed).replace(" ", ""),
        }
        s2 = {
            "editable": False,
            "visible": True,
            "name": "smooth path",
            "showlegend": False,
            "legend": "legend",
            "legendgroup": "",
            "legendgrouptitle": {
                "text": "",
                "font": {"weight": "normal", "style": "normal", "variant": "normal"},
            },
            "legendrank": 1000,
            "label": {"text": "", "texttemplate": ""},
            "xref": "x",
            "yref": "y",
            "layer": "above",
            "opacity": 1,
            "line": {"color": "#444", "width": 4, "dash": "solid"},
            "type": "path",
            "path": spath.d(use_closed_attrib=path_is_closed).replace(" ", ""),
        }

        self.g._send_relayout_msg({"shapes": (s1, s2)})

        times = np.linspace(0, 1, 100 * n)
        if n >= 5:
            points = px_bspl(times) + 1j * py_bspl(times)
        else:
            points = np.array([path.point(t) for t in times])

        col_x: str = self.controls.selected_axes.children[0].value
        col_y: str = self.controls.selected_axes.children[1].value
        x = self.data_for_plot[col_x].values
        y = self.data_for_plot[col_y].values

        X = x + 1j * y
        P = points
        T = times

        self.broker.publish(
            "dplt_soft_path_computed",
            time_points=T,
            data_points=X,
            path_points=P,
            path_is_closed=path_is_closed,
        )

    def dplt_plot_figure_request_callback(
        self,
        figure: go.Figure | None = None,
        metadata: pd.DataFrame | None = None,
        colorby: str | None = None,
        sizeby: str | None = None,
        **figure_kwargs,
    ):
        if figure is None:
            self.make_new_figure(
                metadata=metadata,
                colorby=colorby,
                sizeby=sizeby,
                **figure_kwargs,
            )
            return

        if colorby in self.controls.color.options:
            self.controls.color.value = colorby
        else:
            self.controls.color.value = None

        if sizeby in self.controls.sizeby.options:
            self.controls.sizeby.value = sizeby
        else:
            self.controls.sizeby.value = None

        figure.layout.template = self.g.layout.template

        with self.g.batch_update():
            self.g.update(figure.to_dict(), overwrite=True)
            self.update_marker_sizes(colorby=colorby, sizeby=sizeby)
            self.g.plotly_relayout({"dragmode": False})

    def dplt_add_trace_request_callback(self, trace: go.Scatter | go.Scattergl):
        with self.g.batch_update():
            self.g.add_trace(trace)

    def dplt_add_vline_request_callback(self, vlines: float | list[float], **kwargs):
        if not isinstance(vlines, list):
            vlines = [vlines]

        with self.g.batch_update():
            for vline in vlines:
                self.g.add_vline(x=vline, **kwargs)

    def dplt_add_hline_request_callback(self, hlines: float | list[float], **kwargs):
        if not isinstance(hlines, list):
            hlines = [hlines]

        with self.g.batch_update():
            for hline in hlines:
                self.g.add_hline(y=hline, **kwargs)

    def dplt_add_data_as_line_trace_request_callback(
        self, data_key: str, data: pd.DataFrame, **kvargs
    ):
        if data_key != self.dataset._selected_data_key:
            return

        ndims = self.controls.n_dimensions.value
        col_x: str = self.controls.selected_axes.children[0].value
        col_y: str = self.controls.selected_axes.children[1].value
        col_z: str = self.controls.selected_axes.children[2].value

        data = rotate_multiple_steps(data, self.controls.rotate_steps.value)

        if ndims == "1D":
            if col_x + "_height" not in data.columns:
                return

            x = data[col_x].values
            y = data[col_x + "_height"].values
            trace = go.Scattergl(x=x, y=y, mode="lines", **kvargs)

        elif ndims == "2D":
            x = data[col_x].values
            y = data[col_y].values
            trace = go.Scattergl(x=x, y=y, mode="lines", **kvargs)

        elif ndims == "3D":
            x = data[col_x].values
            y = data[col_y].values
            z = data[col_z].values
            trace = go.Scatter3d(x=x, y=y, z=z, mode="lines", **kvargs)

        else:
            return

        with self.g.batch_update():
            self.g.add_trace(trace)
