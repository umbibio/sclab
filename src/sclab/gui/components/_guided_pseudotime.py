import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import Button, Dropdown, FloatLogSlider, FloatSlider, HBox, Text
from numpy import floating
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from sclab.dataset.processor import Processor
from sclab.dataset.processor.step import ProcessorStepBase

# TODO: remove self.drawn_path and self.drawn_path_residue from the class
#       and add them to the dataset as _drawn_path and _drawn_path_residue


_2PI = 2 * np.pi


class GuidedPseudotime(ProcessorStepBase):
    parent: Processor
    name: str = "guided_pseudotime"
    description: str = "Guided Pseudotime"

    run_button_description = "Compute Pseudotime"

    def __init__(self, parent: Processor) -> None:
        variable_controls = dict(
            residue_threshold=FloatLogSlider(
                value=1,
                min=-3,
                max=0,
                description="Filter by Dist.",
                continuous_update=True,
            ),
            use_rep=Dropdown(
                options=tuple(parent.dataset.adata.obsm.keys()),
                value=None,
                description="Use rep.",
            ),
            roughness=FloatSlider(
                value=0,
                min=0,
                max=3,
                step=0.05,
                description="Roughness",
                continuous_update=False,
            ),
            min_snr=FloatSlider(
                value=0.25,
                min=0,
                max=1,
                step=0.05,
                description="SNR",
                continuous_update=False,
            ),
            key_added=Text(
                value="pseudotime",
                description="Key added",
                placeholder="",
            ),
        )

        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )
        self.run_button.disabled = True
        self.run_button.button_style = ""
        self.estimate_start_time_button.layout.visibility = "hidden"
        self.estimate_start_time_button.layout.height = "0px"

        self.start_drawing_button.on_click(self.toggle_drawing_callback)
        self.plot_signal.on_click(self.send_signal_plot)
        self.estimate_start_time_button.on_click(
            self.estimate_periodic_pseudotime_start
        )
        self.plot_fitted_pseudotime_curve.on_click(self.send_fitted_pseudotime_plot)
        self.variable_controls["use_rep"].observe(
            self.update_buttons_state, names="value", type="change"
        )
        self.variable_controls["residue_threshold"].observe(
            self._assign_drawn_path_values, names="value", type="change"
        )

        self.variable_controls["use_rep"].observe(
            self.close_signal_plot, names="value", type="change"
        )
        self.variable_controls["roughness"].observe(
            self.close_signal_plot, names="value", type="change"
        )
        self.variable_controls["min_snr"].observe(
            self.close_signal_plot, names="value", type="change"
        )

    def make_controls(self):
        self.start_drawing_button = Button(
            description="Start Drawing", button_style="primary"
        )

        self.auto_drawing_button = Button(
            description="Automatic Drawing", button_style="primary"
        )
        self.auto_drawing_button.on_click(self._automatic_periodic_path_drawing)

        self.plot_signal = Button(
            description="Plot Signal", button_style="info", disabled=True
        )

        self.estimate_start_time_button = Button(
            description="Estimate Pseudotime Start",
            button_style="",
            disabled=True,
        )

        self.plot_fitted_pseudotime_curve = Button(
            description="Plot Fitted Pseudotime Curve",
            button_style="info",
            disabled=True,
        )

        self.controls_list = [
            HBox([self.auto_drawing_button, self.start_drawing_button]),
            *self.variable_controls.values(),
            self.plot_signal,
            self.run_button,
            self.plot_fitted_pseudotime_curve,
            self.estimate_start_time_button,
            self.output,
        ]
        super().make_controls()

    def update_buttons_state(self, *args, **kwargs):
        drawing = self.start_drawing_button.description != "Start Drawing"
        if self.variable_controls["use_rep"].value is None or drawing:
            self.run_button.disabled = True
            self.run_button.button_style = ""

            self.plot_signal.disabled = True
            self.plot_signal.button_style = ""

            self.plot_fitted_pseudotime_curve.disabled = True
            self.plot_fitted_pseudotime_curve.button_style = ""

            self.estimate_start_time_button.disabled = True
            self.estimate_start_time_button.button_style = ""
            self.estimate_start_time_button.layout.visibility = "hidden"
            self.estimate_start_time_button.layout.height = "0px"

            return

        self.run_button.disabled = False
        self.run_button.button_style = "primary"

        self.plot_signal.disabled = False
        self.plot_signal.button_style = "info"

        self.plot_fitted_pseudotime_curve.disabled = False
        self.plot_fitted_pseudotime_curve.button_style = "info"

    def toggle_drawing_callback(self, _: Button | None = None):
        if self.start_drawing_button.description == "Start Drawing":
            self.start_drawing_button.disabled = False
            self.start_drawing_button.button_style = "warning"
            self.start_drawing_button.description = "--> click here when ready <--"

            self.update_buttons_state()

            self.broker.publish("dplt_start_drawing_request")
            self.update_output("Use your mouse pointer to draw a path on the figure")
        else:
            self.start_drawing_button.disabled = False
            self.start_drawing_button.button_style = "primary"
            self.start_drawing_button.description = "Start Drawing"

            self.update_buttons_state()

            self.broker.publish("dplt_end_drawing_request")
            self.update_output(
                "Click on the **Run** button to fit a pseudotime curve to the data"
                + " points. Make sure to select a data representation before"
                + " running the analysis."
            )

    def estimate_periodic_pseudotime_start(self, _: Button | None = None):
        from cellflow.pseudotime._pseudotime import estimate_periodic_pseudotime_start

        time_key = self.variable_controls["key_added"].value
        estimate_periodic_pseudotime_start(self.parent.dataset.adata, time_key=time_key)
        self.broker.publish(
            "dset_metadata_change", self.parent.dataset.metadata, time_key
        )
        self.estimate_start_time_button.button_style = "success"

    def send_signal_plot(self, _: Button | None = None):
        from cellflow.utils.interpolate import NDBSpline

        if self.plot_signal.description == "Plot Signal":
            adata = self.parent.dataset.adata
            use_rep = self.variable_controls["use_rep"].value
            roughness = self.variable_controls["roughness"].value
            min_snr = self.variable_controls["min_snr"].value
            periodic = self.parent.dataset.adata.uns["drawn_path"]["path_is_closed"]

            t_range = (0.0, 1.0)
            tmin, tmax = t_range

            t = adata.obs["drawn_path"].values
            X = adata.obsm[use_rep]

            df = pd.DataFrame(
                X,
                columns=[f"Dim {i + 1}" for i in range(X.shape[1])],
                index=adata.obs_names,
            )
            df = df.join(self.parent.dataset.metadata)
            df["index"] = df.index

            t_mask = (tmin <= t) * (t <= tmax)
            t = t[t_mask]
            X = X[t_mask]
            df = df.loc[t_mask]

            max_dims = 16
            ndims = min(X.shape[1], max_dims)

            F = NDBSpline(t_range=t_range, periodic=periodic, roughness=roughness)
            F.fit(t, X)

            SNR: NDArray[floating] = F(t).var(axis=0) / X.var(axis=0)
            SNR = SNR / SNR.max()

            x = np.linspace(*t_range, 200)
            Y = F(x)

            rows = cols = int(np.ceil(np.sqrt(ndims)))
            titles = [f"Dim {i + 1}. SNR: {SNR[i]:.2f}" for i in range(ndims)]
            fig = make_subplots(
                rows=rows,
                cols=cols,
                shared_xaxes=True,
                shared_yaxes=False,
                x_title="Drawn path",
                y_title="Signal",
                subplot_titles=titles,
            )

            for i in range(ndims):
                row = i // cols + 1
                col = i % cols + 1
                snr = SNR[i]
                marker_color = "blue" if snr >= min_snr else "lightgray"
                line_color = "red" if snr >= min_snr else "gray"

                scatter = px.scatter(
                    df,
                    x="drawn_path",
                    y=f"Dim {i + 1}",
                    template="simple_white",
                    hover_name="index",
                )
                scatter.update_traces(marker=dict(size=5, color=marker_color))

                for trace in scatter.data:
                    fig.add_trace(trace, row=row, col=col)

                line = go.Scattergl(
                    x=x,
                    y=Y[:, i],
                    mode="lines",
                    line_color=line_color,
                )
                fig.add_trace(line, row=row, col=col)

            fig.update_layout(showlegend=False, title=f"{use_rep} Signal Plot")
            self.plot_signal.description = "Close Signal Plot"
            self.plot_signal.button_style = "warning"

        else:
            fig = None
            self.plot_signal.description = "Plot Signal"
            self.plot_signal.button_style = "info"

        self.broker.publish("dplt_plot_figure_request", figure=fig)

    def close_signal_plot(self, *args, **kwargs):
        self.plot_signal.description = "Plot Signal"
        self.plot_signal.button_style = "info"
        self.broker.publish("dplt_plot_figure_request", figure=None)

    def function(
        self,
        use_rep: str,
        roughness: float,
        min_snr: float,
        key_added: str,
        **kwargs,
    ):
        from cellflow.pseudotime._pseudotime import pseudotime

        self.plot_signal.description = "Plot Signal"
        self.plot_signal.button_style = "info"

        periodic = self.parent.dataset.adata.uns["drawn_path"]["path_is_closed"]

        self.output.clear_output(wait=True)
        with self.output:
            pseudotime(
                adata=self.parent.dataset.adata,
                use_rep=use_rep,
                t_key="drawn_path",
                t_range=(0.0, 1.0),
                min_snr=min_snr,
                periodic=periodic,
                method="splines",
                roughness=roughness,
                key_added=key_added,
            )

        self.parent.dataset.clear_selected_rows()
        self.broker.publish("ctrl_data_key_value_change_request", use_rep)
        self.broker.publish(
            "dset_metadata_change", self.parent.dataset.metadata, key_added
        )

        self.send_fitted_pseudotime_plot()
        self.update_output("")

        if periodic:
            self.estimate_start_time_button.disabled = False
            self.estimate_start_time_button.button_style = "primary"
            self.estimate_start_time_button.layout.visibility = "visible"
            self.estimate_start_time_button.layout.height = "28px"

    def send_fitted_pseudotime_plot(self, *args, **kwargs):
        use_rep = self.variable_controls["use_rep"].value
        key_added = self.variable_controls["key_added"].value

        t: NDArray = self.parent.dataset.adata.obs[key_added].values
        t_mask = ~np.isnan(t)
        t = t[t_mask]

        X_path = self.parent.dataset.adata.obsm[f"{key_added}_path"]
        data = self.parent.dataset.data.copy()
        data.values[:] = X_path
        data: pd.DataFrame = data.loc[t_mask]
        data = data.iloc[t.argsort()]
        self.broker.publish(
            "dplt_add_data_as_line_trace_request",
            use_rep,
            data,
            name=key_added,
            line_color="red",
        )

    def _assign_drawn_path_values(self, *args, **kwargs):
        dataset = self.parent.dataset

        drawn_path = self.drawn_path.copy()
        drawn_path_residue = self.drawn_path_residue.copy()

        residue_threshold = self.variable_controls["residue_threshold"].value
        x = drawn_path_residue / drawn_path_residue.max()
        drawn_path.loc[x > residue_threshold] = np.nan

        # detecting outliers: points with projection to the curve, but that are not
        # part of the cluster of points where the user drew the path
        x = drawn_path_residue.loc[drawn_path.notna()]
        x = x[~np.isnan(x)].values
        # sort in descending order
        x = np.sort(x)[::-1]
        # normalize
        y = x / x.max()
        # detect jumps in the normalized values
        d = y[:-1] - y[1:]
        if (d > 0.25).any():
            # if there is a spike in the residue values, and the spike is larger than
            # 25% of the maximum residue value, then remove the points with residue
            # values larger than the threshold
            thr = x[:-1][d > 0.25].min()
            drawn_path.loc[drawn_path_residue >= thr] = np.nan

        selected: NDArray = drawn_path.notna().values
        if selected.any():
            dataset.selected_rows = dataset.row_names[selected]
        else:
            dataset.selected_rows = None

        publish_change = "drawn_path" not in dataset._metadata
        dataset._metadata["drawn_path"] = drawn_path.clip(0, 1)
        if publish_change:
            self.broker.publish("dset_metadata_change", dataset.metadata)

    def _automatic_periodic_path_drawing(self, *args, **kwargs):
        from cellflow.pseudotime._pseudotime import periodic_parameter
        from cellflow.utils.interpolate import NDFourier

        data_points_array = self.parent.plotter.data_for_plot.values[:, :2]
        ordr_points_array = periodic_parameter(data_points_array) / _2PI
        F = NDFourier(t_range=(0, 1), grid_size=128, smoothing_fn=np.median)
        F.fit(ordr_points_array, data_points_array)

        T = time_points = np.linspace(0, 1, 1024 + 1)
        path_points_array = F(time_points)

        X = data_points_array[:, 0] + 1j * data_points_array[:, 1]
        P = path_points_array[:, 0] + 1j * path_points_array[:, 1]
        self._compute_drawn_path(T, X, P, path_is_closed=True)
        self._assign_drawn_path_values()
        line = go.Scattergl(x=P.real, y=P.imag, mode="lines", line_color="black")
        self.broker.publish("dplt_add_trace_request", trace=line)

    def _compute_drawn_path(
        self,
        time_points: NDArray[floating],
        data_points: NDArray[floating],
        path_points: NDArray[floating],
        path_is_closed: bool,
    ):
        dataset = self.parent.dataset

        T = time_points
        X = data_points
        P = path_points

        idxs = np.sort(
            np.argsort(np.abs(X[:, None] - P[None, :]), axis=1)[:, :3], axis=1
        )

        drawn_path = pd.Series(index=dataset._metadata.index, dtype=float)
        drawn_path_residue = pd.Series(index=dataset._metadata.index, dtype=float)

        T1, T2, T3 = T[idxs].T
        P1, P2, P3 = P[idxs].T

        A = P2 - P1
        B = X - P1
        C = P2 - X
        d = B.real * A.real + B.imag * A.imag
        e = A.real * C.real + A.imag * C.imag
        gap = np.abs(B - d * A / np.abs(A) ** 2)
        m = mask1 = (d > 0) * (e > 0)
        pseudotime = T1[m] + d[m] / np.abs(A[m]) * (T2[m] - T1[m])
        drawn_path.loc[m] = pseudotime
        drawn_path.loc[m] = pseudotime
        drawn_path_residue.loc[m] = gap[m]

        if (~mask1).sum() > 0:
            A = P3[~mask1] - P1[~mask1]
            B = X[~mask1] - P1[~mask1]
            C = P3[~mask1] - X[~mask1]
            d = B.real * A.real + B.imag * A.imag
            e = A.real * C.real + A.imag * C.imag
            gap = np.abs(B - d * A / np.abs(A) ** 2)
            m = mask2 = np.zeros_like(mask1)
            mask2[~mask1] = submsk = (d > 0) * (e > 0)
            pseudotime = T1[m] + d[submsk] / np.abs(A[submsk]) * (T3[m] - T1[m])
            drawn_path.loc[m] = pseudotime
            drawn_path_residue.loc[m] = gap[submsk]

        self.drawn_path = drawn_path.copy()
        self.drawn_path_residue = drawn_path_residue.copy()

        dataset.adata.uns["drawn_path"] = dict(
            t_range=[0.0, 1.0],
            periodic=path_is_closed,
            path_is_closed=path_is_closed,
        )

    def dplt_soft_path_computed_callback(
        self,
        time_points: NDArray[floating],
        data_points: NDArray[floating],
        path_points: NDArray[floating],
        path_is_closed: bool,
    ):
        self._compute_drawn_path(time_points, data_points, path_points, path_is_closed)
        self._assign_drawn_path_values()
        line = go.Scattergl(
            x=path_points.real, y=path_points.imag, mode="lines", line_color="black"
        )
        self.broker.publish("dplt_add_trace_request", trace=line)
