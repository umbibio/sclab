import pandas as pd
from ipywidgets import (
    Accordion,
    Button,
    Checkbox,
    Dropdown,
    FloatSlider,
    HBox,
    IntSlider,
    Layout,
    Tab,
    Textarea,
    ToggleButtons,
    ValueWidget,
    VBox,
)

from ...event import EventBroker, EventClient
from .._dataset import SCLabDataset


class PlotterControls(EventClient, VBox):
    data_key: Dropdown
    n_dimensions: ToggleButtons
    aspect_equal: Checkbox
    enable_hover_info: Checkbox
    marker_size: FloatSlider
    marker_opacity: FloatSlider
    plot_width: IntSlider
    plot_height: IntSlider
    selected_axes: VBox
    log_axes: VBox
    histogram_nbins: IntSlider
    color: Dropdown
    show_density: Checkbox
    density_grid_resolution: IntSlider
    density_line_smoothing: FloatSlider
    density_bandwidth_factor: FloatSlider
    density_contours: IntSlider
    refresh_button: Button
    save_button: Button

    preemptions: dict[str, list[str]] = {
        "ctrl_data_key_change": [
            "ctrl_n_dimensions_change",
        ]
    }

    def __init__(self, broker: EventBroker):
        self._make_controls()
        self._make_controls_layout()
        VBox.__init__(self, self.controls_layout, layout=Layout(width="100%"))

        self.dict: dict[str, ValueWidget | Button] = {
            "data_key": self.data_key,
            "color": self.color,
            "sizeby": self.sizeby,
            "n_dimensions": self.n_dimensions,
            "selected_axes_1": self.selected_axes.children[0],
            "selected_axes_2": self.selected_axes.children[1],
            "selected_axes_3": self.selected_axes.children[2],
            "log_axes_1": self.log_axes.children[0],
            "log_axes_2": self.log_axes.children[1],
            "log_axes_3": self.log_axes.children[2],
            "rotate_steps": self.rotate_steps,
            "marker_size": self.marker_size,
            "marker_opacity": self.marker_opacity,
            "plot_width": self.plot_width,
            "plot_height": self.plot_height,
            "histogram_nbins": self.histogram_nbins,
            "aspect_equal": self.aspect_equal,
            "enable_hover_info": self.enable_hover_info,
            "show_density": self.show_density,
            "density_grid_resolution": self.density_grid_resolution,
            "density_line_smoothing": self.density_line_smoothing,
            "density_bandwidth_factor": self.density_bandwidth_factor,
            "density_contours": self.density_contours,
            "refresh_button": self.refresh_button,
            "save_button": self.save_button,
        }

        self.events = (
            [
                f"ctrl_{control_name}_change"
                for control_name, widget in self.dict.items()
                if not isinstance(widget, Button)
            ]
            + [
                f"ctrl_{control_name}_click"
                for control_name, widget in self.dict.items()
                if isinstance(widget, Button)
            ]
            + [
                f"ctrl_{control_name}_value_change_request"
                for control_name, widget in self.dict.items()
                if isinstance(widget, ValueWidget)
            ]
            + [
                "ctrl_value_change_request",
            ]
        )
        EventClient.__init__(self, broker)

        for control_name, widget in self.dict.items():
            if isinstance(widget, Button):
                widget.on_click(self.button_click_event_publisher("ctrl", control_name))
            else:
                widget.observe(
                    self.control_value_change_event_publisher("ctrl", control_name),
                    names="value",
                    type="change",
                )
        for control_name, widget in self.dict.items():
            if isinstance(widget, ValueWidget):
                callback = self.control_value_change_request_callback_generator(widget)
                self.broker.subscribe(
                    f"ctrl_{control_name}_value_change_request", callback
                )

        def get_callback(checkbox: Checkbox):
            def callback(_):
                checkbox.value = False

            return callback

        for i in range(3):
            dropdown: Dropdown = self.selected_axes.children[i]
            checkbox: Checkbox = self.log_axes.children[i]
            dropdown.observe(get_callback(checkbox), names="value", type="change")

        self.broker.subscribe("dplt_dataset_change", self.populate)

        self.broker.subscribe("dset_data_dict_change", self.set_data_key_dd_options)
        self.broker.subscribe(
            "dset_data_key_selection_change", self.set_selected_axes_dd_options
        )
        self.broker.subscribe("dset_metadata_change", self.set_color_dd_options)
        self.broker.subscribe("dset_metadata_change", self.set_sizeby_dd_options)

    def control_value_change_request_callback_generator(self, widget: ValueWidget):
        def callback(new_value):
            widget.value = new_value

        return callback

    def _make_controls(self):
        self.data_key: Dropdown = Dropdown(
            description="Data",
            options={"": None},
            value=None,
            layout=dict(margin="10px 0px 0px 0px"),
        )
        self.n_dimensions: ToggleButtons = ToggleButtons(
            options=[
                "NA",  # Will be initialized later and populated based on the data
                # Options should be ["1D", "2D", "3D"]
            ],
            value="NA",
            button_style="primary",
            layout=dict(margin="10px 0px 0px 0px"),
            disabled=True,
        )
        self.aspect_equal: Checkbox = Checkbox(
            description="Aspect equal",
            value=False,
        )
        self.enable_hover_info: Checkbox = Checkbox(
            description="Enable hover info",
            value=True,
        )
        self.marker_size: FloatSlider = FloatSlider(
            min=0.01,
            max=20.0,
            description="Marker size",
            value=8.0,
            step=0.01,
            continuous_update=True,
            readout=False,
        )
        self.marker_opacity: FloatSlider = FloatSlider(
            min=0,
            max=1,
            description="Opacity",
            value=1,
            step=0.01,
            continuous_update=True,
            readout=False,
        )
        self.plot_width: IntSlider = IntSlider(
            min=0,
            max=2000,
            description="Plot width",
            value=0,
            step=10,
            continuous_update=True,
            readout=False,
        )
        self.plot_height: IntSlider = IntSlider(
            min=100,
            max=2000,
            description="Plot height",
            value=600,
            step=10,
            continuous_update=True,
            readout=False,
        )
        self.selected_axes = VBox(
            [
                Dropdown(description=f"{ax} axis", options={"": None}, value=None)
                for ax in ["X", "Y", "Z"]
            ]
        )
        self.log_axes = VBox(
            [
                Checkbox(description=f"{ax} axis log scale", value=False)
                for ax in ["X", "Y", "Z"]
            ]
        )
        self.rotate_steps: Textarea = Textarea(
            description="Rotation steps",
            value="",
            rows=5,
        )
        self.histogram_nbins: IntSlider = IntSlider(
            min=5,
            max=200,
            description="Histo. bins",
            value=100,
            step=5,
            continuous_update=True,
            readout=False,
        )
        self.color: Dropdown = Dropdown(
            description="Color",
            options={"": None},
            value=None,
            layout=dict(margin="10px 0px 0px 0px"),
        )
        self.sizeby: Dropdown = Dropdown(
            description="Size by",
            options={"": None},
            value=None,
            layout=Layout(margin="10px 0px 0px 0px"),
        )
        self.show_density: Checkbox = Checkbox(
            description="Show density",
            value=False,
        )
        self.density_grid_resolution: IntSlider = IntSlider(
            min=4,
            max=8,
            description="Resolution",
            value=6,
            step=1,
            continuous_update=False,
            readout=False,
        )
        self.density_line_smoothing: FloatSlider = FloatSlider(
            min=0,
            max=1.3,
            description="Smoothing",
            value=1.3,
            step=0.05,
            continuous_update=True,
            readout=False,
        )
        self.density_bandwidth_factor: FloatSlider = FloatSlider(
            min=0.002,
            max=0.02,
            description="Bandwidth",
            value=0.017,
            step=0.003,
            continuous_update=False,
            readout=False,
        )
        self.density_contours: IntSlider = IntSlider(
            min=5,
            max=100,
            description="Contours",
            value=20,
            step=1,
            continuous_update=True,
            readout=False,
        )
        self.refresh_button: Button = Button(
            description="Refresh", button_style="primary"
        )
        self.save_button: Button = Button(description="Save", button_style="primary")

    def _make_controls_layout(self):
        plot_settings_tab = Tab(
            children=[
                self.selected_axes,
                self.log_axes,
                self.rotate_steps,
            ],
            titles=["Axes", "Log axes", "Rotate"],
            layout=Layout(margin="10px 0px 0px 0px"),
        )

        plot_settings_grid = VBox(
            [
                self.data_key,
                self.color,
                self.sizeby,
                self.n_dimensions,
                plot_settings_tab,
            ]
        )

        general_visual_settings = VBox(
            [
                self.marker_size,
                self.marker_opacity,
                self.plot_width,
                self.plot_height,
                self.aspect_equal,
                self.enable_hover_info,
            ]
        )

        histogram_visual_settings = VBox(
            [
                self.histogram_nbins,
            ]
        )

        density_visual_settings = VBox(
            [
                self.show_density,
                self.density_bandwidth_factor,
                self.density_grid_resolution,
                self.density_line_smoothing,
                self.density_contours,
            ]
        )

        visual_settings = Tab(
            [
                general_visual_settings,
                histogram_visual_settings,
                density_visual_settings,
            ],
            titles=["General", "Histogram", "Density"],
        )

        sliders_grid = Accordion(
            [
                plot_settings_grid,
                visual_settings,
            ],
            titles=["Plot Settings", "Visual Settings"],
            selected_index=0,
            layout=dict(width="auto"),
        )

        self.plot_width.layout.width = "95%"
        self.plot_height.layout.width = "95%"
        self.marker_size.layout.width = "95%"
        self.marker_opacity.layout.width = "95%"
        self.histogram_nbins.layout.width = "95%"
        self.aspect_equal.layout.width = "95%"
        self.enable_hover_info.layout.width = "95%"
        self.show_density.layout.width = "95%"
        self.density_grid_resolution.layout.width = "95%"
        self.density_line_smoothing.layout.width = "95%"
        self.density_bandwidth_factor.layout.width = "95%"
        self.density_contours.layout.width = "95%"

        self.data_key.layout.width = "95%"
        self.data_key.style.description_width = "15%"
        self.color.layout.width = "95%"
        self.color.style.description_width = "15%"
        self.sizeby.layout.width = "95%"
        self.sizeby.style.description_width = "15%"

        self.n_dimensions.layout.width = "auto"
        self.n_dimensions.style.button_width = "30%"
        self.n_dimensions.layout.padding = "0px 0px 0px 17%"

        self.selected_axes.layout.width = "95%"
        self.log_axes.layout.width = "95%"
        self.rotate_steps.layout.width = "95%"
        for i in range(3):
            self.selected_axes.children[i].layout.width = "95%"
            self.log_axes.children[i].layout.width = "95%"

        self.controls_layout = [
            sliders_grid,
            HBox([self.refresh_button, self.save_button]),
        ]

    def clear(self):
        self.data_key.options = {"": None}
        self.data_key.value = None

        self.color.options = {"": None}
        self.color.value = None

        self.sizeby.options = {"": None}
        self.sizeby.value = None

        self.n_dimensions.value = None

        dropdown_list: list[Dropdown] = self.selected_axes.children
        for dropdown in dropdown_list:
            dropdown.options = {"": None}
            dropdown.value = None
            dropdown.disabled = True
            dropdown.layout.visibility = "hidden"

        checkbox_list: list[Checkbox] = self.log_axes.children
        for checkbox in checkbox_list:
            checkbox.value = False
            checkbox.disabled = True
            checkbox.layout.visibility = "hidden"

        self.marker_size.disabled = True
        self.marker_opacity.disabled = True
        self.histogram_nbins.disabled = True

    def dset_metadata_change_callback(
        self, metadata: pd.DataFrame, value: str | None = None
    ):
        if self.data_key.value == "metadata":
            data = metadata.select_dtypes(include=["number"])
            self.set_selected_axes_dd_options(data)

    def set_color_dd_options(self, metadata: pd.DataFrame, value: str | None = None):
        options = metadata.select_dtypes(include=["bool", "category", "number"]).columns

        if value is not None:
            pass
        elif self.color.value not in options:
            value = None
        else:
            value = self.color.value

        self.color.options = {"": None, **{c: c for c in options}}
        self.color.value = value

    def set_sizeby_dd_options(self, metadata: pd.DataFrame, _: str | None = None):
        df = metadata.select_dtypes(include=["number"])

        # only show columns that have non-negative values for all rows
        options = df.columns[(df >= 0).sum() == df.shape[0]]

        if self.sizeby.value not in options:
            value = None
        else:
            value = self.sizeby.value

        self.sizeby.options = {"": None, **{c: c for c in options}}
        self.sizeby.value = value

    def set_data_key_dd_options(
        self, data_dict: dict[str, pd.DataFrame], value: str | None = None
    ):
        options = list(data_dict.keys())

        if value is not None:
            pass
        elif self.data_key.value not in options:
            value = None
        else:
            value = self.data_key.value

        # self.data_key.disabled = True
        self.data_key.options = {"": None, **{c: c for c in options}}
        # self.data_key.disabled = False

        self.data_key.value = value

    def set_selected_axes_dd_options(self, data: pd.DataFrame):
        columns = data.columns
        data_ncols = len(columns)

        dropdown_list: list[Dropdown] = self.selected_axes.children
        checkbox_list: list[Checkbox] = self.log_axes.children

        options = {c: c for c in columns}
        for i, (column, dropdown, checkbox) in enumerate(
            zip(columns, dropdown_list, checkbox_list)
        ):
            current_value = dropdown.value

            if i == 0:
                dropdown.options = {**options}
            else:
                dropdown.options = options

            if current_value is not None and current_value in dropdown.options:
                dropdown.value = current_value
            else:
                dropdown.value = column
                checkbox.value = False

        n = data_ncols
        for dropdown, checkbox in zip(dropdown_list[:n], checkbox_list[:n]):
            dropdown.disabled = False
            dropdown.layout.visibility = "visible"

            checkbox.disabled = False
            checkbox.layout.visibility = "visible"

        for dropdown, checkbox in zip(dropdown_list[n:], checkbox_list[n:]):
            dropdown.disabled = True
            dropdown.layout.visibility = "hidden"

            checkbox.disabled = True
            checkbox.layout.visibility = "hidden"

        for dropdown, checkbox in zip(
            dropdown_list[data_ncols:], checkbox_list[data_ncols:]
        ):
            dropdown.options = None
            dropdown.value = None
            checkbox.value = False

    def ctrl_n_dimensions_change_callback(self, new_value: str | None):
        dropdown_list: list[Dropdown] = self.selected_axes.children
        checkbox_list: list[Checkbox] = self.log_axes.children
        if new_value is None:
            for dropdown, checkbox in zip(dropdown_list, checkbox_list):
                dropdown.disabled = True
                dropdown.layout.visibility = "hidden"
                checkbox.disabled = True
                checkbox.value = False
                checkbox.layout.visibility = "hidden"
        else:
            # ndim = int(new_value[0])
            for i, (dropdown, checkbox) in enumerate(zip(dropdown_list, checkbox_list)):
                dropdown.disabled = False
                dropdown.layout.visibility = "visible"
                checkbox.disabled = False
                checkbox.value = False
                checkbox.layout.visibility = "visible"

        if new_value is None:
            self.marker_size.disabled = True
            self.marker_opacity.disabled = True
            self.histogram_nbins.disabled = False
        elif new_value == "1D":
            self.marker_size.disabled = True
            self.marker_opacity.disabled = True
            self.histogram_nbins.disabled = False
        else:
            self.marker_size.disabled = False
            self.marker_opacity.disabled = False
            self.histogram_nbins.disabled = True

    def ctrl_value_change_request_callback(self, **kwargs):
        for control_name, new_value in kwargs.items():
            control = self.dict[control_name]
            assert isinstance(control, ValueWidget)

            control.value = new_value

    def dset_data_key_selection_change_callback(self, data: pd.DataFrame):
        data_ncols = len(data.columns)

        if data_ncols == 1:
            options = ["1D"]
            value = "1D"
        elif data_ncols == 2:
            options = ["1D", "2D"]
            value = "2D"
        elif data_ncols >= 3:
            options = ["1D", "2D", "3D"]
            value = "2D"
        else:
            options = ["NA"]
            value = "NA"

        self.n_dimensions.options = options
        self.n_dimensions.value = value
        self.n_dimensions.disabled = data_ncols == 0

    def populate(self, dataset: SCLabDataset):
        if not isinstance(dataset, SCLabDataset):
            raise TypeError("dataset must be an instance of SCLabDataset")

        data_dict = dataset.data_dict
        data = dataset.data
        metadata = dataset.metadata

        # if not data_dict:
        #     self.clear()
        #     return

        self.set_data_key_dd_options(data_dict)
        self.set_selected_axes_dd_options(data)
        self.set_color_dd_options(metadata)
        self.set_sizeby_dd_options(metadata)
