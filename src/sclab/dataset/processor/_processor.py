import html
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from ipywidgets.widgets import (
    HTML,
    Accordion,
    Button,
    Checkbox,
    Dropdown,
    FloatRangeSlider,
    HBox,
    IntRangeSlider,
    Output,
    SelectMultiple,
    Tab,
    Text,
    ToggleButton,
    VBox,
)
from ipywidgets.widgets.valuewidget import ValueWidget
from ipywidgets.widgets.widget_description import DescriptionWidget
from pandas import CategoricalDtype
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)
from traitlets import TraitError

from ..._methods_registry import get_sclab_methods
from ...event import EventBroker, EventClient
from ...utils import (
    CodeActions,
    layer_keys,
    set_options_keep_value,
    text_or_category_columns,
)
from .._dataset import SCLabDataset
from ..plotter import Plotter
from . import _codegen, _history
from ._citations_panel import CitationsPanel
from ._results_panel import ResultsPanel
from ._selection_code import selection_mask, store_selection, stored_selections

logger = logging.getLogger(__name__)


def _accepted_params(function: callable, params: dict) -> dict:
    """The subset of ``params`` that ``function`` accepts as keywords."""
    parameters = inspect.signature(function).parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in parameters):
        return params
    names = {p.name for p in parameters}
    return {k: v for k, v in params.items() if k in names}


class _UnavailableStep:
    """The panel of a step that could not be built: the package it needs.

    Added by :meth:`Processor.add_steps` when the step's constructor raises
    ``ImportError`` (a missing optional dependency); running it re-raises.
    """

    def __init__(self, step: type, error: ImportError):
        self.name, self.description = step.name, step.description
        self.error = error
        reason = str(error) or f"Needs {error.name or 'a missing package'}"
        self.controls = VBox(
            [HTML(f"<span style='color:#b26a00'>{html.escape(reason)}</span>")]
        )

    def run(self, **_):
        raise self.error

    function = run


# forward declaration
class ProcessorStepBase(EventClient):
    events: list[str] = None
    parent: "Processor"
    name: str
    description: str
    fixed_params: dict[str, Any]
    variable_controls: dict[str, DescriptionWidget | ValueWidget]
    output: Output
    run_button: Button
    controls_list: list[DescriptionWidget | ValueWidget | Button]
    controls: VBox
    run_button_description = "Run"


# forward declaration
class BasicProcessorStep(EventClient):
    events: list[str] = None
    parent: "Processor"
    description: str
    function_name: str
    function: callable
    fixed_params: dict[str, Any]
    variable_controls: dict[str, DescriptionWidget | ValueWidget]
    output: Output
    run_button: Button
    controls: VBox


_ProcessorStep = BasicProcessorStep | ProcessorStepBase


class Processor(EventClient):
    dataset: SCLabDataset
    plotter: Plotter
    results_panel: ResultsPanel
    batch_key: str | None
    batch_values: list[str] | None
    broker: EventBroker
    metadata_table: pd.DataFrame
    selection_controls_list: list[DescriptionWidget | ValueWidget]
    selection_controls_dict: dict[str, DescriptionWidget | ValueWidget]
    selection_controls_container: VBox
    selection_labeling_controls_dict: dict[
        str, DescriptionWidget | ValueWidget | Accordion
    ]
    selection_buttons_dict: dict[str, Button | Accordion]
    all_controls_list: list[DescriptionWidget | ValueWidget]
    steps: dict[str, BasicProcessorStep | ProcessorStepBase]
    step_history: list[dict[str, str | tuple | dict]]
    main_accordion: Accordion
    _loaded_step_history: list[dict[str, str | tuple | dict]] | None = None
    # last plot (box/lasso) selection and its shape; None if none or not replayable
    _plot_selection: pd.Index | None = None
    _plot_shape: dict | None = None
    # replaces history_to_code in generate_code, e.g. a Workspace's merge script
    code_generator: Callable[[], str] | None = None

    events = [
        "dspr_selection_values_change",
        "dspr_clear_selection_click",
        "dspr_keep_selected_click",
        "dspr_drop_selected_click",
        "dspr_apply_label_click",
        "dspr_batch_key_change",  # (old, new), see set_batch_key
    ]

    def __init__(
        self,
        dataset: SCLabDataset,
        plotter: Plotter,
        *,
        batch_key: str | None = None,
        source: str | None = None,
    ):
        self.dataset = dataset
        self.source = source  # path or URL the data was loaded from, for the script
        self.plotter = plotter
        self.results_panel = ResultsPanel(self.dataset.broker)
        self.broker = self.dataset.broker
        self.selection_controls_list = []
        self.selection_controls_dict = {}
        self.selection_controls_container = VBox(layout=dict(width="100%"))
        self.selection_labeling_controls_dict = {}
        self.selection_buttons_dict = {}
        self.selection_status = HTML()  # the last Keep/Drop/Label error, if any
        self.all_controls_list = []
        # a dataset saved by sclab carries its history; summarized, display-only
        self.step_history = _history.loads_uns(self.dataset.adata)
        self.history_table = HTML()
        self.code_toggle = ToggleButton(description="Show code", icon="code")
        self.code_actions = CodeActions()
        self.code_view = HTML(layout=dict(display="none"))  # <pre>: selectable
        self.code_toggle.observe(self._toggle_code_view, "value")
        self.history_panel = VBox(
            [
                self.history_table,
                HBox([self.code_toggle, self.code_actions]),
                self.code_view,
                CitationsPanel(self.dataset),
            ]
        )
        self._update_code = self.broker.coalesce(self._render_code)
        self._render_step_history()

        def update_category_control_visibility_callback(change):
            if "new" not in change or not change["new"]:
                return
            new_category = change["new"]

            if "old" in change and change["old"] == change["new"]:
                return

            if "old" in change and change["old"]:
                old_category = change["old"]
                control = self.selection_controls_dict[old_category]
                control.layout.visibility = "hidden"
                control.layout.height = "0px"
            else:
                for control in self.selection_controls_list:
                    if isinstance(control, SelectMultiple):
                        control.layout.visibility = "hidden"
                        control.layout.height = "0px"

            control = self.selection_controls_dict[new_category]
            n_options = len(control.options)
            h = np.clip(n_options * 18 + 15, 20, 150)
            control.layout.height = f"{h}px"
            control.layout.visibility = "visible"

        self.visible_category_dropdown = Dropdown(options=[], description="Category")
        self.visible_category_dropdown.layout.width = "95%"
        self.visible_category_dropdown.layout.margin = "0px 0px 10px"
        self.visible_category_dropdown.style.description_width = "0px"
        self.visible_category_dropdown.observe(
            update_category_control_visibility_callback, "value", "change"
        )
        self._make_selection_buttons()
        self._make_selection_labeling_controls()
        self._make_selection_controls()

        self.main_accordion = Accordion(
            [
                self.selection_controls_container,
            ],
            titles=[
                "Selection Controls",
            ],
        )
        self.steps = {}
        self.plotter.data_hash = self.data_hash  # for pinned views
        self.add_step_func("filter_rows", self.filter_rows, use_run_button=False)
        self.add_step_func("apply_label", self.apply_label, use_run_button=False)
        for name in ("filter_rows", "apply_label"):  # their panels are not shown
            self.steps[name].status = self.selection_status

        super().__init__(self.broker)
        self.broker.subscribe("dset_metadata_change", self._make_selection_controls)

        # None: "batch" if there is such a column; "": no batch key
        if "batch" in self.dataset.adata.obs.columns and batch_key is None:
            batch_key = "batch"
        self.batch_key = self.batch_values = None
        self._batch_dropdowns: list[Dropdown] = []  # see make_selectbatch_drowpdown
        self.set_batch_key(batch_key)

        registered_methods = get_sclab_methods()
        self.add_steps(registered_methods)
        self._update_code()  # the steps' code generators are known now

    @property
    def step_groups(self) -> dict[str, Accordion]:
        return dict(zip(self.main_accordion.titles, self.main_accordion.children))

    def _create_new_step_group(
        self,
        group_name: str,
        group_steps: dict[str, _ProcessorStep] | None = None,
    ):
        if group_steps is None:
            group_steps = {}

        # accordion childrens are widgets, we need to extract the corresponding controls
        group_steps_dict = {k: v.controls for k, v in group_steps.items()}

        # create a new step group accordion
        step_group_accordion = Accordion(
            titles=list(group_steps_dict.keys()),
            children=list(group_steps_dict.values()),
        )

        # add the new step group accordion to the main accordion
        children = list(self.main_accordion.children)
        titles = list(self.main_accordion.titles)
        children.append(step_group_accordion)
        titles.append(group_name)
        self.main_accordion.children = tuple(children)
        self.main_accordion.titles = tuple(titles)

    def _update_step_group(self, group_name: str, new_steps: dict[str, _ProcessorStep]):
        # get the current group steps accordion
        group_steps_accordion = self.step_groups[group_name]
        current_steps_dict = dict(
            zip(group_steps_accordion.titles, group_steps_accordion.children)
        )

        # accordion childrens are widgets, we need to extract the corresponding controls
        new_steps_dict = {k: v.controls for k, v in new_steps.items()}

        # update the current group steps accordion. We merge the current and new steps
        steps = {**current_steps_dict, **new_steps_dict}
        group_steps_accordion.children = tuple(steps.values())
        group_steps_accordion.titles = tuple(steps.keys())

    def add_steps(
        self,
        _steps: _ProcessorStep
        | type
        | list[_ProcessorStep | type]
        | dict[str, _ProcessorStep | type]
        | dict[str, list[_ProcessorStep | type]],
        step_group_name: str = "Processing",
    ):
        # we make sure _steps is a dictionary of lists of steps
        """
        Add one or more steps to the dataset processor. The steps can be given as a single
        step instance, a type of step to be instantiated, a list of steps, or a dictionary
        of step group names to lists of steps.

        Parameters
        ----------
        _steps: _ProcessorStep | type | list[_ProcessorStep | type] | dict[str, _ProcessorStep | type] | dict[str, list[_ProcessorStep | type]]
            The steps to add to the dataset processor.
        step_group_name: str, optional
            The name of the step group to add the steps to. If the step group does not exist,
            it will be created. Defaults to "Processing".

        Raises
        ------
        ValueError
            If the step has already been added to the dataset processor.
        """
        from .step._processor_step_base import ProcessorStepBase

        if not isinstance(_steps, list | dict):
            steps = {step_group_name: [_steps]}

        elif isinstance(_steps, list):
            steps = {step_group_name: _steps}

        elif isinstance(_steps, dict):
            steps = _steps
            for key, value in _steps.items():
                if not isinstance(value, list):
                    steps[key] = [value]

        # if there are uninstantiated steps, we instantiate them
        for step_group_name, steps_list in steps.items():
            for i, step in enumerate(steps_list):
                if isinstance(step, type):
                    assert issubclass(step, ProcessorStepBase), (
                        f"{step} must be a subclass of {ProcessorStepBase}"
                    )
                    try:
                        steps_list[i] = step(self)
                    except ImportError as error:  # a missing optional dependency
                        steps_list[i] = _UnavailableStep(step, error)

        steps: dict[str, list[_ProcessorStep]]
        steps_list: list[_ProcessorStep]

        # we make sure the steps have not been previously added
        for step_group_name, steps_list in steps.items():
            for step in steps_list:
                assert step.description not in self.steps, (
                    f"Step {step.description} already exists"
                )

        # we add the new steps
        group_steps_dict: dict[str, _ProcessorStep]
        for step_group_name, steps_list in steps.items():
            group_steps_dict = {step.description: step for step in steps_list}

            if step_group_name in self.step_groups:
                # update the existing step group
                self._update_step_group(step_group_name, group_steps_dict)
            else:
                # create a new step group
                self._create_new_step_group(step_group_name, group_steps_dict)

            # we register the new steps
            for step in steps_list:
                self.steps[step.description] = step
                if isinstance(step, ProcessorStepBase):
                    step.refresh()  # enable its Run button if its inputs are valid

    def add_step_func(
        self,
        description: str,
        function: callable,
        fixed_params: dict[str, Any] = {},
        variable_controls: dict[str, DescriptionWidget | ValueWidget] = {},
        use_run_button: bool = True,
        accordion: Accordion | None = None,
    ):
        from .step import BasicProcessorStep

        step = BasicProcessorStep(
            self, description, function, fixed_params, variable_controls, use_run_button
        )
        self.add_step_object(step, accordion)

    def add_step_object(
        self, step: ProcessorStepBase, accordion: Accordion | None = None
    ):
        assert step.description not in self.steps, (
            f"Step {step.description} already exists"
        )
        self.steps[step.description] = step

        if accordion is not None:
            self.append_to_accordion(accordion, step.controls, step.description)

    def lockable_controls(self) -> list:
        """Widgets disabled while a background step runs: those changing the data."""
        return [
            *self.all_controls_list,
            *self.selection_buttons_dict["keep_drop_buttons"].children,
            self.selection_labeling_controls_dict["apply_button"],
        ]

    def append_to_accordion(self, accordion: Accordion, panel: VBox, title: str):
        children = list(accordion.children)
        children.append(panel)
        accordion.children = tuple(children)
        accordion.set_title(len(children) - 1, title)

    def _active_selection_values(self) -> dict:
        # same notion of "active" as dspr_selection_values_change_callback
        return {
            column: control.value
            for column, control in self.selection_controls_dict.items()
            if (
                bool(control.value)
                if isinstance(control, SelectMultiple)
                else control.value[1] > control.value[0]
            )
        }

    def _selection_filters(self) -> dict:
        """Active filter controls, in the schema of :mod:`._selection_code`."""
        filters = {}
        obs = self.dataset.adata.obs
        for column, value in self._active_selection_values().items():
            series = obs[column]
            if is_numeric_dtype(series) and not is_bool_dtype(series):
                filters[column] = {"between": list(value)}  # range slider
            else:
                filters[column] = {"isin": list(value)}  # select multiple
        return filters

    def _make_selection_controls(self, *args, **kwargs):
        before = self._active_selection_values()

        for column in self.dataset.metadata.columns:
            if column in self.selection_controls_dict:
                self.update_column_selection_control(column)
            else:
                self.add_column_selection_control(column)

        for column in list(self.selection_controls_dict):
            if column not in self.dataset.metadata.columns:
                self.remove_column_selection_control(column)

        after = self._active_selection_values()
        if any(
            after.get(c) != v
            for c, v in before.items()
            if c in self.selection_controls_dict
        ):
            # called directly: publishing is disabled while dset_metadata_change runs
            self.dspr_selection_values_change_callback("", None)

    def _make_selection_buttons(self):
        """
        Create buttons for selection actions

        We make sure these buttons are created only once and are not recreated when
        updating the selection controls.
        """

        clear_selection_button = Button(
            description="Clear Selection",
            button_style="primary",
            layout=dict(width="98%"),
        )
        clear_selection_button.on_click(
            self.button_click_event_publisher("dspr", "clear_selection")
        )

        keep_selected_button = Button(
            description="Keep Selected", button_style="danger"
        )
        keep_selected_button.on_click(
            self.button_click_event_publisher("dspr", "keep_selected")
        )

        drop_selected_button = Button(
            description="Drop Selected", button_style="danger"
        )
        drop_selected_button.on_click(
            self.button_click_event_publisher("dspr", "drop_selected")
        )

        keep_drop_buttons = HBox([keep_selected_button, drop_selected_button])
        keep_drop_buttons_accordion = Accordion(
            [keep_drop_buttons], titles=["Keep/Drop"]
        )

        self.selection_buttons_dict["clear_selection"] = clear_selection_button
        self.selection_buttons_dict["keep_drop_buttons"] = keep_drop_buttons
        self.selection_buttons_dict["keep_drop_accordion"] = keep_drop_buttons_accordion

    def _make_selection_labeling_controls(self):
        # widgets visible when a new key is to be created
        new_medatadata_key = Text(
            description="Key",
            layout=dict(width="98%"),
            placeholder="New metadata key",
        )
        new_label = Text(
            description="Label",
            layout=dict(width="98%"),
            disabled=True,
        )
        new_medatadata_key.layout.visibility = "hidden"
        new_label.layout.visibility = "hidden"
        new_medatadata_key.layout.height = "0px"
        new_medatadata_key.layout.margin = "0px"
        new_label.layout.height = "0px"
        new_label.layout.margin = "0px"

        # widgets visible when an existing key is to be updated
        existing_metadata_key = Dropdown(
            description="Key",
            layout=dict(width="98%"),
        )
        existing_label = Dropdown(
            description="Label",
            layout=dict(width="98%"),
            disabled=True,
        )

        create_new_key_checkbox = Checkbox(
            value=False,
            description="Create new key/label",
            layout=dict(width="98%"),
        )

        def _update_widget_visibility(change: dict):
            df = self.dataset.metadata.select_dtypes(include=["bool", "category"])
            create_new_key = change["new"]
            if create_new_key:
                new_medatadata_key.disabled = False
                new_label.disabled = False
                existing_metadata_key.disabled = True
                existing_label.disabled = True

                new_medatadata_key.layout.visibility = "visible"
                new_label.layout.visibility = "visible"
                existing_metadata_key.layout.visibility = "hidden"
                existing_label.layout.visibility = "hidden"

                new_medatadata_key.layout.height = "28px"
                new_medatadata_key.layout.margin = "2px"
                new_label.layout.height = "28px"
                new_label.layout.margin = "2px"
                existing_metadata_key.layout.height = "0px"
                existing_metadata_key.layout.margin = "0px"
                existing_label.layout.height = "0px"
                existing_label.layout.margin = "0px"

                new_medatadata_key.value = ""
                new_label.value = ""
            else:
                new_medatadata_key.disabled = True
                new_label.disabled = True
                existing_metadata_key.disabled = False
                existing_label.disabled = False

                new_medatadata_key.layout.visibility = "hidden"
                new_label.layout.visibility = "hidden"
                existing_metadata_key.layout.visibility = "visible"
                existing_label.layout.visibility = "visible"

                new_medatadata_key.layout.height = "0px"
                new_medatadata_key.layout.margin = "0px"
                new_label.layout.height = "0px"
                new_label.layout.margin = "0px"
                existing_metadata_key.layout.height = "28px"
                existing_metadata_key.layout.margin = "2px"
                existing_label.layout.height = "28px"
                existing_label.layout.margin = "2px"

                existing_metadata_key.options = [""] + df.columns.to_list()
                existing_metadata_key.value = ""
                existing_label.options = [""]
                existing_label.value = ""

        create_new_key_checkbox.observe(_update_widget_visibility, "value", "change")

        def _update_new_label_options_callback(change: dict):
            metadata_key_value = change["new"]
            if metadata_key_value is None or metadata_key_value == "":
                existing_label.disabled = True
                existing_label.options = [""]
                existing_label.value = ""
            else:
                series: pd.Series = self.dataset.metadata[metadata_key_value]
                if isinstance(series.dtype, CategoricalDtype):
                    existing_label.options = [""] + series.cat.categories.to_list()
                elif is_bool_dtype(series):
                    existing_label.options = ["", True, False]
                elif is_integer_dtype(series) or is_float_dtype(series):
                    existing_label.options = []

                existing_label.disabled = False
            existing_label.value = ""

        existing_metadata_key.observe(
            _update_new_label_options_callback, "value", "change"
        )

        apply_button = Button(
            description="Apply", disabled=True, button_style="primary"
        )
        apply_button.on_click(self.button_click_event_publisher("dspr", "apply_label"))

        def _update_button_disabled(_):
            new_key = new_medatadata_key.value
            existing_key = existing_metadata_key.value

            if create_new_key_checkbox.value:
                apply_button.disabled = new_key == ""
            else:
                apply_button.disabled = existing_key == ""

        new_medatadata_key.observe(_update_button_disabled, "value", "change")
        new_label.observe(_update_button_disabled, "value", "change")
        existing_metadata_key.observe(_update_button_disabled, "value", "change")
        existing_label.observe(_update_button_disabled, "value", "change")

        container = VBox(
            [
                create_new_key_checkbox,
                new_medatadata_key,
                new_label,
                existing_metadata_key,
                existing_label,
                apply_button,
            ]
        )

        self.selection_labeling_controls_dict["create_new_key_checkbox"] = (
            create_new_key_checkbox
        )
        self.selection_labeling_controls_dict["existing_metadata_key"] = (
            existing_metadata_key
        )
        self.selection_labeling_controls_dict["existing_label"] = existing_label
        self.selection_labeling_controls_dict["new_medatadata_key"] = new_medatadata_key
        self.selection_labeling_controls_dict["new_label"] = new_label
        self.selection_labeling_controls_dict["apply_button"] = apply_button
        self.selection_labeling_controls_dict["container"] = container
        self.selection_labeling_controls_dict["accordion"] = Accordion(
            [container],
            titles=["Label"],
        )

    def add_column_selection_control(self, column: str):
        if column in self.selection_controls_dict:
            return

        if column == "is_selected":
            return

        series = self.dataset.metadata[column]
        allna = series.isna().all()
        dtype = series.dtype

        if isinstance(dtype, CategoricalDtype):
            control = SelectMultiple(
                options=series.cat.categories,
                description=column,
                tooltip=column,
            )
            control.style.description_width = "0px"
            control.layout.margin = "0px"

        elif is_bool_dtype(dtype):
            control = SelectMultiple(
                options=[True, False],
                description=column,
                tooltip=column,
            )
            control.style.description_width = "0px"
            control.layout.margin = "0px"

        elif is_integer_dtype(dtype):
            if allna:
                min_value, max_value = 0.0, 0.0
            else:
                min_value, max_value = series.min(), series.max()

            control = IntRangeSlider(
                min=min_value,
                max=max_value,
                value=(min_value, min_value),
                description=column,
                tooltip=column,
            )
            control.style.description_width = "0px"

        elif is_float_dtype(dtype):
            eps = np.finfo(dtype).eps
            if allna:
                min_value, max_value = 0.0, 0.0
            else:
                min_value, max_value = series.min() - eps, series.max() + eps

            span = max_value - min_value
            step = span / 100
            control = FloatRangeSlider(
                min=min_value,
                max=max_value,
                step=step,
                description=column,
                tooltip=column,
                value=(min_value, min_value),
            )
            control.style.description_width = "0px"

        else:
            raise TypeError(f"Unsupported dtype: {series.dtype}")
        control.layout.width = "98%"
        control.observe(self._publish_selection_value_change, "value")

        self.selection_controls_list.append(control)
        self.selection_controls_dict[column] = control
        self._make_selection_controls_container()

    def _make_selection_controls_container(self):
        categorical_controls = [self.visible_category_dropdown]
        range_slider_controls = []

        for control in self.selection_controls_list:
            if isinstance(control, SelectMultiple):
                control.layout.visibility = "hidden"
                control.layout.height = "0px"
                control.layout.margin = "0px"
                categorical_controls.append(control)
            elif isinstance(control, IntRangeSlider | FloatRangeSlider):
                label = HTML(f"{control.description}", layout={"height": "20px"})
                control.layout.height = "20px"
                range_slider_controls.append(label)
                range_slider_controls.append(control)
            else:
                raise RuntimeError(f"Unsupported control type {type(control)}")

        options = [c.description for c in categorical_controls[1:]]
        set_options_keep_value(self.visible_category_dropdown, options)

        tabs = Tab(
            [
                VBox(categorical_controls, layout={"height": "200px"}),
                VBox(range_slider_controls, layout={"height": "200px"}),
            ],
            layout=dict(width="98%"),
            titles=["Categorical", "Numeric"],
        )
        selection_criteria = VBox(
            [
                tabs,
            ]
        )
        selection_actions = VBox(
            [
                Accordion(
                    [
                        self.selection_labeling_controls_dict["container"],
                        self.selection_buttons_dict["keep_drop_buttons"],
                    ],
                    titles=["Label", "Keep/Drop"],
                ),
                self.selection_status,
            ]
        )

        self.selection_controls_container.children = tuple(
            [
                Accordion([selection_criteria], titles=["Selection Criteria"]),
                Accordion([selection_actions], titles=["Selection Actions"]),
                self.selection_buttons_dict["clear_selection"],
            ]
        )

    def update_column_selection_control(self, column: str):
        if column not in self.selection_controls_dict:
            return

        control = self.selection_controls_dict[column]
        series = self.dataset.metadata[column]
        allna = series.isna().all()
        dtype = series.dtype
        if isinstance(dtype, CategoricalDtype):
            old_value = control.value
            control.options = series.cat.categories  # resets value
            control.value = tuple(v for v in old_value if v in series.cat.categories)

        elif is_bool_dtype(dtype):
            pass

        elif is_integer_dtype(dtype):
            old_value = control.value
            if allna:
                min_value, max_value = 0.0, 0.0
            else:
                min_value, max_value = series.min(), series.max()

            try:
                control.min, control.max = min_value, max_value
            except TraitError:
                try:
                    control.max, control.min = max_value, min_value
                except TraitError:
                    pass

            self._restore_slider_value(control, old_value)

        elif is_float_dtype(dtype):
            old_value = control.value
            dtype = series.dtype
            eps = np.finfo(dtype).eps
            min_value, max_value = series.min() - eps, series.max() + eps
            span = max_value - min_value
            step = span / 100
            try:
                control.min = min_value
                control.max = max_value
                control.step = step
            except TraitError:
                try:
                    control.max = max_value
                    control.min = min_value
                    control.step = step
                except TraitError:
                    pass
            self._restore_slider_value(control, old_value)

        else:
            raise TypeError(f"Unsupported dtype: {series.dtype}")

    @staticmethod
    def _restore_slider_value(control, old_value):
        # inactive (no range selected) stays inactive; active range is clipped to the new bounds
        lo, hi = old_value
        if hi > lo:
            lo, hi = (min(max(v, control.min), control.max) for v in (lo, hi))
            control.value = (lo, hi)
        else:
            control.value = (control.min, control.min)

    def remove_column_selection_control(self, column: str):
        if column not in self.selection_controls_dict:
            return

        control = self.selection_controls_dict.pop(column)
        self.selection_controls_list.remove(control)
        if isinstance(control, SelectMultiple):
            was_filtering = bool(control.value)
        else:
            was_filtering = control.value[1] > control.value[0]
        control.close()
        self._make_selection_controls_container()

        if was_filtering:
            # called directly: publishing is disabled while dset_metadata_change runs
            self.dspr_selection_values_change_callback(column, None)

    def append_to_step_history(
        self,
        step_description: str,
        params: dict,
        status: Literal["success", "failed"] = "success",
        error: str | None = None,
        shape_before: tuple[int, int] | None = None,
    ):
        """Record a step run; see :mod:`sclab.dataset.processor._history`."""
        entry = _history.make_entry(
            step_description,
            params,
            status,
            error,
            shape_before,
            self.dataset.adata.shape,
        )
        self.step_history.append(entry)
        self._sync_step_history()

    def data_hash(self) -> str:
        """``history_hash`` of the steps that changed the data.

        Steps with ``plot_only = True`` (e.g. Gene Expression) are left out, so
        plotting does not mark the pinned views as made on other data.
        """
        changed = [
            e
            for e in self.step_history
            if not getattr(self.steps.get(e["step"]), "plot_only", False)
            and e["step"] != "load"  # added by a Workspace; the data is the same
        ]
        return _history.history_hash(changed)

    def _sync_step_history(self):
        uns = self.dataset.adata.uns.setdefault(_history.UNS_KEY, {})
        uns["history"] = _history.dumps_uns(self.step_history)
        self._render_step_history()
        self._update_code()
        self.broker.publish("dset_state_change")

    def generate_code(self, header: bool = True) -> str:
        """Python script reproducing the successful steps of ``step_history``.

        See :func:`sclab.dataset.processor._codegen.history_to_code`.
        """
        if self.code_generator is not None:
            return self.code_generator()
        return _codegen.history_to_code(
            self.step_history,
            steps=self.steps,
            header=header,
            source=self.source,
            counts_layer=self.dataset.counts_layer,
            batch_key=self.batch_key,
        )

    def _render_code(self):
        code = self.generate_code()
        self.code_actions.text = code
        self.code_view.value = (
            "<pre style='font-family:monospace;font-size:12px;user-select:text;"
            f"white-space:pre;overflow:auto'>{html.escape(code)}</pre>"
        )

    def _toggle_code_view(self, change: dict):
        self.code_view.layout.display = "" if change["new"] else "none"
        self.code_toggle.description = "Hide code" if change["new"] else "Show code"

    def _render_step_history(self):
        header = ["#", "Step", "Params", "Status", "Time", "Shape after"]
        rows = []
        for i, entry in enumerate(self.step_history, 1):
            status = entry.get("status", "success")
            if entry.get("error"):
                status = f"{status}: {entry['error']}"
            shape = entry.get("shape_after")
            cells = [
                i,
                entry["step"],
                _history.format_params(entry["params"]),
                status,
                (entry.get("time") or "")[:19].replace("T", " "),
                " x ".join(map(str, shape)) if shape else "",
            ]
            color = "" if status == "success" else " style='color:#c62828'"
            tds = "".join(f"<td>{html.escape(str(c))}</td>" for c in cells)
            rows.append(f"<tr{color}>{tds}</tr>")

        if not rows:
            self.history_table.value = "<i>No steps applied yet.</i>"
            return

        ths = "".join(f"<th>{h}</th>" for h in header)
        self.history_table.value = (
            "<style>.sclab-history td,.sclab-history th"
            "{padding:2px 8px;text-align:left;vertical-align:top}"
            ".sclab-history td:not(:nth-child(3)){white-space:nowrap}</style>"
            f"<table class='sclab-history'><tr>{ths}</tr>{''.join(rows)}</table>"
        )

    def save_step_history(
        self, path: str | Path, format: Literal["pickle", "json"] | None = None
    ):
        """Save ``step_history`` (every entry, failed ones included) to ``path``.

        ``format`` is ``"json"`` or ``"pickle"``; by default the file extension
        (``.json`` / ``.pickle``) decides. ``ValueError`` for anything else.
        Reload it with :meth:`load_step_history` and :meth:`apply_step_history`.
        For a script that reproduces the analysis, use :meth:`generate_code`.
        """
        path = Path(path)
        if format is None:
            format = path.suffix[1:]

        if format == "pickle":
            import pickle

            with open(path, "wb") as f:
                pickle.dump(self.step_history, f)
        elif format == "json":
            with open(path, "w") as f:
                json.dump(self.step_history, f, indent=4)
        else:
            raise ValueError(f"Unsupported format {format}")

    def load_step_history(self, path: str):
        """Load a history saved by :meth:`save_step_history` for replay.

        The format follows the extension (``.json`` or ``.pickle``; ``ValueError``
        otherwise). Nothing is applied: call :meth:`apply_step_history`. A pickle
        runs code on load: only load files you trust.
        """
        path = Path(path)
        format = path.suffix[1:]

        if format == "pickle":
            import pickle

            with open(path, "rb") as f:
                history: list = pickle.load(f)
        elif format == "json":
            with open(path, "r") as f:
                history: list = json.load(f)
        else:
            raise ValueError(f"Unsupported format {format}")

        self._loaded_step_history = [_history.normalize_entry(e) for e in history]

    def apply_step_history(self):
        """Replay the successful steps of the loaded history not yet applied.

        Continues where this dashboard left off: the successful steps already
        in ``step_history`` must match the start of the loaded history (step and
        params), else ``AssertionError``; failed and cancelled entries are
        skipped. Before applying any, every step to replay is checked: if one
        cannot run here (not a step of this dashboard, a missing package, see
        ``ProcessorStepBase.unavailable``, or summarized params), ``ValueError``
        lists them all and nothing is applied.

        Each step runs with its recorded params, including the ``backend`` it
        ran with (``"native"`` is not replaced by scanpy, nor the reverse).

        The replay stops at the first step that fails: ``RuntimeError`` naming
        its number, name and error (the exception it raised, or the error it
        recorded as ``failed``/``cancelled``), and no later step runs. A step
        also fails if a param it recorded is replayed with another value (e.g.
        an implicit input such as ``batch_key`` resolved differently here); params
        the step no longer accepts, or new defaults it adds, are not a mismatch.
        The steps before it stay applied and are in ``step_history`` (the
        failed one too, as ``failed``): fix the cause and call this again to
        continue from the failed step.
        """
        assert self._loaded_step_history is not None, "No step history loaded"

        current_step_history = _history.succeeded(self.step_history)
        new_step_history = _history.succeeded(self._loaded_step_history)
        n = len(current_step_history)
        N = len(new_step_history)
        assert N >= n, "Step history mismatch"

        for i, (present_step, incoming_step) in enumerate(
            zip(current_step_history, new_step_history)
        ):
            present_key = _history.entry_key(present_step)
            assert present_key == _history.entry_key(incoming_step), (
                "Step history mismatch"
            )
            logger.info(f"Step {i + 1: 2d}/{N} already applied: {present_key['step']}")

        problems = []
        for i, entry in enumerate(new_step_history[n:], n + 1):
            step = self.steps.get(entry["step"])
            if step is None:
                reason = "not a step of this dashboard"
            elif isinstance(step, _UnavailableStep):
                reason = str(step.error)
            elif _history.is_summarized(entry["params"]):
                reason = "summarized params (from adata.uns) cannot be replayed"
            elif hasattr(step, "unavailable"):  # not BasicProcessorStep
                reason = step.unavailable(entry["params"])
            else:
                reason = None
            if reason:
                problems.append(f"step {i} ({entry['step']}): {reason}")
        if problems:
            raise ValueError(
                "Cannot replay the loaded history here, nothing was applied:\n"
                + "\n".join(problems)
            )

        for i, entry in enumerate(new_step_history[n:], n + 1):
            step_description = entry["step"]
            logger.info(f"Applying step {i: 2d}/{N}: {step_description}")
            step = self.steps[step_description]
            n_recorded = len(self.step_history)
            try:
                step.run(**_accepted_params(step.function, entry["params"]))
            except Exception as e:  # e.g. refused by validate() before running
                cause, reason = e, f"{type(e).__name__}: {e}"
            else:
                cause, reason = None, self._replay_failure(entry, n_recorded)
            if reason:
                raise RuntimeError(
                    f"Replay stopped at step {i}/{N} ({step_description}): {reason}. "
                    f"Steps 1-{i - 1} stay applied (see step_history); later steps "
                    "were not run."
                ) from cause

    def _replay_failure(self, entry: dict, n_recorded: int) -> str | None:
        """Why the step just replayed for ``entry`` counts as failed, if it does.

        Steps catch their own errors and record them as ``failed`` entries. The
        recorded params must come back unchanged, ignoring only keys the step
        dropped or added (see :func:`_accepted_params`, ``resolved_params``).
        """
        if len(self.step_history) == n_recorded:
            return "the step recorded nothing"
        applied = self.step_history[-1]
        if applied["status"] != "success":
            return applied.get("error") or applied["status"]
        got = _history.entry_key(applied)["params"]
        want = _history.entry_key(entry)["params"]
        if differ := sorted(k for k in got.keys() & want.keys() if got[k] != want[k]):
            return f"replayed with other params than recorded ({', '.join(differ)})"
        return None

    def print_step_history(self, with_hash: bool = False):
        for i, step in enumerate(self.step_history):
            desc = step["step"]
            params = _history.format_params(step["params"])
            if with_hash:
                history_hash = self._get_step_history_hash(self.step_history[: i + 1])
                history_hash = history_hash[:8] + " ..."
            else:
                history_hash = ""
            print(f"({i + 1: 3d}) {history_hash} {desc}({params})")

    def _get_step_history_hash(self, history: list[dict]):
        return _history.history_hash(history)

    @property
    def step_history_hash(self):
        """Hash of the successful steps (step and params) of ``step_history``."""
        return self._get_step_history_hash(self.step_history)

    @property
    def selection_values(self):
        return {
            column: control.value
            for column, control in self.selection_controls_dict.items()
        }

    def _publish_selection_value_change(self, change: dict):
        owner: DescriptionWidget = change["owner"]
        column = owner.description
        new_value = change["new"]
        self.broker.publish("dspr_selection_values_change", column, new_value=new_value)

    def _recorded_selection(self, filters, shape, mask, n) -> pd.Index:
        adata = self.dataset.adata
        if shape and "id" in shape and shape["id"] not in stored_selections(adata):
            store_selection(adata, shape, shape["id"])  # replay: for code export
        selected = selection_mask(adata, filters, shape, mask)
        if n is not None and selected.sum() != n:
            logger.warning(f"Selection matched {selected.sum()} cells, recorded {n}")
        return self.dataset.adata.obs_names[selected]

    def filter_rows(
        self,
        index: pd.Index | list | None = None,
        filters: dict | None = None,
        shape: dict | None = None,
        mask: str | None = None,
        n: int | None = None,
        drop: bool = False,
    ):
        """Keep (or drop) the selected cells; see :mod:`._selection_code`.

        ``index`` (pre-0.5 entries) lists the obs names to keep.
        """
        if index is None:
            selected = self._recorded_selection(filters, shape, mask, n)
            obs_names = self.dataset.adata.obs_names
            index = obs_names[~obs_names.isin(selected)] if drop else selected
        self.dataset.filter_rows(pd.Index(index))

    def apply_label(
        self,
        index: pd.Index | list | None = None,
        column: str = "",
        label: str = "",
        filters: dict | None = None,
        shape: dict | None = None,
        mask: str | None = None,
        n: int | None = None,
    ):
        """Label the selected cells; see :mod:`._selection_code`.

        ``index`` (pre-0.5 entries) lists the obs names to label.
        """
        if index is None:
            index = self._recorded_selection(filters, shape, mask, n)
        self.dataset.apply_label(pd.Index(index), column, label)

    def set_batch_key(self, batch_key: str | None):
        """Set ``batch_key`` (None or "": none); steps update their defaults.

        Controls made only with a batch key (e.g. Neighbors' "Group By Batch")
        appear when the dashboard is built with one.
        """
        old, self.batch_key = self.batch_key, batch_key or None
        self.batch_values = None
        if self.batch_key:
            column = self.dataset.adata.obs[self.batch_key]
            self.batch_values = column.sort_values().astype(str).unique().tolist()
        options = {"": None, **{v: v for v in self.batch_values or ()}}
        for dropdown in self._batch_dropdowns:
            set_options_keep_value(dropdown, options)
        if self.batch_key != old:
            self.broker.publish("dspr_batch_key_change", old, self.batch_key)

    def make_selectbatch_drowpdown(self, description="Select Batch"):
        control = dict()
        if self.batch_key:
            control_key = description.lower().replace(" ", "_")
            control[control_key] = Dropdown(
                options={"": None, **{v: v for v in self.batch_values}},
                value=None,
                description=description,
            )
            self._batch_dropdowns.append(control[control_key])
        return control

    def make_groupbybatch_checkbox(self, description="Group By Batch"):
        control = dict()
        if self.batch_key:
            control["group_by_batch"] = Checkbox(
                value=True,
                description=description,
            )
        return control

    def dspr_selection_values_change_callback(self, column_changed: str, new_value):
        self._update_selected_rows()

    def _update_selected_rows(self):
        """Select the cells in the active filters AND the plot selection."""
        filters = self._selection_filters()
        obs_names = self.dataset.adata.obs_names
        # None means nothing active; an empty Index means active but nothing matched
        selected = None
        if filters or self._plot_selection is not None:
            mask = selection_mask(self.dataset.adata, filters)
            if self._plot_selection is not None:
                mask &= obs_names.isin(self._plot_selection)
            selected = obs_names[mask]

        current = self.dataset._selected_rows
        if (
            selected is not None
            and current is not None
            and len(current) == len(selected)
            and current.isin(selected).all()
        ):
            return  # e.g. the dataset already took the plot selection as is
        self.dataset.selected_rows = selected

    def _selection_params(self) -> dict | None:
        """Provenance of the current selection, see :mod:`._selection_code`."""
        if self.dataset._selected_rows is None:
            return None
        adata = self.dataset.adata
        selected = adata.obs_names.isin(self.dataset._selected_rows)
        filters = self._selection_filters()
        shape = self._plot_shape if self._plot_selection is not None else None
        params = {"n": int(selected.sum())}
        if (filters or shape) and np.array_equal(
            selection_mask(adata, filters, shape), selected
        ):
            if shape is not None and "id" not in shape:
                shape = self._plot_shape = {
                    "id": store_selection(adata, shape),
                    **shape,
                }
            provenance = {"filters": filters, "shape": shape}
            return {**{k: v for k, v in provenance.items() if v}, **params}
        # e.g. set by code, or a plot selection whose shape was not captured
        payload = {"type": "mask", "mask": selected}
        return {"mask": store_selection(adata, payload), **params}

    def dspr_clear_selection_click_callback(self):
        self._plot_selection = self._plot_shape = None
        for control in self.selection_controls_dict.values():
            if isinstance(control, SelectMultiple):
                control.value = tuple()

            elif isinstance(control, IntRangeSlider | FloatRangeSlider):
                control.value = control.min, control.min

            else:
                raise RuntimeError(f"Unsupported control type {type(control)}")

    def dspr_apply_label_click_callback(self):
        create_new_key = self.selection_labeling_controls_dict[
            "create_new_key_checkbox"
        ].value
        if create_new_key:
            column = self.selection_labeling_controls_dict["new_medatadata_key"].value
            label = self.selection_labeling_controls_dict["new_label"].value
        else:
            column = self.selection_labeling_controls_dict[
                "existing_metadata_key"
            ].value
            label = self.selection_labeling_controls_dict["existing_label"].value

        if column == "":
            return

        self._run_on_selection("apply_label", column=column, label=label)

    def dspr_keep_selected_click_callback(self):
        self._run_on_selection("filter_rows")

    def dspr_drop_selected_click_callback(self):
        self._run_on_selection("filter_rows", drop=True)

    def _run_on_selection(self, step: str, **params):
        """Run ``step`` on the selection, if any; its errors show below the buttons."""
        from .step._processor_step_base import error_html

        try:
            if (selection := self._selection_params()) is not None:
                self.steps[step].run(**selection, **params)
        except Exception as e:  # e.g. a background job runs
            self.broker._log_exception(e)
            self.selection_status.value = error_html(e)

    def dplt_selected_points_change_callback(self, new_value: pd.Index | None):
        # a new plot selection replaces the previous one; the filters stay
        self._plot_selection = new_value
        self._plot_shape = (
            self.plotter.selection_shape if new_value is not None else None
        )
        self._update_selected_rows()

    def dset_total_rows_change_callback(self, metadata: pd.DataFrame):
        for column in self.selection_controls_dict.keys():
            self.update_column_selection_control(column)

    def dset_metadata_change_callback(self, *args, **kwargs):
        metadata = self.dataset._metadata
        df = metadata.select_dtypes(include=["bool", "category"])
        ctrl: Dropdown = self.selection_labeling_controls_dict["existing_metadata_key"]
        ctrl.options = [""] + df.columns.to_list()

        options = {
            "": None,
            **{c: c for c in text_or_category_columns(self.dataset._metadata)},
        }
        for control in self.all_controls_list:
            if not isinstance(control, Dropdown):
                continue
            description: str = control.description
            if description.lower().strip(" :.").startswith("group"):
                current_value = control.value
                control.options = options
                if current_value not in control.options:
                    control.value = None
                else:
                    control.value = current_value

        metadata = self.dataset._metadata
        options = {"": None, **{c: c for c in metadata.columns}}
        for control in self.all_controls_list:
            if not isinstance(control, Dropdown):
                continue
            description: str = control.description
            if description.lower().strip(" :.").endswith("axis"):
                current_value = control.value
                control.options = options
                if current_value not in control.options:
                    control.value = None
                else:
                    control.value = current_value

    def dset_anndata_layers_change_callback(self, *args, **kwargs):
        layer_options = {key: key for key in layer_keys(self.dataset.adata)}
        obsm_options = {key: key for key in self.dataset.adata.obsm.keys()}

        for control in self.all_controls_list:
            if not isinstance(control, Dropdown):
                continue
            description: str = control.description

            if description.lower().strip(" :.") == "layer":
                current_value = control.value
                control.options = layer_options
                if current_value not in control.options:
                    control.value = None
                else:
                    control.value = current_value

            if description.lower().strip(" :.") == "use rep":
                current_value = control.value
                control.options = {**layer_options, **obsm_options}
                if current_value not in control.options:
                    control.value = None
                else:
                    control.value = current_value

    def dset_data_dict_change_callback(self, *args, **kwargs):
        options = [
            *layer_keys(self.dataset.adata),
            *self.dataset.adata.obsm.keys(),
        ]
        options = {v: v for v in options}
        for control in self.all_controls_list:
            if not isinstance(control, Dropdown):
                continue
            description: str = control.description
            if description.lower().strip(" :.") == "use rep":
                current_value = control.value
                control.options = options
                if current_value is None and "X_pca" in control.options:
                    control.value = "X_pca"
                elif current_value not in control.options:
                    control.value = None
                else:
                    control.value = current_value
