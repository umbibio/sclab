import json
import logging
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from ipywidgets.widgets import (
    HTML,
    Accordion,
    Button,
    Checkbox,
    Combobox,
    Dropdown,
    FloatRangeSlider,
    HBox,
    IntRangeSlider,
    Output,
    SelectMultiple,
    Tab,
    Text,
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

from ...event import EventBroker, EventClient
from .._dataset import SCLabDataset
from ..plotter import Plotter

logger = logging.getLogger(__name__)


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

    events = [
        "dspr_selection_values_change",
        "dspr_clear_selection_click",
        "dspr_keep_selected_click",
        "dspr_drop_selected_click",
        "dspr_apply_label_click",
    ]

    def __init__(
        self,
        dataset: SCLabDataset,
        plotter: Plotter,
        *,
        batch_key: str | None = None,
    ):
        self.dataset = dataset
        self.plotter = plotter
        self.broker = self.dataset.broker
        self.selection_controls_list = []
        self.selection_controls_dict = {}
        self.selection_controls_container = VBox(layout=dict(width="100%"))
        self.selection_labeling_controls_dict = {}
        self.selection_buttons_dict = {}
        self.all_controls_list = []
        self.step_history = []

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
        self.add_step_func("filter_rows", self.filter_rows, use_run_button=False)
        self.add_step_func("apply_label", self.apply_label, use_run_button=False)

        if "batch" in self.dataset.adata.obs.columns and not batch_key:
            batch_key = "batch"

        if batch_key:
            batch_values = (
                self.dataset.adata.obs[batch_key]
                .sort_values()
                .astype(str)
                .unique()
                .tolist()
            )

        else:
            batch_values = None

        self.batch_key = batch_key
        self.batch_values = batch_values

        super().__init__(self.broker)
        self.broker.subscribe("dset_metadata_change", self._make_selection_controls)

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
                    assert issubclass(
                        step, ProcessorStepBase
                    ), f"{step} must be a subclass of {ProcessorStepBase}"
                    steps_list[i] = step(self)

        steps: dict[str, list[_ProcessorStep]]
        steps_list: list[_ProcessorStep]

        # we make sure the steps have not been previously added
        for step_group_name, steps_list in steps.items():
            for step in steps_list:
                assert (
                    step.description not in self.steps
                ), f"Step {step.description} already exists"

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
        assert (
            step.description not in self.steps
        ), f"Step {step.description} already exists"
        self.steps[step.description] = step

        if accordion is not None:
            self.append_to_accordion(accordion, step.controls, step.description)

    def append_to_accordion(self, accordion: Accordion, panel: VBox, title: str):
        children = list(accordion.children)
        children.append(panel)
        accordion.children = tuple(children)
        accordion.set_title(len(children) - 1, title)

    def _make_selection_controls(self, *args, **kwargs):
        for column in self.dataset.metadata.columns:
            if column in self.selection_controls_dict:
                self.update_column_selection_control(column)
            else:
                self.add_column_selection_control(column)

        for column in self.selection_controls_dict.keys():
            if column not in self.dataset.metadata.columns:
                self.remove_column_selection_control(column)

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

        self.selection_labeling_controls_dict[
            "create_new_key_checkbox"
        ] = create_new_key_checkbox
        self.selection_labeling_controls_dict[
            "existing_metadata_key"
        ] = existing_metadata_key
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

        old_value = self.visible_category_dropdown.value
        options = [c.description for c in categorical_controls[1:]]
        self.visible_category_dropdown.options = options
        if old_value in options:
            self.visible_category_dropdown.value = old_value

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
            control.options = series.cat.categories
            control.value = tuple()

        elif is_bool_dtype(dtype):
            control.value = tuple()

        elif is_integer_dtype(dtype):
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

            control.value = (control.min, control.min)

        elif is_float_dtype(dtype):
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
            control.value = (control.min, control.min)

        else:
            raise TypeError(f"Unsupported dtype: {series.dtype}")

    def remove_column_selection_control(self, column: str):
        if column not in self.selection_controls_dict:
            return

        control = self.selection_controls_dict.pop(column)
        self.all_controls_list.remove(control)
        self._make_selection_controls_container()

    def append_to_step_history(self, step_description: str, params: dict):
        params_ = {}
        for key, value in params.items():
            if isinstance(value, pd.Index):
                value = value.to_list()
            params_[key] = value

        self.step_history.append(
            {
                "step_description": step_description,
                "params": params_,
            }
        )

    def save_step_history(
        self, path: str | Path, format: Literal["pickle", "json"] | None = None
    ):
        path = Path(path)
        if format is None:
            format = path.suffix[1:]

        if format == "pickle":
            import pickle

            with open(path, "wb") as f:
                pickle.dump(self.step_history, f)
        elif format == "json":
            import json

            with open(path, "w") as f:
                json.dump(self.step_history, f, indent=4)
        else:
            raise ValueError(f"Unsupported format {format}")

    def load_step_history(self, path: str):
        path = Path(path)
        format = path.suffix[1:]

        if format == "pickle":
            import pickle

            with open(path, "rb") as f:
                self._loaded_step_history: list = pickle.load(f)
        elif format == "json":
            import json

            with open(path, "r") as f:
                self._loaded_step_history: list = json.load(f)
        else:
            raise ValueError(f"Unsupported format {format}")

    def apply_step_history(self):
        assert self._loaded_step_history is not None, "No step history loaded"

        current_step_history = self.step_history
        new_step_history = self._loaded_step_history
        n = len(current_step_history)
        N = len(new_step_history)
        assert N >= n, "Step history mismatch"

        for i, (present_step, incoming_step) in enumerate(
            zip(current_step_history, new_step_history)
        ):
            assert present_step == incoming_step, "Step history mismatch"
            step_description = present_step["step_description"]
            logger.info(f"Step {i + 1: 2d}/{N} already applied: {step_description}")

        new_steps_to_apply = new_step_history[n:]
        for i, step in enumerate(new_steps_to_apply):
            step_description = step["step_description"]
            params = step["params"]
            logger.info(f"Applying step {n + i + 1: 2d}/{N}: {step_description}")
            self.steps[step_description].run(**params)

    def print_step_history(self, with_hash: bool = False):
        for i, step in enumerate(self.step_history):
            desc = step["step_description"]
            params = step["params"]
            p = []
            for k, v in params.items():
                if isinstance(v, list):
                    v = f"list({len(v)})"
                if v is None:
                    v = "None"
                if not isinstance(v, int | float | str):
                    v = type(v)
                p.append(f"{k}={v}")
            if with_hash:
                history_hash = self._get_step_history_hash(self.step_history[: i + 1])
                history_hash = history_hash[:8] + " ..."
            else:
                history_hash = ""
            print(f"({i + 1: 3d}) {history_hash} {desc}({', '.join(p)})")

    def _get_step_history_hash(self, history: list[dict]):
        history_json = json.dumps(history)
        history_json_hash = sha256(history_json.encode()).hexdigest()
        return history_json_hash

    @property
    def step_history_hash(self):
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

    def filter_rows(self, index: pd.Index | list):
        if isinstance(index, list):
            index = pd.Index(index)
        self.dataset.filter_rows(index)

    def apply_label(self, index: pd.Index | list, column: str, label: str):
        if isinstance(index, list):
            index = pd.Index(index)
        self.dataset.apply_label(index, column, label)

    def make_selectbatch_drowpdown(self, description="Select Batch"):
        control = dict()
        if self.batch_key:
            control_key = description.lower().replace(" ", "_")
            control[control_key] = Dropdown(
                options={"": None, **{v: v for v in self.batch_values}},
                value=None,
                description=description,
            )
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
        row_names = self.dataset.metadata.index
        selected_rows = pd.Index([])

        # we will check if we intended to make a selection
        selection_attempted = False
        for column, value in self.selection_values.items():
            series = self.dataset.metadata[column]
            subset = pd.Index([])

            if is_numeric_dtype(series) and not is_bool_dtype(series):
                # this must be a range slider with value = tuple(min, max)
                min_value, max_value = value
                if max_value > min_value:
                    subset = row_names[(series >= min_value) & (series <= max_value)]
                    selection_attempted = True

            elif value:
                # this must be a select multiple with value = tuple(selected_values)
                subset = row_names[series.isin(value)]
                selection_attempted = True

            if selected_rows.empty:
                # we found the first non-empty subset, initialize selected_rows
                selected_rows = subset

            elif not subset.empty:
                # we found another non-empty subset, intersect with previously selected_rows
                selected_rows = selected_rows.intersection(subset)

                if selected_rows.empty:
                    # control values don't intersect, we will return an empty selection
                    break

        # if no selection was attempted, we will set None
        # if a selection was attempted but no rows were selected, we will set an empty index
        selected_rows = selected_rows if selection_attempted else None
        self.dataset.selected_rows = selected_rows

    def dspr_clear_selection_click_callback(self):
        for control in self.selection_controls_dict.values():
            if isinstance(control, SelectMultiple):
                control.value = tuple()

            elif isinstance(control, IntRangeSlider | FloatRangeSlider):
                control.value = control.min, control.min

            else:
                raise RuntimeError(f"Unsupported control type {type(control)}")

    def dspr_apply_label_click_callback(self):
        if self.dataset.selected_rows is None:
            return

        rows_to_label = self.dataset.selected_rows

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

        self.steps["apply_label"].run(index=rows_to_label, column=column, label=label)

    def dspr_keep_selected_click_callback(self):
        if self.dataset.selected_rows is not None:
            rows_to_keep = self.dataset.selected_rows
            self.steps["filter_rows"].run(index=rows_to_keep)

    def dspr_drop_selected_click_callback(self):
        if self.dataset.selected_rows is not None:
            rows_to_keep = self.dataset.row_names.difference(self.dataset.selected_rows)
            self.steps["filter_rows"].run(index=rows_to_keep)

    def dplt_selected_points_change_callback(self, new_value: pd.Index):
        for column, control in self.selection_controls_dict.items():
            if isinstance(control, SelectMultiple):
                control.value = tuple()

            elif isinstance(control, IntRangeSlider | FloatRangeSlider):
                control.value = control.min, control.min

            else:
                raise RuntimeError(
                    f"Unsupported control type {type(control)} for column {column}"
                )

    def dset_total_rows_change_callback(self, metadata: pd.DataFrame):
        for column in self.selection_controls_dict.keys():
            self.update_column_selection_control(column)

    def dset_metadata_change_callback(self, *args, **kwargs):
        metadata = self.dataset._metadata
        df = metadata.select_dtypes(include=["bool", "category"])
        ctrl: Dropdown = self.selection_labeling_controls_dict["existing_metadata_key"]
        ctrl.options = [""] + df.columns.to_list()

        metadata = self.dataset._metadata.select_dtypes(include=["object", "category"])
        options = {"": None, **{c: c for c in metadata.columns}}
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

    def dset_anndata_layers_change_callback(self, layers):
        options = {layer: layer for layer in layers}
        for control in self.all_controls_list:
            if not isinstance(control, Dropdown):
                continue
            description: str = control.description
            if description.lower().strip(" :.") == "layer":
                current_value = control.value
                control.options = options
                if current_value not in control.options:
                    control.value = None
                else:
                    control.value = current_value

    def dset_data_dict_change_callback(self, *args, **kwargs):
        options = {v: v for v in self.dataset.adata.obsm.keys()}
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

    def dset_total_vars_change_callback(self, *args, **kwargs):
        options = {v: v for v in self.dataset.adata.var_names}
        for control in self.all_controls_list:
            if not isinstance(control, Dropdown | Combobox):
                continue
            description: str = control.description
            if description.lower().strip(" :.") == "gene":
                current_value = control.value
                control.options = options
                if current_value not in control.options:
                    control.value = None
                else:
                    control.value = current_value
