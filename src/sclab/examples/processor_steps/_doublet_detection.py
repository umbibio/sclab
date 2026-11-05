import functools
import inspect

from ipywidgets import Checkbox, Dropdown

from sclab.dataset.processor import Processor
from sclab.dataset.processor._codegen import ADATA, call
from sclab.dataset.processor.step import ProcessorStepBase
from sclab.tools.doublet_detection import (
    doubletdetection,
    scdblfinder,
    scrublet,
)
from sclab.utils import layer_keys

# flavor -> (sclab.tools.doublet_detection function name, its extra kwargs)
FLAVORS = {
    "scDblFinder": ("scdblfinder", dict(clusters_col="leiden")),
    "doubletdetection": (
        "doubletdetection",
        dict(
            pseudocount=1,
            clustering_algorithm="leiden",
            clustering_kwargs=dict(resolution=5.0),
        ),
    ),
    "scrublet": ("scrublet", {}),
}


def default_seed(flavor: str) -> int | None:
    """The flavor's library default ``random_state`` (None for unknown flavors)."""
    if flavor not in FLAVORS:
        return None
    import sclab.tools.doublet_detection as tools

    function = getattr(tools, FLAVORS[flavor][0])
    return inspect.signature(function).parameters["random_state"].default


class DoubletDetection(ProcessorStepBase):
    """Doublet detection step.

    Identifies putative doublet cells using one of the available flavors
    (scrublet, doubletdetection, scDblFinder). Scores and binary labels
    (``"singlet"`` / ``"doublet"``) are added to ``adata.obs`` under keys
    ``{flavor}_score`` and ``{flavor}_label``.

    All flavors require raw counts. By default the dataset's counts layer
    is used; tick "Override layer" to pick a different layer.

    Requires at least one of the flavors' packages to be installed
    (e.g. ``pip install scrublet``).
    """

    parent: Processor
    name: str = "doublet_detection"
    description: str = "Doublet Detection"
    background = True  # but not scDblFinder: R (rpy2) runs on the main thread

    def __init__(self, parent: Processor) -> None:
        from sclab.tools.doublet_detection import (
            any_doublet_detection_available,
            get_available_doublet_detection_methods,
        )

        if not any_doublet_detection_available():
            raise ImportError

        methods_available = get_available_doublet_detection_methods()

        layers = tuple(layer_keys(parent.dataset.adata))
        counts_layer = parent.dataset.counts_layer
        self._layer_dropdown = Dropdown(
            options=layers,
            value=counts_layer if counts_layer in layers else None,
            description="Layer",
            disabled=True,
        )
        self._override_checkbox = Checkbox(value=False, description="Override layer")
        self._override_checkbox.observe(self._on_override_change, names="value")

        variable_controls = dict(
            flavor=Dropdown(
                options=methods_available,
                description="Flavor",
            ),
            override_layer=self._override_checkbox,
            layer=self._layer_dropdown,
        )

        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )

    def _on_override_change(self, change) -> None:
        override = change["new"]
        self._layer_dropdown.disabled = not override
        if not override:
            counts_layer = self.parent.dataset.counts_layer
            if counts_layer in self._layer_dropdown.options:
                self._layer_dropdown.value = counts_layer

    def validate(self, params: dict) -> str | None:
        layers = layer_keys(self.parent.dataset.adata)
        if params.get("override_layer"):
            if params.get("layer") not in layers:
                return "Choose a layer with raw counts"
        elif (counts_layer := self.parent.dataset.counts_layer) not in layers:
            return (
                f"No counts layer {counts_layer!r}: tick 'Override layer' to choose one"
            )
        return None

    def resolved_params(self, params: dict) -> dict:
        resolved = super().resolved_params(params)
        if not resolved.get("override_layer"):
            resolved["layer"] = self.parent.dataset.counts_layer
        if resolved.get("random_state") is None:  # recorded so the code shows it
            resolved["random_state"] = default_seed(resolved.get("flavor"))
        return resolved

    def to_code(self, params: dict) -> str:
        function, extra = FLAVORS[params["flavor"]]
        return call(
            f"sclab.tools.doublet_detection.{function}",
            ADATA,
            layer=params["layer"],
            key_added=params["flavor"],
            **extra,
            random_state=params["random_state"],
        )

    def function(
        self,
        flavor: str,
        override_layer: bool = False,
        layer: str | None = None,
        random_state: int | None = None,
    ):
        adata = self.parent.dataset.adata
        if not override_layer:
            layer = self.parent.dataset.counts_layer

        if flavor not in FLAVORS:
            raise ValueError(f"Unknown flavor: {flavor}")
        function, extra = FLAVORS[flavor]
        if random_state is None:
            random_state = default_seed(flavor)
        functions = dict(
            scdblfinder=scdblfinder,
            doubletdetection=doubletdetection,
            scrublet=scrublet,
        )

        detect = functools.partial(
            functions[function],
            layer=layer,
            key_added=flavor,
            **extra,
            random_state=random_state,
        )

        def work(adata):
            with self.broker.jobs.output():
                detect(adata=adata)

        self.broker.std_output.clear_output(wait=False)
        if function == "scdblfinder":
            work(adata)
        else:
            yield from self.compute(work)

        self.broker.publish(
            "dset_metadata_change",
            self.parent.dataset.metadata,
            f"{flavor}_label",
        )
