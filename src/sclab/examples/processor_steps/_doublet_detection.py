import functools
import inspect

from ipywidgets import Checkbox, Dropdown

from sclab._compat import requirement
from sclab.dataset.processor import Processor
from sclab.dataset.processor._codegen import ADATA, call
from sclab.dataset.processor.step import ProcessorStepBase
from sclab.tools.doublet_detection import (
    doubletdetection,
    scdblfinder,
    scrublet,
)
from sclab.tools.doublet_detection._doubletdetection import (
    doubletdetection_is_available,
)
from sclab.tools.doublet_detection._scdblfinder import (
    _pyscdblfinder_available,
    _r_available,
)
from sclab.tools.doublet_detection._scrublet import scrublet_is_available
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

# shown below the Run button, see ProcessorStepBase.tool
TOOLS = {
    "doubletdetection": "Python · doubletdetection.BoostClassifier",
    "scrublet": "Python · scrublet.Scrublet",
    ("scDblFinder", "r"): "R · scDblFinder::scDblFinder",
    ("scDblFinder", "python"): "Python · pyscdblfinder (scDblFinder port)",
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

    scDblFinder runs in R (``backend="r"``) or in Python (pyscdblfinder,
    ``backend="python"``). A Backend dropdown is shown only when both are
    installed; it defaults to R, as ``backend="auto"`` does.

    Requires at least one of the flavors' packages to be installed
    (e.g. ``pip install scrublet``).
    """

    parent: Processor
    name: str = "doublet_detection"
    description: str = "Doublet Detection"
    background = True  # but not scDblFinder in R: rpy2 runs on the main thread

    def __init__(self, parent: Processor) -> None:
        # checked once per dashboard: checking for R starts R, which is slow
        backends = {"R": _r_available(), "Python": _pyscdblfinder_available()}
        backends = [(label, label.lower()) for label, ok in backends.items() if ok]
        self.scdblfinder_backends = [value for _, value in backends]
        available = {
            "doubletdetection": doubletdetection_is_available(),
            "scDblFinder": bool(backends),
            "scrublet": scrublet_is_available(),
        }
        methods_available = [m for m, ok in available.items() if ok]
        if not methods_available:
            raise ImportError(
                f"Needs a doublet detection package, e.g. {requirement('scrublet')}"
            )

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
            backend=Dropdown(options=backends, description="Backend"),
            override_layer=self._override_checkbox,
            layer=self._layer_dropdown,
        )

        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )
        variable_controls["flavor"].observe(self._show_backend, names="value")
        self._show_backend()

    def _show_backend(self, *_):
        """The Backend dropdown: for scDblFinder, if both backends are installed."""
        shown = len(self.scdblfinder_backends) > 1
        shown = shown and self.variable_controls["flavor"].value == "scDblFinder"
        self.variable_controls["backend"].layout.display = None if shown else "none"

    def _default_backend(self) -> str | None:
        """What ``backend="auto"`` picks here: R if installed."""
        return next(iter(self.scdblfinder_backends), None)

    def tool(self, params: dict) -> str:
        flavor = params.get("flavor")
        if flavor == "scDblFinder":
            backend = params.get("backend") or self._default_backend()
            return TOOLS.get((flavor, backend), "")
        return TOOLS.get(flavor, "")

    def _on_override_change(self, change) -> None:
        override = change["new"]
        self._layer_dropdown.disabled = not override
        if not override:
            counts_layer = self.parent.dataset.counts_layer
            if counts_layer in self._layer_dropdown.options:
                self._layer_dropdown.value = counts_layer

    def dset_counts_layer_change_callback(self, old: str, new: str):
        self._on_override_change({"new": self._override_checkbox.value})

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
        if resolved.get("flavor") == "scDblFinder":
            resolved["backend"] = resolved.get("backend") or self._default_backend()
        else:  # only scDblFinder has backends
            resolved.pop("backend", None)
        return resolved

    def to_code(self, params: dict) -> str:
        function, extra = FLAVORS[params["flavor"]]
        backend = params.get("backend")
        if backend is not None and backend != self._default_backend():
            extra = {**extra, "backend": backend}  # else what backend="auto" picks
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
        backend: str | None = None,
    ):
        adata = self.parent.dataset.adata
        if not override_layer:
            layer = self.parent.dataset.counts_layer

        if flavor not in FLAVORS:
            raise ValueError(f"Unknown flavor: {flavor}")
        function, extra = FLAVORS[flavor]
        if function == "scdblfinder":  # explicit: "auto" would check for R again
            backend = backend or self._default_backend() or "python"
            extra = {**extra, "backend": backend}
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
        if function == "scdblfinder" and backend == "r":
            work(adata)
        else:
            yield from self.compute(work)

        self.broker.publish(
            "dset_metadata_change",
            self.parent.dataset.metadata,
            f"{flavor}_label",
        )
