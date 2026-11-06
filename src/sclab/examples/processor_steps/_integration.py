from ipywidgets import Dropdown, IntText

from sclab.dataset.processor import Processor
from sclab.dataset.processor._codegen import ADATA, call
from sclab.dataset.processor.step import ProcessorStepBase
from sclab.event._jobs import capture
from sclab.utils import text_or_category_columns


class Integration(ProcessorStepBase):
    """Batch integration step for multi-sample single-cell data.

    Corrects batch effects in a cell embedding by applying one of three
    integration algorithms: CCA (canonical correlation analysis), Harmony,
    or Scanorama. The corrected embedding is stored under a new
    ``obsm`` key named ``{use_rep}_{flavor}`` and selected in the plotter.

    Available flavors:

    - ``"cca"`` — canonical correlation analysis via
      :func:`~sclab.preprocess.cca_integrate`.
    - ``"harmony"`` — Harmony via :func:`~sclab.preprocess.harmony_integrate`.
    - ``"scanorama"`` — Scanorama via
      :func:`scanpy.external.pp.scanorama_integrate`.
    """

    parent: Processor
    name: str = "integration"
    description: str = "Integration"

    def __init__(self, parent: Processor) -> None:
        cat_options = {
            "": None,
            **{c: c for c in text_or_category_columns(parent.dataset._metadata)},
        }

        variable_controls = dict(
            use_rep=Dropdown(
                options=tuple(parent.dataset.adata.obsm.keys()),
                value=None,
                description="Use rep.",
            ),
            group_by=Dropdown(
                options=cat_options,
                # the dashboard's batch key ("batch" if there is such a column)
                value=parent.batch_key if parent.batch_key in cat_options else None,
                description="GroupBy",
            ),
            reference_batch=Dropdown(
                description="Reference Batch",
            ),
            flavor=Dropdown(
                options=["cca", "harmony", "scanorama"],
                value="cca",
                description="Flavor",
            ),
            max_iters=IntText(
                value=20,
                description="Max iters",
            ),
        )

        def update_reference_batch(*args, **kwargs):
            group_by = variable_controls["group_by"].value
            if (
                group_by is None
                or group_by not in self.parent.dataset.adata.obs.columns
            ):
                variable_controls["reference_batch"].options = {"": None}
                return

            options = {
                "": None,
                **{
                    c: c
                    for c in self.parent.dataset.adata.obs[group_by]
                    .sort_values()
                    .unique()
                },
            }
            variable_controls["reference_batch"].options = options

        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )

        variable_controls["group_by"].observe(
            self.guarded(update_reference_batch), names="value"
        )
        update_reference_batch()

    def dspr_batch_key_change_callback(self, old: str | None, new: str | None):
        group_by = self.variable_controls["group_by"]
        if group_by.value == old and new in group_by.options.values():  # not chosen
            group_by.value = new

    def validate(self, params: dict) -> str | None:
        obs = self.parent.dataset.adata.obs
        group_by, flavor = params.get("group_by"), params.get("flavor")
        if group_by is None:
            return "Choose a 'GroupBy' column with at least 2 batches"
        if group_by not in obs:
            return f"No column {group_by!r} in the data"
        groups = obs[group_by].dropna().unique()
        if len(groups) < 2:
            return f"{group_by!r} has a single batch: integration needs at least 2"
        if flavor == "cca" and len(groups) != 2:
            return f"CCA integrates 2 batches, {group_by!r} has {len(groups)}"
        use_rep = params.get("use_rep")
        if use_rep is None and flavor != "cca":  # cca can use X
            return "Choose a 'Use rep.' representation (run PCA first)"
        if use_rep is not None and use_rep not in self.parent.dataset.adata.obsm:
            return f"No representation {use_rep!r} in the data"
        reference = params.get("reference_batch")
        if reference is not None:
            if flavor == "cca":
                return "CCA does not take a reference batch: clear it or use harmony"
            if reference not in groups:
                return f"Reference batch {reference!r} is not in {group_by!r}"
        return None

    @staticmethod
    def _kwargs(
        use_rep, group_by, flavor, reference_batch, max_iters, random_state
    ) -> dict:
        basis = use_rep or "X"
        kwargs = dict(key=group_by, basis=basis, adjusted_basis=f"{basis}_{flavor}")
        if flavor == "cca":
            kwargs.update(reference_batch=reference_batch, random_state=random_state)
        elif flavor == "harmony":
            kwargs.update(
                reference_batch=reference_batch,
                max_iter_harmony=max_iters,
                random_state=random_state,
            )
        return kwargs

    def to_code(self, params: dict) -> str:
        functions = {
            "cca": "sclab.preprocess.cca_integrate",
            "harmony": "sclab.preprocess.harmony_integrate",
            "scanorama": "sc.external.pp.scanorama_integrate",
        }
        kwargs = self._kwargs(
            params["use_rep"],
            params["group_by"],
            params["flavor"],
            params["reference_batch"],
            params["max_iters"],
            params["random_state"],
        )
        return call(functions[params["flavor"]], ADATA, **kwargs)

    def function(
        self,
        use_rep: str | None,
        group_by: str,
        flavor: str,
        reference_batch: str | None,
        max_iters: int,
        random_state: int = 0,
    ):
        adata = self.parent.dataset.adata
        kwargs = self._kwargs(
            use_rep, group_by, flavor, reference_batch, max_iters, random_state
        )

        self.broker.std_output.clear_output(wait=False)
        with capture(self.broker.std_output):
            match flavor:
                case "cca":
                    from sclab.preprocess import cca_integrate

                    cca_integrate(adata, **kwargs)

                case "harmony":
                    from sclab.preprocess import harmony_integrate

                    harmony_integrate(adata, **kwargs)

                case "scanorama":
                    from scanpy.external.pp import scanorama_integrate

                    scanorama_integrate(adata, **kwargs)
                case _:
                    raise ValueError(f"Unknown flavor: {flavor}")

        self.broker.publish(
            "dset_data_dict_change",
            self.parent.dataset.data_dict,
            kwargs["adjusted_basis"],
        )
