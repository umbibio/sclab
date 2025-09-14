from ipywidgets import Dropdown, IntText

from sclab.dataset.processor import Processor
from sclab.dataset.processor.step import ProcessorStepBase


class Integration(ProcessorStepBase):
    parent: Processor
    name: str = "integration"
    description: str = "Integration"

    def __init__(self, parent: Processor) -> None:
        cat_metadata = parent.dataset._metadata.select_dtypes(
            include=["object", "category"]
        )
        cat_options = {"": None, **{c: c for c in cat_metadata.columns}}

        variable_controls = dict(
            use_rep=Dropdown(
                options=tuple(parent.dataset.adata.obsm.keys()),
                value=None,
                description="Use rep.",
            ),
            group_by=Dropdown(
                options=cat_options,
                value="batch" if "batch" in cat_options else None,
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

        variable_controls["group_by"].observe(update_reference_batch, names="value")

        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )

    def function(
        self,
        use_rep: str,
        group_by: str,
        flavor: str,
        reference_batch: str | None,
        max_iters: int,
    ):
        adata = self.parent.dataset.adata

        key_added = f"{use_rep}_{flavor}"
        kvargs = {
            "adata": adata,
            "key": group_by,
            "basis": use_rep,
            "adjusted_basis": key_added,
        }

        self.broker.std_output.clear_output(wait=False)
        with self.broker.std_output:
            match flavor:
                case "cca":
                    from sclab.preprocess import cca_integrate

                    cca_integrate(
                        **kvargs,
                        reference_batch=reference_batch,
                    )

                case "harmony":
                    from sclab.preprocess import harmony_integrate

                    harmony_integrate(
                        **kvargs,
                        reference_batch=reference_batch,
                        max_iter_harmony=max_iters,
                    )

                case "scanorama":
                    from scanpy.external.pp import scanorama_integrate

                    scanorama_integrate(**kvargs)
                case _:
                    raise ValueError(f"Unknown flavor: {flavor}")

        self.broker.publish(
            "dset_data_dict_change",
            self.parent.dataset.data_dict,
            key_added,
        )
