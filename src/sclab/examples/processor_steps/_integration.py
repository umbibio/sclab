from ipywidgets import Dropdown

from sclab.dataset.processor import Processor
from sclab.dataset.processor.step import ProcessorStepBase


class Integration(ProcessorStepBase):
    parent: Processor
    name: str = "integration"
    description: str = "Integration"

    def __init__(self, parent: Processor) -> None:
        try:
            from scanpy.external.pp import harmony_integrate  # noqa
        except ImportError:
            try:
                from scanpy.external.pp import scanorama_integrate  # noqa
            except ImportError:
                raise ImportError(
                    "Integration requires scanorama or harmony to be installed.\n"
                    "\nInstall with one of:\n"
                    "\npip install harmony"
                    "\npip install scanorama"
                    "\n"
                )

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
            flavor=Dropdown(
                options=["harmony", "scanorama"],
                value="harmony",
                description="Flavor",
            ),
        )

        super().__init__(
            parent=parent,
            fixed_params={},
            variable_controls=variable_controls,
        )

    def function(self, use_rep: str, group_by: str, flavor: str):
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
                case "harmony":
                    from scanpy.external.pp import harmony_integrate

                    harmony_integrate(**kvargs)

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
