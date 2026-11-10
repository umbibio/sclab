import html

from ipywidgets import HTML, HBox, ToggleButton, VBox

from ..._citations import active_citations
from ...utils import CodeActions
from .._dataset import SCLabDataset


class CitationsPanel(VBox):
    """History tab's Citations: the references of the methods run on the dataset.

    Shows :func:`sclab.active_citations` for the dataset's adata, with Copy
    (the text) and Download .bib buttons. Refreshed after every step run.
    """

    def __init__(self, dataset: SCLabDataset):
        self.dataset = dataset
        self.toggle = ToggleButton(description="Citations", icon="book")
        self.actions = CodeActions(
            filename="sclab_citations.bib",
            mime="application/x-bibtex",
            copy_label="Copy",
            download_label="Download .bib",
        )
        self.view = HTML(layout=dict(display="none"))
        self.toggle.observe(self._toggle, "value")
        super().__init__([HBox([self.toggle, self.actions]), self.view])
        dataset.broker.subscribe("dset_state_change", self.refresh)
        self.refresh()

    def refresh(self, *args) -> None:
        adata = self.dataset.adata
        entries = active_citations(adata, format="dict")
        text = active_citations(adata)
        self.actions.text = text
        self.actions.download_text = active_citations(adata, format="bibtex")
        items = "".join(
            f"<li><b>{html.escape(e['use'])}</b><br>{html.escape(t)}</li>"
            for e, t in zip(entries, text.split("\n\n"))
        )
        self.view.value = (
            f"<ol style='user-select:text'>{items}</ol>"
            if entries
            else "<i>No citations recorded yet: run a step.</i>"
        )

    def _toggle(self, change: dict) -> None:
        self.view.layout.display = "" if change["new"] else "none"
