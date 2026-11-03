"""One redraw per step: coalescing renders must not change what ends up drawn."""

import json
from collections import Counter

import itables
import plotly.utils
import pytest

from sclab import SCLabDashboard
from sclab.dataset.plotter import Plotter

from .test_utils import run_step, simple_loop_adata

STEPS = ["QC", "Preprocess", "PCA", "Guided Pseudotime", "Neighbors", "UMAP", "Cluster"]


def figure_state(dash: SCLabDashboard) -> dict:
    fig = json.loads(
        json.dumps(dash.plotter.g.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
    )
    for trace in fig["data"]:
        trace.pop("uid", None)  # random per trace
    return fig


def run_workflow(coalescing: bool) -> tuple[dict, dict]:
    counts: dict[str, Counter] = {}
    states: dict[str, dict] = {}
    current = Counter()
    make_new_figure, show = Plotter.make_new_figure, itables.show

    def counting_make_new_figure(self, *args, **kwargs):
        current["figure"] += 1
        return make_new_figure(self, *args, **kwargs)

    def counting_show(df, *args, table_id="", **kwargs):
        current["obs" if "_obs_" in table_id else "var"] += 1
        return show(df, *args, table_id=table_id, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Plotter, "make_new_figure", counting_make_new_figure)
        mp.setattr(itables, "show", counting_show)
        dash = SCLabDashboard(simple_loop_adata(n_obs=1000))
        dash.broker.coalescing = coalescing

        for step in STEPS:
            if step == "Guided Pseudotime":
                gp = dash.processor.steps[step]
                gp._automatic_periodic_path_drawing()
                states["drawn path"] = figure_state(dash)
                gp.variable_controls["roughness"].value = 1
            current.clear()
            run_step(dash, step)
            counts[step] = Counter(current)
            states[step] = figure_state(dash)

    assert not dash.broker.exceptions_log
    return counts, states


@pytest.fixture(scope="module")
def coalesced() -> tuple[dict, dict]:
    return run_workflow(coalescing=True)


@pytest.fixture(scope="module")
def immediate() -> tuple[dict, dict]:
    return run_workflow(coalescing=False)


@pytest.mark.parametrize("step", ["QC", "Preprocess", "PCA", "UMAP", "Cluster"])
def test_one_figure_render_per_step(coalesced, step):
    counts, _ = coalesced
    assert counts[step]["figure"] == 1
    assert counts[step]["obs"] <= 1
    assert counts[step]["var"] <= 1


@pytest.mark.parametrize("step", ["drawn path", *STEPS])
def test_final_figure_matches_immediate_rendering(coalesced, immediate, step):
    assert coalesced[1][step] == immediate[1][step]


def test_fitted_pseudotime_curve_survives_the_deferred_redraw(coalesced):
    _, states = coalesced
    names = [trace.get("name") for trace in states["Guided Pseudotime"]["data"]]
    assert "pseudotime" in names
