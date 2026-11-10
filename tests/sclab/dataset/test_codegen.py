"""Generated code round trips: steps run in the dashboard vs ``exec`` of the
script that ``generate_code`` writes for them, on a copy of the same input."""

import ast
import tempfile

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sclab import SCLabDashboard
from sclab.dataset.processor import _codegen
from sclab.dataset.processor._codegen import LOAD_PLACEHOLDER, history_to_code
from sclab.dataset.processor._selection_code import (
    MAX_INLINE_VERTICES,
    SELECTIONS_FILE,
)
from sclab.tools.doublet_detection import scrublet_is_available

from ..test_utils import run_step, simple_loop_adata
from .test_selection_provenance import box, keep, label, lasso

BASE = [("QC", {}), ("Preprocess", {}), ("PCA", {})]


def make_adata() -> AnnData:
    adata = simple_loop_adata(n_obs=300, n_vars=300)
    adata.obs["batch"] = pd.Categorical(
        np.where(np.arange(adata.n_obs) % 2 == 0, "b1", "b2")
    )
    adata.obs["grp"] = pd.cut(adata.obs["time"], bins=3, labels=["A", "B", "C"])
    # stand-in for a path drawn in the dashboard (Guided Pseudotime)
    adata.obs["drawn_path"] = adata.obs["time"].to_numpy() / (2 * np.pi)
    adata.uns["drawn_path"] = {"path_is_closed": True}
    return adata


def run_steps(dash: SCLabDashboard, steps: list[tuple[str, dict]]):
    for name, params in steps:
        controls = dash.processor.steps[name].variable_controls
        for key, value in params.items():  # as a user would, via the widgets
            controls[key].value = value
        run_step(dash, name)


def exec_code(code: str, adata: AnnData, saved: AnnData | None = None) -> AnnData:
    """Run ``code`` on ``adata``; ``saved`` is the adata the dashboard saved, from
    which the code reads the lasso polygons and masks that are not inlined."""
    namespace = {"input_adata": adata}
    code = code.replace(LOAD_PLACEHOLDER, "adata = input_adata")
    with tempfile.TemporaryDirectory() as tmp:
        if saved is not None:
            saved.write_h5ad(f"{tmp}/saved.h5ad")
            code = code.replace(f'"{SELECTIONS_FILE}"', f'"{tmp}/saved.h5ad"')
        exec(code, namespace)
    return namespace["adata"]


def dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def assert_same_adata(a: AnnData, b: AnnData):
    assert a.obs_names.equals(b.obs_names)
    assert a.var_names.equals(b.var_names)
    pd.testing.assert_frame_equal(a.obs, b.obs, check_exact=False, rtol=1e-6)
    pd.testing.assert_frame_equal(a.var, b.var, check_exact=False, rtol=1e-6)
    np.testing.assert_allclose(dense(a.X), dense(b.X), rtol=1e-6, atol=1e-8)
    for attr in ("layers", "obsm", "obsp"):
        ma, mb = getattr(a, attr), getattr(b, attr)
        assert set(ma.keys()) == set(mb.keys()), attr
        for key in ma.keys():
            np.testing.assert_allclose(
                dense(ma[key]), dense(mb[key]), rtol=1e-6, atol=1e-8, err_msg=key
            )
    # the script records citations too, the dashboard also its history
    assert set(a.uns) - {"sclab"} == set(b.uns) - {"sclab"}


def assert_readable(code: str):
    tree = ast.parse(code)
    for line in code.splitlines():
        assert len(line) <= 120, line
    for node in ast.walk(tree):
        if isinstance(node, ast.List | ast.Tuple | ast.Set | ast.Dict):
            n = len(node.keys if isinstance(node, ast.Dict) else node.elts)
            coords = isinstance(node, ast.List) and all(
                isinstance(e, ast.List) and len(e.elts) == 2 for e in node.elts
            )  # an inlined lasso polygon, one vertex per line
            assert n <= 10 or coords, ast.unparse(node)


def roundtrip(steps: list[tuple[str, dict]]):
    adata = make_adata()
    dash = SCLabDashboard(adata.copy())
    run_steps(dash, steps)
    code = dash.generate_code()
    assert_readable(code)
    result = exec_code(code, adata.copy())
    assert_same_adata(dash.dataset.adata, result)
    return code, dash.dataset.adata, result


# --------------------------------------------------------------------------
# round trips
# --------------------------------------------------------------------------
def test_full_pipeline_roundtrip():
    steps = [*BASE, ("Neighbors", {}), ("UMAP", {}), ("Cluster", {})]
    code, adata, _ = roundtrip(steps)
    assert {"X_pca", "X_2Dumap"} <= set(adata.obsm)
    assert "leiden" in adata.obs
    for step in ["QC", "Preprocess", "PCA", "Neighbors", "UMAP", "Cluster"]:
        assert f"# {step}\n" in code
    assert "random_state=0" in code


@pytest.mark.parametrize("n_vertices", [4, MAX_INLINE_VERTICES + 10])
def test_pipeline_with_selections_roundtrip(n_vertices):
    """Keep by category, Keep by box, Label by lasso, replayed on the input."""
    adata = make_adata()
    dash = SCLabDashboard(adata.copy())
    controls = dash.processor.selection_controls_dict
    run_steps(dash, [*BASE, ("Neighbors", {}), ("UMAP", {}), ("Cluster", {})])
    plotter = dash.plotter
    plotter.controls.data_key.value = "X_2Dumap"
    assert plotter.controls.n_dimensions.value == "2D"

    clusters = list(dash.dataset.adata.obs["leiden"].cat.categories)
    controls["leiden"].value = tuple(clusters[: max(1, len(clusters) - 1)])
    keep(dash)
    dash.processor.dspr_clear_selection_click_callback()

    xy = dash.dataset.adata.obsm["X_2Dumap"][:, :2]
    (x0, y0), (x1, y1) = np.quantile(xy, 0.05, axis=0), np.quantile(xy, 0.95, axis=0)
    box(dash, x=(x0, x1), y=(y0, y1))
    keep(dash)
    dash.processor.dspr_clear_selection_click_callback()

    xy = dash.dataset.adata.obsm["X_2Dumap"][:, :2]
    t = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    center, radius = np.median(xy, axis=0), np.ptp(xy, axis=0) / 3
    lasso(dash, (center + radius * np.c_[np.cos(t), np.sin(t)]).tolist())
    params = label(dash, "region", "core")
    assert 0 < params["n"] < dash.dataset.adata.n_obs

    code = dash.generate_code()
    assert_readable(code)
    assert code.count("select_box(") == 1 and code.count("select_polygon(") == 1
    assert {"import sclab.tools", "import pandas as pd"} <= set(code.splitlines())
    assert ("read_adata(" in code.replace(LOAD_PLACEHOLDER, "")) == (n_vertices > 20)
    result = exec_code(code, adata.copy(), saved=dash.dataset.adata)
    assert result.obs_names.equals(dash.dataset.adata.obs_names)
    assert list(result.obs.columns) == list(dash.dataset.adata.obs.columns)
    assert result.obs["region"].notna().sum() == params["n"]
    assert_same_adata(dash.dataset.adata, result)


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(
            [
                ("QC", {"min_cells": 3, "max_rank": 250}),
                (
                    "Preprocess",
                    {
                        "group_by": "batch",
                        "normalization_method": "Weighted library size",
                        "regress_n_genes": True,
                    },
                ),
            ],
            id="qc-preprocess-options",
        ),
        pytest.param(
            [
                ("QC", {}),
                ("Preprocess", {"normalization_method": "No normalization"}),
                (
                    "PCA",
                    {
                        "n_comps": 12,
                        "mask_var": "highly_variable",
                        "reference_batch": "b1",
                        "zero_center": True,
                    },
                ),
            ],
            id="pca-reference-batch",
        ),
        pytest.param(
            [
                *BASE,
                (
                    "Neighbors",
                    {
                        "use_rep": "X_pca",
                        "dims": (2, 8),
                        "metric": "cosine",
                        "group_by_batch": False,
                    },
                ),
                ("UMAP", {"n_components": 3, "zero_center": True}),
                ("Cluster", {"resolution": 0.5}),
            ],
            id="neighbors-trimmed-umap-3d",
        ),
        pytest.param(
            [*BASE, ("Integration", {"use_rep": "X_pca", "flavor": "cca"})],
            id="integration-cca",
        ),
        pytest.param(
            [
                *BASE,
                (
                    "Integration",
                    {"use_rep": "X_pca", "flavor": "harmony", "max_iters": 5},
                ),
            ],
            id="integration-harmony",
        ),
        pytest.param(
            [
                *BASE,
                ("Neighbors", {}),
                (
                    "Transfer Metadata",
                    {"group_key": "grp", "source_group": "A", "column": "time"},
                ),
            ],
            id="transfer-metadata",
        ),
        pytest.param(
            [
                *BASE,
                ("Differential Expression", {"groupby": "grp", "name": "x"}),
                (
                    "Differential Expression",
                    {"groupby": "grp", "groups": ("A",), "name": "x"},
                ),
            ],
            id="differential-expression",
        ),
        pytest.param(
            [
                *BASE,
                ("Guided Pseudotime", {"use_rep": "Y_array2", "roughness": 1.0}),
            ],
            id="guided-pseudotime",
        ),
    ],
)
def test_step_roundtrip(steps):
    roundtrip(steps)


@pytest.mark.skipif(not scrublet_is_available(), reason="scrublet not installed")
def test_doublet_detection_roundtrip():
    code, *_ = roundtrip([("Doublet Detection", {"flavor": "scrublet"})])
    assert "sclab.tools.doublet_detection.scrublet(" in code
    assert "layer=counts_layer" in code


def test_differential_expression_records_the_key_it_added():
    steps = [*BASE, ("Differential Expression", {"groupby": "grp", "name": "x"})]
    code, adata, result = roundtrip(steps + steps[-1:])
    assert 'key_added="differential_expression_x"' in code
    assert 'key_added="differential_expression_x_2"' in code
    for key in ("differential_expression_x", "differential_expression_x_2"):
        pd.testing.assert_frame_equal(adata.uns[key]["table"], result.uns[key]["table"])
    assert "sclab.tools.de(" in code and "import sclab.tools" in code


def test_guided_pseudotime_flags_the_drawn_path():
    steps = [*BASE, ("Guided Pseudotime", {"use_rep": "Y_array2"})]
    code, *_ = roundtrip(steps)
    assert 'adata.obs["drawn_path"]' in code
    assert "periodic=True" in code


def test_load_line_from_path(tmp_path):
    path = tmp_path / "data.h5ad"
    make_adata().write_h5ad(path)
    dash = SCLabDashboard(str(path))
    run_steps(dash, BASE)
    code = dash.generate_code()
    assert "adata = sclab._io.read_adata(" in code
    assert f'"{path}"' in code
    assert "import sclab._io" in code
    namespace = {}
    exec(code, namespace)
    assert_same_adata(dash.dataset.adata, namespace["adata"])


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def test_skips_failed_and_plot_only_steps():
    dash = SCLabDashboard(make_adata())
    run_steps(dash, BASE)
    dash.processor.steps["Integration"].run(use_rep="X_pca", flavor="bogus")
    assert dash.processor.step_history[-1]["status"] == "failed"
    dash.processor.steps["Gene Expression"].run(gene_input="var_0001", layer="counts")
    code = dash.generate_code()
    assert code.count("# PCA\n") == 1
    assert "Integration" not in code
    assert "Gene Expression" not in code


def test_header_imports_versions_and_variables():
    dash = SCLabDashboard(make_adata())
    run_steps(dash, BASE)
    code = dash.generate_code()
    lines = code.splitlines()
    assert lines[1].startswith("# sclab ")
    assert "import sclab.preprocess" in lines
    assert "import scanpy as sc" not in lines  # no sc.* call
    assert LOAD_PLACEHOLDER in lines
    assert 'counts_layer = "counts"' in lines
    assert 'batch_key = "batch"' in lines
    assert "counts_layer=counts_layer," in code
    assert dash.generate_code(header=False).lstrip().startswith("# QC\n")


def test_selection_steps_use_the_emitters_and_header_imports():
    dash = SCLabDashboard(make_adata())
    dash.dataset.selected_rows = dash.dataset.adata.obs_names[:100]
    dash.processor.selection_controls_dict["batch"].value = ("b1",)
    label(dash, "sel", "yes")
    keep(dash)
    code = dash.generate_code()
    assert_readable(code)
    assert '# apply_label\nmask = adata.obs["batch"].isin(["b1"])' in code
    assert "# filter_rows\n" in code and "adata = adata[mask].copy()" in code
    assert "placeholder" not in code and "not available yet" not in code
    lines = code.splitlines()
    assert "import pandas as pd" in lines
    assert "import numpy as np" not in lines  # only an empty label uses np.nan
    entry = {"step": "apply_label", "status": "success", "params": {"mask": "sel1"}}
    entry["params"].update(column="c", label="")
    assert "import numpy as np" in history_to_code([entry]).splitlines()


def test_legacy_selection_entries_have_no_code():
    dash = SCLabDashboard(make_adata())
    keep_index = dash.dataset.adata.obs_names[:100]
    dash.processor.steps["apply_label"].run(index=keep_index, column="sel", label="y")
    dash.processor.steps["filter_rows"].run(index=keep_index)
    code = dash.generate_code()
    assert "# apply_label: cells recorded as a list of obs names (pre-0.5)" in code
    assert "# filter_rows: cells recorded as a list of obs names (pre-0.5)" in code


def test_registered_codegen_overrides_and_unknown_steps(monkeypatch):
    dash = SCLabDashboard(make_adata())
    dash.processor.add_step_func("custom", lambda: None)
    dash.processor.steps["custom"].run()
    dash.processor.steps["filter_rows"].run(index=dash.dataset.adata.obs_names[:10])
    assert "# custom: no code available for this step" in dash.generate_code()

    def filter_rows_code(params):
        return f"adata = adata[{len(params['index'])}:]"

    monkeypatch.setitem(_codegen._CODEGEN, "filter_rows", filter_rows_code)
    monkeypatch.setitem(_codegen._CODEGEN, "custom", lambda params: None)
    code = dash.generate_code()
    assert "# filter_rows\nadata = adata[10:]" in code
    assert "custom" not in code


def test_history_to_code_without_steps_or_header():
    history = [{"step": "QC", "params": {}, "status": "success"}]
    assert history_to_code(history, header=False) == (
        "# QC: no code available for this step\n"
    )


def test_history_tab_code_widgets_refresh():
    dash = SCLabDashboard(make_adata())
    processor = dash.processor
    assert processor.code_toggle in processor.history_panel.children[1].children
    assert processor.code_view.layout.display == "none"
    processor.code_toggle.value = True
    assert processor.code_view.layout.display == ""

    with dash.broker.batch():  # coalesced: one render after the batch
        run_steps(dash, BASE[:2])
        assert "sclab.preprocess.qc" not in processor.code_actions.text
    assert processor.code_actions.text == dash.generate_code()
    assert "sclab.preprocess.preprocess(" in processor.code_view.value
