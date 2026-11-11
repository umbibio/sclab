# Reproducing an Analysis

Every step you run in the dashboard (including Keep/Drop selections and labels) is recorded in the **History** tab, with the parameters it actually ran with. There are two ways to reproduce an analysis from it.

## The generated script (recommended)

The History tab can show the analysis as a Python script, with **Copy code** and **Download .py** buttons. The script calls the same library functions as the steps, with every parameter explicit (including `backend="native"` for steps that ran with sclab's own port instead of scanpy). It runs without the dashboard, and you can read, review and edit it. The same script is available as `dashboard.processor.generate_code()`.

## Replaying a saved history

To apply the steps of one dashboard to another, for example the same data loaded again:

```python
# in the first session
dashboard.processor.save_step_history("history.json")  # or "history.pickle"

# in a new dashboard on the same data
dashboard.processor.load_step_history("history.json")
dashboard.processor.apply_step_history()
```

- `save_step_history(path)` writes the format given by the extension, `.json` or `.pickle`. Prefer JSON: loading a pickle runs code, so only load pickles you trust.
- `load_step_history(path)` only reads the file.
- `apply_step_history()` continues where the current dashboard left off. The steps already run in it must match the start of the loaded history (step and parameters), otherwise it stops with an error. A fresh dashboard on the same data replays everything.

Before applying anything, `apply_step_history()` checks that every remaining step can run here. If some cannot (a step the dashboard does not have, a missing package such as scanpy for a step recorded with scanpy), it raises a single `ValueError` listing all of them and applies nothing.

If a step then fails while replaying (or runs with other parameter values than recorded), the replay stops there with a `RuntimeError` naming the step number, the step and the error. Later steps are not run, because they would work on incomplete data. The steps before it stay applied, and the failed one is in the History tab marked as failed. Fix the cause (for example a missing column) and call `apply_step_history()` again: it continues from the failed step.

A Keep/Drop or Label selection is replayed by its criteria (ranges, categories, the drawn shape), and the number of cells it selected is recorded. If on replay it selects a different number of cells, for example because the data differ, the replay does not stop: it continues with the cells selected here and emits a `UserWarning` naming the step and both counts.

### Which backend runs

Steps with a choice of implementation (QC, Preprocess, PCA, Neighbors, Cluster, UMAP, Differential Expression with scanpy methods) record whether scanpy or sclab's native port ran. A replay uses the recorded one, so a history recorded with the native port replays natively even where scanpy is installed. A history recorded with scanpy is not replayed with the native port when scanpy is missing, because results may differ: run the step again interactively to use the native port.

### The copy in the dataset

The history is also stored in `adata.uns["sclab"]["history"]`, so a dataset saved by sclab shows its history when opened again. This copy is for display: long lists, such as the cells of a Keep/Drop selection, are replaced by a summary (length and hash) and cannot be replayed. `apply_step_history()` reports such steps in its `ValueError`. Use the generated script, or a history file saved with `save_step_history`, to reproduce an analysis that contains selections.
