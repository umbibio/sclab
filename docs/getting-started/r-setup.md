# R Setup (optional)

R is an **optional** dependency required only for:

- **Differential expression** — `pseudobulk_edger`, `pseudobulk_limma`
- **Imputation** — ALRA

All other SCLab features work without R.

---

## Step 1 — Install R

Download and install R (≥ 4.2) from [https://cran.r-project.org](https://cran.r-project.org).

Verify the installation:

```bash
R --version
```

---

## Step 2 — Install R packages

Open an R console and run:

```r
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")

# For differential expression
BiocManager::install(c("edgeR", "limma", "MAST", "SingleCellExperiment"))

# For imputation (ALRA)
BiocManager::install("ALRA")
```

---

## Step 3 — Install Python R bridge

```bash
pip install rpy2 anndata2ri
```

Or with uv:

```bash
uv add rpy2 anndata2ri
```

---

## Step 4 — Verify

```python
from sclab.tools.differential_expression import edger_is_available, limma_is_available

print("edgeR available:", edger_is_available())
print("limma available:", limma_is_available())
```

---

## Troubleshooting

**`rpy2` cannot find R**

Make sure R is on your `PATH`. You can set the `R_HOME` environment variable explicitly:

```bash
export R_HOME=$(R RHOME)
```

**Package not found in R**

Re-run the BiocManager install commands inside the same R version that `rpy2` is pointing to.
