# Third-party notices

sclab is licensed under the GNU General Public License v3.0 or later (see
[LICENSE](LICENSE)). It includes code copied or ported from the projects
below. All of their licenses are compatible with GPL-3.0.

| sclab files | Origin | Version | License |
| --- | --- | --- | --- |
| `src/sclab/scanpy/` (`_compat.py`, `_settings.py`, `logging.py`, `readwrite.py`, `plotting/`) | [scanpy](https://github.com/scverse/scanpy), copied and trimmed | 1.10.4 | BSD-3-Clause |
| `src/sclab/preprocess/_harmony_integrate.py` | scanpy, `scanpy.external.pp.harmony_integrate`, adapted | 1.10.x | BSD-3-Clause |
| `src/sclab/preprocess/_harmony.py` | [harmonypy](https://github.com/slowkow/harmonypy), `harmonypy/harmony.py`, modified (reference batches) | based on 0.0.10 | GPL-3.0-or-later |
| `src/sclab/tools/differential_expression/_edgepython_kernels.py` | [edgePython](https://github.com/pachterlab/edgePython), `edgepython/glm_kernels.py`, `ql_weights.py`, `dispersion_lowlevel.py`, `weighted_lowess.py` (numba ports of edgeR's and limma's C code), vectorised with NumPy | 0.5.1 | GPL-3.0-or-later |
| `src/sclab/tools/differential_expression/_edger_preprocessing.py` | edgePython, `edgepython/filtering.py` (`filter_by_expr`) and `normalization.py` (TMM `calc_norm_factors`), reduced | 0.5.1 | GPL-3.0-or-later |
| `src/sclab/tools/doublet_detection/_fastcluster.py` | [scDblFinder](https://github.com/plger/scDblFinder) (R/Bioconductor), `R/clustering.R` (`fastcluster`, `.prepSCE`, `.getDR`), ported to Python | 1.20.2 | GPL-3 |

## scanpy

The files under `src/sclab/scanpy/` and `src/sclab/preprocess/_harmony_integrate.py`
are distributed under the following license:

```
BSD 3-Clause License

Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## harmonypy

Copyright (C) 2018 Ilya Korsunsky, 2019 Kamil Slowikowski. GPL-3.0-or-later;
the notice is kept at the top of `src/sclab/preprocess/_harmony.py`.

Korsunsky I, Millard N, Fan J, et al. Fast, sensitive and accurate integration
of single-cell data with Harmony. Nat Methods 16, 1289-1296 (2019).
https://doi.org/10.1038/s41592-019-0619-0

## scDblFinder

Copyright the scDblFinder authors (Pierre-Luc Germain et al.). GPL-3.

Germain P-L, Lun A, Garcia Meixide C, Macnair W, Robinson MD. Doublet
identification in single-cell sequencing data using scDblFinder.
F1000Research 2021, 10:979. https://doi.org/10.12688/f1000research.73600.2

## edgePython

By Lior Pachter. GPL-3.0-or-later. A Python port of the Bioconductor package
edgeR (GPL >= 2; its C code `glm.c`, `compute_nbdev.c`, `ql_weights.c`,
`ql_glm.c`, `compute_adj_profile_ll`, `fmm_spline` from R's `splines.c`, and
the R functions `filterByExpr` and `calcNormFactors`) and of limma's `weighted_lowess.c` (GPL >= 2).

Pachter L. Differential analysis of genomics count data with edge*. bioRxiv
(2026). https://doi.org/10.64898/2026.02.16.706223

Robinson MD, McCarthy DJ, Smyth GK. edgeR: a Bioconductor package for
differential expression analysis of digital gene expression data.
Bioinformatics 26, 139-140 (2010). https://doi.org/10.1093/bioinformatics/btp616

McCarthy DJ, Chen Y, Smyth GK. Differential expression analysis of multifactor
RNA-Seq experiments with respect to biological variation. Nucleic Acids Res 40,
4288-4297 (2012). https://doi.org/10.1093/nar/gks042

Lun ATL, Chen Y, Smyth GK. It's DE-licious: a recipe for differential
expression analyses of RNA-seq experiments using quasi-likelihood methods in
edgeR. Methods Mol Biol 1418, 391-416 (2016).
https://doi.org/10.1007/978-1-4939-3578-9_19

Chen Y, Lun ATL, Smyth GK. From reads to genes to pathways: differential
expression analysis of RNA-Seq experiments using Rsubread and the edgeR
quasi-likelihood pipeline. F1000Research 5, 1438 (2016).
https://doi.org/10.12688/f1000research.8987.2

Robinson MD, Oshlack A. A scaling normalization method for differential
expression analysis of RNA-seq data. Genome Biol 11, R25 (2010).
https://doi.org/10.1186/gb-2010-11-3-r25

Chen Y, Chen L, Lun ATL, Baldoni PL, Smyth GK. edgeR v4: powerful differential
analysis of sequencing data with expanded functionality and improved support
for small counts and larger datasets. Nucleic Acids Res 53, gkaf018 (2025).
https://doi.org/10.1093/nar/gkaf018
