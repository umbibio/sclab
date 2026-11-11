# Third-party notices

sclab is licensed under the GNU General Public License v3.0 or later (see
[LICENSE](LICENSE)). It includes code copied or ported from the projects
below. All of their licenses are compatible with GPL-3.0.

| sclab files | Origin | Version | License |
| --- | --- | --- | --- |
| `src/sclab/scanpy/` (`_compat.py`, `_settings.py`, `logging.py`, `readwrite.py`, `plotting/`) | [scanpy](https://github.com/scverse/scanpy), copied and trimmed | 1.10.4 | BSD-3-Clause |
| `src/sclab/preprocess/_harmony_integrate.py` | scanpy, `scanpy.external.pp.harmony_integrate`, adapted | 1.10.x | BSD-3-Clause |
| `src/sclab/preprocess/_harmony.py` | [harmonypy](https://github.com/slowkow/harmonypy), `harmonypy/harmony.py`, modified (reference batches) | based on 0.0.10 | GPL-3.0-or-later |

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
