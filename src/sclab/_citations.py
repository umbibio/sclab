"""References for the methods and tools sclab runs, and a record of the ones used.

Running a library method records its reference keys (see :func:`cites` and
:func:`record_citations`) in ``adata.uns["sclab"]["citations"]``, an ordered
list of unique keys saved with the dataset, and in a session-wide list (also
for calls without an AnnData). :func:`active_citations` renders them;
:func:`citation` renders one entry, :func:`known_citations` all of them.

Every entry was checked against its DOI record (Crossref / DataCite) or, where
there is none, the tool's own citation instructions.
"""

import functools
import threading
import warnings

UNS_KEY = "sclab"  # as in sclab.dataset.processor._history (circular import)
CITATIONS_KEY = "citations"
FORMATS = ("text", "bibtex", "ris", "dict")

# key -> fields. authors: (family, given) pairs. kind: article, preprint or
# software. pages: page range, article number or None.
REFERENCES: dict[str, dict] = {
    "sclab": dict(
        use="SCLab",
        kind="software",
        authors=[("Arriojas", "Argenis")],
        title="SCLab: Interactive Single-Cell Analysis Toolkit",
        journal="GitHub repository",
        year=2025,
        url="https://github.com/umbibio/sclab",
        # shortcut: placeholder (no paper or archived DOI yet), replace when published
    ),
    "anndata": dict(
        use="AnnData data structure",
        kind="article",
        authors=[
            ("Virshup", "Isaac"),
            ("Rybakov", "Sergei"),
            ("Theis", "Fabian J."),
            ("Angerer", "Philipp"),
            ("Wolf", "F. Alexander"),
        ],
        title="anndata: Access and store annotated data matrices",
        journal="Journal of Open Source Software",
        year=2024,
        volume="9",
        issue="101",
        pages="4371",
        doi="10.21105/joss.04371",
    ),
    "scanpy": dict(
        use="Scanpy",
        kind="article",
        authors=[
            ("Wolf", "F. Alexander"),
            ("Angerer", "Philipp"),
            ("Theis", "Fabian J."),
        ],
        title="SCANPY: large-scale single-cell gene expression data analysis",
        journal="Genome Biology",
        year=2018,
        volume="19",
        pages="15",
        doi="10.1186/s13059-017-1382-0",
    ),
    "scikit-learn": dict(
        use="scikit-learn",
        kind="article",
        authors=[
            ("Pedregosa", "Fabian"),
            ("Varoquaux", "Gaël"),
            ("Gramfort", "Alexandre"),
            ("Michel", "Vincent"),
            ("Thirion", "Bertrand"),
            ("Grisel", "Olivier"),
            ("Blondel", "Mathieu"),
            ("Prettenhofer", "Peter"),
            ("Weiss", "Ron"),
            ("Dubourg", "Vincent"),
            ("Vanderplas", "Jake"),
            ("Passos", "Alexandre"),
            ("Cournapeau", "David"),
            ("Brucher", "Matthieu"),
            ("Perrot", "Matthieu"),
            ("Duchesnay", "Édouard"),
        ],
        title="Scikit-learn: Machine Learning in Python",
        journal="Journal of Machine Learning Research",
        year=2011,
        volume="12",
        issue="85",
        pages="2825-2830",
        url="https://jmlr.org/papers/v12/pedregosa11a.html",
    ),
    "umap": dict(
        use="UMAP algorithm",
        kind="preprint",
        authors=[("McInnes", "Leland"), ("Healy", "John"), ("Melville", "James")],
        title="UMAP: Uniform Manifold Approximation and Projection for "
        "Dimension Reduction",
        journal="arXiv",
        year=2018,
        pages="1802.03426",
        doi="10.48550/arXiv.1802.03426",
    ),
    "umap-learn": dict(
        use="UMAP software (umap-learn)",
        kind="article",
        authors=[
            ("McInnes", "Leland"),
            ("Healy", "John"),
            ("Saul", "Nathaniel"),
            ("Großberger", "Lukas"),
        ],
        title="UMAP: Uniform Manifold Approximation and Projection",
        journal="Journal of Open Source Software",
        year=2018,
        volume="3",
        issue="29",
        pages="861",
        doi="10.21105/joss.00861",
    ),
    "leiden": dict(
        use="Leiden clustering",
        kind="article",
        authors=[("Traag", "V. A."), ("Waltman", "L."), ("van Eck", "N. J.")],
        title="From Louvain to Leiden: guaranteeing well-connected communities",
        journal="Scientific Reports",
        year=2019,
        volume="9",
        pages="5233",
        doi="10.1038/s41598-019-41695-z",
    ),
    "igraph": dict(
        use="igraph (Leiden implementation)",
        kind="article",
        authors=[("Csardi", "Gabor"), ("Nepusz", "Tamas")],
        title="The igraph software package for complex network research",
        journal="InterJournal, Complex Systems",
        year=2006,
        pages="1695",
        url="https://igraph.org",
    ),
    "satija2015": dict(
        use="Highly variable genes (Seurat flavor)",
        kind="article",
        authors=[
            ("Satija", "Rahul"),
            ("Farrell", "Jeffrey A."),
            ("Gennert", "David"),
            ("Schier", "Alexander F."),
            ("Regev", "Aviv"),
        ],
        title="Spatial reconstruction of single-cell gene expression data",
        journal="Nature Biotechnology",
        year=2015,
        volume="33",
        issue="5",
        pages="495-502",
        doi="10.1038/nbt.3192",
    ),
    "stuart2019": dict(
        use="Highly variable genes (Seurat v3 flavor); CCA integration",
        kind="article",
        authors=[
            ("Stuart", "Tim"),
            ("Butler", "Andrew"),
            ("Hoffman", "Paul"),
            ("Hafemeister", "Christoph"),
            ("Papalexi", "Efthymia"),
            ("Mauck", "William M."),
            ("Hao", "Yuhan"),
            ("Stoeckius", "Marlon"),
            ("Smibert", "Peter"),
            ("Satija", "Rahul"),
        ],
        title="Comprehensive Integration of Single-Cell Data",
        journal="Cell",
        year=2019,
        volume="177",
        issue="7",
        pages="1888-1902.e21",
        doi="10.1016/j.cell.2019.05.031",
    ),
    "harmony": dict(
        use="Harmony integration",
        kind="article",
        authors=[
            ("Korsunsky", "Ilya"),
            ("Millard", "Nghia"),
            ("Fan", "Jean"),
            ("Slowikowski", "Kamil"),
            ("Zhang", "Fan"),
            ("Wei", "Kevin"),
            ("Baglaenko", "Yuriy"),
            ("Brenner", "Michael"),
            ("Loh", "Po-ru"),
            ("Raychaudhuri", "Soumya"),
        ],
        title="Fast, sensitive and accurate integration of single-cell data "
        "with Harmony",
        journal="Nature Methods",
        year=2019,
        volume="16",
        issue="12",
        pages="1289-1296",
        doi="10.1038/s41592-019-0619-0",
    ),
    "scanorama": dict(
        use="Scanorama integration",
        kind="article",
        authors=[("Hie", "Brian"), ("Bryson", "Bryan"), ("Berger", "Bonnie")],
        title="Efficient integration of heterogeneous single-cell "
        "transcriptomes using Scanorama",
        journal="Nature Biotechnology",
        year=2019,
        volume="37",
        issue="6",
        pages="685-691",
        doi="10.1038/s41587-019-0113-3",
    ),
    "bbknn": dict(
        use="BBKNN batch-balanced neighbors",
        kind="article",
        authors=[
            ("Polański", "Krzysztof"),
            ("Young", "Matthew D."),
            ("Miao", "Zhichao"),
            ("Meyer", "Kerstin B."),
            ("Teichmann", "Sarah A."),
            ("Park", "Jong-Eun"),
        ],
        title="BBKNN: fast batch alignment of single cell transcriptomes",
        journal="Bioinformatics",
        year=2020,
        volume="36",
        issue="3",
        pages="964-965",
        doi="10.1093/bioinformatics/btz625",
    ),
    "scrublet": dict(
        use="Scrublet doublet detection",
        kind="article",
        authors=[
            ("Wolock", "Samuel L."),
            ("Lopez", "Romain"),
            ("Klein", "Allon M."),
        ],
        title="Scrublet: Computational Identification of Cell Doublets in "
        "Single-Cell Transcriptomic Data",
        journal="Cell Systems",
        year=2019,
        volume="8",
        issue="4",
        pages="281-291.e9",
        doi="10.1016/j.cels.2018.11.005",
    ),
    "doubletdetection": dict(
        use="DoubletDetection doublet detection",
        kind="software",
        authors=[
            ("Gayoso", "Adam"),
            ("Shor", "Jonathan"),
            ("Carr", "Ambrose J."),
            ("Sharma", "Roshan"),
            ("Pe'er", "Dana"),
        ],
        title="DoubletDetection",
        journal="Zenodo",
        year=2020,
        doi="10.5281/zenodo.2678041",
    ),
    "scdblfinder": dict(
        use="scDblFinder doublet detection",
        kind="article",
        authors=[
            ("Germain", "Pierre-Luc"),
            ("Lun", "Aaron"),
            ("Garcia Meixide", "Carlos"),
            ("Macnair", "Will"),
            ("Robinson", "Mark D."),
        ],
        title="Doublet identification in single-cell sequencing data using scDblFinder",
        journal="F1000Research",
        year=2022,
        volume="10",
        pages="979",
        doi="10.12688/f1000research.73600.2",
    ),
    "ntranos2019": dict(
        use="Logistic-regression marker genes",
        kind="article",
        authors=[
            ("Ntranos", "Vasilis"),
            ("Yi", "Lynn"),
            ("Melsted", "Páll"),
            ("Pachter", "Lior"),
        ],
        title="A discriminative learning approach to differential expression "
        "analysis for single-cell RNA-seq",
        journal="Nature Methods",
        year=2019,
        volume="16",
        issue="2",
        pages="163-166",
        doi="10.1038/s41592-018-0303-9",
    ),
    "limma": dict(
        use="limma linear models",
        kind="article",
        authors=[
            ("Ritchie", "Matthew E."),
            ("Phipson", "Belinda"),
            ("Wu", "Di"),
            ("Hu", "Yifang"),
            ("Law", "Charity W."),
            ("Shi", "Wei"),
            ("Smyth", "Gordon K."),
        ],
        title="limma powers differential expression analyses for RNA-sequencing "
        "and microarray studies",
        journal="Nucleic Acids Research",
        year=2015,
        volume="43",
        issue="7",
        pages="e47",
        doi="10.1093/nar/gkv007",
    ),
    "voom": dict(
        use="voom precision weights",
        kind="article",
        authors=[
            ("Law", "Charity W."),
            ("Chen", "Yunshun"),
            ("Shi", "Wei"),
            ("Smyth", "Gordon K."),
        ],
        title="voom: precision weights unlock linear model analysis tools for "
        "RNA-seq read counts",
        journal="Genome Biology",
        year=2014,
        volume="15",
        issue="2",
        pages="R29",
        doi="10.1186/gb-2014-15-2-r29",
    ),
    "edger": dict(
        use="edgeR",
        kind="article",
        authors=[
            ("Chen", "Yunshun"),
            ("Chen", "Lizhong"),
            ("Lun", "Aaron T. L."),
            ("Baldoni", "Pedro L."),
            ("Smyth", "Gordon K."),
        ],
        title="edgeR v4: powerful differential analysis of sequencing data with "
        "expanded functionality and improved support for small counts and "
        "larger datasets",
        journal="Nucleic Acids Research",
        year=2025,
        volume="53",
        issue="2",
        pages="gkaf018",
        doi="10.1093/nar/gkaf018",
    ),
    "edger_ql": dict(
        use="edgeR quasi-likelihood pipeline",
        kind="article",
        authors=[
            ("Chen", "Yunshun"),
            ("Lun", "Aaron T. L."),
            ("Smyth", "Gordon K."),
        ],
        title="From reads to genes to pathways: differential expression analysis "
        "of RNA-Seq experiments using Rsubread and the edgeR quasi-likelihood "
        "pipeline",
        journal="F1000Research",
        year=2016,
        volume="5",
        pages="1438",
        doi="10.12688/f1000research.8987.2",
    ),
    "tmm": dict(
        use="TMM normalization",
        kind="article",
        authors=[("Robinson", "Mark D."), ("Oshlack", "Alicia")],
        title="A scaling normalization method for differential expression "
        "analysis of RNA-seq data",
        journal="Genome Biology",
        year=2010,
        volume="11",
        issue="3",
        pages="R25",
        doi="10.1186/gb-2010-11-3-r25",
    ),
    "pylimma": dict(
        use="pylimma (Python limma)",
        kind="preprint",
        authors=[("Mulvey", "John F.")],
        title="pylimma: a faithful, AnnData-native Python port of R limma for "
        "differential expression analysis",
        journal="bioRxiv",
        year=2026,
        doi="10.64898/2026.07.06.736732",
    ),
    "edgepython": dict(
        use="edgePython (Python edgeR)",
        kind="preprint",
        authors=[("Pachter", "Lior")],
        title="Differential analysis of genomics count data with edge*",
        journal="bioRxiv",
        year=2026,
        doi="10.64898/2026.02.16.706223",
    ),
    "sctype": dict(
        use="ScType cell-type annotation",
        kind="article",
        authors=[
            ("Ianevski", "Aleksandr"),
            ("Giri", "Anil K."),
            ("Aittokallio", "Tero"),
        ],
        title="Fully-automated and ultra-fast cell-type identification using "
        "specific marker combinations from single-cell transcriptomic data",
        journal="Nature Communications",
        year=2022,
        volume="13",
        pages="1246",
        doi="10.1038/s41467-022-28803-w",
    ),
    "alra": dict(
        use="ALRA imputation",
        kind="article",
        authors=[
            ("Linderman", "George C."),
            ("Zhao", "Jun"),
            ("Roulis", "Manolis"),
            ("Bielecki", "Piotr"),
            ("Flavell", "Richard A."),
            ("Nadler", "Boaz"),
            ("Kluger", "Yuval"),
        ],
        title="Zero-preserving imputation of single-cell RNA-seq data",
        journal="Nature Communications",
        year=2022,
        volume="13",
        pages="192",
        doi="10.1038/s41467-021-27729-z",
    ),
}

_session: list[str] = []
_lock = threading.Lock()


def _check_keys(keys) -> None:
    unknown = [k for k in keys if k not in REFERENCES]
    if unknown:
        raise KeyError(f"unknown citation key(s) {unknown}; see known_citations()")


def _add(keys: list[str], new) -> list[str]:
    return keys + [k for k in dict.fromkeys(new) if k not in keys]


def _adata_keys(adata) -> list[str]:
    value = adata.uns.get(UNS_KEY, {}).get(CITATIONS_KEY)
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    return [str(k) for k in value]  # an array after an h5ad round trip


def record_citations(adata, *keys: str) -> None:
    """Record ``keys`` (plus ``"sclab"``, and ``"anndata"`` with an AnnData).

    Adds them to the session record and, unless ``adata`` is None or a view,
    to ``adata.uns["sclab"]["citations"]``. ``adata.uns["sclab"]`` is replaced,
    not modified, so the dashboard's shallow copies stay independent.
    """
    _check_keys(keys)
    keys = ("sclab", *(() if adata is None else ("anndata",)), *keys)
    with _lock:
        _session[:] = _add(_session, keys)
    if adata is None or adata.is_view:
        return
    old = _adata_keys(adata)
    new = _add(old, keys)
    if new != old:
        adata.uns[UNS_KEY] = {**adata.uns.get(UNS_KEY, {}), CITATIONS_KEY: new}


def _find_adata(args, kwargs):
    from anndata import AnnData

    return next((v for v in (*args, *kwargs.values()) if isinstance(v, AnnData)), None)


def cites(*keys: str):
    """Decorator: after a successful call, record ``keys`` (see :func:`record_citations`)
    for the first AnnData argument (or only for the session, without one)."""
    _check_keys(keys)

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            record_citations(_find_adata(args, kwargs), *keys)
            return result

        return wrapper

    return decorator


# --- rendering -------------------------------------------------------------


def _initials(given: str) -> str:
    # "F. Alexander" -> "F. A.", "Pierre-Luc" -> "P.-L."
    return " ".join(
        "-".join(p[0] + "." for p in word.split("-") if p) for word in given.split()
    )


def _text(e: dict) -> str:
    names = [f"{f}, {_initials(g)}" for f, g in e["authors"]]
    authors = (
        names[0] if len(names) == 1 else ", ".join(names[:-1]) + ", & " + names[-1]
    )
    source = e["journal"]
    if e.get("volume"):
        source += f", {e['volume']}"
        if e.get("issue"):
            source += f"({e['issue']})"
    if e.get("pages"):
        source += f", {e['pages'].replace('-', '–')}"
    link = f"https://doi.org/{e['doi']}" if e.get("doi") else e["url"]
    return f"{authors} ({e['year']}). {e['title']}. {source}. {link}"


def _bibtex(key: str, e: dict) -> str:
    fields = {
        "author": " and ".join(f"{f}, {g}" for f, g in e["authors"]),
        "title": "{" + e["title"] + "}",
        ("journal" if e["kind"] != "software" else "howpublished"): e["journal"],
        "year": str(e["year"]),
        "volume": e.get("volume"),
        "number": e.get("issue"),
        "pages": e["pages"].replace("-", "--") if e.get("pages") else None,
        "doi": e.get("doi"),
        "url": e.get("url"),
    }
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    kind = "misc" if e["kind"] == "software" else "article"
    return f"@{kind}{{{key},\n{body}\n}}"


def _ris(e: dict) -> str:
    tags = [("TY", "COMP" if e["kind"] == "software" else "JOUR")]
    tags += [("AU", f"{f}, {g}") for f, g in e["authors"]]
    tags += [("TI", e["title"]), ("T2", e["journal"]), ("PY", str(e["year"]))]
    tags += [("VL", e.get("volume")), ("IS", e.get("issue"))]
    first, _, last = (e.get("pages") or "").partition("-")
    tags += [("SP", first), ("EP", last), ("DO", e.get("doi")), ("UR", e.get("url"))]
    return "\n".join(f"{t}  - {v}" for t, v in tags if v) + "\nER  - "


def _as_dict(key: str, e: dict) -> dict:
    d = {"key": key, **e}
    d["authors"] = [f"{f}, {g}" for f, g in e["authors"]]
    if d.get("doi") and not d.get("url"):
        d["url"] = f"https://doi.org/{d['doi']}"
    return d


def _render(keys: list[str], format: str):
    if format not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, got {format!r}")
    known = [k for k in keys if k in REFERENCES]
    if len(known) < len(keys):  # e.g. a dataset saved by a newer sclab
        unknown = [k for k in keys if k not in REFERENCES]
        warnings.warn(f"skipping unknown citation key(s) {unknown}", stacklevel=3)
    entries = [(k, REFERENCES[k]) for k in known]
    if format == "dict":
        return [_as_dict(k, e) for k, e in entries]
    if format == "text":
        return "\n\n".join(_text(e) for _, e in entries)
    if format == "bibtex":
        return "\n\n".join(_bibtex(k, e) for k, e in entries)
    return "\n\n".join(_ris(e) for _, e in entries)


# --- public API ------------------------------------------------------------


def active_citations(adata=None, format: str = "text"):
    """References for the methods run on ``adata``, or in this session.

    Parameters
    ----------
    adata : AnnData or None, optional
        Dataset whose ``adata.uns["sclab"]["citations"]`` to render (recorded
        by sclab methods, saved with the h5ad). If None, the methods run in
        this Python session (on any dataset). Default is None.
    format : {"text", "bibtex", "ris", "dict"}, optional
        ``"text"`` (APA-like), ``"bibtex"`` or ``"ris"`` give one string,
        entries separated by blank lines; ``"dict"`` a list of dicts with the
        fields (``key``, ``use``, ``authors``, ``title``, ``journal``,
        ``year``, ``volume``, ``issue``, ``pages``, ``doi``, ``url``).
        Default is ``"text"``.

    Examples
    --------
    >>> import sclab
    >>> sclab.preprocess.pca(adata)  # doctest: +SKIP
    >>> print(sclab.active_citations(adata, format="bibtex"))  # doctest: +SKIP
    """
    with _lock:
        keys = list(_session) if adata is None else _adata_keys(adata)
    return _render(keys, format)


def citation(key: str, format: str = "text"):
    """One reference by key (see :func:`known_citations`), in ``format``.

    ``format`` is as in :func:`active_citations`; ``"dict"`` gives one dict.
    """
    _check_keys([key])
    result = _render([key], format)
    return result[0] if format == "dict" else result


def known_citations(format: str = "text"):
    """Every reference sclab knows, in ``format`` (see :func:`active_citations`)."""
    return _render(list(REFERENCES), format)
