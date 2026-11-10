import re

import numpy as np
import pytest
from anndata import AnnData, read_h5ad

import sclab
from sclab import _citations
from sclab._citations import REFERENCES, cites, record_citations


@pytest.fixture(autouse=True)
def fresh_session(monkeypatch):
    monkeypatch.setattr(_citations, "_session", [])


def _adata(n_obs=60, n_vars=30, seed=0) -> AnnData:
    rng = np.random.default_rng(seed)
    return AnnData(rng.poisson(2, (n_obs, n_vars)).astype(np.float32))


def _parse_bibtex(text: str) -> dict[str, dict[str, str]]:
    """Minimal BibTeX parser: ``@type{key, field = {value}, ...}`` entries."""
    entries = {}
    for chunk in text.split("\n\n"):
        m = re.fullmatch(r"@(\w+)\{([^,\s]+),\n(.*)\n\}", chunk, flags=re.S)
        assert m, chunk
        assert chunk.count("{") == chunk.count("}"), chunk
        fields = dict(re.findall(r"^  (\w+) = \{(.*)\},?$", m.group(3), flags=re.M))
        assert len(fields) == len(m.group(3).splitlines()), chunk
        entries[m.group(2)] = fields
    return entries


class TestRegistry:
    @pytest.mark.parametrize("key", list(REFERENCES))
    def test_entry_renders_in_every_format(self, key):
        e = REFERENCES[key]
        text = sclab.citation(key)
        assert e["title"] in text and str(e["year"]) in text
        bib = _parse_bibtex(sclab.citation(key, "bibtex"))
        assert bib[key]["year"] == str(e["year"])
        assert e["authors"][0][0] in bib[key]["author"]
        ris = sclab.citation(key, "ris")
        assert ris.startswith("TY  - ") and ris.endswith("ER  - ")
        assert f"TI  - {e['title']}" in ris
        d = sclab.citation(key, "dict")
        assert d["key"] == key and d["use"] and d["url"].startswith("https://")

    @pytest.mark.parametrize("key", list(REFERENCES))
    def test_doi_well_formed(self, key):
        e = REFERENCES[key]
        assert e.get("doi") or e.get("url")
        if e.get("doi"):
            assert re.fullmatch(r"10\.\d{4,9}/\S+", e["doi"]), e["doi"]
            assert not e["doi"].lower().startswith(("http", "doi:"))

    def test_known_citations_lists_all(self):
        bib = _parse_bibtex(sclab.known_citations("bibtex"))
        assert list(bib) == list(REFERENCES)
        assert len(sclab.known_citations("dict")) == len(REFERENCES)

    def test_unknown_key_and_format(self):
        with pytest.raises(KeyError):
            sclab.citation("nope")
        with pytest.raises(KeyError):
            cites("nope")
        with pytest.raises(ValueError):
            sclab.citation("scanpy", "markdown")


class TestRecording:
    def test_success_records_keys(self):
        @cites("harmony")
        def method(adata, value):
            adata.uns["value"] = value

        adata = _adata()
        method(adata, value=1)
        assert adata.uns["sclab"]["citations"] == ["sclab", "anndata", "harmony"]
        method(adata, 2)  # no duplicates
        assert adata.uns["sclab"]["citations"] == ["sclab", "anndata", "harmony"]
        keys = [d["key"] for d in sclab.active_citations(adata, format="dict")]
        assert keys == ["sclab", "anndata", "harmony"]

    def test_failure_records_nothing(self):
        @cites("harmony")
        def method(adata):
            raise RuntimeError("boom")

        adata = _adata()
        with pytest.raises(RuntimeError):
            method(adata)
        assert "sclab" not in adata.uns
        assert sclab.active_citations() == ""

    def test_keeps_other_sclab_entries_and_replaces_dict(self):
        adata = _adata()
        before = {"history": "[]"}
        adata.uns["sclab"] = before
        record_citations(adata, "scanpy")
        assert adata.uns["sclab"]["history"] == "[]"
        assert "citations" not in before  # replaced, not modified in place

    def test_view_records_session_only(self):
        adata = _adata()
        record_citations(adata[:10], "scanpy")
        assert "sclab" not in adata.uns
        assert "scanpy" in sclab.active_citations(format="bibtex")

    def test_library_method_records(self):
        adata = _adata()
        sclab.preprocess.normalize_weighted(adata)
        assert adata.uns["sclab"]["citations"] == ["sclab", "anndata"]
        sclab.preprocess.pca(adata, n_comps=5)
        assert adata.uns["sclab"]["citations"] == [
            "sclab",
            "anndata",
            "scanpy",
            "scikit-learn",
        ]

    def test_failing_library_method_records_nothing(self):
        adata = _adata()
        with pytest.raises(KeyError):
            sclab.preprocess.harmony_integrate(adata, "batch")  # no X_pca
        assert "sclab" not in adata.uns
        assert sclab.active_citations(format="dict") == []


class TestScopes:
    def test_session_vs_adata(self):
        a, b = _adata(), _adata()
        record_citations(a, "harmony")
        record_citations(b, "scrublet")
        record_citations(None, "alra")  # e.g. an ndarray-only function
        keys = lambda x: [d["key"] for d in sclab.active_citations(x, "dict")]  # noqa: E731
        assert keys(a) == ["sclab", "anndata", "harmony"]
        assert keys(b) == ["sclab", "anndata", "scrublet"]
        assert keys(None) == ["sclab", "anndata", "harmony", "scrublet", "alra"]

    def test_without_records(self):
        assert sclab.active_citations(_adata()) == ""
        assert sclab.active_citations(_adata(), "dict") == []

    def test_unknown_stored_key_is_skipped(self):
        adata = _adata()
        adata.uns["sclab"] = {"citations": ["scanpy", "from-the-future"]}
        with pytest.warns(UserWarning, match="from-the-future"):
            out = sclab.active_citations(adata, "dict")
        assert [d["key"] for d in out] == ["scanpy"]


def test_h5ad_round_trip(tmp_path):
    adata = _adata()
    record_citations(adata, "harmony", "scanpy")
    path = tmp_path / "a.h5ad"
    adata.write_h5ad(path)
    back = read_h5ad(path)
    assert sclab.active_citations(back) == sclab.active_citations(adata)
    record_citations(back, "umap")  # appends to the stored (array) keys
    keys = [d["key"] for d in sclab.active_citations(back, "dict")]
    assert keys == ["sclab", "anndata", "harmony", "scanpy", "umap"]
    back.write_h5ad(path)
    assert sclab.active_citations(read_h5ad(path)) == sclab.active_citations(back)
