from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import requests
from anndata import AnnData

from sclab import _io
from sclab._io import (
    fetch_file,
    get_file_hash,
    is_valid_url,
    _read_adata_bytes,
    load_adata_from_url,
    read_adata,
)

from .test_utils import _tenx_files as tx


def _dense(adata: AnnData) -> np.ndarray:
    X = adata.X
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


# --------------------------------------------------------------------------
# read_adata
# --------------------------------------------------------------------------
class TestReadAdata:
    def test_h5_v3_filters_gex_and_promotes_gene_ids(self, tmp_path):
        p = tx.write_10x_h5_v3(tmp_path / "v3.h5")
        adata = read_adata(p)
        # Antibody Capture feature dropped; index promoted from gene_ids
        assert adata.shape == (4, 2)
        assert adata.var_names.tolist() == tx.GENE_IDS[:2]
        assert adata.obs_names.tolist() == tx.BARCODES
        np.testing.assert_array_equal(_dense(adata), tx.counts_cxg()[:, :2])

    def test_h5_legacy(self, tmp_path):
        p = tx.write_10x_h5_legacy(tmp_path / "legacy.h5")
        adata = read_adata(p)
        assert adata.shape == (4, 3)
        assert adata.var_names.tolist() == tx.GENE_IDS
        np.testing.assert_array_equal(_dense(adata), tx.counts_cxg())

    def test_var_names_none_found_keeps_default_index(self, tmp_path):
        p = tx.write_10x_h5_v3(tmp_path / "v3.h5")
        adata = read_adata(p, var_names="not_a_column")
        assert adata.var_names.tolist() == tx.GENE_NAMES[:2]
        assert "gene_ids" in adata.var

    def test_h5ad_roundtrip(self, tmp_path):
        import pandas as pd

        src = AnnData(
            tx.counts_cxg(),
            obs=pd.DataFrame(index=tx.BARCODES),
            var=pd.DataFrame({"gene_ids": tx.GENE_IDS}, index=tx.GENE_NAMES),
        )
        p = tmp_path / "x.h5ad"
        src.write_h5ad(p)
        adata = read_adata(p)
        assert adata.var_names.tolist() == tx.GENE_IDS
        assert "gene_ids" not in adata.var  # moved to the index
        np.testing.assert_array_equal(_dense(adata), tx.counts_cxg())

    def test_h5ad_without_var_column(self, tmp_path):
        src = AnnData(tx.counts_cxg())
        p = tmp_path / "x.h5ad"
        src.write_h5ad(p)
        adata = read_adata(str(p))  # str path accepted
        assert adata.shape == (4, 3)
        assert adata.var_names.tolist() == ["0", "1", "2"]

    def test_mtx_folder_v3(self, tmp_path):
        folder = tx.write_10x_mtx_v3(tmp_path / "mtx")
        adata = read_adata(folder)
        assert adata.shape == (4, 2)
        assert adata.var_names.tolist() == tx.GENE_IDS[:2]
        assert adata.obs_names.tolist() == tx.BARCODES
        np.testing.assert_array_equal(_dense(adata), tx.counts_cxg()[:, :2])

    def test_mtx_folder_legacy(self, tmp_path):
        folder = tx.write_10x_mtx_legacy(tmp_path / "mtx")
        adata = read_adata(folder)
        assert adata.shape == (4, 3)
        assert adata.var_names.tolist() == tx.GENE_IDS
        np.testing.assert_array_equal(_dense(adata), tx.counts_cxg())

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="10x h5, h5ad"):
            read_adata(p)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(AssertionError):
            read_adata(tmp_path / "does_not_exist")

    def test_missing_h5_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            read_adata(tmp_path / "nope.h5")


# --------------------------------------------------------------------------
# vendored scanpy readers used by sclab
# --------------------------------------------------------------------------
class TestVendoredReaders:
    def test_read_10x_h5_genome_filter(self, tmp_path):
        from sclab.scanpy.readwrite import read_10x_h5

        p = tx.write_10x_h5_v3(
            tmp_path / "v3.h5",
            feature_types=("Gene Expression",) * 3,
            genomes=("hg", "mm", "mm"),
        )
        adata = read_10x_h5(p, genome="mm")
        assert adata.var_names.tolist() == tx.GENE_NAMES[1:]
        assert not adata.is_view
        with pytest.raises(ValueError, match="Could not find data"):
            read_10x_h5(p, genome="zz")

    def test_read_10x_h5_gex_only_false_keeps_all(self, tmp_path):
        from sclab.scanpy.readwrite import read_10x_h5

        p = tx.write_10x_h5_v3(tmp_path / "v3.h5")
        adata = read_10x_h5(p, gex_only=False)
        assert adata.n_vars == 3
        assert adata.var["feature_types"].tolist() == [
            "Gene Expression",
            "Gene Expression",
            "Antibody Capture",
        ]

    def test_read_10x_h5_legacy_multi_genome_requires_genome(self, tmp_path):
        import h5py

        from sclab.scanpy.readwrite import read_10x_h5

        p = tx.write_10x_h5_legacy(tmp_path / "legacy.h5", genome="hg19")
        with h5py.File(p, "a") as f:
            f.copy("hg19", "mm10")
        with pytest.raises(ValueError, match="more than one genome"):
            read_10x_h5(p)
        assert read_10x_h5(p, genome="mm10").shape == (4, 3)
        with pytest.raises(ValueError, match="Could not find genome"):
            read_10x_h5(p, genome="nope")

    def test_read_10x_mtx_var_names_options(self, tmp_path):
        from sclab.scanpy.readwrite import read_10x_mtx

        folder = tx.write_10x_mtx_v3(tmp_path / "mtx")
        by_symbol = read_10x_mtx(folder)
        assert by_symbol.var_names.tolist() == tx.GENE_NAMES[:2]
        assert by_symbol.var["gene_ids"].tolist() == tx.GENE_IDS[:2]
        by_id = read_10x_mtx(folder, var_names="gene_ids", gex_only=False)
        assert by_id.var_names.tolist() == tx.GENE_IDS
        assert by_id.var["gene_symbols"].tolist() == tx.GENE_NAMES
        with pytest.raises(ValueError, match="var_names"):
            read_10x_mtx(folder, var_names="bogus")

    def test_read_10x_mtx_prefix_and_make_unique(self, tmp_path):
        from sclab.scanpy.readwrite import read_10x_mtx

        folder = tx.write_10x_mtx_legacy(tmp_path / "mtx")
        for f in list(folder.iterdir()):
            f.rename(folder / f"pA_{f.name}")
        # duplicate gene symbol
        genes = folder / "pA_genes.tsv"
        genes.write_text("ENSG1\tDUP\nENSG2\tDUP\nENSG3\tGC\n")
        adata = read_10x_mtx(folder, prefix="pA_")
        assert adata.var_names.is_unique
        assert adata.var_names.tolist() == ["DUP", "DUP-1", "GC"]


# --------------------------------------------------------------------------
# URL helpers
# --------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, content: bytes, status: int = 200, headers=None):
        self._content = content
        self.status_code = status
        self.headers = (
            headers if headers is not None else {"content-length": str(len(content))}
        )

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def iter_content(self, block_size):
        for i in range(0, len(self._content), block_size):
            yield self._content[i : i + block_size]


@pytest.fixture
def fake_get(monkeypatch):
    calls = []
    store = {}

    def get(url, stream=False, **kw):
        calls.append((url, stream))
        return store[url]

    monkeypatch.setattr(_io.requests, "get", get)
    get.calls = calls
    get.store = store
    return get


class TestUrlHelpers:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://example.com/a.h5ad", True),
            ("http://localhost:8000/x", True),
            ("ftp://host/file", True),
            ("example.com/a.h5ad", False),
            ("/local/path.h5ad", False),
            ("https://", False),
            ("", False),
            (None, False),
            (123, False),
        ],
    )
    def test_is_valid_url(self, url, expected):
        assert is_valid_url(url) is expected

    def test_fetch_file_concatenates_chunks(self, fake_get):
        payload = bytes(range(256)) * 20  # 5120 bytes -> several 1 KiB blocks
        fake_get.store["https://x.org/f"] = _FakeResponse(payload)
        buf = fetch_file("https://x.org/f", progress=False)
        assert isinstance(buf, BytesIO)
        assert buf.getvalue() == payload
        assert fake_get.calls == [("https://x.org/f", True)]

    def test_fetch_file_with_progress_and_missing_length(self, fake_get):
        payload = b"abc" * 1000
        fake_get.store["https://x.org/f"] = _FakeResponse(payload, headers={})
        buf = fetch_file("https://x.org/f", progress=True)
        assert buf.getvalue() == payload

    def test_fetch_file_http_error(self, fake_get):
        fake_get.store["https://x.org/f"] = _FakeResponse(b"", status=404)
        with pytest.raises(requests.HTTPError):
            fetch_file("https://x.org/f", progress=False)

    def test_load_adata_from_url_h5ad(self, tmp_path, fake_get):
        import pandas as pd

        src = AnnData(
            tx.counts_cxg(),
            obs=pd.DataFrame(index=tx.BARCODES),
            var=pd.DataFrame({"gene_ids": tx.GENE_IDS}, index=tx.GENE_NAMES),
        )
        p = tmp_path / "x.h5ad"
        src.write_h5ad(p)
        url = "https://x.org/data/x.h5ad?download=1"
        fake_get.store[url] = _FakeResponse(p.read_bytes())
        adata = load_adata_from_url(url, progress=False)
        assert adata.var_names.tolist() == tx.GENE_IDS
        np.testing.assert_array_equal(_dense(adata), tx.counts_cxg())

    @pytest.mark.parametrize("url", ["not a url", "example.com/a.h5ad", ""])
    def test_load_adata_from_url_rejects_invalid_url(self, url):
        with pytest.raises(ValueError, match="URL is not valid"):
            load_adata_from_url(url)

    def test_load_adata_from_url_rejects_unknown_suffix(self, fake_get):
        url = "https://x.org/data.csv"
        fake_get.store[url] = _FakeResponse(b"a,b\n")
        with pytest.raises(ValueError, match=r"'\.csv'.*\.h5.*\.h5ad"):
            load_adata_from_url(url, progress=False)

    def test_load_adata_from_url_10x_h5(self, tmp_path, fake_get):
        p = tx.write_10x_h5_v3(tmp_path / "v3.h5")
        url = "https://x.org/data/v3.h5"
        fake_get.store[url] = _FakeResponse(p.read_bytes())
        adata = load_adata_from_url(url, progress=False)
        assert adata.shape == (4, 2)
        assert adata.var_names.tolist() == tx.GENE_IDS[:2]
        np.testing.assert_array_equal(_dense(adata), tx.counts_cxg()[:, :2])


class TestReadAdataBytes:
    def test_h5_from_bytesio(self, tmp_path):
        p = tx.write_10x_h5_v3(tmp_path / "v3.h5")
        adata = _read_adata_bytes(BytesIO(p.read_bytes()), ".h5")
        assert adata.shape == (4, 2)
        assert adata.var_names.tolist() == tx.GENE_IDS[:2]
        np.testing.assert_array_equal(_dense(adata), tx.counts_cxg()[:, :2])

    def test_h5_from_raw_bytes_and_var_names(self, tmp_path):
        p = tx.write_10x_h5_v3(tmp_path / "v3.h5")
        adata = _read_adata_bytes(p.read_bytes(), ".h5", var_names="not_a_column")
        assert adata.var_names.tolist() == tx.GENE_NAMES[:2]

    def test_h5ad_roundtrip_is_in_memory(self, tmp_path):
        import pandas as pd

        src = AnnData(
            tx.counts_cxg(),
            obs=pd.DataFrame(index=tx.BARCODES),
            var=pd.DataFrame({"gene_ids": tx.GENE_IDS}, index=tx.GENE_NAMES),
        )
        p = tmp_path / "x.h5ad"
        src.write_h5ad(p)
        adata = _read_adata_bytes(BytesIO(p.read_bytes()), ".h5ad")
        assert not adata.isbacked
        assert adata.var_names.tolist() == tx.GENE_IDS
        np.testing.assert_array_equal(_dense(adata), tx.counts_cxg())

    def test_temp_dir_is_removed(self, tmp_path, monkeypatch):
        import tempfile

        seen = []
        real = tempfile.TemporaryDirectory

        class Spy(real):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                seen.append(self.name)

        monkeypatch.setattr(_io.tempfile, "TemporaryDirectory", Spy)
        p = tx.write_10x_h5_v3(tmp_path / "v3.h5")
        _read_adata_bytes(p.read_bytes(), ".h5")
        assert len(seen) == 1
        assert not Path(seen[0]).exists()

    def test_unsupported_suffix(self):
        with pytest.raises(ValueError, match="10x h5, h5ad"):
            _read_adata_bytes(b"a,b\n", ".csv")


class TestDataLoaderOnUpload:
    @staticmethod
    def _loader():
        from sclab._sclab import DataLoader

        return DataLoader(None)

    @staticmethod
    def _upload(name: str, content: bytes) -> tuple:
        return (
            {
                "name": name,
                "type": "application/octet-stream",
                "size": len(content),
                "last_modified": datetime.now(timezone.utc),
                "content": memoryview(content),
            },
        )

    def test_upload_10x_h5(self, tmp_path):
        loader = self._loader()
        p = tx.write_10x_h5_v3(tmp_path / "v3.h5")
        loader.upload.value = self._upload("v3.h5", p.read_bytes())
        assert loader.adata.shape == (4, 2)
        assert loader.adata.var_names.tolist() == tx.GENE_IDS[:2]

    def test_upload_unsupported_suffix_leaves_adata_unset(self):
        loader = self._loader()
        loader.upload.value = self._upload("data.csv", b"a,b\n")
        assert getattr(loader, "adata", None) is None


# --------------------------------------------------------------------------
# get_file_hash
# --------------------------------------------------------------------------
class TestFileHash:
    def test_known_digests(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_bytes(b"abc")
        assert (
            get_file_hash(p)
            == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )
        assert get_file_hash(p, "md5") == "900150983cd24fb0d6963f7d28e17f72"
        assert get_file_hash(str(p), algorithm="sha1") == (
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        )

    def test_empty_file_and_content_sensitivity(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.write_bytes(b"")
        b.write_bytes(b"x")
        assert get_file_hash(a, "md5") == "d41d8cd98f00b204e9800998ecf8427e"
        assert get_file_hash(a) != get_file_hash(b)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_file_hash(tmp_path / "nope")

    def test_unknown_algorithm(self, tmp_path):
        p = tmp_path / "f"
        p.write_bytes(b"x")
        with pytest.raises(ValueError):
            get_file_hash(p, "not-a-hash")
