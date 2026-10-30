"""Helpers that write tiny 10x-style files (h5 v3, h5 legacy, mtx) to disk."""

import gzip
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

BARCODES = ["AAAC-1", "AAAG-1", "AACT-1", "AAGT-1"]
GENE_IDS = ["ENSG1", "ENSG2", "ENSG3"]
GENE_NAMES = ["GA", "GB", "GC"]
# genes x cells (10x orientation); entries are integer counts
COUNTS_GXC = np.array([[1, 0, 2, 0], [0, 3, 0, 4], [5, 0, 0, 6]], dtype=np.int32)


def counts_cxg() -> np.ndarray:
    return COUNTS_GXC.T.astype(np.float32)


def write_10x_h5_v3(
    path: Path,
    feature_types=("Gene Expression", "Gene Expression", "Antibody Capture"),
    genomes=("hg", "hg", "hg"),
):
    # 10x stores the matrix genes x cells in CSC layout
    from scipy.sparse import csc_matrix

    m = csc_matrix(COUNTS_GXC)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=m.data.astype(np.int32))
        g.create_dataset("indices", data=m.indices.astype(np.int64))
        g.create_dataset("indptr", data=m.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array(m.shape, dtype=np.int32))
        g.create_dataset("barcodes", data=np.array(BARCODES, dtype="S"))
        feat = g.create_group("features")
        feat.create_dataset("id", data=np.array(GENE_IDS, dtype="S"))
        feat.create_dataset("name", data=np.array(GENE_NAMES, dtype="S"))
        feat.create_dataset("feature_type", data=np.array(feature_types, dtype="S"))
        feat.create_dataset("genome", data=np.array(genomes, dtype="S"))
    return path


def write_10x_h5_legacy(path: Path, genome="hg19"):
    from scipy.sparse import csc_matrix

    m = csc_matrix(COUNTS_GXC)
    with h5py.File(path, "w") as f:
        g = f.create_group(genome)
        g.create_dataset("data", data=m.data.astype(np.int32))
        g.create_dataset("indices", data=m.indices.astype(np.int64))
        g.create_dataset("indptr", data=m.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array(m.shape, dtype=np.int32))
        g.create_dataset("barcodes", data=np.array(BARCODES, dtype="S"))
        g.create_dataset("genes", data=np.array(GENE_IDS, dtype="S"))
        g.create_dataset("gene_names", data=np.array(GENE_NAMES, dtype="S"))
    return path


def write_10x_mtx_v3(
    folder: Path,
    feature_types=("Gene Expression", "Gene Expression", "Antibody Capture"),
):
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "tmp_matrix.mtx"
    mmwrite(str(tmp), csr_matrix(COUNTS_GXC), field="integer")
    with open(tmp, "rb") as src, gzip.open(folder / "matrix.mtx.gz", "wb") as dst:
        dst.write(src.read())
    tmp.unlink()
    feats = pd.DataFrame([GENE_IDS, GENE_NAMES, list(feature_types)]).T
    feats.to_csv(folder / "features.tsv.gz", sep="\t", header=False, index=False)
    pd.Series(BARCODES).to_csv(folder / "barcodes.tsv.gz", header=False, index=False)
    return folder


def write_10x_mtx_legacy(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    mmwrite(str(folder / "matrix.mtx"), csr_matrix(COUNTS_GXC), field="integer")
    pd.DataFrame([GENE_IDS, GENE_NAMES]).T.to_csv(
        folder / "genes.tsv", sep="\t", header=False, index=False
    )
    pd.Series(BARCODES).to_csv(folder / "barcodes.tsv", header=False, index=False)
    return folder
