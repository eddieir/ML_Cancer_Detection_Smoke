"""data/converters.py — raw GEO/TCGA/NLST downloads -> clean loader formats."""
import gzip
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import data.converters as converters


def _write_series_matrix(path, sample_ids, characteristics, expr):
    """
    Minimal synthetic GEO `*_series_matrix.txt.gz` file — enough of the real
    format (!Sample_geo_accession, !Sample_characteristics_ch1,
    !series_matrix_table_begin/end) for _parse_series_matrix() to read.
    """
    lines = ['!Sample_geo_accession\t' + "\t".join(f'"{s}"' for s in sample_ids)]
    for key, values in characteristics.items():
        lines.append(f'!Sample_characteristics_ch1\t' +
                      "\t".join(f'"{key}: {v}"' for v in values))
    lines.append("!series_matrix_table_begin")
    header = "ID_REF\t" + "\t".join(sample_ids)
    lines.append(header)
    for gene, row in expr.items():
        lines.append(gene + "\t" + "\t".join(str(v) for v in row))
    lines.append("!series_matrix_table_end")

    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines))


@pytest.fixture()
def tmp_paths(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        converted = Path(tmp) / "converted"
        monkeypatch.setattr(converters, "CONVERTED", converted)
        yield Path(tmp), converted


def test_parse_series_matrix_reads_expr_and_metadata(tmp_paths):
    tmp, _ = tmp_paths
    gz = tmp / "GSEXXX_series_matrix.txt.gz"
    _write_series_matrix(
        gz,
        sample_ids=["GSM1", "GSM2"],
        characteristics={"smoking status": ["current smoker", "never smoker"]},
        expr={"G1": [1.0, 2.0], "G2": [3.0, 4.0]},
    )
    expr, meta = converters._parse_series_matrix(gz)
    assert list(expr.columns) == ["GSM1", "GSM2"]
    assert expr.loc["G1", "GSM1"] == 1.0
    assert meta.loc["GSM1", "smoking status"] == "current smoker"


def test_convert_microarray_infers_per_sample_smoke_type(tmp_paths):
    tmp, converted = tmp_paths
    gz = tmp / "GSE994_series_matrix.txt.gz"
    _write_series_matrix(
        gz,
        sample_ids=["GSM1", "GSM2"],
        characteristics={"smoking status": ["current smoker", "never smoker"]},
        expr={"G1": [1.0, 2.0], "G2": [3.0, 4.0]},
    )
    csv_path = converters.convert_microarray("GSE994", gz, "cigarette")
    assert csv_path.exists()

    meta = pd.read_csv(converted / "GSE994_samples_meta.csv").set_index("sample_id")
    assert meta.loc["GSM1", "smoke_type"] == "cigarette"
    assert meta.loc["GSM2", "smoke_type"] == "unexposed"


def test_convert_canuck_classifies_smoke_type_from_three_fields(tmp_paths):
    """
    GSE307690 (CANUCK) ships its real expression matrix as a separate
    space-delimited supplementary file keyed by generic "sampleN" labels,
    while the series matrix embeds no expression table at all — only
    metadata, in GSM column order. convert_canuck() must join the two
    positionally and derive smoke_type from three independent fields
    (cannabis group / cigarette / vape), including the dual_use case a
    single-field classifier would miss.
    """
    tmp, converted = tmp_paths
    gz = tmp / "GSE307690_series_matrix.txt.gz"
    _write_series_matrix(
        gz,
        sample_ids=["GSM1", "GSM2", "GSM3", "GSM4"],
        characteristics={
            "cannabis group": ["Cannabis", "Cannabis", "Non-Cannabis", "Non-Cannabis"],
            "cigarette":      ["current", "never", "former", "never"],
            "vape":           ["No", "No", "No", "No"],
        },
        expr={"G1": [1.0, 2.0, 3.0, 4.0]},  # embedded table is unused by convert_canuck
    )
    processed = tmp / "GSE307690_processed_data.txt.gz"
    with gzip.open(processed, "wt") as f:
        f.write("sample1 sample2 sample3 sample4\n")
        f.write("ENSG00000000003.16_9 1.0 2.0 3.0 4.0\n")
        f.write("ENSG00000000005.6_2 5.0 6.0 7.0 8.0\n")

    out_path = converters.convert_canuck("GSE307690", gz, processed)
    expr = pd.read_csv(out_path, index_col=0)
    assert list(expr.index) == ["ENSG00000000003", "ENSG00000000005"]  # suffix stripped

    meta = pd.read_csv(converted / "GSE307690_samples_meta.csv").set_index("sample_id")
    assert meta.loc["GSM1", "smoke_type"] == "dual_use"   # cannabis + current cigarette
    assert meta.loc["GSM2", "smoke_type"] == "cannabis"
    assert meta.loc["GSM3", "smoke_type"] == "cigarette"  # former cigarette, no cannabis
    assert meta.loc["GSM4", "smoke_type"] == "unexposed"


def test_convert_scrna_10x_without_donor_map_defaults_unknown(tmp_paths, capsys):
    import gzip as gzip_module
    import scipy.io as sio
    import scipy.sparse as sp

    tmp, converted = tmp_paths
    src = tmp / "GSE_TEST"
    src.mkdir()
    n_genes, n_cells = 5, 4
    mtx = sp.random(n_genes, n_cells, density=0.5, format="csr")  # 10x convention: genes x cells
    sio.mmwrite(str(src / "matrix.mtx"), mtx)
    with open(src / "matrix.mtx", "rb") as f_in, gzip_module.open(src / "matrix.mtx.gz", "wb") as f_out:
        f_out.write(f_in.read())
    (src / "matrix.mtx").unlink()

    with gzip_module.open(src / "barcodes.tsv.gz", "wt") as f:
        f.write("\n".join(f"BC{i}" for i in range(n_cells)) + "\n")
    with gzip_module.open(src / "features.tsv.gz", "wt") as f:
        f.write("\n".join(f"ENSG{i}\tGENE{i}\tGene Expression" for i in range(n_genes)) + "\n")

    out_path = converters.convert_scrna_10x("GSE_TEST", src)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    import anndata as ad
    adata = ad.read_h5ad(out_path)
    assert (adata.obs["donor_id"] == "unknown").all()


def test_convert_scrna_10x_combined_matrix_transposes_genes_x_cells(tmp_paths):
    """
    Regression test: GEO's combined `*_RawCounts_Sparse.mtx.gz` supplementary
    files (e.g. GSE136831) ship genes x cells, same as standard 10x triples,
    but convert_scrna_10x's non-triples branch used to build the AnnData
    directly from that orientation without transposing — AnnData requires
    obs (cells) x var (genes), so this silently produced a shape mismatch /
    swapped cells and genes. Fixed by orienting against the real barcode count.
    """
    import gzip as gzip_module
    import scipy.io as sio
    import scipy.sparse as sp

    tmp, converted = tmp_paths
    src = tmp / "GSE_COMBINED"
    src.mkdir()
    n_genes, n_cells = 7, 5
    mtx = sp.random(n_genes, n_cells, density=0.5, format="csr")  # genes x cells, like real GEO files
    sio.mmwrite(str(src / "RawCounts_Sparse.mtx"), mtx)
    with open(src / "RawCounts_Sparse.mtx", "rb") as f_in, \
         gzip_module.open(src / "RawCounts_Sparse.mtx.gz", "wb") as f_out:
        f_out.write(f_in.read())
    (src / "RawCounts_Sparse.mtx").unlink()

    with gzip_module.open(src / "cellBarcodes.txt.gz", "wt") as f:
        f.write("\n".join(f"BC{i}" for i in range(n_cells)) + "\n")
    with gzip_module.open(src / "GeneIDs.txt.gz", "wt") as f:
        f.write("\n".join(f"ENSG{i}" for i in range(n_genes)) + "\n")

    out_path = converters.convert_scrna_10x("GSE_COMBINED", src)
    import anndata as ad
    adata = ad.read_h5ad(out_path)
    assert adata.n_obs == n_cells
    assert adata.n_vars == n_genes
    assert list(adata.obs_names) == [f"BC{i}" for i in range(n_cells)]
    assert list(adata.var_names) == [f"ENSG{i}" for i in range(n_genes)]


def test_convert_nlst_outcomes(tmp_paths):
    tmp, converted = tmp_paths
    prsn = tmp / "prsn.csv"
    pd.DataFrame({"pid": ["p1", "p2", "p3"], "candx": [1, 0, 1]}).to_csv(prsn, index=False)

    out_path = converters.convert_nlst_outcomes(prsn)
    out = pd.read_csv(out_path)
    assert out.set_index("subject_id")["cancer_label"].to_dict() == {"p1": 1, "p2": 0, "p3": 1}


def test_convert_tcga_splits_tumor_and_normal(tmp_paths):
    tmp, converted = tmp_paths
    src = tmp / "TCGA-LUAD"
    src.mkdir()

    pd.DataFrame([
        {"file_id": "f1", "file_name": "f1.htseq.counts", "case_id": "case1", "sample_type": "Primary Tumor"},
        {"file_id": "f2", "file_name": "f2.htseq.counts", "case_id": "case2", "sample_type": "Solid Tissue Normal"},
    ]).to_csv(src / "file_meta.csv", index=False)

    for fid, counts in [("f1", {"ENSG001.1": 10, "ENSG002.3": 20, "__no_feature": 5}),
                         ("f2", {"ENSG001.1": 1,  "ENSG002.3": 2,  "__no_feature": 1})]:
        d = src / fid
        d.mkdir()
        with open(d / f"{fid}.htseq.counts", "w") as f:
            for gene, count in counts.items():
                f.write(f"{gene}\t{count}\n")

    csv_path = converters.convert_tcga("TCGA-LUAD", src)
    expr = pd.read_csv(csv_path, index_col=0)
    assert list(expr.index) == ["ENSG001", "ENSG002"]   # Ensembl version stripped, __no_feature dropped
    assert expr.loc["ENSG001", "f1"] == 10

    meta = pd.read_csv(converted / "TCGA-LUAD_samples_meta.csv").set_index("sample_id")
    assert meta.loc["f1", "malignancy"] == 1.0
    assert meta.loc["f2", "malignancy"] == 0.0
    assert meta.loc["f1", "subject_id"] == "case1"

    outcomes = pd.read_csv(converted / "TCGA-LUAD_outcomes.csv").set_index("subject_id")
    assert outcomes.loc["case1", "cancer_label"] == 1
    assert outcomes.loc["case2", "cancer_label"] == 0


def test_convert_tcga_missing_file_meta_returns_none(tmp_paths):
    tmp, _ = tmp_paths
    src = tmp / "TCGA-LUSC"
    src.mkdir()
    assert converters.convert_tcga("TCGA-LUSC", src) is None
