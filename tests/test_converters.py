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


def _write_gpl_annot(path, probe_to_symbol):
    """
    Minimal synthetic GPL platform annotation file — enough of the real
    `!platform_table_begin/end`-delimited TSV format for
    _load_probe_to_symbol_map() to parse.
    """
    lines = [
        "!Annotation_platform = GPLTEST",
        "!platform_table_begin",
        "ID\tGene title\tGene symbol\tGene ID",
    ]
    for probe, symbol in probe_to_symbol.items():
        lines.append(f"{probe}\tsome gene title\t{symbol}\t123")
    lines.append("!platform_table_end")
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def test_convert_microarray_maps_illumina_probes_to_gene_symbols(tmp_paths):
    """
    GSE123352 ships Illumina HumanHT-12 probe IDs (ILMN_...) that BioMart
    can't resolve — GEO's own GPL annotation file is the only way to map
    them to gene symbols. Two probes for the same gene (ILMN_A, ILMN_C
    both -> GAPDH) must collapse via mean, and an unannotated probe
    (ILMN_D, no symbol) must be dropped rather than kept as a bogus gene.
    """
    tmp, converted = tmp_paths
    gz = tmp / "GSE123352_series_matrix.txt.gz"
    _write_series_matrix(
        gz,
        sample_ids=["GSM1", "GSM2"],
        characteristics={"smoking status": ["ever smoker", "never smoker"]},
        expr={
            "ILMN_A": [10.0, 20.0],   # -> GAPDH
            "ILMN_B": [1.0, 2.0],     # -> EEF1A1
            "ILMN_C": [30.0, 40.0],   # -> GAPDH (same gene, different probe)
            "ILMN_D": [5.0, 5.0],     # unannotated -> dropped
        },
    )
    annot = tmp / "GPLTEST.annot.gz"
    _write_gpl_annot(annot, {"ILMN_A": "GAPDH", "ILMN_B": "EEF1A1", "ILMN_C": "GAPDH"})

    csv_path = converters.convert_microarray("GSE123352", gz, "cigarette",
                                              platform_annot_path=annot)
    expr = pd.read_csv(csv_path, index_col=0)

    assert set(expr.index) == {"GAPDH", "EEF1A1"}
    assert expr.loc["GAPDH", "GSM1"] == 20.0   # mean(10, 30)
    assert expr.loc["GAPDH", "GSM2"] == 30.0   # mean(20, 40)
    assert expr.loc["EEF1A1", "GSM1"] == 1.0


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


def test_convert_scrna_10x_cell_metadata_overrides_prefix_guess(tmp_paths):
    """
    Regression test for the real GSE136831 quirk: subject "1372C" has
    barcodes prefixed "137C-a_..." — the barcode prefix does NOT equal the
    subject ID, so the old prefix-regex donor_map path would mislabel this
    cell's donor. A full-barcode cell_metadata join must get it right.
    """
    import gzip as gzip_module
    import scipy.io as sio
    import scipy.sparse as sp

    tmp, converted = tmp_paths
    src = tmp / "GSE_META"
    src.mkdir()
    n_genes, n_cells = 3, 2
    mtx = sp.random(n_genes, n_cells, density=0.5, format="csr")
    sio.mmwrite(str(src / "matrix.mtx"), mtx)
    with open(src / "matrix.mtx", "rb") as f_in, gzip_module.open(src / "matrix.mtx.gz", "wb") as f_out:
        f_out.write(f_in.read())
    (src / "matrix.mtx").unlink()

    barcodes = ["137C-a_AAACCTGCAGCGAACA", "001C_AAACCTGCATCGGGTC"]
    with gzip_module.open(src / "barcodes.tsv.gz", "wt") as f:
        f.write("\n".join(barcodes) + "\n")
    with gzip_module.open(src / "features.tsv.gz", "wt") as f:
        f.write("\n".join(f"ENSG{i}\tGENE{i}\tGene Expression" for i in range(n_genes)) + "\n")

    cell_metadata = pd.DataFrame({
        "donor_id":         ["1372C", "001C"],
        "disease_identity": ["Control", "Control"],
    }, index=barcodes)

    out_path = converters.convert_scrna_10x("GSE_META", src, cell_metadata=cell_metadata)
    import anndata as ad
    adata = ad.read_h5ad(out_path)
    assert adata.obs.loc["137C-a_AAACCTGCAGCGAACA", "donor_id"] == "1372C"
    assert adata.obs.loc["001C_AAACCTGCATCGGGTC", "donor_id"] == "001C"
    assert adata.obs.loc["001C_AAACCTGCATCGGGTC", "disease_identity"] == "Control"


def test_load_gse136831_cell_metadata_parses_real_column_names(tmp_paths):
    tmp, converted = tmp_paths
    src = tmp / "GSE136831"
    src.mkdir()
    meta_path = src / "GSE136831_AllCells.Samples.CellType.MetadataTable.txt.gz"
    with gzip.open(meta_path, "wt") as f:
        f.write('"CellBarcode_Identity"\t"nUMI"\t"nGene"\t"CellType_Category"\t'
                '"Manuscript_Identity"\t"Subclass_Cell_Identity"\t"Disease_Identity"\t'
                '"Subject_Identity"\t"Library_Identity"\n')
        f.write('"137C-a_AAACCTGCAGCGAACA"\t1759\t1100\t"Lymphoid"\t"T_Cytotoxic"\t'
                '"T_Cytotoxic_C"\t"Control"\t"1372C"\t"137C-a"\n')

    meta = converters._load_gse136831_cell_metadata(src)
    assert meta is not None
    assert meta.loc["137C-a_AAACCTGCAGCGAACA", "donor_id"] == "1372C"
    assert meta.loc["137C-a_AAACCTGCAGCGAACA", "disease_identity"] == "Control"


def test_read_id_list_prefers_symbol_column(tmp_paths):
    """
    GSE136831's GeneIDs.txt.gz ships [Ensembl_GeneID, HGNC_EnsemblAlt_GeneID]
    (a real gene symbol) — prefer_symbol_col=True should pick the symbol
    column so harmonize_gene_ids() doesn't need a live BioMart lookup for
    genes that already have one.
    """
    tmp, converted = tmp_paths
    path = tmp / "GeneIDs.txt.gz"
    with gzip.open(path, "wt") as f:
        f.write('"Ensembl_GeneID"\t"HGNC_EnsemblAlt_GeneID"\n')
        f.write('"ENSG00000000003"\t"TSPAN6"\n')
        f.write('"ENSG00000000005"\t"TNMD"\n')

    ids = converters._read_id_list(path, expected_len=2, prefer_symbol_col=True)
    assert ids == ["TSPAN6", "TNMD"]

    ids_default = converters._read_id_list(path, expected_len=2)
    assert ids_default == ["ENSG00000000003", "ENSG00000000005"]


def test_read_id_list_single_column_unaffected_by_prefer_symbol(tmp_paths):
    tmp, converted = tmp_paths
    path = tmp / "barcodes.tsv.gz"
    with gzip.open(path, "wt") as f:
        f.write("BC0\nBC1\n")
    assert converters._read_id_list(path, prefer_symbol_col=True) == ["BC0", "BC1"]


def test_convert_gse288003_labels_con_and_ecig_separately(tmp_paths):
    """
    Regression test for the real correctness bug: GSE288003's two GSM
    samples are "Con" (unexposed control mouse) and "E-cigs" (e-cig
    exposed). Blanket-labelling every cell "vape" (the accession's
    GEO_DATASETS default) would mislabel the entire control sample.
    convert_gse288003 must tag each sample with its real condition.
    """
    import gzip as gzip_module
    import scipy.io as sio
    import scipy.sparse as sp

    tmp, converted = tmp_paths
    src = tmp / "GSE288003"
    src.mkdir()

    def _write_sample(prefix, n_genes, n_cells):
        mtx = sp.random(n_genes, n_cells, density=0.5, format="csr")  # genes x cells
        sio.mmwrite(str(src / f"{prefix}_matrix.mtx"), mtx)
        with open(src / f"{prefix}_matrix.mtx", "rb") as f_in, \
             gzip_module.open(src / f"{prefix}_matrix.mtx.gz", "wb") as f_out:
            f_out.write(f_in.read())
        (src / f"{prefix}_matrix.mtx").unlink()
        with gzip_module.open(src / f"{prefix}_barcodes.tsv.gz", "wt") as f:
            f.write("\n".join(f"BC{i}" for i in range(n_cells)) + "\n")
        with gzip_module.open(src / f"{prefix}_genes.tsv.gz", "wt") as f:
            f.write("\n".join(f"ENSG{i}\tGENE{i}\tGene Expression" for i in range(n_genes)) + "\n")

    _write_sample("GSM8757329_Con", n_genes=6, n_cells=3)
    _write_sample("GSM8757330_E-cigs", n_genes=6, n_cells=4)

    out_path = converters.convert_gse288003("GSE288003", src)
    import anndata as ad
    adata = ad.read_h5ad(out_path)

    assert adata.n_obs == 7
    con_cells = adata.obs[adata.obs["donor_id"] == "GSM8757329"]
    ecig_cells = adata.obs[adata.obs["donor_id"] == "GSM8757330"]
    assert len(con_cells) == 3
    assert len(ecig_cells) == 4
    assert (con_cells["smoke_type_name"] == "unexposed").all()
    assert (ecig_cells["smoke_type_name"] == "vape").all()


def test_convert_nlst_outcomes(tmp_paths):
    tmp, converted = tmp_paths
    prsn = tmp / "prsn.csv"
    pd.DataFrame({"pid": ["p1", "p2", "p3"], "candx": [1, 0, 1]}).to_csv(prsn, index=False)

    out_path = converters.convert_nlst_outcomes(prsn)
    out = pd.read_csv(out_path)
    assert out.set_index("subject_id")["cancer_label"].to_dict() == {"p1": 1, "p2": 0, "p3": 1}


def test_convert_nlst_outcomes_drops_missing_candx_instead_of_defaulting_to_zero(tmp_paths):
    """A subject with no candx value recorded must be EXCLUDED from the
    outcomes CSV (unknown outcome), never written as cancer_label=0."""
    tmp, converted = tmp_paths
    prsn = tmp / "prsn.csv"
    pd.DataFrame({"pid": ["p1", "p2", "p3"], "candx": [1, None, 0]}).to_csv(prsn, index=False)

    out_path = converters.convert_nlst_outcomes(prsn)
    out = pd.read_csv(out_path)
    assert set(out["subject_id"].astype(str)) == {"p1", "p3"}
    assert out.set_index("subject_id")["cancer_label"].to_dict() == {"p1": 1, "p3": 0}


def test_convert_nlst_outcomes_missing_candx_column_writes_empty_not_all_negative(tmp_paths):
    tmp, converted = tmp_paths
    prsn = tmp / "prsn.csv"
    pd.DataFrame({"pid": ["p1", "p2"]}).to_csv(prsn, index=False)

    out_path = converters.convert_nlst_outcomes(prsn)
    out = pd.read_csv(out_path)
    assert len(out) == 0


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
