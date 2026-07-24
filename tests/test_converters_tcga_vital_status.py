"""
tests/test_converters_tcga_vital_status.py — adversarial coverage for
data/converters.py::convert_tcga_vital_status and _read_tcga_star_gene_counts,
the real GDC open-access "STAR - Counts" parser and vital-status outcome
converter (Issue #16 follow-up: cancer-prediction evidence via a genuinely
linked TCGA-LUAD/TCGA-LUSC expression<->vital-status cohort).
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data import converters
from data.converters import _read_tcga_star_gene_counts, convert_tcga_vital_status


@pytest.fixture()
def tmp_paths(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        converted = Path(tmp) / "converted"
        monkeypatch.setattr(converters, "CONVERTED", converted)
        yield Path(tmp), converted

STAR_HEADER = "# gene-model: GENCODE v36\ngene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second\ttpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded\n"
STAR_STAT_ROWS = (
    "N_unmapped\t\t\t100\t100\t100\t\t\t\n"
    "N_multimapping\t\t\t200\t200\t200\t\t\t\n"
    "N_noFeature\t\t\t300\t300\t300\t\t\t\n"
    "N_ambiguous\t\t\t400\t400\t400\t\t\t\n"
)


def _write_star_file(path: Path, genes: dict) -> None:
    lines = [STAR_HEADER, STAR_STAT_ROWS]
    for gene_id, tpm in genes.items():
        lines.append(f"{gene_id}\tSYM_{gene_id}\tprotein_coding\t10\t5\t5\t{tpm}\t1.0\t1.0\n")
    path.write_text("".join(lines))


def test_read_star_gene_counts_drops_stat_rows_and_strips_version(tmp_path):
    f = tmp_path / "sample.tsv.gz"
    _write_star_file(f, {"ENSG00000000003.15": 94.7, "ENSG00000000005.6": 0.0})
    series = _read_tcga_star_gene_counts(f)
    assert set(series.index) == {"ENSG00000000003", "ENSG00000000005"}
    assert series["ENSG00000000003"] == pytest.approx(94.7)
    assert "N_unmapped" not in series.index


def test_convert_tcga_vital_status_excludes_unknown_status(tmp_path, tmp_paths):
    src = tmp_path
    _write_star_file(src / "file1.tsv.gz", {"ENSG00000000003.15": 1.0})
    _write_star_file(src / "file2.tsv.gz", {"ENSG00000000003.15": 2.0})
    pd.DataFrame({
        "file_id": ["file1", "file2"],
        "case_id": ["case1", "case2"],
        "submitter_id": ["TCGA-01", "TCGA-02"],
        "vital_status": ["Alive", "Not Reported"],
        "days_to_death": [None, None],
    }).to_csv(src / "outcome_meta.csv", index=False)

    csv_path = convert_tcga_vital_status("TCGA-TEST", src)
    assert csv_path is not None
    meta = pd.read_csv(csv_path.with_name(csv_path.stem + "_samples_meta.csv"))
    assert list(meta["sample_id"]) == ["file1"]
    assert meta["vital_status"].iloc[0] == 0.0  # Alive


def test_convert_tcga_vital_status_excludes_duplicate_case_id(tmp_path, tmp_paths):
    src = tmp_path
    _write_star_file(src / "file1.tsv.gz", {"ENSG00000000003.15": 1.0})
    _write_star_file(src / "file2.tsv.gz", {"ENSG00000000003.15": 2.0})
    pd.DataFrame({
        "file_id": ["file1", "file2"],
        "case_id": ["case1", "case1"],  # same case, two files
        "submitter_id": ["TCGA-01", "TCGA-01"],
        "vital_status": ["Alive", "Dead"],
        "days_to_death": [None, 100],
    }).to_csv(src / "outcome_meta.csv", index=False)

    csv_path = convert_tcga_vital_status("TCGA-TEST", src)
    assert csv_path is None  # both excluded as duplicates -> no usable samples


def test_convert_tcga_vital_status_excludes_missing_file(tmp_path, tmp_paths):
    src = tmp_path
    _write_star_file(src / "file1.tsv.gz", {"ENSG00000000003.15": 1.0})
    pd.DataFrame({
        "file_id": ["file1", "file2"],  # file2 never downloaded
        "case_id": ["case1", "case2"],
        "submitter_id": ["TCGA-01", "TCGA-02"],
        "vital_status": ["Alive", "Dead"],
        "days_to_death": [None, 50],
    }).to_csv(src / "outcome_meta.csv", index=False)

    csv_path = convert_tcga_vital_status("TCGA-TEST", src)
    meta = pd.read_csv(csv_path.with_name(csv_path.stem + "_samples_meta.csv"))
    assert list(meta["sample_id"]) == ["file1"]


def test_convert_tcga_vital_status_dead_encoded_as_one_alive_as_zero(tmp_path, tmp_paths):
    src = tmp_path
    _write_star_file(src / "file1.tsv.gz", {"ENSG00000000003.15": 1.0})
    _write_star_file(src / "file2.tsv.gz", {"ENSG00000000003.15": 2.0})
    pd.DataFrame({
        "file_id": ["file1", "file2"],
        "case_id": ["case1", "case2"],
        "submitter_id": ["TCGA-01", "TCGA-02"],
        "vital_status": ["Dead", "Alive"],
        "days_to_death": [365, None],
    }).to_csv(src / "outcome_meta.csv", index=False)

    csv_path = convert_tcga_vital_status("TCGA-TEST", src)
    meta = pd.read_csv(csv_path.with_name(csv_path.stem + "_samples_meta.csv")).set_index("sample_id")
    assert meta.loc["file1", "vital_status"] == 1.0
    assert meta.loc["file2", "vital_status"] == 0.0
    assert bool(meta.loc["file1", "subject_id_verified"])


def test_convert_tcga_vital_status_no_meta_returns_none(tmp_path, tmp_paths):
    assert convert_tcga_vital_status("TCGA-TEST", tmp_path) is None
