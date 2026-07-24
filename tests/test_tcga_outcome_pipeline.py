"""
tests/test_tcga_outcome_pipeline.py — adversarial coverage for
data/tcga_outcome_pipeline.py, which combines TCGA-LUAD/TCGA-LUSC
vital-status outputs on their shared gene set for the real, subject-level
cancer-outcome-prediction task.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.tcga_outcome_pipeline import (
    TCGAOutcomePipelineError,
    build_tcga_outcome_dataset,
    split_tcga_outcome_subjects,
)


def _write_project(tmp_path, name, genes, samples):
    """samples: list of (sample_id, subject_id, verified, vital_known, vital, values)"""
    expr = pd.DataFrame({s[0]: s[5] for s in samples}, index=genes)
    csv_path = tmp_path / f"{name}.csv"
    expr.to_csv(csv_path)
    pd.DataFrame({
        "sample_id": [s[0] for s in samples],
        "subject_id": [s[1] for s in samples],
        "subject_id_verified": [s[2] for s in samples],
        "vital_status_known": [s[3] for s in samples],
        "vital_status": [s[4] for s in samples],
    }).to_csv(tmp_path / f"{name}_samples_meta.csv", index=False)
    return csv_path


def test_combines_two_projects_on_shared_genes(tmp_path):
    genes_a = ["G1", "G2", "G3"]
    genes_b = ["G2", "G3", "G4"]
    p1 = _write_project(tmp_path, "PROJ_A", genes_a, [
        ("s1", "case1", True, True, 0, [1.0, 2.0, 3.0]),
        ("s2", "case2", True, True, 1, [4.0, 5.0, 6.0]),
    ])
    p2 = _write_project(tmp_path, "PROJ_B", genes_b, [
        ("s3", "case3", True, True, 0, [7.0, 8.0, 9.0]),
    ])
    ds = build_tcga_outcome_dataset([p1, p2])
    assert set(ds.gene_names) == {"G2", "G3"}
    assert sorted(ds.subject_ids) == ["case1", "case2", "case3"]
    assert ds.project_of_subject["case1"] == "PROJ_A"
    assert ds.project_of_subject["case3"] == "PROJ_B"


def test_excludes_unverified_subject(tmp_path):
    genes = ["G1", "G2"]
    p1 = _write_project(tmp_path, "PROJ_A", genes, [
        ("s1", "case1", True, True, 0, [1.0, 2.0]),
        ("s2", "case2", False, True, 1, [3.0, 4.0]),  # unverified
    ])
    ds = build_tcga_outcome_dataset([p1])
    assert ds.subject_ids == ["case1"]
    assert ds.excluded_reason_counts.get("unverified_or_unknown") == 1


def test_excludes_unknown_vital_status(tmp_path):
    genes = ["G1", "G2"]
    p1 = _write_project(tmp_path, "PROJ_A", genes, [
        ("s1", "case1", True, True, 0, [1.0, 2.0]),
        ("s2", "case2", True, False, 1, [3.0, 4.0]),  # vital status unknown
    ])
    ds = build_tcga_outcome_dataset([p1])
    assert ds.subject_ids == ["case1"]


def test_excludes_duplicate_subject_across_projects(tmp_path):
    genes = ["G1", "G2"]
    p1 = _write_project(tmp_path, "PROJ_A", genes, [
        ("s1", "case1", True, True, 0, [1.0, 2.0]),
    ])
    p2 = _write_project(tmp_path, "PROJ_B", genes, [
        ("s2", "case1", True, True, 1, [3.0, 4.0]),  # same case_id, different project
    ])
    ds = build_tcga_outcome_dataset([p1, p2])
    assert ds.subject_ids == ["case1"]  # only the first occurrence kept
    assert ds.excluded_reason_counts.get("duplicate_subject_across_projects") == 1


def test_no_shared_genes_raises(tmp_path):
    p1 = _write_project(tmp_path, "PROJ_A", ["G1"], [("s1", "case1", True, True, 0, [1.0])])
    p2 = _write_project(tmp_path, "PROJ_B", ["G2"], [("s2", "case2", True, True, 1, [2.0])])
    with pytest.raises(TCGAOutcomePipelineError):
        build_tcga_outcome_dataset([p1, p2])


def test_split_is_subject_level_and_disjoint(tmp_path):
    genes = ["G1", "G2"]
    samples = [(f"s{i}", f"case{i}", True, True, i % 2, [float(i), float(i + 1)]) for i in range(20)]
    p1 = _write_project(tmp_path, "PROJ_A", genes, samples)
    ds = build_tcga_outcome_dataset([p1])
    split = split_tcga_outcome_subjects(ds, seed=1, train_frac=0.7, test_frac=0.3)
    assert set(split.train_subjects).isdisjoint(set(split.test_subjects))
    assert (set(split.train_subjects) | set(split.test_subjects)) <= set(ds.subject_ids)
    assert len(split.train_subjects) + len(split.test_subjects) >= 0.9 * len(ds.subject_ids)
