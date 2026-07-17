"""tests/test_label_quality_report.py — machine-readable label-quality report."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import ALL_SMOKE_MARKERS


def _synthetic_h5ad_consistent_labels(path, n_subjects=12, cells_per_subject=30, g=500, n_classes=3):
    genes = [f"G{i}" for i in range(g)]
    genes[:5] = [f"MT-{i}" for i in range(5)]
    for i, mk in enumerate(ALL_SMOKE_MARKERS[:4]):
        genes[50 + i] = mk

    subject_ids, smoke_types = [], []
    for i in range(n_subjects):
        subject_ids += [f"sub_{i}"] * cells_per_subject
        smoke_types  += [i % n_classes] * cells_per_subject
    n = len(subject_ids)

    raw = np.random.negative_binomial(5, 0.7, (n, g)).astype("float32")
    obs = pd.DataFrame({
        "donor_id":        subject_ids,
        "subject_id":      subject_ids,
        "smoke_type":      smoke_types,
        "smoke_type_name": "mixed",
        "data_modality":   "scrna",
        "is_pseudo_bulk":  False,
        "malignancy":      0.0,
        "cell_type_id":    0,
    }, index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=sp.csr_matrix(raw), obs=obs, var=pd.DataFrame(index=genes))
    adata.write_h5ad(path)


def _run_pipeline(tmp):
    from preprocess import run_pipeline_split_aware
    h5ad = str(Path(tmp) / "test.h5ad")
    _synthetic_h5ad_consistent_labels(h5ad)
    return run_pipeline_split_aware({
        "data": {
            "scrna_sources": [(h5ad, "cigarette", "donor_id")],
            "n_hvgs": 50,
            "min_cells_per_subject": 5,
            "out_dir": str(Path(tmp) / "processed"),
            "cell_type_allow_diagnostic_fallback": True,
        },
        "split": {"seed": 1, "train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2},
    })


def test_report_flags_all_unknown_cancer_outcomes_when_no_outcomes_configured():
    from data.label_quality_report import build_label_quality_report
    with tempfile.TemporaryDirectory() as tmp:
        result = _run_pipeline(tmp)
        report = build_label_quality_report(result)
        assert any("no_subjects_with_known_cancer_outcome" in f for f in report.flags)
        assert report.has_critical_violations is True


def test_report_per_split_counts_match_split_manifest_report():
    from data.label_quality_report import build_label_quality_report
    with tempfile.TemporaryDirectory() as tmp:
        result = _run_pipeline(tmp)
        report = build_label_quality_report(result)
        manifest_report = result["split_manifest"].report["splits"]
        for split_name, s in manifest_report.items():
            assert report.per_split[split_name]["n_subjects"] == s["n_subjects"]
            assert report.per_split[split_name]["n_cells"] == s["n_cells"]


def test_report_save_json_and_csv(tmp_path):
    from data.label_quality_report import build_label_quality_report
    with tempfile.TemporaryDirectory() as tmp:
        result = _run_pipeline(tmp)
        report = build_label_quality_report(result)
        json_path = tmp_path / "report.json"
        csv_path = tmp_path / "summary.csv"
        report.save_json(json_path)
        report.save_csv_summary(csv_path)
        assert json_path.exists() and csv_path.exists()
        with open(json_path) as f:
            payload = json.load(f)
        assert payload["schema_version"] == "1"
        assert "flags" in payload
