"""
tests/test_evidence_publication.py — Issue #16 blocker 5: sanitized
evidence-run publication.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evidence.artifact_bundle import write_evidence_run
from evidence.errors import ArtifactValidationError
from evidence.evidence_contract import build_report
from evidence.publication import PublicationError, publish_evidence_run

FAKE_FP = "f" * 64


def _identity(**overrides):
    identity = {
        "task": "smoke_classification", "endpoint": "e", "endpoint_definition": "e",
        "prediction_unit": "subject", "biological_specimen_type": "airway_epithelium",
        "assay_modality": "bulk_microarray", "dataset_accession": "GSE123352",
        "cohort_role": "development", "species": "human", "sample_count": 10,
        "unique_subject_count": 10, "class_counts": {"a": 5, "b": 5},
        "verified_label_count": 10, "unknown_label_count": 0, "excluded_subject_count": 0,
        "split_role": "development_holdout", "split_manifest_fingerprint": FAKE_FP,
        "dataset_manifest_fingerprint": FAKE_FP, "preprocessing_artifact_fingerprint": FAKE_FP,
        "model_fingerprint": FAKE_FP, "random_seed": 0, "code_commit_sha": "0" * 40,
        "environment_fingerprint": FAKE_FP, "synthetic_flag": False,
        "development_only_flag": True, "frozen_test_access_status": "not_applicable",
        "evidence_level": "development_only_real_data", "limitations": ["test fixture"],
        "evaluation_timestamp": "2026-01-01T00:00:00Z", "report_schema_version": "1",
    }
    identity.update(overrides)
    return identity


def _write_run(tmp_path, *, include_split_manifest=True, subject_ids_in_split=("GSM1", "GSM2")):
    report = build_report(_identity(), {"macro_f1": 0.5, "per_class": {}}).to_dict()
    files = {
        "metrics/report.json": report,
        "predictions/test_predictions.csv": [{"subject_id": "GSM1", "y_true": 0, "y_pred": 0}],
    }
    if include_split_manifest:
        files["split_manifest.json"] = {
            "schema_version": "1", "task": "smoke_classification", "endpoint": "e",
            "dataset_accession": "GSE123352",
            "train_subject_ids": list(subject_ids_in_split),
            "validation_subject_ids": [], "development_holdout_subject_ids": ["GSM3"],
            "internal_test_subject_ids": [], "external_cohort_accession": None,
            "external_subject_ids": [], "seed": 42, "train_frac": 0.7, "val_frac": 0.0,
            "test_frac": 0.3, "stratification_policy": "x", "class_mapping": {"a": 0, "b": 1},
            "rare_class_policy": "x", "label_state_policy": "x", "weak_label_policy": "x",
            "subject_identity_field": "subject_id", "grouping_policy": "x",
            "per_partition_class_counts": {}, "dataset_manifest_fingerprint": FAKE_FP,
        }
    root = tmp_path / "artifacts" / "evidence"
    write_evidence_run(root, "run1", files, extra_manifest_fields={"dataset_accession": "GSE123352"})
    return root


def test_publish_produces_sanitized_summary(tmp_path):
    private_root = _write_run(tmp_path)
    published_root = tmp_path / "published"
    summary = publish_evidence_run(
        private_root, "run1", report_relative_path="metrics/report.json",
        published_root=published_root,
    )
    assert summary["dataset_accession"] == "GSE123352"
    assert summary["metrics"]["macro_f1"] == 0.5
    assert (published_root / "run1" / "summary.json").exists()


def test_publish_excludes_subject_and_sample_ids(tmp_path):
    private_root = _write_run(tmp_path)
    published_root = tmp_path / "published"
    summary = publish_evidence_run(
        private_root, "run1", report_relative_path="metrics/report.json",
        published_root=published_root,
    )
    blob = json.dumps(summary)
    assert "GSM1" not in blob and "GSM2" not in blob and "GSM3" not in blob
    assert "train_subject_ids" not in summary["split_configuration"]
    assert "development_holdout_subject_ids" not in summary["split_configuration"]


def test_publish_split_configuration_has_counts_not_raw_ids(tmp_path):
    private_root = _write_run(tmp_path)
    published_root = tmp_path / "published"
    summary = publish_evidence_run(
        private_root, "run1", report_relative_path="metrics/report.json",
        published_root=published_root,
    )
    split_cfg = summary["split_configuration"]
    assert split_cfg["train_subject_ids_count"] == 2
    assert "train_subject_ids" not in split_cfg
    assert "fingerprint" in split_cfg


def test_publish_binds_to_private_bundle_fingerprint(tmp_path):
    private_root = _write_run(tmp_path)
    published_root = tmp_path / "published"
    summary = publish_evidence_run(
        private_root, "run1", report_relative_path="metrics/report.json",
        published_root=published_root,
    )
    assert len(summary["source_private_bundle_fingerprint"]) == 64


def test_publish_fails_closed_on_tampered_private_bundle(tmp_path):
    private_root = _write_run(tmp_path)
    (private_root / "run1" / "metrics" / "report.json").write_text('{"tampered": true}')
    with pytest.raises(ArtifactValidationError):
        publish_evidence_run(
            private_root, "run1", report_relative_path="metrics/report.json",
            published_root=tmp_path / "published",
        )


def test_publish_fails_on_missing_component(tmp_path):
    private_root = _write_run(tmp_path)
    with pytest.raises(PublicationError):
        publish_evidence_run(
            private_root, "run1", report_relative_path="metrics/does_not_exist.json",
            published_root=tmp_path / "published",
        )


def test_publish_reports_are_checksummed_and_reloadable(tmp_path):
    private_root = _write_run(tmp_path)
    published_root = tmp_path / "published"
    summary = publish_evidence_run(
        private_root, "run1", report_relative_path="metrics/report.json",
        published_root=published_root,
    )
    with open(published_root / "run1" / "summary.json") as f:
        reloaded = json.load(f)
    assert reloaded == summary


def test_hand_tampered_summary_checksum_is_detectably_wrong(tmp_path):
    from evidence.publication import _content_checksum

    private_root = _write_run(tmp_path)
    published_root = tmp_path / "published"
    publish_evidence_run(
        private_root, "run1", report_relative_path="metrics/report.json",
        published_root=published_root,
    )
    summary_path = published_root / "run1" / "summary.json"
    data = json.loads(summary_path.read_text())
    data["metrics"]["macro_f1"] = 0.999  # tamper without recomputing checksum_sha256
    stored_checksum = data.pop("checksum_sha256")
    recomputed = _content_checksum(data)
    assert recomputed != stored_checksum


def test_republishing_recomputes_checksum_matching_current_content(tmp_path):
    private_root = _write_run(tmp_path)
    published_root = tmp_path / "published"
    summary1 = publish_evidence_run(
        private_root, "run1", report_relative_path="metrics/report.json",
        published_root=published_root,
    )
    summary2 = publish_evidence_run(
        private_root, "run1", report_relative_path="metrics/report.json",
        published_root=published_root,
    )
    assert summary1 == summary2
