"""
Regression tests for PR7 requirement 7: ExperimentContext must reject both
missing and unexpected manifest subjects, reject blank/placeholder bag
subject IDs, and expose fingerprinted run identity that a checkpoint/result
reload can verify against.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.context import ExperimentContext
from benchmarks.runner import build_synthetic_context
from data.label_mapping import identity_label_mapping
from data.preprocessing import PreprocessingArtifact
from data.splitting import SplitManifest
from train import CellLevelDataset


def _base_result(n=20):
    all_ids = np.array([f"s{i}" for i in range(n)], dtype=object)
    ds = CellLevelDataset(
        np.random.rand(n, 5).astype("float32"), np.zeros(n, dtype=int),
        np.zeros(n, dtype="float32"), np.zeros(n, dtype=int), subject_ids=all_ids,
    )
    train_ids, val_ids, test_ids = all_ids[:14].tolist(), all_ids[14:17].tolist(), all_ids[17:].tolist()
    artifact = PreprocessingArtifact(
        version="1", gene_list=[f"g{i}" for i in range(5)],
        gene_means=[0.0] * 5, gene_stds=[1.0] * 5, n_hvgs=5,
        smoke_marker_genes_forced=[], fit_n_cells=14, fit_n_subjects=14,
    )
    manifest = SplitManifest(seed=1, train_subjects=train_ids, val_subjects=val_ids, test_subjects=test_ids)
    return dict(
        train_cell_dataset=ds.subset_by_subjects(train_ids),
        val_cell_dataset=ds.subset_by_subjects(val_ids),
        test_cell_dataset=ds.subset_by_subjects(test_ids),
        train_bags=[], val_bags=[], test_bags=[],
        split_manifest=manifest, preprocessing_artifact=artifact,
        label_mapping=identity_label_mapping(), rare_class_report={},
        label_provenance_report={}, transductive_batch_correction=False,
    ), train_ids, val_ids, test_ids


def test_rejects_manifest_subject_missing_from_cell_dataset():
    """Would have passed silently before the fix: only the reverse direction
    (dataset subject not in manifest) was checked."""
    result, train_ids, val_ids, test_ids = _base_result()
    manifest = SplitManifest(seed=1, train_subjects=train_ids + ["ghost_subject"],
                              val_subjects=val_ids, test_subjects=test_ids)
    result["split_manifest"] = manifest
    with pytest.raises(ValueError, match="absent from train_cell_dataset"):
        ExperimentContext.from_pipeline_result(result, config={})


def test_rejects_unexpected_subject_in_cell_dataset():
    result, train_ids, val_ids, test_ids = _base_result()
    manifest = SplitManifest(seed=1, train_subjects=train_ids[:-1],  # drop one
                              val_subjects=val_ids, test_subjects=test_ids)
    result["split_manifest"] = manifest
    with pytest.raises(ValueError, match="not present in split_manifest"):
        ExperimentContext.from_pipeline_result(result, config={})


@pytest.mark.parametrize("bad_id", ["", "unknown", "none", "nan"])
def test_rejects_blank_or_placeholder_bag_subject_id(bad_id):
    result, *_ = _base_result()
    result["train_bags"] = [{
        "subject_id": bad_id, "gene_matrix": np.zeros((5, 5), dtype="float32"),
        "cell_type_ids": np.zeros(5, dtype=int), "smoke_labels": np.zeros(5, dtype=int),
        "malig_labels": np.zeros(5, dtype="float32"), "malig_known": np.zeros(5, dtype=bool),
        "cancer_label": 0, "cancer_label_known": True,
    }]
    with pytest.raises(ValueError, match="blank/placeholder subject_id"):
        ExperimentContext.from_pipeline_result(result, config={})


def test_run_identity_round_trips_and_detects_mismatch():
    ctx = build_synthetic_context(seed=1, fast=True)
    identity = ctx.run_identity("run123")
    assert identity["run_id"] == "run123"
    assert identity["config_fingerprint"] == ctx.config_fingerprint
    assert identity["preprocessing_artifact_fingerprint"] == ctx.preprocessing_artifact_fingerprint
    assert identity["label_mapping_fingerprint"] == ctx.label_mapping_fingerprint

    # Round-trips cleanly against itself.
    ctx.validate_run_identity(identity)

    # A tampered config fingerprint must be detected.
    tampered = dict(identity, config_fingerprint="deadbeef")
    with pytest.raises(ValueError, match="config_fingerprint"):
        ctx.validate_run_identity(tampered)


def test_config_fingerprint_changes_when_config_content_changes():
    ctx1 = build_synthetic_context(seed=1, fast=True)
    ctx2 = build_synthetic_context(seed=1, fast=True)
    assert ctx1.config_fingerprint == ctx2.config_fingerprint  # same config -> same fingerprint
    import dataclasses
    ctx3 = dataclasses.replace(ctx2, config={**ctx2.config, "extra_marker": True})
    assert ctx3.config_fingerprint != ctx1.config_fingerprint
