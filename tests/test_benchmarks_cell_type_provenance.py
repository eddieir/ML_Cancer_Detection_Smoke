"""
Regression tests for PR7-5 Issue 3: strict, fail-closed cell-type
annotation provenance validation. The historical check
(`getattr(artifact, "cell_type_annotation_degraded", False)`) treated a
MISSING provenance field as "not degraded" / safe — these tests prove that
gap is closed: every one of the three provenance fields must be explicitly
present and valid before a real ExperimentContext will build.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.context import ExperimentContext
from benchmarks.runner import build_synthetic_context
from data.label_mapping import identity_label_mapping
from data.preprocessing import (
    CellTypeProvenanceError,
    PreprocessingArtifact,
    validate_cell_type_provenance,
)
from data.splitting import SplitManifest
from data.transforms import cell_type_map_fingerprint
from train import CellLevelDataset

_CURRENT_FP = cell_type_map_fingerprint()


def _artifact(**overrides) -> PreprocessingArtifact:
    kwargs = dict(
        version="1", gene_list=["g0", "g1"], gene_means=[0.0, 0.0], gene_stds=[1.0, 1.0],
        n_hvgs=2, smoke_marker_genes_forced=[], fit_n_cells=10, fit_n_subjects=10,
        # Valid assay-policy provenance by default (see data/assay_policy.py,
        # GitHub issue #13) — this file's own tests are about cell-type
        # provenance specifically; assay provenance defaults to a valid,
        # non-degraded state so it never masks what these tests actually
        # exercise, unless a test overrides it on purpose.
        assay_policy="single_cell_only", assay_policy_version="1",
        observed_assay_modes=["human_single_cell"],
        pseudo_bulk_rows_present_at_fit=False,
        training_data_modality="single_cell", allowed_inference_modality="single_cell",
    )
    kwargs.update(overrides)
    return PreprocessingArtifact(**kwargs)


# --- unit-level: validate_cell_type_provenance directly ------------------


def test_valid_inductive_provenance_passes():
    validate_cell_type_provenance(_artifact(
        cell_type_annotation_degraded=False,
        cell_type_annotation_mode="inductive_per_cell",
        cell_type_map_fingerprint=_CURRENT_FP,
    ))  # must not raise


def test_valid_pseudo_bulk_provenance_passes():
    validate_cell_type_provenance(_artifact(
        cell_type_annotation_degraded=False,
        cell_type_annotation_mode="pseudo_bulk_no_cell_type_identity",
        cell_type_map_fingerprint=None,
    ))  # must not raise


def test_missing_degraded_flag_fails():
    """The core regression: an artifact with NO provenance fields set at
    all (all default None) must be REJECTED, not silently accepted."""
    with pytest.raises(CellTypeProvenanceError, match="degraded"):
        validate_cell_type_provenance(_artifact())


def test_degraded_true_fails():
    with pytest.raises(CellTypeProvenanceError, match="degraded"):
        validate_cell_type_provenance(_artifact(
            cell_type_annotation_degraded=True,
            cell_type_annotation_mode="diagnostic_fallback",
            cell_type_map_fingerprint=_CURRENT_FP,
        ))


def test_degraded_none_fails():
    with pytest.raises(CellTypeProvenanceError, match="degraded"):
        validate_cell_type_provenance(_artifact(
            cell_type_annotation_degraded=None,
            cell_type_annotation_mode="inductive_per_cell",
            cell_type_map_fingerprint=_CURRENT_FP,
        ))


def test_missing_mode_fails():
    with pytest.raises(CellTypeProvenanceError, match="mode"):
        validate_cell_type_provenance(_artifact(
            cell_type_annotation_degraded=False,
            cell_type_annotation_mode=None,
            cell_type_map_fingerprint=_CURRENT_FP,
        ))


def test_majority_voting_mode_fails():
    with pytest.raises(CellTypeProvenanceError, match="mode"):
        validate_cell_type_provenance(_artifact(
            cell_type_annotation_degraded=False,
            cell_type_annotation_mode="majority_voting",
            cell_type_map_fingerprint=_CURRENT_FP,
        ))


def test_diagnostic_fallback_mode_fails():
    with pytest.raises(CellTypeProvenanceError, match="mode"):
        validate_cell_type_provenance(_artifact(
            cell_type_annotation_degraded=False,  # even if degraded were somehow miscoded False
            cell_type_annotation_mode="diagnostic_fallback",
            cell_type_map_fingerprint=_CURRENT_FP,
        ))


def test_arbitrary_unexpected_mode_fails():
    with pytest.raises(CellTypeProvenanceError, match="mode"):
        validate_cell_type_provenance(_artifact(
            cell_type_annotation_degraded=False,
            cell_type_annotation_mode="some_future_mode_nobody_registered",
            cell_type_map_fingerprint=_CURRENT_FP,
        ))


def test_missing_map_fingerprint_fails_for_inductive_mode():
    with pytest.raises(CellTypeProvenanceError, match="fingerprint"):
        validate_cell_type_provenance(_artifact(
            cell_type_annotation_degraded=False,
            cell_type_annotation_mode="inductive_per_cell",
            cell_type_map_fingerprint=None,
        ))


def test_malformed_fingerprint_fails():
    with pytest.raises(CellTypeProvenanceError, match="fingerprint"):
        validate_cell_type_provenance(_artifact(
            cell_type_annotation_degraded=False,
            cell_type_annotation_mode="inductive_per_cell",
            cell_type_map_fingerprint="not-a-valid-hex-digest",
        ))


def test_blank_fingerprint_fails():
    with pytest.raises(CellTypeProvenanceError, match="fingerprint"):
        validate_cell_type_provenance(_artifact(
            cell_type_annotation_degraded=False,
            cell_type_annotation_mode="inductive_per_cell",
            cell_type_map_fingerprint="",
        ))


def test_stale_mismatched_fingerprint_fails():
    stale_but_well_formed = "0" * 64
    assert stale_but_well_formed != _CURRENT_FP
    with pytest.raises(CellTypeProvenanceError, match="does not match"):
        validate_cell_type_provenance(_artifact(
            cell_type_annotation_degraded=False,
            cell_type_annotation_mode="inductive_per_cell",
            cell_type_map_fingerprint=stale_but_well_formed,
        ))


def test_pseudo_bulk_mode_with_a_fingerprint_set_fails():
    """The pseudo-bulk exemption means NO fingerprint applies — a
    fingerprint accidentally present alongside this mode is itself a
    provenance inconsistency, not something to silently ignore."""
    with pytest.raises(CellTypeProvenanceError, match="must be None"):
        validate_cell_type_provenance(_artifact(
            cell_type_annotation_degraded=False,
            cell_type_annotation_mode="pseudo_bulk_no_cell_type_identity",
            cell_type_map_fingerprint=_CURRENT_FP,
        ))


def test_round_trip_serialization_preserves_provenance():
    artifact = _artifact(
        cell_type_annotation_degraded=False,
        cell_type_annotation_mode="inductive_per_cell",
        cell_type_map_fingerprint=_CURRENT_FP,
    )
    d = artifact.to_dict()
    reloaded = PreprocessingArtifact(**d)
    assert reloaded.cell_type_annotation_degraded is False
    assert reloaded.cell_type_annotation_mode == "inductive_per_cell"
    assert reloaded.cell_type_map_fingerprint == _CURRENT_FP
    validate_cell_type_provenance(reloaded)  # must not raise


# --- integration-level: wired through ExperimentContext.from_pipeline_result


def _pipeline_result(artifact: PreprocessingArtifact, n=20) -> dict:
    all_ids = np.array([f"s{i}" for i in range(n)], dtype=object)
    ds = CellLevelDataset(
        np.random.rand(n, 2).astype("float32"), np.zeros(n, dtype=int),
        np.zeros(n, dtype="float32"), np.zeros(n, dtype=int), subject_ids=all_ids,
    )
    train_ids, val_ids, test_ids = all_ids[:14].tolist(), all_ids[14:17].tolist(), all_ids[17:].tolist()
    manifest = SplitManifest(seed=1, train_subjects=train_ids, val_subjects=val_ids, test_subjects=test_ids)
    return dict(
        train_cell_dataset=ds.subset_by_subjects(train_ids),
        val_cell_dataset=ds.subset_by_subjects(val_ids),
        test_cell_dataset=ds.subset_by_subjects(test_ids),
        train_bags=[], val_bags=[], test_bags=[],
        split_manifest=manifest, preprocessing_artifact=artifact,
        label_mapping=identity_label_mapping(), rare_class_report={},
        label_provenance_report={}, transductive_batch_correction=False,
    )


def test_from_pipeline_result_accepts_valid_inductive_provenance():
    artifact = _artifact(
        gene_list=["g0", "g1"], gene_means=[0.0, 0.0], gene_stds=[1.0, 1.0], n_hvgs=2,
        cell_type_annotation_degraded=False,
        cell_type_annotation_mode="inductive_per_cell",
        cell_type_map_fingerprint=_CURRENT_FP,
    )
    ctx = ExperimentContext.from_pipeline_result(_pipeline_result(artifact), config={})
    assert ctx is not None


def test_from_pipeline_result_rejects_missing_provenance():
    artifact = _artifact(gene_list=["g0", "g1"], gene_means=[0.0, 0.0], gene_stds=[1.0, 1.0], n_hvgs=2)
    with pytest.raises(CellTypeProvenanceError):
        ExperimentContext.from_pipeline_result(_pipeline_result(artifact), config={})


def test_explicitly_synthetic_context_is_not_subject_to_this_validation():
    """build_synthetic_context never calls from_pipeline_result / this
    validation at all — proving the strict check does not weaken the
    explicit, clearly-labelled synthetic construction path."""
    ctx = build_synthetic_context(seed=1, fast=True)
    assert ctx is not None
