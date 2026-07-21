"""
tests/test_artifact_bundle.py — Phase 7 Step 15 tests: the immutable
evidence-run artifact directory writer/reader.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evidence.artifact_bundle import (
    RunAlreadyCompleteError,
    list_run_ids,
    read_manifest,
    validate_evidence_run,
    write_evidence_run,
)
from evidence.errors import ArtifactBundleError, ArtifactValidationError


@pytest.fixture
def tmp_root(tmp_path):
    return tmp_path / "artifacts" / "evidence"


def _sample_files():
    return {
        "configuration.json": {"config": "value"},
        "data_audit.json": {"status": "audited"},
        "metrics/development.json": {"macro_f1": 0.5},
        "reports/evidence_report.md": "# Evidence report\n\nnot_evaluable.\n",
        "predictions/development_oof.csv": [
            {"subject_id": "s1", "y_true": 0, "y_prob": 0.2},
            {"subject_id": "s2", "y_true": 1, "y_prob": 0.8},
        ],
    }


def test_write_then_validate_round_trips(tmp_root):
    write_evidence_run(tmp_root, "run_a", _sample_files(), run_status="complete")
    manifest = validate_evidence_run(tmp_root, "run_a")
    assert manifest["run_status"] == "complete"
    assert set(manifest["files"]) == set(_sample_files())


def test_only_provided_files_are_written(tmp_root):
    files = {"configuration.json": {"x": 1}}
    run_dir = write_evidence_run(tmp_root, "run_devonly", files, run_status="incomplete")
    assert (run_dir / "configuration.json").exists()
    assert not (run_dir / "predictions" / "internal_test.csv").exists()
    manifest = read_manifest(tmp_root, "run_devonly")
    assert manifest["run_status"] == "incomplete"
    assert list(manifest["files"]) == ["configuration.json"]


def test_checksums_json_written_for_every_file(tmp_root):
    files = _sample_files()
    run_dir = write_evidence_run(tmp_root, "run_b", files)
    with open(run_dir / "checksums.json") as f:
        checksums = json.load(f)
    assert set(checksums) == set(files)
    for digest in checksums.values():
        assert len(digest) == 64


def test_manifest_written_last_binds_every_file_and_checksum(tmp_root):
    files = _sample_files()
    run_dir = write_evidence_run(tmp_root, "run_c", files)
    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    for rel_path in files:
        assert rel_path in manifest["files"]
        assert "sha256" in manifest["files"][rel_path]
    assert manifest["run_status"] == "complete"


def test_completed_run_cannot_be_written_again(tmp_root):
    write_evidence_run(tmp_root, "run_d", _sample_files(), run_status="complete")
    with pytest.raises(RunAlreadyCompleteError):
        write_evidence_run(tmp_root, "run_d", {"configuration.json": {"x": 2}})


def test_incomplete_run_can_be_written_again(tmp_root):
    write_evidence_run(tmp_root, "run_e", {"configuration.json": {"x": 1}}, run_status="incomplete")
    # Retrying/continuing an incomplete run is permitted — only a
    # 'complete' manifest makes a run directory immutable.
    write_evidence_run(tmp_root, "run_e", {"configuration.json": {"x": 2}}, run_status="complete")
    manifest = read_manifest(tmp_root, "run_e")
    assert manifest["run_status"] == "complete"


def test_validate_raises_on_missing_file(tmp_root):
    run_dir = write_evidence_run(tmp_root, "run_f", _sample_files())
    (run_dir / "configuration.json").unlink()
    with pytest.raises(ArtifactValidationError):
        validate_evidence_run(tmp_root, "run_f")


def test_validate_raises_on_checksum_mismatch(tmp_root):
    run_dir = write_evidence_run(tmp_root, "run_g", _sample_files())
    (run_dir / "configuration.json").write_text('{"config": "TAMPERED"}')
    with pytest.raises(ArtifactValidationError):
        validate_evidence_run(tmp_root, "run_g")


def test_validate_raises_on_missing_run(tmp_root):
    with pytest.raises(ArtifactValidationError):
        validate_evidence_run(tmp_root, "does_not_exist")


def test_write_rejects_path_traversal(tmp_root):
    with pytest.raises(ArtifactBundleError):
        write_evidence_run(tmp_root, "run_h", {"../escape.json": {"x": 1}})


def test_write_rejects_reserved_relative_path(tmp_root):
    with pytest.raises(ArtifactBundleError):
        write_evidence_run(tmp_root, "run_i", {"manifest.json": {"x": 1}})
    with pytest.raises(ArtifactBundleError):
        write_evidence_run(tmp_root, "run_i2", {"checksums.json": {"x": 1}})


def test_write_rejects_unsupported_extension(tmp_root):
    with pytest.raises(ArtifactBundleError):
        write_evidence_run(tmp_root, "run_j", {"model.bin": b"\x00\x01"})


def test_write_rejects_wrong_content_type_for_json(tmp_root):
    with pytest.raises(ArtifactBundleError):
        write_evidence_run(tmp_root, "run_k", {"configuration.json": "not a dict"})


def test_write_rejects_wrong_content_type_for_csv(tmp_root):
    with pytest.raises(ArtifactBundleError):
        write_evidence_run(tmp_root, "run_l", {"predictions/development_oof.csv": {"not": "a list"}})


def test_list_run_ids_read_only(tmp_root):
    write_evidence_run(tmp_root, "run_m1", {"configuration.json": {"x": 1}})
    write_evidence_run(tmp_root, "run_m2", {"configuration.json": {"x": 1}})
    ids = list_run_ids(tmp_root)
    assert ids == ["run_m1", "run_m2"]


def test_list_run_ids_empty_when_root_absent(tmp_path):
    assert list_run_ids(tmp_path / "nonexistent") == []


def test_invalid_run_status_rejected(tmp_root):
    with pytest.raises(ArtifactBundleError):
        write_evidence_run(tmp_root, "run_n", {"configuration.json": {"x": 1}}, run_status="bogus")
