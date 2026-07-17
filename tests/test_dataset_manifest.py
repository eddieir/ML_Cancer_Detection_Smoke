"""
tests/test_dataset_manifest.py — versioned dataset provenance manifest.
Never fabricates a checksum for a file that isn't actually on disk.
"""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.manifest import (
    DatasetManifestEntry,
    ManifestValidationError,
    build_dataset_manifest,
    checksum_existing_files,
    manifest_fingerprint,
    save_manifest,
    sha256_of_file,
)


def _minimal_entry(**overrides) -> DatasetManifestEntry:
    base = dict(
        dataset_id="fixture_ds", accession="GSEFIXTURE",
        source_url="https://example.invalid/fixture",
        official_record_url="https://example.invalid/fixture",
        species="human", assay_type="single_cell_rna_seq",
        matrix_representation="sparse_mtx_triplet",
        subject_identifier_field="subject_id",
        license_or_access_level="public_geo", controlled_access=False,
        synthetic_or_real="synthetic_fixture",
    )
    base.update(overrides)
    return DatasetManifestEntry(**base)


def test_entry_validates_with_all_required_fields():
    _minimal_entry().validate()  # must not raise


def test_entry_missing_required_field_raises():
    entry = _minimal_entry()
    entry.species = ""
    with pytest.raises(ManifestValidationError):
        entry.validate()


def test_entry_rejects_checksum_without_files_present():
    entry = _minimal_entry(raw_file_checksums={"foo.txt": "a" * 64}, files_present=False)
    with pytest.raises(ManifestValidationError):
        entry.validate()


def test_controlled_access_false_is_not_treated_as_missing_field():
    """controlled_access=False is a legitimate, present value — validate()
    must not mistake Python falsiness for a missing field."""
    _minimal_entry(controlled_access=False).validate()


def test_sha256_of_file_matches_known_hash(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"hello world")
    import hashlib
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert sha256_of_file(p) == expected


def test_checksum_existing_files_never_fabricates_for_missing_file(tmp_path):
    (tmp_path / "present.txt").write_bytes(b"data")
    result = checksum_existing_files(tmp_path, ["present.txt", "absent.txt"])
    assert result["present.txt"] is not None
    assert result["absent.txt"] is None


def test_build_dataset_manifest_from_seed_with_no_local_files(tmp_path):
    seed = {
        "datasets": [
            {
                "dataset_id": "fixture_ds", "accession": "GSEFIXTURE",
                "raw_subdir": "does/not/exist",
                "source_url": "https://example.invalid/fixture",
                "official_record_url": "https://example.invalid/fixture",
                "species": "human", "assay_type": "single_cell_rna_seq",
                "matrix_representation": "sparse_mtx_triplet",
                "subject_identifier_field": "subject_id",
                "license_or_access_level": "public_geo", "controlled_access": False,
                "synthetic_or_real": "synthetic_fixture",
                "raw_file_names": ["a.mtx.gz"],
            }
        ]
    }
    seed_path = tmp_path / "datasets.yaml"
    with open(seed_path, "w") as f:
        yaml.safe_dump(seed, f)

    entries = build_dataset_manifest(seed_path, raw_root=tmp_path / "raw_root")
    assert len(entries) == 1
    assert entries[0].files_present is False
    assert entries[0].raw_file_checksums == {"a.mtx.gz": None}


def test_build_dataset_manifest_computes_real_checksum_when_file_present(tmp_path):
    raw_root = tmp_path / "raw_root"
    src_dir = raw_root / "cigarette" / "FIXTURE"
    src_dir.mkdir(parents=True)
    (src_dir / "a.mtx.gz").write_bytes(b"fake matrix bytes")

    seed = {
        "datasets": [
            {
                "dataset_id": "fixture_ds", "accession": "GSEFIXTURE",
                "raw_subdir": "cigarette/FIXTURE",
                "source_url": "https://example.invalid/fixture",
                "official_record_url": "https://example.invalid/fixture",
                "species": "human", "assay_type": "single_cell_rna_seq",
                "matrix_representation": "sparse_mtx_triplet",
                "subject_identifier_field": "subject_id",
                "license_or_access_level": "public_geo", "controlled_access": False,
                "synthetic_or_real": "synthetic_fixture",
                "raw_file_names": ["a.mtx.gz"],
            }
        ]
    }
    seed_path = tmp_path / "datasets.yaml"
    with open(seed_path, "w") as f:
        yaml.safe_dump(seed, f)

    entries = build_dataset_manifest(seed_path, raw_root=raw_root)
    assert entries[0].files_present is True
    assert entries[0].raw_file_checksums["a.mtx.gz"] == sha256_of_file(src_dir / "a.mtx.gz")


def test_manifest_fingerprint_changes_when_an_entry_changes():
    e1 = _minimal_entry()
    e2 = _minimal_entry()
    fp_same = manifest_fingerprint([e1]) == manifest_fingerprint([e2])
    assert fp_same  # identical content -> identical fingerprint

    e3 = _minimal_entry(label_policy_version="2")
    assert manifest_fingerprint([e1]) != manifest_fingerprint([e3])


def test_manifest_fingerprint_ignores_download_date_only():
    e1 = _minimal_entry(download_date="2024-01-01")
    e2 = _minimal_entry(download_date="2024-06-01")
    assert manifest_fingerprint([e1]) == manifest_fingerprint([e2])


def test_save_manifest_writes_atomic_valid_json(tmp_path):
    entry = _minimal_entry()
    out_path = tmp_path / "dataset_manifest.json"
    fp = save_manifest([entry], out_path)
    assert out_path.exists()
    import json
    with open(out_path) as f:
        payload = json.load(f)
    assert payload["fingerprint"] == fp
    assert payload["datasets"][0]["dataset_id"] == "fixture_ds"


def test_real_datasets_yaml_seed_builds_and_validates_cleanly():
    """The checked-in configs/datasets.yaml must itself build into a valid
    manifest (no files present in this environment/CI, so every entry is
    metadata-only — files_present=False, checksums all None — but every
    entry must still carry complete, valid provenance)."""
    repo_root = Path(__file__).parents[1]
    seed_path = repo_root / "configs" / "datasets.yaml"
    entries = build_dataset_manifest(seed_path, raw_root=repo_root / "data" / "raw")
    assert len(entries) >= 7
    ids = {e.dataset_id for e in entries}
    assert {"gse136831", "gse288003", "gse123352", "tcga_luad", "tcga_lusc", "nlst"} <= ids
    for e in entries:
        e.validate()  # must not raise for any real seed entry
