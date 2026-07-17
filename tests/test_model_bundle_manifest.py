"""
Phase 4 — full model-bundle manifest (Step 10).

Covers adversarial items 33-39:
  33. model loads with its matching artifact
  34. model rejects another fold's artifact
  35. model rejects an altered gene list
  36. model rejects an incompatible class vocabulary
  37. model rejects a corrupt artifact
  38. legacy checkpoint behavior follows explicit policy
  39. resume logic validates identity before reuse
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.bundle import (
    BundleCorruptionError,
    BundleValidationError,
    LegacyBundleError,
    load_and_validate_bundle,
    validate_bundle_for_model,
    validate_bundle_matches_identity,
    write_model_bundle,
)
from data.preprocessing import fit_preprocessing
from model import MultiSmokeCancerNet

GENES = 10


def _artifact(seed=0, n_genes=GENES, n_subjects=6):
    rng = np.random.default_rng(seed)
    subject_ids = []
    for i in range(n_subjects):
        subject_ids += [f"sub_{i}"] * 10
    n = len(subject_ids)
    X = rng.random((n, n_genes)).astype("float32")
    genes = [f"G{i}" for i in range(n_genes)]
    obs = pd.DataFrame({"subject_id": subject_ids, "batch": ["b0"] * n}, index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    return fit_preprocessing(adata, set(subject_ids), n_hvgs=n_genes)


def _fake_checkpoint(path):
    torch.save({"model_state_dict": {"w": torch.zeros(2)}}, path)


def _write_bundle(tmp_path, artifact=None, class_vocabulary=None):
    artifact = artifact or _artifact(seed=1)
    ckpt_path = tmp_path / "phase3_best.pt"
    _fake_checkpoint(ckpt_path)
    return write_model_bundle(
        tmp_path / "bundle", ckpt_path, artifact,
        model_config={"input_dim": len(artifact.gene_list), "num_smoke_types": 6},
        class_vocabulary=class_vocabulary or ["cigarette", "vape", "cannabis", "dual_use", "cigar", "unexposed"],
        label_policy="verified_only", species_policy="human_only", assay_mode="human_single_cell",
        dataset_manifest_fingerprint="dmfp123", split_fingerprint="splitfp456",
    )


# ─── Item 33: model loads with its matching artifact ───────────────────────

def test_model_loads_with_matching_artifact(tmp_path):
    bundle_dir = _write_bundle(tmp_path)
    manifest = load_and_validate_bundle(bundle_dir)
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, num_smoke=6,
                                 num_cell_types=4, attention_dim=4)
    validate_bundle_for_model(manifest, model)  # must not raise


# ─── Item 34: model rejects another fold's artifact ────────────────────────

def test_rejects_another_folds_artifact_swapped_in(tmp_path):
    bundle_dir = _write_bundle(tmp_path)
    other_artifact = _artifact(seed=2)  # different fit -> different fingerprint, same gene count
    other_artifact.save(bundle_dir / "preprocessing_artifact.json")
    with pytest.raises(BundleValidationError):
        load_and_validate_bundle(bundle_dir)


# ─── Item 35: model rejects an altered gene list ───────────────────────────

def test_rejects_altered_gene_list(tmp_path):
    bundle_dir = _write_bundle(tmp_path)
    art_path = bundle_dir / "preprocessing_artifact.json"
    d = json.loads(art_path.read_text())
    d["gene_list"] = list(reversed(d["gene_list"]))
    art_path.write_text(json.dumps(d))
    with pytest.raises(BundleValidationError):
        load_and_validate_bundle(bundle_dir)


# ─── Item 36: model rejects an incompatible class vocabulary ──────────────

def test_rejects_incompatible_class_vocabulary(tmp_path):
    bundle_dir = _write_bundle(tmp_path, class_vocabulary=["a", "b", "c"])
    manifest = load_and_validate_bundle(bundle_dir)
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, num_smoke=6,
                                 num_cell_types=4, attention_dim=4)
    with pytest.raises(BundleValidationError):
        validate_bundle_for_model(manifest, model)


def test_rejects_altered_input_dim(tmp_path):
    bundle_dir = _write_bundle(tmp_path)
    manifest = load_and_validate_bundle(bundle_dir)
    model = MultiSmokeCancerNet(input_dim=GENES + 5, embedding_dim=8, num_smoke=6,
                                 num_cell_types=4, attention_dim=4)
    with pytest.raises(BundleValidationError):
        validate_bundle_for_model(manifest, model)


# ─── Item 37: model rejects a corrupt artifact ─────────────────────────────

def test_rejects_corrupt_artifact_file(tmp_path):
    bundle_dir = _write_bundle(tmp_path)
    (bundle_dir / "preprocessing_artifact.json").write_text("{not valid json")
    with pytest.raises(BundleValidationError):
        load_and_validate_bundle(bundle_dir)


def test_rejects_corrupt_checkpoint_file(tmp_path):
    bundle_dir = _write_bundle(tmp_path)
    (tmp_path / "phase3_best.pt").write_bytes(b"garbage")
    with pytest.raises(BundleValidationError):
        load_and_validate_bundle(bundle_dir)


def test_rejects_tampered_manifest(tmp_path):
    bundle_dir = _write_bundle(tmp_path)
    manifest_path = bundle_dir / "bundle_manifest.json"
    d = json.loads(manifest_path.read_text())
    d["label_policy"] = "weak_labels_enabled"  # tamper with a field, leave fingerprint stale
    manifest_path.write_text(json.dumps(d))
    with pytest.raises(BundleValidationError):
        load_and_validate_bundle(bundle_dir)


def test_missing_bundle_manifest_without_allow_legacy_rejected(tmp_path):
    (tmp_path / "bundle").mkdir()
    with pytest.raises(LegacyBundleError):
        load_and_validate_bundle(tmp_path / "bundle")


# ─── Item 38: legacy checkpoint behavior follows explicit policy ──────────

def test_legacy_bundle_requires_explicit_opt_in():
    with pytest.raises(LegacyBundleError):
        load_and_validate_bundle("/nonexistent/bundle/dir")


def test_legacy_bundle_allowed_with_explicit_flag(tmp_path):
    (tmp_path / "bundle").mkdir()
    result = load_and_validate_bundle(tmp_path / "bundle", allow_legacy=True)
    assert result == {}


def test_missing_checkpoint_file_rejected(tmp_path):
    bundle_dir = _write_bundle(tmp_path)
    (tmp_path / "phase3_best.pt").unlink()
    with pytest.raises(BundleCorruptionError):
        load_and_validate_bundle(bundle_dir)


def test_unrecognized_schema_version_rejected(tmp_path):
    bundle_dir = _write_bundle(tmp_path)
    manifest_path = bundle_dir / "bundle_manifest.json"
    d = json.loads(manifest_path.read_text())
    d["schema_version"] = "999"
    manifest_path.write_text(json.dumps(d))
    with pytest.raises(BundleCorruptionError):
        load_and_validate_bundle(bundle_dir)


# ─── Item 39: resume logic validates identity before reuse ────────────────

def test_resume_matches_expected_identity(tmp_path):
    bundle_dir = _write_bundle(tmp_path)
    manifest = load_and_validate_bundle(bundle_dir)
    validate_bundle_matches_identity(
        manifest, {"dataset_manifest_fingerprint": "dmfp123", "split_fingerprint": "splitfp456"},
    )  # must not raise


def test_resume_rejects_mismatched_dataset_manifest_fingerprint(tmp_path):
    bundle_dir = _write_bundle(tmp_path)
    manifest = load_and_validate_bundle(bundle_dir)
    with pytest.raises(BundleValidationError):
        validate_bundle_matches_identity(manifest, {"dataset_manifest_fingerprint": "different"})


def test_resume_rejects_mismatched_split_fingerprint(tmp_path):
    bundle_dir = _write_bundle(tmp_path)
    manifest = load_and_validate_bundle(bundle_dir)
    with pytest.raises(BundleValidationError):
        validate_bundle_matches_identity(manifest, {"split_fingerprint": "different"})
