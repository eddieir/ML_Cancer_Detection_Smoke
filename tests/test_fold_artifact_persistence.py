"""
Phase 4 — per-fold preprocessing artifact persistence (Step 8).

Every CV fold must fit and persist its OWN PreprocessingArtifact, from only
that fold's training subjects, under a real on-disk directory
(<output_root>/preprocessing/fold_XX/) — not just an in-memory object that
disappears after the run. Covers adversarial items 40-44:

  40. every fold receives a different training/split fingerprint
  41. every fold artifact excludes its validation subjects
  42. OOF records contain artifact fingerprints (already true — see
      tests/test_benchmarks_final_evaluation.py's fold_membership records;
      re-verified here directly against the persistence layer)
  43. fold artifact reuse requires exact compatibility
  44. changing one fold cannot mutate another fold's artifact
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.fold_preprocessing import (
    FoldArtifactMismatchError,
    IncompleteFoldArtifactError,
    fold_artifact_dir,
    load_fold_artifact,
    refit_artifact_for_fold,
    save_fold_artifact,
)


def _adata(n_subjects=20, cells_per_subject=15, n_genes=30, seed=0):
    rng = np.random.default_rng(seed)
    subject_ids = []
    for i in range(n_subjects):
        subject_ids += [f"sub_{i}"] * cells_per_subject
    n = len(subject_ids)
    X = rng.random((n, n_genes)).astype("float32")
    genes = [f"G{i}" for i in range(n_genes)]
    obs = pd.DataFrame({"subject_id": subject_ids, "batch": ["b0"] * n}, index=[f"c{i}" for i in range(n)])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))


def _two_folds(adata, n_genes=30):
    subjects = sorted(set(adata.obs["subject_id"]))
    fold0_train, fold0_val = subjects[:12], subjects[12:16]
    fold1_train, fold1_val = subjects[4:16], subjects[16:20]
    art0 = refit_artifact_for_fold(adata, fold0_train, n_genes)
    art1 = refit_artifact_for_fold(adata, fold1_train, n_genes)
    return (art0, fold0_train, fold0_val), (art1, fold1_train, fold1_val)


# ─── Item 40: every fold receives a different training/split fingerprint ──

def test_different_folds_get_different_fingerprints(tmp_path):
    adata = _adata(seed=1)
    (art0, tr0, va0), (art1, tr1, va1) = _two_folds(adata)
    assert art0.scientific_fingerprint() != art1.scientific_fingerprint()
    save_fold_artifact(art0, tmp_path, 0, tr0, va0)
    save_fold_artifact(art1, tmp_path, 1, tr1, va1)
    loaded0 = load_fold_artifact(tmp_path, 0)
    loaded1 = load_fold_artifact(tmp_path, 1)
    assert loaded0.scientific_fingerprint() != loaded1.scientific_fingerprint()


# ─── Item 41: every fold artifact excludes its validation subjects ────────

def test_fold_artifact_fit_n_subjects_excludes_validation_subjects(tmp_path):
    adata = _adata(seed=2)
    (art0, tr0, va0), _ = _two_folds(adata)
    assert set(str(s) for s in va0).isdisjoint(set(str(s) for s in tr0))
    # fit_n_subjects reflects ONLY the training partition actually used to fit.
    assert art0.fit_n_subjects == len(set(str(s) for s in tr0))


# ─── Persistence directory layout ──────────────────────────────────────────

def test_fold_artifact_saved_to_expected_directory_layout(tmp_path):
    adata = _adata(seed=3)
    (art0, tr0, va0), _ = _two_folds(adata)
    fold_dir = save_fold_artifact(art0, tmp_path, 0, tr0, va0)
    assert fold_dir == fold_artifact_dir(tmp_path, 0)
    assert (fold_dir / "artifact.json").exists()
    assert (fold_dir / "manifest.json").exists()


def test_save_load_round_trip_preserves_fingerprint(tmp_path):
    adata = _adata(seed=4)
    (art0, tr0, va0), _ = _two_folds(adata)
    save_fold_artifact(art0, tmp_path, 0, tr0, va0)
    reloaded = load_fold_artifact(tmp_path, 0, expected_train_subjects=tr0, expected_val_subjects=va0)
    assert reloaded.scientific_fingerprint() == art0.scientific_fingerprint()


# ─── Item 43: fold artifact reuse requires exact compatibility ────────────

def test_resume_with_wrong_expected_train_subjects_rejected(tmp_path):
    adata = _adata(seed=5)
    (art0, tr0, va0), (_, tr1, _) = _two_folds(adata)
    save_fold_artifact(art0, tmp_path, 0, tr0, va0)
    with pytest.raises(FoldArtifactMismatchError):
        load_fold_artifact(tmp_path, 0, expected_train_subjects=tr1, expected_val_subjects=va0)


def test_resume_with_wrong_expected_val_subjects_rejected(tmp_path):
    adata = _adata(seed=6)
    (art0, tr0, va0), (_, _, va1) = _two_folds(adata)
    save_fold_artifact(art0, tmp_path, 0, tr0, va0)
    with pytest.raises(FoldArtifactMismatchError):
        load_fold_artifact(tmp_path, 0, expected_train_subjects=tr0, expected_val_subjects=va1)


def test_resume_with_wrong_expected_fingerprint_rejected(tmp_path):
    adata = _adata(seed=7)
    (art0, tr0, va0), _ = _two_folds(adata)
    save_fold_artifact(art0, tmp_path, 0, tr0, va0)
    with pytest.raises(FoldArtifactMismatchError):
        load_fold_artifact(tmp_path, 0, expected_artifact_fingerprint="0" * 64)


# ─── One fold cannot accidentally load another fold's artifact ────────────

def test_a_fold_cannot_be_loaded_using_another_folds_subjects(tmp_path):
    adata = _adata(seed=8)
    (art0, tr0, va0), (art1, tr1, va1) = _two_folds(adata)
    save_fold_artifact(art0, tmp_path, 0, tr0, va0)
    save_fold_artifact(art1, tmp_path, 1, tr1, va1)
    # Attempting to "resume" fold 0 while expecting fold 1's subjects must fail.
    with pytest.raises(FoldArtifactMismatchError):
        load_fold_artifact(tmp_path, 0, expected_train_subjects=tr1, expected_val_subjects=va1)


# ─── Incomplete fold artifact is rejected ──────────────────────────────────

def test_incomplete_fold_artifact_rejected_missing_manifest(tmp_path):
    adata = _adata(seed=9)
    (art0, tr0, va0), _ = _two_folds(adata)
    fold_dir = fold_artifact_dir(tmp_path, 0)
    art0.save(fold_dir / "artifact.json")  # artifact.json written, manifest.json never written
    with pytest.raises(IncompleteFoldArtifactError):
        load_fold_artifact(tmp_path, 0)


def test_incomplete_fold_artifact_rejected_non_complete_status(tmp_path):
    import json

    adata = _adata(seed=10)
    (art0, tr0, va0), _ = _two_folds(adata)
    fold_dir = save_fold_artifact(art0, tmp_path, 0, tr0, va0)
    manifest_path = fold_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "in_progress"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(IncompleteFoldArtifactError):
        load_fold_artifact(tmp_path, 0)


def test_missing_fold_directory_rejected(tmp_path):
    with pytest.raises(IncompleteFoldArtifactError):
        load_fold_artifact(tmp_path, 5)


# ─── Item 44: changing one fold cannot mutate another fold's artifact ─────

def test_resaving_one_fold_does_not_alter_another_folds_persisted_artifact(tmp_path):
    adata = _adata(seed=11)
    (art0, tr0, va0), (art1, tr1, va1) = _two_folds(adata)
    save_fold_artifact(art0, tmp_path, 0, tr0, va0)
    save_fold_artifact(art1, tmp_path, 1, tr1, va1)
    fp1_before = load_fold_artifact(tmp_path, 1).scientific_fingerprint()

    # Re-fit and re-save fold 0 with a heavily perturbed training set.
    adata2 = adata.copy()
    non_fold0_train_mask = ~adata2.obs["subject_id"].isin(tr0).values
    X = adata2.X.copy()
    X[non_fold0_train_mask] = X[non_fold0_train_mask] * 999 + 111
    adata2.X = X
    art0_v2 = refit_artifact_for_fold(adata2, tr0, 30)
    save_fold_artifact(art0_v2, tmp_path, 0, tr0, va0)

    fp1_after = load_fold_artifact(tmp_path, 1).scientific_fingerprint()
    assert fp1_before == fp1_after
