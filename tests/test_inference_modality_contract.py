"""
inference.py — raw-array modality contract enforcement (GitHub issue #13,
third remediation round). predict_subject()/predict_batch() accept
preprocessed NumPy matrices with no gene names, no obs columns, nothing to
check on their own — every real call must declare input_modality/
input_assay_policy/is_pseudo_bulk explicitly. Covers Predictor.from_config's
legacy-artifact rejection and Predictor.from_bundle's bundle-backed modality
enforcement as well.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import anndata as ad
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import ASSAY_POLICY_SINGLE_CELL_ONLY, ASSAY_POLICY_VERSION, N_CELL_TYPES
from data.assay_policy import (
    AssayPolicyError,
    AssayPolicyMismatchError,
    BulkTrainingNotImplementedError,
    InvalidAssayProvenanceError,
    MultimodalTrainingNotImplementedError,
)
from data.preprocessing import CellTypeProvenanceError, PreprocessingArtifact, fit_preprocessing
from inference import Predictor, PredictorInputContractError, PredictorBundleCompatibilityError
from model import MultiSmokeCancerNet

GENES = 10


def _model():
    return MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)


def _valid_kwargs(n):
    return dict(
        input_modality="single_cell", input_assay_policy="single_cell_only",
        is_pseudo_bulk=np.zeros(n, dtype=bool),
    )


# ─── predict_subject: modality contract ────────────────────────────────────

def test_predict_subject_requires_input_modality():
    predictor = Predictor(_model())
    with pytest.raises(TypeError):
        predictor.predict_subject(
            np.random.randn(4, GENES).astype("float32"),
            np.zeros(4, dtype=int),
            input_assay_policy="single_cell_only", is_pseudo_bulk=np.zeros(4, dtype=bool),
        )


def test_predict_subject_rejects_missing_is_pseudo_bulk():
    predictor = Predictor(_model())
    with pytest.raises(PredictorInputContractError):
        predictor.predict_subject(
            np.random.randn(4, GENES).astype("float32"),
            np.zeros(4, dtype=int),
            input_modality="single_cell", input_assay_policy="single_cell_only",
            is_pseudo_bulk=None,
        )


def test_predict_subject_rejects_bulk_input_assay_policy():
    predictor = Predictor(_model())
    with pytest.raises(BulkTrainingNotImplementedError):
        predictor.predict_subject(
            np.random.randn(4, GENES).astype("float32"),
            np.zeros(4, dtype=int),
            input_modality="bulk", input_assay_policy="bulk_only",
            is_pseudo_bulk=np.ones(4, dtype=bool),
        )


def test_predict_subject_rejects_multimodal_input_assay_policy():
    predictor = Predictor(_model())
    with pytest.raises(MultimodalTrainingNotImplementedError):
        predictor.predict_subject(
            np.random.randn(4, GENES).astype("float32"),
            np.zeros(4, dtype=int),
            input_modality="multimodal", input_assay_policy="multimodal",
            is_pseudo_bulk=np.zeros(4, dtype=bool),
        )


def test_predict_subject_rejects_modality_policy_mismatch():
    predictor = Predictor(_model())
    with pytest.raises(PredictorInputContractError):
        predictor.predict_subject(
            np.random.randn(4, GENES).astype("float32"),
            np.zeros(4, dtype=int),
            input_modality="bulk",  # declared bulk but policy says single_cell_only
            input_assay_policy="single_cell_only",
            is_pseudo_bulk=np.zeros(4, dtype=bool),
        )


def test_predict_subject_rejects_pseudo_bulk_rows_under_single_cell_only():
    predictor = Predictor(_model())
    with pytest.raises(AssayPolicyError):
        predictor.predict_subject(
            np.random.randn(4, GENES).astype("float32"),
            np.zeros(4, dtype=int),
            input_modality="single_cell", input_assay_policy="single_cell_only",
            is_pseudo_bulk=np.array([False, False, True, False]),
        )


def test_predict_subject_rejects_array_length_mismatch():
    predictor = Predictor(_model())
    with pytest.raises(InvalidAssayProvenanceError):
        predictor.predict_subject(
            np.random.randn(4, GENES).astype("float32"),
            np.zeros(4, dtype=int),
            input_modality="single_cell", input_assay_policy="single_cell_only",
            is_pseudo_bulk=np.zeros(3, dtype=bool),  # wrong length
        )


def test_predict_subject_rejects_invalid_boolean_strings():
    predictor = Predictor(_model())
    with pytest.raises(InvalidAssayProvenanceError):
        predictor.predict_subject(
            np.random.randn(3, GENES).astype("float32"),
            np.zeros(3, dtype=int),
            input_modality="single_cell", input_assay_policy="single_cell_only",
            is_pseudo_bulk=["maybe", "false", "true"],
        )


def test_predict_subject_accepts_valid_explicit_single_cell_input():
    predictor = Predictor(_model())
    n = 6
    result = predictor.predict_subject(
        np.random.randn(n, GENES).astype("float32"),
        np.zeros(n, dtype=int),
        **_valid_kwargs(n),
    )
    assert result["diagnostic"] is False


def test_predict_subject_validates_before_model_forward_pass():
    """A spy proves the model is never invoked when the modality contract
    is invalid — validation happens before any tensor/model execution."""
    model = _model()
    model.forward_subject = MagicMock(side_effect=AssertionError("model must not run"))
    predictor = Predictor(model)
    with pytest.raises(PredictorInputContractError):
        predictor.predict_subject(
            np.random.randn(4, GENES).astype("float32"),
            np.zeros(4, dtype=int),
            input_modality="single_cell", input_assay_policy="single_cell_only",
            is_pseudo_bulk=None,
        )
    model.forward_subject.assert_not_called()


def test_predict_subject_diagnostic_mode_stamps_result():
    predictor = Predictor(_model())
    n = 5
    result = predictor.predict_subject(
        np.random.randn(n, GENES).astype("float32"),
        np.zeros(n, dtype=int),
        input_modality="single_cell", input_assay_policy="single_cell_only",
        is_pseudo_bulk=np.zeros(n, dtype=bool),
        diagnostic_mode=True,
    )
    assert result["diagnostic"] is True


# ─── predict_batch: batch-level modality contract ──────────────────────────

def _subject(n, sid="s0", **overrides):
    d = {
        "subject_id": sid,
        "gene_matrix": np.random.randn(n, GENES).astype("float32"),
        "cell_type_ids": np.zeros(n, dtype=int),
        "input_modality": "single_cell",
        "input_assay_policy": "single_cell_only",
        "is_pseudo_bulk": np.zeros(n, dtype=bool),
    }
    d.update(overrides)
    return d


def test_predict_batch_rejects_subject_missing_required_keys():
    predictor = Predictor(_model())
    subjects = [_subject(4, "s0"), {"subject_id": "s1", "gene_matrix": np.zeros((4, GENES), "float32"),
                                     "cell_type_ids": np.zeros(4, dtype=int)}]
    with pytest.raises(PredictorInputContractError):
        predictor.predict_batch(subjects)


def test_predict_batch_bad_final_subject_prevents_all_predictions():
    """Every subject is pre-validated before any subject is predicted — an
    invalid LAST subject must still reject the whole batch with zero
    predictions produced, not len(subjects)-1 partial results."""
    model = _model()
    model.forward_subject = MagicMock(side_effect=AssertionError("model must not run"))
    predictor = Predictor(model)
    subjects = [_subject(4, "s0"), _subject(4, "s1"), _subject(4, "s2", is_pseudo_bulk=None)]
    with pytest.raises(PredictorInputContractError):
        predictor.predict_batch(subjects)
    model.forward_subject.assert_not_called()


def test_predict_batch_rejects_mixed_modalities():
    predictor = Predictor(_model())
    subjects = [
        _subject(4, "s0"),
        _subject(4, "s1", input_modality="bulk", input_assay_policy="bulk_only",
                 is_pseudo_bulk=np.ones(4, dtype=bool)),
    ]
    with pytest.raises(AssayPolicyMismatchError):
        predictor.predict_batch(subjects)


def test_predict_batch_rejects_mixed_diagnostic_status():
    predictor = Predictor(_model())
    subjects = [
        _subject(4, "s0", diagnostic_mode=False),
        _subject(4, "s1", diagnostic_mode=True),
    ]
    with pytest.raises(AssayPolicyMismatchError):
        predictor.predict_batch(subjects)


def test_predict_batch_valid_batch_still_works():
    predictor = Predictor(_model())
    subjects = [_subject(4, f"s{i}") for i in range(3)]
    results = predictor.predict_batch(subjects)
    assert len(results) == 3
    assert all(r["subject_id"] == f"s{i}" for i, r in enumerate(results))
    assert all(r["diagnostic"] is False for r in results)


# ─── Predictor.from_config: legacy-artifact rejection ──────────────────────

def _train_artifact(tmp_path, n=60, g=GENES, seed=0):
    subject_ids = [f"s{i // 10}" for i in range(n)]
    adata = ad.AnnData(
        X=np.random.default_rng(seed).random((n, g)).astype("float32"),
        obs=pd.DataFrame({"subject_id": subject_ids, "is_pseudo_bulk": [False] * n}),
        var=pd.DataFrame(index=[f"G{i}" for i in range(g)]),
    )
    return fit_preprocessing(adata, {f"s{i}" for i in range(n // 10)}, n_hvgs=g)


def _write_checkpoint(ckpt_dir, model):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "format_version": 2},
               ckpt_dir / "phase3_best.pt")


def test_from_config_rejects_legacy_artifact_missing_assay_policy(tmp_path):
    model = _model()
    ckpt_dir = tmp_path / "ckpt"
    _write_checkpoint(ckpt_dir, model)
    artifact = _train_artifact(tmp_path)
    d = artifact.to_dict()
    d["assay_policy"] = None
    d["assay_policy_version"] = None
    legacy = PreprocessingArtifact(**d)
    legacy.save(ckpt_dir / "preprocessing_artifact.json")

    config = {"model": {"input_dim": GENES, "embedding_dim": 8, "attention_dim": 4},
              "train": {"checkpoint_dir": str(ckpt_dir)}}
    with pytest.raises(CellTypeProvenanceError):
        Predictor.from_config(config, phase=3)


def test_from_config_unsafe_legacy_mode_allows_legacy_artifact_and_stamps_diagnostic(tmp_path):
    model = _model()
    ckpt_dir = tmp_path / "ckpt"
    _write_checkpoint(ckpt_dir, model)
    artifact = _train_artifact(tmp_path)
    d = artifact.to_dict()
    d["assay_policy"] = None
    d["assay_policy_version"] = None
    legacy = PreprocessingArtifact(**d)
    legacy.save(ckpt_dir / "preprocessing_artifact.json")

    config = {"model": {"input_dim": GENES, "embedding_dim": 8, "attention_dim": 4},
              "train": {"checkpoint_dir": str(ckpt_dir)}}
    predictor = Predictor.from_config(config, phase=3, unsafe_legacy_mode=True)
    n = 5
    result = predictor.predict_subject(
        np.random.randn(n, GENES).astype("float32"), np.zeros(n, dtype=int), **_valid_kwargs(n),
    )
    assert result["diagnostic"] is True


def test_from_config_accepts_real_artifact(tmp_path):
    model = _model()
    ckpt_dir = tmp_path / "ckpt"
    _write_checkpoint(ckpt_dir, model)
    artifact = _train_artifact(tmp_path)
    artifact.save(ckpt_dir / "preprocessing_artifact.json")

    config = {"model": {"input_dim": GENES, "embedding_dim": 8, "attention_dim": 4},
              "train": {"checkpoint_dir": str(ckpt_dir)}}
    predictor = Predictor.from_config(config, phase=3)
    n = 5
    result = predictor.predict_subject(
        np.random.randn(n, GENES).astype("float32"), np.zeros(n, dtype=int), **_valid_kwargs(n),
    )
    assert result["diagnostic"] is False


# ─── Predictor.from_bundle: bundle-backed modality enforcement ─────────────

def _write_bundle(tmp_path, model, artifact):
    from benchmarks.bundle import write_model_bundle
    ckpt_path = tmp_path / "phase3_best.pt"
    torch.save({"model_state_dict": model.state_dict(), "format_version": 2}, ckpt_path)
    return write_model_bundle(
        tmp_path / "bundle", ckpt_path, artifact,
        model_config={"input_dim": GENES, "embedding_dim": 8, "attention_dim": 4},
        class_vocabulary=["cigarette", "vape_ecig", "cigar", "cannabis", "dual_use", "unexposed"],
        label_policy="verified_only", species_policy="human_only", assay_mode="human_single_cell",
    )


def test_from_bundle_loads_and_validates_input_modality(tmp_path):
    model = _model()
    artifact = _train_artifact(tmp_path)
    bundle_dir = _write_bundle(tmp_path, model, artifact)
    predictor = Predictor.from_bundle(bundle_dir)
    n = 5
    result = predictor.predict_subject(
        np.random.randn(n, GENES).astype("float32"), np.zeros(n, dtype=int), **_valid_kwargs(n),
    )
    assert result["diagnostic"] is False


def test_from_bundle_rejects_pseudo_bulk_input_against_single_cell_bundle(tmp_path):
    model = _model()
    artifact = _train_artifact(tmp_path)
    bundle_dir = _write_bundle(tmp_path, model, artifact)
    predictor = Predictor.from_bundle(bundle_dir)
    n = 4
    with pytest.raises(AssayPolicyError):
        predictor.predict_subject(
            np.random.randn(n, GENES).astype("float32"), np.zeros(n, dtype=int),
            input_modality="single_cell", input_assay_policy="single_cell_only",
            is_pseudo_bulk=np.array([False, False, False, True]),
        )


def test_from_bundle_rejects_corrupted_bundle_before_prediction(tmp_path):
    from benchmarks.bundle import BundleCorruptionError

    model = _model()
    artifact = _train_artifact(tmp_path)
    bundle_dir = _write_bundle(tmp_path, model, artifact)
    manifest_path = bundle_dir / "bundle_manifest.json"
    with open(manifest_path, "a") as f:
        f.write("tampered-not-json")
    with pytest.raises(BundleCorruptionError):
        Predictor.from_bundle(bundle_dir)
