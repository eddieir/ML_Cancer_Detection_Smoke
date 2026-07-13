"""inference.py — Predictor preprocessing-compatibility validation."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import N_CELL_TYPES
from data.label_mapping import build_effective_label_mapping
from data.preprocessing import PreprocessingArtifact
from inference import Predictor, VALID_INPUT_STAGES
from model import MultiSmokeCancerNet

GENES = 10


def _artifact(genes):
    return PreprocessingArtifact(
        version="1", gene_list=list(genes),
        gene_means=[0.0] * len(genes), gene_stds=[1.0] * len(genes),
        n_hvgs=len(genes), smoke_marker_genes_forced=[],
        fit_n_cells=100, fit_n_subjects=10,
    )


def _h5ad(path, genes, n=20):
    obs = pd.DataFrame({
        "subject_id":   ["s1"] * n,
        "cell_type_id": np.zeros(n, dtype=int),
    }, index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(X=np.random.randn(n, len(genes)).astype("float32"), obs=obs,
                    var=pd.DataFrame(index=genes))
    a.write_h5ad(path)


def test_predict_h5ad_rejects_incompatible_gene_panel():
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact([f"G{i}" for i in range(GENES)])
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mismatched.h5ad"
        _h5ad(path, [f"OTHER{i}" for i in range(GENES)])  # completely different gene panel
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path))


def test_predict_h5ad_accepts_compatible_gene_panel():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "matched.h5ad"
        _h5ad(path, genes)
        results = predictor.predict_h5ad(str(path))
        assert len(results) == 1


def test_predict_h5ad_rejects_raw_input_without_artifact_by_default():
    """No artifact loaded and no explicit unsafe_legacy_mode — refuse rather
    than silently run unreordered/unscaled genes through the model."""
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model)  # no artifact, unsafe_legacy_mode=False

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "any.h5ad"
        _h5ad(path, [f"WHATEVER{i}" for i in range(GENES)])
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path))


def test_predict_h5ad_unsafe_legacy_mode_bypasses_missing_artifact():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model, unsafe_legacy_mode=True)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "any.h5ad"
        _h5ad(path, genes)
        results = predictor.predict_h5ad(str(path))
        assert len(results) == 1


def test_predict_h5ad_reorders_genes_that_are_present_but_shuffled():
    """A raw file with the right genes in the WRONG order must still be
    correctly reordered by apply_preprocessing before scoring."""
    genes = [f"G{i}" for i in range(GENES)]
    shuffled = list(reversed(genes))
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shuffled.h5ad"
        _h5ad(path, shuffled)
        results = predictor.predict_h5ad(str(path))
        assert len(results) == 1


def test_predict_h5ad_rejects_extra_genes_not_in_artifact():
    """Extra genes beyond the artifact's panel must be dropped, not error —
    only MISSING required genes are fatal (see verify_compatible)."""
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "extra.h5ad"
        _h5ad(path, genes + ["EXTRA_GENE_1", "EXTRA_GENE_2"])
        results = predictor.predict_h5ad(str(path))
        assert len(results) == 1


def test_predict_h5ad_already_preprocessed_requires_exact_gene_order():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shuffled.h5ad"
        _h5ad(path, list(reversed(genes)))
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path), already_preprocessed=True)


def test_predict_h5ad_already_preprocessed_accepts_exact_match():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "exact.h5ad"
        _h5ad(path, genes)
        results = predictor.predict_h5ad(str(path), already_preprocessed=True)
        assert len(results) == 1


def test_predict_h5ad_rejects_wrong_model_input_width():
    """Model expects GENES+5 features but the artifact/H5AD only has GENES —
    must fail with a clear error, not a confusing shape mismatch deep in the model."""
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES + 5, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "matched.h5ad"
        _h5ad(path, genes)
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path))


def test_predict_h5ad_rejects_out_of_range_cell_type_id():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_ct.h5ad"
        n = 10
        obs = pd.DataFrame({
            "subject_id":   ["s1"] * n,
            "cell_type_id": [0, 1, 2, 3, 99, 0, 0, 0, 0, 0],  # 99 is out of range
        }, index=[f"c{i}" for i in range(n)])
        a = ad.AnnData(X=np.random.randn(n, GENES).astype("float32"), obs=obs,
                        var=pd.DataFrame(index=genes))
        a.write_h5ad(path)
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path))


def test_predict_h5ad_rejects_negative_cell_type_id():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "neg_ct.h5ad"
        n = 5
        obs = pd.DataFrame({
            "subject_id":   ["s1"] * n,
            "cell_type_id": [0, -1, 2, 0, 0],
        }, index=[f"c{i}" for i in range(n)])
        a = ad.AnnData(X=np.random.randn(n, GENES).astype("float32"), obs=obs,
                        var=pd.DataFrame(index=genes))
        a.write_h5ad(path)
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path))


def test_predict_subject_rejects_fractional_cell_type_id():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model, unsafe_legacy_mode=True)
    with pytest.raises(ValueError):
        predictor.predict_subject(
            np.random.randn(4, GENES).astype("float32"),
            np.array([0.0, 1.5, 2.0, 0.0]),
        )


def test_predict_h5ad_handles_sparse_input():
    import scipy.sparse as sp
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    predictor = Predictor(model, preprocessing_artifact=artifact)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sparse.h5ad"
        n = 15
        obs = pd.DataFrame({
            "subject_id":   ["s1"] * n,
            "cell_type_id": np.zeros(n, dtype=int),
        }, index=[f"c{i}" for i in range(n)])
        a = ad.AnnData(
            X=sp.csr_matrix(np.random.randn(n, len(genes)).astype("float32")),
            obs=obs, var=pd.DataFrame(index=genes),
        )
        a.write_h5ad(path)
        results = predictor.predict_h5ad(str(path))
        assert len(results) == 1


# ─── input_stage explicit API ─────────────────────────────────────────────────

def test_predict_h5ad_rejects_raw_counts_input_stage():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model, preprocessing_artifact=_artifact(genes))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "raw.h5ad"
        _h5ad(path, genes)
        with pytest.raises(ValueError, match="raw_counts"):
            predictor.predict_h5ad(str(path), input_stage="raw_counts")


def test_predict_h5ad_rejects_invalid_input_stage_value():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model, preprocessing_artifact=_artifact(genes))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "any.h5ad"
        _h5ad(path, genes)
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path), input_stage="totally_bogus_stage")


def test_predict_h5ad_model_ready_accepts_exact_match():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model, preprocessing_artifact=_artifact(genes))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "exact.h5ad"
        _h5ad(path, genes)
        results = predictor.predict_h5ad(str(path), input_stage="model_ready")
        assert len(results) == 1


def test_predict_h5ad_model_ready_rejects_shuffled_order():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model, preprocessing_artifact=_artifact(genes))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shuffled.h5ad"
        _h5ad(path, list(reversed(genes)))
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path), input_stage="model_ready")


def test_predict_h5ad_model_ready_rejects_non_finite_values():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model, preprocessing_artifact=_artifact(genes))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nan.h5ad"
        n = 5
        X = np.random.randn(n, GENES).astype("float32")
        X[0, 0] = np.nan
        obs = pd.DataFrame({
            "subject_id":   ["s1"] * n,
            "cell_type_id": np.zeros(n, dtype=int),
        }, index=[f"c{i}" for i in range(n)])
        a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
        a.write_h5ad(path)
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path), input_stage="model_ready")


def test_predict_h5ad_model_ready_rejects_artifact_version_mismatch():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    artifact = _artifact(genes)
    artifact.version = "999"  # not the current ARTIFACT_VERSION
    predictor = Predictor(model, preprocessing_artifact=artifact)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "exact.h5ad"
        _h5ad(path, genes)
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path), input_stage="model_ready")


def test_predict_h5ad_normalized_expression_reorders_and_scales():
    genes = [f"G{i}" for i in range(GENES)]
    shuffled = list(reversed(genes))
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model, preprocessing_artifact=_artifact(genes))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shuffled.h5ad"
        _h5ad(path, shuffled)
        results = predictor.predict_h5ad(str(path), input_stage="normalized_expression")
        assert len(results) == 1


def test_predict_h5ad_normalized_expression_rejects_missing_genes():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model, preprocessing_artifact=_artifact(genes))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "missing.h5ad"
        _h5ad(path, genes[:-2])  # missing 2 required genes
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path), input_stage="normalized_expression")


def test_predict_h5ad_normalized_expression_rejects_duplicate_genes():
    genes = [f"G{i}" for i in range(GENES)]
    dup_genes = genes + [genes[0]]  # GENES+1 columns, first gene duplicated
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model, preprocessing_artifact=_artifact(genes))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dup.h5ad"
        _h5ad(path, dup_genes)
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path), input_stage="normalized_expression")


def test_predict_h5ad_normalized_expression_rejects_expected_input_stage_mismatch():
    genes = [f"G{i}" for i in range(GENES)]
    artifact = _artifact(genes)
    artifact.expected_input_stage = "some_future_stage"
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model, preprocessing_artifact=artifact)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "any.h5ad"
        _h5ad(path, genes)
        with pytest.raises(ValueError):
            predictor.predict_h5ad(str(path), input_stage="normalized_expression")


def test_legacy_already_preprocessed_true_maps_to_model_ready_and_warns():
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model, preprocessing_artifact=_artifact(genes))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "exact.h5ad"
        _h5ad(path, genes)
        with pytest.warns(DeprecationWarning):
            results = predictor.predict_h5ad(str(path), already_preprocessed=True)
        assert len(results) == 1


def test_legacy_already_preprocessed_false_never_silently_claims_raw_support():
    """already_preprocessed=False must map to 'normalized_expression', which
    still requires an artifact/reorder/scale — it must NOT silently accept
    truly raw, un-normalized input as if raw support existed."""
    genes = [f"G{i}" for i in range(GENES)]
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)
    predictor = Predictor(model)  # no artifact, unsafe_legacy_mode=False
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "raw.h5ad"
        _h5ad(path, genes)
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError):
                predictor.predict_h5ad(str(path), already_preprocessed=False)


def test_valid_input_stages_contains_exactly_three_values():
    assert VALID_INPUT_STAGES == {"model_ready", "normalized_expression", "raw_counts"}


# ─── Effective label mapping wiring (Predictor) ───────────────────────────────

def _merged_mapping():
    report = {
        "policy": "merge_into_dual_use_or_other",
        "affected_classes": {"cigar": {"n_subjects": 1, "too_rare": True, "action": "merged_into_dual_use"}},
    }
    return build_effective_label_mapping(report)


def test_predictor_rejects_label_mapping_k_mismatch():
    mapping = _merged_mapping()  # K=5
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4)  # still num_smoke=6
    with pytest.raises(ValueError):
        Predictor(model, label_mapping=mapping)


def test_predictor_smoke_profile_uses_wired_mapping_class_names():
    mapping = _merged_mapping()
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=8, attention_dim=4, num_smoke=mapping.k)
    predictor = Predictor(model, label_mapping=mapping)
    n = 20
    r = predictor.predict_subject(
        np.random.randn(n, GENES).astype("float32"),
        np.random.randint(0, N_CELL_TYPES, n),
    )
    assert set(r["smoke_profile"].keys()) == set(mapping.class_names)
    assert "cigar" not in r["smoke_profile"]
