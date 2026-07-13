"""train.py — CellLevelDataset class-weighting and Trainer checkpoint selection."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from constants import N_SMOKE_CLASSES, N_CELL_TYPES
from model import MultiSmokeCancerNet
from train import (
    CellLevelDataset, SubjectLevelDataset, MILEligibilityError, Trainer,
    assert_disjoint_subjects, check_mil_eligibility, validate_experiment_partitions,
)
from metrics import validate_cell_type_ids

GENES = 20


def _dataset(smoke_labels: np.ndarray) -> CellLevelDataset:
    n = len(smoke_labels)
    return CellLevelDataset(
        gene_matrix       = np.random.randn(n, GENES).astype("float32"),
        smoke_labels      = smoke_labels,
        malignancy_labels = np.random.randint(0, 2, n).astype("float32"),
        cell_type_ids     = np.zeros(n, dtype="int64"),
        diagnostic_mode    = True,
    )


def test_smoke_class_weights_balanced_dataset_all_equal():
    labels = np.tile(np.arange(N_SMOKE_CLASSES), 5)   # equal counts per class
    weights = _dataset(labels).smoke_class_weights()
    assert torch.allclose(weights, torch.full((N_SMOKE_CLASSES,), 1.0), atol=1e-5)


def test_smoke_class_weights_favors_minority_classes():
    """
    README's real-data run: cigarette=83, dual_use=30, vape=7, cannabis=6,
    cigar=1, unexposed=10 out of a 6-class label space. The rare classes
    must get strictly larger weight than the majority class, otherwise the
    weighting doesn't do what it's for.
    """
    counts = [83, 7, 1, 6, 30, 10]
    labels = np.concatenate([np.full(c, cls) for cls, c in enumerate(counts)])
    weights = _dataset(labels).smoke_class_weights()

    majority_cls, minority_cls = 0, 2  # cigarette (83) vs cigar (1)
    assert weights[minority_cls] > weights[majority_cls]
    # sanity check against the closed-form balanced-weight formula
    n = sum(counts)
    expected = n / (N_SMOKE_CLASSES * counts[majority_cls])
    assert abs(weights[majority_cls].item() - expected) < 1e-4


def test_smoke_class_weights_zero_for_absent_classes():
    """A class with zero samples in this merge gets weight 0, not inf/nan."""
    labels = np.array([0, 0, 1, 1])  # classes 2..5 entirely absent
    weights = _dataset(labels).smoke_class_weights()
    assert torch.isfinite(weights).all()
    assert (weights[2:] == 0).all()
    assert (weights[:2] > 0).all()


def _bag(subject_id, cancer_label=None, cancer_label_known=False, n=10):
    return {
        "subject_id":         subject_id,
        "gene_matrix":        np.random.randn(n, GENES).astype("float32"),
        "cell_type_ids":      np.zeros(n, dtype="int64"),
        "smoke_labels":       np.zeros(n, dtype="int64"),
        "malig_labels":       np.zeros(n, dtype="float32"),
        "cancer_label":       cancer_label,
        "cancer_label_known": cancer_label_known,
    }


def test_subject_level_dataset_excludes_unknown_outcome_by_default():
    bags = [
        _bag("known_pos", cancer_label=1, cancer_label_known=True),
        _bag("known_neg", cancer_label=0, cancer_label_known=True),
        _bag("unknown",   cancer_label=None, cancer_label_known=False),
    ]
    ds = SubjectLevelDataset(bags)
    assert len(ds) == 2
    subject_ids = {ds[i]["subject_id"] for i in range(len(ds))}
    assert subject_ids == {"known_pos", "known_neg"}
    assert ds.n_excluded_unknown_outcome == 1


def test_subject_level_dataset_can_keep_unknown_outcome_when_requested():
    bags = [
        _bag("known", cancer_label=1, cancer_label_known=True),
        _bag("unknown", cancer_label=None, cancer_label_known=False),
    ]
    ds = SubjectLevelDataset(bags, require_known_outcome=False)
    assert len(ds) == 2


def test_mil_eligibility_rejects_too_few_subjects():
    bags = [_bag(f"s{i}", cancer_label=i % 2, cancer_label_known=True) for i in range(4)]
    ds = SubjectLevelDataset(bags)
    with pytest.raises(MILEligibilityError):
        check_mil_eligibility(ds, min_subjects=10)


def test_mil_eligibility_rejects_single_class():
    bags = [_bag(f"s{i}", cancer_label=1, cancer_label_known=True) for i in range(15)]
    ds = SubjectLevelDataset(bags)
    with pytest.raises(MILEligibilityError):
        check_mil_eligibility(ds, min_subjects=10, min_positive=2, min_negative=2)


def test_mil_eligibility_passes_balanced_sufficient_data():
    bags = [_bag(f"s{i}", cancer_label=i % 2, cancer_label_known=True) for i in range(20)]
    ds = SubjectLevelDataset(bags)
    report = check_mil_eligibility(ds, min_subjects=10, min_positive=2, min_negative=2)
    assert report["n_total"] == 20
    assert report["problems"] == []


# ─── Real subject_ids required outside diagnostic_mode ─────────────────────

def test_cell_level_dataset_rejects_missing_subject_ids_by_default():
    with pytest.raises(ValueError):
        CellLevelDataset(
            gene_matrix=np.random.randn(5, GENES).astype("float32"),
            smoke_labels=np.zeros(5, dtype="int64"),
            malignancy_labels=np.zeros(5, dtype="float32"),
            cell_type_ids=np.zeros(5, dtype="int64"),
        )


def test_cell_level_dataset_rejects_unknown_subject_id_outside_diagnostic_mode():
    with pytest.raises(ValueError):
        CellLevelDataset(
            gene_matrix=np.random.randn(3, GENES).astype("float32"),
            smoke_labels=np.zeros(3, dtype="int64"),
            malignancy_labels=np.zeros(3, dtype="float32"),
            cell_type_ids=np.zeros(3, dtype="int64"),
            subject_ids=["sub_a", "unknown", "sub_b"],
        )


def test_cell_level_dataset_diagnostic_mode_allows_missing_subject_ids():
    ds = CellLevelDataset(
        gene_matrix=np.random.randn(5, GENES).astype("float32"),
        smoke_labels=np.zeros(5, dtype="int64"),
        malignancy_labels=np.zeros(5, dtype="float32"),
        cell_type_ids=np.zeros(5, dtype="int64"),
        diagnostic_mode=True,
    )
    assert len(ds) == 5


# ─── subset_by_subjects keeps a subject's cells together ────────────────────

def test_subset_by_subjects_keeps_one_subjects_cells_entirely_in_one_split():
    n_per_subject = 40
    subject_ids = np.repeat(["sub_a", "sub_b", "sub_c"], n_per_subject)
    n = len(subject_ids)
    ds = CellLevelDataset(
        gene_matrix=np.random.randn(n, GENES).astype("float32"),
        smoke_labels=np.random.randint(0, N_SMOKE_CLASSES, n),
        malignancy_labels=np.random.randint(0, 2, n).astype("float32"),
        cell_type_ids=np.zeros(n, dtype="int64"),
        subject_ids=subject_ids,
    )
    train_ds = ds.subset_by_subjects(["sub_a", "sub_b"])
    val_ds   = ds.subset_by_subjects(["sub_c"])
    assert len(train_ds) == 2 * n_per_subject
    assert len(val_ds) == n_per_subject
    assert set(train_ds.subject_ids.tolist()) == {"sub_a", "sub_b"}
    assert set(val_ds.subject_ids.tolist()) == {"sub_c"}


def test_subset_by_subjects_preserves_label_alignment():
    subject_ids = np.array(["a", "a", "b", "b", "b"])
    smoke = np.array([0, 1, 2, 3, 4])
    ds = CellLevelDataset(
        gene_matrix=np.arange(5 * GENES, dtype="float32").reshape(5, GENES),
        smoke_labels=smoke,
        malignancy_labels=np.zeros(5, dtype="float32"),
        cell_type_ids=np.zeros(5, dtype="int64"),
        subject_ids=subject_ids,
    )
    sub_b = ds.subset_by_subjects(["b"])
    assert sub_b.smoke.tolist() == [2, 3, 4]
    # gene rows must still correspond to the same original cells
    assert torch.equal(sub_b.X[0], ds.X[2])


# ─── assert_disjoint_subjects / Trainer leakage guard ───────────────────────

def test_assert_disjoint_subjects_passes_for_disjoint_sets():
    assert_disjoint_subjects(["a", "b"], ["c", "d"])  # no raise


def test_assert_disjoint_subjects_raises_on_overlap():
    with pytest.raises(ValueError):
        assert_disjoint_subjects(["a", "b"], ["b", "c"])


def _real_cell_ds(subject_ids, n_per_subject=20):
    n = len(subject_ids) * n_per_subject
    sid_col = np.repeat(subject_ids, n_per_subject)
    return CellLevelDataset(
        gene_matrix=np.random.randn(n, GENES).astype("float32"),
        smoke_labels=np.random.randint(0, N_SMOKE_CLASSES, n),
        malignancy_labels=np.random.randint(0, 2, n).astype("float32"),
        cell_type_ids=np.zeros(n, dtype="int64"),
        subject_ids=sid_col,
    )


def _trainer():
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=16, attention_dim=8)
    return Trainer(model, {"train": {"phase1_epochs": 1, "checkpoint_dir": "/tmp/_test_ckpt_leakage"}})


def test_trainer_phase1_rejects_overlapping_train_val_subjects():
    trainer = _trainer()
    shared = _real_cell_ds(["s1", "s2"])
    with pytest.raises(ValueError):
        trainer.phase1(shared, shared)  # same subjects in "train" and "val"


def test_trainer_phase1_rejects_empty_train_dataset():
    trainer = _trainer()
    empty = _real_cell_ds([])
    val = _real_cell_ds(["s1"])
    with pytest.raises(ValueError):
        trainer.phase1(empty, val)


def test_trainer_phase1_accepts_disjoint_train_val_subjects():
    trainer = _trainer()
    trainer.cfg["phase1_epochs"] = 1
    train_ds = _real_cell_ds(["s1", "s2", "s3"])
    val_ds   = _real_cell_ds(["s4"])
    result = trainer.phase1(train_ds, val_ds)
    assert "best_smoke_macro_f1" in result


# ─── Structured checkpoints ──────────────────────────────────────────────────

def test_trainer_save_writes_structured_checkpoint_with_metadata(tmp_path):
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=16, attention_dim=8)
    trainer = Trainer(model, {"train": {"checkpoint_dir": str(tmp_path)}}, seed=7)
    trainer.split_manifest_path = "some/manifest.json"
    trainer.rare_class_policy = "keep_with_warning"
    trainer._save(1, 0.42, metric_name="val_smoke_macro_f1")

    obj = torch.load(tmp_path / "phase1_best.pt", weights_only=False)
    assert obj["format_version"] >= 2
    assert "model_state_dict" in obj
    assert obj["metric_name"] == "val_smoke_macro_f1"
    assert obj["metric_value"] == 0.42
    assert obj["random_seed"] == 7
    assert obj["split_manifest_path"] == "some/manifest.json"
    assert obj["rare_class_policy"] == "keep_with_warning"
    assert obj["input_dim"] == GENES


def test_trainer_load_best_round_trips_structured_checkpoint(tmp_path):
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=16, attention_dim=8)
    trainer = Trainer(model, {"train": {"checkpoint_dir": str(tmp_path)}})
    trainer._save(1, 0.9, metric_name="val_smoke_macro_f1")
    original_weight = next(model.parameters()).clone()

    # Perturb the live model, then reload — should restore the saved weights.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    meta = trainer._load_best(1)
    assert torch.allclose(next(model.parameters()), original_weight)
    assert meta["metric_value"] == 0.9


def test_trainer_load_best_handles_legacy_raw_state_dict_checkpoint(tmp_path):
    """A checkpoint saved before structured format existed (bare
    state_dict) must still load, with a warning, and no crash."""
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=16, attention_dim=8)
    ckpt_dir = tmp_path
    torch.save(model.state_dict(), ckpt_dir / "phase1_best.pt")  # legacy format

    trainer = Trainer(model, {"train": {"checkpoint_dir": str(ckpt_dir)}})
    meta = trainer._load_best(1)
    assert meta == {}  # no metadata available for legacy checkpoints


def test_trainer_phase1_class_weights_computed_from_train_only():
    """Class weights must reflect ONLY the train dataset, never validation."""
    trainer = _trainer()
    trainer.cfg["phase1_epochs"] = 1
    n_tr, n_va = 60, 60
    train_ds = CellLevelDataset(
        gene_matrix=np.random.randn(n_tr, GENES).astype("float32"),
        smoke_labels=np.zeros(n_tr, dtype="int64"),  # all class 0
        malignancy_labels=np.zeros(n_tr, dtype="float32"),
        cell_type_ids=np.zeros(n_tr, dtype="int64"),
        subject_ids=np.array(["s1"] * n_tr),
    )
    val_ds = CellLevelDataset(
        gene_matrix=np.random.randn(n_va, GENES).astype("float32"),
        smoke_labels=np.ones(n_va, dtype="int64"),  # all class 1 — must NOT affect weights
        malignancy_labels=np.zeros(n_va, dtype="float32"),
        cell_type_ids=np.zeros(n_va, dtype="int64"),
        subject_ids=np.array(["s2"] * n_va),
    )
    expected = train_ds.smoke_class_weights()
    result = trainer.phase1(train_ds, val_ds)
    assert torch.allclose(expected, train_ds.smoke_class_weights())


# ─── validate_experiment_partitions: cross-task leakage ─────────────────────

def _real_subject_ds(subject_ids, cancer_label=1):
    return SubjectLevelDataset([
        _bag(sid, cancer_label=cancer_label, cancer_label_known=True) for sid in subject_ids
    ])


def test_validate_experiment_partitions_passes_for_clean_split():
    groups = validate_experiment_partitions(
        train_cell_dataset=_real_cell_ds(["s1", "s2"]),
        val_cell_dataset=_real_cell_ds(["s3"]),
        train_subject_dataset=_real_subject_ds(["s1", "s2"]),
        val_subject_dataset=_real_subject_ds(["s3"]),
    )
    assert groups["train"] == {"s1", "s2"}
    assert groups["val"] == {"s3"}


def test_validate_experiment_partitions_rejects_train_cell_val_bag_overlap():
    """Same-modality checks (train_cell vs val_cell) would miss this: s3's
    CELLS are only in train, but s3's BAG is in val — cross-modality leakage
    that a subject-disjoint-per-phase check alone does not catch."""
    with pytest.raises(ValueError):
        validate_experiment_partitions(
            train_cell_dataset=_real_cell_ds(["s1", "s3"]),
            val_cell_dataset=_real_cell_ds(["s2"]),
            train_subject_dataset=_real_subject_ds(["s1"]),
            val_subject_dataset=_real_subject_ds(["s2", "s3"]),  # s3 bag leaked into val
        )


def test_validate_experiment_partitions_rejects_train_bag_val_cell_overlap():
    with pytest.raises(ValueError):
        validate_experiment_partitions(
            train_cell_dataset=_real_cell_ds(["s1"]),
            val_cell_dataset=_real_cell_ds(["s2", "s3"]),  # s3 cells leaked into val
            train_subject_dataset=_real_subject_ds(["s1", "s3"]),
            val_subject_dataset=_real_subject_ds(["s2"]),
        )


def test_validate_experiment_partitions_allows_same_split_cell_bag_overlap():
    """A subject's cells and that SAME subject's bag both being in "train" is
    fine and expected — only cross-split overlap is leakage."""
    groups = validate_experiment_partitions(
        train_cell_dataset=_real_cell_ds(["s1"]),
        val_cell_dataset=_real_cell_ds(["s2"]),
        train_subject_dataset=_real_subject_ds(["s1"]),
        val_subject_dataset=_real_subject_ds(["s2"]),
    )
    assert groups["train"] == {"s1"}


def test_validate_experiment_partitions_checks_test_partition_too():
    with pytest.raises(ValueError):
        validate_experiment_partitions(
            train_cell_dataset=_real_cell_ds(["s1"]),
            val_cell_dataset=_real_cell_ds(["s2"]),
            train_subject_dataset=_real_subject_ds(["s1"]),
            val_subject_dataset=_real_subject_ds(["s2"]),
            test_cell_dataset=_real_cell_ds(["s1"]),  # s1 leaked into test
            test_subject_dataset=_real_subject_ds([]),
        )


def test_validate_experiment_partitions_skips_diagnostic_datasets():
    diag = _dataset(np.zeros(10, dtype="int64"))  # diagnostic_mode=True
    groups = validate_experiment_partitions(
        train_cell_dataset=diag, val_cell_dataset=diag,
        train_subject_dataset=_real_subject_ds(["s1"]),
        val_subject_dataset=_real_subject_ds(["s1"]),  # would overlap if enforced
    )
    assert groups == {}


def test_validate_experiment_partitions_rejects_empty_partition():
    with pytest.raises(ValueError):
        validate_experiment_partitions(
            train_cell_dataset=_real_cell_ds([]),
            val_cell_dataset=_real_cell_ds(["s1"]),
            train_subject_dataset=_real_subject_ds([]),
            val_subject_dataset=_real_subject_ds(["s1"]),
        )


def test_trainer_phase3_rejects_train_cell_val_bag_overlap():
    trainer = _trainer()
    trainer.cfg.update({"phase1_epochs": 1, "phase2_epochs": 1, "phase3_epochs": 1})
    with pytest.raises(ValueError):
        trainer.phase3(
            train_cell_dataset=_real_cell_ds(["s1", "s3"]),
            val_cell_dataset=_real_cell_ds(["s2"]),
            train_subject_dataset=_real_subject_ds(["s1"]),
            val_subject_dataset=_real_subject_ds(["s2", "s3"]),
            skip_eligibility_check=True,
        )


# ─── Cell-type ID validation ──────────────────────────────────────────────────

def test_validate_cell_type_ids_accepts_valid_boundary_ids():
    ids = validate_cell_type_ids(np.array([0, 1, 2, 3]), num_cell_types=4, n_expected=4)
    assert ids.tolist() == [0, 1, 2, 3]


def test_validate_cell_type_ids_rejects_negative_id():
    with pytest.raises(ValueError):
        validate_cell_type_ids(np.array([0, -1, 2]), num_cell_types=4, n_expected=3)


def test_validate_cell_type_ids_rejects_id_equal_to_num_cell_types():
    with pytest.raises(ValueError):
        validate_cell_type_ids(np.array([0, 4]), num_cell_types=4, n_expected=2)


def test_validate_cell_type_ids_rejects_fractional_id():
    with pytest.raises(ValueError):
        validate_cell_type_ids(np.array([0.0, 1.5]), num_cell_types=4, n_expected=2)


def test_validate_cell_type_ids_rejects_nan():
    with pytest.raises(ValueError):
        validate_cell_type_ids(np.array([0.0, np.nan]), num_cell_types=4, n_expected=2)


def test_validate_cell_type_ids_rejects_wrong_length():
    with pytest.raises(ValueError):
        validate_cell_type_ids(np.array([0, 1, 2]), num_cell_types=4, n_expected=5)


def test_trainer_predict_rejects_invalid_cell_type_id():
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=16, attention_dim=8)
    trainer = Trainer(model, {"train": {"checkpoint_dir": "/tmp/_test_ckpt_predict"}})
    n = 10
    with pytest.raises(ValueError):
        trainer.predict(
            np.random.randn(n, GENES).astype("float32"),
            np.array([0, 1, 2, 99, 0, 0, 0, 0, 0, 0]),  # 99 out of range
        )


# ─── Held-out test evaluation enforcement ─────────────────────────────────────

def _trainer_with_seen_subjects(train_subjects, val_subjects, tmp_path):
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=16, attention_dim=8)
    trainer = Trainer(model, {"train": {"phase1_epochs": 1, "checkpoint_dir": str(tmp_path)}})
    train_ds = _real_cell_ds(train_subjects)
    val_ds   = _real_cell_ds(val_subjects)
    trainer.phase1(train_ds, val_ds)
    return trainer


def test_final_test_evaluation_rejects_test_subject_seen_in_train(tmp_path):
    trainer = _trainer_with_seen_subjects(["s1", "s2"], ["s3"], tmp_path)
    with pytest.raises(ValueError):
        trainer.final_test_evaluation(
            test_cell_dataset=_real_cell_ds(["s1"]),  # s1 was in train
            test_subject_dataset=_real_subject_ds(["s1"]),
        )


def test_final_test_evaluation_rejects_test_subject_seen_in_val(tmp_path):
    trainer = _trainer_with_seen_subjects(["s1", "s2"], ["s3"], tmp_path)
    with pytest.raises(ValueError):
        trainer.final_test_evaluation(
            test_cell_dataset=_real_cell_ds(["s3"]),  # s3 was in val
            test_subject_dataset=_real_subject_ds(["s3"]),
        )


def test_final_test_evaluation_accepts_genuinely_held_out_subjects(tmp_path):
    trainer = _trainer_with_seen_subjects(["s1", "s2"], ["s3"], tmp_path)
    report = trainer.final_test_evaluation(
        test_cell_dataset=_real_cell_ds(["s4"]),
        test_subject_dataset=_real_subject_ds(["s4"]),
        phase=1,
    )
    assert report["provenance"]["split_name"] == "test"
    assert report["provenance"]["is_held_out"] is True
    assert report["provenance"]["is_pristine"] is True
    assert (tmp_path / "heldout_test_report.json").exists()
    assert (tmp_path / "heldout_test_predictions.json").exists()


def test_final_test_evaluation_blocks_repeat_run_by_default(tmp_path):
    trainer = _trainer_with_seen_subjects(["s1", "s2"], ["s3"], tmp_path)
    trainer.final_test_evaluation(
        test_cell_dataset=_real_cell_ds(["s4"]),
        test_subject_dataset=_real_subject_ds(["s4"]),
        phase=1,
    )
    with pytest.raises(RuntimeError):
        trainer.final_test_evaluation(
            test_cell_dataset=_real_cell_ds(["s4"]),
            test_subject_dataset=_real_subject_ds(["s4"]),
            phase=1,
        )


def test_final_test_evaluation_allow_repeat_marks_result_non_pristine(tmp_path):
    trainer = _trainer_with_seen_subjects(["s1", "s2"], ["s3"], tmp_path)
    trainer.final_test_evaluation(
        test_cell_dataset=_real_cell_ds(["s4"]),
        test_subject_dataset=_real_subject_ds(["s4"]),
        phase=1,
    )
    report = trainer.final_test_evaluation(
        test_cell_dataset=_real_cell_ds(["s4"]),
        test_subject_dataset=_real_subject_ds(["s4"]),
        allow_repeat=True,
        phase=1,
    )
    assert report["provenance"]["test_evaluation_run_count"] == 2
    assert report["provenance"]["is_pristine"] is False
