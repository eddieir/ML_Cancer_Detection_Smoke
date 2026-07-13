"""evaluate.py — binary metric safety and malignancy known/unknown masking."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import N_CELL_TYPES, N_SMOKE_CLASSES
from data.splitting import subject_train_val_test_split
from evaluate import Evaluator, _binary_metrics, _smoke_metrics, describe_split
from model import MultiSmokeCancerNet
from train import CellLevelDataset

GENES = 20


def test_binary_metrics_handles_empty_input():
    m = _binary_metrics([], [])
    assert m["roc_auc"] is None
    assert m["pr_auc"] is None
    assert m["n"] == 0


def test_binary_metrics_handles_single_class_target():
    m = _binary_metrics([1.0, 1.0, 1.0], [0.9, 0.8, 0.7])
    assert m["roc_auc"] is None  # undefined with one class, not a fabricated number
    assert m["pr_auc"] is None
    assert m["note"] is not None


def test_binary_metrics_normal_case_has_no_note():
    m = _binary_metrics([0.0, 1.0, 0.0, 1.0], [0.1, 0.9, 0.2, 0.8])
    assert m["roc_auc"] is not None
    assert m["note"] is None


def _cell_ds_with_known_mask(malig_known):
    n = len(malig_known)
    return CellLevelDataset(
        gene_matrix       = torch.randn(n, GENES).numpy(),
        smoke_labels      = torch.zeros(n, dtype=torch.long).numpy(),
        malignancy_labels = torch.zeros(n).numpy(),
        cell_type_ids     = torch.zeros(n, dtype=torch.long).numpy(),
        malignancy_known  = malig_known,
    )


def test_cell_level_evaluation_skips_malignancy_metrics_when_all_unknown():
    import numpy as np
    ds = _cell_ds_with_known_mask(np.zeros(10, dtype=bool))
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=16, attention_dim=8)
    ev = Evaluator(model)
    result = ev.cell_level(ds)
    assert result["malignancy"]["roc_auc"] is None
    assert result["malignancy"]["n"] == 0
    assert result["malignancy"]["n_unknown"] == 10


def test_smoke_metrics_handles_empty_input():
    m = _smoke_metrics([], [])
    assert m["n"] == 0
    assert "note" in m


def test_smoke_metrics_reports_full_breakdown():
    y_true = [0, 0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 1, 1, 2]
    m = _smoke_metrics(y_true, y_pred)
    assert set(["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]).issubset(m)
    assert len(m["per_class"]) == N_SMOKE_CLASSES
    for cls_metrics in m["per_class"].values():
        assert set(["precision", "recall", "f1", "support"]).issubset(cls_metrics)
    cm = m["confusion_matrix"]
    assert len(cm) == N_SMOKE_CLASSES and len(cm[0]) == N_SMOKE_CLASSES
    assert sum(sum(row) for row in cm) == len(y_true)


def test_smoke_metrics_normalized_confusion_matrix_rows_sum_to_one_or_zero():
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 1, 0, 1, 1]
    m = _smoke_metrics(y_true, y_pred)
    for row in m["confusion_matrix_normalized"]:
        s = sum(row)
        assert abs(s - 1.0) < 1e-6 or s == 0.0


def test_describe_split_reports_provenance_fields():
    subject_ids = [f"s{i}" for i in range(15) for _ in range(4)]
    labels = [i % 3 for i in range(15) for _ in range(4)]
    manifest = subject_train_val_test_split(subject_ids, labels, seed=1)
    info = describe_split("test", manifest, is_held_out=True, data_modality="single_cell")
    assert info["split_name"] == "test"
    assert info["is_held_out"] is True
    assert info["data_modality"] == "single_cell"
    assert info["n_subjects"] == len(manifest.test_subjects)
    assert info["class_distribution"] is not None


def test_cell_level_evaluation_reports_known_positive_negative_counts():
    import numpy as np
    known = np.array([True] * 6 + [False] * 4)
    ds = _cell_ds_with_known_mask(known)
    # overwrite malignancy labels directly for a deterministic known split
    ds.malig[:6] = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    model = MultiSmokeCancerNet(input_dim=GENES, embedding_dim=16, attention_dim=8)
    ev = Evaluator(model)
    result = ev.cell_level(ds)
    assert result["malignancy"]["n_known_positive"] == 3
    assert result["malignancy"]["n_known_negative"] == 3
    assert result["malignancy"]["n_unknown"] == 4
