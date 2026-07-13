"""evaluate.py — binary metric safety and malignancy known/unknown masking."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from constants import N_CELL_TYPES
from evaluate import Evaluator, _binary_metrics
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
