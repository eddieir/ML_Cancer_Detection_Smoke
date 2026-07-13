"""
metrics.py — Metric definitions shared identically by Trainer (checkpoint
selection) and Evaluator (reporting).

Trainer.phase1 used to pick the "best" checkpoint with
f1_score(targets, preds, average="macro") — no explicit label list. sklearn's
average="macro" with labels=None restricts averaging to classes observed in
y_true/y_pred for that call, so a validation batch missing one smoke class
would silently compute macro-F1 over 5 classes instead of 6. evaluate.py's
_smoke_metrics passed an explicit label list and so was NOT affected the same
way — meaning the metric used to select a checkpoint and the metric reported
for that checkpoint could silently disagree. This module is the one place
"macro-F1 over the smoke-type task" is defined, so both call sites agree.
"""

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import f1_score


def validate_cell_type_ids(cell_type_ids, num_cell_types: int, n_expected: int) -> "np.ndarray":
    """
    Strict validation of cell-type IDs before they're used to index into a
    model's cell-type embedding table. Previously only column *presence* was
    checked (inference.py::_validate_h5ad), so a NaN, a negative id, an id
    equal to num_cell_types (out of range — valid ids are 0..num_cell_types-1),
    or a fractional id could silently corrupt the embedding lookup or crash
    deep inside the model with a confusing index error. Used identically by
    Predictor.predict_subject/predict_batch/predict_h5ad and Trainer.predict.

    Returns the ids as an int64 ndarray on success; raises ValueError naming
    the actual invalid values and the expected range otherwise.
    """
    arr = np.asarray(cell_type_ids)
    if arr.shape[0] != n_expected:
        raise ValueError(
            f"cell_type_ids has {arr.shape[0]} entries but {n_expected} expected "
            "(must equal the number of expression rows)."
        )
    if arr.dtype.kind == "f":
        if np.isnan(arr).any():
            n_nan = int(np.isnan(arr).sum())
            raise ValueError(f"cell_type_ids contains {n_nan} NaN value(s) — cell type must be known for every cell.")
        if not np.all(np.mod(arr, 1.0) == 0.0):
            bad = arr[np.mod(arr, 1.0) != 0.0][:5]
            raise ValueError(f"cell_type_ids contains non-integer value(s) {bad.tolist()} — must be whole numbers.")
    elif arr.dtype.kind not in ("i", "u"):
        raise ValueError(f"cell_type_ids has unsupported dtype {arr.dtype} — must be integer or safely integer-convertible float.")

    ids = arr.astype(np.int64)
    if (ids < 0).any():
        bad = sorted(set(ids[ids < 0].tolist()))[:5]
        raise ValueError(f"cell_type_ids contains negative value(s) {bad} — valid range is [0, {num_cell_types - 1}].")
    if (ids >= num_cell_types).any():
        bad = sorted(set(ids[ids >= num_cell_types].tolist()))[:5]
        raise ValueError(
            f"cell_type_ids contains value(s) {bad} >= num_cell_types={num_cell_types} — "
            f"valid range is [0, {num_cell_types - 1}]."
        )
    return ids


def multiclass_f1_report(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> Dict:
    """
    Macro/weighted F1 computed over the full effective class list
    (0..num_classes-1) regardless of which classes happen to appear in this
    particular y_true/y_pred — classes with zero support contribute F1=0 to
    the macro average (sklearn's zero_division=0 behaviour), which is the
    standard, honest interpretation: "this class exists in the model's
    output space and the model was not evaluated as having gotten it right."
    classes_absent_from_targets records which classes had zero true
    examples in y_true, so a partial validation split's macro-F1 isn't
    mistaken for a full 6-class number.
    """
    all_labels = list(range(num_classes))
    if len(y_true) == 0:
        return {
            "macro_f1": 0.0, "weighted_f1": 0.0, "num_classes": num_classes,
            "classes_absent_from_targets": all_labels, "is_partial": True,
        }
    absent_from_targets = sorted(set(all_labels) - set(y_true))
    macro = f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)
    return {
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "num_classes": num_classes,
        "classes_absent_from_targets": absent_from_targets,
        "is_partial": len(absent_from_targets) > 0,
    }
