"""
benchmarks/metrics.py — metric definitions shared by every baseline, the
neural adapter, and the grouped-CV runner, so no two models in a comparison
can be scored by subtly different code paths.

Two rules enforced throughout:
  1. An undefined metric (e.g. AUROC with one class present) is returned as
     None with a reason string — NEVER coerced to 0.5 or 0.0. Averaging code
     downstream must skip Nones explicitly.
  2. Cell-weighted vs subject-weighted are always reported as two distinct
     numbers for Task A: a subject with 10,000 cells must not get 100x the
     voting power of a subject with 100 cells in the metric that's actually
     used for model comparison.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from metrics import multiclass_f1_report  # top-level src/metrics.py — single source of truth


# ─── Task A: smoke-type classification ────────────────────────────────────────

def cell_weighted_smoke_metrics(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> Dict:
    """Every cell counts equally — subjects with more cells dominate."""
    return multiclass_f1_report(y_true, y_pred, num_classes)


def subject_weighted_smoke_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], subject_ids: Sequence[str], num_classes: int,
) -> Dict:
    """
    One vote per subject: majority-vote the cell-level predictions within
    each subject into a single subject-level prediction/label first, then
    score. This is the metric that must NOT be dominated by cell count, and
    is the primary Task A comparison metric (see eligibility.py / runner.py).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    subject_ids = np.asarray([str(s) for s in subject_ids])

    subj_true, subj_pred = [], []
    for sid in sorted(set(subject_ids.tolist())):
        mask = subject_ids == sid
        true_vals = y_true[mask]
        pred_vals = y_pred[mask]
        # every cell of one subject shares one true label (subject-level split
        # invariant enforced upstream by splitting.py's _unique_subject_labels)
        subj_true.append(int(np.bincount(true_vals).argmax()))
        subj_pred.append(int(np.bincount(pred_vals, minlength=num_classes).argmax()))

    return multiclass_f1_report(subj_true, subj_pred, num_classes)


def _majority_vote_by_subject(
    y_true: Sequence[int], y_pred: Sequence[int], subject_ids: Sequence[str], num_classes: int,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    subject_ids = np.asarray([str(s) for s in subject_ids])
    subj_true, subj_pred = [], []
    for sid in sorted(set(subject_ids.tolist())):
        mask = subject_ids == sid
        subj_true.append(int(np.bincount(y_true[mask]).argmax()))
        subj_pred.append(int(np.bincount(y_pred[mask], minlength=num_classes).argmax()))
    return subj_true, subj_pred


def subject_weighted_full_smoke_metrics_report(
    y_true: Sequence[int], y_pred: Sequence[int], subject_ids: Sequence[str], num_classes: int,
) -> Dict:
    """
    The PRIMARY smoke-classification report: one vote per subject (see
    subject_weighted_smoke_metrics) carried through the FULL secondary-
    metric bundle (balanced accuracy, per-class precision/recall/F1/
    support, confusion matrix) — not just macro/weighted F1. "support" in
    the returned per_class dict therefore counts SUBJECTS, not cells, and a
    subject with many cells cannot dominate any of these numbers any more
    than a subject with few cells can. Class ordering is always
    range(num_classes), so confusion-matrix rows/columns and per_class keys
    are stable across strategies/folds/runs.
    """
    subj_true, subj_pred = _majority_vote_by_subject(y_true, y_pred, subject_ids, num_classes)
    return full_smoke_metrics_report(subj_true, subj_pred, num_classes)


def full_smoke_metrics_report(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> Dict:
    """multiclass_f1_report plus the secondary metrics section 2 requires:
    balanced accuracy, per-class precision/recall/F1/support, confusion matrix."""
    base = multiclass_f1_report(y_true, y_pred, num_classes)
    if len(y_true) == 0:
        base.update({"balanced_accuracy": None, "per_class": {}, "confusion_matrix": None})
        return base
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0,
    )
    base["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    base["per_class"] = {
        str(c): {"precision": float(precision[c]), "recall": float(recall[c]),
                 "f1": float(f1[c]), "support": int(support[c])}
        for c in labels
    }
    base["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return base


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> Optional[Dict]:
    """Percentile bootstrap CI over an already-computed list of per-fold/per-seed
    metric values. Returns None (undefined) if fewer than 2 values are given."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=float)
    boot_means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": float(lo), "hi": float(hi), "mean": float(arr.mean()), "n": len(arr)}


def aggregate_metric(values: Sequence[Optional[float]], seed: int = 42) -> Dict:
    """
    Descriptive fold-level aggregation: mean, std, median, bootstrap CI, and
    an explicit count of defined vs undefined values. Never silently drops
    the undefined count — a model with 3/5 folds undefined must not look
    identical to one with 5/5 defined folds just because the mean is similar.

    The bootstrap CI here resamples raw per-fold values as if they were
    independent — with repeated seeds over the same (overlapping) subject
    pool, folds are NOT independent samples (a subject reappears in many
    folds across seeds), so this CI is optimistic/descriptive, not a
    statistically rigorous uncertainty estimate. `resampling_unit: "fold"`
    marks this explicitly. Prefer aggregate_metric_by_seed (resamples whole
    seeds, which ARE independent random re-partitions) whenever >=2 seeds
    are available — see its docstring.
    """
    defined = [v for v in values if v is not None]
    n_undefined = len(values) - len(defined)
    if not defined:
        return {
            "mean": None, "std": None, "median": None, "ci": None,
            "n_valid": 0, "n_undefined": n_undefined, "values": list(values),
            "resampling_unit": "fold",
        }
    arr = np.asarray(defined, dtype=float)
    return {
        "mean": float(arr.mean()), "std": float(arr.std(ddof=0)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)), "ci": bootstrap_ci(defined, seed=seed),
        "n_valid": len(defined), "n_undefined": n_undefined, "values": list(values),
        "resampling_unit": "fold",
    }


def aggregate_metric_by_seed(
    values: Sequence[Optional[float]], seeds: Sequence[int], seed: int = 42,
) -> Dict:
    """
    Statistically preferred aggregation for repeated-seed grouped CV: average
    each seed's fold values into ONE per-seed mean first, then bootstrap
    across those per-seed means. Distinct seeds are genuinely independent
    random subject re-partitions (unlike folds within/across seeds, which
    overlap), so this CI's resampling unit is actually valid — at the cost
    of far fewer "samples" (one per seed, not one per fold). Requires >=2
    distinct seeds with at least one defined value each; returns
    n_valid=0/ci=None with a note otherwise rather than silently falling
    back to the (statistically weaker) per-fold bootstrap.
    """
    by_seed: Dict[int, list] = {}
    for v, s in zip(values, seeds):
        by_seed.setdefault(s, []).append(v)
    seed_means = []
    for s, vals in by_seed.items():
        defined = [v for v in vals if v is not None]
        if defined:
            seed_means.append(float(np.mean(defined)))
    if len(seed_means) < 2:
        return {
            "mean": float(np.mean(seed_means)) if seed_means else None,
            "std": None, "median": None, "ci": None, "n_valid": len(seed_means),
            "n_undefined": len(by_seed) - len(seed_means), "values": seed_means,
            "resampling_unit": "seed",
            "note": "fewer than 2 seeds had a defined value — cannot bootstrap across independent seeds",
        }
    arr = np.asarray(seed_means, dtype=float)
    return {
        "mean": float(arr.mean()), "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)), "ci": bootstrap_ci(seed_means, seed=seed),
        "n_valid": len(seed_means), "n_undefined": len(by_seed) - len(seed_means),
        "values": seed_means, "resampling_unit": "seed",
    }


# ─── Task B: subject-level cancer prediction ──────────────────────────────────

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> List[Dict]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    curve = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        curve.append({
            "bin_lo": float(lo), "bin_hi": float(hi), "n": int(mask.sum()),
            "mean_predicted": float(y_prob[mask].mean()) if mask.sum() else None,
            "empirical_rate": float(y_true[mask].mean()) if mask.sum() else None,
        })
    return curve


def cancer_metrics_at_threshold(y_true: Sequence[int], y_prob: Sequence[float], threshold: float) -> Dict:
    """
    All metrics that need a hard decision. AUROC/AUPRC are threshold-free and
    computed separately in cancer_prediction_metrics(). Returns None for any
    metric that's undefined given the data (single-class y_true).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    classes_present = set(y_true.tolist())
    if len(classes_present) < 2:
        return {
            "balanced_accuracy": None, "sensitivity": None, "specificity": None,
            "f1": None, "reason": f"only class(es) {classes_present} present in y_true",
        }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity) if sensitivity is not None else None,
        "specificity": float(specificity) if specificity is not None else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "reason": None,
    }


def cancer_prediction_metrics(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5,
) -> Dict:
    """
    Full Task B metric bundle at a given (already frozen, if this is a test
    evaluation) threshold. AUROC/AUPRC are None with an explicit reason when
    y_true has only one class — this is the "undefined AUROC stays undefined,
    never 0.5" rule from the spec.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    classes_present = set(y_true.tolist())

    if len(classes_present) < 2:
        auroc, auprc, reason = None, None, f"only class(es) {classes_present} present in y_true"
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
        reason = None

    at_thresh = cancer_metrics_at_threshold(y_true, y_prob, threshold)
    brier = float(brier_score_loss(y_true, y_prob)) if len(classes_present) >= 1 else None
    ece = expected_calibration_error(y_true, y_prob) if len(y_true) > 0 else None

    return {
        "auroc": auroc, "auprc": auprc, "auroc_auprc_undefined_reason": reason,
        "brier": brier, "ece": ece, "threshold": threshold,
        **at_thresh,
    }
