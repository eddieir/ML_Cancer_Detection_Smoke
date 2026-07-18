"""
benchmarks/domain_shift.py — label-free domain-shift diagnostics and the
optional source-predictability diagnostic classifier.

Every function here is a DIAGNOSTIC: it characterizes how different a
held-out source's subject-level representation looks relative to the
development pool, using development-fitted preprocessing/features only. None
of these functions decide model selection, calibration, or thresholds by
themselves, and none of them are a substitute for the labeled source-held-out
evaluation in source_held_out.py — a low computed "shift" does not mean the
model will generalize, and a high one does not by itself mean it will fail;
these numbers are reported alongside the labeled evaluation, not instead of
it.
"""

import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .domain_losses import _median_bandwidth, _pairwise_sq_dists, _rbf_kernel_sum, _subject_covariance


def _to_tensor(x: np.ndarray):
    import torch
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64)


# ─── Gene-space compatibility ───────────────────────────────────────────────

def gene_space_compatibility(required_genes: Sequence[str], present_genes: Sequence[str]) -> Dict:
    required = list(required_genes)
    present_set = set(present_genes)
    required_set = set(required)
    missing = sorted(required_set - present_set)
    unexpected = sorted(present_set - required_set)
    order_compatible = list(present_genes[:len(required)]) == required if len(present_genes) >= len(required) else False
    coverage = 1.0 - (len(missing) / len(required)) if required else 1.0
    return {
        "n_required_genes": len(required), "n_present_genes": len(present_genes),
        "n_missing_genes": len(missing), "missing_genes": missing[:50],
        "n_unexpected_genes": len(unexpected), "unexpected_genes": unexpected[:50],
        "gene_coverage": coverage, "gene_order_compatible": order_compatible,
    }


def module_coverage_compatibility(module_gene_counts: Dict[str, int], min_genes_per_module: int) -> Dict:
    empty = [m for m, n in module_gene_counts.items() if n == 0]
    below_min = [m for m, n in module_gene_counts.items() if 0 < n < min_genes_per_module]
    return {
        "n_modules": len(module_gene_counts), "n_empty_modules": len(empty), "empty_modules": empty,
        "n_below_minimum_modules": len(below_min), "below_minimum_modules": below_min,
    }


# ─── Subject/cell composition ───────────────────────────────────────────────

def composition_summary(
    cells_per_subject: Dict[str, int], cell_type_counts_per_subject: Dict[str, Dict[int, int]],
    known_label_mask_per_subject: Optional[Dict[str, float]] = None,
) -> Dict:
    counts = list(cells_per_subject.values())
    arr = np.asarray(counts, dtype=np.float64) if counts else np.zeros(0)
    all_types = sorted({t for d in cell_type_counts_per_subject.values() for t in d})
    proportions = {}
    for sid, d in cell_type_counts_per_subject.items():
        total = sum(d.values())
        proportions[sid] = {str(t): (d.get(t, 0) / total if total else 0.0) for t in all_types}
    missing_types_per_subject = {
        sid: sorted(str(t) for t in all_types if d.get(t, 0) == 0)
        for sid, d in cell_type_counts_per_subject.items()
    }
    return {
        "n_subjects": len(counts),
        "cells_per_subject_summary": {
            "min": float(arr.min()) if len(arr) else None,
            "median": float(np.median(arr)) if len(arr) else None,
            "iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))] if len(arr) else None,
            "max": float(arr.max()) if len(arr) else None,
        },
        "cell_type_proportions_per_subject": proportions,
        "missing_cell_types_per_subject": missing_types_per_subject,
        "known_label_rate_per_subject": known_label_mask_per_subject or {},
    }


# ─── Distribution shift (unsupervised, development-fitted features only) ──

def centroid_distance(dev_features: np.ndarray, held_out_features: np.ndarray) -> float:
    return float(np.linalg.norm(dev_features.mean(axis=0) - held_out_features.mean(axis=0)))


def energy_distance(dev_features: np.ndarray, held_out_features: np.ndarray) -> float:
    """E-statistic energy distance: 2*E|X-Y| - E|X-X'| - E|Y-Y'|, computed
    exactly (all pairwise distances) — fine at the subject-level scale this
    is always used at (tens to low hundreds of subjects, never cells)."""
    x = np.asarray(dev_features, dtype=np.float64)
    y = np.asarray(held_out_features, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    dxy = np.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1) + 1e-12)
    dxx = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1) + 1e-12)
    dyy = np.sqrt(((y[:, None, :] - y[None, :, :]) ** 2).sum(-1) + 1e-12)
    return float(2 * dxy.mean() - dxx.mean() - dyy.mean())


def coral_distance(dev_features: np.ndarray, held_out_features: np.ndarray) -> float:
    x, y = _to_tensor(dev_features), _to_tensor(held_out_features)
    d = x.shape[1]
    diff = _subject_covariance(x) - _subject_covariance(y)
    return float(((diff ** 2).sum() / (4.0 * d * d)).item())


def mmd_distance(dev_features: np.ndarray, held_out_features: np.ndarray) -> float:
    x, y = _to_tensor(dev_features), _to_tensor(held_out_features)
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    bw = _median_bandwidth(x, y)
    kxx, kyy, kxy = _rbf_kernel_sum(x, x, bw), _rbf_kernel_sum(y, y, bw), _rbf_kernel_sum(x, y, bw)
    return float((kxx.mean() + kyy.mean() - 2 * kxy.mean()).clamp(min=0.0).item())


def distribution_shift_report(dev_features: np.ndarray, held_out_features: np.ndarray) -> Dict:
    """
    Label-free — takes only feature matrices, never labels. Every metric
    here is computed from development-fitted preprocessing (the caller is
    responsible for ensuring held_out_features was produced by transforming,
    never refitting, the development preprocessing artifact) and the
    held-out source's own raw feature values; no held-out LABEL is used
    anywhere in this function.
    """
    return {
        "n_dev_subjects": int(len(dev_features)), "n_held_out_subjects": int(len(held_out_features)),
        "centroid_distance": centroid_distance(dev_features, held_out_features),
        "energy_distance": energy_distance(dev_features, held_out_features),
        "coral_distance": coral_distance(dev_features, held_out_features),
        "mmd_distance": mmd_distance(dev_features, held_out_features),
    }


# ─── Source-predictability diagnostic ──────────────────────────────────────

def source_predictability_diagnostic(
    features: np.ndarray, sources: Sequence[str], subject_groups: Sequence[str], seed: int = 42, n_folds: int = 3,
) -> Dict:
    """
    Fits a small subject-grouped-CV classifier predicting SOURCE from
    development representations only — never from held-out-source data,
    which this function does not even accept as an argument. Reports
    balanced accuracy and macro-F1 against a label-permutation baseline
    (chance level given the actual source-count imbalance), NOT an absolute
    pass/fail threshold: a high score means source identity remains encoded
    in the representation (a fact about the representation), not by itself
    evidence that the primary task model is broken; a low score is a
    diagnostic signal, not proof of "domain invariance."
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, f1_score

    from data.splitting import grouped_kfold

    features = np.asarray(features, dtype=np.float64)
    sources = np.asarray([str(s) for s in sources])
    groups = np.asarray([str(g) for g in subject_groups])
    unique_sources = sorted(set(sources.tolist()))
    if len(unique_sources) < 2:
        return {"status": "insufficient_evidence", "reason": "fewer than 2 development sources present"}
    src_idx = {s: i for i, s in enumerate(unique_sources)}
    y = np.array([src_idx[s] for s in sources])

    folds = grouped_kfold(groups, y, n_folds=n_folds, seed=seed)
    accs, f1s, perm_accs = [], [], []
    rng = np.random.RandomState(seed)
    for fold in folds:
        train_mask = np.isin(groups, fold["train"])
        val_mask = np.isin(groups, fold["val"])
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        clf = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=4)
        clf.fit(features[train_mask], y[train_mask])
        pred = clf.predict(features[val_mask])
        if len(set(y[val_mask].tolist())) < 2:
            continue
        accs.append(balanced_accuracy_score(y[val_mask], pred))
        f1s.append(f1_score(y[val_mask], pred, average="macro"))

        permuted_train_y = rng.permutation(y[train_mask])
        clf_perm = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=4)
        clf_perm.fit(features[train_mask], permuted_train_y)
        pred_perm = clf_perm.predict(features[val_mask])
        perm_accs.append(balanced_accuracy_score(y[val_mask], pred_perm))

    if not accs:
        return {"status": "insufficient_evidence", "reason": "no fold had both classes represented in validation"}
    return {
        "status": "evaluated", "balanced_accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s)),
        "permutation_baseline_balanced_accuracy": float(np.mean(perm_accs)),
        "n_folds_evaluated": len(accs), "n_sources": len(unique_sources),
        "note": "high balanced accuracy indicates source identity remains encoded in the "
                "representation; this is a diagnostic about the representation, not a pass/fail "
                "judgment about the primary task model, and low accuracy is not proof of domain "
                "invariance.",
    }
