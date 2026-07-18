"""
benchmarks/uncertainty.py — development-defined uncertainty and abstention
diagnostics for the source-held-out protocol.

Every threshold here (abstention cutoff) is selected from DEVELOPMENT data
only — held-out-source outcomes are never read by any selection function in
this module, only by the reporting functions that summarize an
already-fixed threshold's effect afterward. MC-dropout dispersion is
reported as a dispersion diagnostic, never as a calibrated confidence
interval (see pathway_hierarchical_mil.mc_dropout_predict's own docstring,
reused here unchanged).
"""

from typing import Dict, Optional, Sequence

import numpy as np


def binary_predictive_uncertainty(prob: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-subject uncertainty summaries for binary (cancer) probabilities —
    no labels involved, purely a function of the predicted probability."""
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-12, 1 - 1e-12)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    max_class_prob = np.maximum(p, 1 - p)
    energy_score = -np.log(p / (1 - p) + 1e-12)  # negative logit, a common energy-score proxy
    return {"entropy": entropy, "max_class_probability": max_class_prob, "energy_score": energy_score}


def select_abstention_threshold_from_development(
    dev_uncertainty: np.ndarray, dev_correct: np.ndarray, target_coverage: float = 0.8,
) -> Dict:
    """
    Chooses the uncertainty value below which a subject is retained, using
    ONLY development (OOF) predictions and their correctness — never
    held-out-source outcomes. target_coverage is the fraction of
    DEVELOPMENT subjects that must remain after abstention; the threshold
    is the corresponding development-uncertainty quantile.
    """
    if not (0.0 < target_coverage <= 1.0):
        raise ValueError(f"target_coverage must be in (0, 1], got {target_coverage}.")
    dev_uncertainty = np.asarray(dev_uncertainty, dtype=np.float64)
    if len(dev_uncertainty) == 0:
        return {"status": "insufficient_evidence", "reason": "no development subjects"}
    threshold = float(np.quantile(dev_uncertainty, target_coverage))
    retained = dev_uncertainty <= threshold
    dev_accuracy_retained = float(np.mean(dev_correct[retained])) if retained.any() else None
    return {
        "status": "selected", "uncertainty_threshold": threshold, "target_coverage": target_coverage,
        "development_realized_coverage": float(retained.mean()),
        "development_accuracy_at_coverage": dev_accuracy_retained,
    }


def apply_abstention_threshold(
    held_out_uncertainty: np.ndarray, held_out_correct: np.ndarray, threshold: float,
) -> Dict:
    """
    Reports coverage/performance on the held-out source AFTER a threshold
    already selected from development data is applied — this function does
    not choose the threshold, only reports its effect. The MAIN (non-
    abstention) source-held-out metric always covers every eligible
    subject; this is reported as a separate, secondary diagnostic, never a
    substitute that drops difficult subjects from the headline result.
    """
    held_out_uncertainty = np.asarray(held_out_uncertainty, dtype=np.float64)
    held_out_correct = np.asarray(held_out_correct, dtype=bool)
    retained = held_out_uncertainty <= threshold
    return {
        "threshold": threshold, "n_total": int(len(held_out_uncertainty)),
        "n_retained": int(retained.sum()), "coverage": float(retained.mean()) if len(retained) else None,
        "accuracy_at_coverage": float(held_out_correct[retained].mean()) if retained.any() else None,
        "accuracy_full_population": float(held_out_correct.mean()) if len(held_out_correct) else None,
    }


def multiclass_predictive_uncertainty(proba: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-subject uncertainty summaries for a multi-class softmax
    probability matrix ([n_subjects, n_classes]) — predictive entropy and
    maximum class probability, no labels involved."""
    p = np.clip(np.asarray(proba, dtype=np.float64), 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    entropy = -(p * np.log(p)).sum(axis=1)
    max_class_prob = p.max(axis=1)
    return {"entropy": entropy, "max_class_probability": max_class_prob}


def mc_dropout_uncertainty_report(adapter, bags: Sequence[dict], n_passes: int = 20, seed: int = 0) -> Dict:
    """Thin wrapper around pathway_hierarchical_mil.mc_dropout_predict —
    explicitly NOT a calibrated confidence interval, purely a dispersion
    diagnostic (see that function's own docstring)."""
    import torch
    from pathway_hierarchical_mil import mc_dropout_predict
    from .pathway_hierarchical_adapter import bags_to_pathway_batch

    torch.manual_seed(seed)
    batch = bags_to_pathway_batch(bags)
    out = mc_dropout_predict(
        adapter.model,
        {"expression": batch["expression"], "cell_type_ids": batch["cell_type_ids"], "cell_mask": batch["cell_mask"]},
        n_passes=n_passes,
    )
    return {
        "n_passes": n_passes,
        "cancer_proba_mean": out["cancer_mean"].tolist(),
        "cancer_proba_variance": out["cancer_variance"].tolist(),
        "cancer_entropy": out["cancer_entropy"].tolist(),
        "note": "MC-dropout dispersion is a diagnostic measure of predictive instability under "
                "stochastic dropout, not a calibrated confidence interval.",
    }
