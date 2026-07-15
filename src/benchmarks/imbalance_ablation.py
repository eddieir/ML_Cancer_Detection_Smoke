"""
benchmarks/imbalance_ablation.py — development-only comparison of Phase 2
class-imbalance strategies (data/sampling.py, model.py's FocalLoss).

Reuses the SAME grouped-subject-CV machinery cross_validation.py's
run_smoke_cv already uses (data/splitting.py::grouped_kfold,
fold_preprocessing.py's per-fold artifact refit), so every strategy sees
identical outer folds, identical preprocessing boundaries, and the same
candidate architecture — only train.smoke_imbalance differs between
strategies. CV always runs over context's TRAIN+VAL subject pool; the test
split is never touched here (same contract as run_smoke_cv/run_cancer_cv).

Implemented as a focused, standalone function rather than folded into
cross_validation.py's nested-hyperparameter-search machinery
(select_nested_hyperparameters_with_refit) because imbalance strategy is a
qualitatively different axis than a hyperparameter grid: this compares
named, fully-specified STRATEGIES against each other directly, it does not
select one strategy's own hyperparameters via inner CV.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from train import assert_disjoint_subjects
from data.splitting import grouped_kfold

from .cross_validation import DEFAULT_SEEDS, _fold_context, _majority_label_by_subject
from .features import cap_cell_dataset
from .fold_preprocessing import artifact_fingerprint, fold_train_val_datasets, require_normalized_adata
from .metrics import (
    aggregate_metric,
    aggregate_metric_by_seed,
    full_smoke_metrics_report,
    subject_weighted_smoke_metrics,
)
from .neural import NeuralSmokeAdapter

# The five ablations requirement 6 (STEP 12) asks for at minimum. Every
# strategy is a COMPLETE, self-consistent train.smoke_imbalance override —
# never a partial diff — so results are never contaminated by a stale key
# left over from a previous strategy dict.
IMBALANCE_ABLATION_STRATEGIES: Dict[str, dict] = {
    "natural_no_weight": {
        "sampler": "shuffle", "class_selection": "uniform", "subject_selection": "uniform",
        "batches_per_epoch": None, "samples_per_epoch": None, "cells_per_subject_per_batch": None,
        "replacement": True, "class_weighting": "none", "loss": "cross_entropy",
        "focal_gamma": 2.0, "focal_alpha_mode": "class_weights", "seed_offset": 0,
    },
    "natural_inverse_frequency": {
        "sampler": "shuffle", "class_selection": "uniform", "subject_selection": "uniform",
        "batches_per_epoch": None, "samples_per_epoch": None, "cells_per_subject_per_batch": None,
        "replacement": True, "class_weighting": "inverse_frequency", "loss": "cross_entropy",
        "focal_gamma": 2.0, "focal_alpha_mode": "class_weights", "seed_offset": 0,
    },
    "subject_balanced_no_weight": {
        "sampler": "subject_balanced", "class_selection": "uniform", "subject_selection": "uniform",
        "batches_per_epoch": None, "samples_per_epoch": None, "cells_per_subject_per_batch": 8,
        "replacement": True, "class_weighting": "none", "loss": "cross_entropy",
        "focal_gamma": 2.0, "focal_alpha_mode": "class_weights", "seed_offset": 0,
    },
    "subject_balanced_inverse_frequency": {
        "sampler": "subject_balanced", "class_selection": "uniform", "subject_selection": "uniform",
        "batches_per_epoch": None, "samples_per_epoch": None, "cells_per_subject_per_batch": 8,
        "replacement": True, "class_weighting": "inverse_frequency", "loss": "cross_entropy",
        "focal_gamma": 2.0, "focal_alpha_mode": "class_weights", "seed_offset": 0,
    },
    # Sampling already does the class correction here, so alpha is
    # deliberately "none" — avoids stacking inverse-frequency loss
    # weighting on top of inverse-frequency-ish sampling (the
    # "double-correction" risk documented in README.md/model.py).
    "subject_balanced_focal": {
        "sampler": "subject_balanced", "class_selection": "uniform", "subject_selection": "uniform",
        "batches_per_epoch": None, "samples_per_epoch": None, "cells_per_subject_per_batch": 8,
        "replacement": True, "class_weighting": "none", "loss": "focal",
        "focal_gamma": 2.0, "focal_alpha_mode": "none", "seed_offset": 0,
    },
}
BASELINE_STRATEGY = "natural_no_weight"


def run_smoke_imbalance_ablation(
    context,
    strategies:              Optional[Sequence[str]] = None,
    n_folds:                  int = 5,
    seeds:                    Sequence[int] = DEFAULT_SEEDS,
    device:                   str = "cpu",
    max_cells_per_subject:     Optional[int] = None,
) -> Dict:
    """
    Compare named smoke-imbalance strategies (IMBALANCE_ABLATION_STRATEGIES)
    with identical outer folds/seeds/preprocessing/architecture, development
    data only. Never touches context's test split; never uses this
    comparison's outcome to invoke the frozen-test guard.
    """
    strategies = list(strategies) if strategies is not None else list(IMBALANCE_ABLATION_STRATEGIES)
    unknown = set(strategies) - set(IMBALANCE_ABLATION_STRATEGIES)
    if unknown:
        raise ValueError(f"run_smoke_imbalance_ablation: unknown strategy name(s) {sorted(unknown)} "
                          f"— must be a subset of {sorted(IMBALANCE_ABLATION_STRATEGIES)}")
    if not strategies:
        raise ValueError("run_smoke_imbalance_ablation: strategies must be non-empty.")

    normalized_adata = require_normalized_adata(context)
    num_classes = context.num_smoke_classes
    n_hvgs = context.preprocessing_artifact.n_hvgs
    max_cells_per_subject = (
        max_cells_per_subject if max_cells_per_subject is not None
        else context.config.get("benchmarks", {}).get("max_cells_per_subject_cv", 200)
    )

    pool_subjects = sorted(set(context.subjects_for("train")) | set(context.subjects_for("val")))
    label_by_subject = _majority_label_by_subject(normalized_adata, pool_subjects, num_classes)
    subject_ids = np.array(pool_subjects)
    y_full = np.array([label_by_subject[s] for s in pool_subjects])

    results: Dict[str, Dict] = {name: {"folds": []} for name in strategies}

    for seed in seeds:
        folds = grouped_kfold(subject_ids, y_full, n_folds=n_folds, seed=seed)
        for fold_idx, fold in enumerate(folds):
            artifact, train_ds, val_ds = fold_train_val_datasets(
                context, fold["train"], fold["val"], n_hvgs=n_hvgs,
            )
            assert_disjoint_subjects(train_ds, val_ds, names=["fold_train", "fold_val"])
            fp = artifact_fingerprint(artifact)
            # Same cap applied identically to every strategy in this fold —
            # cap_cell_dataset is deterministic given (seed, max cells), so
            # every strategy trains/evaluates on the exact same fold cells.
            train_ds_capped = cap_cell_dataset(train_ds, max_cells_per_subject, seed=seed)
            val_ds_capped = cap_cell_dataset(val_ds, max_cells_per_subject, seed=seed)

            for strategy_name in strategies:
                strategy_cfg = IMBALANCE_ABLATION_STRATEGIES[strategy_name]
                fold_context = _fold_context(context, artifact, train_ds_capped, val_ds_capped)
                overridden = dict(fold_context.config)
                overridden["train"] = dict(overridden.get("train", {}))
                overridden["train"]["smoke_imbalance"] = dict(strategy_cfg)
                fold_context.config = overridden

                adapter = NeuralSmokeAdapter(fold_context.config, device=device)
                adapter.fit(fold_context, train_ds_capped, val_ds_capped, seed=seed)
                preds = adapter.predict(val_ds_capped)
                cell_report = full_smoke_metrics_report(val_ds_capped.smoke.numpy(), preds, num_classes)
                subj_report = subject_weighted_smoke_metrics(
                    val_ds_capped.smoke.numpy(), preds, val_ds_capped.subject_ids, num_classes,
                )
                results[strategy_name]["folds"].append({
                    "seed": seed, "fold": fold_idx,
                    "strategy": strategy_name, "strategy_config": strategy_cfg,
                    "classes_absent_from_val": fold["classes_absent_from_val"],
                    "preprocessing_fingerprint": fp,
                    "subject_weighted_macro_f1": subj_report["macro_f1"],
                    "cell_weighted_macro_f1": cell_report["macro_f1"],
                    "balanced_accuracy": cell_report.get("balanced_accuracy"),
                    "per_class": cell_report.get("per_class"),
                    "hyperparameters": adapter.metadata(),
                })

    for name in strategies:
        folds = results[name]["folds"]
        fold_seeds = [f["seed"] for f in folds]
        results[name]["subject_weighted_macro_f1"] = aggregate_metric(
            [f["subject_weighted_macro_f1"] for f in folds],
        )
        results[name]["subject_weighted_macro_f1_by_seed"] = aggregate_metric_by_seed(
            [f["subject_weighted_macro_f1"] for f in folds], fold_seeds,
        )
        results[name]["n_folds_run"] = len(folds)

    baseline = BASELINE_STRATEGY if BASELINE_STRATEGY in strategies else strategies[0]
    paired = _paired_fold_differences(results, baseline)

    return {
        "task": "smoke_imbalance_ablation", "primary_metric": "subject_weighted_macro_f1",
        "n_folds_requested": n_folds, "seeds": list(seeds), "strategies": strategies,
        "baseline_strategy": baseline,
        "results": results, "paired_fold_differences": paired,
        "development_only": True,
        "note": "Development-only comparison over context's train+val subject pool "
                "(grouped subject CV) — no test data was accessed and this comparison "
                "does not invoke the frozen-test guard. A strategy is not 'better' "
                "merely because its mean subject_weighted_macro_f1 is nominally "
                "higher than another's — inspect paired_fold_differences (per "
                "seed/fold) and each strategy's per-class support before drawing any "
                "conclusion, and treat differences within noise as inconclusive.",
    }


def _paired_fold_differences(
    results: Dict[str, Dict], baseline: str,
) -> Dict[str, List[Optional[float]]]:
    """
    Per-(seed, fold) subject_weighted_macro_f1 difference vs `baseline`, for
    every other strategy — lets a caller see whether a strategy wins on
    EVERY fold or only on average, rather than collapsing straight to a
    single mean comparison.
    """
    base_folds = {(f["seed"], f["fold"]): f["subject_weighted_macro_f1"] for f in results[baseline]["folds"]}
    out: Dict[str, List[Optional[float]]] = {}
    for name, r in results.items():
        if name == baseline:
            continue
        diffs = []
        for f in r["folds"]:
            key = (f["seed"], f["fold"])
            b = base_folds.get(key)
            v = f["subject_weighted_macro_f1"]
            diffs.append((v - b) if (b is not None and v is not None) else None)
        out[name] = diffs
    return out
