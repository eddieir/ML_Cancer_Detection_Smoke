"""
benchmarks/cross_validation.py — grouped-CV experiment runner around
data/splitting.py::grouped_kfold.

CV always runs over the TRAIN+VAL subject pool of an ExperimentContext —
the test split is never touched here (see runner.py for the one place a
frozen final test evaluation happens, after CV has picked a configuration).
Baselines use subject-summary features (features.py); the neural model uses
its native cell/bag datasets, subset per fold by subject id.

Undefined per-fold metrics (e.g. AUROC with one class in a fold's val split)
are recorded as None, never coerced to a filler value — see metrics.py.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from data.splitting import grouped_kfold
from train import CellLevelDataset, MILEligibilityError, SubjectLevelDataset

from .baselines import CANCER_BASELINES, SMOKE_BASELINES
from .features import build_cancer_subject_features, build_smoke_subject_summary_features
from .metrics import (
    aggregate_metric,
    cancer_prediction_metrics,
    full_smoke_metrics_report,
    subject_weighted_smoke_metrics,
)
from .neural import NeuralCancerAdapter, NeuralSmokeAdapter

DEFAULT_SEEDS = [42, 43, 44]


def concat_cell_datasets(a: CellLevelDataset, b: CellLevelDataset) -> CellLevelDataset:
    """Merge two CellLevelDatasets (e.g. a context's train + val splits) into
    one pool for grouped CV. Never used to merge across train and TEST."""
    return CellLevelDataset(
        gene_matrix       = np.concatenate([a.X.numpy(), b.X.numpy()]),
        smoke_labels      = np.concatenate([a.smoke.numpy(), b.smoke.numpy()]),
        malignancy_labels = np.concatenate([a.malig.numpy(), b.malig.numpy()]),
        cell_type_ids     = np.concatenate([a.ctype.numpy(), b.ctype.numpy()]),
        exposure_dose     = np.concatenate([a.dose.numpy(), b.dose.numpy()]),
        malignancy_known  = np.concatenate([a.malig_known.numpy(), b.malig_known.numpy()]),
        subject_ids       = np.concatenate([a.subject_ids, b.subject_ids]),
        dataset_source     = np.concatenate([a.dataset_source, b.dataset_source]),
        diagnostic_mode    = a.diagnostic_mode or b.diagnostic_mode,
    )


# ─── Task A: smoke classification CV ──────────────────────────────────────────

def run_smoke_cv(
    context, model_names: Sequence[str], n_folds: int = 5,
    seeds: Sequence[int] = DEFAULT_SEEDS, device: str = "cpu",
) -> Dict:
    pool = concat_cell_datasets(context.train_cell_dataset, context.val_cell_dataset)
    num_classes = context.num_smoke_classes
    num_cell_types = context.config.get("model", {}).get("num_cell_types", 4)

    X_full, y_full, subject_ids, feature_names = build_smoke_subject_summary_features(
        pool, num_cell_types=num_cell_types, num_classes=num_classes,
    )
    subject_ids = np.array(subject_ids)

    results: Dict[str, Dict] = {name: {"folds": [], "hyperparameters": []} for name in model_names}

    for seed in seeds:
        folds = grouped_kfold(subject_ids, y_full, n_folds=n_folds, seed=seed)
        for fold_idx, fold in enumerate(folds):
            train_mask = np.isin(subject_ids, fold["train"])
            val_mask = np.isin(subject_ids, fold["val"])
            Xtr, ytr = X_full[train_mask], y_full[train_mask]
            Xva, yva = X_full[val_mask], y_full[val_mask]
            val_subj_pool = pool.subset_by_subjects(fold["val"])

            for name in model_names:
                fold_record = {"seed": seed, "fold": fold_idx,
                                "classes_absent_from_val": fold["classes_absent_from_val"],
                                "stratified": fold["stratified"]}
                if name == "neural":
                    train_subj_pool = pool.subset_by_subjects(fold["train"])
                    adapter = NeuralSmokeAdapter(context.config, device=device)
                    adapter.fit(context, train_subj_pool, val_subj_pool, seed=seed)
                    preds = adapter.predict(val_subj_pool)
                    cell_report = full_smoke_metrics_report(val_subj_pool.smoke.numpy(), preds, num_classes)
                    subj_report = subject_weighted_smoke_metrics(
                        val_subj_pool.smoke.numpy(), preds, val_subj_pool.subject_ids, num_classes,
                    )
                    fold_record["hyperparameters"] = adapter.metadata()
                else:
                    model = SMOKE_BASELINES[name]()
                    model.fit(Xtr, ytr, seed=seed)
                    preds = model.predict(Xva)
                    cell_report = full_smoke_metrics_report(yva, preds, num_classes)
                    # subject-summary features are already one row per subject,
                    # so cell-weighted == subject-weighted for these baselines.
                    subj_report = cell_report
                    fold_record["hyperparameters"] = model.metadata()

                fold_record["cell_weighted_macro_f1"] = cell_report["macro_f1"]
                fold_record["subject_weighted_macro_f1"] = subj_report["macro_f1"]
                fold_record["balanced_accuracy"] = cell_report.get("balanced_accuracy")
                fold_record["weighted_f1"] = cell_report["weighted_f1"]
                fold_record["per_class"] = cell_report.get("per_class")
                results[name]["folds"].append(fold_record)

    for name in model_names:
        folds = results[name]["folds"]
        results[name]["subject_weighted_macro_f1"] = aggregate_metric(
            [f["subject_weighted_macro_f1"] for f in folds],
        )
        results[name]["cell_weighted_macro_f1"] = aggregate_metric(
            [f["cell_weighted_macro_f1"] for f in folds],
        )
        results[name]["n_folds_run"] = len(folds)

    return {
        "task": "smoke_classification", "primary_metric": "subject_weighted_macro_f1",
        "n_folds_requested": n_folds, "seeds": list(seeds),
        "feature_names_n": len(feature_names), "results": results,
    }


# ─── Task B: cancer prediction CV ─────────────────────────────────────────────

def run_cancer_cv(
    context, model_names: Sequence[str], n_folds: int = 5,
    seeds: Sequence[int] = DEFAULT_SEEDS, device: str = "cpu", pooling: Optional[str] = None,
) -> Dict:
    pool_bags = list(context.train_bags) + list(context.val_bags)
    known_bags = [b for b in pool_bags if b.get("cancer_label_known")]
    num_cell_types = context.config.get("model", {}).get("num_cell_types", 4)

    X_full, y_full, subject_ids, feature_names = build_cancer_subject_features(known_bags, num_cell_types)
    subject_ids = np.array(subject_ids)
    bags_by_subject = {str(b["subject_id"]): b for b in known_bags}

    mil_names = [n for n in model_names if n in ("neural", "mean_mil", "max_mil", "attention_mil")]
    baseline_names = [n for n in model_names if n not in mil_names]

    results: Dict[str, Dict] = {name: {"folds": []} for name in model_names}
    pooling_for = {
        "mean_mil": "mean", "max_mil": "max", "attention_mil": "attention", "neural": pooling or "attention",
    }

    for seed in seeds:
        folds = grouped_kfold(subject_ids, y_full, n_folds=n_folds, seed=seed)
        for fold_idx, fold in enumerate(folds):
            train_mask = np.isin(subject_ids, fold["train"])
            val_mask = np.isin(subject_ids, fold["val"])
            Xtr, ytr = X_full[train_mask], y_full[train_mask]
            Xva, yva = X_full[val_mask], y_full[val_mask]

            for name in baseline_names:
                model = CANCER_BASELINES[name]()
                model.fit(Xtr, ytr, seed=seed)
                proba = model.predict_proba(Xva)[:, 1]
                report = cancer_prediction_metrics(yva, proba)
                report.update({"seed": seed, "fold": fold_idx, "hyperparameters": model.metadata()})
                results[name]["folds"].append(report)

            for name in mil_names:
                train_bags_fold = [bags_by_subject[s] for s in fold["train"]]
                val_bags_fold = [bags_by_subject[s] for s in fold["val"]]
                train_sd = SubjectLevelDataset(train_bags_fold)
                val_sd = SubjectLevelDataset(val_bags_fold)
                try:
                    adapter = NeuralCancerAdapter(pooling=pooling_for[name], device=device)
                    adapter.fit(
                        context, context.train_cell_dataset, context.val_cell_dataset,
                        train_sd, val_sd, seed=seed, pretrain_epochs=2,
                    )
                    proba = adapter.predict_proba(val_sd)
                    y_val_ordered = np.array([b["cancer_label"] for b in val_sd.bags])
                    report = cancer_prediction_metrics(y_val_ordered, proba)
                    report.update({"seed": seed, "fold": fold_idx, "hyperparameters": adapter.metadata()})
                except MILEligibilityError as e:
                    # A fold this small failing MIL eligibility (see
                    # train.check_mil_eligibility) is an honest, expected
                    # outcome of grouped CV on a small real cohort — record
                    # it as an undefined fold with its reason, don't crash
                    # the whole CV run or silently drop the fold.
                    report = {
                        "auroc": None, "auprc": None, "auroc_auprc_undefined_reason": str(e),
                        "seed": seed, "fold": fold_idx, "hyperparameters": None,
                    }
                results[name]["folds"].append(report)

    for name in model_names:
        folds = results[name]["folds"]
        results[name]["auroc"] = aggregate_metric([f["auroc"] for f in folds])
        results[name]["auprc"] = aggregate_metric([f["auprc"] for f in folds])
        results[name]["n_folds_run"] = len(folds)

    return {
        "task": "cancer_prediction", "primary_metric": "auroc",
        "n_folds_requested": n_folds, "seeds": list(seeds),
        "feature_names_n": len(feature_names), "results": results,
    }
