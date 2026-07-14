"""
benchmarks/cross_validation.py — grouped-CV experiment runner around
data/splitting.py::grouped_kfold.

CV always runs over the TRAIN+VAL subject pool of an ExperimentContext —
the test split is never touched here (see runner.py for the one place a
frozen final test evaluation happens, after CV has picked a configuration).

Preprocessing (gene scaling + HVG selection) is refit PER FOLD from
context.normalized_adata_for_refit, using only that fold's training
subjects (see fold_preprocessing.py) — reusing the context's outer
PreprocessingArtifact across folds would leak an inner-CV-validation
subject's influence on scaling/HVG selection into its own "held-out"
evaluation. Baselines use subject-summary features built from the fold's
own refit data; the neural/MIL models train on the SAME fold-specific cell
and bag datasets, so Phase 1 encoder pretraining never sees a cancer-fold's
validation subjects either (the cross-task leakage the pooled outer
train+val cell dataset used to allow).

Undefined per-fold metrics (e.g. AUROC with one class in a fold's val
split) are recorded as None, never coerced to a filler value — see metrics.py.
"""

import dataclasses
from typing import Dict, List, Optional, Sequence

import numpy as np

from data.splitting import grouped_kfold
from train import (
    CellLevelDataset,
    MILEligibilityError,
    SubjectLevelDataset,
    assert_disjoint_subjects,
    validate_experiment_partitions,
)

from .baselines import CANCER_BASELINES, CANCER_SEARCH_SPACE, SMOKE_BASELINES, SMOKE_SEARCH_SPACE, positive_class_proba
from .features import build_cancer_subject_features, build_smoke_subject_summary_features, cap_cell_dataset
from .fold_preprocessing import (
    artifact_fingerprint,
    bags_from_fold_cell_dataset,
    fold_train_val_datasets,
    require_normalized_adata,
)
from .hyperparameter_search import build_param_grid, select_nested_hyperparameters_with_refit
from .metrics import (
    aggregate_metric,
    aggregate_metric_by_seed,
    cancer_prediction_metrics,
    full_smoke_metrics_report,
    subject_weighted_smoke_metrics,
)
from .neural import NeuralCancerAdapter, NeuralSmokeAdapter

DEFAULT_SEEDS = [42, 43, 44]

# Bounded, deterministic neural/MIL config search — deliberately NOT a large
# hyperparameter grid (section 6: "do not perform a large search... use the
# current configuration plus at most a small predefined comparison"). Each
# candidate is compared via the SAME inner-grouped-CV/refit-per-fold
# machinery as the classical baselines (select_nested_hyperparameters_with_refit).
NEURAL_SEARCH_SPACE = {"phase1_epochs": [1, 2]}
MIL_SEARCH_SPACE = {"pretrain_epochs": [1, 2]}
# Inner CV cost multiplies across outer folds x seeds x candidates x inner
# folds; kept small since inner folds retrain a real model per candidate.
DEFAULT_INNER_FOLDS = 2


def concat_cell_datasets(a: CellLevelDataset, b: CellLevelDataset) -> CellLevelDataset:
    """Merge two CellLevelDatasets (e.g. an outer context's train + val
    splits) into one pool. Used by ood.py's per-source held-out evaluation
    and by tests; run_smoke_cv/run_cancer_cv build fold data straight from
    normalized_adata_for_refit instead (see fold_preprocessing.py) so they
    no longer need this for CV itself."""
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


def _majority_label_by_subject(normalized_adata, subjects: Sequence[str], num_classes: int) -> Dict[str, int]:
    obs = normalized_adata.obs
    subj_series = obs["subject_id"].astype(str)
    smoke_series = obs["smoke_type"].astype(int)
    out = {}
    for s in subjects:
        vals = smoke_series[subj_series == str(s)].values
        out[s] = int(np.bincount(vals, minlength=num_classes).argmax())
    return out


def _smoke_baseline_fit_score_fn(baseline_cls, num_cell_types: int, num_classes: int):
    """fit_score_fn for select_nested_hyperparameters_with_refit: fits
    baseline_cls(**params) on the inner-train fold's own subject-summary
    features (built from an artifact refit on ONLY that inner fold's train
    subjects) and scores subject-summary macro-F1 on the inner-val fold."""
    def fn(params, artifact, inner_train_ds, inner_val_ds, seed):
        Xtr, ytr, _, _ = build_smoke_subject_summary_features(inner_train_ds, num_cell_types, num_classes)
        Xva, yva, _, _ = build_smoke_subject_summary_features(inner_val_ds, num_cell_types, num_classes)
        model = baseline_cls(**params).fit(Xtr, ytr, seed=seed)
        preds = model.predict(Xva)
        return full_smoke_metrics_report(yva, preds, num_classes)["macro_f1"]
    return fn


def _neural_smoke_fit_score_fn(context, device: str, num_classes: int, max_cells_per_subject: int):
    """fit_score_fn variant for the neural adapter: params carries a small
    config override (e.g. {"phase1_epochs": N}), applied on top of
    context.config for exactly this candidate's fit."""
    def fn(params, artifact, inner_train_ds, inner_val_ds, seed):
        train_capped = cap_cell_dataset(inner_train_ds, max_cells_per_subject, seed=seed)
        val_capped = cap_cell_dataset(inner_val_ds, max_cells_per_subject, seed=seed)
        fold_ctx = _fold_context(context, artifact, train_capped, val_capped)
        # dataclasses.replace only shallow-copies the dataclass itself — the
        # nested config dict is still the SAME object as context.config, so
        # this candidate's override must go into a fresh dict, never mutate
        # the shared one in place (which would leak into other candidates).
        overridden_config = dict(fold_ctx.config)
        overridden_config["train"] = dict(overridden_config.get("train", {}), **params)
        fold_ctx.config = overridden_config
        adapter = NeuralSmokeAdapter(fold_ctx.config, device=device)
        adapter.fit(fold_ctx, train_capped, val_capped, seed=seed)
        preds = adapter.predict(val_capped)
        return full_smoke_metrics_report(val_capped.smoke.numpy(), preds, num_classes)["macro_f1"]
    return fn


def _auroc_or_none(y_val: np.ndarray, prob_val: np.ndarray) -> Optional[float]:
    """AUROC, or None (never 0.5) when the validation rows carry only one
    class — a real, expected outcome of small inner/outer CV folds."""
    if len(set(np.asarray(y_val).tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_val, prob_val)


def _cancer_baseline_fit_score_fn(baseline_cls, num_cell_types: int, outcomes_by_subject: dict, min_cells: int):
    def fn(params, artifact, inner_train_ds, inner_val_ds, seed):
        train_bags = bags_from_fold_cell_dataset(inner_train_ds, outcomes_by_subject, min_cells)
        val_bags = bags_from_fold_cell_dataset(inner_val_ds, outcomes_by_subject, min_cells)
        if not train_bags or not val_bags:
            return None
        Xtr, ytr, _, _ = build_cancer_subject_features(train_bags, num_cell_types)
        Xva, yva, _, _ = build_cancer_subject_features(val_bags, num_cell_types)
        model = baseline_cls(**params).fit(Xtr, ytr, seed=seed)
        return _auroc_or_none(yva, positive_class_proba(model, Xva))
    return fn


def _mil_fit_score_fn(context, pooling: str, device: str, outcomes_by_subject: dict, min_cells: int):
    def fn(params, artifact, inner_train_ds, inner_val_ds, seed):
        train_bags = bags_from_fold_cell_dataset(inner_train_ds, outcomes_by_subject, min_cells)
        val_bags = bags_from_fold_cell_dataset(inner_val_ds, outcomes_by_subject, min_cells)
        if not train_bags or not val_bags:
            return None
        train_sd = SubjectLevelDataset(train_bags)
        val_sd = SubjectLevelDataset(val_bags)
        try:
            validate_experiment_partitions(
                train_cell_dataset=inner_train_ds, val_cell_dataset=inner_val_ds,
                train_subject_dataset=train_sd, val_subject_dataset=val_sd,
            )
            fold_ctx = _fold_context(context, artifact, inner_train_ds, inner_val_ds)
            adapter = NeuralCancerAdapter(pooling=pooling, device=device)
            adapter.fit(fold_ctx, inner_train_ds, inner_val_ds, train_sd, val_sd,
                        seed=seed, pretrain_epochs=params.get("pretrain_epochs"))
            proba = adapter.predict_proba(val_sd)
            y_val_ordered = np.array([b["cancer_label"] for b in val_sd.bags])
            return _auroc_or_none(y_val_ordered, proba)
        except MILEligibilityError:
            return None
    return fn


def _fold_context(context, artifact, train_cell_dataset, val_cell_dataset):
    """Shallow-copied ExperimentContext with the fold's own artifact/cell
    datasets swapped in, so Trainer.from_experiment_context builds a model
    with the fold's own input_dim/gene space instead of the outer one."""
    return dataclasses.replace(
        context, preprocessing_artifact=artifact,
        train_cell_dataset=train_cell_dataset, val_cell_dataset=val_cell_dataset,
    )


# ─── Task A: smoke classification CV ──────────────────────────────────────────

def run_smoke_cv(
    context, model_names: Sequence[str], n_folds: int = 5,
    seeds: Sequence[int] = DEFAULT_SEEDS, device: str = "cpu",
    max_cells_per_subject: Optional[int] = None,
) -> Dict:
    normalized_adata = require_normalized_adata(context)
    num_classes = context.num_smoke_classes
    num_cell_types = context.config.get("model", {}).get("num_cell_types", 4)
    n_hvgs = context.preprocessing_artifact.n_hvgs
    max_cells_per_subject = (
        max_cells_per_subject
        if max_cells_per_subject is not None
        else context.config.get("benchmarks", {}).get("max_cells_per_subject_cv", 200)
    )

    pool_subjects = sorted(set(context.subjects_for("train")) | set(context.subjects_for("val")))
    label_by_subject = _majority_label_by_subject(normalized_adata, pool_subjects, num_classes)
    subject_ids = np.array(pool_subjects)
    y_full = np.array([label_by_subject[s] for s in pool_subjects])

    results: Dict[str, Dict] = {name: {"folds": []} for name in model_names}

    for seed in seeds:
        folds = grouped_kfold(subject_ids, y_full, n_folds=n_folds, seed=seed)
        for fold_idx, fold in enumerate(folds):
            artifact, train_ds, val_ds = fold_train_val_datasets(context, fold["train"], fold["val"], n_hvgs=n_hvgs)
            assert_disjoint_subjects(train_ds, val_ds, names=["fold_train", "fold_val"])
            fp = artifact_fingerprint(artifact)

            Xtr, ytr, _, feature_names = build_smoke_subject_summary_features(train_ds, num_cell_types, num_classes)
            Xva, yva, _, _ = build_smoke_subject_summary_features(val_ds, num_cell_types, num_classes)

            for name in model_names:
                fold_record = {"seed": seed, "fold": fold_idx,
                                "classes_absent_from_val": fold["classes_absent_from_val"],
                                "stratified": fold["stratified"],
                                "preprocessing_fingerprint": fp,
                                "fit_subject_ids": sorted(str(s) for s in fold["train"]),
                                "gene_list_n": len(artifact.gene_list)}
                if name == "neural":
                    # Cap each split's per-subject cell count independently
                    # (never using the other split's data to decide what to
                    # keep) so a subject with far more cells than others
                    # cannot dominate a Phase 1 epoch. Deterministic given
                    # seed — same fold + same seed always keeps the same cells.
                    train_ds_capped = cap_cell_dataset(train_ds, max_cells_per_subject, seed=seed)
                    val_ds_capped = cap_cell_dataset(val_ds, max_cells_per_subject, seed=seed)

                    # Bounded, deterministic config search (section 6) —
                    # inner-CV selected ENTIRELY from this outer fold's own
                    # training subjects (fold["train"]), each inner candidate
                    # refitting its own preprocessing artifact from only its
                    # inner-train subjects (select_nested_hyperparameters_with_refit).
                    neural_hp = select_nested_hyperparameters_with_refit(
                        context, fold["train"], label_by_subject,
                        build_param_grid(NEURAL_SEARCH_SPACE),
                        fit_score_fn=_neural_smoke_fit_score_fn(context, device, num_classes, max_cells_per_subject),
                        seed=seed, n_inner_folds=DEFAULT_INNER_FOLDS, n_hvgs=n_hvgs,
                    )
                    fold_context = _fold_context(context, artifact, train_ds_capped, val_ds_capped)
                    selected_config = dict(fold_context.config)
                    selected_config["train"] = dict(selected_config.get("train", {}), **neural_hp["selected_params"])
                    fold_context.config = selected_config
                    adapter = NeuralSmokeAdapter(fold_context.config, device=device)
                    adapter.fit(fold_context, train_ds_capped, val_ds_capped, seed=seed)
                    preds = adapter.predict(val_ds_capped)
                    cell_report = full_smoke_metrics_report(val_ds_capped.smoke.numpy(), preds, num_classes)
                    subj_report = subject_weighted_smoke_metrics(
                        val_ds_capped.smoke.numpy(), preds, val_ds_capped.subject_ids, num_classes,
                    )
                    fold_record["hyperparameters"] = adapter.metadata()
                    fold_record["hyperparameter_search"] = neural_hp
                    fold_record["feature_mode"] = "cell_capped"
                    fold_record["max_cells_per_subject"] = max_cells_per_subject
                    fold_record["n_cells_before_cap"] = {"train": len(train_ds), "val": len(val_ds)}
                    fold_record["n_cells_after_cap"] = {"train": len(train_ds_capped), "val": len(val_ds_capped)}
                else:
                    # Nested, leakage-free hyperparameter selection ENTIRELY
                    # from this outer fold's own training subjects — never
                    # this fold's val subjects, never any other fold's or the
                    # outer context's val/test subjects (blocker 4).
                    hp_search = select_nested_hyperparameters_with_refit(
                        context, fold["train"], label_by_subject,
                        build_param_grid(SMOKE_SEARCH_SPACE.get(name, {})),
                        fit_score_fn=_smoke_baseline_fit_score_fn(SMOKE_BASELINES[name], num_cell_types, num_classes),
                        seed=seed, n_inner_folds=DEFAULT_INNER_FOLDS, n_hvgs=n_hvgs,
                    )
                    model = SMOKE_BASELINES[name](**hp_search["selected_params"])
                    model.fit(Xtr, ytr, seed=seed)
                    preds = model.predict(Xva)
                    cell_report = full_smoke_metrics_report(yva, preds, num_classes)
                    # subject-summary features are already one row per subject:
                    # cell-weighted is NOT APPLICABLE here (there is no per-cell
                    # prediction to weight), never claimed equal to subject-weighted.
                    subj_report = cell_report
                    fold_record["hyperparameters"] = model.metadata()
                    fold_record["hyperparameter_search"] = hp_search
                    fold_record["evaluation_mode"] = "subject_summary"
                    fold_record["feature_mode"] = "subject_summary"

                fold_record["cell_weighted_macro_f1"] = cell_report["macro_f1"] if name == "neural" else None
                fold_record["cell_weighted_macro_f1_not_applicable"] = name != "neural"
                fold_record["subject_weighted_macro_f1"] = subj_report["macro_f1"]
                fold_record["balanced_accuracy"] = cell_report.get("balanced_accuracy")
                fold_record["weighted_f1"] = cell_report["weighted_f1"]
                fold_record["per_class"] = cell_report.get("per_class")
                results[name]["folds"].append(fold_record)

    for name in model_names:
        folds = results[name]["folds"]
        fold_seeds = [f["seed"] for f in folds]
        results[name]["subject_weighted_macro_f1"] = aggregate_metric(
            [f["subject_weighted_macro_f1"] for f in folds],
        )
        results[name]["subject_weighted_macro_f1_by_seed"] = aggregate_metric_by_seed(
            [f["subject_weighted_macro_f1"] for f in folds], fold_seeds,
        )
        results[name]["cell_weighted_macro_f1"] = aggregate_metric(
            [f["cell_weighted_macro_f1"] for f in folds],
        )
        results[name]["n_folds_run"] = len(folds)

    return {
        "task": "smoke_classification", "primary_metric": "subject_weighted_macro_f1",
        "n_folds_requested": n_folds, "seeds": list(seeds),
        "results": results,
    }


# ─── Task B: cancer prediction CV ─────────────────────────────────────────────

def run_cancer_cv(
    context, model_names: Sequence[str], n_folds: int = 5,
    seeds: Sequence[int] = DEFAULT_SEEDS, device: str = "cpu", pooling: Optional[str] = None,
) -> Dict:
    normalized_adata = require_normalized_adata(context)
    all_bags = list(context.train_bags) + list(context.val_bags)
    outcomes_by_subject = {str(b["subject_id"]): b["cancer_label"] for b in all_bags if b.get("cancer_label_known")}
    known_subjects = sorted(outcomes_by_subject.keys())
    if len(known_subjects) < 2:
        raise ValueError("run_cancer_cv: fewer than 2 subjects with a known cancer outcome in train+val")

    num_cell_types = context.config.get("model", {}).get("num_cell_types", 4)
    n_hvgs = context.preprocessing_artifact.n_hvgs
    min_cells = context.config.get("data", context.config).get("min_cells_per_subject", 50)

    subject_ids = np.array(known_subjects)
    y_full = np.array([outcomes_by_subject[s] for s in known_subjects])

    mil_names = [n for n in model_names if n in ("neural", "mean_mil", "max_mil", "attention_mil")]
    baseline_names = [n for n in model_names if n not in mil_names]
    pooling_for = {
        "mean_mil": "mean", "max_mil": "max", "attention_mil": "attention", "neural": pooling or "attention",
    }

    results: Dict[str, Dict] = {name: {"folds": []} for name in model_names}

    for seed in seeds:
        folds = grouped_kfold(subject_ids, y_full, n_folds=n_folds, seed=seed)
        for fold_idx, fold in enumerate(folds):
            artifact, train_cell_ds, val_cell_ds = fold_train_val_datasets(
                context, fold["train"], fold["val"], n_hvgs=n_hvgs,
            )
            fp = artifact_fingerprint(artifact)
            train_bags_fold = bags_from_fold_cell_dataset(train_cell_ds, outcomes_by_subject, min_cells)
            val_bags_fold = bags_from_fold_cell_dataset(val_cell_ds, outcomes_by_subject, min_cells)
            if not train_bags_fold or not val_bags_fold:
                for name in model_names:
                    results[name]["folds"].append({
                        "auroc": None, "auprc": None,
                        "auroc_auprc_undefined_reason": "fold has no subject with >= min_cells_per_subject cells",
                        "seed": seed, "fold": fold_idx, "preprocessing_fingerprint": fp,
                    })
                continue

            Xtr, ytr, _, _ = build_cancer_subject_features(train_bags_fold, num_cell_types)
            Xva, yva, _, _ = build_cancer_subject_features(val_bags_fold, num_cell_types)

            for name in baseline_names:
                # Nested selection ENTIRELY from this outer fold's own
                # training subjects (fold["train"]) — never fold["val"],
                # never any other fold's or the outer val/test subjects.
                hp_search = select_nested_hyperparameters_with_refit(
                    context, fold["train"], outcomes_by_subject,
                    build_param_grid(CANCER_SEARCH_SPACE.get(name, {})),
                    fit_score_fn=_cancer_baseline_fit_score_fn(CANCER_BASELINES[name], num_cell_types,
                                                                outcomes_by_subject, min_cells),
                    seed=seed, n_inner_folds=DEFAULT_INNER_FOLDS, n_hvgs=n_hvgs,
                )
                model = CANCER_BASELINES[name](**hp_search["selected_params"])
                model.fit(Xtr, ytr, seed=seed)
                proba = positive_class_proba(model, Xva)
                report = cancer_prediction_metrics(yva, proba)
                report.update({"seed": seed, "fold": fold_idx, "hyperparameters": model.metadata(),
                               "hyperparameter_search": hp_search,
                               "preprocessing_fingerprint": fp, "train_classes_present": sorted(set(ytr.tolist()))})
                results[name]["folds"].append(report)

            for name in mil_names:
                train_sd = SubjectLevelDataset(train_bags_fold)
                val_sd = SubjectLevelDataset(val_bags_fold)
                try:
                    validate_experiment_partitions(
                        train_cell_dataset=train_cell_ds, val_cell_dataset=val_cell_ds,
                        train_subject_dataset=train_sd, val_subject_dataset=val_sd,
                    )
                    mil_hp = select_nested_hyperparameters_with_refit(
                        context, fold["train"], outcomes_by_subject,
                        build_param_grid(MIL_SEARCH_SPACE),
                        fit_score_fn=_mil_fit_score_fn(context, pooling_for[name], device, outcomes_by_subject, min_cells),
                        seed=seed, n_inner_folds=DEFAULT_INNER_FOLDS, n_hvgs=n_hvgs,
                    )
                    fold_context = _fold_context(context, artifact, train_cell_ds, val_cell_ds)
                    adapter = NeuralCancerAdapter(pooling=pooling_for[name], device=device)
                    adapter.fit(
                        fold_context, train_cell_ds, val_cell_ds,
                        train_sd, val_sd, seed=seed,
                        pretrain_epochs=mil_hp["selected_params"].get("pretrain_epochs", 2),
                    )
                    proba = adapter.predict_proba(val_sd)
                    y_val_ordered = np.array([b["cancer_label"] for b in val_sd.bags])
                    report = cancer_prediction_metrics(y_val_ordered, proba)
                    report.update({"seed": seed, "fold": fold_idx, "hyperparameters": adapter.metadata(),
                                   "hyperparameter_search": mil_hp,
                                   "preprocessing_fingerprint": fp})
                except MILEligibilityError as e:
                    # A fold this small failing MIL eligibility (see
                    # train.check_mil_eligibility) is an honest, expected
                    # outcome of grouped CV on a small real cohort — record
                    # it as an undefined fold with its reason, don't crash
                    # the whole CV run or silently drop the fold.
                    report = {
                        "auroc": None, "auprc": None, "auroc_auprc_undefined_reason": str(e),
                        "seed": seed, "fold": fold_idx, "hyperparameters": None, "preprocessing_fingerprint": fp,
                    }
                results[name]["folds"].append(report)

    for name in model_names:
        folds = results[name]["folds"]
        fold_seeds = [f["seed"] for f in folds]
        results[name]["auroc"] = aggregate_metric([f["auroc"] for f in folds])
        results[name]["auroc_by_seed"] = aggregate_metric_by_seed([f["auroc"] for f in folds], fold_seeds)
        results[name]["auprc"] = aggregate_metric([f["auprc"] for f in folds])
        results[name]["n_folds_run"] = len(folds)

    return {
        "task": "cancer_prediction", "primary_metric": "auroc",
        "n_folds_requested": n_folds, "seeds": list(seeds),
        "results": results,
    }
