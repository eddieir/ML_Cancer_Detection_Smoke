"""
benchmarks/final_evaluation.py — the one-shot frozen-test evaluation
protocol for Task B (blockers 2 and 5 of PR7):

  1. select_final_candidate: rank every CV-eligible candidate — classical
     baseline OR neural/MIL — using ONLY the development/CV evidence already
     computed by cross_validation.run_cancer_cv. No test access.
  2. generate_subject_oof_predictions: subject-grouped out-of-fold
     predictions for the selected candidate across the WHOLE development
     pool (train+val subjects), each OOF subject predicted by a model whose
     own fold-refit preprocessing artifact and training set never included
     that subject.
  3. fit_frozen_calibration_policy: calibration + threshold selection fitted
     EXCLUSIVELY from those OOF predictions (never validation-only, and
     never test).
  4. refit_final_candidate_on_dev_pool: ONE final preprocessing artifact +
     model fit, on ALL and ONLY the development subjects, using the
     already-selected model/hyperparameters (never reselected here).
  5. runner.py then acquires the durable FrozenTestGuard and calls this
     refit model exactly once on the untouched test subjects.

MIL candidates ("neural", "mean_mil", "max_mil", "attention_mil") and
classical baselines share this exact same protocol — no separate, silently
baseline-only code path.
"""

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from data.splitting import grouped_kfold
from train import MILEligibilityError, SubjectLevelDataset, validate_experiment_partitions

from .baselines import CANCER_BASELINES, positive_class_proba
from .features import build_cancer_subject_features
from .fold_preprocessing import (
    artifact_fingerprint,
    bags_from_fold_cell_dataset,
    build_fold_cell_dataset,
    refit_artifact_for_fold,
    require_normalized_adata,
)
from .neural import NeuralCancerAdapter

MIL_CANDIDATE_NAMES = ("neural", "mean_mil", "max_mil", "attention_mil")


def is_mil_candidate(name: str) -> bool:
    return name in MIL_CANDIDATE_NAMES


class NoEligibleFinalCandidateError(ValueError):
    """Raised when not one requested model produced a defined CV primary
    metric — there is nothing legitimate to select as the final candidate."""


def select_final_candidate(
    cv_report: Dict, model_names: Sequence[str], primary_metric: str = "auroc",
) -> Tuple[str, Dict]:
    """
    Rank every requested model by its CV development-evidence mean of
    `primary_metric` (cross_validation.run_cancer_cv's aggregated result) —
    classical baselines and MIL candidates are ranked on exactly the same
    footing, so a MIL model winning is never silently overridden. A model
    with an undefined CV mean (e.g. every fold undefined) is recorded as
    ineligible with a reason, never crashes the whole selection or is
    silently dropped without a trace.
    """
    ranked, ineligible = [], []
    for name in model_names:
        res = cv_report.get("results", {}).get(name)
        if res is None:
            ineligible.append({"name": name, "reason": "no CV results recorded for this model"})
            continue
        mean = res.get(primary_metric, {}).get("mean")
        if mean is None:
            ineligible.append({"name": name, "reason": f"{primary_metric} undefined across all CV folds/seeds"})
            continue
        ranked.append({"name": name, "score": mean, "kind": "mil" if is_mil_candidate(name) else "baseline"})
    ranked.sort(key=lambda r: r["score"], reverse=True)

    if not ranked:
        raise NoEligibleFinalCandidateError(
            f"select_final_candidate: no model among {list(model_names)} produced a defined "
            f"CV {primary_metric} — nothing eligible to select as the final frozen-test candidate. "
            f"Ineligible: {ineligible}"
        )
    selection_report = {"primary_metric": primary_metric, "ranked": ranked, "ineligible": ineligible}
    return ranked[0]["name"], selection_report


def _fit_baseline(name: str, hp_params: Dict, Xtr: np.ndarray, ytr: np.ndarray, seed: int):
    model = CANCER_BASELINES[name](**hp_params)
    model.fit(Xtr, ytr, seed=seed)
    return model


def _predict_baseline(model, X: np.ndarray) -> np.ndarray:
    return positive_class_proba(model, X)


def _fit_mil(context, artifact, pooling: str, device: str, train_ds, val_ds,
             train_bags: List[dict], val_bags: List[dict], hp_params: Dict, seed: int) -> NeuralCancerAdapter:
    """train_ds/val_ds and train_bags/val_bags must already be subject-
    disjoint — callers are responsible for that (see
    generate_subject_oof_predictions's per-fold split, and
    refit_final_candidate_on_dev_pool's internal held-out slice of the
    development pool used only for Trainer's own checkpoint selection)."""
    train_sd = SubjectLevelDataset(train_bags)
    val_sd = SubjectLevelDataset(val_bags)
    validate_experiment_partitions(
        train_cell_dataset=train_ds, val_cell_dataset=val_ds,
        train_subject_dataset=train_sd, val_subject_dataset=val_sd,
    )
    fold_ctx = dataclasses.replace(context, preprocessing_artifact=artifact,
                                    train_cell_dataset=train_ds, val_cell_dataset=val_ds)
    adapter = NeuralCancerAdapter(pooling=pooling, device=device)
    adapter.fit(fold_ctx, train_ds, val_ds, train_sd, val_sd, seed=seed,
                pretrain_epochs=hp_params.get("pretrain_epochs", 2))
    return adapter


def generate_subject_oof_predictions(
    context, candidate_name: str, dev_subjects: Sequence[str], outcomes_by_subject: Dict[str, int],
    selected_params: Dict, num_cell_types: int, min_cells_per_subject: int, n_hvgs: int,
    pooling: str = "attention", device: str = "cpu", seed: int = 42, n_folds: int = 5,
) -> Dict:
    """
    Subject-grouped out-of-fold probabilities for candidate_name across the
    WHOLE development pool, using the ALREADY-SELECTED `selected_params`
    (this function performs no hyperparameter search of its own — those were
    chosen once, from CV/dev-pool evidence, before this runs). Each dev
    subject's OOF prediction comes from a model fit only on the OTHER
    subjects in its fold; every fold's PreprocessingArtifact is refit from
    only that fold's own training subjects (fold_train_val_datasets), so no
    OOF subject's expression ever influenced the scaling/HVG selection its
    own prediction is computed under.

    Returns {"oof_by_subject": {sid: proba}, "fold_membership": [...],
    "candidate": candidate_name, "n_folds_requested": n_folds}. Raises if any
    dev subject with a known outcome does not end up with exactly one OOF
    prediction, or if a subject's OOF prediction came from a fold that also
    trained on that subject (grouped_kfold's own subject-disjoint guarantee
    means this can only happen from a bug, not real data).
    """
    normalized_adata = require_normalized_adata(context)
    dev_subjects = sorted({str(s) for s in dev_subjects if str(s) in outcomes_by_subject})
    if len(dev_subjects) < 2:
        raise ValueError("generate_subject_oof_predictions: fewer than 2 known-outcome development subjects")

    y_dev = np.array([outcomes_by_subject[s] for s in dev_subjects])
    folds = grouped_kfold(np.array(dev_subjects), y_dev, n_folds=n_folds, seed=seed)

    oof_by_subject: Dict[str, float] = {}
    fold_membership = []
    for fold_idx, fold in enumerate(folds):
        artifact = refit_artifact_for_fold(normalized_adata, fold["train"], n_hvgs)
        train_ds = build_fold_cell_dataset(normalized_adata, artifact, fold["train"])
        val_ds = build_fold_cell_dataset(normalized_adata, artifact, fold["val"])
        train_bags = bags_from_fold_cell_dataset(train_ds, outcomes_by_subject, min_cells_per_subject)
        val_bags = bags_from_fold_cell_dataset(val_ds, outcomes_by_subject, min_cells_per_subject)

        record = {
            "fold": fold_idx, "train_subject_ids": sorted(str(s) for s in fold["train"]),
            "val_subject_ids": sorted(str(s) for s in fold["val"]),
            "preprocessing_fingerprint": artifact_fingerprint(artifact),
        }
        if not train_bags or not val_bags:
            record["skipped_reason"] = "fold has no subject with >= min_cells_per_subject cells"
            fold_membership.append(record)
            continue

        if is_mil_candidate(candidate_name):
            try:
                adapter = _fit_mil(context, artifact, pooling, device, train_ds, val_ds,
                                    train_bags, val_bags, selected_params, seed)
                val_sd = SubjectLevelDataset(val_bags)
                proba = adapter.predict_proba(val_sd)
                subj_order = [str(b["subject_id"]) for b in val_sd.bags]
            except MILEligibilityError as e:
                record["skipped_reason"] = f"MIL ineligible: {e}"
                fold_membership.append(record)
                continue
        else:
            Xtr, ytr, _, _ = build_cancer_subject_features(train_bags, num_cell_types)
            Xva, _, subj_va, _ = build_cancer_subject_features(val_bags, num_cell_types)
            model = _fit_baseline(candidate_name, selected_params, Xtr, ytr, seed)
            proba = _predict_baseline(model, Xva)
            subj_order = subj_va

        for sid, p in zip(subj_order, proba):
            sid = str(sid)
            if sid in oof_by_subject:
                raise RuntimeError(
                    f"generate_subject_oof_predictions: duplicate OOF prediction for subject {sid!r} "
                    "— grouped_kfold's val partitions must be disjoint across folds."
                )
            if sid in record["train_subject_ids"]:
                raise RuntimeError(
                    f"generate_subject_oof_predictions: subject {sid!r} was predicted by a model "
                    "trained on that same subject — in-fold leakage."
                )
            oof_by_subject[sid] = float(p)
        record["n_predicted"] = len(subj_order)
        fold_membership.append(record)

    missing = sorted(set(dev_subjects) - set(oof_by_subject))
    if missing:
        raise RuntimeError(
            f"generate_subject_oof_predictions: {len(missing)} development subject(s) never received "
            f"an OOF prediction (e.g. every fold containing them was skipped): {missing[:5]}"
        )

    return {
        "oof_by_subject": oof_by_subject, "fold_membership": fold_membership,
        "candidate": candidate_name, "n_folds_requested": n_folds, "seed": seed,
        "dev_subjects": dev_subjects,
    }


def refit_final_candidate_on_dev_pool(
    context, candidate_name: str, dev_subjects: Sequence[str], test_subjects: Sequence[str],
    outcomes_by_subject: Dict[str, int], selected_params: Dict, num_cell_types: int,
    min_cells_per_subject: int, n_hvgs: int, pooling: str = "attention", device: str = "cpu", seed: int = 42,
) -> Dict:
    """
    The ONE final fit: a single PreprocessingArtifact refit on ALL (and
    ONLY) development subjects, and a single candidate_name model fit on
    that artifact's ALL development-subject bags, using the already-selected
    `selected_params` (never reselected here). Applies that same artifact to
    the test subjects to produce test-set probabilities — this is the only
    place test expression is transformed, and it happens with a
    preprocessing artifact that has never seen test data during FITTING
    (only TRANSFORM, exactly like every other split in this codebase).

    Returns {"test_subject_ids", "test_proba", "preprocessing_artifact",
    "preprocessing_artifact_fingerprint", "model_metadata"} — everything
    runner.py needs to persist provenance for the frozen test evaluation.
    """
    normalized_adata = require_normalized_adata(context)
    dev_subjects = sorted({str(s) for s in dev_subjects if str(s) in outcomes_by_subject})
    test_subjects = sorted({str(s) for s in test_subjects})

    final_artifact = refit_artifact_for_fold(normalized_adata, dev_subjects, n_hvgs)
    dev_ds = build_fold_cell_dataset(normalized_adata, final_artifact, dev_subjects)
    test_ds = build_fold_cell_dataset(normalized_adata, final_artifact, test_subjects)
    dev_bags = bags_from_fold_cell_dataset(dev_ds, outcomes_by_subject, min_cells_per_subject)
    # Test bags may legitimately include subjects with unknown outcomes —
    # build_cancer_subject_features filters to known-outcome bags itself.
    test_outcomes = {sid: outcomes_by_subject[sid] for sid in test_subjects if sid in outcomes_by_subject}
    test_bags = bags_from_fold_cell_dataset(test_ds, test_outcomes, min_cells_per_subject)
    if not dev_bags:
        raise ValueError("refit_final_candidate_on_dev_pool: no development subject has >= "
                          "min_cells_per_subject cells after the final refit")
    if not test_bags:
        raise ValueError("refit_final_candidate_on_dev_pool: no test subject has >= "
                          "min_cells_per_subject cells and a known outcome after the final refit")

    if is_mil_candidate(candidate_name):
        # Trainer.phase1/phase2 require a subject-disjoint validation split
        # for their own checkpoint/early-stopping bookkeeping — unlike the
        # classical baselines, the MIL final refit therefore carves a small
        # internal held-out slice OUT OF the development pool for that
        # purpose only (never test, never a subject outside dev_subjects).
        # This means a minority of development subjects contribute to
        # Trainer's validation metric rather than to the fitted weights
        # themselves for this final pass — a documented limitation of
        # reusing the existing Trainer architecture for the final refit; see
        # README/ARCHITECTURE's Phase 1 limitations section.
        dev_subject_ids_mil = sorted({str(b["subject_id"]) for b in dev_bags})
        y_dev_mil = np.array([outcomes_by_subject[s] for s in dev_subject_ids_mil])
        if len(dev_subject_ids_mil) >= 4:
            # n_folds=2 (an approximately even split) rather than a smaller
            # held-out slice — MIL eligibility requires a minimum subject
            # count on EACH side (train.check_mil_eligibility), which a
            # small 80/20-style split can starve on the held-out side for a
            # modest development pool.
            internal_fold = grouped_kfold(np.array(dev_subject_ids_mil), y_dev_mil, n_folds=2, seed=seed)[0]
            internal_train, internal_val = set(internal_fold["train"]), set(internal_fold["val"])
        else:
            internal_train = set(dev_subject_ids_mil[:-1])
            internal_val = set(dev_subject_ids_mil[-1:])
        internal_train_bags = [b for b in dev_bags if str(b["subject_id"]) in internal_train]
        internal_val_bags = [b for b in dev_bags if str(b["subject_id"]) in internal_val]
        internal_train_ds = dev_ds.subset_by_subjects(sorted(internal_train))
        internal_val_ds = dev_ds.subset_by_subjects(sorted(internal_val))
        adapter = _fit_mil(context, final_artifact, pooling, device, internal_train_ds, internal_val_ds,
                            internal_train_bags, internal_val_bags, selected_params, seed)
        test_sd = SubjectLevelDataset(test_bags)
        test_proba = adapter.predict_proba(test_sd)
        test_subject_order = [str(b["subject_id"]) for b in test_sd.bags]
        model_metadata = adapter.metadata()
    else:
        Xtr, ytr, _, _ = build_cancer_subject_features(dev_bags, num_cell_types)
        Xte, _, test_subject_order, _ = build_cancer_subject_features(test_bags, num_cell_types)
        model = _fit_baseline(candidate_name, selected_params, Xtr, ytr, seed)
        test_proba = _predict_baseline(model, Xte)
        model_metadata = model.metadata()

    y_test_ordered = np.array([outcomes_by_subject[s] for s in test_subject_order])
    return {
        "test_subject_ids": test_subject_order, "test_labels": y_test_ordered, "test_proba": test_proba,
        "preprocessing_artifact": final_artifact,
        "preprocessing_artifact_fingerprint": artifact_fingerprint(final_artifact),
        "dev_subject_ids": dev_subjects, "model_metadata": model_metadata,
    }
