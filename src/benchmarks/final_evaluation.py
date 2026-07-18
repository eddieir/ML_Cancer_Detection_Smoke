"""
benchmarks/final_evaluation.py — the one-shot frozen-test evaluation
protocol for Task B.

Two structurally separate stages:

  A. Development-only final fit (select_final_candidate,
     generate_subject_oof_predictions, fit_final_candidate_on_dev_pool).
     These functions never accept test subject IDs, test bags, test labels,
     or test expression as arguments — it is not merely convention but a
     structural property of their signatures that they cannot read test
     data. Every hyperparameter/config decision (including each OOF fold's
     own inner-CV selection) is made from development evidence only.

  B. Guarded frozen-test execution (evaluate_frozen_test). This is the ONE
     function in this module that touches test data — it transforms test
     expression through an ALREADY-FITTED preprocessing artifact (never
     refits it) and generates raw probabilities from an ALREADY-FITTED
     model. It carries no guard of its own by design: runner.py is the
     single call site, and it is responsible for acquiring the durable
     FrozenTestGuard (test_guard.py) immediately before calling this
     function and for marking the guard failed/completed around it. This
     keeps "which stage may touch test data" a property of which function
     is called, not of a flag threaded through shared code.

MIL candidates ("neural", "mean_mil", "max_mil", "attention_mil") and
classical baselines share this exact same protocol — no separate, silently
baseline-only code path. The final MIL fit uses Trainer.phase1_final_fit /
phase2_final_fit (train.py), which train on every supplied subject with no
internal validation carve-out for checkpoint selection — unlike CV/OOF
fold fits, which still use the normal validation-selected phase1/phase2.
"""

import dataclasses
import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from data.splitting import grouped_kfold
from train import MILEligibilityError, SubjectLevelDataset, check_mil_eligibility, validate_experiment_partitions

from .baselines import CANCER_BASELINES, CANCER_SEARCH_SPACE, positive_class_proba
from .cross_validation import MIL_SEARCH_SPACE, _cancer_baseline_fit_score_fn, _mil_fit_score_fn, _pathway_cancer_fit_score_fn
from .features import build_cancer_subject_features
from .fold_preprocessing import (
    artifact_fingerprint,
    bags_from_fold_cell_dataset,
    build_fold_cell_dataset,
    refit_artifact_for_fold,
    require_normalized_adata,
)
from .hyperparameter_search import build_param_grid, select_nested_hyperparameters_with_refit
from .mil_registry import MIL_CANDIDATE_NAMES, build_mil_adapter, pathway_search_space
from .neural import NeuralCancerAdapter
from .pathway_hierarchical_adapter import MODEL_NAME as PATHWAY_MODEL_NAME

DEFAULT_OOF_INNER_FOLDS = 2


def is_mil_candidate(name: str) -> bool:
    return name in MIL_CANDIDATE_NAMES


def _canonical_hash(obj) -> str:
    """SHA-256 of a canonical (sorted-keys) JSON encoding — used for every
    subject-list / selected-params fingerprint here, never a raw JSON blob
    stored under a field literally named "fingerprint" (issue 4)."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _subject_list_fingerprint(subject_ids: Sequence[str]) -> str:
    return _canonical_hash(sorted(str(s) for s in subject_ids))




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


def _fit_mil(context, candidate_name: str, artifact, pooling: str, device: str, train_ds, val_ds,
             train_bags: List[dict], val_bags: List[dict], hp_params: Dict, seed: int,
             domain_robustness_config: Optional[Dict] = None):
    """Used for OOF-fold fits only (each OOF fold still uses a real,
    subject-disjoint validation split for Trainer's own checkpoint
    selection, exactly like CV) — the final dev-pool refit uses
    fit_final_candidate_on_dev_pool below instead, which trains on every
    development subject via the adapter's own fit_final(). candidate_name
    selects the adapter class via benchmarks/mil_registry.py, so this
    function works unchanged for every MIL-kind candidate."""
    train_sd = SubjectLevelDataset(train_bags)
    val_sd = SubjectLevelDataset(val_bags)
    validate_experiment_partitions(
        train_cell_dataset=train_ds, val_cell_dataset=val_ds,
        train_subject_dataset=train_sd, val_subject_dataset=val_sd,
    )
    is_pathway = candidate_name == PATHWAY_MODEL_NAME
    if is_pathway:
        # NeuralCancerAdapter.fit gets this same train+val eligibility
        # check for free inside Trainer.phase2 — this adapter has no
        # Trainer, so it is applied explicitly here to match.
        check_mil_eligibility(train_sd)
        check_mil_eligibility(val_sd)
    fold_ctx = dataclasses.replace(context, preprocessing_artifact=artifact,
                                    train_cell_dataset=train_ds, val_cell_dataset=val_ds)
    adapter = build_mil_adapter(
        candidate_name, pooling, device, config_overrides=hp_params if is_pathway else None,
        domain_robustness_config=domain_robustness_config if is_pathway else None,
    )
    adapter.fit(fold_ctx, train_ds, val_ds, train_sd, val_sd, seed=seed,
                pretrain_epochs=None if is_pathway else hp_params.get("pretrain_epochs", 2))
    return adapter


def _oof_fold_hyperparameters(
    context, candidate_name: str, oof_train_subjects: Sequence[str], outcomes_by_subject: Dict[str, int],
    num_cell_types: int, min_cells_per_subject: int, n_hvgs: int, pooling: str, device: str,
    seed: int, n_inner_folds: int,
) -> Dict:
    """
    One inner-CV hyperparameter/config selection computed ENTIRELY from
    oof_train_subjects (never oof_val_subjects, never any other OOF fold's
    subjects, never test) — this is what makes each OOF fold's prediction
    come from a configuration that never saw the subject it predicts,
    rather than one globally selected configuration whose selection was
    influenced by every development subject's label (including the one
    being predicted).
    """
    if candidate_name == PATHWAY_MODEL_NAME:
        candidates = build_param_grid(pathway_search_space(fast=False))
        fit_score_fn = _pathway_cancer_fit_score_fn(context, device, outcomes_by_subject, min_cells_per_subject)
    elif is_mil_candidate(candidate_name):
        candidates = build_param_grid(MIL_SEARCH_SPACE)
        fit_score_fn = _mil_fit_score_fn(context, pooling, device, outcomes_by_subject, min_cells_per_subject)
    else:
        candidates = build_param_grid(CANCER_SEARCH_SPACE.get(candidate_name, {}))
        fit_score_fn = _cancer_baseline_fit_score_fn(CANCER_BASELINES[candidate_name], num_cell_types,
                                                       outcomes_by_subject, min_cells_per_subject)
    return select_nested_hyperparameters_with_refit(
        context, oof_train_subjects, outcomes_by_subject, candidates, fit_score_fn=fit_score_fn,
        seed=seed, n_inner_folds=n_inner_folds, n_hvgs=n_hvgs,
    )


def generate_subject_oof_predictions(
    context, candidate_name: str, dev_subjects: Sequence[str], outcomes_by_subject: Dict[str, int],
    num_cell_types: int, min_cells_per_subject: int, n_hvgs: int,
    pooling: str = "attention", device: str = "cpu", seed: int = 42, n_folds: int = 5,
    n_inner_folds: int = DEFAULT_OOF_INNER_FOLDS, domain_robustness_config: Optional[Dict] = None,
) -> Dict:
    """
    Subject-grouped, SELECTION-CLEAN out-of-fold probabilities for
    candidate_name across the WHOLE development pool. For every OOF fold,
    hyperparameters/config are selected by a fresh inner-CV run using ONLY
    that fold's OOF-training subjects (_oof_fold_hyperparameters) — an
    OOF-held-out subject's label/expression never influences the
    configuration used to predict it, not just the model weights that
    predict it. The fold model is then fit on the WHOLE OOF-training set
    with that fold-selected configuration and used to predict the
    OOF-held-out subjects, mirroring the standard nested-CV pattern already
    used by cross_validation.py's outer-fold loop.

    Returns {"oof_by_subject": {sid: proba}, "fold_membership": [...] (each
    entry carries its own "hyperparameter_search"), "candidate":
    candidate_name, "n_folds_requested": n_folds}. Raises if any dev subject
    with a known outcome does not end up with exactly one OOF prediction, or
    if a subject's OOF prediction came from a fold that also trained on that
    subject.
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
        oof_train_subjects = fold["train"]
        oof_val_subjects = fold["val"]

        hp_search = _oof_fold_hyperparameters(
            context, candidate_name, oof_train_subjects, outcomes_by_subject,
            num_cell_types, min_cells_per_subject, n_hvgs, pooling, device, seed, n_inner_folds,
        )
        selected_params = hp_search["selected_params"]

        artifact = refit_artifact_for_fold(normalized_adata, oof_train_subjects, n_hvgs)
        train_ds = build_fold_cell_dataset(normalized_adata, artifact, oof_train_subjects)
        val_ds = build_fold_cell_dataset(normalized_adata, artifact, oof_val_subjects)
        train_bags = bags_from_fold_cell_dataset(train_ds, outcomes_by_subject, min_cells_per_subject)
        val_bags = bags_from_fold_cell_dataset(val_ds, outcomes_by_subject, min_cells_per_subject)

        train_ids = sorted(str(s) for s in oof_train_subjects)
        val_ids = sorted(str(s) for s in oof_val_subjects)
        record = {
            "fold": fold_idx, "train_subject_ids": train_ids, "val_subject_ids": val_ids,
            "training_subjects_fingerprint": _subject_list_fingerprint(train_ids),
            "validation_subjects_fingerprint": _subject_list_fingerprint(val_ids),
            "preprocessing_fingerprint": artifact_fingerprint(artifact),
            "hyperparameter_search": hp_search,
            "selected_params_fingerprint": _canonical_hash(selected_params),
            "inner_selection_fingerprint": _canonical_hash(hp_search),
        }
        if not train_bags or not val_bags:
            record["skipped_reason"] = "fold has no subject with >= min_cells_per_subject cells"
            fold_membership.append(record)
            continue

        if is_mil_candidate(candidate_name):
            try:
                adapter = _fit_mil(context, candidate_name, artifact, pooling, device, train_ds, val_ds,
                                    train_bags, val_bags, selected_params, seed,
                                    domain_robustness_config=domain_robustness_config)
                val_sd = SubjectLevelDataset(val_bags)
                proba = adapter.predict_proba(val_sd)
                subj_order = [str(b["subject_id"]) for b in val_sd.bags]
                record["model_state_fingerprint"] = adapter.model_state_fingerprint()
                if candidate_name == PATHWAY_MODEL_NAME:
                    record["module_fingerprint"] = adapter.modules.fingerprint() if adapter.modules else None
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
            record["model_state_fingerprint"] = model.model_state_fingerprint()

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


@dataclasses.dataclass
class FittedFinalCandidate:
    """
    The frozen, development-only output of fit_final_candidate_on_dev_pool.
    Carries everything evaluate_frozen_test/runner.py need to transform and
    score the test split, and nothing that could have been derived from
    test data — this dataclass has no field a test-touching value could be
    smuggled through.
    """
    candidate_name: str
    kind: str  # "baseline" or "mil"
    pooling: Optional[str]
    selected_params: Dict
    preprocessing_artifact: object
    preprocessing_artifact_fingerprint: str
    dev_subject_ids: List[str]
    model_metadata: Dict
    model_state_fingerprint: str
    predictor: object  # a fitted baseline model, or a fitted NeuralCancerAdapter


def fit_final_candidate_on_dev_pool(
    context, candidate_name: str, dev_subjects: Sequence[str], outcomes_by_subject: Dict[str, int],
    selected_params: Dict, num_cell_types: int, min_cells_per_subject: int, n_hvgs: int,
    pooling: str = "attention", device: str = "cpu", seed: int = 42,
    domain_robustness_config: Optional[Dict] = None,
) -> FittedFinalCandidate:
    """
    Development-only final fit (blocker 1's stage A / blocker 3): ONE
    PreprocessingArtifact refit on ALL (and ONLY) development subjects, and
    ONE candidate_name model/adapter fit on that artifact's ALL
    development-subject bags, using the already-selected `selected_params`
    (never reselected here). This function's signature accepts no test
    subject IDs, test bags, or test labels — it is structurally unable to
    read test data, not merely documented not to.

    For a MIL candidate, this calls NeuralCancerAdapter.fit_final, which
    trains via Trainer.phase1_final_fit/phase2_final_fit — every
    development subject contributes to gradient updates, with no internal
    validation carve-out for checkpoint selection (blocker 3).
    """
    normalized_adata = require_normalized_adata(context)
    dev_subjects = sorted({str(s) for s in dev_subjects if str(s) in outcomes_by_subject})

    final_artifact = refit_artifact_for_fold(normalized_adata, dev_subjects, n_hvgs)
    dev_ds = build_fold_cell_dataset(normalized_adata, final_artifact, dev_subjects)
    dev_bags = bags_from_fold_cell_dataset(dev_ds, outcomes_by_subject, min_cells_per_subject)
    if not dev_bags:
        raise ValueError("fit_final_candidate_on_dev_pool: no development subject has >= "
                          "min_cells_per_subject cells after the final refit")

    kind = "mil" if is_mil_candidate(candidate_name) else "baseline"
    if kind == "mil":
        dev_sd = SubjectLevelDataset(dev_bags)
        is_pathway = candidate_name == PATHWAY_MODEL_NAME
        # No eligibility check here, for any MIL-kind candidate — this
        # mirrors Trainer.phase2_final_fit exactly: the final dev-pool fit
        # has no internal validation split to check, by design (blocker 3;
        # see fit_final_candidate_on_dev_pool's own docstring).
        fold_ctx = dataclasses.replace(context, preprocessing_artifact=final_artifact)
        adapter = build_mil_adapter(
            candidate_name, pooling, device, config_overrides=selected_params if is_pathway else None,
            domain_robustness_config=domain_robustness_config if is_pathway else None,
        )
        adapter.fit_final(fold_ctx, dev_ds, dev_sd, seed=seed,
                           pretrain_epochs=None if is_pathway else selected_params.get("pretrain_epochs"))
        model_metadata = adapter.metadata()
        model_state_fp = adapter.model_state_fingerprint()
        predictor = adapter
    else:
        Xtr, ytr, _, _ = build_cancer_subject_features(dev_bags, num_cell_types)
        model = _fit_baseline(candidate_name, selected_params, Xtr, ytr, seed)
        model_metadata = model.metadata()
        model_state_fp = model.model_state_fingerprint()
        predictor = model

    return FittedFinalCandidate(
        candidate_name=candidate_name, kind=kind, pooling=pooling if kind == "mil" else None,
        selected_params=selected_params, preprocessing_artifact=final_artifact,
        preprocessing_artifact_fingerprint=artifact_fingerprint(final_artifact),
        dev_subject_ids=sorted({str(b["subject_id"]) for b in dev_bags}),
        model_metadata=model_metadata, model_state_fingerprint=model_state_fp, predictor=predictor,
    )


def evaluate_frozen_test(
    fitted: FittedFinalCandidate, context, test_subjects: Sequence[str],
    test_outcomes_by_subject: Dict[str, int], num_cell_types: int, min_cells_per_subject: int,
) -> Dict:
    """
    The ONE function in this module that touches test data: transforms test
    expression through fitted.preprocessing_artifact (TRANSFORM only, never
    fit — that artifact was already fit exclusively on development subjects
    by fit_final_candidate_on_dev_pool) and generates raw, uncalibrated
    probabilities from fitted.predictor (already fit, never refit here).

    This function carries no guard of its own — runner.py is the single
    sanctioned call site, and MUST call FrozenTestGuard.acquire() before
    calling this function and mark it completed/failed around the call.
    Every argument here (test_subjects, test_outcomes_by_subject) is
    exactly the test-data surface a guarded call site must resolve only
    after acquiring the guard.
    """
    normalized_adata = require_normalized_adata(context)
    test_subjects = sorted({str(s) for s in test_subjects})
    test_ds = build_fold_cell_dataset(normalized_adata, fitted.preprocessing_artifact, test_subjects)
    test_outcomes = {sid: test_outcomes_by_subject[sid] for sid in test_subjects if sid in test_outcomes_by_subject}
    test_bags = bags_from_fold_cell_dataset(test_ds, test_outcomes, min_cells_per_subject)
    if not test_bags:
        raise ValueError("evaluate_frozen_test: no test subject has >= min_cells_per_subject "
                          "cells and a known outcome after the final preprocessing transform")

    if fitted.kind == "mil":
        test_sd = SubjectLevelDataset(test_bags)
        test_proba = fitted.predictor.predict_proba(test_sd)
        test_subject_order = [str(b["subject_id"]) for b in test_sd.bags]
    else:
        Xte, _, test_subject_order, _ = build_cancer_subject_features(test_bags, num_cell_types)
        test_proba = _predict_baseline(fitted.predictor, Xte)

    y_test_ordered = np.array([test_outcomes_by_subject[s] for s in test_subject_order])
    return {
        "test_subject_ids": test_subject_order, "test_labels": y_test_ordered, "test_proba": test_proba,
    }
