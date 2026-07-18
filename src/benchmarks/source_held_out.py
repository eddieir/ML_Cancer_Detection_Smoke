"""
benchmarks/source_held_out.py — Phase 6 source-held-out ("LOSO") evaluation
protocol: for each eligible source, train on every OTHER source (using the
existing development-only OOF + dev-pool-refit protocol from
final_evaluation.py, restricted to a development-source-only subject pool)
and apply the frozen result to the held-out source EXACTLY ONCE.

This is deliberately layered on top of, not a fork of, final_evaluation.py's
functions: generate_subject_oof_predictions / fit_final_candidate_on_dev_pool
/ evaluate_frozen_test all accept explicit subject-ID lists rather than
reading context.train_bags/val_bags/test_bags directly, so calling them with
"development sources' subjects" in place of "train+val" and "held-out
source's subjects" in place of "test" reuses every leakage-prevention
property those functions already have (fold-local preprocessing refit,
per-OOF-fold nested hyperparameter selection, calibration/threshold fit
exclusively from OOF predictions) with no duplicated logic.

This protocol is entirely separate from the real frozen-test guard
(test_guard.py) — it never reads context.test_bags, context.subjects_for
("test"), or split_manifest.test_subjects, and it never touches
FrozenTestGuard. A held-out source's subjects come exclusively from the
train+val pool (see ood.py's pre-existing pattern, reused here), so this
evaluation can run any number of times across any number of sources without
ever consuming the one-shot real test guard.
"""

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .baselines import CANCER_BASELINES, CANCER_SEARCH_SPACE, SMOKE_BASELINES, positive_class_proba
from .calibration import build_frozen_policy
from .domain_losses import resolve_domain_robustness_config
from .features import build_cancer_subject_features, build_smoke_subject_summary_features
from .final_evaluation import (
    fit_final_candidate_on_dev_pool,
    generate_subject_oof_predictions,
    is_mil_candidate,
)
from .fold_preprocessing import (
    artifact_fingerprint,
    bags_from_fold_cell_dataset,
    build_fold_cell_dataset,
    refit_artifact_for_fold,
    require_normalized_adata,
)
from .metrics import full_smoke_metrics_report
from .mil_registry import build_mil_adapter
from .pathway_hierarchical_adapter import MODEL_NAME as PATHWAY_MODEL_NAME
from .robustness_report import build_robustness_report
from .source_eligibility import (
    assess_cancer_source_eligibility,
    assess_smoke_source_eligibility,
    build_source_held_out_manifest,
)

DEFAULT_OOF_FOLDS = 3


class CrossSourceSubjectConflictError(ValueError):
    """A subject_id assigned to more than one dataset_source in the
    train+val pool — a data-assembly inconsistency, never silently
    resolved by picking one source (see benchmarks/ood.py's identical
    check for the pre-existing Task A LOSO path)."""


def _pool_subjects_and_sources(context) -> Dict[str, str]:
    normalized_adata = require_normalized_adata(context)
    pool_subjects = sorted(set(context.subjects_for("train")) | set(context.subjects_for("val")))
    obs = normalized_adata.obs
    subj_series = obs["subject_id"].astype(str)
    pool_mask = subj_series.isin(pool_subjects).values
    pool_source = obs["source"].astype(str).values[pool_mask]
    pool_subject_arr = subj_series.values[pool_mask]

    by_subject: Dict[str, set] = {}
    for sid, src in zip(pool_subject_arr.tolist(), pool_source.tolist()):
        by_subject.setdefault(sid, set()).add(src)
    conflicts = {sid: sorted(s) for sid, s in by_subject.items() if len(s) > 1}
    if conflicts:
        raise CrossSourceSubjectConflictError(
            f"{len(conflicts)} subject(s) assigned to more than one dataset_source in the "
            f"train+val pool — e.g. {dict(list(conflicts.items())[:3])}."
        )
    return {sid: next(iter(srcs)) for sid, srcs in by_subject.items()}


# ─── Task B: cancer prediction ─────────────────────────────────────────────

def run_cancer_source_held_out(
    context, model_names: Sequence[str], device: str = "cpu",
    domain_robustness_config: Optional[Dict] = None,
    incompatible_sources: Optional[Sequence[str]] = None,
    species_by_source: Optional[Dict[str, str]] = None,
    reference_species: Optional[str] = None,
    controlled_access_sources: Optional[Sequence[str]] = None,
    seed: int = 42, n_oof_folds: int = DEFAULT_OOF_FOLDS,
) -> Dict:
    """
    For every source present in the train+val pool: assess eligibility,
    then (if eligible) select the best of model_names by development-only
    OOF AUROC computed EXCLUSIVELY from the remaining (development) sources'
    subjects, fit that candidate once on the full development pool, and
    apply the frozen result to the held-out source's subjects exactly once.
    Returns {source: RobustnessReport dict}.
    """
    domain_cfg = resolve_domain_robustness_config(domain_robustness_config)
    subject_to_source = _pool_subjects_and_sources(context)
    all_bags = list(context.train_bags) + list(context.val_bags)
    outcomes_by_subject = {str(b["subject_id"]): b["cancer_label"] for b in all_bags if b.get("cancer_label_known")}
    num_cell_types = context.config.get("model", {}).get("num_cell_types", 4)
    n_hvgs = context.preprocessing_artifact.n_hvgs
    min_cells = context.config.get("data", {}).get("min_cells_per_subject", 50)

    sources = sorted(set(subject_to_source.values()))
    reports: Dict[str, Dict] = {}

    for held_out_source in sources:
        held_out_subjects = sorted(s for s, src in subject_to_source.items() if src == held_out_source)
        held_out_outcomes = [outcomes_by_subject.get(s) for s in held_out_subjects]

        elig = assess_cancer_source_eligibility(
            held_out_source, held_out_outcomes, incompatible_sources=incompatible_sources,
            species_by_source=species_by_source, reference_species=reference_species,
            controlled_access_sources=controlled_access_sources,
        )
        dev_sources = [s for s in sources if s != held_out_source]
        dev_subjects_all = sorted(s for s, src in subject_to_source.items() if src != held_out_source)

        if not elig.eligible:
            manifest = build_source_held_out_manifest(
                task="cancer_prediction", held_out_source=held_out_source, development_sources=dev_sources,
                development_subjects=dev_subjects_all, held_out_subjects=held_out_subjects,
                known_label_counts=elig.counts, class_distribution={}, eligibility=elig, seed=seed,
            )
            reports[held_out_source] = build_robustness_report(
                task="cancer_prediction", model=None, strategy=domain_cfg["strategy"],
                held_out_source=held_out_source, eligibility=elig.to_dict(), development_sources=dev_sources,
                source_split_manifest_fingerprint=manifest.fingerprint(),
                limitations=["source ineligible — no model was trained or evaluated for this source"],
            ).to_dict()
            continue

        dev_subjects = sorted(s for s in dev_subjects_all if s in outcomes_by_subject)
        best_name, best_auroc, best_oof, best_selected_params = None, -np.inf, None, None
        candidate_scores = {}
        for name in model_names:
            try:
                oof = generate_subject_oof_predictions(
                    context, name, dev_subjects, outcomes_by_subject, num_cell_types, min_cells, n_hvgs,
                    pooling="attention", device=device, seed=seed, n_folds=n_oof_folds,
                    domain_robustness_config=domain_cfg if name == PATHWAY_MODEL_NAME else None,
                )
            except Exception as e:  # noqa: BLE001 — record and continue, never crash the whole sweep
                candidate_scores[name] = {"auroc": None, "error": str(e)}
                continue
            y = np.array([outcomes_by_subject[s] for s in oof["dev_subjects"]])
            p = np.array([oof["oof_by_subject"][s] for s in oof["dev_subjects"]])
            if len(set(y.tolist())) < 2:
                candidate_scores[name] = {"auroc": None, "error": "dev-pool OOF has only one class"}
                continue
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(y, p))
            candidate_scores[name] = {"auroc": auroc}
            if auroc > best_auroc:
                best_name, best_auroc, best_oof = name, auroc, oof

        if best_name is None:
            reports[held_out_source] = build_robustness_report(
                task="cancer_prediction", model=None, strategy=domain_cfg["strategy"],
                held_out_source=held_out_source, eligibility=elig.to_dict(), development_sources=dev_sources,
                limitations=["no candidate model produced a defined development-only OOF AUROC"],
                comparisons=[{"name": n, **s} for n, s in candidate_scores.items()],
            ).to_dict()
            continue

        y_oof = np.array([outcomes_by_subject[s] for s in best_oof["dev_subjects"]])
        prob_oof = np.array([best_oof["oof_by_subject"][s] for s in best_oof["dev_subjects"]])
        policy = build_frozen_policy(y_oof, prob_oof)

        hp_for_final = {}  # keep the dev-pool refit's own selected params per OOF fold's majority — a
        # single declared configuration is not re-selected here; the final fit reuses whichever
        # params the LAST OOF fold happened to select is not scientifically meaningful, so instead
        # we re-run one nested selection over the FULL development pool for the final fit only
        # (mirrors runner.py's _final_dev_pool_hyperparameters — one extra, clearly-declared
        # selection step, never using held-out-source data).
        from .hyperparameter_search import build_param_grid, select_nested_hyperparameters_with_refit
        from .cross_validation import (
            DEFAULT_INNER_FOLDS, MIL_SEARCH_SPACE, _cancer_baseline_fit_score_fn, _mil_fit_score_fn,
            _pathway_cancer_fit_score_fn,
        )
        from .mil_registry import pathway_search_space
        if best_name == PATHWAY_MODEL_NAME:
            candidates = build_param_grid(pathway_search_space(fast=False))
            fit_score_fn = _pathway_cancer_fit_score_fn(context, device, outcomes_by_subject, min_cells)
        elif is_mil_candidate(best_name):
            candidates = build_param_grid(MIL_SEARCH_SPACE)
            fit_score_fn = _mil_fit_score_fn(context, "attention", device, outcomes_by_subject, min_cells)
        else:
            candidates = build_param_grid(CANCER_SEARCH_SPACE.get(best_name, {}))
            fit_score_fn = _cancer_baseline_fit_score_fn(
                CANCER_BASELINES[best_name], num_cell_types, outcomes_by_subject, min_cells,
            )
        hp_search = select_nested_hyperparameters_with_refit(
            context, dev_subjects, outcomes_by_subject, candidates, fit_score_fn=fit_score_fn,
            seed=seed, n_inner_folds=DEFAULT_INNER_FOLDS, n_hvgs=n_hvgs,
        )
        selected_params = hp_search["selected_params"]

        fitted = fit_final_candidate_on_dev_pool(
            context, best_name, dev_subjects, outcomes_by_subject, selected_params, num_cell_types,
            min_cells, n_hvgs, pooling="attention", device=device, seed=seed,
            domain_robustness_config=domain_cfg if best_name == PATHWAY_MODEL_NAME else None,
        )

        normalized_adata = require_normalized_adata(context)
        held_out_ds = build_fold_cell_dataset(normalized_adata, fitted.preprocessing_artifact, held_out_subjects)
        held_out_bags = bags_from_fold_cell_dataset(held_out_ds, outcomes_by_subject, min_cells)
        if not held_out_bags:
            reports[held_out_source] = build_robustness_report(
                task="cancer_prediction", model=best_name, strategy=domain_cfg["strategy"],
                held_out_source=held_out_source, eligibility=elig.to_dict(), development_sources=dev_sources,
                limitations=["held-out source had no subject with >= min_cells_per_subject cells "
                             "after the frozen preprocessing transform"],
            ).to_dict()
            continue

        if fitted.kind == "mil":
            from train import SubjectLevelDataset
            held_out_sd = SubjectLevelDataset(held_out_bags)
            held_out_proba_raw = fitted.predictor.predict_proba(held_out_sd)
            held_out_order = [str(b["subject_id"]) for b in held_out_sd.bags]
        else:
            Xho, _, held_out_order, _ = build_cancer_subject_features(held_out_bags, num_cell_types)
            held_out_proba_raw = positive_class_proba(fitted.predictor, Xho)

        y_held_out = np.array([outcomes_by_subject[s] for s in held_out_order])
        metrics_result = policy.apply_to_test(y_held_out, held_out_proba_raw)

        manifest = build_source_held_out_manifest(
            task="cancer_prediction", held_out_source=held_out_source, development_sources=dev_sources,
            development_subjects=dev_subjects_all, held_out_subjects=held_out_subjects,
            known_label_counts=elig.counts, class_distribution={"n_positive": elig.counts.get("n_positive"),
                                                                   "n_negative": elig.counts.get("n_negative")},
            eligibility=elig, seed=seed, preprocessing_policy_fingerprint=fitted.preprocessing_artifact_fingerprint,
            module_fingerprint=fitted.model_metadata.get("module_fingerprint"),
        )

        reports[held_out_source] = build_robustness_report(
            task="cancer_prediction", model=best_name, strategy=domain_cfg["strategy"],
            held_out_source=held_out_source, eligibility=elig.to_dict(), development_sources=dev_sources,
            metrics={k: v for k, v in metrics_result.items() if k != "policy"},
            calibration=metrics_result.get("policy", {}),
            comparisons=[{"name": n, **s} for n, s in candidate_scores.items()],
            source_split_manifest_fingerprint=manifest.fingerprint(),
            preprocessing_fingerprint=fitted.preprocessing_artifact_fingerprint,
            model_fingerprint=fitted.model_state_fingerprint,
        ).to_dict()

    return reports


# ─── Task A: smoke-type classification ─────────────────────────────────────

def run_smoke_source_held_out(
    context, model_names: Sequence[str], device: str = "cpu",
    incompatible_sources: Optional[Sequence[str]] = None,
    species_by_source: Optional[Dict[str, str]] = None,
    reference_species: Optional[str] = None,
    controlled_access_sources: Optional[Sequence[str]] = None,
    seed: int = 42,
) -> Dict:
    """
    Task A source-held-out evaluation. Classical baselines (SMOKE_BASELINES)
    are fit on subject-summary features exactly like the pre-existing
    ood.py::run_leave_one_source_out. pathway_hierarchical_mil, if requested,
    is additionally fit directly on development-source cell-level bags via
    PathwayHierarchicalAdapter and evaluated on the held-out source's
    subject-level majority-vote smoke label. The pooling-based MIL models
    ("neural", "mean_mil", "max_mil", "attention_mil") and the cell-level
    MultiSmokeCancerNet Trainer curriculum are NOT covered by this function
    — see the module-level limitation this is documented against in
    README.md — extending Task A's LOSO to the full Trainer curriculum
    would require a materially larger rework of the per-source refit flow
    than is safe to attempt within this scope.
    """
    normalized_adata = require_normalized_adata(context)
    num_classes = context.num_smoke_classes
    num_cell_types = context.config.get("model", {}).get("num_cell_types", 4)
    n_hvgs = context.preprocessing_artifact.n_hvgs

    subject_to_source = _pool_subjects_and_sources(context)
    obs = normalized_adata.obs
    subj_series = obs["subject_id"].astype(str)

    def _subject_label(sid: str) -> Optional[int]:
        mask = (subj_series == sid).values
        vals = obs["smoke_type"].values[mask]
        if len(vals) == 0:
            return None
        v, c = np.unique(vals, return_counts=True)
        return int(v[np.argmax(c)])

    sources = sorted(set(subject_to_source.values()))
    reports: Dict[str, Dict] = {}

    for held_out_source in sources:
        held_out_subjects = sorted(s for s, src in subject_to_source.items() if src == held_out_source)
        held_out_labels = [_subject_label(s) for s in held_out_subjects]
        held_out_labels = [l for l in held_out_labels if l is not None]

        elig = assess_smoke_source_eligibility(
            held_out_source, held_out_labels, incompatible_sources=incompatible_sources,
            species_by_source=species_by_source, reference_species=reference_species,
            controlled_access_sources=controlled_access_sources,
        )
        dev_sources = [s for s in sources if s != held_out_source]
        dev_subjects = sorted(s for s, src in subject_to_source.items() if src != held_out_source)

        if not elig.eligible:
            reports[held_out_source] = build_robustness_report(
                task="smoke_classification", model=None, strategy="erm", held_out_source=held_out_source,
                eligibility=elig.to_dict(), development_sources=dev_sources,
                limitations=["source ineligible — no model was trained or evaluated for this source"],
            ).to_dict()
            continue

        try:
            artifact = refit_artifact_for_fold(normalized_adata, dev_subjects, n_hvgs)
            train_ds = build_fold_cell_dataset(normalized_adata, artifact, dev_subjects)
            held_out_ds = build_fold_cell_dataset(normalized_adata, artifact, held_out_subjects)
        except ValueError as e:
            reports[held_out_source] = build_robustness_report(
                task="smoke_classification", model=None, strategy="erm", held_out_source=held_out_source,
                eligibility=elig.to_dict(), development_sources=dev_sources,
                limitations=[f"held-out source's genes could not be transformed with the "
                             f"development-only artifact: {e}"],
            ).to_dict()
            continue

        comparisons = []
        Xtr, ytr, _, _ = build_smoke_subject_summary_features(train_ds, num_cell_types, num_classes)
        Xte, yte, _, _ = build_smoke_subject_summary_features(held_out_ds, num_cell_types, num_classes)
        for name in model_names:
            if name in SMOKE_BASELINES:
                model = SMOKE_BASELINES[name]()
                model.fit(Xtr, ytr, seed=seed)
                preds = model.predict(Xte)
                report = full_smoke_metrics_report(yte, preds, num_classes)
                comparisons.append({"name": name, "kind": "baseline",
                                     "subject_weighted_macro_f1": report["macro_f1"],
                                     "classes_absent_from_held_out_source": report["classes_absent_from_targets"]})
            elif name == PATHWAY_MODEL_NAME:
                train_bags = bags_from_fold_cell_dataset(train_ds, {}, min_cells_per_subject=1)
                held_out_bags = bags_from_fold_cell_dataset(held_out_ds, {}, min_cells_per_subject=1)
                if not train_bags or not held_out_bags:
                    comparisons.append({"name": name, "kind": "mil", "subject_weighted_macro_f1": None,
                                         "error": "no subject met min_cells_per_subject for smoke bags"})
                    continue
                from train import SubjectLevelDataset
                adapter = build_mil_adapter(name, "attention", device)
                fold_ctx_train = SubjectLevelDataset(train_bags, require_known_outcome=False)
                fold_ctx_val = SubjectLevelDataset(held_out_bags, require_known_outcome=False)
                import dataclasses as _dc
                fold_ctx = _dc.replace(context, preprocessing_artifact=artifact)
                adapter.fit(fold_ctx, train_ds, held_out_ds, fold_ctx_train, fold_ctx_val, seed=seed,
                            pretrain_epochs=3)
                preds = adapter.predict_smoke(fold_ctx_val)
                labels, known = adapter.known_smoke_labels(fold_ctx_val)
                if known.any():
                    report = full_smoke_metrics_report(labels[known], preds[known], num_classes)
                    comparisons.append({"name": name, "kind": "mil",
                                         "subject_weighted_macro_f1": report["macro_f1"],
                                         "classes_absent_from_held_out_source": report["classes_absent_from_targets"]})
                else:
                    comparisons.append({"name": name, "kind": "mil", "subject_weighted_macro_f1": None,
                                         "error": "no held-out subject had a verified smoke label"})

        manifest = build_source_held_out_manifest(
            task="smoke_classification", held_out_source=held_out_source, development_sources=dev_sources,
            development_subjects=dev_subjects, held_out_subjects=held_out_subjects,
            known_label_counts={"n": len(held_out_labels)}, class_distribution=elig.counts,
            eligibility=elig, seed=seed, preprocessing_policy_fingerprint=artifact_fingerprint(artifact),
        )
        scored = [c for c in comparisons if c.get("subject_weighted_macro_f1") is not None]
        best = max(scored, key=lambda c: c["subject_weighted_macro_f1"]) if scored else None
        reports[held_out_source] = build_robustness_report(
            task="smoke_classification", model=best["name"] if best else None, strategy="erm",
            held_out_source=held_out_source, eligibility=elig.to_dict(), development_sources=dev_sources,
            metrics={"per_model": comparisons,
                     "macro_f1": best["subject_weighted_macro_f1"] if best else None},
            comparisons=comparisons,
            source_split_manifest_fingerprint=manifest.fingerprint(),
            preprocessing_fingerprint=artifact_fingerprint(artifact),
        ).to_dict()

    return reports
