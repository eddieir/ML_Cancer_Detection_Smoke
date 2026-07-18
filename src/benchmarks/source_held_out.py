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

import dataclasses
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from data.manifest import DatasetManifestEntry, manifest_fingerprint
from data.splitting import grouped_kfold
from train import MILEligibilityError, SubjectLevelDataset

from .baselines import CANCER_BASELINES, CANCER_SEARCH_SPACE, SMOKE_BASELINES, SMOKE_SEARCH_SPACE, positive_class_proba
from .calibration import build_frozen_policy
from .cross_validation import (
    DEFAULT_INNER_FOLDS, MIL_SEARCH_SPACE, _cancer_baseline_fit_score_fn, _mil_fit_score_fn,
    _pathway_cancer_fit_score_fn, _pathway_smoke_fit_score_fn, _smoke_baseline_fit_score_fn,
)
from .domain_losses import DomainLossConfigurationError, resolve_domain_robustness_config
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
    fold_train_val_datasets,
    refit_artifact_for_fold,
    require_normalized_adata,
)
from .hyperparameter_search import build_param_grid, select_nested_hyperparameters_with_refit
from .metrics import full_smoke_metrics_report
from .mil_registry import build_mil_adapter, pathway_search_space
from .pathway_hierarchical_adapter import MODEL_NAME as PATHWAY_MODEL_NAME
from .robustness_report import build_robustness_report, not_applicable
from .source_held_out_diagnostics import (
    cancer_biological_stability_report,
    cancer_domain_shift_report,
    cancer_uncertainty_report,
    smoke_domain_shift_report,
    smoke_uncertainty_report,
)
from .source_eligibility import (
    assess_cancer_source_eligibility,
    assess_smoke_source_eligibility,
    build_source_held_out_manifest,
    resolve_source_policy,
)

DEFAULT_OOF_FOLDS = 3
DEFAULT_SMOKE_DEV_CV_FOLDS = 3


class ConflictingSmokeLabelError(ValueError):
    """A subject's own VERIFIED (smoke_type_known=True) cells do not agree on
    a single smoke_type value — never silently resolved by majority vote
    across conflicting verified evidence (distinct from a subject simply
    having no verified cells at all, which is reported as unknown, not a
    conflict)."""


class LabelSchemaError(ValueError):
    """Raised when normalized_adata.obs is missing the smoke_type_known
    column this protocol depends on — the absence of a verified-label gate
    must never be silently treated as "every label is verified"; see
    data/label_state.py's known/unknown vocabulary, which this function
    reads rather than reimplements."""


def _verified_smoke_subject_labels(
    normalized_adata, subjects: Sequence[str], weak_labels_enabled: bool = False,
) -> Tuple[Dict[str, int], Dict]:
    """
    Subject -> smoke_type label, restricted to subjects whose smoke_type_known
    cells agree on exactly one value. A subject with zero verified cells (all
    unknown, or all weak-proxy cells with weak-proxy promotion disabled) is
    reported as unknown and excluded from the returned mapping rather than
    silently defaulting to whatever the majority raw smoke_type happens to be
    — see data/label_state.py and data/labellers.py's smoke_type_known gate,
    which this function reads rather than reimplements. Raises
    LabelSchemaError if smoke_type_known is absent — never defaults to
    "every label is verified".
    """
    obs = normalized_adata.obs
    if "smoke_type_known" not in obs.columns:
        raise LabelSchemaError(
            "normalized_adata.obs has no 'smoke_type_known' column — this protocol requires an "
            "explicit verified-label gate and never assumes every smoke_type value is verified."
        )
    subj_series = obs["subject_id"].astype(str)
    smoke_series = obs["smoke_type"].astype(int)
    known_arr = obs["smoke_type_known"].astype(bool).values
    weak_proxy_arr = (
        obs["weak_smoke_proxy_known"].astype(bool).values
        if "weak_smoke_proxy_known" in obs.columns
        else np.zeros(len(obs), dtype=bool)
    )

    label_by_subject: Dict[str, int] = {}
    unknown_subjects: List[str] = []
    weak_proxy_only_subjects: List[str] = []
    excluded_by_policy_subjects: List[str] = []
    conflicts: Dict[str, List[int]] = {}
    for sid in subjects:
        sid = str(sid)
        subj_mask = (subj_series == sid).values
        verified_mask = subj_mask & known_arr
        if not verified_mask.any():
            if (subj_mask & weak_proxy_arr).any():
                weak_proxy_only_subjects.append(sid)
                if not weak_labels_enabled:
                    excluded_by_policy_subjects.append(sid)
            else:
                unknown_subjects.append(sid)
            continue
        vals = sorted(set(smoke_series.values[verified_mask].tolist()))
        if len(vals) > 1:
            conflicts[sid] = vals
            continue
        label_by_subject[sid] = vals[0]

    if conflicts:
        raise ConflictingSmokeLabelError(
            f"{len(conflicts)} subject(s) have conflicting VERIFIED smoke_type labels across "
            f"their own cells (e.g. {dict(list(conflicts.items())[:3])}) — never resolved by "
            "majority vote."
        )

    class_distribution: Dict[str, int] = {}
    for v in label_by_subject.values():
        class_distribution[str(v)] = class_distribution.get(str(v), 0) + 1

    diagnostics = {
        "total_subjects": len(subjects),
        "verified_subjects": sorted(label_by_subject),
        "unknown_subjects": sorted(unknown_subjects),
        "weak_proxy_only_subjects": sorted(weak_proxy_only_subjects),
        "excluded_by_policy_subjects": sorted(excluded_by_policy_subjects),
        "verified_labels": len(label_by_subject),
        "unknown_labels": len(unknown_subjects),
        "conflicting_labels": len(conflicts),
        "verified_class_distribution": class_distribution,
        "weak_labels_enabled": weak_labels_enabled,
        "weak_label_policy_fingerprint": _sha256_json({"weak_labels_enabled": weak_labels_enabled}),
    }
    return label_by_subject, diagnostics


class CrossSourceSubjectConflictError(ValueError):
    """A subject_id assigned to more than one dataset_source in the
    train+val pool — a data-assembly inconsistency, never silently
    resolved by picking one source (see benchmarks/ood.py's identical
    check for the pre-existing Task A LOSO path)."""


class MissingSourceProvenanceError(ValueError):
    """A subject's dataset_source is blank, a placeholder value, or absent
    entirely — a source-aware protocol must reject this outright rather than
    silently mapping it to a literal "unknown" bucket, which would let an
    unprovenanced subject quietly participate in source-held-out selection,
    CORAL/MMD, source-balanced sampling, or domain-adversarial training."""


_PLACEHOLDER_SOURCE_VALUES = frozenset({"", "unknown", "none", "nan", "null", "n/a", "na"})


def _reject_placeholder_sources(by_subject: Dict[str, str]) -> None:
    bad = {sid: src for sid, src in by_subject.items() if src.strip().lower() in _PLACEHOLDER_SOURCE_VALUES}
    if bad:
        raise MissingSourceProvenanceError(
            f"{len(bad)} subject(s) have a blank or placeholder dataset_source value (e.g. "
            f"{dict(list(bad.items())[:5])}) — a source-aware protocol requires a real source "
            "identity for every subject, never a placeholder silently treated as a source."
        )


def _pool_subjects_and_sources(context) -> Dict[str, str]:
    normalized_adata = require_normalized_adata(context)
    if "source" not in normalized_adata.obs.columns:
        raise MissingSourceProvenanceError(
            "normalized_adata.obs has no 'source' column — a source-aware protocol cannot run "
            "without per-cell dataset-source provenance."
        )
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
    resolved = {sid: next(iter(srcs)) for sid, srcs in by_subject.items()}
    _reject_placeholder_sources(resolved)
    return resolved


# ─── Task B: cancer prediction ─────────────────────────────────────────────

def _sha256_json(payload) -> str:
    import json as _json
    return __import__("hashlib").sha256(_json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _domain_head_fingerprint(adapter) -> Optional[str]:
    """SHA-256 of the domain-adversarial head's weights + fixed source
    vocabulary, or None when this adapter has no domain head (ERM/
    source_balanced/coral/mmd, or an adapter that was never built with
    strategy='domain_adversarial')."""
    if getattr(adapter, "domain_head", None) is None:
        return None
    import torch
    state = adapter.domain_head.state_dict()
    blob = b"".join(t.detach().cpu().numpy().tobytes() for t in state.values())
    vocab_blob = _sha256_json(sorted(adapter.domain_source_vocabulary or [])).encode("utf-8")
    return __import__("hashlib").sha256(blob + vocab_blob).hexdigest()


class EnvironmentSnapshotError(RuntimeError):
    """Raised when the running environment's core package versions cannot be
    collected in full — an evaluated report must never silently record an
    unknown/None environment identity."""


def _environment_fingerprint() -> str:
    from .env_versions import collect_core_package_versions
    try:
        versions = collect_core_package_versions(required=True)
    except RuntimeError as e:
        raise EnvironmentSnapshotError(
            f"cannot build a validated environment fingerprint: {e}"
        ) from e
    return _sha256_json(versions)


def run_cancer_source_held_out(
    context, model_names: Sequence[str], device: str = "cpu",
    domain_robustness_config: Optional[Dict] = None,
    incompatible_sources: Optional[Sequence[str]] = None,
    species_by_source: Optional[Dict[str, str]] = None,
    reference_species: Optional[str] = None,
    controlled_access_sources: Optional[Sequence[str]] = None,
    reference_assay_mode: Optional[str] = None,
    seed: int = 42, n_oof_folds: int = DEFAULT_OOF_FOLDS,
    dataset_manifest_entries: Optional[Sequence[DatasetManifestEntry]] = None,
    stability_extra_seeds: Sequence[int] = (),
) -> Dict:
    """
    For every source present in the train+val pool: assess eligibility,
    then (if eligible) select the best of model_names by development-only
    OOF AUROC computed EXCLUSIVELY from the remaining (development) sources'
    subjects, fit that candidate once on the full development pool, and
    apply the frozen result to the held-out source's subjects exactly once.
    Returns {source: RobustnessReport dict}.

    dataset_manifest_entries, if supplied, makes source_eligibility.py
    prefer the canonical dataset manifest (data/manifest.py) over caller-
    supplied species_by_source/controlled_access_sources for any source it
    declares, and raises SourcePolicyDriftError if the caller-supplied value
    contradicts it (see source_eligibility.py::resolve_source_policy).

    stability_extra_seeds, if supplied, opts every pathway_hierarchical_mil
    candidate's biological_stability.cross_run_stability field into GENUINE
    independent multi-seed refits (see cancer_biological_stability_report) —
    left empty (the default) because each extra seed is a full extra model
    fit; cross_run_stability reports insufficient_evidence rather than a
    fabricated single-run "stability" when left empty.
    """
    domain_cfg = resolve_domain_robustness_config(domain_robustness_config)
    subject_to_source = _pool_subjects_and_sources(context)
    all_bags = list(context.train_bags) + list(context.val_bags)
    outcomes_by_subject = {str(b["subject_id"]): b["cancer_label"] for b in all_bags if b.get("cancer_label_known")}
    num_cell_types = context.config.get("model", {}).get("num_cell_types", 4)
    n_hvgs = context.preprocessing_artifact.n_hvgs
    min_cells = context.config.get("data", {}).get("min_cells_per_subject", 50)

    manifest_by_source: Dict[str, DatasetManifestEntry] = {}
    dataset_manifest_fp: Optional[str] = None
    if dataset_manifest_entries:
        for e in dataset_manifest_entries:
            manifest_by_source[e.dataset_id] = e
            manifest_by_source[e.accession] = e
        dataset_manifest_fp = manifest_fingerprint(list(dataset_manifest_entries))
    environment_fp = _environment_fingerprint()

    sources = sorted(set(subject_to_source.values()))
    reports: Dict[str, Dict] = {}

    for held_out_source in sources:
        held_out_subjects = sorted(s for s, src in subject_to_source.items() if src == held_out_source)
        held_out_outcomes = [outcomes_by_subject.get(s) for s in held_out_subjects]

        elig = assess_cancer_source_eligibility(
            held_out_source, held_out_outcomes, incompatible_sources=incompatible_sources,
            species_by_source=species_by_source, reference_species=reference_species,
            reference_assay_mode=reference_assay_mode,
            controlled_access_sources=controlled_access_sources, manifest_by_source=manifest_by_source,
        )
        source_policy_fp = _sha256_json(resolve_source_policy(
            held_out_source, species_by_source, controlled_access_sources, manifest_by_source,
        ))
        dev_sources = [s for s in sources if s != held_out_source]
        dev_subjects_all = sorted(s for s, src in subject_to_source.items() if src != held_out_source)

        if not elig.eligible:
            manifest = build_source_held_out_manifest(
                task="cancer_prediction", held_out_source=held_out_source, development_sources=dev_sources,
                development_subjects=dev_subjects_all, held_out_subjects=held_out_subjects,
                known_label_counts=elig.counts, class_distribution={}, eligibility=elig, seed=seed,
                dataset_manifest_fingerprint=dataset_manifest_fp,
            )
            reports[held_out_source] = build_robustness_report(
                task="cancer_prediction", model=None, strategy=domain_cfg["strategy"],
                held_out_source=held_out_source, eligibility=elig.to_dict(), development_sources=dev_sources,
                source_split_manifest_fingerprint=manifest.fingerprint(),
                dataset_manifest_fingerprint=dataset_manifest_fp, source_policy_fingerprint=source_policy_fp,
                environment_fingerprint=environment_fp, seed=seed,
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
            except MILEligibilityError as e:
                candidate_scores[name] = {"auroc": None, "error": f"MIL ineligible: {e}"}
                continue
            except DomainLossConfigurationError:
                raise  # a configuration error is never a per-candidate ineligibility outcome
            except ValueError as e:
                # Expected candidate-ineligibility outcomes only (e.g. "fewer than 2 known-outcome
                # development subjects") — never RuntimeError (leakage/duplicate-prediction bugs),
                # never a configuration/provenance/label-integrity error, all of which must abort
                # the whole sweep rather than being recorded as a candidate score.
                candidate_scores[name] = {"auroc": None, "error": str(e)}
                continue
            except RuntimeError as e:
                # generate_subject_oof_predictions raises RuntimeError for two structurally
                # different situations: (a) a dev subject never received an OOF prediction because
                # every fold containing it was skipped — an expected small-dev-pool-for-this-
                # candidate outcome, recorded as a candidate ineligibility; (b) a subject predicted
                # by a fold that also trained on it — genuine in-fold leakage, which must always
                # abort the sweep rather than being silently recorded as a low score.
                if "never received an OOF prediction" in str(e):
                    candidate_scores[name] = {"auroc": None, "error": str(e)}
                    continue
                raise
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
                dataset_manifest_fingerprint=dataset_manifest_fp, source_policy_fingerprint=source_policy_fp,
                environment_fingerprint=environment_fp, seed=seed,
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
                dataset_manifest_fingerprint=dataset_manifest_fp, source_policy_fingerprint=source_policy_fp,
                environment_fingerprint=environment_fp, seed=seed,
                preprocessing_fingerprint=fitted.preprocessing_artifact_fingerprint,
            ).to_dict()
            continue

        if fitted.kind == "mil":
            held_out_sd = SubjectLevelDataset(held_out_bags)
            held_out_proba_raw = fitted.predictor.predict_proba(held_out_sd)
            held_out_order = [str(b["subject_id"]) for b in held_out_sd.bags]
        else:
            Xho, _, held_out_order, _ = build_cancer_subject_features(held_out_bags, num_cell_types)
            held_out_proba_raw = positive_class_proba(fitted.predictor, Xho)

        y_held_out = np.array([outcomes_by_subject[s] for s in held_out_order])
        metrics_result = policy.apply_to_test(y_held_out, held_out_proba_raw)
        policy_dict = metrics_result.get("policy", {})
        calibration_fp = _sha256_json(policy_dict.get("calibration", {}))
        threshold_policy_fp = _sha256_json({
            "threshold": policy_dict.get("threshold"), "threshold_strategy": policy_dict.get("threshold_strategy"),
            "threshold_reason": policy_dict.get("threshold_reason"),
        })

        dev_ds_for_diag = build_fold_cell_dataset(normalized_adata, fitted.preprocessing_artifact, dev_subjects)
        dev_bags_for_diag = bags_from_fold_cell_dataset(dev_ds_for_diag, outcomes_by_subject, min_cells)
        module_info = None
        if fitted.kind == "mil" and best_name == PATHWAY_MODEL_NAME:
            module_info = fitted.predictor.modules
        domain_shift = cancer_domain_shift_report(
            dev_bags_for_diag, held_out_bags, num_cell_types, subject_to_source, seed=seed,
            required_gene_list=list(fitted.preprocessing_artifact.gene_list), modules=module_info,
        )
        uncertainty = cancer_uncertainty_report(
            y_oof, prob_oof, y_held_out, held_out_proba_raw, fitted=fitted, held_out_bags=held_out_bags,
        )
        biological_stability = cancer_biological_stability_report(
            context, fitted, dev_bags_for_diag, held_out_bags, outcomes_by_subject, num_cell_types, min_cells,
            seed=seed, extra_seeds=stability_extra_seeds,
        )

        gene_list_fp = _sha256_json(list(fitted.preprocessing_artifact.gene_list))
        if fitted.kind == "mil" and best_name == PATHWAY_MODEL_NAME:
            domain_vocab_fp = _sha256_json(sorted(fitted.model_metadata.get("domain_source_vocabulary") or []))
            domain_head_fp = _domain_head_fingerprint(fitted.predictor)
            if domain_head_fp is None:
                domain_head_fp = not_applicable("this candidate was not built with strategy='domain_adversarial'")
        else:
            domain_vocab_fp = not_applicable(f"{best_name} has no domain-source vocabulary")
            domain_head_fp = not_applicable(f"{best_name} has no domain-adversarial head")

        manifest = build_source_held_out_manifest(
            task="cancer_prediction", held_out_source=held_out_source, development_sources=dev_sources,
            development_subjects=dev_subjects_all, held_out_subjects=held_out_subjects,
            known_label_counts=elig.counts, class_distribution={"n_positive": elig.counts.get("n_positive"),
                                                                   "n_negative": elig.counts.get("n_negative")},
            eligibility=elig, seed=seed, preprocessing_policy_fingerprint=fitted.preprocessing_artifact_fingerprint,
            module_fingerprint=fitted.model_metadata.get("module_fingerprint"),
            dataset_manifest_fingerprint=dataset_manifest_fp,
        )

        module_fp = fitted.model_metadata.get("module_fingerprint")
        if module_fp is None:
            module_fp = not_applicable(f"{best_name} has no gene-module structure")

        reports[held_out_source] = build_robustness_report(
            task="cancer_prediction", model=best_name, strategy=domain_cfg["strategy"],
            held_out_source=held_out_source, eligibility=elig.to_dict(), development_sources=dev_sources,
            metrics={k: v for k, v in metrics_result.items() if k != "policy"},
            calibration=metrics_result.get("policy", {}),
            uncertainty=uncertainty, domain_shift=domain_shift, biological_stability=biological_stability,
            comparisons=[{"name": n, **s} for n, s in candidate_scores.items()],
            source_split_manifest_fingerprint=manifest.fingerprint(),
            preprocessing_fingerprint=fitted.preprocessing_artifact_fingerprint,
            module_fingerprint=module_fp,
            model_fingerprint=fitted.model_state_fingerprint,
            calibration_fingerprint=calibration_fp, threshold_policy_fingerprint=threshold_policy_fp,
            dataset_manifest_fingerprint=dataset_manifest_fp,
            source_policy_fingerprint=source_policy_fp, environment_fingerprint=environment_fp,
            gene_list_fingerprint=gene_list_fp, domain_vocabulary_fingerprint=domain_vocab_fp,
            domain_head_fingerprint=domain_head_fp, seed=seed, evaluated=True,
            is_module_based_candidate=(best_name == PATHWAY_MODEL_NAME),
        ).to_dict()

    return reports


# ─── Task A: smoke-type classification ─────────────────────────────────────

class UnsupportedSmokeCandidateError(ValueError):
    """Raised for a requested Task A source-held-out candidate that is
    neither a classical SMOKE_BASELINES entry nor pathway_hierarchical_mil.
    Task A source-held-out deliberately does not cover the pooling-based MIL
    models ("neural", "mean_mil", "max_mil", "attention_mil") or the
    cell-level MultiSmokeCancerNet Trainer curriculum (see this module's
    docstring and README.md's honest-limitations section) — an unsupported
    name must fail loudly, never be silently skipped."""


class UnsupportedSmokeDomainStrategyError(ValueError):
    """Task A source-held-out supports ERM only. CORAL/MMD/domain-adversarial/
    source-balanced training is cancer-only (Task B) in this repository — see
    README.md's task/strategy support matrix. A caller requesting a
    non-"erm" strategy for the smoke task must be rejected explicitly rather
    than silently downgraded to ERM."""


_SUPPORTED_SMOKE_CANDIDATES = frozenset(set(SMOKE_BASELINES) | {PATHWAY_MODEL_NAME})


def _align_proba_to_full_classes(proba: np.ndarray, model_classes: Sequence[int], num_classes: int) -> np.ndarray:
    """
    A fold's training data can legitimately miss a class entirely (a small
    grouped-CV fold, especially for a rare smoke type) — sklearn's
    predict_proba then returns columns only for model.classes_ (whatever
    subset the model actually saw), never the full [0, num_classes) range.
    Re-embeds those columns into a FIXED-width, fixed-order [n, num_classes]
    array (missing classes explicitly zero-filled) so every fold's OOF
    probabilities share one consistent class axis regardless of which
    classes that fold's training data happened to contain.
    """
    out = np.zeros((proba.shape[0], num_classes), dtype=np.float64)
    for col, cls in enumerate(model_classes):
        cls = int(cls)
        if 0 <= cls < num_classes:
            out[:, cls] = proba[:, col]
    return out


def _smoke_candidate_dev_score(
    context, name: str, dev_subjects: Sequence[str], label_by_subject: Dict[str, int],
    num_cell_types: int, num_classes: int, n_hvgs: int, device: str, seed: int,
    n_dev_folds: int, n_inner_folds: int,
) -> Tuple[Optional[float], Dict]:
    """
    Development-only grouped-CV mean macro-F1 for one candidate, computed
    ENTIRELY from dev_subjects (never held-out-source subjects). Each fold's
    hyperparameters are selected by a fresh inner-CV run restricted to that
    fold's own training subjects (select_nested_hyperparameters_with_refit),
    mirroring cross_validation.run_smoke_cv's per-fold nested-selection
    pattern but scoped to this held-out source's development pool only.

    Every dev subject is predicted by EXACTLY ONE fold (the one it was held
    out of) — never a fold it also trained on — so evidence["oof_by_subject"]
    is a genuine out-of-fold probability vector (fixed num_classes columns,
    see _align_proba_to_full_classes) per subject with a defined fold
    prediction, never an in-sample probability. A subject whose every
    containing fold was skipped (empty train/val bags after the min-cells/
    known-label mask) legitimately has no OOF entry — the caller must not
    assume every dev_subject appears in oof_by_subject.

    Returns (mean_macro_f1_or_None, evidence_dict). evidence_dict always has
    an "oof_by_subject" key (possibly empty) so a caller can rely on its
    presence without a hasattr/get-with-default dance.
    """
    labeled_dev_subjects = sorted(s for s in dev_subjects if s in label_by_subject)
    oof_by_subject: Dict[str, np.ndarray] = {}
    if len(labeled_dev_subjects) < 2:
        return None, {"error": "fewer than 2 verified-label development subjects", "oof_by_subject": oof_by_subject}
    y_full = np.array([label_by_subject[s] for s in labeled_dev_subjects])
    if len(set(y_full.tolist())) < 2:
        return None, {"error": "development pool has only one verified smoke class", "oof_by_subject": oof_by_subject}

    folds = grouped_kfold(np.array(labeled_dev_subjects), y_full, n_folds=n_dev_folds, seed=seed)
    fold_scores, fold_records = [], []
    seen_oof_subjects: set = set()
    for fold_idx, fold in enumerate(folds):
        artifact, train_ds, val_ds = fold_train_val_datasets(
            context, fold["train"], fold["val"], n_hvgs=n_hvgs,
        )
        if name in SMOKE_BASELINES:
            hp_search = select_nested_hyperparameters_with_refit(
                context, fold["train"], label_by_subject, build_param_grid(SMOKE_SEARCH_SPACE.get(name, {})),
                fit_score_fn=_smoke_baseline_fit_score_fn(SMOKE_BASELINES[name], num_cell_types, num_classes),
                seed=seed, n_inner_folds=n_inner_folds, n_hvgs=n_hvgs,
            )
            _, _, subj_tr, _ = build_smoke_subject_summary_features(train_ds, num_cell_types, num_classes)
            Xva_raw, _, subj_va, _ = build_smoke_subject_summary_features(val_ds, num_cell_types, num_classes)
            tr_mask = [s in label_by_subject for s in subj_tr]
            va_mask = np.array([s in label_by_subject for s in subj_va])
            if not any(tr_mask) or not va_mask.any():
                continue
            Xtr_full, _, _, _ = build_smoke_subject_summary_features(train_ds, num_cell_types, num_classes)
            Xtr = Xtr_full[tr_mask]
            ytr = np.array([label_by_subject[s] for s, keep in zip(subj_tr, tr_mask) if keep])
            model = SMOKE_BASELINES[name](**hp_search["selected_params"]).fit(Xtr, ytr, seed=seed)
            preds = model.predict(Xva_raw[va_mask])
            yva = np.array([label_by_subject[s] for s, keep in zip(subj_va, va_mask) if keep])
            report = full_smoke_metrics_report(yva, preds, num_classes)
            proba_va = model.predict_proba(Xva_raw[va_mask])
            proba_va_aligned = _align_proba_to_full_classes(proba_va, model.model.classes_, num_classes)
            va_subjects = [s for s, keep in zip(subj_va, va_mask) if keep]
        elif name == PATHWAY_MODEL_NAME:
            train_bags = bags_from_fold_cell_dataset(train_ds, {}, min_cells_per_subject=1)
            val_bags = bags_from_fold_cell_dataset(val_ds, {}, min_cells_per_subject=1)
            if not train_bags or not val_bags:
                continue
            hp_search = select_nested_hyperparameters_with_refit(
                context, fold["train"], label_by_subject, build_param_grid(pathway_search_space(fast=False)),
                fit_score_fn=_pathway_smoke_fit_score_fn(context, device, num_classes),
                seed=seed, n_inner_folds=n_inner_folds, n_hvgs=n_hvgs,
            )
            train_sd = SubjectLevelDataset(train_bags, require_known_outcome=False)
            val_sd = SubjectLevelDataset(val_bags, require_known_outcome=False)
            fold_ctx = dataclasses.replace(context, preprocessing_artifact=artifact)
            adapter = build_mil_adapter(PATHWAY_MODEL_NAME, None, device, config_overrides=hp_search["selected_params"])
            adapter.fit(fold_ctx, train_ds, val_ds, train_sd, val_sd, seed=seed)
            preds = adapter.predict_smoke(val_sd)
            y_val, known = adapter.known_smoke_labels(val_sd)
            if not known.any():
                continue
            report = full_smoke_metrics_report(y_val[known], preds[known], num_classes)
            proba_va_aligned = adapter.predict_smoke_proba(val_sd)[known]
            va_subjects = [str(b["subject_id"]) for b, k in zip(val_sd.bags, known) if k]
        else:
            raise UnsupportedSmokeCandidateError(
                f"{name!r} is not a supported Task A source-held-out candidate — "
                f"supported names are {sorted(_SUPPORTED_SMOKE_CANDIDATES)}."
            )
        fold_scores.append(report["macro_f1"])
        fold_records.append({"fold": fold_idx, "macro_f1": report["macro_f1"], "hyperparameter_search": hp_search})
        for sid, proba_row in zip(va_subjects, proba_va_aligned):
            if sid in seen_oof_subjects:
                raise RuntimeError(
                    f"_smoke_candidate_dev_score: subject {sid!r} received an OOF prediction from more "
                    "than one fold — grouped_kfold must assign every subject to exactly one validation "
                    "fold; this indicates a fold-construction bug, never an expected outcome."
                )
            seen_oof_subjects.add(sid)
            oof_by_subject[sid] = proba_row

    if not fold_scores:
        return None, {"error": "no development fold produced a defined macro-F1", "folds": fold_records,
                       "oof_by_subject": oof_by_subject}
    return float(np.mean(fold_scores)), {
        "folds": fold_records, "n_folds_scored": len(fold_scores), "oof_by_subject": oof_by_subject,
    }


def run_smoke_source_held_out(
    context, model_names: Sequence[str], device: str = "cpu",
    domain_robustness_config: Optional[Dict] = None,
    incompatible_sources: Optional[Sequence[str]] = None,
    species_by_source: Optional[Dict[str, str]] = None,
    reference_species: Optional[str] = None,
    controlled_access_sources: Optional[Sequence[str]] = None,
    reference_assay_mode: Optional[str] = None,
    seed: int = 42, n_dev_cv_folds: int = DEFAULT_SMOKE_DEV_CV_FOLDS, n_inner_folds: int = DEFAULT_INNER_FOLDS,
    dataset_manifest_entries: Optional[Sequence[DatasetManifestEntry]] = None,
) -> Dict:
    """
    Task A source-held-out evaluation. Only ERM is supported (see
    UnsupportedSmokeDomainStrategyError) — CORAL/MMD/domain-adversarial/
    source-balanced training is Task B (cancer) only in this repository.

    For every source: model/hyperparameter selection is a development-only
    grouped-CV ranking computed EXCLUSIVELY from the remaining (development)
    sources' verified-label subjects (_smoke_candidate_dev_score); the
    selected candidate's hyperparameters are then reselected once more from
    the FULL development pool (never the held-out source), the model is
    refit once on the full development pool with no internal validation
    carve-out (adapter.fit_final for pathway_hierarchical_mil; a plain
    baseline .fit has no validation concept to begin with), and the frozen
    result is applied to the held-out source's subjects exactly once. The
    pooling-based MIL models ("neural", "mean_mil", "max_mil",
    "attention_mil") and the cell-level MultiSmokeCancerNet Trainer
    curriculum remain out of scope for this function — see README.md.

    Smoke labels are VERIFIED-only (smoke_type_known=True cells, see
    _verified_smoke_subject_labels) — unknown and (unless explicitly
    enabled) weak-proxy cells never contribute to a subject's ground-truth
    label, and a subject whose verified cells disagree raises
    ConflictingSmokeLabelError rather than being resolved by majority vote.
    """
    domain_cfg = resolve_domain_robustness_config(domain_robustness_config)
    if domain_cfg["strategy"] != "erm":
        raise UnsupportedSmokeDomainStrategyError(
            f"run_smoke_source_held_out: strategy={domain_cfg['strategy']!r} is not supported for "
            "Task A — domain-robust training strategies are cancer-only in this repository."
        )
    unsupported = sorted(set(model_names) - _SUPPORTED_SMOKE_CANDIDATES)
    if unsupported:
        raise UnsupportedSmokeCandidateError(
            f"run_smoke_source_held_out: unsupported candidate name(s) {unsupported} — "
            f"supported names are {sorted(_SUPPORTED_SMOKE_CANDIDATES)}."
        )

    normalized_adata = require_normalized_adata(context)
    num_classes = context.num_smoke_classes
    num_cell_types = context.config.get("model", {}).get("num_cell_types", 4)
    n_hvgs = context.preprocessing_artifact.n_hvgs
    weak_labels_enabled = bool(context.config.get("data", {}).get("weak_labels", {}).get("enabled", False))

    manifest_by_source: Dict[str, DatasetManifestEntry] = {}
    dataset_manifest_fp: Optional[str] = None
    if dataset_manifest_entries:
        for e in dataset_manifest_entries:
            manifest_by_source[e.dataset_id] = e
            manifest_by_source[e.accession] = e
        dataset_manifest_fp = manifest_fingerprint(list(dataset_manifest_entries))
    environment_fp = _environment_fingerprint()

    subject_to_source = _pool_subjects_and_sources(context)
    sources = sorted(set(subject_to_source.values()))
    reports: Dict[str, Dict] = {}

    for held_out_source in sources:
        held_out_subjects = sorted(s for s, src in subject_to_source.items() if src == held_out_source)
        held_out_label_by_subject, held_out_label_diagnostics = _verified_smoke_subject_labels(
            normalized_adata, held_out_subjects, weak_labels_enabled=weak_labels_enabled,
        )
        held_out_labels = list(held_out_label_by_subject.values())

        elig = assess_smoke_source_eligibility(
            held_out_source, held_out_labels, incompatible_sources=incompatible_sources,
            species_by_source=species_by_source, reference_species=reference_species,
            reference_assay_mode=reference_assay_mode,
            controlled_access_sources=controlled_access_sources, manifest_by_source=manifest_by_source,
        )
        source_policy_fp = _sha256_json(resolve_source_policy(
            held_out_source, species_by_source, controlled_access_sources, manifest_by_source,
        ))
        dev_sources = [s for s in sources if s != held_out_source]
        dev_subjects = sorted(s for s, src in subject_to_source.items() if src != held_out_source)

        if not elig.eligible:
            reports[held_out_source] = build_robustness_report(
                task="smoke_classification", model=None, strategy="erm", held_out_source=held_out_source,
                eligibility=elig.to_dict(), development_sources=dev_sources,
                limitations=["source ineligible — no model was trained or evaluated for this source"],
                label_state={"held_out": held_out_label_diagnostics, "weak_labels_enabled": weak_labels_enabled},
                dataset_manifest_fingerprint=dataset_manifest_fp, source_policy_fingerprint=source_policy_fp,
                environment_fingerprint=environment_fp, seed=seed,
            ).to_dict()
            continue

        dev_label_by_subject, dev_label_diagnostics = _verified_smoke_subject_labels(
            normalized_adata, dev_subjects, weak_labels_enabled=weak_labels_enabled,
        )

        candidate_scores: Dict[str, Dict] = {}
        best_name, best_score = None, -np.inf
        best_oof_by_subject: Dict[str, np.ndarray] = {}
        for name in model_names:
            score, evidence = _smoke_candidate_dev_score(
                context, name, dev_subjects, dev_label_by_subject, num_cell_types, num_classes,
                n_hvgs, device, seed, n_dev_cv_folds, n_inner_folds,
            )
            # oof_by_subject carries raw subject IDs and per-subject
            # probability vectors — never let it flow into candidate_scores/
            # comparisons, which are embedded verbatim in the persisted
            # report; it is retained here only for the WINNING candidate's
            # internal uncertainty computation below.
            oof_by_subject = evidence.pop("oof_by_subject", {})
            candidate_scores[name] = {"development_macro_f1": score, **evidence}
            if score is not None and score > best_score:
                best_name, best_score = name, score
                best_oof_by_subject = oof_by_subject

        if best_name is None:
            reports[held_out_source] = build_robustness_report(
                task="smoke_classification", model=None, strategy="erm", held_out_source=held_out_source,
                eligibility=elig.to_dict(), development_sources=dev_sources,
                limitations=["no candidate produced a defined development-only macro-F1"],
                comparisons=[{"name": n, **s} for n, s in candidate_scores.items()],
                label_state={"development": dev_label_diagnostics, "held_out": held_out_label_diagnostics,
                             "weak_labels_enabled": weak_labels_enabled},
                dataset_manifest_fingerprint=dataset_manifest_fp, source_policy_fingerprint=source_policy_fp,
                environment_fingerprint=environment_fp, seed=seed,
            ).to_dict()
            continue

        # One more, clearly-declared nested hyperparameter selection over the
        # FULL development pool for the final fit (mirrors
        # run_cancer_source_held_out's identical "hp_for_final" pattern) —
        # never re-selected using any per-fold subset or held-out data.
        if best_name in SMOKE_BASELINES:
            fit_score_fn = _smoke_baseline_fit_score_fn(SMOKE_BASELINES[best_name], num_cell_types, num_classes)
            candidates = build_param_grid(SMOKE_SEARCH_SPACE.get(best_name, {}))
        else:
            fit_score_fn = _pathway_smoke_fit_score_fn(context, device, num_classes)
            candidates = build_param_grid(pathway_search_space(fast=False))
        labeled_dev_subjects = sorted(dev_label_by_subject)
        final_hp_search = select_nested_hyperparameters_with_refit(
            context, labeled_dev_subjects, dev_label_by_subject, candidates, fit_score_fn=fit_score_fn,
            seed=seed, n_inner_folds=n_inner_folds, n_hvgs=n_hvgs,
        )
        selected_params = final_hp_search["selected_params"]

        try:
            final_artifact = refit_artifact_for_fold(normalized_adata, dev_subjects, n_hvgs)
            dev_ds = build_fold_cell_dataset(normalized_adata, final_artifact, dev_subjects)
            held_out_ds = build_fold_cell_dataset(normalized_adata, final_artifact, held_out_subjects)
        except ValueError as e:
            reports[held_out_source] = build_robustness_report(
                task="smoke_classification", model=best_name, strategy="erm", held_out_source=held_out_source,
                eligibility=elig.to_dict(), development_sources=dev_sources,
                limitations=[f"held-out source's genes could not be transformed with the "
                             f"development-only artifact: {e}"],
                comparisons=[{"name": n, **s} for n, s in candidate_scores.items()],
                dataset_manifest_fingerprint=dataset_manifest_fp, source_policy_fingerprint=source_policy_fp,
                environment_fingerprint=environment_fp, seed=seed,
            ).to_dict()
            continue

        if best_name in SMOKE_BASELINES:
            Xdev_full, _, subj_dev, _ = build_smoke_subject_summary_features(dev_ds, num_cell_types, num_classes)
            dev_mask = [s in dev_label_by_subject for s in subj_dev]
            Xdev = Xdev_full[dev_mask]
            ydev = np.array([dev_label_by_subject[s] for s, keep in zip(subj_dev, dev_mask) if keep])
            model = SMOKE_BASELINES[best_name](**selected_params).fit(Xdev, ydev, seed=seed)
            Xte_full, _, subj_te, _ = build_smoke_subject_summary_features(held_out_ds, num_cell_types, num_classes)
            te_mask = np.array([s in held_out_label_by_subject for s in subj_te])
            if not te_mask.any():
                reports[held_out_source] = build_robustness_report(
                    task="smoke_classification", model=best_name, strategy="erm", held_out_source=held_out_source,
                    eligibility=elig.to_dict(), development_sources=dev_sources,
                    limitations=["no held-out subject had a verified smoke label"],
                    comparisons=[{"name": n, **s} for n, s in candidate_scores.items()],
                    dataset_manifest_fingerprint=dataset_manifest_fp, source_policy_fingerprint=source_policy_fp,
                    environment_fingerprint=environment_fp, seed=seed,
                ).to_dict()
                continue
            preds = model.predict(Xte_full[te_mask])
            yte = np.array([held_out_label_by_subject[s] for s, keep in zip(subj_te, te_mask) if keep])
            report = full_smoke_metrics_report(yte, preds, num_classes)
            model_state_fp = model.model_state_fingerprint()
            held_out_proba = model.predict_proba(Xte_full[te_mask])
            held_out_labels_for_uncertainty = yte
        else:
            dev_bags = bags_from_fold_cell_dataset(dev_ds, {}, min_cells_per_subject=1)
            held_out_bags = bags_from_fold_cell_dataset(held_out_ds, {}, min_cells_per_subject=1)
            if not dev_bags or not held_out_bags:
                reports[held_out_source] = build_robustness_report(
                    task="smoke_classification", model=best_name, strategy="erm", held_out_source=held_out_source,
                    eligibility=elig.to_dict(), development_sources=dev_sources,
                    limitations=["no subject met min_cells_per_subject for smoke bags"],
                    comparisons=[{"name": n, **s} for n, s in candidate_scores.items()],
                    dataset_manifest_fingerprint=dataset_manifest_fp, source_policy_fingerprint=source_policy_fp,
                    environment_fingerprint=environment_fp, seed=seed,
                ).to_dict()
                continue
            dev_sd = SubjectLevelDataset(dev_bags, require_known_outcome=False)
            held_out_sd = SubjectLevelDataset(held_out_bags, require_known_outcome=False)
            fold_ctx = dataclasses.replace(context, preprocessing_artifact=final_artifact)
            adapter = build_mil_adapter(PATHWAY_MODEL_NAME, None, device, config_overrides=selected_params)
            adapter.fit_final(fold_ctx, dev_ds, dev_sd, seed=seed)
            preds = adapter.predict_smoke(held_out_sd)
            labels, known = adapter.known_smoke_labels(held_out_sd)
            if not known.any():
                reports[held_out_source] = build_robustness_report(
                    task="smoke_classification", model=best_name, strategy="erm", held_out_source=held_out_source,
                    eligibility=elig.to_dict(), development_sources=dev_sources,
                    limitations=["no held-out subject had a verified smoke label"],
                    comparisons=[{"name": n, **s} for n, s in candidate_scores.items()],
                    dataset_manifest_fingerprint=dataset_manifest_fp, source_policy_fingerprint=source_policy_fp,
                    environment_fingerprint=environment_fp, seed=seed,
                ).to_dict()
                continue
            report = full_smoke_metrics_report(labels[known], preds[known], num_classes)
            model_state_fp = adapter.model_state_fingerprint()
            held_out_proba_full = adapter.predict_smoke_proba(held_out_sd)
            held_out_proba = held_out_proba_full[known]
            held_out_labels_for_uncertainty = labels[known]

        module_info = None
        if best_name == PATHWAY_MODEL_NAME:
            module_info = adapter.modules
        domain_shift = smoke_domain_shift_report(
            dev_ds, held_out_ds, num_cell_types, num_classes, subject_to_source, seed=seed,
            required_gene_list=list(final_artifact.gene_list), modules=module_info,
        )
        gene_list_fp = _sha256_json(list(final_artifact.gene_list))
        if best_name == PATHWAY_MODEL_NAME:
            module_fp = adapter.modules.fingerprint()
        else:
            module_fp = not_applicable(f"{best_name} has no gene-module structure")

        # Development-only OOF probabilities from the SAME dev-only nested
        # grouped-CV sweep that selected best_name (best_oof_by_subject,
        # captured above) — never the final dev-pool-fitted model's
        # in-sample predictions. A subject with no OOF entry (every fold
        # containing it was skipped) is simply absent from this array, not
        # imputed.
        oof_dev_subjects = sorted(best_oof_by_subject)
        if oof_dev_subjects:
            dev_proba_oof = np.stack([best_oof_by_subject[s] for s in oof_dev_subjects])
            dev_labels_oof = np.array([dev_label_by_subject[s] for s in oof_dev_subjects])
        else:
            dev_proba_oof, dev_labels_oof = None, None

        smoke_uncertainty = smoke_uncertainty_report(
            dev_proba_oof, dev_labels_oof, held_out_proba, held_out_labels_for_uncertainty, num_classes,
        )

        manifest = build_source_held_out_manifest(
            task="smoke_classification", held_out_source=held_out_source, development_sources=dev_sources,
            development_subjects=dev_subjects, held_out_subjects=held_out_subjects,
            known_label_counts={"n": len(held_out_labels)}, class_distribution=elig.counts,
            eligibility=elig, seed=seed, preprocessing_policy_fingerprint=artifact_fingerprint(final_artifact),
            dataset_manifest_fingerprint=dataset_manifest_fp,
        )
        reports[held_out_source] = build_robustness_report(
            task="smoke_classification", model=best_name, strategy="erm",
            held_out_source=held_out_source, eligibility=elig.to_dict(), development_sources=dev_sources,
            metrics={"macro_f1": report["macro_f1"], "weighted_f1": report["weighted_f1"],
                     "balanced_accuracy": report.get("balanced_accuracy"), "per_class": report.get("per_class"),
                     "classes_absent_from_held_out_source": report["classes_absent_from_targets"]},
            domain_shift=domain_shift, uncertainty=smoke_uncertainty,
            comparisons=[{"name": n, **s} for n, s in candidate_scores.items()],
            source_split_manifest_fingerprint=manifest.fingerprint(),
            preprocessing_fingerprint=artifact_fingerprint(final_artifact),
            module_fingerprint=module_fp,
            model_fingerprint=model_state_fp,
            label_state={"development": dev_label_diagnostics, "held_out": held_out_label_diagnostics,
                         "weak_labels_enabled": weak_labels_enabled},
            dataset_manifest_fingerprint=dataset_manifest_fp, source_policy_fingerprint=source_policy_fp,
            environment_fingerprint=environment_fp, gene_list_fingerprint=gene_list_fp, seed=seed,
            evaluated=True, is_module_based_candidate=(best_name == PATHWAY_MODEL_NAME),
        ).to_dict()

    return reports
