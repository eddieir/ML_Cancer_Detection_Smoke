"""
evidence/tracks.py — the three Phase 7 evaluation tracks (Step 5).

Each track has two entry points:

  run_track_<x>_against_registry(cohorts, ...)
      The real path. Consults the cohort registry (evidence/cohort_registry.py)
      for a cohort that genuinely supports the track's task at the requested
      role. Never opens a dataset file, never fabricates a subject/class
      count. In this environment configs/cohorts.yaml has no cohort with
      task_support[...]='yes' for any of the three tasks (every real
      cohort is either structurally incompatible or 'not_currently'
      supported pending a pipeline that does not exist yet), so this path
      always returns a structured not_evaluable() object here — that is the
      correct, honest answer, not a stub.

  run_track_<x>_on_fixture(...)
      The synthetic/development proof path. Takes already-assembled
      subject-level arrays (as the existing benchmark fixtures already used
      throughout tests/ provide) and runs the real metric-computation code
      from benchmarks/metrics.py, returning a fully-identified EvidenceReport
      with synthetic_flag=True / evidence_level='synthetic_software_validation'.
      This proves the scoring code path actually executes; it never claims
      real-world evidence.
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmarks.metrics import (
    cancer_prediction_metrics,
    expected_calibration_error,
    reliability_curve,
    subject_weighted_full_smoke_metrics_report,
)

from .cohort_registry import Cohort
from .evidence_contract import build_report, not_evaluable

TASK_SMOKE = "smoke_classification"
TASK_MALIGNANCY = "malignancy_classification"
TASK_CANCER_PREDICTION = "subject_level_cancer_prediction"

MIN_SUBJECT_SUPPORT_FOR_LEARNABILITY_CLAIM = 5


def _git_commit_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=5, check=True,
        )
        sha = out.stdout.strip()
        if sha:
            return sha
    except Exception:
        pass
    return "0" * 40


def _fp(*parts: object) -> str:
    import hashlib
    import json
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def eligible_cohorts_for_track(cohorts: Sequence[Cohort], task: str, role: str = "development") -> List[Cohort]:
    """Real-data eligibility check — never opens a dataset, only consults
    the already-loaded, already-validated cohort registry."""
    return [c for c in cohorts if c.supports(task) and c.eligible_for_role(role)]


def _no_eligible_cohort(task: str, role: str, extra_reason: str) -> dict:
    return not_evaluable(
        reason_code="NO_ELIGIBLE_COHORT",
        reason=(
            f"No cohort in configs/cohorts.yaml has task_support[{task!r}]='yes' with "
            f"role_eligibility including {role!r}. {extra_reason}"
        ),
        required_next_action=(
            "Add or update a cohort_registry entry once a genuinely supported dataset "
            "adapter/pipeline and verified labels exist for this task/cohort combination, "
            "then re-run this track against the registry."
        ),
        task=task, role=role,
    )


# ─── Track A — verified smoke classification ──────────────────────────────

def run_track_a_against_registry(cohorts: Sequence[Cohort], role: str = "development") -> dict:
    eligible = eligible_cohorts_for_track(cohorts, TASK_SMOKE, role)
    if not eligible:
        return _no_eligible_cohort(
            TASK_SMOKE, role,
            "Verified per-subject smoke-exposure labels on single-cell-compatible input "
            "are not currently supported by any registered cohort in this environment "
            "(gse136831 has only a weak Disease_Identity proxy; gse288003 is mouse-only "
            "and role_eligibility=excluded; gse123352 is bulk and excluded).",
        )
    return not_evaluable(
        reason_code="REAL_FITTING_NOT_IMPLEMENTED",
        reason=(
            f"cohort(s) {[c.cohort_id for c in eligible]} are marked eligible for "
            f"{TASK_SMOKE!r}/{role!r}, but no local raw data was verified present in this "
            "environment (see evidence.audit) and no fitting was attempted."
        ),
        required_next_action="Run evidence.audit to confirm local data presence, then invoke evidence.runner development.",
        task=TASK_SMOKE, role=role, candidate_cohorts=[c.cohort_id for c in eligible],
    )


def run_track_a_on_fixture(
    y_true: Sequence[int], y_pred: Sequence[int], subject_ids: Sequence[str], num_classes: int,
    *, class_names: Optional[List[str]] = None, weak_label_experiment: bool = False,
    random_seed: int = 0,
) -> dict:
    """Runs the real subject-level (one-vote-per-subject) scoring path
    against an already-assembled synthetic/development fixture. Never mixes
    weak-proxy labels into this primary result — `weak_label_experiment`
    must be explicitly True and is stamped into the report identity/
    limitations so it can never be confused with a verified-label result."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    subject_ids = [str(s) for s in subject_ids]
    n_subjects = len(set(subject_ids))

    report = subject_weighted_full_smoke_metrics_report(y_true, y_pred, subject_ids, num_classes)

    class_counts: Dict[str, int] = {}
    for cls, info in report.get("per_class", {}).items():
        name = class_names[int(cls)] if class_names else cls
        class_counts[name] = info["support"]

    below_support_threshold = [
        (class_names[int(c)] if class_names else c)
        for c, info in report.get("per_class", {}).items()
        if info["support"] < MIN_SUBJECT_SUPPORT_FOR_LEARNABILITY_CLAIM
    ]

    limitations = [
        "Synthetic/development fixture only — no real-world evidence claimed.",
        "One vote per subject via cell-level majority vote; abstention/coverage is secondary only.",
    ]
    if weak_label_experiment:
        limitations.append(
            "weak_label_experiment=True — this result uses proxy/weak exposure labels and "
            "must never be merged with a verified-label primary result."
        )
    if below_support_threshold:
        limitations.append(
            f"classes {below_support_threshold} are below the subject-support threshold "
            f"({MIN_SUBJECT_SUPPORT_FOR_LEARNABILITY_CLAIM}) — no strong learnability claim is made for them."
        )

    identity = {
        "task": TASK_SMOKE,
        "endpoint": "subject_level_smoke_class" if not weak_label_experiment else "subject_level_smoke_class_weak_label_sensitivity",
        "endpoint_definition": "verified cigarette/vape/never exposure at time of sampling, one label per subject",
        "prediction_unit": "subject",
        "biological_specimen_type": "lung_tissue",
        "assay_modality": "single_cell_rna_seq",
        "dataset_accession": "SYNTHETIC_FIXTURE",
        "cohort_role": "development",
        "species": "human",
        "sample_count": len(y_true),
        "unique_subject_count": n_subjects,
        "class_counts": class_counts,
        "verified_label_count": 0 if weak_label_experiment else len(y_true),
        "unknown_label_count": 0,
        "excluded_subject_count": 0,
        "split_role": "train",
        "split_manifest_fingerprint": _fp("track_a_split", subject_ids),
        "dataset_manifest_fingerprint": _fp("track_a_fixture", y_true),
        "preprocessing_artifact_fingerprint": _fp("track_a_preprocessing"),
        "model_fingerprint": _fp("track_a_model", y_pred),
        "random_seed": random_seed,
        "code_commit_sha": _git_commit_sha(),
        "environment_fingerprint": _fp("track_a_environment"),
        "synthetic_flag": True,
        "development_only_flag": True,
        "frozen_test_access_status": "not_applicable",
        "evidence_level": "synthetic_software_validation",
        "limitations": limitations,
        "evaluation_timestamp": "1970-01-01T00:00:00Z",
        "report_schema_version": "1",
    }
    metrics = {
        "primary_metric": "macro_f1",
        "macro_f1": report["macro_f1"],
        "weighted_f1": report["weighted_f1"],
        "balanced_accuracy": report["balanced_accuracy"],
        "per_class": report["per_class"],
        "confusion_matrix": report["confusion_matrix"],
        "accuracy": (
            float(np.mean(np.asarray(report["confusion_matrix"]).diagonal().sum() /
                           max(1, np.asarray(report["confusion_matrix"]).sum())))
            if report["confusion_matrix"] else None
        ),
        "effective_label_mapping": (
            {str(i): class_names[i] for i in range(num_classes)} if class_names
            else {str(i): i for i in range(num_classes)}
        ),
        "weak_label_experiment": weak_label_experiment,
        "classes_below_support_threshold": below_support_threshold,
    }
    return build_report(identity, metrics).to_dict()


# ─── Track B — cell/sample malignancy classification ──────────────────────

def run_track_b_against_registry(cohorts: Sequence[Cohort], role: str = "development") -> dict:
    eligible = [
        c for c in cohorts
        if c.supports(TASK_MALIGNANCY) and c.eligible_for_role(role) and c.single_cell_or_bulk == "single_cell"
    ]
    if not eligible:
        return _no_eligible_cohort(
            TASK_MALIGNANCY, role,
            "No registered single-cell cohort carries a genuine per-cell/per-sample "
            "malignancy label at a biological unit compatible with this model — TCGA-LUAD/"
            "LUSC carry only bulk sample_type labels, which must never be translated into "
            "per-cell malignancy labels (see cohort_registry source_limitations); the "
            "single-cell cohorts (gse136831, gse288003) carry no malignancy_label_fields at all.",
        )
    return not_evaluable(
        reason_code="REAL_FITTING_NOT_IMPLEMENTED",
        reason=f"cohort(s) {[c.cohort_id for c in eligible]} are structurally eligible but no local data was verified present.",
        required_next_action="Run evidence.audit, then evidence.runner development.",
        task=TASK_MALIGNANCY, role=role, candidate_cohorts=[c.cohort_id for c in eligible],
    )


def run_track_b_on_fixture(
    y_true: Sequence[int], y_prob: Sequence[float], sample_ids: Sequence[str],
    *, threshold: float = 0.5, threshold_method: str = "fixed_default", random_seed: int = 0,
) -> dict:
    y_true = list(y_true)
    y_prob = list(y_prob)
    metrics = cancer_prediction_metrics(y_true, y_prob, threshold=threshold)
    curve = reliability_curve(np.asarray(y_true, dtype=float), np.asarray(y_prob, dtype=float))

    identity = {
        "task": TASK_MALIGNANCY,
        "endpoint": "sample_malignancy_status",
        "endpoint_definition": "genuine tumour/malignant vs normal label at a biological unit compatible with the model",
        "prediction_unit": "sample",
        "biological_specimen_type": "lung_tissue",
        "assay_modality": "single_cell_rna_seq",
        "dataset_accession": "SYNTHETIC_FIXTURE",
        "cohort_role": "development",
        "species": "human",
        "sample_count": len(y_true),
        "unique_subject_count": len(set(sample_ids)),
        "class_counts": {"malignant": int(sum(1 for v in y_true if v == 1)),
                          "normal": int(sum(1 for v in y_true if v == 0))},
        "verified_label_count": len(y_true),
        "unknown_label_count": 0,
        "excluded_subject_count": 0,
        "split_role": "train",
        "split_manifest_fingerprint": _fp("track_b_split", sample_ids),
        "dataset_manifest_fingerprint": _fp("track_b_fixture", y_true),
        "preprocessing_artifact_fingerprint": _fp("track_b_preprocessing"),
        "model_fingerprint": _fp("track_b_model", y_prob),
        "random_seed": random_seed,
        "code_commit_sha": _git_commit_sha(),
        "environment_fingerprint": _fp("track_b_environment"),
        "synthetic_flag": True,
        "development_only_flag": True,
        "frozen_test_access_status": "not_applicable",
        "evidence_level": "synthetic_software_validation",
        "limitations": [
            "Synthetic/development fixture only — no real-world evidence claimed.",
            "No registered cohort currently supports this task on real data; see run_track_b_against_registry.",
        ],
        "evaluation_timestamp": "1970-01-01T00:00:00Z",
        "report_schema_version": "1",
        "threshold_fingerprint": _fp("track_b_threshold", threshold, threshold_method),
    }
    out_metrics = dict(metrics)
    out_metrics["calibration_curve"] = curve
    out_metrics["threshold_method"] = threshold_method
    return build_report(identity, out_metrics).to_dict()


# ─── Track C — subject-level cancer prediction ────────────────────────────

def run_track_c_against_registry(cohorts: Sequence[Cohort], role: str = "development") -> dict:
    eligible = [
        c for c in cohorts
        if c.supports(TASK_CANCER_PREDICTION) and c.eligible_for_role(role)
        and c.expression_outcome_linkable_at_subject_level
    ]
    if not eligible:
        return not_evaluable(
            reason_code="NO_ELIGIBLE_EXPRESSION_OUTCOME_LINKAGE",
            reason=(
                "No registered cohort has both compatible expression input and a genuinely "
                "linked, verified subject-level cancer outcome. NLST carries verified outcome "
                "fields (candx) but is controlled-access, has no expression data at all, and "
                "expression_outcome_linkable_at_subject_level=False; TCGA-LUAD/LUSC carry bulk "
                "outcome-adjacent fields but no adapter in this repository links them to the "
                "single-cell MIL model's expression input."
            ),
            required_next_action=(
                "Obtain authorized access to a cohort with genuine subject-level expression<->"
                "outcome linkage (e.g. an approved NLST DUA plus a scientifically validated "
                "cross-assay linkage adapter, neither of which exists in this repository "
                "today), register it in configs/cohorts.yaml with "
                "expression_outcome_linkable_at_subject_level=True, then re-run."
            ),
            task=TASK_CANCER_PREDICTION, role=role,
        )
    return not_evaluable(
        reason_code="REAL_FITTING_NOT_IMPLEMENTED",
        reason=f"cohort(s) {[c.cohort_id for c in eligible]} are structurally eligible but no local data was verified present.",
        required_next_action="Run evidence.audit, then evidence.runner development.",
        task=TASK_CANCER_PREDICTION, role=role, candidate_cohorts=[c.cohort_id for c in eligible],
    )


def run_track_c_on_fixture(
    y_true: Sequence[int], y_prob: Sequence[float], subject_ids: Sequence[str],
    *, event_indicator_field: str = "cancer_diagnosis", time_origin: str = "enrollment",
    prediction_horizon_years: float = 5.0, follow_up_complete: bool = True,
    threshold: float = 0.5, threshold_method: str = "fixed_default", random_seed: int = 0,
) -> dict:
    """Binary fixed-horizon subject-level cancer prediction only — this
    fixture never fabricates time-to-event/censoring semantics, so no
    survival metric (C-index, time-dependent AUROC, integrated Brier score)
    is computed here; that requires genuinely valid follow-up/censoring
    data, which no available fixture or cohort attests to."""
    y_true = list(y_true)
    y_prob = list(y_prob)
    metrics = cancer_prediction_metrics(y_true, y_prob, threshold=threshold)
    curve = reliability_curve(np.asarray(y_true, dtype=float), np.asarray(y_prob, dtype=float))

    identity = {
        "task": TASK_CANCER_PREDICTION,
        "endpoint": "subject_level_cancer_outcome_fixed_horizon",
        "endpoint_definition": (
            f"binary cancer diagnosis within {prediction_horizon_years} years of {time_origin}; "
            f"event_indicator_field={event_indicator_field}"
        ),
        "prediction_unit": "subject",
        "biological_specimen_type": "lung_tissue",
        "assay_modality": "single_cell_rna_seq",
        "dataset_accession": "SYNTHETIC_FIXTURE",
        "cohort_role": "development",
        "species": "human",
        "sample_count": len(y_true),
        "unique_subject_count": len(set(subject_ids)),
        "class_counts": {"event": int(sum(1 for v in y_true if v == 1)),
                          "no_event": int(sum(1 for v in y_true if v == 0))},
        "verified_label_count": len(y_true),
        "unknown_label_count": 0,
        "excluded_subject_count": 0,
        "split_role": "train",
        "split_manifest_fingerprint": _fp("track_c_split", subject_ids),
        "dataset_manifest_fingerprint": _fp("track_c_fixture", y_true),
        "preprocessing_artifact_fingerprint": _fp("track_c_preprocessing"),
        "model_fingerprint": _fp("track_c_model", y_prob),
        "random_seed": random_seed,
        "code_commit_sha": _git_commit_sha(),
        "environment_fingerprint": _fp("track_c_environment"),
        "synthetic_flag": True,
        "development_only_flag": True,
        "frozen_test_access_status": "not_applicable",
        "evidence_level": "synthetic_software_validation",
        "limitations": [
            "Synthetic/development fixture only — no real-world evidence or outcome linkage claimed.",
            "Binary fixed-horizon only; no time-to-event/survival metric computed against this fixture.",
            "No registered cohort currently supports this task on real data; see run_track_c_against_registry.",
        ],
        "evaluation_timestamp": "1970-01-01T00:00:00Z",
        "report_schema_version": "1",
        "threshold_fingerprint": _fp("track_c_threshold", threshold, threshold_method),
    }
    out_metrics = dict(metrics)
    out_metrics["calibration_curve"] = curve
    out_metrics["threshold_method"] = threshold_method
    out_metrics["endpoint_fields"] = {
        "event_indicator_field": event_indicator_field,
        "time_origin": time_origin,
        "prediction_horizon_years": prediction_horizon_years,
        "follow_up_complete": follow_up_complete,
        "censoring_status": "not_applicable_binary_fixed_horizon",
    }
    return build_report(identity, out_metrics).to_dict()
