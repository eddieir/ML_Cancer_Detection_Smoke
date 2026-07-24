"""
evidence/development.py — Issue #16 blocker 4: canonical orchestration for
`evidence.runner development` / `internal-test` / `external-test`.

`run_development()` is the one real implementation of the "development"
subcommand — dataset-specific scripts (scripts/run_gse123352_verified_label_evidence.py)
are thin wrappers that call this same function, so there is exactly one
place that decides how a cohort/task combination is (or is not) actually
evaluated.

`run_internal_test()` / `run_external_test()` are honest gates, not stubs
that always say "not yet integrated": each names the SPECIFIC reason a
frozen internal-test or external-validation result cannot be produced right
now (no frozen-test partition has ever been created/guarded for any
registered cohort; no cohort in configs/cohorts.yaml carries
role_eligibility=[external_validation]) — never a generic "the pipeline
isn't wired up" placeholder, and never a path that fabricates a result.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from .evidence_contract import not_evaluable

ROOT = Path(__file__).resolve().parents[2]

GSE123352_RAW_DIR = ROOT / "data" / "raw" / "cigarette" / "GSE123352"
GSE123352_RAW_FILE_NAMES = (
    "GSE123352_series_matrix.txt.gz",
    "GSE123352_non-normalized.txt.gz",
    "GPL10558.annot.gz",
)


def _no_eligible_cohort(cohort_id: str, task: str, role: str) -> Dict[str, Any]:
    return not_evaluable(
        reason_code="NO_ELIGIBLE_COHORT",
        reason=f"cohort {cohort_id!r} is not registered with task_support[{task!r}]='yes' and role_eligibility including {role!r}.",
        required_next_action="Register (or correct) a configs/cohorts.yaml entry for this cohort/task/role combination.",
        cohort_id=cohort_id, task=task, role=role,
    )


def run_development(
    cohort_id: str, task: str, *,
    cohorts: list,
    output_root: Union[str, Path],
    run_id: Optional[str] = None,
    seed: int = 42, train_frac: float = 0.70, test_frac: float = 0.30,
    n_top_variance_genes: int = 2000,
) -> Dict[str, Any]:
    """The one real 'development' orchestration path. `cohorts` is an
    already-loaded evidence.cohort_registry.Cohort list (see
    evidence.cohort_registry.load_cohort_registry). Never accesses
    internal-test or external data — the only data this function ever opens
    is the development-eligible cohort's own raw files, read exactly once."""
    from .cohort_registry import find_cohort, CohortRegistryError
    from .evidence_contract import is_not_evaluable

    try:
        cohort = find_cohort(cohorts, cohort_id)
    except CohortRegistryError:
        return _no_eligible_cohort(cohort_id, task, "development")
    if not cohort.supports(task) or not cohort.eligible_for_role("development"):
        return _no_eligible_cohort(cohort_id, task, "development")

    if cohort_id == "gse123352" and task == "smoke_classification":
        return _run_gse123352_development(
            output_root=output_root, run_id=run_id or "gse123352_verified_label_smoke_v1",
            seed=seed, train_frac=train_frac, test_frac=test_frac,
            n_top_variance_genes=n_top_variance_genes,
        )

    if cohort_id == "tcga_lung_vital_status" and task == "subject_level_cancer_prediction":
        return _run_tcga_lung_vital_status_development(
            output_root=output_root, run_id=run_id or "tcga_lung_vital_status_v1",
            seed=seed, train_frac=train_frac, test_frac=test_frac,
            n_top_variance_genes=n_top_variance_genes,
        )

    return not_evaluable(
        reason_code="TRACK_RUNNER_NOT_YET_INTEGRATED",
        reason=(
            f"cohort {cohort_id!r} is structurally eligible for {task!r}/'development', but no "
            "canonical development orchestration exists for this specific cohort/task "
            "combination in this repository yet (only gse123352/smoke_classification is wired up)."
        ),
        required_next_action=(
            "Add a dedicated branch to evidence.development.run_development for this "
            "cohort/task combination once a genuine dataset-specific pipeline exists."
        ),
        cohort_id=cohort_id, task=task,
    )


def _run_gse123352_development(*, output_root, run_id, seed, train_frac, test_frac, n_top_variance_genes) -> Dict[str, Any]:
    raw_files = [GSE123352_RAW_DIR / name for name in GSE123352_RAW_FILE_NAMES]
    missing = [str(p) for p in raw_files if not p.exists()]
    if missing:
        return not_evaluable(
            reason_code="REAL_DATA_NOT_PRESENT",
            reason=f"missing real raw GSE123352 file(s) under {GSE123352_RAW_DIR}: {missing}",
            required_next_action="Download the real GSE123352 raw files to this path before running development.",
            cohort_id="gse123352", task="smoke_classification",
        )

    from data.converters import convert_accession
    from data import bulk_pipeline
    from . import tracks
    from .artifact_bundle import write_evidence_run
    from .run_identity import (
        build_environment_snapshot,
        build_split_manifest,
        bulk_model_fingerprint,
        bulk_preprocessing_fingerprint,
        environment_snapshot_fingerprint,
        real_git_commit_sha,
        split_manifest_fingerprint as compute_split_manifest_fingerprint,
    )

    csv_path = convert_accession("GSE123352")
    if csv_path is None:
        return not_evaluable(
            reason_code="CONVERSION_FAILED",
            reason="convert_accession('GSE123352') returned no path.",
            required_next_action="Investigate data/converters.py::convert_accession for GSE123352.",
            cohort_id="gse123352", task="smoke_classification",
        )

    result = bulk_pipeline.run_bulk_smoke_classification(
        csv_path, seed=seed, train_frac=train_frac, test_frac=test_frac,
        n_top_variance_genes=n_top_variance_genes,
    )
    dataset = result["dataset"]
    test_true, test_pred = result["test"]["y_true"], result["test"]["y_pred"]
    test_subjects, train_subjects = result["test"]["subject_ids"], result["train"]["subject_ids"]
    fitted = result["fitted_classifier"]

    preprocessing_fp = bulk_preprocessing_fingerprint(
        gene_indices=fitted.gene_indices, gene_names_selected=fitted.gene_names_selected,
        n_genes_available=fitted.n_genes_available, n_top_variance_genes_requested=n_top_variance_genes,
        selection_policy="top_variance_train_only", scaler=fitted.scaler,
        train_subject_ids=train_subjects,
        missing_value_handling="none — converter drops probes with no annotated gene symbol upstream",
    )
    model_fp = bulk_model_fingerprint(
        model=fitted.model,
        hyperparameters={"C": float(fitted.model.C), "class_weight": "balanced", "random_state": seed},
        class_names=list(bulk_pipeline.SUPPORTED_BULK_CLASSES),
        preprocessing_fingerprint=preprocessing_fp, train_subject_ids=train_subjects,
    )
    commit_sha = real_git_commit_sha(ROOT)
    env_snapshot = build_environment_snapshot(commit_sha, ROOT)
    split_manifest = build_split_manifest(
        task=tracks.TASK_SMOKE, endpoint="subject_level_smoke_class_verified_bulk",
        dataset_accession="GSE123352", train_subject_ids=train_subjects, validation_subject_ids=[],
        development_holdout_subject_ids=test_subjects, seed=seed, train_frac=train_frac,
        val_frac=0.0, test_frac=test_frac,
        stratification_policy="subject_train_val_test_split (label-stratified subject-level split)",
        class_mapping={name: i for i, name in enumerate(bulk_pipeline.SUPPORTED_BULK_CLASSES)},
        rare_class_policy="not_applicable_binary_verified_cohort",
        label_state_policy="verified_only — smoke_type_known=True required, strict boolean parsing",
        weak_label_policy="not_applicable — GSE123352 carries a verified ever_never_smoker label",
        subject_identity_field="subject_id (verified via GEO Sample_title patient_<N> parsing)",
        grouping_policy="one verified subject per row; duplicate subject_id rejected upstream",
        dataset_manifest_fingerprint=tracks._real_raw_file_fingerprint(raw_files),
    )

    report = tracks.run_track_a_on_real_verified_label_data(
        y_true=test_true, y_pred=test_pred, subject_ids=test_subjects, num_classes=2,
        raw_file_paths=raw_files,
        split_manifest_fingerprint=compute_split_manifest_fingerprint(split_manifest),
        model_fingerprint=model_fp, environment_fingerprint=environment_snapshot_fingerprint(env_snapshot),
        preprocessing_artifact_fingerprint=preprocessing_fp,
        class_names=list(bulk_pipeline.SUPPORTED_BULK_CLASSES), dataset_accession="GSE123352",
        excluded_subject_count=len(dataset.excluded_sample_ids),
        excluded_subject_reason=(f"excluded_sample_ids: {dataset.excluded_reason_counts}" if dataset.excluded_sample_ids else None),
        random_seed=seed,
    )

    files = {
        "configuration.json": {
            "accession": "GSE123352", "seed": seed, "train_frac": train_frac,
            "test_frac": test_frac, "n_top_variance_genes": n_top_variance_genes,
        },
        "predictions/test_predictions.csv": [
            {"subject_id": sid, "y_true": yt, "y_pred": yp}
            for sid, yt, yp in zip(test_subjects, test_true, test_pred)
        ],
        "metrics/track_a_real_verified_label.json": report,
        "cohort_flow.json": {
            "total_samples_parsed": dataset.X.shape[0] + len(dataset.excluded_sample_ids),
            "verified_label_samples_kept": dataset.X.shape[0],
            "excluded_sample_ids": dataset.excluded_sample_ids,
            "excluded_reason_counts": dataset.excluded_reason_counts,
            "train_subjects": train_subjects, "test_subjects": test_subjects,
        },
        "environment.json": env_snapshot,
        "split_manifest.json": split_manifest,
    }
    run_dir = write_evidence_run(
        output_root, run_id, files,
        extra_manifest_fields={"dataset_accession": "GSE123352", "track": "A", "task": tracks.TASK_SMOKE},
    )
    return {"status": "complete", "run_dir": str(run_dir), "report": report}


TCGA_LUAD_CSV = ROOT / "data" / "processed" / "converted" / "TCGA-LUAD_vital_status.csv"
TCGA_LUSC_CSV = ROOT / "data" / "processed" / "converted" / "TCGA-LUSC_vital_status.csv"
TCGA_LUAD_RAW_DIR = ROOT / "data" / "raw" / "malignancy" / "TCGA-LUAD"
TCGA_LUSC_RAW_DIR = ROOT / "data" / "raw" / "malignancy" / "TCGA-LUSC"


def _tcga_lung_vital_status_dataset():
    """Builds the real, converted TCGA-LUAD+TCGA-LUSC vital-status dataset,
    converting from real downloaded GDC files first if the converted CSVs
    are not already present. Returns None (never fabricates a dataset) if
    the real raw GDC files are not present locally."""
    from data import tcga_outcome_pipeline
    from data.converters import convert_tcga_vital_status

    if not TCGA_LUAD_CSV.exists():
        if not (TCGA_LUAD_RAW_DIR / "outcome_meta.csv").exists():
            return None, "TCGA-LUAD"
        if convert_tcga_vital_status("TCGA-LUAD", TCGA_LUAD_RAW_DIR) is None:
            return None, "TCGA-LUAD"
    if not TCGA_LUSC_CSV.exists():
        if not (TCGA_LUSC_RAW_DIR / "outcome_meta.csv").exists():
            return None, "TCGA-LUSC"
        if convert_tcga_vital_status("TCGA-LUSC", TCGA_LUSC_RAW_DIR) is None:
            return None, "TCGA-LUSC"

    dataset = tcga_outcome_pipeline.build_tcga_outcome_dataset([TCGA_LUAD_CSV, TCGA_LUSC_CSV])
    return dataset, None


def _run_tcga_lung_vital_status_development(*, output_root, run_id, seed, train_frac, test_frac, n_top_variance_genes) -> Dict[str, Any]:
    dataset, missing = _tcga_lung_vital_status_dataset()
    if dataset is None:
        return not_evaluable(
            reason_code="REAL_DATA_NOT_PRESENT",
            reason=f"missing real downloaded GDC files for {missing} under data/raw/malignancy/{missing}/ "
                   "(run the GDC open-access download for STAR-Counts gene expression + case vital_status first).",
            required_next_action="Download the real TCGA-LUAD/TCGA-LUSC open-access GDC files before running development.",
            cohort_id="tcga_lung_vital_status", task="subject_level_cancer_prediction",
        )

    from data import bulk_pipeline, tcga_outcome_pipeline
    from . import tracks
    from .artifact_bundle import write_evidence_run
    from .run_identity import (
        build_environment_snapshot,
        build_split_manifest,
        bulk_model_fingerprint,
        bulk_preprocessing_fingerprint,
        environment_snapshot_fingerprint,
        real_git_commit_sha,
        split_manifest_fingerprint as compute_split_manifest_fingerprint,
    )
    import hashlib

    split = tcga_outcome_pipeline.split_tcga_outcome_subjects(
        dataset, seed=seed, train_frac=train_frac, test_frac=test_frac,
    )
    id_to_row = {sid: i for i, sid in enumerate(dataset.subject_ids)}
    train_idx = [id_to_row[s] for s in split.train_subjects]
    test_idx = [id_to_row[s] for s in split.test_subjects]
    _assert_no_subject_overlap({"train": split.train_subjects, "test": split.test_subjects})

    fitted = bulk_pipeline.fit_bulk_logistic_regression(
        dataset.X[train_idx], dataset.y[train_idx], dataset.gene_names,
        seed=seed, n_top_variance_genes=n_top_variance_genes, C=1.0,
    )
    y_pred, y_prob = bulk_pipeline.apply_bulk_classifier(fitted, dataset.X[test_idx])
    test_subjects = [dataset.subject_ids[i] for i in test_idx]
    train_subjects = [dataset.subject_ids[i] for i in train_idx]
    test_true = dataset.y[test_idx].tolist()

    preprocessing_fp = bulk_preprocessing_fingerprint(
        gene_indices=fitted.gene_indices, gene_names_selected=fitted.gene_names_selected,
        n_genes_available=fitted.n_genes_available, n_top_variance_genes_requested=n_top_variance_genes,
        selection_policy="top_variance_train_only", scaler=fitted.scaler,
        train_subject_ids=train_subjects, missing_value_handling="none — missing genes filled 0 at conversion",
    )
    model_fp = bulk_model_fingerprint(
        model=fitted.model, hyperparameters={"C": 1.0, "class_weight": "balanced", "random_state": seed},
        class_names=list(tcga_outcome_pipeline.SUPPORTED_VITAL_CLASSES),
        preprocessing_fingerprint=preprocessing_fp, train_subject_ids=train_subjects,
    )
    commit_sha = real_git_commit_sha(ROOT)
    env_snapshot = build_environment_snapshot(commit_sha, ROOT)
    split_manifest = build_split_manifest(
        task=tracks.TASK_CANCER_PREDICTION, endpoint="subject_level_vital_status_at_last_follow_up",
        dataset_accession="TCGA-LUAD+TCGA-LUSC", train_subject_ids=train_subjects, validation_subject_ids=[],
        development_holdout_subject_ids=test_subjects, seed=seed, train_frac=train_frac,
        val_frac=0.0, test_frac=test_frac,
        stratification_policy="subject_train_val_test_split (label-stratified subject-level split)",
        class_mapping={name: i for i, name in enumerate(tcga_outcome_pipeline.SUPPORTED_VITAL_CLASSES)},
        rare_class_policy="not_applicable_binary_cohort",
        label_state_policy="verified_only — vital_status_known=True required from real GDC demographic record",
        weak_label_policy="not_applicable — vital_status is a real, independently-recorded GDC outcome field",
        subject_identity_field="subject_id (real GDC case_id — authoritative, not inferred)",
        grouping_policy="one verified subject (case_id) per row; duplicate case_id rejected upstream",
        dataset_manifest_fingerprint=hashlib.sha256(
            (TCGA_LUAD_CSV.read_bytes() + TCGA_LUSC_CSV.read_bytes())
        ).hexdigest(),
    )

    report = tracks.run_track_c_on_real_tcga_vital_status_data(
        y_true=test_true, y_prob=y_prob.tolist(), subject_ids=test_subjects,
        dataset_manifest_fingerprint=split_manifest["dataset_manifest_fingerprint"],
        split_manifest_fingerprint=compute_split_manifest_fingerprint(split_manifest),
        model_fingerprint=model_fp, environment_fingerprint=environment_snapshot_fingerprint(env_snapshot),
        preprocessing_artifact_fingerprint=preprocessing_fp,
        excluded_subject_count=len(dataset.excluded_sample_ids),
        excluded_subject_reason=(f"excluded_sample_ids: {dataset.excluded_reason_counts}" if dataset.excluded_sample_ids else None),
        random_seed=seed,
    )

    files = {
        "configuration.json": {
            "accession": "TCGA-LUAD+TCGA-LUSC", "seed": seed, "train_frac": train_frac, "test_frac": test_frac,
            "n_top_variance_genes": n_top_variance_genes,
        },
        "predictions/test_predictions.csv": [
            {"subject_id": sid, "y_true": yt, "y_prob": float(yp)}
            for sid, yt, yp in zip(test_subjects, test_true, y_prob.tolist())
        ],
        "metrics/track_c_real_tcga_vital_status.json": report,
        "cohort_flow.json": {
            "total_samples_parsed": dataset.X.shape[0] + len(dataset.excluded_sample_ids),
            "verified_samples_kept": dataset.X.shape[0],
            "excluded_sample_ids": dataset.excluded_sample_ids,
            "excluded_reason_counts": dataset.excluded_reason_counts,
            "train_subjects": train_subjects, "test_subjects": test_subjects,
        },
        "environment.json": env_snapshot,
        "split_manifest.json": split_manifest,
    }
    run_dir = write_evidence_run(
        output_root, run_id, files,
        extra_manifest_fields={
            "dataset_accession": "TCGA-LUAD+TCGA-LUSC", "track": "C", "task": tracks.TASK_CANCER_PREDICTION,
        },
    )
    return {"status": "complete", "run_dir": str(run_dir), "report": report}


DEFAULT_GSE123352_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
_C_GRID = (0.1, 1.0, 10.0)

DEFAULT_FROZEN_INTERNAL_TEST_POLICY = {
    "min_total_subjects": 300,
    "min_per_class_subjects": 50,
    "train_frac": 0.60,
    "development_holdout_frac": 0.20,
    "internal_test_frac": 0.20,
}


class SubjectOverlapError(ValueError):
    """Raised when a hard partition-disjointness check fails — never
    silently tolerated. See _assert_no_subject_overlap."""


def _assert_no_subject_overlap(partitions: Dict[str, Sequence[str]]) -> None:
    """Hard validation (Issue #16 step 3.1): no subject may appear in more
    than one named partition (e.g. {'train': [...], 'test': [...]} or
    {'inner_train': [...], 'inner_val': [...]}). Raises SubjectOverlapError
    naming the exact overlapping subjects and partitions on failure — this
    is a redundant, explicit re-check on top of
    data.splitting.SplitManifest's own internal overlap assertion, so a
    caller that builds a split without going through SplitManifest is still
    caught."""
    seen: Dict[str, str] = {}
    overlaps: Dict[str, list] = {}
    for name, subjects in partitions.items():
        for s in subjects:
            if s in seen and seen[s] != name:
                overlaps.setdefault(s, [seen[s]]).append(name)
            else:
                seen[s] = name
    if overlaps:
        raise SubjectOverlapError(
            f"Subject(s) appear in more than one partition: {overlaps}"
        )


def _load_evidence_config(evidence_config_path: Union[str, Path] = "configs/evidence.yaml") -> Dict[str, Any]:
    import yaml
    try:
        with open(evidence_config_path) as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _load_gse123352_repeat_seeds(evidence_config_path: Union[str, Path] = "configs/evidence.yaml") -> list:
    raw = _load_evidence_config(evidence_config_path)
    seeds = raw.get("development_repeat_seeds", {}).get("gse123352_smoke_classification")
    if seeds:
        return [int(s) for s in seeds]
    return list(DEFAULT_GSE123352_SEEDS)


def _load_frozen_internal_test_policy(evidence_config_path: Union[str, Path] = "configs/evidence.yaml") -> Dict[str, Any]:
    raw = _load_evidence_config(evidence_config_path)
    policy = dict(DEFAULT_FROZEN_INTERNAL_TEST_POLICY)
    policy.update(raw.get("frozen_internal_test_policy") or {})
    return policy


def assess_frozen_internal_test_eligibility(
    *, unique_subject_count: int, class_counts: Dict[str, int],
    config_path: Union[str, Path] = "configs/evidence.yaml",
) -> Dict[str, Any]:
    """Issue #16 step 3.7: a deterministic, versioned-policy eligibility
    assessment for carving out a frozen internal-test partition — never a
    bare 'no frozen partition exists' statement. Reports the real computed
    subject/class counts against the configured minimum support policy and
    the exact reason a frozen partition can or cannot be created. Does not
    itself create or evaluate any partition."""
    import hashlib
    import json

    policy = _load_frozen_internal_test_policy(config_path)
    reasons = []
    if unique_subject_count < policy["min_total_subjects"]:
        reasons.append(
            f"total verified subjects ({unique_subject_count}) below the configured minimum "
            f"({policy['min_total_subjects']})"
        )
    under_min_class = {c: n for c, n in class_counts.items() if n < policy["min_per_class_subjects"]}
    if under_min_class:
        reasons.append(
            f"per-class subject count(s) below the configured minimum ({policy['min_per_class_subjects']}): "
            f"{under_min_class}"
        )
    eligible = not reasons
    result = {
        "eligible": eligible,
        "unique_subject_count": unique_subject_count,
        "class_counts": dict(class_counts),
        "configured_policy": policy,
        "policy_fingerprint": hashlib.sha256(
            json.dumps(policy, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    if not eligible:
        result["reason_code"] = "INSUFFICIENT_SUPPORT_FOR_FROZEN_INTERNAL_TEST"
        result["reason"] = "; ".join(reasons)
        result["required_next_action"] = (
            "Acquire additional verified subjects for this cohort (or a compatible pooled cohort) "
            "until both the total and per-class minimums in configured_policy are met, then re-run "
            "this assessment before attempting to carve out a frozen internal-test partition."
        )
    return result


def _macro_f1_metric(y_true, y_pred) -> Optional[float]:
    from sklearn.metrics import f1_score
    if len(set(y_true)) < 2:
        return None
    return float(f1_score(y_true, [round(p) for p in y_pred], average="macro"))


def _balanced_accuracy_metric(y_true, y_pred) -> Optional[float]:
    from sklearn.metrics import balanced_accuracy_score
    if len(set(y_true)) < 2:
        return None
    return float(balanced_accuracy_score(y_true, [round(p) for p in y_pred]))


def _fold_local_selection(dataset, train_subject_ids, seed, n_top_variance_genes) -> Dict[str, Any]:
    """Development-only hyperparameter search: splits the OUTER TRAIN
    partition (never the outer held-out partition) into an inner
    train/validation pair, fits each candidate C on the inner-train rows,
    and picks the C with the best macro-F1 on the inner-validation rows.
    The winning C is then refit on the FULL outer-train partition by the
    caller — this function never touches outer-test data. The inner
    train/validation row indices are also returned so callers can reuse the
    SAME inner split for a leakage-safe, development-only calibration
    pathway (evidence/development.py's calibration section below) without
    building a second, uncoordinated inner split."""
    from data.splitting import subject_train_val_test_split
    from data.bulk_pipeline import fit_bulk_logistic_regression, apply_bulk_classifier

    id_to_row = {sid: i for i, sid in enumerate(dataset.subject_ids)}
    train_idx_all = [id_to_row[s] for s in train_subject_ids if s in id_to_row]
    train_labels_all = [int(dataset.y[i]) for i in train_idx_all]

    inner_split = subject_train_val_test_split(
        subject_ids=[dataset.subject_ids[i] for i in train_idx_all], labels=train_labels_all,
        train_frac=0.75, val_frac=0.0, test_frac=0.25, seed=seed * 1000 + 1,
    )
    inner_train_idx = [id_to_row[s] for s in inner_split.train_subjects if s in id_to_row]
    inner_val_idx = [id_to_row[s] for s in inner_split.test_subjects if s in id_to_row]
    _assert_no_subject_overlap({
        "inner_train": inner_split.train_subjects, "inner_val": inner_split.test_subjects,
    })

    if len(inner_train_idx) < 4 or len(inner_val_idx) < 2 or len(set(dataset.y[inner_train_idx].tolist())) < 2:
        return {
            "selected_c": 1.0, "inner_train_idx": inner_train_idx, "inner_val_idx": inner_val_idx,
            "feasible": False,
            "reason": (
                f"only {len(inner_train_idx)} inner-train / {len(inner_val_idx)} inner-val subjects "
                "(or a single-class inner-train partition) — too few for an honest inner split; "
                "falling back to the untuned default C=1.0, calibration NOT_EVALUABLE for this seed"
            ),
        }

    best_c, best_score = 1.0, -1.0
    for c in _C_GRID:
        try:
            fitted = fit_bulk_logistic_regression(
                dataset.X[inner_train_idx], dataset.y[inner_train_idx], dataset.gene_names,
                seed=seed, n_top_variance_genes=n_top_variance_genes, C=c,
            )
        except Exception:
            continue
        y_pred, _ = apply_bulk_classifier(fitted, dataset.X[inner_val_idx])
        score = _macro_f1_metric(dataset.y[inner_val_idx].tolist(), y_pred.tolist())
        if score is not None and score > best_score:
            best_c, best_score = c, score
    return {
        "selected_c": best_c, "inner_train_idx": inner_train_idx, "inner_val_idx": inner_val_idx,
        "feasible": True, "reason": None, "selection_metric": "macro_f1", "selection_score": best_score,
        "candidate_grid": list(_C_GRID),
    }


def _full_metric_bundle(y_true: Sequence[int], y_pred: Sequence[int], y_prob: Sequence[float]) -> Dict[str, Any]:
    """The complete per-repeat metric bundle (Issue #16 step 3.5): macro-F1,
    balanced accuracy, ordinary accuracy, weighted F1, per-class
    precision/recall/F1/support, confusion matrix (via
    benchmarks.metrics.full_smoke_metrics_report), plus AUROC/AUPRC/Brier/ECE
    (via benchmarks.metrics.cancer_prediction_metrics, reused rather than
    reimplemented) and log loss. Every metric that is mathematically
    undefined for this repeat's class support (e.g. AUROC with a single
    class present) stays None with an explicit reason — never silently 0."""
    from sklearn.metrics import log_loss
    from benchmarks.metrics import full_smoke_metrics_report, cancer_prediction_metrics

    bundle = full_smoke_metrics_report(list(y_true), list(y_pred), num_classes=2)
    proba_bundle = cancer_prediction_metrics(list(y_true), list(y_prob), threshold=0.5)
    bundle.update({
        "auroc": proba_bundle["auroc"], "auprc": proba_bundle["auprc"],
        "auroc_auprc_undefined_reason": proba_bundle["auroc_auprc_undefined_reason"],
        "brier": proba_bundle["brier"], "ece": proba_bundle["ece"],
    })
    if len(set(y_true)) < 2:
        bundle["log_loss"] = None
        bundle["log_loss_undefined_reason"] = f"only class(es) {set(y_true)} present in y_true"
    else:
        bundle["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
        bundle["log_loss_undefined_reason"] = None
    return bundle


def _baseline_predictions(dataset, train_idx: Sequence[int], test_idx: Sequence[int], seed: int) -> Dict[str, tuple]:
    """Fits the required Issue #16 step 3.4 baselines on the EXACT SAME
    subject partition as the candidate (train_idx/test_idx, built once by
    the caller and passed unchanged here) and returns
    {baseline_name: (y_pred, y_prob)}. No legitimate non-leaking
    clinical/metadata covariate exists for GSE123352 beyond expression
    itself (see docs/DATA_CARD.md) — a 'clinical baseline' is deliberately
    NOT invented here rather than fabricated from something leaky."""
    from benchmarks.baselines import SmokeMajorityBaseline, SmokeLogisticRegression

    X_train, y_train = dataset.X[train_idx], dataset.y[train_idx]
    X_test = dataset.X[test_idx]

    majority = SmokeMajorityBaseline().fit(X_train, y_train, seed=seed)
    maj_pred = majority.predict(X_test)
    maj_classes = list(majority.classes_)
    maj_prob = (
        majority.predict_proba(X_test)[:, maj_classes.index(1)] if 1 in maj_classes
        else np.zeros(len(X_test)) if maj_classes else np.full(len(X_test), 0.5)
    )

    prevalence = float(np.mean(y_train)) if len(y_train) else 0.5
    prev_prob = np.full(len(X_test), prevalence)
    prev_pred = (prev_prob >= 0.5).astype(int)

    linear = SmokeLogisticRegression(C=1.0).fit(X_train, y_train, seed=seed)
    lin_pred = linear.predict(X_test)
    lin_classes = list(linear.classes_)
    lin_prob = (
        linear.predict_proba(X_test)[:, lin_classes.index(1)] if 1 in lin_classes
        else np.zeros(len(X_test))
    )

    return {
        "majority": (maj_pred.tolist(), maj_prob.tolist()),
        "prevalence": (prev_pred.tolist(), prev_prob.tolist()),
        "bulk_linear_untuned": (lin_pred.tolist(), lin_prob.tolist()),
    }


def run_gse123352_repeated_development(
    *, seeds: Optional[Sequence[int]] = None, train_frac: float = 0.70, test_frac: float = 0.30,
    n_top_variance_genes: int = 2000, _dataset_override: Optional[Any] = None,
) -> Dict[str, Any]:
    """Repeated grouped development evaluation (Issue #16 blocker 6):
    for each predeclared seed, splits GSE123352 subjects into a fresh
    train/development-holdout partition, selects C via a fold-local inner
    split of the TRAIN partition only (never the outer held-out partition),
    refits on the full outer-train partition with the selected C, and
    scores the outer held-out partition — producing exactly one
    development-holdout prediction per seed. Returns not_evaluable if the
    real raw GSE123352 files are not present locally.

    The resulting per-seed macro-F1/balanced-accuracy values are aggregated
    via evidence.uncertainty's repeated_metric_summary (reused, not
    reimplemented) into a mean/median/std/IQR and a seed-level bootstrap CI,
    plus a subject-level bootstrap CI computed on the pooled
    development-holdout predictions across all seeds.

    This is a NEW result, distinct from and never conflated with the
    original single 70/30-split exploratory result
    (evidence.tracks.run_track_a_on_real_verified_label_data) — that
    earlier result remains valid as a labeled single-split exploratory
    finding; this function's output is the repeated-development primary
    estimate.
    """
    from .uncertainty import (
        build_development_repeated_oof, build_repeat_record, paired_candidate_comparison,
        repeated_metric_summary, subject_level_bootstrap_ci,
    )
    from data import bulk_pipeline
    from benchmarks.calibration import build_frozen_policy

    seeds = list(seeds) if seeds is not None else _load_gse123352_repeat_seeds()

    if _dataset_override is not None:
        # test-only injection point — bypasses the real raw-file/conversion
        # path so the statistical aggregation logic below can be exercised
        # against a small synthetic dataset without real GSE123352 files.
        dataset = _dataset_override
    else:
        raw_files = [GSE123352_RAW_DIR / name for name in GSE123352_RAW_FILE_NAMES]
        missing = [str(p) for p in raw_files if not p.exists()]
        if missing:
            return not_evaluable(
                reason_code="REAL_DATA_NOT_PRESENT",
                reason=f"missing real raw GSE123352 file(s) under {GSE123352_RAW_DIR}: {missing}",
                required_next_action="Download the real GSE123352 raw files to this path before running repeated development.",
                cohort_id="gse123352", task="smoke_classification",
            )
        from data.converters import convert_accession
        csv_path = convert_accession("GSE123352")
        if csv_path is None:
            return not_evaluable(
                reason_code="CONVERSION_FAILED", reason="convert_accession('GSE123352') returned no path.",
                required_next_action="Investigate data/converters.py::convert_accession for GSE123352.",
                cohort_id="gse123352", task="smoke_classification",
            )
        dataset = bulk_pipeline.build_bulk_smoke_dataset(csv_path)

    candidate_repeats, majority_repeats, prevalence_repeats, linear_repeats = [], [], [], []
    selected_hyperparameters: Dict[int, float] = {}
    excluded_seeds: list = []
    full_metric_bundle_by_seed: Dict[str, Any] = {}
    calibration_by_seed: Dict[str, Any] = {}
    inner_selection_by_seed: Dict[str, Any] = {}

    for seed in seeds:
        split = bulk_pipeline.split_bulk_subjects(
            dataset, seed=seed, train_frac=train_frac, val_frac=0.0, test_frac=test_frac,
        )
        id_to_row = {sid: i for i, sid in enumerate(dataset.subject_ids)}
        train_idx = [id_to_row[s] for s in split.train_subjects if s in id_to_row]
        test_idx = [id_to_row[s] for s in split.test_subjects if s in id_to_row]
        _assert_no_subject_overlap({"outer_train": split.train_subjects, "outer_test": split.test_subjects})
        if len(test_idx) == 0 or len(set(dataset.y[train_idx].tolist())) < 2:
            excluded_seeds.append({
                "seed": seed,
                "reason": f"outer split infeasible for this seed (n_test={len(test_idx)}, "
                          f"train classes={sorted(set(dataset.y[train_idx].tolist()))})",
            })
            continue

        selection = _fold_local_selection(dataset, split.train_subjects, seed, n_top_variance_genes)
        selected_c = selection["selected_c"]
        selected_hyperparameters[seed] = selected_c
        inner_selection_by_seed[str(seed)] = {
            "feasible": selection["feasible"], "reason": selection["reason"],
            "selected_c": selected_c, "candidate_grid": list(_C_GRID),
            "n_inner_train": len(selection["inner_train_idx"]), "n_inner_val": len(selection["inner_val_idx"]),
        }

        fitted = bulk_pipeline.fit_bulk_logistic_regression(
            dataset.X[train_idx], dataset.y[train_idx], dataset.gene_names,
            seed=seed, n_top_variance_genes=n_top_variance_genes, C=selected_c,
        )
        y_pred, y_prob = bulk_pipeline.apply_bulk_classifier(fitted, dataset.X[test_idx])
        test_subject_ids = [dataset.subject_ids[i] for i in test_idx]
        test_y_true = dataset.y[test_idx].tolist()

        candidate_repeats.append(build_repeat_record(seed, test_subject_ids, test_y_true, y_pred.tolist()))
        full_metric_bundle_by_seed[str(seed)] = _full_metric_bundle(test_y_true, y_pred.tolist(), y_prob.tolist())

        # Required baselines (Issue #16 step 3.4) — fit on the IDENTICAL
        # train_idx/test_idx partition as the candidate above, never a
        # separately-drawn split, so paired comparison is valid.
        baseline_preds = _baseline_predictions(dataset, train_idx, test_idx, seed)
        majority_repeats.append(build_repeat_record(seed, test_subject_ids, test_y_true, baseline_preds["majority"][0]))
        prevalence_repeats.append(build_repeat_record(seed, test_subject_ids, test_y_true, baseline_preds["prevalence"][0]))
        linear_repeats.append(build_repeat_record(seed, test_subject_ids, test_y_true, baseline_preds["bulk_linear_untuned"][0]))

        # Development-only calibration pathway (Issue #16 step 3.6). Reuses
        # the SAME inner train/validation split _fold_local_selection just
        # built (never a second, uncoordinated inner split) so the
        # calibration-fitting set (inner_val) is disjoint from both
        # inner_train (what the calibration-source model is fit on) and the
        # outer test set. This calibration-source model is fit on
        # inner_train ONLY, distinct from the primary candidate above
        # (fit on the FULL outer train) — no outer-train-disjoint data
        # would otherwise remain to fit and evaluate calibration without
        # touching outer test, so this is reported as a separate, clearly
        # labeled pathway rather than applied to the primary candidate's
        # own probabilities.
        if not selection["feasible"]:
            calibration_by_seed[str(seed)] = {
                "status": "not_evaluable", "reason": selection["reason"],
                "calibration_fingerprint": "not_applicable",
            }
        else:
            inner_train_idx, inner_val_idx = selection["inner_train_idx"], selection["inner_val_idx"]
            calib_source = bulk_pipeline.fit_bulk_logistic_regression(
                dataset.X[inner_train_idx], dataset.y[inner_train_idx], dataset.gene_names,
                seed=seed, n_top_variance_genes=n_top_variance_genes, C=selected_c,
            )
            _, inner_val_prob = bulk_pipeline.apply_bulk_classifier(calib_source, dataset.X[inner_val_idx])
            _, outer_test_prob_from_calib_source = bulk_pipeline.apply_bulk_classifier(calib_source, dataset.X[test_idx])
            policy = build_frozen_policy(
                dataset.y[inner_val_idx], inner_val_prob, calibration_method="auto", threshold_strategy="youden",
            )
            uncalibrated = _full_metric_bundle(
                test_y_true, (outer_test_prob_from_calib_source >= 0.5).astype(int).tolist(),
                outer_test_prob_from_calib_source.tolist(),
            )
            calibrated = policy.apply_to_test(np.asarray(test_y_true), outer_test_prob_from_calib_source)
            calibration_by_seed[str(seed)] = {
                "status": "complete",
                "calibration_source_model": "fit_on_inner_train_only_not_the_primary_candidate",
                "calibration_fitting_set_size": len(inner_val_idx),
                "calibration_method": policy.calibrator.method,
                "calibration_params": policy.calibrator.to_dict(),
                "threshold": policy.threshold, "threshold_strategy": policy.threshold_strategy,
                "threshold_reason": policy.threshold_reason,
                "uncalibrated_outer_test_metrics": uncalibrated,
                "calibrated_outer_test_metrics": calibrated,
            }

    if not candidate_repeats:
        return not_evaluable(
            reason_code="INSUFFICIENT_REPEATS", reason="No seed produced a usable development-holdout partition.",
            required_next_action="Check GSE123352 subject counts against the configured split fractions.",
            cohort_id="gse123352", task="smoke_classification",
        )

    candidate_oof = build_development_repeated_oof(candidate_repeats, role="development")
    macro_f1_summary = repeated_metric_summary(candidate_oof, _macro_f1_metric)
    balanced_accuracy_summary = repeated_metric_summary(candidate_oof, _balanced_accuracy_metric)

    # A per-repeat (not pooled-across-repeats) subject-level bootstrap CI —
    # pooling raw predictions across repeats would let the same subject
    # (held out in more than one repeat, which is the expected, normal
    # outcome of repeated resampling of the same development pool) count
    # more than once in a single bootstrap resample, silently understating
    # variance. Each repeat's subject set is unique within itself
    # (RepeatRecord enforces this), so per-repeat CIs stay genuinely
    # subject-level.
    per_seed_macro_f1_ci = {
        str(r.seed): subject_level_bootstrap_ci(r, _macro_f1_metric) for r in candidate_repeats
    }

    baseline_oofs = {
        "majority": build_development_repeated_oof(majority_repeats, role="development"),
        "prevalence": build_development_repeated_oof(prevalence_repeats, role="development"),
        "bulk_linear_untuned": build_development_repeated_oof(linear_repeats, role="development"),
    }
    baseline_comparisons = {}
    for name, baseline_oof in baseline_oofs.items():
        comparison = paired_candidate_comparison(candidate_oof, baseline_oof, _macro_f1_metric)
        n_defined_pairs = len(comparison["common_seeds"]) - comparison["n_undefined"]
        comparison["candidate_name"] = "bulk_logistic_regression_fold_local_c"
        comparison["baseline_name"] = name
        comparison["shared_subject_partition"] = (
            "identical train_idx/test_idx per seed by construction — both fit inside the same "
            "loop iteration on the same outer split"
        )
        comparison["ci"] = None
        if n_defined_pairs >= 2:
            defined_diffs = [p["diff"] for p in comparison["per_seed"] if p["diff"] is not None]
            from benchmarks.metrics import bootstrap_ci
            comparison["ci"] = bootstrap_ci(defined_diffs)
            comparison["status"] = "reported"
        else:
            comparison["status"] = "insufficient_evidence"
            comparison["insufficient_evidence_reason"] = (
                f"only {n_defined_pairs} seed(s) produced a defined paired macro-F1 difference — "
                "at least 2 are required for a paired confidence interval."
            )
        baseline_comparisons[name] = comparison

    unique_subject_count = len(set(dataset.subject_ids))
    class_counts = {
        name: int((dataset.y == i).sum()) for i, name in enumerate(dataset.class_names)
    }
    frozen_test_eligibility = assess_frozen_internal_test_eligibility(
        unique_subject_count=unique_subject_count, class_counts=class_counts,
    )

    return {
        "status": "complete",
        "task": "smoke_classification",
        "dataset_accession": "GSE123352",
        "protocol": "repeated_grouped_development_holdout",
        "seeds": seeds,
        "n_requested_seeds": len(seeds),
        "n_completed_seeds": len(candidate_repeats),
        "n_excluded_seeds": len(excluded_seeds),
        "excluded_seeds": excluded_seeds,
        "independent_development_holdout_subject_count": candidate_oof.independent_subject_count(),
        "selected_hyperparameters_by_seed": {str(k): v for k, v in selected_hyperparameters.items()},
        "inner_selection_by_seed": inner_selection_by_seed,
        "primary_metric": "macro_f1",
        "macro_f1": macro_f1_summary,
        "balanced_accuracy": balanced_accuracy_summary,
        "per_seed_macro_f1_subject_bootstrap_ci": per_seed_macro_f1_ci,
        "full_metric_bundle_by_seed": full_metric_bundle_by_seed,
        "baseline_comparisons": baseline_comparisons,
        "calibration_by_seed": calibration_by_seed,
        "frozen_internal_test_eligibility": frozen_test_eligibility,
        "limitations": [
            "Repeated grouped development-holdout estimate — every held-out subject here is "
            "still a development-role subject (gse123352's role_eligibility is [development] "
            "only); this is NOT internal-test or external-validation evidence.",
            "Hyperparameter (C) selection is fold-local per seed, using only that seed's "
            "outer-train partition — no outer held-out subject ever influenced selection.",
            "The original single 70/30-split exploratory result "
            "(evidence.tracks.run_track_a_on_real_verified_label_data) remains a separate, "
            "still-valid single-split exploratory finding and is not superseded by this result.",
            "Calibration is evaluated on a model fit only on the inner-train partition (roughly "
            "75% of each seed's outer-train subjects), not on the primary full-outer-train "
            "candidate reported above — no outer-train-disjoint data remains to fit and evaluate "
            "calibration for the primary candidate without touching outer test.",
            "No legitimate non-leaking clinical/metadata covariate exists for GSE123352 beyond "
            "expression itself, so no separate clinical/metadata baseline is reported (see "
            "docs/DATA_CARD.md).",
            f"Frozen internal-test eligibility: {frozen_test_eligibility.get('reason', 'eligible under configured policy')}.",
        ],
    }


DEFAULT_TCGA_VITAL_STATUS_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)


def _macro_f1_metric_from_prob(y_true, y_prob) -> Optional[float]:
    """Same convention as _macro_f1_metric — rounds probabilities to a hard
    label at 0.5 before scoring. Kept separate (not reused for smoke's
    y_pred, which is already a hard label) purely for naming clarity at
    call sites that pass probabilities."""
    return _macro_f1_metric(y_true, y_prob)


def run_tcga_lung_vital_status_repeated_development(
    *, seeds: Optional[Sequence[int]] = None, train_frac: float = 0.70, test_frac: float = 0.30,
    n_top_variance_genes: int = 2000, _dataset_override: Optional[Any] = None,
) -> Dict[str, Any]:
    """Repeated grouped development evaluation for the real TCGA-LUAD+
    TCGA-LUSC vital-status cohort — the same protocol as
    run_gse123352_repeated_development() (fold-local hyperparameter
    selection, required baselines on the identical partition, the full
    per-seed metric bundle, a development-only calibration pathway, and a
    frozen-internal-test eligibility decision), applied to this repository's
    first genuinely linked expression<->cancer-outcome cohort. Returns
    not_evaluable if the real downloaded GDC data is not present locally."""
    from .uncertainty import (
        build_development_repeated_oof, build_repeat_record, paired_candidate_comparison,
        repeated_metric_summary, subject_level_bootstrap_ci,
    )
    from data import bulk_pipeline
    from benchmarks.calibration import build_frozen_policy

    seeds = list(seeds) if seeds is not None else list(DEFAULT_TCGA_VITAL_STATUS_SEEDS)

    if _dataset_override is not None:
        dataset = _dataset_override
    else:
        dataset, missing = _tcga_lung_vital_status_dataset()
        if dataset is None:
            return not_evaluable(
                reason_code="REAL_DATA_NOT_PRESENT",
                reason=f"missing real downloaded GDC files for {missing}.",
                required_next_action="Download the real TCGA-LUAD/TCGA-LUSC open-access GDC files before running repeated development.",
                cohort_id="tcga_lung_vital_status", task="subject_level_cancer_prediction",
            )

    candidate_repeats, majority_repeats, prevalence_repeats, linear_repeats = [], [], [], []
    selected_hyperparameters: Dict[int, float] = {}
    excluded_seeds: list = []
    full_metric_bundle_by_seed: Dict[str, Any] = {}
    calibration_by_seed: Dict[str, Any] = {}
    inner_selection_by_seed: Dict[str, Any] = {}

    for seed in seeds:
        from data.splitting import subject_train_val_test_split
        split = subject_train_val_test_split(
            subject_ids=dataset.subject_ids, labels=list(dataset.y),
            train_frac=train_frac, val_frac=0.0, test_frac=test_frac, seed=seed,
        )
        id_to_row = {sid: i for i, sid in enumerate(dataset.subject_ids)}
        train_idx = [id_to_row[s] for s in split.train_subjects if s in id_to_row]
        test_idx = [id_to_row[s] for s in split.test_subjects if s in id_to_row]
        _assert_no_subject_overlap({"outer_train": split.train_subjects, "outer_test": split.test_subjects})
        if len(test_idx) == 0 or len(set(dataset.y[train_idx].tolist())) < 2:
            excluded_seeds.append({
                "seed": seed,
                "reason": f"outer split infeasible for this seed (n_test={len(test_idx)}, "
                          f"train classes={sorted(set(dataset.y[train_idx].tolist()))})",
            })
            continue

        selection = _fold_local_selection(dataset, split.train_subjects, seed, n_top_variance_genes)
        selected_c = selection["selected_c"]
        selected_hyperparameters[seed] = selected_c
        inner_selection_by_seed[str(seed)] = {
            "feasible": selection["feasible"], "reason": selection["reason"],
            "selected_c": selected_c, "candidate_grid": list(_C_GRID),
            "n_inner_train": len(selection["inner_train_idx"]), "n_inner_val": len(selection["inner_val_idx"]),
        }

        fitted = bulk_pipeline.fit_bulk_logistic_regression(
            dataset.X[train_idx], dataset.y[train_idx], dataset.gene_names,
            seed=seed, n_top_variance_genes=n_top_variance_genes, C=selected_c,
        )
        y_pred, y_prob = bulk_pipeline.apply_bulk_classifier(fitted, dataset.X[test_idx])
        test_subject_ids = [dataset.subject_ids[i] for i in test_idx]
        test_y_true = dataset.y[test_idx].tolist()

        candidate_repeats.append(build_repeat_record(seed, test_subject_ids, test_y_true, y_prob.tolist()))
        full_metric_bundle_by_seed[str(seed)] = _full_metric_bundle(test_y_true, y_pred.tolist(), y_prob.tolist())

        baseline_preds = _baseline_predictions(dataset, train_idx, test_idx, seed)
        majority_repeats.append(build_repeat_record(seed, test_subject_ids, test_y_true, baseline_preds["majority"][1]))
        prevalence_repeats.append(build_repeat_record(seed, test_subject_ids, test_y_true, baseline_preds["prevalence"][1]))
        linear_repeats.append(build_repeat_record(seed, test_subject_ids, test_y_true, baseline_preds["bulk_linear_untuned"][1]))

        if not selection["feasible"]:
            calibration_by_seed[str(seed)] = {
                "status": "not_evaluable", "reason": selection["reason"],
                "calibration_fingerprint": "not_applicable",
            }
        else:
            inner_train_idx, inner_val_idx = selection["inner_train_idx"], selection["inner_val_idx"]
            calib_source = bulk_pipeline.fit_bulk_logistic_regression(
                dataset.X[inner_train_idx], dataset.y[inner_train_idx], dataset.gene_names,
                seed=seed, n_top_variance_genes=n_top_variance_genes, C=selected_c,
            )
            _, inner_val_prob = bulk_pipeline.apply_bulk_classifier(calib_source, dataset.X[inner_val_idx])
            _, outer_test_prob_from_calib_source = bulk_pipeline.apply_bulk_classifier(calib_source, dataset.X[test_idx])
            policy = build_frozen_policy(
                dataset.y[inner_val_idx], inner_val_prob, calibration_method="auto", threshold_strategy="youden",
            )
            uncalibrated = _full_metric_bundle(
                test_y_true, (outer_test_prob_from_calib_source >= 0.5).astype(int).tolist(),
                outer_test_prob_from_calib_source.tolist(),
            )
            calibrated = policy.apply_to_test(np.asarray(test_y_true), outer_test_prob_from_calib_source)
            calibration_by_seed[str(seed)] = {
                "status": "complete",
                "calibration_source_model": "fit_on_inner_train_only_not_the_primary_candidate",
                "calibration_fitting_set_size": len(inner_val_idx),
                "calibration_method": policy.calibrator.method,
                "calibration_params": policy.calibrator.to_dict(),
                "threshold": policy.threshold, "threshold_strategy": policy.threshold_strategy,
                "threshold_reason": policy.threshold_reason,
                "uncalibrated_outer_test_metrics": uncalibrated,
                "calibrated_outer_test_metrics": calibrated,
            }

    if not candidate_repeats:
        return not_evaluable(
            reason_code="INSUFFICIENT_REPEATS", reason="No seed produced a usable development-holdout partition.",
            required_next_action="Check TCGA-LUAD+TCGA-LUSC subject counts against the configured split fractions.",
            cohort_id="tcga_lung_vital_status", task="subject_level_cancer_prediction",
        )

    candidate_oof = build_development_repeated_oof(candidate_repeats, role="development")
    macro_f1_summary = repeated_metric_summary(candidate_oof, _macro_f1_metric_from_prob)
    balanced_accuracy_summary = repeated_metric_summary(candidate_oof, _balanced_accuracy_metric)
    per_seed_macro_f1_ci = {
        str(r.seed): subject_level_bootstrap_ci(r, _macro_f1_metric_from_prob) for r in candidate_repeats
    }

    baseline_oofs = {
        "majority": build_development_repeated_oof(majority_repeats, role="development"),
        "prevalence": build_development_repeated_oof(prevalence_repeats, role="development"),
        "bulk_linear_untuned": build_development_repeated_oof(linear_repeats, role="development"),
    }
    baseline_comparisons = {}
    for name, baseline_oof in baseline_oofs.items():
        comparison = paired_candidate_comparison(candidate_oof, baseline_oof, _macro_f1_metric_from_prob)
        n_defined_pairs = len(comparison["common_seeds"]) - comparison["n_undefined"]
        comparison["candidate_name"] = "bulk_logistic_regression_fold_local_c"
        comparison["baseline_name"] = name
        comparison["shared_subject_partition"] = (
            "identical train_idx/test_idx per seed by construction — both fit inside the same "
            "loop iteration on the same outer split"
        )
        comparison["ci"] = None
        if n_defined_pairs >= 2:
            defined_diffs = [p["diff"] for p in comparison["per_seed"] if p["diff"] is not None]
            from benchmarks.metrics import bootstrap_ci
            comparison["ci"] = bootstrap_ci(defined_diffs)
            comparison["status"] = "reported"
        else:
            comparison["status"] = "insufficient_evidence"
            comparison["insufficient_evidence_reason"] = (
                f"only {n_defined_pairs} seed(s) produced a defined paired macro-F1 difference — "
                "at least 2 are required for a paired confidence interval."
            )
        baseline_comparisons[name] = comparison

    unique_subject_count = len(set(dataset.subject_ids))
    class_counts = {name: int((dataset.y == i).sum()) for i, name in enumerate(dataset.class_names)}
    frozen_test_eligibility = assess_frozen_internal_test_eligibility(
        unique_subject_count=unique_subject_count, class_counts=class_counts,
    )

    return {
        "status": "complete",
        "task": "subject_level_cancer_prediction",
        "dataset_accession": "TCGA-LUAD+TCGA-LUSC",
        "protocol": "repeated_grouped_development_holdout",
        "seeds": seeds,
        "n_requested_seeds": len(seeds),
        "n_completed_seeds": len(candidate_repeats),
        "n_excluded_seeds": len(excluded_seeds),
        "excluded_seeds": excluded_seeds,
        "independent_development_holdout_subject_count": candidate_oof.independent_subject_count(),
        "selected_hyperparameters_by_seed": {str(k): v for k, v in selected_hyperparameters.items()},
        "inner_selection_by_seed": inner_selection_by_seed,
        "primary_metric": "macro_f1",
        "macro_f1": macro_f1_summary,
        "balanced_accuracy": balanced_accuracy_summary,
        "per_seed_macro_f1_subject_bootstrap_ci": per_seed_macro_f1_ci,
        "full_metric_bundle_by_seed": full_metric_bundle_by_seed,
        "baseline_comparisons": baseline_comparisons,
        "calibration_by_seed": calibration_by_seed,
        "frozen_internal_test_eligibility": frozen_test_eligibility,
        "limitations": [
            "Repeated grouped development-holdout estimate — every held-out subject here is "
            "still a development-role subject; this is NOT internal-test or external-validation "
            "evidence, even if the frozen-internal-test eligibility decision below is 'eligible' — "
            "eligibility means a frozen partition COULD be created, not that one has been.",
            "vital_status is a binary outcome at last recorded GDC follow-up, not a time-to-event "
            "survival label — no censoring or follow-up duration is modeled.",
            "Hyperparameter (C) selection is fold-local per seed, using only that seed's "
            "outer-train partition — no outer held-out subject ever influenced selection.",
            "Calibration is evaluated on a model fit only on the inner-train partition, not on the "
            "primary full-outer-train candidate reported above — no outer-train-disjoint data "
            "remains to fit and evaluate calibration for the primary candidate without touching "
            "outer test.",
            "No legitimate non-leaking clinical/metadata baseline is reported beyond expression "
            "itself for this development-only bulk pipeline.",
            f"Frozen internal-test eligibility: {frozen_test_eligibility.get('reason', 'eligible under configured policy')}.",
        ],
    }


def run_internal_test(
    run_dir: Union[str, Path], authorization_file: Optional[str],
    *, cohort_id: Optional[str] = None, task: Optional[str] = None,
) -> Dict[str, Any]:
    """Honest gate: no cohort in this repository has ever had a frozen
    internal-test partition created and guarded (see configs/cohorts.yaml —
    every registered cohort's role_eligibility is [development] only), so
    this always returns a structured not_evaluable — never a generic 'not
    yet integrated' stub, and never a fabricated frozen-test result
    regardless of what run_dir or authorization_file is supplied.

    When cohort_id='gse123352'/task='smoke_classification' is given, the
    not_evaluable reason is the REAL, computed
    assess_frozen_internal_test_eligibility() decision (Issue #16 step
    3.7) — not a bare 'no partition exists' statement — proving, with real
    subject/class counts, why a frozen partition cannot honestly be carved
    out today. Any other cohort/task (or none given) falls back to the
    structural NO_FROZEN_TEST_PARTITION_EXISTS gate, since no eligibility
    assessment has been wired up for it."""
    from .artifact_bundle import validate_evidence_run
    from .errors import ArtifactValidationError

    artifacts_root, run_id = Path(run_dir).parent, Path(run_dir).name
    try:
        validate_evidence_run(artifacts_root, run_id)
    except ArtifactValidationError as exc:
        return not_evaluable(
            reason_code="DEVELOPMENT_RUN_INVALID",
            reason=f"the development run at {run_dir} did not validate: {exc}",
            required_next_action="Re-run evidence.runner development to produce a valid, complete run bundle first.",
        )

    if cohort_id == "gse123352" and task == "smoke_classification":
        raw_files = [GSE123352_RAW_DIR / name for name in GSE123352_RAW_FILE_NAMES]
        missing = [str(p) for p in raw_files if not p.exists()]
        if missing:
            return not_evaluable(
                reason_code="REAL_DATA_NOT_PRESENT",
                reason=f"missing real raw GSE123352 file(s) under {GSE123352_RAW_DIR}: {missing}",
                required_next_action="Download the real GSE123352 raw files to this path before assessing frozen-test eligibility.",
                cohort_id="gse123352", task="smoke_classification",
            )
        from data.converters import convert_accession
        from data import bulk_pipeline
        csv_path = convert_accession("GSE123352")
        dataset = bulk_pipeline.build_bulk_smoke_dataset(csv_path)
        unique_subject_count = len(set(dataset.subject_ids))
        class_counts = {name: int((dataset.y == i).sum()) for i, name in enumerate(dataset.class_names)}
        eligibility = assess_frozen_internal_test_eligibility(
            unique_subject_count=unique_subject_count, class_counts=class_counts,
        )
        if eligibility["eligible"]:
            return not_evaluable(
                reason_code="FROZEN_PARTITION_ELIGIBLE_BUT_NOT_YET_CREATED",
                reason=(
                    f"gse123352/smoke_classification meets the configured frozen internal-test "
                    f"support policy ({unique_subject_count} subjects, class counts {class_counts}) "
                    "but no frozen partition manifest has been created and guarded for it yet in "
                    "this repository."
                ),
                required_next_action=(
                    "Create and freeze an internal-test subject partition (persisting its manifest "
                    "without ever reading its expression/labels during development), update "
                    "configs/cohorts.yaml role_eligibility to include 'internal_test', and acquire "
                    "the one-shot frozen-test guard (benchmarks.test_guard) before calling this "
                    "subcommand again."
                ),
                **eligibility,
            )
        return not_evaluable(
            reason_code=eligibility["reason_code"], reason=eligibility["reason"],
            required_next_action=eligibility["required_next_action"], cohort_id="gse123352",
            task="smoke_classification", **{k: v for k, v in eligibility.items() if k not in ("reason_code", "reason", "required_next_action")},
        )

    return not_evaluable(
        reason_code="NO_FROZEN_TEST_PARTITION_EXISTS",
        reason=(
            "No cohort registered in configs/cohorts.yaml has ever had a frozen internal-test "
            "partition created and guarded — every registered cohort's role_eligibility is "
            "[development] only. An internal-test result requires a frozen partition that was "
            "carved out and sealed BEFORE any model/feature/hyperparameter selection touched it, "
            "which has never happened for any cohort in this repository."
        ),
        required_next_action=(
            "Create and freeze an internal-test subject partition for the target cohort "
            "(persisting its manifest without ever reading its expression/labels during "
            "development), update its configs/cohorts.yaml role_eligibility to include "
            "'internal_test', and acquire the one-shot frozen-test guard "
            "(benchmarks.test_guard) before calling this subcommand again."
        ),
    )


def run_external_test(run_dir: Union[str, Path], external_cohort: Optional[str]) -> Dict[str, Any]:
    """Honest gate: no cohort in configs/cohorts.yaml carries
    role_eligibility=[external_validation] — see the registry file. This
    always returns NO_ELIGIBLE_EXTERNAL_COHORT rather than fabricating an
    external-validation result."""
    return not_evaluable(
        reason_code="NO_ELIGIBLE_EXTERNAL_COHORT",
        reason=(
            f"No cohort in configs/cohorts.yaml has role_eligibility including "
            f"'external_validation'{f' (requested external_cohort={external_cohort!r})' if external_cohort else ''}. "
            "External validation requires a genuinely independent cohort with zero subject "
            "overlap with any development cohort, which is not currently registered."
        ),
        required_next_action=(
            "Identify and register an independent external cohort with verified zero subject "
            "overlap, add it to configs/cohorts.yaml with role_eligibility including "
            "'external_validation', then re-run this subcommand."
        ),
    )
