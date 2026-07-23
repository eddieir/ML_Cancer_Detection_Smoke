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


DEFAULT_GSE123352_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
_C_GRID = (0.1, 1.0, 10.0)


def _load_gse123352_repeat_seeds(evidence_config_path: Union[str, Path] = "configs/evidence.yaml") -> list:
    import yaml
    try:
        with open(evidence_config_path) as f:
            raw = yaml.safe_load(f) or {}
        seeds = raw.get("development_repeat_seeds", {}).get("gse123352_smoke_classification")
        if seeds:
            return [int(s) for s in seeds]
    except (OSError, yaml.YAMLError):
        pass
    return list(DEFAULT_GSE123352_SEEDS)


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


def _select_c_fold_local(dataset, train_subject_ids, seed, n_top_variance_genes) -> float:
    """Development-only hyperparameter search: splits the OUTER TRAIN
    partition (never the outer held-out partition) into an inner
    train/validation pair, fits each candidate C on the inner-train rows,
    and picks the C with the best macro-F1 on the inner-validation rows.
    The winning C is then refit on the FULL outer-train partition by the
    caller — this function never touches outer-test data."""
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
    if len(inner_train_idx) < 4 or len(inner_val_idx) < 2 or len(set(dataset.y[inner_train_idx].tolist())) < 2:
        return 1.0  # too few subjects for an honest inner split — fall back to the untuned default

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
    return best_c


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
    from .uncertainty import build_development_repeated_oof, build_repeat_record, repeated_metric_summary, subject_level_bootstrap_ci  # noqa: F401
    from data import bulk_pipeline

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

    repeats = []
    selected_hyperparameters = {}

    for seed in seeds:
        split = bulk_pipeline.split_bulk_subjects(
            dataset, seed=seed, train_frac=train_frac, val_frac=0.0, test_frac=test_frac,
        )
        id_to_row = {sid: i for i, sid in enumerate(dataset.subject_ids)}
        train_idx = [id_to_row[s] for s in split.train_subjects if s in id_to_row]
        test_idx = [id_to_row[s] for s in split.test_subjects if s in id_to_row]
        if len(test_idx) == 0 or len(set(dataset.y[train_idx].tolist())) < 2:
            continue

        selected_c = _select_c_fold_local(dataset, split.train_subjects, seed, n_top_variance_genes)
        selected_hyperparameters[seed] = selected_c

        fitted = bulk_pipeline.fit_bulk_logistic_regression(
            dataset.X[train_idx], dataset.y[train_idx], dataset.gene_names,
            seed=seed, n_top_variance_genes=n_top_variance_genes, C=selected_c,
        )
        y_pred, y_prob = bulk_pipeline.apply_bulk_classifier(fitted, dataset.X[test_idx])
        test_subject_ids = [dataset.subject_ids[i] for i in test_idx]
        test_y_true = dataset.y[test_idx].tolist()

        repeats.append(build_repeat_record(seed, test_subject_ids, test_y_true, y_pred.tolist()))

    if not repeats:
        return not_evaluable(
            reason_code="INSUFFICIENT_REPEATS", reason="No seed produced a usable development-holdout partition.",
            required_next_action="Check GSE123352 subject counts against the configured split fractions.",
            cohort_id="gse123352", task="smoke_classification",
        )

    oof = build_development_repeated_oof(repeats, role="development")
    macro_f1_summary = repeated_metric_summary(oof, _macro_f1_metric)
    balanced_accuracy_summary = repeated_metric_summary(oof, _balanced_accuracy_metric)

    # A per-repeat (not pooled-across-repeats) subject-level bootstrap CI —
    # pooling raw predictions across repeats would let the same subject
    # (held out in more than one repeat, which is the expected, normal
    # outcome of repeated resampling of the same development pool) count
    # more than once in a single bootstrap resample, silently understating
    # variance. Each repeat's subject set is unique within itself
    # (RepeatRecord enforces this), so per-repeat CIs stay genuinely
    # subject-level.
    per_seed_macro_f1_ci = {
        str(r.seed): subject_level_bootstrap_ci(r, _macro_f1_metric) for r in repeats
    }

    return {
        "status": "complete",
        "task": "smoke_classification",
        "dataset_accession": "GSE123352",
        "protocol": "repeated_grouped_development_holdout",
        "seeds": seeds,
        "n_repeats": len(repeats),
        "independent_development_holdout_subject_count": oof.independent_subject_count(),
        "selected_hyperparameters_by_seed": {str(k): v for k, v in selected_hyperparameters.items()},
        "primary_metric": "macro_f1",
        "macro_f1": macro_f1_summary,
        "balanced_accuracy": balanced_accuracy_summary,
        "per_seed_macro_f1_subject_bootstrap_ci": per_seed_macro_f1_ci,
        "limitations": [
            "Repeated grouped development-holdout estimate — every held-out subject here is "
            "still a development-role subject (gse123352's role_eligibility is [development] "
            "only); this is NOT internal-test or external-validation evidence.",
            "Hyperparameter (C) selection is fold-local per seed, using only that seed's "
            "outer-train partition — no outer held-out subject ever influenced selection.",
            "The original single 70/30-split exploratory result "
            "(evidence.tracks.run_track_a_on_real_verified_label_data) remains a separate, "
            "still-valid single-split exploratory finding and is not superseded by this result.",
        ],
    }


def run_internal_test(run_dir: Union[str, Path], authorization_file: Optional[str]) -> Dict[str, Any]:
    """Honest gate: no cohort in this repository has ever had a frozen
    internal-test partition created and guarded (see configs/cohorts.yaml —
    every registered cohort's role_eligibility is [development] only), so
    this always returns a structured not_evaluable naming that specific
    blocker — never a generic 'not yet integrated' stub, and never a
    fabricated frozen-test result regardless of what run_dir or
    authorization_file is supplied."""
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
