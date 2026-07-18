"""
benchmarks/runner.py — CLI entry point tying the whole Phase 1 framework
together: ExperimentContext -> eligibility -> grouped CV -> (Task B only)
frozen calibration/threshold -> statistical comparison -> immutable report.

    python -m benchmarks.runner --config configs/default.yaml --task smoke \\
        --models majority logistic random_forest gradient_boosting small_mlp neural \\
        --cv-folds 5 --seeds 42 43 44 --output artifacts/benchmarks

    python -m benchmarks.runner --config configs/default.yaml --task cancer \\
        --models prevalence logistic random_forest gradient_boosting small_mlp \\
        mean_mil max_mil attention_mil --calibration auto --output artifacts/benchmarks

    python -m benchmarks.runner --synthetic --fast

--synthetic builds a small synthetic ExperimentContext instead of reading a
real config, for CI. The report is always stamped synthetic=True in that
mode and the run refuses to be mistaken for a real-data result.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .atomic_io import read_and_verify_csv_rows, read_and_verify_json
from .baselines import CANCER_BASELINES, CANCER_SEARCH_SPACE, SMOKE_BASELINES
from .calibration import build_frozen_policy
from .context import ExperimentContext
from .cross_validation import (
    DEFAULT_INNER_FOLDS,
    MIL_SEARCH_SPACE,
    _cancer_baseline_fit_score_fn,
    _mil_fit_score_fn,
    _pathway_cancer_fit_score_fn,
    run_cancer_cv,
    run_smoke_cv,
)
from .eligibility import check_task_a_eligibility, check_task_b_development_eligibility, check_test_evaluability
from .final_evaluation import (
    NoEligibleFinalCandidateError,
    evaluate_frozen_test,
    fit_final_candidate_on_dev_pool,
    generate_subject_oof_predictions,
    is_mil_candidate,
    select_final_candidate,
)
from .hyperparameter_search import build_param_grid, select_nested_hyperparameters_with_refit
from .mil_registry import pathway_search_space
from .pathway_hierarchical_adapter import MODEL_NAME as PATHWAY_MODEL_NAME
from .pathway_hierarchical_ablation import run_pathway_hierarchical_ablation, write_pathway_hierarchical_ablation_artifact
from .reporting import compare_models, new_run_dir, write_benchmark_report, write_csv_table, write_json
from .ood import run_leave_one_source_out
from .test_guard import FrozenTestGuard, FrozenTestGuardDisabledInRealModeError, default_guard_dir
from data.transforms import assert_batch_correction_safe
from .domain_losses import resolve_domain_robustness_config
from .domain_robustness_ablation import run_domain_robustness_ablation
from .robustness_report import aggregate_source_reports, build_aggregate_report, write_aggregate_report
from .source_held_out import UnsupportedSmokeDomainStrategyError, run_cancer_source_held_out, run_smoke_source_held_out


def build_synthetic_context(seed: int = 42, fast: bool = True) -> ExperimentContext:
    """Small, fully-wired, non-diagnostic ExperimentContext for CI. Real
    (fake but internally consistent) subject ids, two smoke classes' worth
    of spread, two dataset sources, and enough known cancer outcomes with
    both classes present in every split to be ELIGIBLE for both tasks."""
    from train import CellLevelDataset
    from data.label_mapping import identity_label_mapping
    from data.preprocessing import PreprocessingArtifact
    from data.splitting import SplitManifest

    rng = np.random.RandomState(seed)
    n_genes = 15
    num_classes = 3
    cells_each = 10 if fast else 30

    def make_split(subjects, source):
        subj_ids, X, y, src = [], [], [], []
        for i, s in enumerate(subjects):
            lab = i % num_classes
            subj_ids += [s] * cells_each
            X.append(rng.randn(cells_each, n_genes).astype("float32") + lab * 2.0)
            y += [lab] * cells_each
            src += [source] * cells_each
        return (np.vstack(X), np.array(y), np.array(subj_ids, dtype=object), np.array(src, dtype=object))

    train_subj = [f"train_{i}" for i in range(16)]
    val_subj   = [f"val_{i}"   for i in range(8)]
    test_subj  = [f"test_{i}"  for i in range(8)]

    Xtr, ytr, str_, srctr = make_split(train_subj[:10], "sourceA")
    Xtr2, ytr2, str2, srctr2 = make_split(train_subj[10:], "sourceB")
    Xtr = np.concatenate([Xtr, Xtr2]); ytr = np.concatenate([ytr, ytr2])
    str_ = np.concatenate([str_, str2]); srctr = np.concatenate([srctr, srctr2])
    Xva, yva, sva, srcva = make_split(val_subj, "sourceA")
    Xte, yte, ste, srcte = make_split(test_subj, "sourceA")

    n_ct = 4
    train_ds = CellLevelDataset(Xtr, ytr, np.zeros(len(ytr), dtype="float32"),
                                 rng.randint(0, n_ct, len(ytr)), subject_ids=str_, dataset_source=srctr)
    val_ds   = CellLevelDataset(Xva, yva, np.zeros(len(yva), dtype="float32"),
                                 rng.randint(0, n_ct, len(yva)), subject_ids=sva, dataset_source=srcva)
    test_ds  = CellLevelDataset(Xte, yte, np.zeros(len(yte), dtype="float32"),
                                 rng.randint(0, n_ct, len(yte)), subject_ids=ste, dataset_source=srcte)

    def make_bag(sid, label, n=20):
        return {
            "subject_id": sid, "gene_matrix": rng.randn(n, n_genes).astype("float32") + label,
            "cell_type_ids": rng.randint(0, n_ct, n), "smoke_labels": rng.randint(0, num_classes, n),
            "malig_labels": rng.rand(n).astype("float32"), "malig_known": np.zeros(n, dtype=bool),
            "cancer_label": label, "cancer_label_known": True,
        }
    train_bags = [make_bag(s, i % 2) for i, s in enumerate(train_subj)]
    val_bags   = [make_bag(s, i % 2) for i, s in enumerate(val_subj)]
    test_bags  = [make_bag(s, i % 2) for i, s in enumerate(test_subj)]

    mapping = identity_label_mapping({0: "cigarette", 1: "vape", 2: "cannabis"})
    artifact = PreprocessingArtifact(
        version="1", gene_list=[f"g{i}" for i in range(n_genes)],
        gene_means=[0.0] * n_genes, gene_stds=[1.0] * n_genes, n_hvgs=n_genes,
        smoke_marker_genes_forced=[], fit_n_cells=len(ytr), fit_n_subjects=len(train_subj),
    )
    manifest = SplitManifest(seed=seed, train_subjects=train_subj, val_subjects=val_subj, test_subjects=test_subj)

    config = {
        "data": {"min_cells_per_subject": 5},
        "model": {
            "embedding_dim": 16, "attention_dim": 8, "num_cell_types": n_ct,
            # Development/CI-only pathway_hierarchical_mil configuration: a
            # deterministic synthetic module scheme (never a real pathway
            # resource) is explicitly opted into here — see
            # pathway_hierarchical_adapter.build_gene_modules_for_context.
            # Real (non-synthetic) runs must supply gene_modules.path
            # instead; configs/default.yaml's own default leaves
            # allow_synthetic_modules false.
            "pathway_hierarchical_mil": {
                "embedding_dim": 16, "attention_dim": 8, "residual_gene_dim": 8, "dropout": 0.1,
                "num_cell_type_buckets": n_ct + 1,
                "gene_modules": {
                    "path": None, "allow_synthetic_modules": True,
                    "synthetic_seed": 0, "synthetic_n_modules": 4, "synthetic_genes_per_module": 5,
                    "minimum_genes_per_module": 2,
                },
            },
        },
        "train": {"phase1_epochs": 1, "phase2_epochs": 1, "phase1_batch_size": 64,
                  "checkpoint_dir": "checkpoints/benchmarks_synthetic"},
        "benchmarks": {"species_by_source": {"sourceA": "human", "sourceB": "human"},
                       "reference_species": "human"},
    }

    # Pre-HVG, pre-scaling "normalized" AnnData standing in for what
    # run_pipeline_split_aware() captures on real data — required so CV can
    # refit preprocessing per fold (see fold_preprocessing.py) instead of
    # reusing this synthetic context's single outer artifact across folds.
    import anndata as ad
    import pandas as pd
    all_X = np.concatenate([Xtr, Xva, Xte])
    all_y = np.concatenate([ytr, yva, yte])
    all_subj = np.concatenate([str_, sva, ste])
    all_src = np.concatenate([srctr, srcva, srcte])
    all_ct = rng.randint(0, n_ct, len(all_y))
    obs = pd.DataFrame({
        "subject_id": all_subj, "smoke_type": all_y, "smoke_type_known": True, "cell_type_id": all_ct,
        "malignancy": 0.0, "malignancy_known": False,
        "exposure_dose": -1.0, "source": all_src,
    })
    normalized_adata = ad.AnnData(X=all_X.astype("float32"), obs=obs,
                                    var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))

    return ExperimentContext(
        train_cell_dataset=train_ds, val_cell_dataset=val_ds, test_cell_dataset=test_ds,
        train_bags=train_bags, val_bags=val_bags, test_bags=test_bags, split_manifest=manifest,
        preprocessing_artifact=artifact, label_mapping=mapping, rare_class_report={"policy": "keep_with_warning"},
        label_provenance_report={}, transductive_batch_correction=False, config=config, seed=seed,
        dataset_source_summary={
            "train": {"sourceA": 100, "sourceB": 60}, "val": {"sourceA": 80}, "test": {"sourceA": 80},
        },
        normalized_adata_for_refit=normalized_adata,
    )


def build_real_context(config_path: str) -> ExperimentContext:
    import yaml
    from preprocess import run_pipeline_split_aware

    with open(config_path) as f:
        full_config = yaml.safe_load(f)
    result = run_pipeline_split_aware(full_config)
    return ExperimentContext.from_pipeline_result(result, full_config)


def _run_manifest(context: ExperimentContext, seeds: List[int], synthetic: bool, run_id: str) -> dict:
    return {
        "run_id": run_id, "git_sha": context.git_sha, "seeds": seeds,
        "split_fingerprint": context.fingerprint(),
        "run_identity": context.run_identity(run_id),
        "transductive_batch_correction": context.transductive_batch_correction,
        "num_smoke_classes": context.num_smoke_classes,
        "label_mapping": context.label_mapping.to_dict(),
        "dataset_source_summary": context.dataset_source_summary,
        "rare_class_report": context.rare_class_report,
        "label_provenance_report": context.label_provenance_report,
        "synthetic": synthetic,
    }


def _domain_robustness_config_from_args(context, args) -> dict:
    """
    Merge configs/default.yaml's benchmarks.domain_robustness section
    (already-declared defaults: ERM, every regularizer disabled) with
    explicit CLI overrides. CLI flags always win over the config file when
    given, so `--domain-strategy coral --coral-weight 0.1` works even
    against a config whose file only declares the section's defaults.
    Unsupported flag combinations (e.g. --coral-weight without
    --domain-strategy coral) are NOT silently ignored — they are recorded
    in the resolved config and will simply have no effect, which
    resolve_domain_robustness_config's own validation does not reject
    since a declared-but-unused weight is not a contradiction, only a
    likely user mistake; --domain-robustness-ablation runs every strategy
    regardless of --domain-strategy, so this combination is not an error.
    """
    base = dict(context.config.get("benchmarks", {}).get("domain_robustness", {}) or {})
    if args.domain_strategy is not None:
        base["strategy"] = args.domain_strategy
    if args.source_balanced:
        base["source_balancing"] = dict(base.get("source_balancing", {}), enabled=True)
    if args.coral_weight is not None:
        base["coral"] = dict(base.get("coral", {}), enabled=True, weight=args.coral_weight)
    if args.mmd_weight is not None:
        base["mmd"] = dict(base.get("mmd", {}), enabled=True, weight=args.mmd_weight)
    if args.domain_loss_weight is not None and base.get("strategy") == "domain_adversarial":
        base["adversarial"] = dict(base.get("adversarial", {}), enabled=True, weight=args.domain_loss_weight)
    if args.gradient_reversal_lambda is not None:
        base["adversarial"] = dict(base.get("adversarial", {}), gradient_reversal_lambda=args.gradient_reversal_lambda)
    return resolve_domain_robustness_config(base)


def run_smoke_task(context, args, run_dir) -> dict:
    eligibility = {"smoke_classification": check_task_a_eligibility(context)}
    if not eligibility["smoke_classification"].eligible:
        print(f"[benchmarks] Task A NOT_EVALUABLE: {eligibility['smoke_classification'].reasons}")
        return {"eligibility": eligibility, "cv_reports": {}, "comparisons": []}

    cv_report = run_smoke_cv(context, args.models, n_folds=args.cv_folds, seeds=args.seeds, device=args.device,
                              artifact_output_root=str(run_dir))
    comparisons = []
    baseline_ref = "majority" if "majority" in args.models else args.models[0]
    for name in args.models:
        if name == baseline_ref:
            continue
        comparisons.append(compare_models(cv_report["results"], "subject_weighted_macro_f1", name, baseline_ref))

    ood_report = None
    if args.leave_one_source_out:
        ood_models = [m for m in args.models if m in SMOKE_BASELINES]
        bench_cfg = context.config.get("benchmarks", {})
        ood_report = run_leave_one_source_out(
            context, ood_models, device=args.device,
            incompatible_sources=bench_cfg.get("incompatible_sources"),
            species_by_source=bench_cfg.get("species_by_source"),
            reference_species=bench_cfg.get("reference_species"),
        )
        write_json(run_dir / "metrics" / "leave_one_source_out.json", ood_report)

    # Phase 6 — richer source-held-out protocol (both classical baselines
    # and, if requested, pathway_hierarchical_mil), reported separately from
    # the pre-existing --leave-one-source-out path above. Never touches the
    # frozen-test guard (see source_held_out.py's module docstring).
    domain_robustness_report = None
    is_ablation_report = False
    if getattr(args, "domain_strategy", None) is not None or getattr(args, "domain_robustness_ablation", False):
        bench_cfg = context.config.get("benchmarks", {})
        loso_kwargs = dict(
            incompatible_sources=bench_cfg.get("incompatible_sources"),
            species_by_source=bench_cfg.get("species_by_source"),
            reference_species=bench_cfg.get("reference_species"),
            reference_assay_mode=bench_cfg.get("reference_assay_mode"),
        )
        if getattr(args, "domain_robustness_ablation", False):
            domain_robustness_report = run_domain_robustness_ablation(
                context, "smoke", args.models, device=args.device, seed=args.seeds[0], **loso_kwargs,
            )
            is_ablation_report = True
        else:
            requested_strategy = _domain_robustness_config_from_args(context, args)["strategy"]
            if requested_strategy != "erm":
                raise UnsupportedSmokeDomainStrategyError(
                    f"--domain-strategy {requested_strategy!r} was requested for --task smoke, but Task A "
                    "source-held-out evaluation supports ERM only — domain-robust training strategies "
                    "(source_balanced/coral/mmd/domain_adversarial) are cancer-only in this repository. "
                    "Use --task cancer, or omit --domain-strategy (defaults to erm) for smoke."
                )
            per_source = run_smoke_source_held_out(context, args.models, device=args.device, seed=args.seeds[0],
                                                     **loso_kwargs)
            domain_robustness_report = build_aggregate_report(
                "smoke_classification", args.models[0], _domain_robustness_config_from_args(context, args)["strategy"],
                list(per_source.values()), primary_metric="macro_f1",
            )
        # An ablation report has a materially different top-level shape
        # (variants/per-seed results, not a single per_source_reports list)
        # — its own child reports are already validated inside
        # run_domain_robustness_ablation at construction time, so it is
        # written with the generic (but still atomic/reload-verified)
        # write_json. The plain aggregate report goes through
        # write_aggregate_report, which re-validates the WHOLE aggregate
        # (schema/stamps + every nested per-source report) at write time —
        # never persisted through an unvalidated path.
        if is_ablation_report:
            write_json(run_dir / "metrics" / "domain_robustness_smoke.json", domain_robustness_report)
            if getattr(args, "robustness_report", None):
                write_json(Path(args.robustness_report), domain_robustness_report)
        else:
            write_aggregate_report(run_dir / "metrics" / "domain_robustness_smoke.json", domain_robustness_report)
            if getattr(args, "robustness_report", None):
                write_aggregate_report(Path(args.robustness_report), domain_robustness_report)

    imbalance_ablation_report = None
    if getattr(args, "imbalance_ablation", False):
        from .imbalance_ablation import run_smoke_imbalance_ablation, write_imbalance_ablation_artifact
        imbalance_ablation_report = run_smoke_imbalance_ablation(
            context, n_folds=args.cv_folds, seeds=args.seeds, device=args.device,
        )
        write_imbalance_ablation_artifact(
            run_dir, context, imbalance_ablation_report, synthetic=bool(getattr(args, "synthetic", False)),
        )

    pathway_hierarchical_ablation_report = None
    if getattr(args, "pathway_hierarchical_ablation", False):
        pathway_hierarchical_ablation_report = run_pathway_hierarchical_ablation(
            context, n_folds=args.cv_folds, seeds=args.seeds, device=args.device,
        )
        write_pathway_hierarchical_ablation_artifact(
            run_dir, context, pathway_hierarchical_ablation_report, synthetic=bool(getattr(args, "synthetic", False)),
        )

    return {"eligibility": eligibility, "cv_reports": {"smoke_classification": cv_report},
            "comparisons": comparisons, "ood_report": ood_report,
            "imbalance_ablation_report": imbalance_ablation_report,
            "pathway_hierarchical_ablation_report": pathway_hierarchical_ablation_report,
            "domain_robustness_report": domain_robustness_report}


def _final_dev_pool_hyperparameters(context, best_name: str, dev_subjects, outcomes_by_subject,
                                     num_cell_types: int, min_cells: int, n_hvgs: int,
                                     pooling: str, device: str, seed: int) -> dict:
    """
    ONE hyperparameter/config selection for the chosen final candidate,
    computed by nested nested-CV over the WHOLE development pool (train+val
    subjects — never test). This selection is used only for the final
    dev-pool fit (fit_final_candidate_on_dev_pool) — generate_subject_oof_
    predictions performs its OWN fold-local selection per OOF fold instead
    of reusing this one, so an OOF-held-out subject's label never
    influences the configuration used to predict it.
    """
    if best_name == PATHWAY_MODEL_NAME:
        candidates = build_param_grid(pathway_search_space(fast=False))
        fit_score_fn = _pathway_cancer_fit_score_fn(context, device, outcomes_by_subject, min_cells)
    elif is_mil_candidate(best_name):
        candidates = build_param_grid(MIL_SEARCH_SPACE)
        fit_score_fn = _mil_fit_score_fn(context, pooling, device, outcomes_by_subject, min_cells)
    else:
        candidates = build_param_grid(CANCER_SEARCH_SPACE.get(best_name, {}))
        fit_score_fn = _cancer_baseline_fit_score_fn(CANCER_BASELINES[best_name], num_cell_types,
                                                       outcomes_by_subject, min_cells)
    return select_nested_hyperparameters_with_refit(
        context, dev_subjects, outcomes_by_subject, candidates, fit_score_fn=fit_score_fn,
        seed=seed, n_inner_folds=DEFAULT_INNER_FOLDS, n_hvgs=n_hvgs,
    )


_OOF_CSV_COLUMNS = [
    "subject_id", "target", "probability", "seed", "outer_oof_fold", "candidate_name",
    "candidate_type", "pooling", "selected_params_json", "selected_params_fingerprint",
    "inner_selection_fingerprint", "training_subjects_fingerprint", "validation_subjects_fingerprint",
    "preprocessing_fingerprint", "module_fingerprint", "model_state_fingerprint",
    "prediction_status", "undefined_reason",
]


def _write_oof_predictions_csv(run_dir, oof: dict, outcomes_by_subject: dict, candidate_name: str, pooling: str) -> str:
    """
    Persist ONE row per development subject with a validated OOF prediction
    (generate_subject_oof_predictions already rejected missing/duplicate/
    in-fold-leaked coverage before returning `oof`) to
    predictions/cancer_<candidate>_oof.csv — the actual subject-level
    probabilities and every real SHA-256 fingerprint
    generate_subject_oof_predictions already computed per fold (training/
    validation subject-list fingerprints, inner-selection fingerprint,
    selected-params fingerprint, fold model-state fingerprint) — never a
    raw JSON blob under a field named "fingerprint" (issue 4). Writes are
    atomic (write_csv_table/write_json -> atomic_io.py); the file is
    reloaded and its row count verified immediately after writing, and the
    complete-file's own SHA-256 is computed and returned so calibration can
    reference exactly this artifact.
    """
    fold_by_subject = {}
    for fold in oof["fold_membership"]:
        hp = fold.get("hyperparameter_search", {})
        selected_params_json = json.dumps(hp.get("selected_params", {}), sort_keys=True)
        for sid in fold.get("val_subject_ids", []):
            fold_by_subject[sid] = {
                "outer_oof_fold": fold["fold"],
                "preprocessing_fingerprint": fold.get("preprocessing_fingerprint"),
                "selected_params_json": selected_params_json,
                "selected_params_fingerprint": fold.get("selected_params_fingerprint"),
                "inner_selection_fingerprint": fold.get("inner_selection_fingerprint"),
                "training_subjects_fingerprint": fold.get("training_subjects_fingerprint"),
                "validation_subjects_fingerprint": fold.get("validation_subjects_fingerprint"),
                "model_state_fingerprint": fold.get("model_state_fingerprint"),
                "module_fingerprint": fold.get("module_fingerprint"),
            }

    rows = []
    for sid in oof["dev_subjects"]:
        meta = fold_by_subject.get(sid, {})
        proba = oof["oof_by_subject"].get(sid)
        status = "predicted" if proba is not None else "undefined"
        if status == "predicted" and not meta.get("model_state_fingerprint"):
            raise RuntimeError(
                f"_write_oof_predictions_csv: subject {sid!r} has a successful prediction but no "
                "recorded fold model-state fingerprint — refusing to persist an OOF row that "
                "claims a prediction without provenance for the model that produced it."
            )
        row = {col: "" for col in _OOF_CSV_COLUMNS}
        row.update({
            "subject_id": sid, "target": outcomes_by_subject.get(sid),
            "probability": proba, "seed": oof.get("seed"),
            "outer_oof_fold": meta.get("outer_oof_fold"), "candidate_name": candidate_name,
            "candidate_type": "mil" if is_mil_candidate(candidate_name) else "baseline",
            "pooling": pooling if is_mil_candidate(candidate_name) else "",
            "selected_params_json": meta.get("selected_params_json", ""),
            "selected_params_fingerprint": meta.get("selected_params_fingerprint", ""),
            "inner_selection_fingerprint": meta.get("inner_selection_fingerprint", ""),
            "training_subjects_fingerprint": meta.get("training_subjects_fingerprint", ""),
            "validation_subjects_fingerprint": meta.get("validation_subjects_fingerprint", ""),
            "preprocessing_fingerprint": meta.get("preprocessing_fingerprint", ""),
            "module_fingerprint": meta.get("module_fingerprint", "") or "",
            "model_state_fingerprint": meta.get("model_state_fingerprint", ""),
            "prediction_status": status,
            "undefined_reason": "" if proba is not None else "subject never received an OOF prediction",
        })
        rows.append(row)

    # Reject duplicate/missing coverage explicitly here too (belt-and-braces
    # on top of generate_subject_oof_predictions's own checks) before ever
    # writing the file.
    seen_subjects = [r["subject_id"] for r in rows]
    if len(seen_subjects) != len(set(seen_subjects)):
        raise RuntimeError("_write_oof_predictions_csv: duplicate subject rows detected before write.")
    missing = sorted(set(oof["dev_subjects"]) - set(seen_subjects))
    if missing:
        raise RuntimeError(f"_write_oof_predictions_csv: missing rows for development subjects: {missing[:5]}")

    oof_path = run_dir / "predictions" / f"cancer_{candidate_name}_oof.csv"
    write_csv_table(oof_path, rows)
    read_and_verify_csv_rows(oof_path, expected_row_count=len(rows))
    oof_file_fingerprint = hashlib.sha256(oof_path.read_bytes()).hexdigest()

    folds_path = run_dir / "metrics" / "cancer_oof_folds.json"
    write_json(folds_path, oof["fold_membership"])
    read_and_verify_json(folds_path, oof["fold_membership"])

    return oof_file_fingerprint


def run_cancer_task(context, args, run_dir, synthetic: bool = False) -> dict:
    # Development-only eligibility (blocker 1): reads ONLY train_bags/
    # val_bags — check_task_b_development_eligibility's signature doesn't
    # even accept test_bags, so test composition can never influence
    # whether this experiment proceeds. Whether test metrics end up
    # mathematically defined is decided later, by check_test_evaluability,
    # strictly inside the guarded stage below.
    eligibility = {"cancer_prediction": check_task_b_development_eligibility(context.train_bags, context.val_bags)}
    if not eligibility["cancer_prediction"].eligible:
        print(f"[benchmarks] Task B NOT_EVALUABLE: {eligibility['cancer_prediction'].reasons}")
        return {"eligibility": eligibility, "cv_reports": {}, "comparisons": [], "calibration_report": None}

    cv_report = run_cancer_cv(context, args.models, n_folds=args.cv_folds, seeds=args.seeds,
                                device=args.device, pooling=args.pooling, artifact_output_root=str(run_dir))
    comparisons = []
    baseline_ref = "prevalence" if "prevalence" in args.models else args.models[0]
    for name in args.models:
        if name == baseline_ref:
            continue
        comparisons.append(compare_models(cv_report["results"], "auroc", name, baseline_ref))

    # Phase 6 — source-held-out domain robustness (development-only,
    # structurally cannot touch the frozen-test guard below — see
    # source_held_out.py). Runs entirely over the train+val pool, exactly
    # like --leave-one-source-out, just with the richer both-task/all-
    # model-kind protocol and (optionally) domain-robust training.
    domain_robustness_report = None
    is_ablation_report = False
    if getattr(args, "domain_strategy", None) is not None or getattr(args, "domain_robustness_ablation", False):
        bench_cfg = context.config.get("benchmarks", {})
        loso_kwargs = dict(
            incompatible_sources=bench_cfg.get("incompatible_sources"),
            species_by_source=bench_cfg.get("species_by_source"),
            reference_species=bench_cfg.get("reference_species"),
            reference_assay_mode=bench_cfg.get("reference_assay_mode"),
        )
        if getattr(args, "domain_robustness_ablation", False):
            domain_robustness_report = run_domain_robustness_ablation(
                context, "cancer", args.models, device=args.device, seed=args.seeds[0], **loso_kwargs,
            )
            is_ablation_report = True
        else:
            domain_cfg = _domain_robustness_config_from_args(context, args)
            per_source = run_cancer_source_held_out(
                context, args.models, device=args.device, domain_robustness_config=domain_cfg,
                seed=args.seeds[0], stability_extra_seeds=getattr(args, "stability_extra_seeds", None) or (),
                **loso_kwargs,
            )
            domain_robustness_report = build_aggregate_report(
                "cancer_prediction", args.models[0], domain_cfg["strategy"],
                list(per_source.values()), primary_metric="auroc",
            )
        # See run_smoke_task's identical branch for why ablation reports
        # (a different top-level shape) use write_json while the plain
        # aggregate goes through write_aggregate_report's full recursive
        # validation at write time.
        if is_ablation_report:
            write_json(run_dir / "metrics" / "domain_robustness_cancer.json", domain_robustness_report)
            if getattr(args, "robustness_report", None):
                write_json(Path(args.robustness_report), domain_robustness_report)
        else:
            write_aggregate_report(run_dir / "metrics" / "domain_robustness_cancer.json", domain_robustness_report)
            if getattr(args, "robustness_report", None):
                write_aggregate_report(Path(args.robustness_report), domain_robustness_report)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE A — development-only. Every call below may read train_bags/
    # val_bags and CV-report evidence, but NEVER context.test_bags, NEVER
    # context.subjects_for("test"), NEVER a test label/probability. This is
    # enforced structurally: select_final_candidate/generate_subject_oof_
    # predictions/fit_final_candidate_on_dev_pool (final_evaluation.py) do
    # not accept test data as an argument at all.
    # ══════════════════════════════════════════════════════════════════════
    num_cell_types = context.config.get("model", {}).get("num_cell_types", 4)
    n_hvgs = context.preprocessing_artifact.n_hvgs
    min_cells = context.config.get("data", context.config).get("min_cells_per_subject", 50)
    all_dev_bags = list(context.train_bags) + list(context.val_bags)
    outcomes_by_subject = {str(b["subject_id"]): b["cancer_label"] for b in all_dev_bags if b.get("cancer_label_known")}
    dev_subjects = sorted(outcomes_by_subject.keys())
    pooling_for = {"mean_mil": "mean", "max_mil": "max", "attention_mil": "attention",
                   "neural": args.pooling or "attention"}

    try:
        best_name, selection_report = select_final_candidate(cv_report, args.models, primary_metric="auroc")
    except NoEligibleFinalCandidateError as e:
        return {"eligibility": eligibility, "cv_reports": {"cancer_prediction": cv_report},
                "comparisons": comparisons,
                "calibration_report": {"selected_model": None, "error": str(e)},
                "domain_robustness_report": domain_robustness_report}

    pooling = pooling_for.get(best_name, args.pooling or "attention")
    seed = args.seeds[0]

    hp_search = _final_dev_pool_hyperparameters(
        context, best_name, dev_subjects, outcomes_by_subject,
        num_cell_types, min_cells, n_hvgs, pooling, args.device, seed,
    )
    selected_params = hp_search["selected_params"]

    # Subject-grouped, SELECTION-CLEAN OOF predictions across the WHOLE
    # development pool: each OOF fold selects its OWN hyperparameters from
    # only its OOF-training subjects (blocker: no global-selection leakage
    # into the configuration used to predict a held-out subject). Calibration
    # /threshold below are fit exclusively from these, never from a plain
    # validation split and never from test.
    oof = generate_subject_oof_predictions(
        context, best_name, dev_subjects, outcomes_by_subject,
        num_cell_types, min_cells, n_hvgs, pooling=pooling, device=args.device,
        seed=seed, n_folds=args.cv_folds,
    )
    y_oof = np.array([outcomes_by_subject[s] for s in oof["dev_subjects"]])
    prob_oof = np.array([oof["oof_by_subject"][s] for s in oof["dev_subjects"]])
    policy = build_frozen_policy(y_oof, prob_oof, calibration_method=args.calibration,
                                  threshold_strategy=args.threshold_strategy)

    # Persist the ACTUAL subject-level OOF predictions (not just fold
    # membership counts) — one row per development subject, atomically
    # written once fold_membership/oof_by_subject have both already passed
    # generate_subject_oof_predictions's own coverage/leakage validation.
    oof_file_fingerprint = _write_oof_predictions_csv(run_dir, oof, outcomes_by_subject, best_name, pooling)

    # ONE final preprocessing + model fit on ALL development subjects, using
    # the dev-pool-selected configuration (never reselected here). This is
    # the last development-only step — nothing past this point in Stage A
    # reads test data, and `fitted` carries no test-derived information.
    fitted = fit_final_candidate_on_dev_pool(
        context, best_name, dev_subjects, outcomes_by_subject, selected_params,
        num_cell_types, min_cells, n_hvgs, pooling=pooling, device=args.device, seed=seed,
    )

    # ══════════════════════════════════════════════════════════════════════
    # STAGE B — guarded frozen-test execution. The durable test guard is
    # MANDATORY for every non-synthetic run: a safe default location is
    # derived from this run's own output root when
    # benchmarks.frozen_test_guard_dir isn't configured, so a real run can
    # never accidentally proceed without one. Disabling it is permitted ONLY
    # through the explicit synthetic-only path (repeated CI/test invocations
    # against the same synthetic context are an intentional, expected
    # pattern); requesting that disable in a non-synthetic run is refused
    # outright, not silently honored. Test subject IDs, test bags, test
    # labels, and test predictions are resolved for the FIRST time only
    # inside the try block below, strictly after guard.acquire() succeeds.
    # ══════════════════════════════════════════════════════════════════════
    # Refuse to even approach guard acquisition if this run's outer artifact
    # was produced under transductive (disclosed, non-leakage-free) batch
    # correction — the one-shot frozen-test guarantee is meaningless if
    # held-out expression already influenced a shared batch-correction
    # embedding before the guard was ever acquired. See data/transforms.py::
    # assert_batch_correction_safe / UnsafeBatchCorrectionError.
    assert_batch_correction_safe(context.transductive_batch_correction, context_name="runner.run (frozen-test stage)")

    bench_cfg = context.config.get("benchmarks", {})
    disable_guard = bool(bench_cfg.get("disable_frozen_test_guard", False))
    if disable_guard and not synthetic:
        raise FrozenTestGuardDisabledInRealModeError(
            "benchmarks.disable_frozen_test_guard=True is only permitted for --synthetic runs — "
            "refusing to disable the durable frozen-test guard for this non-synthetic run."
        )
    # Deterministic, non-volatile calibration/model fingerprints for the
    # guard identity (blocker 2) — no fit_seconds, no timestamps, no
    # parameter counts standing in for actual fitted weights.
    calibration_fingerprint = hashlib.sha256(
        json.dumps(policy.calibrator.to_dict(), sort_keys=True, default=str).encode()
    ).hexdigest()
    # test_membership_fingerprint reads ONLY split_manifest.test_subjects
    # (never test_bags/labels) — safe to compute before guard acquisition,
    # and included so a changed frozen test membership changes identity.
    test_membership_fp = context.test_membership_fingerprint

    guard = None
    identity_fp = None
    if not (synthetic and disable_guard):
        guard_dir = bench_cfg.get("frozen_test_guard_dir") or default_guard_dir(run_dir.parent)
        identity_fp = context.guard_identity_fingerprint(best_name, extra={
            "selected_hyperparameters": selected_params,
            "final_preprocessing_artifact_fingerprint": fitted.preprocessing_artifact_fingerprint,
            "final_model_state_fingerprint": fitted.model_state_fingerprint,
            "calibration_fingerprint": calibration_fingerprint,
            "threshold": policy.threshold,
            "test_membership_fingerprint": test_membership_fp,
        })
        guard = FrozenTestGuard(Path(guard_dir) / f"{identity_fp}.json")
        guard.acquire(context.run_identity(run_dir.name), selected_model=best_name)

    try:
        # ══════════════════════════════════════════════════════════════
        # The complete guarded transaction (blocker 3): resolve test
        # membership/labels, transform+predict, calibrate/threshold,
        # compute metrics, build the immutable frozen-test result object,
        # persist it atomically, reload and verify it, and ONLY THEN mark
        # the guard completed. Any failure at any step below marks the
        # guard failed instead (the `except` clause) — mark_completed is
        # never reached unless persistence+verification both succeeded.
        # ══════════════════════════════════════════════════════════════
        test_subjects = context.subjects_for("test")
        test_outcomes_by_subject = {str(b["subject_id"]): b["cancer_label"]
                                     for b in context.test_bags if b.get("cancer_label_known")}
        test_evaluability = check_test_evaluability(context.test_bags)
        raw = evaluate_frozen_test(fitted, context, test_subjects, test_outcomes_by_subject,
                                    num_cell_types, min_cells)
        test_result = policy.apply_to_test(raw["test_labels"], raw["test_proba"])

        frozen_result = {
            "scientific_identity_fingerprint": identity_fp,
            "selected_model": best_name,
            "candidate_type": "mil" if is_mil_candidate(best_name) else "baseline",
            "model_state_fingerprint": fitted.model_state_fingerprint,
            "preprocessing_fingerprint": fitted.preprocessing_artifact_fingerprint,
            "calibration_fingerprint": calibration_fingerprint,
            "threshold": policy.threshold,
            "metrics": test_result,
            "test_evaluability": test_evaluability,
            "test_membership_fingerprint": test_membership_fp,
            "n_evaluated_subjects": len(raw["test_subject_ids"]),
            "synthetic": synthetic,
        }
        frozen_result["artifact_fingerprint"] = hashlib.sha256(
            json.dumps(frozen_result, sort_keys=True, default=str).encode()
        ).hexdigest()

        result_path = run_dir / "calibration" / "frozen_test_result.json"
        write_json(result_path, frozen_result)
        read_and_verify_json(result_path, frozen_result)
    except Exception as e:
        if guard is not None:
            guard.mark_failed(str(e))
        raise
    if guard is not None:
        guard.mark_completed(threshold=policy.threshold,
                              test_result_fingerprint=frozen_result["artifact_fingerprint"])

    calibration_report = {
        "selected_model": best_name, "selection_report": selection_report,
        "hyperparameter_search": hp_search, "test_result": test_result,
        "test_evaluability": test_evaluability,
        "oof_summary": {
            "candidate": oof["candidate"], "n_dev_subjects": len(oof["dev_subjects"]),
            "fold_membership": oof["fold_membership"],
            "oof_artifact_fingerprint": oof_file_fingerprint,
        },
        "final_model_metadata": fitted.model_metadata,
        "final_model_state_fingerprint": fitted.model_state_fingerprint,
        "final_preprocessing_artifact_fingerprint": fitted.preprocessing_artifact_fingerprint,
        "test_subject_ids": raw["test_subject_ids"],
        "frozen_test_result_artifact_fingerprint": frozen_result["artifact_fingerprint"],
        "frozen_test_result_path": str(run_dir / "calibration" / "frozen_test_result.json"),
        "guard": {"enabled": guard is not None,
                   "path": str(guard.guard_path) if guard is not None else None,
                   "synthetic_disabled": bool(synthetic and disable_guard)},
    }

    return {"eligibility": eligibility, "cv_reports": {"cancer_prediction": cv_report},
            "comparisons": comparisons, "calibration_report": calibration_report,
            "domain_robustness_report": domain_robustness_report}


class DomainRobustnessCLIConfigurationError(ValueError):
    """A Phase 6 domain-robustness CLI flag was supplied in a combination
    that would otherwise have no effect on the resolved configuration or
    execution path — e.g. --coral-weight without --domain-strategy coral.
    Rejected explicitly rather than silently accepted and ignored (see
    _domain_robustness_config_from_args' own docstring for the one
    deliberate exception: --domain-robustness-ablation runs every strategy
    regardless of --domain-strategy)."""


def _validate_domain_robustness_cli_flags(args) -> None:
    ablation = getattr(args, "domain_robustness_ablation", False)
    strategy = getattr(args, "domain_strategy", None)
    if getattr(args, "coral_weight", None) is not None and not ablation and strategy != "coral":
        raise DomainRobustnessCLIConfigurationError(
            "--coral-weight has no effect without --domain-strategy coral (or --domain-robustness-ablation)."
        )
    if getattr(args, "mmd_weight", None) is not None and not ablation and strategy != "mmd":
        raise DomainRobustnessCLIConfigurationError(
            "--mmd-weight has no effect without --domain-strategy mmd (or --domain-robustness-ablation)."
        )
    if (getattr(args, "domain_loss_weight", None) is not None
            and not ablation and strategy != "domain_adversarial"):
        raise DomainRobustnessCLIConfigurationError(
            "--domain-loss-weight has no effect without --domain-strategy domain_adversarial "
            "(or --domain-robustness-ablation)."
        )
    if (getattr(args, "gradient_reversal_lambda", None) is not None
            and not ablation and strategy != "domain_adversarial"):
        raise DomainRobustnessCLIConfigurationError(
            "--gradient-reversal-lambda has no effect without --domain-strategy domain_adversarial "
            "(or --domain-robustness-ablation)."
        )
    if getattr(args, "source_balanced", False) and not ablation and strategy != "source_balanced":
        raise DomainRobustnessCLIConfigurationError(
            "--source-balanced has no effect without --domain-strategy source_balanced "
            "(or --domain-robustness-ablation)."
        )
    if getattr(args, "robustness_report", None) is not None and strategy is None and not ablation:
        raise DomainRobustnessCLIConfigurationError(
            "--robustness-report has no effect without --domain-strategy or --domain-robustness-ablation "
            "— there is no robustness workflow active to write a report for."
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--task", choices=["smoke", "cancer"], default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", type=str, default="artifacts/benchmarks")
    parser.add_argument("--calibration", type=str, default="auto")
    parser.add_argument("--threshold-strategy", type=str, default="youden")
    parser.add_argument("--pooling", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--leave-one-source-out", action="store_true")
    # Development-only comparison of Phase 2 smoke-imbalance strategies
    # (benchmarks/imbalance_ablation.py) — only meaningful for --task smoke.
    # Runs entirely over the train+val subject pool; never touches test
    # data or the frozen-test guard. Writes its own artifact under
    # <run_dir>/metrics/smoke_imbalance_ablation.json.
    parser.add_argument("--imbalance-ablation", action="store_true")
    # Development-only comparison of pathway_hierarchical_mil architecture
    # variants against each other and against the existing gated-attention
    # MIL baseline (benchmarks/pathway_hierarchical_ablation.py) — only
    # meaningful for --task smoke (it runs its own cancer-task-shaped CV
    # internally so both tasks' metrics come from one fit per fold; see
    # that module's docstring). Never touches test data or the frozen-test
    # guard. Writes its own artifact under
    # <run_dir>/metrics/pathway_hierarchical_ablation.json.
    parser.add_argument("--pathway-hierarchical-ablation", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    # The frozen-test guard is mandatory for every non-synthetic run — this
    # flag exists ONLY to make repeated CI/test invocations against the same
    # synthetic context (an intentional, expected pattern, not a real
    # publication run) skip the durable guard. Using it without --synthetic
    # is refused by run_cancer_task, not silently honored.
    parser.add_argument("--disable-frozen-test-guard", action="store_true")

    # ── Phase 6: source-held-out domain robustness ─────────────────────
    # --leave-one-source-out (above) is the pre-existing Task A classical-
    # baseline-only external-domain diagnostic (benchmarks/ood.py) and
    # remains unchanged. --domain-strategy (and any of the flags below it)
    # additionally runs the richer, both-task, all-model-kind source-held-
    # out protocol (benchmarks/source_held_out.py) — a development-only
    # DIAGNOSTIC EVALUATION of external-domain robustness, structurally
    # separate from, and never consuming, the one-shot frozen final test
    # guard (test_guard.py) evaluated later in this same run.
    parser.add_argument("--domain-strategy", type=str, default=None,
                         choices=["erm", "source_balanced", "coral", "mmd", "domain_adversarial"])
    parser.add_argument("--domain-robustness-ablation", action="store_true")
    parser.add_argument("--source-balanced", action="store_true")
    parser.add_argument("--coral-weight", type=float, default=None)
    parser.add_argument("--mmd-weight", type=float, default=None)
    parser.add_argument("--domain-loss-weight", type=float, default=None)
    parser.add_argument("--gradient-reversal-lambda", type=float, default=None)
    parser.add_argument("--robustness-report", type=str, default=None,
                         help="Path to write the aggregate cross-source robustness report JSON.")
    parser.add_argument("--stability-extra-seeds", nargs="+", type=int, default=None,
                         help="Extra seeds for GENUINE independent pathway_hierarchical_mil refits "
                              "used to compute biological_stability.cross_run_stability. Left empty "
                              "by default — each extra seed is a full extra model fit — in which case "
                              "cross_run_stability reports insufficient_evidence.")
    args = parser.parse_args(argv)
    _validate_domain_robustness_cli_flags(args)

    if args.synthetic:
        if args.task is None:
            args.task = "smoke"
        if args.models is None:
            args.models = ["majority", "logistic", "random_forest"] if args.task == "smoke" \
                else ["prevalence", "logistic", "random_forest"]
        if args.fast:
            args.cv_folds = min(args.cv_folds, 3)
            args.seeds = args.seeds[:1]
        context = build_synthetic_context(seed=args.seeds[0], fast=args.fast)
        synthetic = True
        if args.disable_frozen_test_guard:
            context.config.setdefault("benchmarks", {})["disable_frozen_test_guard"] = True
    else:
        if not args.config or not args.task or not args.models:
            parser.error("--config, --task, and --models are required unless --synthetic is given")
        if args.disable_frozen_test_guard:
            parser.error("--disable-frozen-test-guard is only permitted together with --synthetic")
        context = build_real_context(args.config)
        synthetic = False

    run_dir = new_run_dir(args.output, run_id=args.run_id)
    run_manifest = _run_manifest(context, args.seeds, synthetic, run_dir.name)

    if args.task == "smoke":
        outcome = run_smoke_task(context, args, run_dir)
    else:
        outcome = run_cancer_task(context, args, run_dir, synthetic=synthetic)

    write_benchmark_report(
        run_dir, context, run_manifest, outcome["eligibility"], outcome["cv_reports"],
        outcome["comparisons"], outcome.get("calibration_report"), synthetic=synthetic,
    )
    print(f"[benchmarks] done -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
