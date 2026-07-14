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
from pathlib import Path
from typing import List, Optional

import numpy as np

from .baselines import CANCER_BASELINES, CANCER_SEARCH_SPACE, SMOKE_BASELINES
from .calibration import build_frozen_policy
from .context import ExperimentContext
from .cross_validation import (
    DEFAULT_INNER_FOLDS,
    MIL_SEARCH_SPACE,
    _cancer_baseline_fit_score_fn,
    _mil_fit_score_fn,
    run_cancer_cv,
    run_smoke_cv,
)
from .eligibility import check_task_a_eligibility, check_task_b_eligibility
from .final_evaluation import (
    NoEligibleFinalCandidateError,
    generate_subject_oof_predictions,
    is_mil_candidate,
    refit_final_candidate_on_dev_pool,
    select_final_candidate,
)
from .hyperparameter_search import build_param_grid, select_nested_hyperparameters_with_refit
from .reporting import compare_models, new_run_dir, write_benchmark_report
from .ood import run_leave_one_source_out
from .test_guard import FrozenTestGuard, FrozenTestGuardDisabledInRealModeError, default_guard_dir


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
        "model": {"embedding_dim": 16, "attention_dim": 8, "num_cell_types": n_ct},
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
        "subject_id": all_subj, "smoke_type": all_y, "cell_type_id": all_ct,
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


def run_smoke_task(context, args, run_dir) -> dict:
    eligibility = {"smoke_classification": check_task_a_eligibility(context)}
    if not eligibility["smoke_classification"].eligible:
        print(f"[benchmarks] Task A NOT_EVALUABLE: {eligibility['smoke_classification'].reasons}")
        return {"eligibility": eligibility, "cv_reports": {}, "comparisons": []}

    cv_report = run_smoke_cv(context, args.models, n_folds=args.cv_folds, seeds=args.seeds, device=args.device)
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
        from .reporting import write_json
        write_json(run_dir / "metrics" / "leave_one_source_out.json", ood_report)

    return {"eligibility": eligibility, "cv_reports": {"smoke_classification": cv_report},
            "comparisons": comparisons, "ood_report": ood_report}


def _final_dev_pool_hyperparameters(context, best_name: str, dev_subjects, outcomes_by_subject,
                                     num_cell_types: int, min_cells: int, n_hvgs: int,
                                     pooling: str, device: str, seed: int) -> dict:
    """
    ONE hyperparameter/config selection for the chosen final candidate,
    computed by nested nested-CV over the WHOLE development pool (train+val
    subjects — never test). This selection happens exactly once; both
    generate_subject_oof_predictions and refit_final_candidate_on_dev_pool
    reuse its result rather than re-searching.
    """
    if is_mil_candidate(best_name):
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


def run_cancer_task(context, args, run_dir, synthetic: bool = False) -> dict:
    eligibility = {"cancer_prediction": check_task_b_eligibility(context)}
    if not eligibility["cancer_prediction"].eligible:
        print(f"[benchmarks] Task B NOT_EVALUABLE: {eligibility['cancer_prediction'].reasons}")
        return {"eligibility": eligibility, "cv_reports": {}, "comparisons": [], "calibration_report": None}

    cv_report = run_cancer_cv(context, args.models, n_folds=args.cv_folds, seeds=args.seeds,
                                device=args.device, pooling=args.pooling)
    comparisons = []
    baseline_ref = "prevalence" if "prevalence" in args.models else args.models[0]
    for name in args.models:
        if name == baseline_ref:
            continue
        comparisons.append(compare_models(cv_report["results"], "auroc", name, baseline_ref))

    # ── Final one-shot frozen-test evaluation ────────────────────────────
    # Candidates are ranked using CV/development evidence ONLY (blocker 2):
    # a classical baseline and a neural/MIL model compete on exactly the
    # same footing, and the winner is never silently swapped for a baseline.
    num_cell_types = context.config.get("model", {}).get("num_cell_types", 4)
    n_hvgs = context.preprocessing_artifact.n_hvgs
    min_cells = context.config.get("data", context.config).get("min_cells_per_subject", 50)
    all_dev_bags = list(context.train_bags) + list(context.val_bags)
    outcomes_by_subject = {str(b["subject_id"]): b["cancer_label"] for b in all_dev_bags if b.get("cancer_label_known")}
    dev_subjects = sorted(outcomes_by_subject.keys())
    # Test outcomes are needed ONLY by the single sanctioned final test
    # evaluation below (refit_final_candidate_on_dev_pool's test-side
    # lookup) — never merged into outcomes_by_subject/dev_subjects, so no
    # selection, search, or OOF-generation code above can see them.
    test_outcomes_by_subject = {str(b["subject_id"]): b["cancer_label"]
                                 for b in context.test_bags if b.get("cancer_label_known")}
    all_known_outcomes = {**outcomes_by_subject, **test_outcomes_by_subject}
    pooling_for = {"mean_mil": "mean", "max_mil": "max", "attention_mil": "attention",
                   "neural": args.pooling or "attention"}

    try:
        best_name, selection_report = select_final_candidate(cv_report, args.models, primary_metric="auroc")
    except NoEligibleFinalCandidateError as e:
        return {"eligibility": eligibility, "cv_reports": {"cancer_prediction": cv_report},
                "comparisons": comparisons,
                "calibration_report": {"selected_model": None, "error": str(e)}}

    pooling = pooling_for.get(best_name, args.pooling or "attention")
    seed = args.seeds[0]

    hp_search = _final_dev_pool_hyperparameters(
        context, best_name, dev_subjects, outcomes_by_subject,
        num_cell_types, min_cells, n_hvgs, pooling, args.device, seed,
    )
    selected_params = hp_search["selected_params"]

    # Subject-grouped OOF predictions across the WHOLE development pool,
    # using the ALREADY-selected configuration — calibration/threshold below
    # are fit exclusively from these, never from a plain validation split
    # and never from test (blocker 5).
    oof = generate_subject_oof_predictions(
        context, best_name, dev_subjects, outcomes_by_subject, selected_params,
        num_cell_types, min_cells, n_hvgs, pooling=pooling, device=args.device,
        seed=seed, n_folds=args.cv_folds,
    )
    y_oof = np.array([outcomes_by_subject[s] for s in oof["dev_subjects"]])
    prob_oof = np.array([oof["oof_by_subject"][s] for s in oof["dev_subjects"]])
    policy = build_frozen_policy(y_oof, prob_oof, calibration_method=args.calibration,
                                  threshold_strategy=args.threshold_strategy)

    # ONE final preprocessing + model fit on ALL development subjects, using
    # the same already-selected configuration — never reselected here.
    final = refit_final_candidate_on_dev_pool(
        context, best_name, dev_subjects, context.subjects_for("test"), all_known_outcomes,
        selected_params, num_cell_types, min_cells, n_hvgs, pooling=pooling, device=args.device, seed=seed,
    )

    # Durable one-time test guard — MANDATORY for every non-synthetic run
    # (blocker 3). A safe default location is derived from this run's own
    # output root when benchmarks.frozen_test_guard_dir isn't configured, so
    # a real run can never accidentally proceed without one. Disabling it is
    # permitted ONLY through the explicit synthetic-only path (repeated
    # CI/test invocations against the same synthetic context are an
    # intentional, expected pattern); requesting that disable in a
    # non-synthetic run is refused outright, not silently honored.
    bench_cfg = context.config.get("benchmarks", {})
    disable_guard = bool(bench_cfg.get("disable_frozen_test_guard", False))
    if disable_guard and not synthetic:
        raise FrozenTestGuardDisabledInRealModeError(
            "benchmarks.disable_frozen_test_guard=True is only permitted for --synthetic runs — "
            "refusing to disable the durable frozen-test guard for this non-synthetic run."
        )
    guard = None
    if not (synthetic and disable_guard):
        guard_dir = bench_cfg.get("frozen_test_guard_dir") or default_guard_dir(run_dir.parent)
        identity_fp = context.guard_identity_fingerprint(best_name)
        guard = FrozenTestGuard(Path(guard_dir) / f"{identity_fp}.json")
        guard.acquire(context.run_identity(run_dir.name), selected_model=best_name)

    try:
        test_result = policy.apply_to_test(final["test_labels"], final["test_proba"])
    except Exception as e:
        if guard is not None:
            guard.mark_failed(str(e))
        raise
    if guard is not None:
        result_fp = hashlib.sha256(str(sorted(test_result.items())).encode()).hexdigest()
        guard.mark_completed(threshold=policy.threshold, test_result_fingerprint=result_fp)

    calibration_report = {
        "selected_model": best_name, "selection_report": selection_report,
        "hyperparameter_search": hp_search, "test_result": test_result,
        "oof_summary": {
            "candidate": oof["candidate"], "n_dev_subjects": len(oof["dev_subjects"]),
            "fold_membership": oof["fold_membership"],
        },
        "final_model_metadata": final["model_metadata"],
        "final_preprocessing_artifact_fingerprint": final["preprocessing_artifact_fingerprint"],
        "test_subject_ids": final["test_subject_ids"],
        "guard": {"enabled": guard is not None,
                   "path": str(guard.guard_path) if guard is not None else None,
                   "synthetic_disabled": bool(synthetic and disable_guard)},
    }

    return {"eligibility": eligibility, "cv_reports": {"cancer_prediction": cv_report},
            "comparisons": comparisons, "calibration_report": calibration_report}


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
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    # The frozen-test guard is mandatory for every non-synthetic run — this
    # flag exists ONLY to make repeated CI/test invocations against the same
    # synthetic context (an intentional, expected pattern, not a real
    # publication run) skip the durable guard. Using it without --synthetic
    # is refused by run_cancer_task, not silently honored.
    parser.add_argument("--disable-frozen-test-guard", action="store_true")
    args = parser.parse_args(argv)

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
