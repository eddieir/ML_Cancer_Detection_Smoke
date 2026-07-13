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
from pathlib import Path
from typing import List, Optional

import numpy as np

from .baselines import CANCER_BASELINES, SMOKE_BASELINES, positive_class_proba
from .calibration import build_frozen_policy
from .context import ExperimentContext
from .cross_validation import run_cancer_cv, run_smoke_cv
from .eligibility import check_task_a_eligibility, check_task_b_eligibility
from .features import build_cancer_subject_features
from .reporting import compare_models, new_run_dir, write_benchmark_report
from .ood import run_leave_one_source_out


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
        "benchmarks": {"species_by_source": {"sourceA": "human", "sourceB": "human"}},
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
        )
        from .reporting import write_json
        write_json(run_dir / "metrics" / "leave_one_source_out.json", ood_report)

    return {"eligibility": eligibility, "cv_reports": {"smoke_classification": cv_report},
            "comparisons": comparisons, "ood_report": ood_report}


def run_cancer_task(context, args, run_dir) -> dict:
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

    # Final frozen-threshold test evaluation — baseline models only in this
    # Phase 1 CLI (see reporting.generate_markdown_report's limitations
    # section): wiring the neural/MIL adapter into the same one-shot frozen
    # test path is documented follow-up work, not silently skipped.
    calibration_report = None
    baseline_models = [m for m in args.models if m in CANCER_BASELINES]
    if baseline_models:
        best_name = max(
            baseline_models,
            key=lambda n: cv_report["results"][n]["auroc"]["mean"] if cv_report["results"][n]["auroc"]["mean"] is not None else -1,
        )
        num_cell_types = context.config.get("model", {}).get("num_cell_types", 4)
        Xtr, ytr, _, _ = build_cancer_subject_features(context.train_bags, num_cell_types)
        Xva, yva, _, _ = build_cancer_subject_features(context.val_bags, num_cell_types)
        Xte, yte, _, _ = build_cancer_subject_features(context.test_bags, num_cell_types)

        model = CANCER_BASELINES[best_name]()
        model.fit(Xtr, ytr, seed=args.seeds[0])
        prob_val = positive_class_proba(model, Xva)
        policy = build_frozen_policy(yva, prob_val, calibration_method=args.calibration,
                                      threshold_strategy=args.threshold_strategy)
        prob_test = positive_class_proba(model, Xte)
        test_result = policy.apply_to_test(yte, prob_test)
        calibration_report = {"selected_model": best_name, "test_result": test_result}

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
    else:
        if not args.config or not args.task or not args.models:
            parser.error("--config, --task, and --models are required unless --synthetic is given")
        context = build_real_context(args.config)
        synthetic = False

    run_dir = new_run_dir(args.output, run_id=args.run_id)
    run_manifest = _run_manifest(context, args.seeds, synthetic, run_dir.name)

    if args.task == "smoke":
        outcome = run_smoke_task(context, args, run_dir)
    else:
        outcome = run_cancer_task(context, args, run_dir)

    write_benchmark_report(
        run_dir, context, run_manifest, outcome["eligibility"], outcome["cv_reports"],
        outcome["comparisons"], outcome.get("calibration_report"), synthetic=synthetic,
    )
    print(f"[benchmarks] done -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
