"""
benchmarks/pathway_hierarchical_ablation.py — development-only comparison
of pathway_hierarchical_mil architecture variants against each other and
against the existing gated-attention MIL baseline (NeuralCancerAdapter,
pooling="attention").

Reuses the SAME grouped-subject cancer-task CV machinery
cross_validation.py's run_cancer_cv already uses (data/splitting.py::
grouped_kfold, fold_preprocessing.py's per-fold artifact refit), so every
variant sees identical outer folds, identical seeds, and identical
preprocessing boundaries. CV always runs over context's TRAIN+VAL subject
pool; the frozen test split is never touched here.

Because pathway_hierarchical_mil produces both a cancer prediction and a
smoke prediction from ONE fit, each variant's fold record carries both
task's metrics from the SAME fitted model — cancer AUROC/AUPRC (primary,
subject-level) and smoke Macro-F1 (secondary, restricted to subjects with
a known majority smoke label). This is what makes the "single_task_cancer"
vs "single_task_smoke" vs "full_multitask" comparison meaningful without a
second CV loop: the only thing that differs between those three variants is
smoke_loss_weight/cancer_loss_weight, and both tasks' metrics are read off
the same fold fit.

Variants included here are restricted to what this Phase 5 pass actually
implemented — see ARCHITECTURE.md §15.7 for the explicit list of what was
NOT implemented (an adversarial domain-training head, in particular) and
is therefore absent from this ablation rather than faked.

All results produced by this module are software-correctness checks over
synthetic or real development data, never a claim of scientific superiority
— see `run_pathway_hierarchical_ablation`'s returned `note` field.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from data.splitting import grouped_kfold
from train import MILEligibilityError, SubjectLevelDataset, check_mil_eligibility, validate_experiment_partitions

from .atomic_io import atomic_write_csv_rows, atomic_write_json
from .context import get_git_sha
from .cross_validation import DEFAULT_SEEDS, _fold_context
from .fold_preprocessing import artifact_fingerprint, bags_from_fold_cell_dataset, fold_train_val_datasets, require_normalized_adata
from .metrics import aggregate_metric, cancer_prediction_metrics, full_smoke_metrics_report
from .mil_registry import build_mil_adapter
from .neural import NeuralCancerAdapter, count_parameters
from .pathway_hierarchical_adapter import MODEL_NAME as PATHWAY_MODEL_NAME
from .reporting import compare_models, summarize_comparison, write_environment_artifact

ABLATION_ARTIFACT_SCHEMA_VERSION = 1

# Every variant is a COMPLETE, self-consistent spec — never a partial diff
# against another variant, so results are never contaminated by a stale
# key left over from a previously-run variant.
PATHWAY_ABLATION_VARIANTS: Dict[str, Dict] = {
    "existing_attention_mil": {"kind": "baseline_mil"},
    "full_multitask": {"kind": "pathway", "config_overrides": {}},
    "no_gene_residual": {"kind": "pathway", "config_overrides": {"use_gene_residual": False}},
    "no_cell_type_embedding": {"kind": "pathway", "config_overrides": {"use_cell_type_embedding": False}},
    "single_task_cancer": {"kind": "pathway", "config_overrides": {"smoke_loss_weight": 0.0, "cancer_loss_weight": 1.0}},
    "single_task_smoke": {"kind": "pathway", "config_overrides": {"smoke_loss_weight": 1.0, "cancer_loss_weight": 0.0}},
}
BASELINE_VARIANT = "existing_attention_mil"


def run_pathway_hierarchical_ablation(
    context,
    variants: Optional[Sequence[str]] = None,
    n_folds: int = 5,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    device: str = "cpu",
) -> Dict:
    variants = list(variants) if variants is not None else list(PATHWAY_ABLATION_VARIANTS)
    unknown = set(variants) - set(PATHWAY_ABLATION_VARIANTS)
    if unknown:
        raise ValueError(
            f"run_pathway_hierarchical_ablation: unknown variant name(s) {sorted(unknown)} — "
            f"must be a subset of {sorted(PATHWAY_ABLATION_VARIANTS)}"
        )
    if not variants:
        raise ValueError("run_pathway_hierarchical_ablation: variants must be non-empty.")

    normalized_adata = require_normalized_adata(context)
    all_bags = list(context.train_bags) + list(context.val_bags)
    outcomes_by_subject = {str(b["subject_id"]): b["cancer_label"] for b in all_bags if b.get("cancer_label_known")}
    known_subjects = sorted(outcomes_by_subject.keys())
    if len(known_subjects) < 2:
        raise ValueError("run_pathway_hierarchical_ablation: fewer than 2 subjects with a known cancer outcome")

    num_classes = context.num_smoke_classes
    n_hvgs = context.preprocessing_artifact.n_hvgs
    min_cells = context.config.get("data", context.config).get("min_cells_per_subject", 50)

    subject_ids = np.array(known_subjects)
    y_full = np.array([outcomes_by_subject[s] for s in known_subjects])

    results: Dict[str, Dict] = {name: {"folds": []} for name in variants}

    for seed in seeds:
        folds = grouped_kfold(subject_ids, y_full, n_folds=n_folds, seed=seed)
        for fold_idx, fold in enumerate(folds):
            artifact, train_cell_ds, val_cell_ds = fold_train_val_datasets(
                context, fold["train"], fold["val"], n_hvgs=n_hvgs,
            )
            fp = artifact_fingerprint(artifact)
            train_bags = bags_from_fold_cell_dataset(train_cell_ds, outcomes_by_subject, min_cells)
            val_bags = bags_from_fold_cell_dataset(val_cell_ds, outcomes_by_subject, min_cells)
            if not train_bags or not val_bags:
                for name in variants:
                    results[name]["folds"].append({
                        "seed": seed, "fold": fold_idx, "variant": name, "auroc": None, "auprc": None,
                        "smoke_macro_f1": None, "preprocessing_fingerprint": fp,
                        "skipped_reason": "fold has no subject with >= min_cells_per_subject cells",
                    })
                continue

            train_sd = SubjectLevelDataset(train_bags)
            val_sd = SubjectLevelDataset(val_bags)
            fold_context = _fold_context(context, artifact, train_cell_ds, val_cell_ds)
            train_subject_ids = sorted(str(s) for s in fold["train"])
            val_subject_ids = sorted(str(s) for s in fold["val"])

            for name in variants:
                spec = PATHWAY_ABLATION_VARIANTS[name]
                record = {
                    "seed": seed, "fold": fold_idx, "variant": name, "preprocessing_fingerprint": fp,
                    "train_subject_ids": train_subject_ids, "val_subject_ids": val_subject_ids,
                }
                try:
                    validate_experiment_partitions(
                        train_cell_dataset=train_cell_ds, val_cell_dataset=val_cell_ds,
                        train_subject_dataset=train_sd, val_subject_dataset=val_sd,
                    )
                    check_mil_eligibility(train_sd)
                    check_mil_eligibility(val_sd)
                    if spec["kind"] == "baseline_mil":
                        adapter = NeuralCancerAdapter(pooling="attention", device=device)
                        adapter.fit(fold_context, train_cell_ds, val_cell_ds, train_sd, val_sd,
                                    seed=seed, pretrain_epochs=2)
                        smoke_report = None  # this baseline has no subject-level smoke prediction surface
                        n_params = count_parameters(adapter.trainer.model)
                        module_fp = None
                    else:
                        adapter = build_mil_adapter(
                            PATHWAY_MODEL_NAME, None, device, config_overrides=spec["config_overrides"],
                        )
                        adapter.fit(fold_context, train_cell_ds, val_cell_ds, train_sd, val_sd, seed=seed)
                        smoke_preds = adapter.predict_smoke(val_sd)
                        y_smoke, known_smoke = adapter.known_smoke_labels(val_sd)
                        smoke_report = (
                            full_smoke_metrics_report(y_smoke[known_smoke], smoke_preds[known_smoke], num_classes)
                            if known_smoke.any() else None
                        )
                        n_params = count_parameters(adapter.model)
                        module_fp = adapter.modules.fingerprint() if adapter.modules else None

                    cancer_proba = adapter.predict_proba(val_sd)
                    y_val_ordered = np.array([b["cancer_label"] for b in val_sd.bags])
                    cancer_report = cancer_prediction_metrics(y_val_ordered, cancer_proba)
                    record.update({
                        "auroc": cancer_report.get("auroc"), "auprc": cancer_report.get("auprc"),
                        "smoke_macro_f1": smoke_report["macro_f1"] if smoke_report else None,
                        "smoke_evaluated": smoke_report is not None,
                        "n_parameters": n_params, "module_fingerprint": module_fp,
                    })
                except MILEligibilityError as e:
                    record.update({
                        "auroc": None, "auprc": None, "smoke_macro_f1": None,
                        "skipped_reason": f"MIL ineligible: {e}",
                    })
                results[name]["folds"].append(record)

    for name in variants:
        folds = results[name]["folds"]
        results[name]["auroc"] = aggregate_metric([f["auroc"] for f in folds])
        results[name]["auprc"] = aggregate_metric([f["auprc"] for f in folds])
        results[name]["smoke_macro_f1"] = aggregate_metric([f["smoke_macro_f1"] for f in folds])
        results[name]["n_folds_run"] = len(folds)

    baseline = BASELINE_VARIANT if BASELINE_VARIANT in variants else variants[0]
    comparisons: Dict[str, Dict] = {}
    for name in variants:
        if name == baseline:
            continue
        cmp = compare_models(results, "auroc", name, baseline, seed=seeds[0] if seeds else 42)
        cmp["summary"] = summarize_comparison(cmp)
        comparisons[name] = cmp

    return {
        "task": "pathway_hierarchical_ablation", "primary_metric": "auroc",
        "n_folds_requested": n_folds, "seeds": list(seeds), "variants": variants,
        "baseline_variant": baseline, "results": results, "comparisons": comparisons,
        "development_only": True, "software_only": True,
        "note": "Development-only comparison over context's train+val subject pool (grouped "
                "subject CV) — no frozen test data was accessed and this comparison does not "
                "invoke the frozen-test guard. smoke_macro_f1 is a secondary diagnostic read "
                "off the SAME fitted model as the primary cancer AUROC/AUPRC, restricted to "
                "subjects with a known majority smoke label; it is None for "
                "'existing_attention_mil' (that adapter has no subject-level smoke prediction "
                "surface) and is expected to be near-chance for 'single_task_cancer' "
                "(smoke_loss_weight=0.0 leaves the smoke head untrained). A variant is not "
                "'better' merely because its mean AUROC is nominally higher — see "
                "`comparisons[name].summary` for an honest win/tie/loss and confidence-interval "
                "verdict. Every result here is a software-correctness check, not scientific "
                "evidence of architectural superiority.",
    }


def _ablation_folds_csv_rows(report: Dict) -> List[dict]:
    rows = []
    for name, r in report["results"].items():
        for f in r["folds"]:
            row = {"variant": name}
            row.update({k: v for k, v in f.items() if not isinstance(v, (dict, list))})
            rows.append(row)
    return rows


def write_pathway_hierarchical_ablation_artifact(run_dir: Path, context, report: Dict, synthetic: bool) -> Path:
    """Persist `report` (from run_pathway_hierarchical_ablation) under
    run_dir/metrics/ — same atomic-write/environment-snapshot convention
    every other benchmark artifact in this repository uses (see
    imbalance_ablation.py::write_imbalance_ablation_artifact)."""
    write_environment_artifact(run_dir, synthetic)
    payload = {
        "schema_version": ABLATION_ARTIFACT_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "git_sha": get_git_sha(),
        "config_fingerprint": context.config_fingerprint,
        "split_fingerprint": context.fingerprint(),
        "synthetic": synthetic,
        "frozen_test_data_accessed": False,
        **report,
    }
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "pathway_hierarchical_ablation.json"
    atomic_write_json(json_path, payload)
    atomic_write_csv_rows(metrics_dir / "pathway_hierarchical_ablation_folds.csv", _ablation_folds_csv_rows(report))
    return json_path
