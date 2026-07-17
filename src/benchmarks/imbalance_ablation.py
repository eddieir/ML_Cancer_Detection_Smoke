"""
benchmarks/imbalance_ablation.py — development-only comparison of Phase 2
class-imbalance strategies (data/sampling.py, model.py's FocalLoss).

Reuses the SAME grouped-subject-CV machinery cross_validation.py's
run_smoke_cv already uses (data/splitting.py::grouped_kfold,
fold_preprocessing.py's per-fold artifact refit), so every strategy sees
identical outer folds, identical preprocessing boundaries, and the same
candidate architecture — only train.smoke_imbalance differs between
strategies. CV always runs over context's TRAIN+VAL subject pool; the test
split is never touched here (same contract as run_smoke_cv/run_cancer_cv).

Only the TRAINING half of each fold is ever capped/sampled/rebalanced —
every strategy is scored against the SAME complete, uncapped, natural
validation fold, so a difference between strategies can never be an
artifact of two strategies actually being evaluated on different
validation populations.

Primary metrics are SUBJECT-level (one vote per subject — see
metrics.py::subject_weighted_full_smoke_metrics_report), matching Phase 1's
own subject_weighted_macro_f1 convention; cell-level numbers are reported
only as an explicitly-labeled secondary diagnostic (a subject with many
cells must not be able to dominate the primary comparison).

Implemented as a focused, standalone function rather than folded into
cross_validation.py's nested-hyperparameter-search machinery
(select_nested_hyperparameters_with_refit) because imbalance strategy is a
qualitatively different axis than a hyperparameter grid: this compares
named, fully-specified STRATEGIES against each other directly, it does not
select one strategy's own hyperparameters via inner CV.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from train import assert_disjoint_subjects
from data.splitting import grouped_kfold

from .atomic_io import atomic_write_csv_rows, atomic_write_json
from .context import get_git_sha
from .cross_validation import DEFAULT_SEEDS, _fold_context, _majority_label_by_subject
from .features import cap_cell_dataset
from .fold_preprocessing import artifact_fingerprint, fold_train_val_datasets, require_normalized_adata
from .metrics import (
    aggregate_metric,
    aggregate_metric_by_seed,
    full_smoke_metrics_report,
    subject_weighted_full_smoke_metrics_report,
)
from .neural import NeuralSmokeAdapter
from .reporting import compare_models, summarize_comparison, write_environment_artifact

ABLATION_ARTIFACT_SCHEMA_VERSION = 1

# The five ablations requirement 6 (STEP 12) asks for at minimum. Every
# strategy is a COMPLETE, self-consistent train.smoke_imbalance override —
# never a partial diff — so results are never contaminated by a stale key
# left over from a previous strategy dict.
IMBALANCE_ABLATION_STRATEGIES: Dict[str, dict] = {
    "natural_no_weight": {
        "sampler": "shuffle", "class_selection": "uniform", "subject_selection": "uniform",
        "batches_per_epoch": None, "samples_per_epoch": None, "cells_per_subject_per_batch": None,
        "replacement": True, "class_weighting": "none", "loss": "cross_entropy",
        "focal_gamma": 2.0, "focal_alpha_mode": "class_weights", "seed_offset": 0,
    },
    "natural_inverse_frequency": {
        "sampler": "shuffle", "class_selection": "uniform", "subject_selection": "uniform",
        "batches_per_epoch": None, "samples_per_epoch": None, "cells_per_subject_per_batch": None,
        "replacement": True, "class_weighting": "inverse_frequency", "loss": "cross_entropy",
        "focal_gamma": 2.0, "focal_alpha_mode": "class_weights", "seed_offset": 0,
    },
    "subject_balanced_no_weight": {
        "sampler": "subject_balanced", "class_selection": "uniform", "subject_selection": "uniform",
        "batches_per_epoch": None, "samples_per_epoch": None, "cells_per_subject_per_batch": 8,
        "replacement": True, "class_weighting": "none", "loss": "cross_entropy",
        "focal_gamma": 2.0, "focal_alpha_mode": "class_weights", "seed_offset": 0,
    },
    "subject_balanced_inverse_frequency": {
        "sampler": "subject_balanced", "class_selection": "uniform", "subject_selection": "uniform",
        "batches_per_epoch": None, "samples_per_epoch": None, "cells_per_subject_per_batch": 8,
        "replacement": True, "class_weighting": "inverse_frequency", "loss": "cross_entropy",
        "focal_gamma": 2.0, "focal_alpha_mode": "class_weights", "seed_offset": 0,
    },
    # Sampling already does the class correction here, so alpha is
    # deliberately "none" — avoids stacking inverse-frequency loss
    # weighting on top of inverse-frequency-ish sampling (the
    # "double-correction" risk documented in README.md/model.py).
    "subject_balanced_focal": {
        "sampler": "subject_balanced", "class_selection": "uniform", "subject_selection": "uniform",
        "batches_per_epoch": None, "samples_per_epoch": None, "cells_per_subject_per_batch": 8,
        "replacement": True, "class_weighting": "none", "loss": "focal",
        "focal_gamma": 2.0, "focal_alpha_mode": "none", "seed_offset": 0,
    },
}
BASELINE_STRATEGY = "natural_no_weight"


def _validation_fingerprint(val_ds) -> dict:
    """A stable identity for one fold's validation population — used by
    tests/callers to prove every strategy in a fold was scored against the
    IDENTICAL, untouched validation cells (never capped/resampled/mutated).
    Deliberately independent of dict/array iteration order."""
    subj = sorted(val_ds.subject_ids.tolist())
    return {
        "n_cells": len(val_ds),
        "subject_ids": subj,
        "subject_counts": {s: int((val_ds.subject_ids == s).sum()) for s in sorted(set(subj))},
        "smoke_label_sum": int(val_ds.smoke.sum().item()),
    }


def run_smoke_imbalance_ablation(
    context,
    strategies:              Optional[Sequence[str]] = None,
    n_folds:                  int = 5,
    seeds:                    Sequence[int] = DEFAULT_SEEDS,
    device:                   str = "cpu",
    max_cells_per_subject:     Optional[int] = None,
) -> Dict:
    """
    Compare named smoke-imbalance strategies (IMBALANCE_ABLATION_STRATEGIES)
    with identical outer folds/seeds/preprocessing/architecture, development
    data only. Never touches context's test split; never uses this
    comparison's outcome to invoke the frozen-test guard.

    Only the TRAINING split of each fold is capped (cap_cell_dataset) so a
    subject with many more cells than others cannot dominate a gradient
    step — every strategy trains on that SAME capped train set. Validation
    is the fold's COMPLETE, uncapped set of cells for every strategy; it is
    never resampled, duplicated, or otherwise altered.
    """
    strategies = list(strategies) if strategies is not None else list(IMBALANCE_ABLATION_STRATEGIES)
    unknown = set(strategies) - set(IMBALANCE_ABLATION_STRATEGIES)
    if unknown:
        raise ValueError(f"run_smoke_imbalance_ablation: unknown strategy name(s) {sorted(unknown)} "
                          f"— must be a subset of {sorted(IMBALANCE_ABLATION_STRATEGIES)}")
    if not strategies:
        raise ValueError("run_smoke_imbalance_ablation: strategies must be non-empty.")

    normalized_adata = require_normalized_adata(context)
    num_classes = context.num_smoke_classes
    n_hvgs = context.preprocessing_artifact.n_hvgs
    max_cells_per_subject = (
        max_cells_per_subject if max_cells_per_subject is not None
        else context.config.get("benchmarks", {}).get("max_cells_per_subject_cv", 200)
    )

    pool_subjects = sorted(set(context.subjects_for("train")) | set(context.subjects_for("val")))
    label_by_subject = _majority_label_by_subject(normalized_adata, pool_subjects, num_classes)
    subject_ids = np.array(pool_subjects)
    y_full = np.array([label_by_subject[s] for s in pool_subjects])

    results: Dict[str, Dict] = {name: {"folds": []} for name in strategies}

    for seed in seeds:
        folds = grouped_kfold(subject_ids, y_full, n_folds=n_folds, seed=seed)
        for fold_idx, fold in enumerate(folds):
            artifact, train_ds, val_ds = fold_train_val_datasets(
                context, fold["train"], fold["val"], n_hvgs=n_hvgs,
            )
            assert_disjoint_subjects(train_ds, val_ds, names=["fold_train", "fold_val"])
            fp = artifact_fingerprint(artifact)
            val_fingerprint = _validation_fingerprint(val_ds)
            # ONLY the training split is capped — cap_cell_dataset is
            # deterministic given (seed, max cells), so every strategy
            # trains on the exact same fold cells, but validation is the
            # fold's own complete, natural, uncapped cell population,
            # reused unmutated (never re-derived) across every strategy —
            # see _validation_fingerprint's use in the caller-facing tests.
            train_ds_capped = cap_cell_dataset(train_ds, max_cells_per_subject, seed=seed)

            for strategy_name in strategies:
                strategy_cfg = IMBALANCE_ABLATION_STRATEGIES[strategy_name]
                fold_context = _fold_context(context, artifact, train_ds_capped, val_ds)
                overridden = dict(fold_context.config)
                overridden["train"] = dict(overridden.get("train", {}))
                overridden["train"]["smoke_imbalance"] = dict(strategy_cfg)
                fold_context.config = overridden

                adapter = NeuralSmokeAdapter(fold_context.config, device=device)
                adapter.fit(fold_context, train_ds_capped, val_ds, seed=seed)
                preds = adapter.predict(val_ds)

                # PRIMARY: subject-level (one vote per subject) — a subject
                # with many cells cannot dominate this. SECONDARY: raw
                # cell-level numbers, explicitly labeled diagnostic only.
                subject_report = subject_weighted_full_smoke_metrics_report(
                    val_ds.smoke.numpy(), preds, val_ds.subject_ids, num_classes,
                )
                cell_report = full_smoke_metrics_report(val_ds.smoke.numpy(), preds, num_classes)

                results[strategy_name]["folds"].append({
                    "seed": seed, "fold": fold_idx,
                    "strategy": strategy_name, "strategy_config": strategy_cfg,
                    "classes_absent_from_val": fold["classes_absent_from_val"],
                    "preprocessing_fingerprint": fp,
                    "validation_fingerprint": val_fingerprint,
                    "n_train_cells_capped": len(train_ds_capped),
                    "n_train_cells_before_cap": len(train_ds),
                    "n_val_cells": len(val_ds),
                    "subject_weighted_macro_f1": subject_report["macro_f1"],
                    "subject_level": subject_report,
                    "cell_level_diagnostic": cell_report,
                    "cell_weighted_macro_f1_diagnostic": cell_report["macro_f1"],
                    "hyperparameters": adapter.metadata(),
                })

    for name in strategies:
        folds = results[name]["folds"]
        fold_seeds = [f["seed"] for f in folds]
        results[name]["subject_weighted_macro_f1"] = aggregate_metric(
            [f["subject_weighted_macro_f1"] for f in folds],
        )
        results[name]["subject_weighted_macro_f1_by_seed"] = aggregate_metric_by_seed(
            [f["subject_weighted_macro_f1"] for f in folds], fold_seeds,
        )
        results[name]["n_folds_run"] = len(folds)

    baseline = BASELINE_STRATEGY if BASELINE_STRATEGY in strategies else strategies[0]
    paired = _paired_fold_differences(results, baseline)
    comparisons = _strategy_comparisons(results, baseline, strategies, seed=seeds[0] if seeds else 42)

    return {
        "task": "smoke_imbalance_ablation", "primary_metric": "subject_weighted_macro_f1",
        "n_folds_requested": n_folds, "seeds": list(seeds), "strategies": strategies,
        "baseline_strategy": baseline,
        "results": results, "paired_fold_differences": paired,
        "comparisons": comparisons,
        "development_only": True,
        "note": "Development-only comparison over context's train+val subject pool "
                "(grouped subject CV) — no test data was accessed and this comparison "
                "does not invoke the frozen-test guard. A strategy is not 'better' "
                "merely because its mean subject_weighted_macro_f1 is nominally "
                "higher than another's — see `comparisons` for win/tie/loss counts and "
                "confidence intervals, and `comparisons[name].summary` for an honest "
                "'insufficient evidence' verdict when the evidence does not support a "
                "confident selection. Folds drawn from repeated seeds over the same "
                "overlapping subject pool are NOT independent biological replications "
                "— see each comparison's independence_note.",
    }


def _paired_fold_differences(
    results: Dict[str, Dict], baseline: str,
) -> Dict[str, List[Optional[float]]]:
    """
    Per-(seed, fold) subject_weighted_macro_f1 difference vs `baseline`, for
    every other strategy — lets a caller see whether a strategy wins on
    EVERY fold or only on average, rather than collapsing straight to a
    single mean comparison. A None entry marks a fold where either strategy
    produced an undefined value for that (seed, fold) — never silently
    dropped from the list, so len() always matches the baseline's fold count.
    """
    base_folds = {(f["seed"], f["fold"]): f["subject_weighted_macro_f1"] for f in results[baseline]["folds"]}
    out: Dict[str, List[Optional[float]]] = {}
    for name, r in results.items():
        if name == baseline:
            continue
        diffs = []
        for f in r["folds"]:
            key = (f["seed"], f["fold"])
            b = base_folds.get(key)
            v = f["subject_weighted_macro_f1"]
            diffs.append((v - b) if (b is not None and v is not None) else None)
        out[name] = diffs
    return out


def _strategy_comparisons(
    results: Dict[str, Dict], baseline: str, strategies: Sequence[str], seed: int = 42,
) -> Dict[str, Dict]:
    """
    Honest paired comparison of every non-baseline strategy against
    `baseline`, reusing reporting.py's existing compare_models/
    summarize_comparison (the same machinery Phase 1's CV report uses) so
    this never duplicates a weaker win/tie/loss or confidence-interval
    implementation. Adds the median paired difference and an explicit
    missing-pair count, and never claims a winner from the mean alone —
    `summary.meaningfully_better` is False whenever the evidence (win
    fraction, seed-level CI) does not support a confident selection.
    """
    comparisons: Dict[str, Dict] = {}
    for name in strategies:
        if name == baseline:
            continue
        cmp = compare_models(results, "subject_weighted_macro_f1", name, baseline, seed=seed)
        n_folds_total = len(results[name]["folds"])
        n_pairs = cmp.get("n_pairs", 0)
        if n_pairs > 0:
            base_folds = {(f["seed"], f["fold"]): f["subject_weighted_macro_f1"] for f in results[baseline]["folds"]}
            diffs = [
                f["subject_weighted_macro_f1"] - base_folds[(f["seed"], f["fold"])]
                for f in results[name]["folds"]
                if (f["seed"], f["fold"]) in base_folds
                and base_folds[(f["seed"], f["fold"])] is not None
                and f["subject_weighted_macro_f1"] is not None
            ]
            cmp["median_diff"] = float(np.median(diffs)) if diffs else None
        else:
            cmp["median_diff"] = None
        cmp["n_folds_total"] = n_folds_total
        cmp["n_missing_pairs"] = n_folds_total - n_pairs
        cmp["independence_note"] = (
            "Folds drawn from repeated seeds over the same overlapping development subject "
            "pool are NOT independent biological replications — per-fold win/tie/loss counts "
            "are descriptive, not a formal significance test. ci_diff_by_seed (bootstrapped "
            "across independent seeds) is the more rigorous interval when >=2 seeds were run."
        )
        cmp["summary"] = summarize_comparison(cmp)
        comparisons[name] = cmp
    return comparisons


# ─── Artifact persistence ───────────────────────────────────────────────────────

def _ablation_folds_csv_rows(report: Dict) -> List[dict]:
    rows = []
    for name, r in report["results"].items():
        for f in r["folds"]:
            row = {"strategy": name}
            row.update({k: v for k, v in f.items() if not isinstance(v, (dict, list))})
            rows.append(row)
    return rows


def write_imbalance_ablation_artifact(
    run_dir: Path, context, report: Dict, synthetic: bool,
) -> Path:
    """
    Persists `report` (the dict returned by run_smoke_imbalance_ablation)
    as a reproducible artifact under run_dir/metrics/, reusing the SAME
    atomic-write/environment-snapshot machinery every other benchmark
    artifact uses (see reporting.py) rather than a parallel, incompatible
    system. Never overwrites an existing run_dir (see reporting.new_run_dir)
    and never accessed the frozen test split to produce `report` — see
    run_smoke_imbalance_ablation's own docstring/contract.

    Returns the path of the primary JSON artifact written.
    """
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
    json_path = metrics_dir / "smoke_imbalance_ablation.json"
    atomic_write_json(json_path, payload)
    atomic_write_csv_rows(metrics_dir / "smoke_imbalance_ablation_folds.csv", _ablation_folds_csv_rows(report))
    return json_path


def read_imbalance_ablation_artifact(json_path: Path) -> Dict:
    """Round-trip counterpart to write_imbalance_ablation_artifact — plain
    JSON load, kept as a named function so callers/tests have one place to
    point at rather than reimplementing the read side ad hoc."""
    import json
    with open(json_path) as f:
        return json.load(f)
