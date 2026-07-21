"""
evidence/candidate_comparison.py — Step 9: identical-partition candidate
comparison across baseline and MIL-kind models for a single evaluable task.

This module never reimplements a model or a splitting/preprocessing
primitive. Every candidate is fit through the existing Phase 1-6 machinery:

  - benchmarks/candidate_registry.py resolves what KIND a candidate name is
    (classical baseline / non-module MIL / pathway module-based MIL) and
    which domain-robustness strategies it can actually support — never
    guessed here.
  - benchmarks/cross_validation.py::run_smoke_cv / run_cancer_cv drive the
    ERM comparison: called ONCE per task with the full candidate name list,
    so every candidate shares the exact same grouped_kfold fold assignment,
    the exact same per-fold refit preprocessing artifact, and the exact
    same label mapping — "identical partitions" is a structural consequence
    of one shared call, not a promise this module has to keep separately.
  - benchmarks/final_evaluation.py::generate_subject_oof_predictions is the
    one function in this repository that already threads a
    domain_robustness_config through to pathway_hierarchical_mil (the only
    architecture with a domain-robustness attachment point — see
    candidate_registry.py). It is used here, with the SAME dev_subjects/
    outcomes_by_subject/context as the ERM comparison, to add one
    additional row per Phase 6 strategy the pathway candidate genuinely
    supports.

Primary selection metric per task (never silently substituted):
  - smoke_classification: development OOF subject-level macro-F1, scored by
    the exact function tracks.py's Track A wraps
    (benchmarks.metrics.full_smoke_metrics_report /
    subject_weighted_full_smoke_metrics_report).
  - subject_level_cancer_prediction: development OOF AUROC, scored by the
    exact function tracks.py's Track C wraps (benchmarks.metrics.
    cancer_prediction_metrics) — AUPRC and a reliability curve are reported
    alongside it, never used to override AUROC as the ranking metric.

A candidate that cannot run at all (incompatible prediction unit,
insufficient data, ineligible fold structure) is recorded in `ineligible`
with a reason string — it never disappears from the report silently. If
class/event support makes the primary metric undefined for every requested
candidate, select_best_candidate() raises CandidateSelectionError rather
than falling back to a secondary metric or a fixed default.

This module runs against synthetic/development fixtures the same way
tracks.py's `_on_fixture` functions do —
run_candidate_comparison_on_synthetic_fixture() below builds a small
internally-consistent synthetic ExperimentContext
(benchmarks.runner.build_synthetic_context) and drives the real comparison
code against it. That proves the comparison machinery genuinely executes;
it is not, and must never be read as, a real-world model comparison. No
cohort registered in configs/cohorts.yaml is currently eligible for any of
these tasks (see evidence/tracks.py), so no real candidate comparison has
been run anywhere in this repository.
"""

from typing import Dict, List, Optional, Sequence

from benchmarks.candidate_registry import resolve_candidate_kind, resolve_strategy_application
from benchmarks.cross_validation import run_cancer_cv, run_smoke_cv
from benchmarks.domain_losses import DOMAIN_STRATEGIES
from benchmarks.final_evaluation import generate_subject_oof_predictions
from benchmarks.mil_registry import PATHWAY_MODEL_NAME

from .tracks import TASK_CANCER_PREDICTION, TASK_SMOKE, run_track_c_on_fixture

# Candidate sets matching the Step 9 requirement exactly: prevalence/
# majority baseline, regularized logistic regression, random forest, small
# MLP, mean-pooling MIL, max-pooling MIL, gated-attention MIL, pathway
# hierarchical MIL. Every name here is independently validated against
# candidate_registry at call time — this tuple is documentation, not the
# source of truth for "is this a real candidate."
DEFAULT_SMOKE_CANDIDATE_NAMES = ("majority", "logistic", "random_forest", "small_mlp", "neural", PATHWAY_MODEL_NAME)
DEFAULT_CANCER_CANDIDATE_NAMES = (
    "prevalence", "logistic", "random_forest", "small_mlp",
    "mean_mil", "max_mil", "attention_mil", PATHWAY_MODEL_NAME,
)

# Gradient boosting (sklearn's HistGradientBoostingClassifier) IS already a
# dependency-safe registered baseline in this repository (benchmarks/
# baselines.py) — it is simply not part of the Step 9 default comparison
# set because the task list above does not name it. It is not excluded for
# any dependency reason.


class CandidateSelectionError(ValueError):
    """Raised when no requested candidate produced a defined primary
    metric — there is nothing legitimate to select, so this never falls
    back to a secondary metric or a fixed default."""


class UnknownDomainStrategyError(ValueError):
    """Raised for a requested domain-robustness strategy that is not one of
    benchmarks.domain_losses.DOMAIN_STRATEGIES."""


def _validate_candidate_names(candidate_names: Sequence[str]) -> None:
    """Fail hard, before running anything, if any requested name is not
    registered in candidate_registry — never silently drop or default an
    unrecognized name (see UnknownCandidateNameError)."""
    for name in candidate_names:
        resolve_candidate_kind(name)  # raises UnknownCandidateNameError itself


def _sorted_oof_arrays(oof_by_subject: Dict[str, float], outcomes_by_subject: Dict[str, int]):
    subject_ids = sorted(oof_by_subject.keys())
    y_true = [outcomes_by_subject[s] for s in subject_ids]
    y_prob = [oof_by_subject[s] for s in subject_ids]
    return subject_ids, y_true, y_prob


# ─── Smoke-classification comparison (Track A primary metric) ─────────────

def compare_candidates_smoke_track(
    context, candidate_names: Sequence[str] = DEFAULT_SMOKE_CANDIDATE_NAMES,
    *, n_folds: int = 5, seeds: Sequence[int] = (42,), device: str = "cpu",
) -> Dict:
    """
    Runs cross_validation.run_smoke_cv ONCE for every requested candidate —
    identical development subject pool, identical grouped_kfold fold
    assignment, identical per-fold-refit preprocessing contract, identical
    label mapping (context.label_mapping) for every candidate in the list,
    by construction of the shared call. Primary metric:
    'subject_weighted_macro_f1', the same subject-level macro-F1 metric
    tracks.py's Track A wraps.

    Returns {"task", "primary_metric", "candidates": {name: {...}},
    "ineligible": [...], "cv_report": <raw run_smoke_cv output>}.
    """
    _validate_candidate_names(candidate_names)
    cv_report = run_smoke_cv(context, list(candidate_names), n_folds=n_folds, seeds=seeds, device=device)

    candidates: Dict[str, Dict] = {}
    ineligible: List[Dict] = []
    for name in candidate_names:
        res = cv_report["results"].get(name, {})
        agg = res.get("subject_weighted_macro_f1", {})
        if agg.get("mean") is None:
            ineligible.append({
                "candidate": name,
                "reason": "subject_weighted_macro_f1 undefined across every requested fold/seed "
                          "(e.g. every fold's validation split had a single dominant class or no "
                          "eligible subjects) — see cv_report for per-fold detail.",
            })
            continue
        candidates[name] = {
            "candidate_kind": resolve_candidate_kind(name),
            "primary_metric": "subject_weighted_macro_f1",
            "subject_weighted_macro_f1": agg,
            "n_folds_run": res.get("n_folds_run"),
        }

    return {
        "task": TASK_SMOKE, "primary_metric": "subject_weighted_macro_f1",
        "candidates": candidates, "ineligible": ineligible, "cv_report": cv_report,
    }


# ─── Cancer-prediction comparison (Track C primary metric) ────────────────

def compare_candidates_cancer_track(
    context, dev_subjects: Sequence[str], outcomes_by_subject: Dict[str, int],
    candidate_names: Sequence[str] = DEFAULT_CANCER_CANDIDATE_NAMES,
    *, n_folds: int = 5, seeds: Sequence[int] = (42,), device: str = "cpu",
    domain_strategies: Sequence[str] = ("erm",),
) -> Dict:
    """
    Two parts, both against the SAME context/dev_subjects/outcomes_by_subject:

      1. cross_validation.run_cancer_cv, called ONCE with every non-domain-
         robustness candidate — identical fold assignment/preprocessing/
         label mapping across candidates by construction, exactly as in
         compare_candidates_smoke_track.
      2. For PATHWAY_MODEL_NAME only (the sole candidate kind with a domain-
         robustness attachment point — candidate_registry.py), one
         additional generate_subject_oof_predictions() run per requested
         strategy in `domain_strategies` beyond "erm" that
         resolve_strategy_application() confirms is actually supported.
         Every OOF-pooled result (one probability per development subject,
         each from a fold that never trained on it) is scored via
         tracks.run_track_c_on_fixture — the exact scoring code Track C
         uses — never a separately implemented AUROC/AUPRC calculation.

    Primary metric: 'auroc'. A candidate/strategy combination that raises
    (MILEligibilityError, insufficient bag data, a single-class OOF
    pool, ...) is recorded in `ineligible` with the raised message, never
    silently dropped.
    """
    _validate_candidate_names(candidate_names)
    for strategy in domain_strategies:
        if strategy not in DOMAIN_STRATEGIES:
            raise UnknownDomainStrategyError(
                f"compare_candidates_cancer_track: requested_strategy={strategy!r} is not one of "
                f"{DOMAIN_STRATEGIES}."
            )

    cv_report = run_cancer_cv(context, list(candidate_names), n_folds=n_folds, seeds=seeds, device=device)

    candidates: Dict[str, Dict] = {}
    ineligible: List[Dict] = []
    for name in candidate_names:
        res = cv_report["results"].get(name, {})
        agg = res.get("auroc", {})
        if agg.get("mean") is None:
            ineligible.append({
                "candidate": name, "strategy": "erm",
                "reason": "auroc undefined across every requested fold/seed — see cv_report for "
                          "per-fold detail (commonly a single-class validation split in every fold).",
            })
            continue
        candidates[f"{name}[erm]" if name == PATHWAY_MODEL_NAME else name] = {
            "candidate_kind": resolve_candidate_kind(name), "strategy": "erm",
            "strategy_applicable": True,
            "primary_metric": "auroc",
            "auroc": agg, "auprc": res.get("auprc", {}), "n_folds_run": res.get("n_folds_run"),
            "source": "cross_validation.run_cancer_cv",
        }

    num_cell_types = context.config.get("model", {}).get("num_cell_types", 4)
    min_cells = context.config.get("data", {}).get("min_cells_per_subject", 5)
    n_hvgs = context.preprocessing_artifact.n_hvgs
    non_erm_strategies = (
        [s for s in domain_strategies if s != "erm"] if PATHWAY_MODEL_NAME in candidate_names else []
    )
    for strategy in non_erm_strategies:
        applied_strategy, applicable, reason = resolve_strategy_application(PATHWAY_MODEL_NAME, strategy)
        candidate_id = f"{PATHWAY_MODEL_NAME}[{strategy}]"
        if not applicable:
            ineligible.append({"candidate": PATHWAY_MODEL_NAME, "strategy": strategy, "reason": reason})
            continue
        try:
            oof = generate_subject_oof_predictions(
                context, PATHWAY_MODEL_NAME, dev_subjects, outcomes_by_subject,
                num_cell_types, min_cells, n_hvgs, device=device, seed=seeds[0], n_folds=n_folds,
                domain_robustness_config={"strategy": strategy},
            )
        except Exception as exc:  # noqa: BLE001 — recorded, never silently swallowed
            ineligible.append({"candidate": PATHWAY_MODEL_NAME, "strategy": strategy, "reason": str(exc)})
            continue
        subject_ids, y_true, y_prob = _sorted_oof_arrays(oof["oof_by_subject"], outcomes_by_subject)
        if len(set(y_true)) < 2:
            ineligible.append({
                "candidate": PATHWAY_MODEL_NAME, "strategy": strategy,
                "reason": f"OOF-pooled development outcomes have a single class {set(y_true)} — "
                          "auroc is undefined.",
            })
            continue
        report = run_track_c_on_fixture(y_true, y_prob, subject_ids, random_seed=seeds[0])
        candidates[candidate_id] = {
            "candidate_kind": resolve_candidate_kind(PATHWAY_MODEL_NAME), "strategy": applied_strategy,
            "strategy_applicable": True, "primary_metric": "auroc",
            "auroc": {"mean": report["metrics"]["auroc"], "n_valid": 1, "n_undefined": 0,
                      "resampling_unit": "oof_pool"},
            "auprc": {"mean": report["metrics"]["auprc"], "n_valid": 1, "n_undefined": 0,
                      "resampling_unit": "oof_pool"},
            "calibration_curve": report["metrics"]["calibration_curve"],
            "n_oof_subjects": len(subject_ids),
            "source": "final_evaluation.generate_subject_oof_predictions",
        }

    return {
        "task": TASK_CANCER_PREDICTION, "primary_metric": "auroc",
        "candidates": candidates, "ineligible": ineligible, "cv_report": cv_report,
    }


# ─── Selection — fail closed, never silently substitutes a metric ─────────

def select_best_candidate(comparison: Dict) -> Dict:
    """
    Ranks `comparison["candidates"]` by comparison["primary_metric"]'s mean.
    Raises CandidateSelectionError if not one candidate has a defined
    primary-metric mean — this never falls back to a secondary metric
    (e.g. AUPRC when AUROC is undefined) or a fixed default candidate.
    """
    primary_metric = comparison["primary_metric"]
    ranked = []
    for name, info in comparison["candidates"].items():
        mean = info.get(primary_metric, {}).get("mean")
        if mean is not None:
            ranked.append({"candidate": name, "score": mean})
    if not ranked:
        raise CandidateSelectionError(
            f"select_best_candidate: no candidate among {list(comparison['candidates'])} has a "
            f"defined {primary_metric!r} — refusing to select a winner from undefined evidence. "
            f"ineligible={comparison.get('ineligible')}"
        )
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return {"primary_metric": primary_metric, "ranked": ranked, "selected": ranked[0]["candidate"]}


# ─── Synthetic-fixture exercise path — proves the machinery runs ──────────

def run_candidate_comparison_on_synthetic_fixture(
    task: str, *, seed: int = 1, fast: bool = True, n_folds: int = 2,
    candidate_names: Optional[Sequence[str]] = None, domain_strategies: Sequence[str] = ("erm",),
) -> Dict:
    """
    Builds benchmarks.runner.build_synthetic_context and drives the real
    comparison code above against it — the same proof pattern tracks.py's
    `_on_fixture` functions and test_frozen_test_sentinel.py's end-to-end
    test both use. This is a synthetic/development fixture only; it never
    claims a real candidate comparison has been run anywhere in this
    environment (see this module's docstring — no cohort in
    configs/cohorts.yaml is currently eligible for any of these tasks).
    """
    from benchmarks.runner import build_synthetic_context

    ctx = build_synthetic_context(seed=seed, fast=fast)
    if task == TASK_SMOKE:
        names = candidate_names if candidate_names is not None else DEFAULT_SMOKE_CANDIDATE_NAMES
        return compare_candidates_smoke_track(ctx, list(names), n_folds=n_folds, seeds=(seed,))
    if task == TASK_CANCER_PREDICTION:
        names = candidate_names if candidate_names is not None else DEFAULT_CANCER_CANDIDATE_NAMES
        dev_subjects = sorted(set(ctx.subjects_for("train")) | set(ctx.subjects_for("val")))
        outcomes_by_subject = {
            str(b["subject_id"]): b["cancer_label"]
            for b in list(ctx.train_bags) + list(ctx.val_bags) if b.get("cancer_label_known")
        }
        return compare_candidates_cancer_track(
            ctx, dev_subjects, outcomes_by_subject, list(names),
            n_folds=n_folds, seeds=(seed,), domain_strategies=domain_strategies,
        )
    raise ValueError(f"run_candidate_comparison_on_synthetic_fixture: unsupported task {task!r}")
