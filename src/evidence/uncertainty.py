"""
evidence/uncertainty.py — Step 10: repeated grouped-resampling uncertainty
reporting for DEVELOPMENT evidence only.

This module is structurally incapable of touching held-out/frozen-test
data: its entire input surface is DevelopmentRepeatedOOF, a frozen
dataclass whose constructor (build_development_repeated_oof) requires the
caller to pass role="development" as an explicit literal — any other value
raises DevelopmentOnlyError before a single statistic is computed. There is
no parameter anywhere in this module named test_subjects, test_bags, or
similar, and repeated_development_oof_for_cancer_track (the one
orchestration helper that calls real training code) only ever calls
benchmarks.final_evaluation.generate_subject_oof_predictions — itself one
of the functions final_evaluation.py documents as structurally unable to
read test data (see that module's docstring).

Every statistic reported here operates on ONE ROW PER SUBJECT — never a
per-cell array. RepeatRecord's constructor rejects a repeat whose
subject_ids contain a duplicate, which is what a cell-level (many rows per
subject) input would look like; test_subject_level_bootstrap_ci_rejects_
duplicate_subject_rows in tests/test_evidence_uncertainty.py asserts this
directly.

Reused, never reimplemented:
  - benchmarks.metrics.aggregate_metric_by_seed for the seed-level
    (independent-repeat) mean/std/median/bootstrap CI.
  - benchmarks.metrics.bootstrap_ci as the underlying percentile-bootstrap
    primitive for the subject-level CI in this module.
  - benchmarks.final_evaluation.generate_subject_oof_predictions for the
    one real-pipeline orchestration helper.

Overlapping confidence intervals computed by this module are never a
formal equivalence test — every function that could be read that way
labels its output "descriptive only" in both the returned dict and this
docstring; nothing in this module returns a boolean "equivalent" verdict
from CI overlap.
"""

import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from benchmarks.metrics import aggregate_metric_by_seed, bootstrap_ci

DEFAULT_EVIDENCE_CONFIG_PATH = "configs/evidence.yaml"
_FALLBACK_MIN_INDEPENDENT_COHORTS = 3
_FALLBACK_MIN_REPEATS_FOR_CI = 2

DEVELOPMENT_ROLE = "development"


class DevelopmentOnlyError(ValueError):
    """Raised when anything other than the literal role='development' is
    passed to build_development_repeated_oof — this module has no code
    path that can legitimately accept held-out/frozen-test data."""


class UncertaintyInputError(ValueError):
    """Raised for a malformed RepeatRecord/DevelopmentRepeatedOOF input —
    mismatched array lengths, duplicate subject IDs within one repeat
    (which would silently turn a subject-level bootstrap into a cell-level
    one), or fewer than 2 repeats where a repeat-level statistic is
    requested."""


def load_uncertainty_policy(config_path: str = DEFAULT_EVIDENCE_CONFIG_PATH) -> Dict:
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        policy = raw.get("uncertainty_policy") or {}
    except (OSError, yaml.YAMLError):
        policy = {}
    return {
        "min_independent_cohorts": int(policy.get("min_independent_cohorts", _FALLBACK_MIN_INDEPENDENT_COHORTS)),
        "min_repeats_for_ci": int(policy.get("min_repeats_for_ci", _FALLBACK_MIN_REPEATS_FOR_CI)),
    }


@dataclass(frozen=True)
class RepeatRecord:
    """One repeat (one grouped-CV reseeding) of subject-level development
    out-of-fold predictions. Exactly one row per subject — subject_ids must
    be unique within a repeat (enforced below); this is what makes every
    downstream resampling operation genuinely subject-level rather than
    cell-level."""

    seed: int
    subject_ids: Tuple[str, ...]
    y_true: Tuple[float, ...]
    y_pred_or_prob: Tuple[float, ...]
    cohort_source: Tuple[Optional[str], ...] = field(default_factory=tuple)

    def __post_init__(self):
        n = len(self.subject_ids)
        if n == 0:
            raise UncertaintyInputError(f"RepeatRecord(seed={self.seed}): subject_ids cannot be empty.")
        if len(self.y_true) != n or len(self.y_pred_or_prob) != n:
            raise UncertaintyInputError(
                f"RepeatRecord(seed={self.seed}): subject_ids ({n}), y_true "
                f"({len(self.y_true)}), y_pred_or_prob ({len(self.y_pred_or_prob)}) must have equal length."
            )
        if len(set(self.subject_ids)) != n:
            raise UncertaintyInputError(
                f"RepeatRecord(seed={self.seed}): subject_ids contains duplicates — a repeat's "
                "predictions must carry exactly one row per subject, never one row per cell. A "
                "repeated subject_id here would silently turn subject-level resampling into "
                "cell-level resampling."
            )
        if self.cohort_source and len(self.cohort_source) != n:
            raise UncertaintyInputError(
                f"RepeatRecord(seed={self.seed}): cohort_source, if given, must have the same "
                f"length as subject_ids ({n}), got {len(self.cohort_source)}."
            )


def build_repeat_record(
    seed: int, subject_ids: Sequence[str], y_true: Sequence[float], y_pred_or_prob: Sequence[float],
    cohort_source: Optional[Sequence[Optional[str]]] = None,
) -> RepeatRecord:
    return RepeatRecord(
        seed=int(seed),
        subject_ids=tuple(str(s) for s in subject_ids),
        y_true=tuple(float(v) for v in y_true),
        y_pred_or_prob=tuple(float(v) for v in y_pred_or_prob),
        cohort_source=tuple(cohort_source) if cohort_source is not None else tuple(),
    )


@dataclass(frozen=True)
class DevelopmentRepeatedOOF:
    """The only input type every function below accepts. `role` must be the
    literal string 'development' — see build_development_repeated_oof."""

    role: str
    repeats: Tuple[RepeatRecord, ...]

    def __post_init__(self):
        if self.role != DEVELOPMENT_ROLE:
            raise DevelopmentOnlyError(
                f"DevelopmentRepeatedOOF: role must be the literal 'development', got {self.role!r} — "
                "this module has no code path that may accept held-out or frozen-test data."
            )
        if not self.repeats:
            raise UncertaintyInputError("DevelopmentRepeatedOOF: repeats cannot be empty.")

    def independent_subject_count(self) -> int:
        """Unique subjects across ALL repeats — a subject reappearing in
        multiple repeats (the expected, normal case for repeated grouped
        resampling of the same development pool) is counted once, not once
        per repeat."""
        return len({s for r in self.repeats for s in r.subject_ids})

    def independent_cohort_count(self) -> Optional[int]:
        """Unique non-null cohort_source values across all repeats, or None
        if no non-null cohort/source value was ever recorded (either
        because no repeat supplied cohort_source at all, or every entry
        supplied was None/unknown) — this module never fabricates a
        cohort/source label when one wasn't genuinely available."""
        sources = {s for r in self.repeats for s in r.cohort_source if s is not None}
        return len(sources) if sources else None


def build_development_repeated_oof(repeats: Sequence[RepeatRecord], role: str) -> DevelopmentRepeatedOOF:
    """The single sanctioned constructor. `role` has no default — the
    caller must type the literal 'development' at every call site, which
    keeps "this data is development-only" an explicit, grep-able statement
    rather than an implicit assumption."""
    return DevelopmentRepeatedOOF(role=role, repeats=tuple(repeats))


# ─── Per-repeat metric evaluation and aggregation ──────────────────────────

def per_repeat_metric_values(oof: DevelopmentRepeatedOOF, metric_fn: Callable[[Sequence[float], Sequence[float]], Optional[float]]) -> List[Optional[float]]:
    """Applies metric_fn(y_true, y_pred_or_prob) to each repeat independently.
    metric_fn should return None (never a filler value like 0.5) when the
    metric is undefined for that repeat's pooled subjects (e.g. a single
    class present) — the same convention benchmarks.metrics already uses."""
    return [metric_fn(list(r.y_true), list(r.y_pred_or_prob)) for r in oof.repeats]


def repeated_metric_summary(
    oof: DevelopmentRepeatedOOF, metric_fn: Callable[[Sequence[float], Sequence[float]], Optional[float]],
    *, bootstrap_seed: int = 42,
) -> Dict:
    """
    Per-repeat metric values plus mean/median/std/IQR across repeats, and a
    seed-level bootstrap CI computed by reusing
    benchmarks.metrics.aggregate_metric_by_seed (each repeat here IS one
    independent grouped-CV reseeding, exactly the resampling unit that
    function's docstring says is statistically valid to bootstrap over —
    unlike bootstrapping raw fold values, which double-counts overlapping
    subjects across folds of the same seed).
    """
    values = per_repeat_metric_values(oof, metric_fn)
    seeds = [r.seed for r in oof.repeats]
    defined = [v for v in values if v is not None]
    iqr = None
    if len(defined) >= 2:
        q1, q3 = np.percentile(np.asarray(defined, dtype=float), [25, 75])
        iqr = float(q3 - q1)
    return {
        "per_repeat_values": values,
        "seeds": seeds,
        "n_repeats": len(oof.repeats),
        "n_defined": len(defined),
        "n_undefined": len(values) - len(defined),
        "mean": float(np.mean(defined)) if defined else None,
        "median": statistics.median(defined) if defined else None,
        "std": float(np.std(defined, ddof=0)) if len(defined) > 1 else (0.0 if defined else None),
        "iqr": iqr,
        "seed_level_bootstrap": aggregate_metric_by_seed(values, seeds, seed=bootstrap_seed),
    }


# ─── Subject-level bootstrap CI (never cell-level) ─────────────────────────

def subject_level_bootstrap_ci(
    record: RepeatRecord, metric_fn: Callable[[Sequence[float], Sequence[float]], Optional[float]],
    *, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05,
) -> Optional[Dict]:
    """
    Resamples SUBJECTS (rows of `record`, one per subject by construction —
    see RepeatRecord.__post_init__) with replacement n_boot times,
    recomputing metric_fn on each resample, and returns the percentile CI.
    This is a genuinely subject-level bootstrap: the unit being resampled
    is a row of (subject_id, y_true, y_pred_or_prob), never a cell. Returns
    None if metric_fn is undefined on the full (unresampled) record or if
    fewer than 2 subjects are present.
    """
    n = len(record.subject_ids)
    if n < 2:
        return None
    point_estimate = metric_fn(list(record.y_true), list(record.y_pred_or_prob))
    if point_estimate is None:
        return None
    rng = np.random.RandomState(seed)
    y_true_arr = np.asarray(record.y_true, dtype=float)
    y_pred_arr = np.asarray(record.y_pred_or_prob, dtype=float)
    boot_values = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        v = metric_fn(y_true_arr[idx].tolist(), y_pred_arr[idx].tolist())
        if v is not None:
            boot_values.append(v)
    if len(boot_values) < 2:
        return {
            "point_estimate": point_estimate, "lo": None, "hi": None,
            "n_boot_defined": len(boot_values), "n_boot_requested": n_boot,
            "resampling_unit": "subject",
        }
    lo, hi = np.percentile(boot_values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point_estimate": point_estimate, "lo": float(lo), "hi": float(hi),
        "n_boot_defined": len(boot_values), "n_boot_requested": n_boot,
        "resampling_unit": "subject",
    }


def confidence_intervals_overlap(ci_a: Optional[Dict], ci_b: Optional[Dict]) -> Optional[Dict]:
    """
    Descriptive-only check for whether two subject_level_bootstrap_ci()
    outputs' [lo, hi] intervals overlap. THIS IS NOT AN EQUIVALENCE TEST —
    the returned dict is labeled interpretation='descriptive_only' and
    callers must never read `overlaps` as a formal statistical conclusion
    about whether two candidates perform equivalently. Returns None if
    either CI is undefined.
    """
    if not ci_a or not ci_b or ci_a.get("lo") is None or ci_b.get("lo") is None:
        return None
    overlaps = not (ci_a["hi"] < ci_b["lo"] or ci_b["hi"] < ci_a["lo"])
    return {
        "overlaps": overlaps,
        "interpretation": "descriptive_only",
        "disclaimer": (
            "Confidence-interval overlap is a descriptive observation only. It is not a formal "
            "equivalence test and must never be reported or read as evidence that two candidates "
            "perform equivalently — overlapping intervals do not imply no difference, and "
            "non-overlapping intervals alone do not establish one either without an explicit "
            "paired hypothesis test."
        ),
    }


# ─── Paired candidate comparison ───────────────────────────────────────────

def paired_candidate_comparison(
    oof_a: DevelopmentRepeatedOOF, oof_b: DevelopmentRepeatedOOF,
    metric_fn: Callable[[Sequence[float], Sequence[float]], Optional[float]],
) -> Dict:
    """
    Pairs oof_a and oof_b's repeats by seed (both must have been produced
    from the same repeated resampling — a seed present in one but not the
    other is recorded, never silently dropped) and reports per-repeat
    signed differences, win/tie/loss counts, and the mean paired
    difference. Ties are exact float equality of the two repeats' metric
    values — floating point noise from independently retrained models will
    generally NOT tie, which is expected and reported honestly rather than
    rounded to force ties.
    """
    by_seed_a = {r.seed: r for r in oof_a.repeats}
    by_seed_b = {r.seed: r for r in oof_b.repeats}
    common_seeds = sorted(set(by_seed_a) & set(by_seed_b))
    unmatched = sorted(set(by_seed_a) ^ set(by_seed_b))

    diffs, per_seed = [], []
    wins = ties = losses = undefined = 0
    for seed in common_seeds:
        va = metric_fn(list(by_seed_a[seed].y_true), list(by_seed_a[seed].y_pred_or_prob))
        vb = metric_fn(list(by_seed_b[seed].y_true), list(by_seed_b[seed].y_pred_or_prob))
        if va is None or vb is None:
            undefined += 1
            per_seed.append({"seed": seed, "a": va, "b": vb, "diff": None})
            continue
        diff = va - vb
        diffs.append(diff)
        per_seed.append({"seed": seed, "a": va, "b": vb, "diff": diff})
        if diff > 0:
            wins += 1
        elif diff < 0:
            losses += 1
        else:
            ties += 1

    return {
        "common_seeds": common_seeds,
        "unmatched_seeds": unmatched,
        "per_seed": per_seed,
        "n_undefined": undefined,
        "wins_a_over_b": wins, "ties": ties, "losses_a_over_b": losses,
        "mean_paired_diff": float(np.mean(diffs)) if diffs else None,
        "median_paired_diff": float(np.median(diffs)) if diffs else None,
    }


# ─── Independent-cohort-count-gated status ─────────────────────────────────

def uncertainty_status(oof: DevelopmentRepeatedOOF, config_path: str = DEFAULT_EVIDENCE_CONFIG_PATH) -> Dict:
    """
    Returns {"status": "adequate" | "insufficient_evidence_for_reliable_source_level_uncertainty"
    | "cohort_source_not_recorded", "independent_subject_count", "independent_cohort_count",
    "min_independent_cohorts_required"}. Never claims "adequate" source-level
    uncertainty when independent_cohort_count is None (never recorded) or
    below the configured minimum.
    """
    policy = load_uncertainty_policy(config_path)
    n_subjects = oof.independent_subject_count()
    n_cohorts = oof.independent_cohort_count()
    min_required = policy["min_independent_cohorts"]
    if n_cohorts is None:
        status = "cohort_source_not_recorded"
    elif n_cohorts < min_required:
        status = "insufficient_evidence_for_reliable_source_level_uncertainty"
    else:
        status = "adequate"
    return {
        "status": status,
        "independent_subject_count": n_subjects,
        "independent_cohort_count": n_cohorts,
        "min_independent_cohorts_required": min_required,
    }


# ─── Real-pipeline orchestration — cancer track only ───────────────────────

def cohort_source_from_bags(bags: Sequence[dict]) -> Dict[str, Optional[str]]:
    """Builds a subject_id -> cohort/source map from real MIL bag dicts
    (each carries an already-computed majority "source" field — see
    benchmarks/fold_preprocessing.py::bags_from_fold_cell_dataset). Never
    derives a source from a subject_id string or any other heuristic —
    only reads the field the pipeline already attached."""
    return {str(b["subject_id"]): b.get("source") for b in bags}


def repeated_development_oof_for_cancer_track(
    context, candidate_name: str, dev_subjects: Sequence[str], outcomes_by_subject: Dict[str, int],
    num_cell_types: int, min_cells_per_subject: int, n_hvgs: int,
    *, n_repeats: int = 3, base_seed: int = 42, n_folds: int = 5, device: str = "cpu",
    pooling: str = "attention",
) -> DevelopmentRepeatedOOF:
    """
    Runs benchmarks.final_evaluation.generate_subject_oof_predictions once
    per repeat (seeds base_seed, base_seed+1, ..., base_seed+n_repeats-1) —
    each call is a fresh grouped-CV re-partition of the SAME development
    pool (dev_subjects/outcomes_by_subject/context never change across
    repeats), and wraps the results into a DevelopmentRepeatedOOF. This is
    the one function in this module that trains anything; everything else
    is pure statistics over its output. cohort_source is derived per
    subject from the real bag "source" field of the outer context's
    train+val bags (never fabricated) when available.
    """
    from benchmarks.final_evaluation import generate_subject_oof_predictions

    all_bags = list(getattr(context, "train_bags", [])) + list(getattr(context, "val_bags", []))
    source_by_subject = cohort_source_from_bags(all_bags)

    records = []
    for i in range(n_repeats):
        seed = base_seed + i
        oof = generate_subject_oof_predictions(
            context, candidate_name, dev_subjects, outcomes_by_subject,
            num_cell_types, min_cells_per_subject, n_hvgs, pooling=pooling, device=device,
            seed=seed, n_folds=n_folds,
        )
        subject_ids = sorted(oof["oof_by_subject"].keys())
        y_true = [outcomes_by_subject[s] for s in subject_ids]
        y_pred = [oof["oof_by_subject"][s] for s in subject_ids]
        cohort_source = [source_by_subject.get(s) for s in subject_ids]
        records.append(build_repeat_record(seed, subject_ids, y_true, y_pred, cohort_source))

    return build_development_repeated_oof(records, role=DEVELOPMENT_ROLE)
