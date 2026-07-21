"""
evidence/subgroups.py — Step 13: subgroup and fairness diagnostics.

Only evaluates subgroup dimensions that genuinely have a data source
somewhere in this repository's subject-level records today:

  - cohort_source   — the majority dataset_source/accession a subject's
                       cells came from (benchmarks/fold_preprocessing.py's
                       bag "source" field; configs/cohorts.yaml's
                       cohort_id).
  - exposure_type   — the subject's smoke/vape/never-type label
                       (CellLevelDataset.smoke_labels / bag
                       "smoke_labels", subject to the same verified-vs-
                       weak-label distinction the rest of this repository
                       enforces — see evidence/tracks.py Track A).
  - disease_status  — malignancy/cancer outcome known for the subject
                       (CellLevelDataset.malignancy_labels /
                       malignancy_known, bag "cancer_label"/
                       "cancer_label_known").
  - assay_platform  — the cohort's assay_type (configs/cohorts.yaml).
  - species         — the cohort's species (configs/cohorts.yaml).

Sex, age band, race/ethnicity, and site are NOT implemented here — grepping
src/data/*, src/train.py, and configs/cohorts.yaml turns up no field
carrying any of them anywhere in this repository (see the round's research
notes). Adding a subgroup dimension without a genuine upstream field would
mean fabricating one, which this module refuses to do: SubjectRecord's
metadata dict validates every key against SUPPORTED_SUBGROUP_DIMENSIONS,
and requesting an unsupported dimension (e.g. "sex") raises
UnsupportedSubgroupDimensionError rather than silently returning an empty
or fabricated report.

Structural "never infer a sensitive characteristic" guarantee: every
function in this module reads ONLY record.metadata[dimension] — a value
the caller must have already attached from a genuine upstream field. No
function here ever reads record.subject_id's string content, an accession
string, or an expression value to decide a subgroup membership.
test_subgroup_dimensions_never_derived_from_subject_identity_or_expression
in tests/test_evidence_subgroups.py proves this directly: subject IDs that
LOOK like they encode a sensitive attribute (e.g. "female_65_smoker_003")
produce identical subgroup reports regardless of what the ID string says,
because nothing here ever parses it.

No report or summary this module produces may claim fairness has been
established — see BANNED_FAIRNESS_CLAIM_PHRASES and assert_no_banned_claims().
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import yaml

from benchmarks.metrics import bootstrap_ci, expected_calibration_error

DEFAULT_EVIDENCE_CONFIG_PATH = "configs/evidence.yaml"
_FALLBACK_MIN_SUBJECTS_PER_SUBGROUP = 10

SUPPORTED_SUBGROUP_DIMENSIONS = (
    "cohort_source", "exposure_type", "disease_status", "assay_platform", "species",
)

# Phrases that must never appear in any text this module (or a wrapper
# around it) produces — a subgroup/fairness diagnostic report can describe
# what was measured, but it can never assert the conclusion "fairness
# established" in any of its common phrasings.
BANNED_FAIRNESS_CLAIM_PHRASES = (
    "fairness established",
    "fairness is established",
    "bias-free",
    "free of bias",
    "no bias",
    "unbiased across",
    "fair for all subgroups",
    "fair across all subgroups",
    "equitable across all subgroups",
    "confirmed fair",
    "guarantees fairness",
)


class UnsupportedSubgroupDimensionError(ValueError):
    """Raised when a requested dimension is not one of
    SUPPORTED_SUBGROUP_DIMENSIONS — this module never fabricates a
    dimension that has no genuine data source in this repository."""


class SubgroupInputError(ValueError):
    """Raised for a malformed SubjectRecord input (mismatched lengths, an
    unrecognized metadata key)."""


class BannedFairnessClaimError(ValueError):
    """Raised by assert_no_banned_claims() when a generated report string
    contains a banned fairness-established phrase."""


def load_subgroup_policy(config_path: str = DEFAULT_EVIDENCE_CONFIG_PATH) -> Dict:
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        policy = raw.get("subgroup_policy") or {}
    except (OSError, yaml.YAMLError):
        policy = {}
    return {"min_subjects_per_subgroup": int(policy.get("min_subjects_per_subgroup", _FALLBACK_MIN_SUBJECTS_PER_SUBGROUP))}


@dataclass(frozen=True)
class SubjectRecord:
    """One subject's outcome/prediction plus whatever subgroup metadata the
    caller genuinely has for them. `metadata` keys must all be members of
    SUPPORTED_SUBGROUP_DIMENSIONS; a value of None means "missing for this
    subject" (counted, never silently dropped) — it is never inferred."""

    subject_id: str
    y_true: float
    y_pred_or_prob: float
    metadata: Dict[str, Optional[str]] = field(default_factory=dict)

    def __post_init__(self):
        unknown = sorted(set(self.metadata) - set(SUPPORTED_SUBGROUP_DIMENSIONS))
        if unknown:
            raise SubgroupInputError(
                f"SubjectRecord(subject_id={self.subject_id!r}): metadata key(s) {unknown} are not "
                f"in SUPPORTED_SUBGROUP_DIMENSIONS {SUPPORTED_SUBGROUP_DIMENSIONS} — this repository "
                "has no genuine data source for that dimension; refusing to accept it rather than "
                "silently fabricating a subgroup."
            )


def build_subject_record(
    subject_id: str, y_true: float, y_pred_or_prob: float, metadata: Optional[Dict[str, Optional[str]]] = None,
) -> SubjectRecord:
    return SubjectRecord(
        subject_id=str(subject_id), y_true=float(y_true), y_pred_or_prob=float(y_pred_or_prob),
        metadata=dict(metadata) if metadata else {},
    )


def _subject_level_ci(y_true: Sequence[float], y_pred: Sequence[float],
                       metric_fn: Callable[[Sequence[float], Sequence[float]], Optional[float]],
                       *, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> Optional[Dict]:
    """Same subject-level (never cell-level, never fold-level) bootstrap
    pattern as evidence/uncertainty.py::subject_level_bootstrap_ci — kept
    as a small local helper here rather than importing uncertainty.py, so
    this module has no dependency on repeated-resampling machinery it
    doesn't need."""
    n = len(y_true)
    if n < 2:
        return None
    point = metric_fn(list(y_true), list(y_pred))
    if point is None:
        return None
    rng = np.random.RandomState(seed)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    boot = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        v = metric_fn(y_true_arr[idx].tolist(), y_pred_arr[idx].tolist())
        if v is not None:
            boot.append(v)
    if len(boot) < 2:
        return {"point_estimate": point, "lo": None, "hi": None, "n_boot_defined": len(boot),
                "resampling_unit": "subject"}
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point_estimate": point, "lo": float(lo), "hi": float(hi), "n_boot_defined": len(boot),
            "resampling_unit": "subject"}


def _calibration_for_subset(y_true: Sequence[float], y_pred: Sequence[float]) -> Optional[float]:
    """expected_calibration_error is only meaningful for probability-like
    predictions and >=2 classes present — returns None (never a filler
    value) otherwise."""
    y_true_arr = np.asarray(y_true, dtype=float)
    if len(set(y_true_arr.tolist())) < 2:
        return None
    return expected_calibration_error(y_true_arr, np.asarray(y_pred, dtype=float))


def subgroup_report(
    records: Sequence[SubjectRecord], dimension: str,
    metric_fn: Callable[[Sequence[float], Sequence[float]], Optional[float]],
    *, config_path: str = DEFAULT_EVIDENCE_CONFIG_PATH, compute_calibration: bool = True,
) -> Dict:
    """
    Per-value breakdown of `dimension` across `records`: subject count,
    class/event count (via class_counts on y_true), primary metric,
    calibration (ECE), subject-level bootstrap CI, and missing-metadata
    count. A value with fewer than the configured minimum subjects is
    marked not_evaluable rather than silently included or dropped —
    min_subjects_per_subgroup is read from configs/evidence.yaml
    (subgroup_policy.min_subjects_per_subgroup).
    """
    if dimension not in SUPPORTED_SUBGROUP_DIMENSIONS:
        raise UnsupportedSubgroupDimensionError(
            f"subgroup_report: dimension={dimension!r} is not in SUPPORTED_SUBGROUP_DIMENSIONS "
            f"{SUPPORTED_SUBGROUP_DIMENSIONS} — this repository has no genuine data source for it."
        )
    policy = load_subgroup_policy(config_path)
    min_subjects = policy["min_subjects_per_subgroup"]

    missing = [r for r in records if r.metadata.get(dimension) is None]
    present = [r for r in records if r.metadata.get(dimension) is not None]

    values = sorted({r.metadata[dimension] for r in present})
    by_value: Dict[str, Dict] = {}
    for value in values:
        subset = [r for r in present if r.metadata[dimension] == value]
        y_true = [r.y_true for r in subset]
        y_pred = [r.y_pred_or_prob for r in subset]
        class_counts: Dict[str, int] = {}
        for v in y_true:
            key = str(v)
            class_counts[key] = class_counts.get(key, 0) + 1

        if len(subset) < min_subjects:
            by_value[value] = {
                "status": "not_evaluable",
                "reason": f"{len(subset)} subject(s) < configured minimum {min_subjects} for this dimension",
                "subject_count": len(subset), "class_counts": class_counts,
            }
            continue

        metric_value = metric_fn(y_true, y_pred)
        entry = {
            "status": "evaluated" if metric_value is not None else "not_evaluable",
            "subject_count": len(subset),
            "class_counts": class_counts,
            "primary_metric_value": metric_value,
        }
        if metric_value is None:
            entry["reason"] = "primary metric undefined for this subgroup (e.g. a single class/outcome present)"
        else:
            entry["ci"] = _subject_level_ci(y_true, y_pred, metric_fn)
            if compute_calibration:
                entry["calibration_error"] = _calibration_for_subset(y_true, y_pred)
        by_value[value] = entry

    return {
        "dimension": dimension,
        "n_subjects_total": len(records),
        "n_subjects_present": len(present),
        "n_subjects_missing_metadata": len(missing),
        "min_subjects_per_subgroup": min_subjects,
        "by_value": by_value,
    }


def full_subgroup_diagnostics(
    records: Sequence[SubjectRecord], dimensions: Sequence[str],
    metric_fn: Callable[[Sequence[float], Sequence[float]], Optional[float]],
    *, config_path: str = DEFAULT_EVIDENCE_CONFIG_PATH,
) -> Dict:
    """Runs subgroup_report for every requested dimension — every dimension
    must be in SUPPORTED_SUBGROUP_DIMENSIONS (subgroup_report raises
    otherwise; this never silently skips an unsupported dimension request,
    it fails the whole call so the caller notices)."""
    return {d: subgroup_report(records, d, metric_fn, config_path=config_path) for d in dimensions}


# ─── Banned-phrase check ────────────────────────────────────────────────────

def assert_no_banned_claims(text: str) -> None:
    """Raises BannedFairnessClaimError if `text` contains any phrase in
    BANNED_FAIRNESS_CLAIM_PHRASES (case-insensitive, whitespace-normalized).
    Callers that render a subgroup report to human-readable text (this
    module or any wrapper around it) must pass their output through this
    before returning/printing/persisting it."""
    normalized = re.sub(r"\s+", " ", text).lower()
    for phrase in BANNED_FAIRNESS_CLAIM_PHRASES:
        if phrase in normalized:
            raise BannedFairnessClaimError(
                f"assert_no_banned_claims: text contains banned phrase {phrase!r} — no report "
                "produced by this module may claim fairness has been established."
            )


def render_subgroup_summary_text(report: Dict) -> str:
    """Human-readable summary of one subgroup_report() output. Every line
    describes what was measured (counts, metric values, evaluability) and
    never asserts a fairness conclusion — checked directly by
    assert_no_banned_claims() before returning."""
    lines = [
        f"Subgroup dimension: {report['dimension']}",
        f"Subjects evaluated: {report['n_subjects_present']}/{report['n_subjects_total']} "
        f"({report['n_subjects_missing_metadata']} missing metadata for this dimension)",
    ]
    for value, entry in sorted(report["by_value"].items()):
        if entry["status"] == "not_evaluable":
            lines.append(f"  {value}: not evaluable ({entry['reason']}) — n={entry['subject_count']}")
        else:
            metric = entry["primary_metric_value"]
            lines.append(f"  {value}: n={entry['subject_count']}, primary_metric={metric:.4f}")
    text = "\n".join(lines)
    assert_no_banned_claims(text)
    return text
