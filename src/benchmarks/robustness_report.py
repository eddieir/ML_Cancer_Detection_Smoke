"""
benchmarks/robustness_report.py — machine-readable, versioned schema for a
single source-held-out robustness evaluation, plus the cross-source
aggregate report. See ARCHITECTURE.md's "Source-held-out reporting" section
for the full data-flow this feeds into.

Every report built here is stamped development_only=True and
frozen_test_accessed=False — this module has no code path that could read
context.test_bags/split_manifest.test_subjects, so both flags are true by
construction, not by convention.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .atomic_io import atomic_write_json

ROBUSTNESS_REPORT_SCHEMA_VERSION = "1.0"

_REQUIRED_FIELDS = (
    "schema_version", "development_only", "frozen_test_accessed", "task", "model", "strategy",
    "dataset_manifest_fingerprint", "source_split_manifest_fingerprint", "preprocessing_fingerprint",
    "module_fingerprint", "model_fingerprint", "calibration_fingerprint", "held_out_source",
    "eligibility", "development_sources", "metrics", "calibration", "uncertainty", "domain_shift",
    "biological_stability", "comparisons", "limitations",
)


def _sha256_json(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass
class RobustnessReport:
    schema_version: str
    development_only: bool
    frozen_test_accessed: bool
    task: str
    model: str
    strategy: str
    dataset_manifest_fingerprint: Optional[str]
    source_split_manifest_fingerprint: Optional[str]
    preprocessing_fingerprint: Optional[str]
    module_fingerprint: Optional[str]
    model_fingerprint: Optional[str]
    calibration_fingerprint: Optional[str]
    held_out_source: str
    eligibility: Dict
    development_sources: List[str]
    metrics: Dict = field(default_factory=dict)
    calibration: Dict = field(default_factory=dict)
    uncertainty: Dict = field(default_factory=dict)
    domain_shift: Dict = field(default_factory=dict)
    biological_stability: Dict = field(default_factory=dict)
    comparisons: List[Dict] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    label_state: Dict = field(default_factory=dict)
    seed: Optional[int] = None
    gene_list_fingerprint: Optional[str] = None
    source_policy_fingerprint: Optional[str] = None
    domain_vocabulary_fingerprint: Optional[str] = None
    domain_head_fingerprint: Optional[str] = None
    environment_fingerprint: Optional[str] = None

    def fingerprint(self) -> str:
        return _sha256_json(self.to_dict(include_fingerprint=False))

    def to_dict(self, include_fingerprint: bool = True) -> dict:
        d = dict(self.__dict__)
        if include_fingerprint:
            d["report_fingerprint"] = self.fingerprint()
        return d


class RobustnessReportValidationError(ValueError):
    """Raised when a robustness report dict does not satisfy the schema
    contract — missing a required field, or claiming frozen-test access."""


def validate_robustness_report(d: Dict) -> None:
    missing = [f for f in _REQUIRED_FIELDS if f not in d]
    if missing:
        raise RobustnessReportValidationError(f"robustness report missing required field(s): {missing}")
    if d["development_only"] is not True:
        raise RobustnessReportValidationError("robustness report must have development_only=True")
    if d["frozen_test_accessed"] is not False:
        raise RobustnessReportValidationError("robustness report must have frozen_test_accessed=False")
    if d["schema_version"] != ROBUSTNESS_REPORT_SCHEMA_VERSION:
        raise RobustnessReportValidationError(
            f"robustness report schema_version={d['schema_version']!r} != "
            f"expected {ROBUSTNESS_REPORT_SCHEMA_VERSION!r}"
        )


def build_robustness_report(
    task: str, model: str, strategy: str, held_out_source: str, eligibility: Dict,
    development_sources: List[str], metrics: Optional[Dict] = None, calibration: Optional[Dict] = None,
    uncertainty: Optional[Dict] = None, domain_shift: Optional[Dict] = None,
    biological_stability: Optional[Dict] = None, comparisons: Optional[List[Dict]] = None,
    limitations: Optional[List[str]] = None, dataset_manifest_fingerprint: Optional[str] = None,
    source_split_manifest_fingerprint: Optional[str] = None, preprocessing_fingerprint: Optional[str] = None,
    module_fingerprint: Optional[str] = None, model_fingerprint: Optional[str] = None,
    calibration_fingerprint: Optional[str] = None, label_state: Optional[Dict] = None,
    seed: Optional[int] = None, gene_list_fingerprint: Optional[str] = None,
    source_policy_fingerprint: Optional[str] = None, domain_vocabulary_fingerprint: Optional[str] = None,
    domain_head_fingerprint: Optional[str] = None, environment_fingerprint: Optional[str] = None,
) -> RobustnessReport:
    return RobustnessReport(
        schema_version=ROBUSTNESS_REPORT_SCHEMA_VERSION, development_only=True, frozen_test_accessed=False,
        task=task, model=model, strategy=strategy, dataset_manifest_fingerprint=dataset_manifest_fingerprint,
        source_split_manifest_fingerprint=source_split_manifest_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint, module_fingerprint=module_fingerprint,
        model_fingerprint=model_fingerprint, calibration_fingerprint=calibration_fingerprint,
        held_out_source=held_out_source, eligibility=eligibility, development_sources=list(development_sources),
        metrics=metrics or {}, calibration=calibration or {}, uncertainty=uncertainty or {},
        domain_shift=domain_shift or {}, biological_stability=biological_stability or {},
        seed=seed, gene_list_fingerprint=gene_list_fingerprint, source_policy_fingerprint=source_policy_fingerprint,
        domain_vocabulary_fingerprint=domain_vocabulary_fingerprint, domain_head_fingerprint=domain_head_fingerprint,
        environment_fingerprint=environment_fingerprint,
        comparisons=comparisons or [], limitations=limitations or [], label_state=label_state or {},
    )


def write_robustness_report(path, report: RobustnessReport) -> str:
    """Atomic write + immediate reload-and-verify, mirroring
    reporting.write_json's contract elsewhere in this package. Returns the
    written file's own SHA-256."""
    d = report.to_dict()
    atomic_write_json(path, d)
    with open(path) as f:
        reloaded = json.load(f)
    if reloaded != d:
        raise RuntimeError(f"write_robustness_report: reload mismatch at {path} — write was not faithful.")
    validate_robustness_report(reloaded)
    import pathlib
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ─── Cross-source aggregation ───────────────────────────────────────────────

def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def aggregate_source_reports(
    reports: List[Dict], metric_key: str, subject_counts: Optional[Dict[str, int]] = None,
) -> Dict:
    """
    Aggregate per-source reports (each a RobustnessReport.to_dict()) into a
    cross-source summary — never a single pooled average that could hide a
    poor-performing source. subject_counts (held-out subject count per
    source) is required for the subject-weighted average, which is reported
    strictly SECONDARY to the (unweighted) macro-source average.
    """
    evaluated = [r for r in reports if r.get("metrics", {}).get(metric_key) is not None]
    ineligible = [r for r in reports if r not in evaluated]

    values = {r["held_out_source"]: r["metrics"][metric_key] for r in evaluated}
    if not values:
        return {
            "metric": metric_key, "n_evaluated_sources": 0,
            "n_ineligible_sources": len(ineligible),
            "ineligible_sources": [{"source": r["held_out_source"], "eligibility": r.get("eligibility")}
                                     for r in ineligible],
            "status": "insufficient_evidence",
            "reason": f"no source produced a defined {metric_key!r}",
        }

    sorted_items = sorted(values.items(), key=lambda kv: kv[1])
    vals_sorted = [v for _, v in sorted_items]
    macro_avg = sum(vals_sorted) / len(vals_sorted)
    worst_source, worst_value = sorted_items[0]
    best_source, best_value = sorted_items[-1]

    subject_weighted = None
    if subject_counts:
        total_w = sum(subject_counts.get(s, 0) for s in values)
        if total_w > 0:
            subject_weighted = sum(values[s] * subject_counts.get(s, 0) for s in values) / total_w

    return {
        "metric": metric_key,
        "n_evaluated_sources": len(evaluated),
        "n_ineligible_sources": len(ineligible),
        "ineligible_sources": [{"source": r["held_out_source"], "eligibility": r.get("eligibility")}
                                 for r in ineligible],
        "per_source": dict(values),
        "macro_source_average": macro_avg,
        "subject_weighted_average": subject_weighted,
        "worst_source": {"source": worst_source, "value": worst_value},
        "best_source": {"source": best_source, "value": best_value},
        "range": best_value - worst_value,
        "median": _percentile(vals_sorted, 0.5),
        "iqr": [_percentile(vals_sorted, 0.25), _percentile(vals_sorted, 0.75)],
    }


def build_aggregate_report(
    task: str, model: str, strategy: str, per_source_reports: List[Dict], primary_metric: str,
    subject_counts: Optional[Dict[str, int]] = None,
) -> Dict:
    return {
        "schema_version": ROBUSTNESS_REPORT_SCHEMA_VERSION, "development_only": True,
        "frozen_test_accessed": False, "task": task, "model": model, "strategy": strategy,
        "n_sources_considered": len(per_source_reports),
        "primary_metric_summary": aggregate_source_reports(per_source_reports, primary_metric, subject_counts),
        "per_source_reports": per_source_reports,
    }
