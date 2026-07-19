"""
benchmarks/robustness_report.py — machine-readable, versioned schema for a
single source-held-out robustness evaluation, plus the cross-source
aggregate report. See ARCHITECTURE.md's "Source-held-out reporting" section
for the full data-flow this feeds into.

Every report built here is stamped development_only=True and
frozen_test_accessed=False — this module has no code path that could read
context.test_bags/split_manifest.test_subjects, so both flags are true by
construction, not by convention.

Schema v2 (bumped from v1) requires every scientific-identity field to be
present on every report — including on ineligible/not-evaluated branches —
and rejects a bare None for any identity that was ACTUALLY EVALUATED (a
model was fit and/or a held-out prediction was produced). A field that is
scientifically inapplicable for a given report (e.g. domain_head_fingerprint
for a classical baseline, which has no domain head) must use the structured
`not_applicable(reason)` representation below, never a bare None and never a
placeholder string invented ad hoc by a caller.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .atomic_io import atomic_write_json

ROBUSTNESS_REPORT_SCHEMA_VERSION = "2.0"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def not_applicable(reason: str) -> Dict:
    """The one sanctioned structured representation of a scientifically
    inapplicable identity field — never a bare None, never an ad hoc string
    such as "not_applicable" or "n/a" invented at a call site."""
    return {"status": "not_applicable", "reason": str(reason)}


def is_not_applicable(value) -> bool:
    return isinstance(value, dict) and value.get("status") == "not_applicable"


def _is_valid_hash(value: str) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


# Every field here must appear on EVERY report, evaluated or not — an
# ineligible/not-evaluated branch still declares an explicit not_applicable()
# for whichever identity fields it genuinely does not have.
_IDENTITY_FIELDS = (
    "dataset_manifest_fingerprint", "source_policy_fingerprint", "source_split_manifest_fingerprint",
    "preprocessing_fingerprint", "gene_list_fingerprint", "module_fingerprint", "model_fingerprint",
    "domain_head_fingerprint", "domain_vocabulary_fingerprint", "calibration_fingerprint",
    "threshold_policy_fingerprint", "environment_fingerprint",
)

# Fields that, for an EVALUATED report (a model was actually fit and applied
# to the held-out source), must be a real hash regardless of candidate kind
# — never not_applicable and never None. module_fingerprint is deliberately
# NOT here: it is only required when the candidate's REGISTRY-DERIVED kind
# is pathway-module-based (see validate_robustness_report / candidate_
# registry.py) — a classical baseline or non-module MIL candidate has no
# gene-module structure to fingerprint, and requiring one unconditionally
# would reject every legitimate classical-baseline report.
#
# dataset_manifest_fingerprint IS required here: an evaluated report means a
# dataset was actually loaded and a model actually fit against it, so a
# not_applicable dataset-manifest identity would mean the evaluation ran
# with literally no record of what it ran on. Synthetic runs must fingerprint
# their own explicit synthetic manifest rather than being exempted.
_REQUIRED_WHEN_EVALUATED = (
    "preprocessing_fingerprint", "gene_list_fingerprint", "model_fingerprint",
    "source_policy_fingerprint", "source_split_manifest_fingerprint", "environment_fingerprint",
    "dataset_manifest_fingerprint",
)

_REQUIRED_FIELDS = (
    "schema_version", "development_only", "frozen_test_accessed", "task", "model", "strategy",
    "held_out_source", "eligibility", "development_sources", "metrics", "calibration", "uncertainty",
    "domain_shift", "biological_stability", "comparisons", "limitations", "seed", "is_module_based_candidate",
) + _IDENTITY_FIELDS


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
    dataset_manifest_fingerprint: Union[str, Dict]
    source_split_manifest_fingerprint: Union[str, Dict]
    preprocessing_fingerprint: Union[str, Dict]
    module_fingerprint: Union[str, Dict]
    model_fingerprint: Union[str, Dict]
    calibration_fingerprint: Union[str, Dict]
    held_out_source: str
    eligibility: Dict
    development_sources: List[str]
    seed: Optional[int]
    gene_list_fingerprint: Union[str, Dict]
    source_policy_fingerprint: Union[str, Dict]
    domain_vocabulary_fingerprint: Union[str, Dict]
    domain_head_fingerprint: Union[str, Dict]
    environment_fingerprint: Union[str, Dict]
    threshold_policy_fingerprint: Union[str, Dict]
    metrics: Dict = field(default_factory=dict)
    calibration: Dict = field(default_factory=dict)
    uncertainty: Dict = field(default_factory=dict)
    domain_shift: Dict = field(default_factory=dict)
    biological_stability: Dict = field(default_factory=dict)
    comparisons: List[Dict] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    label_state: Dict = field(default_factory=dict)
    evaluated: bool = False
    is_module_based_candidate: bool = False

    def fingerprint(self) -> str:
        return _sha256_json(self.to_dict(include_fingerprint=False))

    def to_dict(self, include_fingerprint: bool = True) -> dict:
        d = dict(self.__dict__)
        if include_fingerprint:
            d["report_fingerprint"] = self.fingerprint()
        return d


class RobustnessReportValidationError(ValueError):
    """Raised when a robustness report dict does not satisfy the schema
    contract — missing a required field, a None/malformed required identity,
    an evaluated report missing a mandatory identity, or a fingerprint
    mismatch after a field was altered post-hoc."""


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
    if d["seed"] is None:
        raise RobustnessReportValidationError("robustness report must record a non-None seed")

    for f in _IDENTITY_FIELDS:
        v = d[f]
        if v is None:
            raise RobustnessReportValidationError(
                f"robustness report field {f!r} is a bare None — use not_applicable(reason) if this "
                "identity is genuinely inapplicable, never a bare None."
            )
        if is_not_applicable(v):
            if not v.get("reason"):
                raise RobustnessReportValidationError(f"robustness report field {f!r}: not_applicable() requires a reason")
            continue
        if not _is_valid_hash(v):
            raise RobustnessReportValidationError(
                f"robustness report field {f!r}={v!r} is neither a valid 64-hex-char SHA-256 digest "
                "nor a structured not_applicable() value."
            )

    if d.get("evaluated"):
        missing_evaluated = [f for f in _REQUIRED_WHEN_EVALUATED if is_not_applicable(d[f])]
        if missing_evaluated:
            raise RobustnessReportValidationError(
                f"robustness report is marked evaluated=True but field(s) {missing_evaluated} are "
                "not_applicable — an evaluated report must record real model/preprocessing/gene/"
                "policy/environment/dataset-manifest identities regardless of candidate kind."
            )
        # Candidate kind is RECOMPUTED from the canonical model registry
        # (candidate_registry.py) — never trusted from the report's own
        # is_module_based_candidate flag. An unrecognized model name raises
        # UnknownCandidateNameError (a typed ValueError subclass) rather than
        # defaulting to any kind. The report may still PERSIST its own
        # is_module_based_candidate value, but validation rejects any
        # disagreement with the registry-derived kind outright.
        from .candidate_registry import is_module_based

        derived_is_module_based = is_module_based(d["model"])
        declared_is_module_based = bool(d.get("is_module_based_candidate"))
        if declared_is_module_based != derived_is_module_based:
            raise RobustnessReportValidationError(
                f"robustness report declares is_module_based_candidate={declared_is_module_based!r} "
                f"for model={d['model']!r}, but the canonical model registry derives "
                f"is_module_based={derived_is_module_based!r} for that name — a candidate cannot "
                "self-declare its own kind; it is derived from the model registry and checked here."
            )
        if derived_is_module_based:
            if is_not_applicable(d["module_fingerprint"]):
                raise RobustnessReportValidationError(
                    "robustness report's model is a registry-derived module-based/pathway candidate "
                    "but module_fingerprint is not_applicable — a module-based/pathway candidate must "
                    "record a real module identity."
                )
        else:
            if not is_not_applicable(d["module_fingerprint"]):
                raise RobustnessReportValidationError(
                    "robustness report's model is a registry-derived non-module candidate but "
                    "module_fingerprint is a real hash — a classical baseline or non-module MIL "
                    "candidate has no gene-module structure and must record a structured "
                    "not_applicable() reason instead."
                )
        strategy = d.get("strategy")
        # Only pathway_hierarchical_mil actually has a domain-adversarial
        # attachment point (see mil_registry.py) — a domain_adversarial run
        # can still legitimately select a classical baseline or non-module
        # MIL candidate as its winner (e.g. that source's OOF sweep simply
        # favored logistic regression over the pathway model), and such a
        # winner has no domain head to fingerprint regardless of the
        # STRATEGY that was requested. Gating on the registry-derived
        # candidate kind (not the bare strategy string) avoids rejecting
        # every legitimate classical-baseline winner under
        # strategy=domain_adversarial.
        if strategy == "domain_adversarial" and derived_is_module_based:
            for f in ("domain_head_fingerprint", "domain_vocabulary_fingerprint"):
                if is_not_applicable(d[f]):
                    raise RobustnessReportValidationError(
                        f"robustness report strategy='domain_adversarial' with a module-based/pathway "
                        f"winning candidate but {f!r} is not_applicable — an adversarial report whose "
                        "winning candidate actually has a domain head must record real domain-head/"
                        "vocabulary identities."
                    )
        if d.get("calibration"):
            for f in ("calibration_fingerprint", "threshold_policy_fingerprint"):
                if is_not_applicable(d[f]):
                    raise RobustnessReportValidationError(
                        f"robustness report has a non-empty calibration block but {f!r} is "
                        "not_applicable — a calibrated cancer report must record real calibration/"
                        "threshold identities."
                    )


def validate_report_fingerprint_unchanged(d: Dict) -> None:
    """Re-derives the report_fingerprint from every OTHER field in `d` and
    raises if it disagrees with the stored value — detects post-hoc
    tampering with any field after the report was written."""
    if "report_fingerprint" not in d:
        raise RobustnessReportValidationError("robustness report missing report_fingerprint")
    stored = d["report_fingerprint"]
    recomputed = _sha256_json({k: v for k, v in d.items() if k != "report_fingerprint"})
    if stored != recomputed:
        raise RobustnessReportValidationError(
            "robustness report report_fingerprint does not match its own content — a field was "
            "altered after the report was written."
        )


def build_robustness_report(
    task: str, model: str, strategy: str, held_out_source: str, eligibility: Dict,
    development_sources: List[str], metrics: Optional[Dict] = None, calibration: Optional[Dict] = None,
    uncertainty: Optional[Dict] = None, domain_shift: Optional[Dict] = None,
    biological_stability: Optional[Dict] = None, comparisons: Optional[List[Dict]] = None,
    limitations: Optional[List[str]] = None, dataset_manifest_fingerprint: Union[str, Dict, None] = None,
    source_split_manifest_fingerprint: Union[str, Dict, None] = None,
    preprocessing_fingerprint: Union[str, Dict, None] = None,
    module_fingerprint: Union[str, Dict, None] = None, model_fingerprint: Union[str, Dict, None] = None,
    calibration_fingerprint: Union[str, Dict, None] = None, label_state: Optional[Dict] = None,
    seed: Optional[int] = None, gene_list_fingerprint: Union[str, Dict, None] = None,
    source_policy_fingerprint: Union[str, Dict, None] = None,
    domain_vocabulary_fingerprint: Union[str, Dict, None] = None,
    domain_head_fingerprint: Union[str, Dict, None] = None,
    environment_fingerprint: Union[str, Dict, None] = None,
    threshold_policy_fingerprint: Union[str, Dict, None] = None,
    evaluated: bool = False, is_module_based_candidate: bool = False,
) -> RobustnessReport:
    def _na(v, reason):
        return v if v is not None else not_applicable(reason)

    return RobustnessReport(
        schema_version=ROBUSTNESS_REPORT_SCHEMA_VERSION, development_only=True, frozen_test_accessed=False,
        task=task, model=model, strategy=strategy,
        dataset_manifest_fingerprint=_na(dataset_manifest_fingerprint, "no dataset manifest supplied"),
        source_split_manifest_fingerprint=_na(source_split_manifest_fingerprint, "split manifest not built for this branch"),
        preprocessing_fingerprint=_na(preprocessing_fingerprint, "no model was fit for this source"),
        module_fingerprint=_na(module_fingerprint, "candidate has no gene-module structure"),
        model_fingerprint=_na(model_fingerprint, "no model was fit for this source"),
        calibration_fingerprint=_na(calibration_fingerprint, "no probability calibration performed for this report"),
        threshold_policy_fingerprint=_na(threshold_policy_fingerprint, "no decision threshold selected for this report"),
        held_out_source=held_out_source, eligibility=eligibility, development_sources=list(development_sources),
        metrics=metrics or {}, calibration=calibration or {}, uncertainty=uncertainty or {},
        domain_shift=domain_shift or {}, biological_stability=biological_stability or {},
        seed=seed,
        gene_list_fingerprint=_na(gene_list_fingerprint, "no model was fit for this source"),
        source_policy_fingerprint=_na(source_policy_fingerprint, "source policy not resolved for this branch"),
        domain_vocabulary_fingerprint=_na(domain_vocabulary_fingerprint, "candidate has no domain-source vocabulary"),
        domain_head_fingerprint=_na(domain_head_fingerprint, "candidate has no domain-adversarial head"),
        environment_fingerprint=_na(environment_fingerprint, "environment fingerprint not collected for this branch"),
        comparisons=comparisons or [], limitations=limitations or [], label_state=label_state or {},
        evaluated=evaluated, is_module_based_candidate=is_module_based_candidate,
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
    validate_report_fingerprint_unchanged(reloaded)
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


_AGGREGATE_REQUIRED_FIELDS = (
    "schema_version", "development_only", "frozen_test_accessed", "task", "model", "strategy",
    "n_sources_considered", "primary_metric_summary", "per_source_reports",
)


def validate_per_source_reports(per_source_reports: List[Dict]) -> None:
    """Validates EVERY per-source report against the full schema-v2
    contract (validate_robustness_report) and its own tamper-detection
    check (validate_report_fingerprint_unchanged) — the one choke point
    every aggregate-building/persisting code path must call before a child
    report is allowed to enter an aggregate or be written to disk."""
    for r in per_source_reports:
        validate_robustness_report(r)
        validate_report_fingerprint_unchanged(r)


def validate_aggregate_report(agg: Dict) -> None:
    """Validates an aggregate report's own schema/version/stamps AND
    recursively validates every per-source report nested inside it — never
    relies on the caller (a test, the CLI) to have already done so."""
    missing = [f for f in _AGGREGATE_REQUIRED_FIELDS if f not in agg]
    if missing:
        raise RobustnessReportValidationError(f"aggregate report missing required field(s): {missing}")
    if agg["schema_version"] != ROBUSTNESS_REPORT_SCHEMA_VERSION:
        raise RobustnessReportValidationError(
            f"aggregate report schema_version={agg['schema_version']!r} != "
            f"expected {ROBUSTNESS_REPORT_SCHEMA_VERSION!r}"
        )
    if agg["development_only"] is not True:
        raise RobustnessReportValidationError("aggregate report must have development_only=True")
    if agg["frozen_test_accessed"] is not False:
        raise RobustnessReportValidationError("aggregate report must have frozen_test_accessed=False")
    if agg["n_sources_considered"] != len(agg["per_source_reports"]):
        raise RobustnessReportValidationError(
            f"aggregate report n_sources_considered={agg['n_sources_considered']!r} does not match "
            f"len(per_source_reports)={len(agg['per_source_reports'])!r}"
        )
    validate_per_source_reports(agg["per_source_reports"])


def build_aggregate_report(
    task: str, model: str, strategy: str, per_source_reports: List[Dict], primary_metric: str,
    subject_counts: Optional[Dict[str, int]] = None,
) -> Dict:
    """Builds the cross-source aggregate report. Validates every per-source
    report BEFORE it is allowed to enter the aggregate — this is the
    production enforcement point every CLI caller goes through by
    construction, not a check tests must remember to call manually."""
    validate_per_source_reports(per_source_reports)
    agg = {
        "schema_version": ROBUSTNESS_REPORT_SCHEMA_VERSION, "development_only": True,
        "frozen_test_accessed": False, "task": task, "model": model, "strategy": strategy,
        "n_sources_considered": len(per_source_reports),
        "primary_metric_summary": aggregate_source_reports(per_source_reports, primary_metric, subject_counts),
        "per_source_reports": per_source_reports,
    }
    validate_aggregate_report(agg)
    return agg


def write_aggregate_report(path, agg: Dict) -> str:
    """Atomic write + immediate reload-and-verify + full recursive
    validation, mirroring write_robustness_report's contract for the
    aggregate report shape. Returns the written file's own SHA-256."""
    atomic_write_json(path, agg)
    with open(path) as f:
        reloaded = json.load(f)
    if reloaded != agg:
        raise RuntimeError(f"write_aggregate_report: reload mismatch at {path} — write was not faithful.")
    validate_aggregate_report(reloaded)
    import pathlib
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
