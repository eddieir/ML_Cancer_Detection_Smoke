"""
benchmarks/ablation_report.py — versioned schema, validator, and atomic
writer for domain_robustness_ablation.py's output.

Mirrors robustness_report.py's validate-before-persist contract:
build_ablation_report validates the WHOLE structure — including recursively
validating every nested per-source robustness report via
validate_per_source_reports — before it ever returns, and
write_ablation_report re-validates the entire reloaded structure immediately
after an atomic write. Nothing calling either function needs to remember to
validate separately; the production path enforces it by construction.
"""

import hashlib
import json
from typing import Dict, Sequence

from .atomic_io import atomic_write_json
from .robustness_report import RobustnessReportValidationError, validate_per_source_reports

ABLATION_REPORT_SCHEMA_VERSION = "1.0"

_TOP_LEVEL_REQUIRED_FIELDS = (
    "schema_version", "task", "primary_metric", "variants", "seeds", "development_only",
    "frozen_test_accessed", "results", "paired_comparison_vs_erm", "aggregate_fingerprint",
)

_PAIRED_COMPARISON_EVALUATED_FIELDS = (
    "n_common_source_seed_pairs", "mean_paired_difference", "median_paired_difference",
    "wins", "losses", "ties",
)


class AblationReportValidationError(ValueError):
    """Raised when a domain-robustness ablation report dict does not
    satisfy its schema contract — a missing top-level field, a malformed
    per-variant/per-seed/paired-comparison structure, an invalid nested
    per-source robustness report, or an aggregate_fingerprint that no
    longer matches the report's own content."""


def _sha256_json(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _validate_seed_result(variant: str, seed, seed_result: Dict) -> None:
    if not isinstance(seed_result, dict):
        raise AblationReportValidationError(
            f"ablation report variant={variant!r} seed={seed!r}: result must be a dict"
        )
    if seed_result.get("status") == "not_evaluable":
        if not seed_result.get("reason"):
            raise AblationReportValidationError(
                f"ablation report variant={variant!r} seed={seed!r}: not_evaluable result missing a reason"
            )
        return
    missing = [f for f in ("per_source", "aggregate") if f not in seed_result]
    if missing:
        raise AblationReportValidationError(
            f"ablation report variant={variant!r} seed={seed!r}: missing field(s) {missing}"
        )
    per_source = seed_result["per_source"]
    if not isinstance(per_source, dict):
        raise AblationReportValidationError(
            f"ablation report variant={variant!r} seed={seed!r}: per_source must be a dict keyed by source name"
        )
    # The one choke point every nested per-source robustness report goes
    # through — the same full schema-v2 validation build_aggregate_report
    # itself uses, never merely exposed for a test to remember to call.
    # Re-raised as AblationReportValidationError so every rejection at this
    # level shares one exception type, with the original message preserved.
    try:
        validate_per_source_reports(list(per_source.values()))
    except RobustnessReportValidationError as e:
        raise AblationReportValidationError(
            f"ablation report variant={variant!r} seed={seed!r}: nested per-source report failed "
            f"schema-v2 validation: {e}"
        ) from e
    aggregate = seed_result["aggregate"]
    if not isinstance(aggregate, dict) or "metric" not in aggregate:
        raise AblationReportValidationError(
            f"ablation report variant={variant!r} seed={seed!r}: aggregate is missing its own 'metric' field"
        )
    if aggregate.get("status") == "insufficient_evidence":
        if not aggregate.get("reason"):
            raise AblationReportValidationError(
                f"ablation report variant={variant!r} seed={seed!r}: aggregate insufficient_evidence "
                "missing a reason"
            )


def _validate_paired_comparison(paired: Dict) -> None:
    if not isinstance(paired, dict):
        raise AblationReportValidationError("ablation report paired_comparison_vs_erm must be a dict")
    if paired.get("status") == "insufficient_evidence":
        if not paired.get("reason"):
            raise AblationReportValidationError(
                "ablation report paired_comparison_vs_erm: insufficient_evidence result missing a reason"
            )
        return
    for variant, comparison in paired.items():
        if not isinstance(comparison, dict) or "status" not in comparison:
            raise AblationReportValidationError(
                f"ablation report paired_comparison_vs_erm[{variant!r}] is missing its own 'status' field"
            )
        status = comparison["status"]
        if status == "insufficient_evidence":
            if not comparison.get("reason"):
                raise AblationReportValidationError(
                    f"ablation report paired_comparison_vs_erm[{variant!r}]: insufficient_evidence "
                    "result missing a reason"
                )
        elif status == "evaluated":
            missing = [f for f in _PAIRED_COMPARISON_EVALUATED_FIELDS if f not in comparison]
            if missing:
                raise AblationReportValidationError(
                    f"ablation report paired_comparison_vs_erm[{variant!r}] missing field(s) {missing}"
                )
        else:
            raise AblationReportValidationError(
                f"ablation report paired_comparison_vs_erm[{variant!r}] has unrecognized status {status!r}"
            )


def _seed_keys_match(per_seed: Dict, seeds: Sequence) -> bool:
    # per_seed's keys are ints in the freshly-built in-memory report but
    # become strings after a JSON round trip (write_ablation_report reloads
    # from disk) — both are valid, so compare by string form.
    return {str(k) for k in per_seed} == {str(s) for s in seeds}


def _validate_structure(report: Dict) -> None:
    """Every check EXCEPT the aggregate_fingerprint field's presence/value
    — used both by build_ablation_report (before the fingerprint exists)
    and by validate_ablation_report (which additionally requires it)."""
    base_required = [f for f in _TOP_LEVEL_REQUIRED_FIELDS if f != "aggregate_fingerprint"]
    missing = [f for f in base_required if f not in report]
    if missing:
        raise AblationReportValidationError(f"ablation report missing required top-level field(s): {missing}")
    if report["schema_version"] != ABLATION_REPORT_SCHEMA_VERSION:
        raise AblationReportValidationError(
            f"ablation report schema_version={report['schema_version']!r} != "
            f"expected {ABLATION_REPORT_SCHEMA_VERSION!r}"
        )
    if report["task"] not in ("smoke", "cancer"):
        raise AblationReportValidationError(
            f"ablation report task must be 'smoke' or 'cancer', got {report['task']!r}"
        )
    if report["development_only"] is not True:
        raise AblationReportValidationError("ablation report must have development_only=True")
    if report["frozen_test_accessed"] is not False:
        raise AblationReportValidationError("ablation report must have frozen_test_accessed=False")
    if not isinstance(report["variants"], list) or not report["variants"]:
        raise AblationReportValidationError("ablation report variants must be a non-empty list")
    if not isinstance(report["seeds"], list) or not report["seeds"]:
        raise AblationReportValidationError("ablation report seeds must be a non-empty list")
    if not isinstance(report["results"], dict) or set(report["results"]) != set(report["variants"]):
        raise AblationReportValidationError(
            "ablation report results keys must exactly match the declared variants list"
        )
    for variant, variant_result in report["results"].items():
        if not isinstance(variant_result, dict) or "per_seed" not in variant_result:
            raise AblationReportValidationError(
                f"ablation report results[{variant!r}] is missing its own 'per_seed' field"
            )
        per_seed = variant_result["per_seed"]
        if not isinstance(per_seed, dict) or not _seed_keys_match(per_seed, report["seeds"]):
            raise AblationReportValidationError(
                f"ablation report results[{variant!r}]['per_seed'] keys must exactly match the "
                "declared seeds list"
            )
        for seed, seed_result in per_seed.items():
            _validate_seed_result(variant, seed, seed_result)
    _validate_paired_comparison(report["paired_comparison_vs_erm"])


def validate_ablation_report(report: Dict) -> None:
    """Full structural validation, requiring aggregate_fingerprint to be
    present (its VALUE is checked separately by
    validate_ablation_report_fingerprint_unchanged)."""
    if "aggregate_fingerprint" not in report:
        raise AblationReportValidationError("ablation report missing required field: aggregate_fingerprint")
    _validate_structure(report)


def validate_ablation_report_fingerprint_unchanged(report: Dict) -> None:
    """Re-derives aggregate_fingerprint from every OTHER field in `report`
    and raises if it disagrees with the stored value — detects any
    post-hoc tampering with top-level metadata, a variant/seed result, a
    nested per-source report, or the paired comparison after the ablation
    report was built."""
    if "aggregate_fingerprint" not in report:
        raise AblationReportValidationError("ablation report missing aggregate_fingerprint")
    stored = report["aggregate_fingerprint"]
    recomputed = _sha256_json({k: v for k, v in report.items() if k != "aggregate_fingerprint"})
    if stored != recomputed:
        raise AblationReportValidationError(
            "ablation report aggregate_fingerprint does not match its own content — a field (top-level "
            "metadata, a variant/seed result, a nested per-source report, or the paired comparison) was "
            "altered after the report was built."
        )


def build_ablation_report(
    task: str, primary_metric: str, variants: Sequence[str], seeds: Sequence[int],
    results: Dict, paired_comparison_vs_erm: Dict,
) -> Dict:
    """
    Assembles, validates (including every nested per-source report,
    recursively), fingerprints, and re-validates the ablation report — the
    production enforcement point every caller goes through, mirroring
    robustness_report.build_aggregate_report's contract for the ablation
    report's own (materially different) top-level shape.
    """
    report = {
        "schema_version": ABLATION_REPORT_SCHEMA_VERSION, "task": task, "primary_metric": primary_metric,
        "variants": list(variants), "seeds": list(seeds), "development_only": True,
        "frozen_test_accessed": False, "results": results,
        "paired_comparison_vs_erm": paired_comparison_vs_erm,
    }
    _validate_structure(report)
    report["aggregate_fingerprint"] = _sha256_json(report)
    validate_ablation_report(report)
    validate_ablation_report_fingerprint_unchanged(report)
    return report


def write_ablation_report(path, report: Dict) -> str:
    """Atomic write + immediate reload-and-verify + full recursive
    validation, mirroring write_aggregate_report's contract for the
    ablation report shape. Returns the written file's own SHA-256.

    per_seed dicts are keyed by integer seeds in the freshly-built
    in-memory report but JSON object keys are always strings — `expected`
    below round-trips `report` through the same json encode/decode step so
    the reload-faithfulness check compares like with like instead of
    spuriously failing on int-vs-str seed keys.
    """
    atomic_write_json(path, report)
    with open(path) as f:
        reloaded = json.load(f)
    expected = json.loads(json.dumps(report, sort_keys=True, default=str))
    if reloaded != expected:
        raise RuntimeError(f"write_ablation_report: reload mismatch at {path} — write was not faithful.")
    validate_ablation_report(reloaded)
    validate_ablation_report_fingerprint_unchanged(reloaded)
    import pathlib
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
