"""Tests for benchmarks/ablation_report.py — the versioned schema,
validator, and atomic writer for domain_robustness_ablation.py's output."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.ablation_report import (
    AblationReportValidationError,
    build_ablation_report,
    validate_ablation_report,
    validate_ablation_report_fingerprint_unchanged,
    write_ablation_report,
)
from benchmarks.robustness_report import build_robustness_report


def _per_source_report(source):
    return build_robustness_report(
        task="cancer_prediction", model=None, strategy="erm", held_out_source=source,
        eligibility={"status": "eligible"}, development_sources=["x"], metrics={}, seed=1,
    ).to_dict()


def _minimal_paired_comparison():
    return {
        "source_balanced": {
            "status": "evaluated", "n_common_source_seed_pairs": 2, "mean_paired_difference": 0.01,
            "median_paired_difference": 0.01, "wins": 1, "losses": 1, "ties": 0,
            "per_source_seed_difference": {},
        },
        "coral": {"status": "insufficient_evidence", "reason": "only 1 common pair"},
    }


def _minimal_results():
    per_source = {"a": _per_source_report("a"), "b": _per_source_report("b")}
    aggregate = {"metric": "auroc", "n_evaluated_sources": 0, "n_ineligible_sources": 2,
                 "ineligible_sources": [], "status": "insufficient_evidence", "reason": "no source evaluated"}
    return {
        "erm": {"per_seed": {1: {"per_source": per_source, "aggregate": aggregate}}},
        "source_balanced": {"per_seed": {1: {"per_source": per_source, "aggregate": aggregate}}},
        "coral": {"per_seed": {1: {"status": "not_evaluable", "reason": "not run"}}},
    }


_VARIANTS = ["erm", "source_balanced", "coral"]


def _build():
    return build_ablation_report("cancer", "auroc", _VARIANTS, [1], _minimal_results(), _minimal_paired_comparison())


def test_build_ablation_report_validates_by_construction():
    report = _build()
    assert report["schema_version"] == "1.0"
    validate_ablation_report(report)
    validate_ablation_report_fingerprint_unchanged(report)


def test_atomic_round_trip_and_reload_validates():
    with tempfile.TemporaryDirectory() as tmp:
        report = _build()
        path = Path(tmp) / "ablation.json"
        file_sha = write_ablation_report(path, report)
        assert file_sha == hashlib.sha256(path.read_bytes()).hexdigest()
        reloaded = json.loads(path.read_text())
        validate_ablation_report(reloaded)
        validate_ablation_report_fingerprint_unchanged(reloaded)


def test_write_ablation_report_detects_file_corruption_after_write():
    with tempfile.TemporaryDirectory() as tmp:
        report = _build()
        path = Path(tmp) / "ablation.json"
        file_sha = write_ablation_report(path, report)
        with open(path, "ab") as f:
            f.write(b" ")
        assert hashlib.sha256(path.read_bytes()).hexdigest() != file_sha


def test_corruption_top_level_metadata_detected():
    report = _build()
    report["task"] = "smoke"  # tamper without recomputing aggregate_fingerprint
    with pytest.raises(AblationReportValidationError):
        validate_ablation_report_fingerprint_unchanged(report)


def test_corruption_one_variant_detected():
    report = _build()
    report["results"]["erm"]["per_seed"][1]["aggregate"]["metric"] = "tampered"
    with pytest.raises(AblationReportValidationError):
        validate_ablation_report_fingerprint_unchanged(report)


def test_corruption_one_seed_detected():
    report = _build()
    report["results"]["source_balanced"]["per_seed"][1]["per_source"]["b"]["metrics"] = {"auroc": 0.999}
    with pytest.raises(AblationReportValidationError):
        validate_ablation_report_fingerprint_unchanged(report)


def test_corruption_nested_source_report_rejected_by_construction():
    """A malformed nested per-source report must be rejected at BUILD time
    — never allowed to enter the ablation report in the first place."""
    results = _minimal_results()
    bad = dict(results["erm"]["per_seed"][1]["per_source"]["a"])
    bad["module_fingerprint"] = None  # a bare None is never valid
    results["erm"]["per_seed"][1] = {
        "per_source": {"a": bad, "b": results["erm"]["per_seed"][1]["per_source"]["b"]},
        "aggregate": results["erm"]["per_seed"][1]["aggregate"],
    }
    with pytest.raises(AblationReportValidationError):
        build_ablation_report("cancer", "auroc", _VARIANTS, [1], results, _minimal_paired_comparison())


def test_corruption_nested_source_report_rejected_after_the_fact():
    """A per-source report tampered with AFTER the ablation report was
    already built (fingerprint recomputation, not build-time validation)
    must also be detected."""
    report = _build()
    report["results"]["erm"]["per_seed"][1]["per_source"]["a"]["metrics"] = {"auroc": 0.5}
    with pytest.raises(AblationReportValidationError):
        validate_ablation_report_fingerprint_unchanged(report)


def test_corruption_paired_comparison_detected():
    report = _build()
    report["paired_comparison_vs_erm"]["source_balanced"]["wins"] = 999
    with pytest.raises(AblationReportValidationError):
        validate_ablation_report_fingerprint_unchanged(report)


def test_corruption_final_fingerprint_detected():
    report = _build()
    report["aggregate_fingerprint"] = "0" * 64
    with pytest.raises(AblationReportValidationError):
        validate_ablation_report_fingerprint_unchanged(report)


def test_missing_top_level_field_rejected():
    report = _build()
    del report["seeds"]
    with pytest.raises(AblationReportValidationError):
        validate_ablation_report(report)


def test_results_keys_must_match_declared_variants():
    with pytest.raises(AblationReportValidationError):
        build_ablation_report(
            "cancer", "auroc", ["erm", "source_balanced"], [1], _minimal_results(), _minimal_paired_comparison(),
        )


def test_seed_keys_must_match_declared_seeds():
    with pytest.raises(AblationReportValidationError):
        build_ablation_report("cancer", "auroc", _VARIANTS, [2], _minimal_results(), _minimal_paired_comparison())


def test_not_evaluable_seed_result_requires_a_reason():
    results = _minimal_results()
    results["coral"]["per_seed"][1] = {"status": "not_evaluable"}  # no reason
    with pytest.raises(AblationReportValidationError):
        build_ablation_report("cancer", "auroc", _VARIANTS, [1], results, _minimal_paired_comparison())


def test_insufficient_evidence_paired_comparison_requires_a_reason():
    paired = _minimal_paired_comparison()
    paired["coral"] = {"status": "insufficient_evidence"}  # no reason
    with pytest.raises(AblationReportValidationError):
        build_ablation_report("cancer", "auroc", _VARIANTS, [1], _minimal_results(), paired)


def test_invalid_task_rejected():
    with pytest.raises(AblationReportValidationError):
        build_ablation_report("not_a_task", "auroc", _VARIANTS, [1], _minimal_results(), _minimal_paired_comparison())
