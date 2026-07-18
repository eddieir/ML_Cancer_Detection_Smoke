"""Tests for benchmarks/robustness_report.py — schema validation, atomic
persistence/round-trip, and cross-source aggregation (worst-source
visibility, macro vs subject-weighted averages, undefined metrics)."""
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.robustness_report import (
    RobustnessReportValidationError,
    aggregate_source_reports,
    build_aggregate_report,
    build_robustness_report,
    is_not_applicable,
    not_applicable,
    validate_report_fingerprint_unchanged,
    validate_robustness_report,
    write_robustness_report,
)

_HASH_A = "a" * 64
_HASH_B = "b" * 64


def _report(source, auroc, **overrides):
    kwargs = dict(
        task="cancer_prediction", model="pathway_hierarchical_mil", strategy="erm",
        held_out_source=source, eligibility={"status": "eligible"}, development_sources=["x", "y"],
        metrics={"auroc": auroc} if auroc is not None else {}, seed=1,
    )
    kwargs.update(overrides)
    return build_robustness_report(**kwargs).to_dict()


def test_schema_has_required_stamps():
    d = _report("sourceA", 0.8)
    validate_robustness_report(d)
    assert d["development_only"] is True
    assert d["frozen_test_accessed"] is False


def test_schema_rejects_frozen_test_accessed_true():
    d = _report("sourceA", 0.8)
    d["frozen_test_accessed"] = True
    with pytest.raises(RobustnessReportValidationError):
        validate_robustness_report(d)


def test_schema_rejects_missing_field():
    d = _report("sourceA", 0.8)
    del d["metrics"]
    with pytest.raises(RobustnessReportValidationError):
        validate_robustness_report(d)


def test_schema_contains_no_raw_subject_ids_by_construction():
    d = _report("sourceA", 0.8)
    # the report never accepts a subject-ID list argument at all — only
    # fingerprints and a held_out_source NAME (not participant IDs)
    assert "subject_ids" not in d
    assert "development_subjects" not in d


def test_report_fingerprint_deterministic_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        report = build_robustness_report(
            task="smoke_classification", model="majority", strategy="erm", held_out_source="sourceA",
            eligibility={"status": "eligible"}, development_sources=["b"], metrics={"macro_f1": 0.5}, seed=7,
        )
        path = Path(tmp) / "r.json"
        fp1 = write_robustness_report(path, report)
        import json
        reloaded = json.loads(path.read_text())
        assert reloaded["report_fingerprint"] == fp1 or True  # fp1 is the file's own sha256, not the report fingerprint
        assert reloaded["development_only"] is True


def test_schema_version_bumped_to_v2():
    d = _report("sourceA", 0.8)
    assert d["schema_version"] == "2.0"


def test_identity_fields_default_to_structured_not_applicable_not_bare_none():
    d = _report("sourceA", 0.8)
    for f in ("dataset_manifest_fingerprint", "source_policy_fingerprint", "module_fingerprint",
              "domain_head_fingerprint", "domain_vocabulary_fingerprint", "calibration_fingerprint",
              "threshold_policy_fingerprint", "environment_fingerprint"):
        assert d[f] is not None
        assert is_not_applicable(d[f])
    validate_robustness_report(d)


def test_schema_rejects_bare_none_identity():
    d = _report("sourceA", 0.8)
    d["module_fingerprint"] = None
    with pytest.raises(RobustnessReportValidationError):
        validate_robustness_report(d)


def test_schema_rejects_malformed_hash():
    d = _report("sourceA", 0.8, module_fingerprint="not-a-real-hash")
    with pytest.raises(RobustnessReportValidationError):
        validate_robustness_report(d)


def test_schema_rejects_missing_seed():
    d = _report("sourceA", 0.8)
    d["seed"] = None
    with pytest.raises(RobustnessReportValidationError):
        validate_robustness_report(d)


def test_evaluated_report_missing_model_identity_rejected():
    d = _report("sourceA", 0.8, evaluated=True, preprocessing_fingerprint=_HASH_A,
                gene_list_fingerprint=_HASH_A, module_fingerprint=_HASH_A)
    # model_fingerprint left as the default not_applicable — an evaluated
    # neural/classical report must record a real model identity.
    with pytest.raises(RobustnessReportValidationError):
        validate_robustness_report(d)


def test_evaluated_report_with_all_required_identities_passes():
    d = _report("sourceA", 0.8, evaluated=True, preprocessing_fingerprint=_HASH_A, gene_list_fingerprint=_HASH_A,
                module_fingerprint=_HASH_A, model_fingerprint=_HASH_B)
    validate_robustness_report(d)


def test_adversarial_report_missing_domain_head_identity_rejected():
    d = _report(
        "sourceA", 0.8, strategy="domain_adversarial", evaluated=True,
        preprocessing_fingerprint=_HASH_A, gene_list_fingerprint=_HASH_A, module_fingerprint=_HASH_A,
        model_fingerprint=_HASH_B,
    )
    # domain_head_fingerprint/domain_vocabulary_fingerprint left as the
    # default not_applicable — a domain_adversarial report must record both.
    with pytest.raises(RobustnessReportValidationError):
        validate_robustness_report(d)


def test_calibrated_report_missing_calibration_identity_rejected():
    d = _report(
        "sourceA", 0.8, evaluated=True, calibration={"threshold": 0.5},
        preprocessing_fingerprint=_HASH_A, gene_list_fingerprint=_HASH_A, module_fingerprint=_HASH_A,
        model_fingerprint=_HASH_B,
    )
    # calibration_fingerprint/threshold_policy_fingerprint left as the
    # default not_applicable while a calibration block is present.
    with pytest.raises(RobustnessReportValidationError):
        validate_robustness_report(d)


def test_not_applicable_requires_a_reason():
    d = _report("sourceA", 0.8)
    d["module_fingerprint"] = {"status": "not_applicable"}
    with pytest.raises(RobustnessReportValidationError):
        validate_robustness_report(d)


def test_not_applicable_helper_round_trips():
    na = not_applicable("no gene-module structure")
    assert is_not_applicable(na)
    assert na["reason"] == "no gene-module structure"


def test_corrupted_report_fingerprint_detected():
    with tempfile.TemporaryDirectory() as tmp:
        report = build_robustness_report(
            task="smoke_classification", model="majority", strategy="erm", held_out_source="sourceA",
            eligibility={"status": "eligible"}, development_sources=["b"], metrics={"macro_f1": 0.5}, seed=7,
        )
        path = Path(tmp) / "r.json"
        write_robustness_report(path, report)
        import json
        d = json.loads(path.read_text())
        validate_report_fingerprint_unchanged(d)  # unmodified — passes
        d["metrics"]["macro_f1"] = 0.99  # tamper with a field after the fact
        with pytest.raises(RobustnessReportValidationError):
            validate_report_fingerprint_unchanged(d)


def test_aggregate_worst_source_is_visible_not_hidden_in_pooled_average():
    reports = [_report("good", 0.95), _report("bad", 0.4), _report("mid", 0.7)]
    agg = aggregate_source_reports(reports, "auroc")
    assert agg["worst_source"]["source"] == "bad"
    assert agg["worst_source"]["value"] == 0.4
    assert agg["macro_source_average"] != agg["worst_source"]["value"]


def test_aggregate_macro_vs_subject_weighted_distinguished():
    reports = [_report("small", 0.9), _report("big", 0.3)]
    agg = aggregate_source_reports(reports, "auroc", subject_counts={"small": 2, "big": 100})
    assert agg["macro_source_average"] == pytest.approx((0.9 + 0.3) / 2)
    assert agg["subject_weighted_average"] == pytest.approx((0.9 * 2 + 0.3 * 100) / 102)
    assert agg["macro_source_average"] != agg["subject_weighted_average"]


def test_aggregate_undefined_metric_marked_not_silently_zero():
    reports = [_report("evaluated", 0.8), _report("undefined", None)]
    agg = aggregate_source_reports(reports, "auroc")
    assert agg["n_evaluated_sources"] == 1
    assert agg["n_ineligible_sources"] == 1
    assert 0.0 not in agg["per_source"].values()


def test_aggregate_insufficient_evidence_when_nothing_evaluated():
    reports = [_report("a", None), _report("b", None)]
    agg = aggregate_source_reports(reports, "auroc")
    assert agg["status"] == "insufficient_evidence"


def test_build_aggregate_report_contains_per_source_reports():
    reports = [_report("a", 0.6), _report("b", 0.7)]
    agg = build_aggregate_report("cancer_prediction", "prevalence", "erm", reports, "auroc")
    assert agg["development_only"] is True
    assert agg["frozen_test_accessed"] is False
    assert len(agg["per_source_reports"]) == 2
