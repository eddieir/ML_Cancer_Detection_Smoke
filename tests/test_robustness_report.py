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
    validate_aggregate_report,
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
    """report_fingerprint (a field INSIDE the JSON, hashing every other
    field) and file_sha256 (write_robustness_report's return value, hashing
    the serialized file's raw bytes) are two independent hashes over
    different inputs — this test checks each on its own terms rather than
    conflating them."""
    with tempfile.TemporaryDirectory() as tmp:
        report = build_robustness_report(
            task="smoke_classification", model="majority", strategy="erm", held_out_source="sourceA",
            eligibility={"status": "eligible"}, development_sources=["b"], metrics={"macro_f1": 0.5}, seed=7,
        )
        path = Path(tmp) / "r.json"
        file_sha256 = write_robustness_report(path, report)
        import hashlib
        import json

        reloaded = json.loads(path.read_text())
        assert reloaded["development_only"] is True

        # report_fingerprint must equal a fresh recomputation over the
        # SAME content, and must be deterministic given identical inputs.
        assert reloaded["report_fingerprint"] == report.fingerprint()
        report_again = build_robustness_report(
            task="smoke_classification", model="majority", strategy="erm", held_out_source="sourceA",
            eligibility={"status": "eligible"}, development_sources=["b"], metrics={"macro_f1": 0.5}, seed=7,
        )
        assert report_again.fingerprint() == report.fingerprint()

        # file_sha256 must equal a fresh SHA-256 of the actual file bytes,
        # and must NOT equal report_fingerprint (they hash different things
        # — the whole JSON document including report_fingerprint itself,
        # versus the report's own fields excluding it).
        assert file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert file_sha256 != reloaded["report_fingerprint"]


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
    d = _report(
        "sourceA", 0.8, evaluated=True, is_module_based_candidate=True,
        preprocessing_fingerprint=_HASH_A, gene_list_fingerprint=_HASH_A, module_fingerprint=_HASH_A,
        model_fingerprint=_HASH_B, source_policy_fingerprint=_HASH_A,
        source_split_manifest_fingerprint=_HASH_A, environment_fingerprint=_HASH_A,
        dataset_manifest_fingerprint=_HASH_A,
    )
    validate_robustness_report(d)


def test_evaluated_classical_baseline_requires_not_applicable_module_fingerprint():
    """A classical baseline (model="prevalence", a registry-derived
    non-module candidate) must record a structured not_applicable
    module_fingerprint — a real hash there would be scientifically
    meaningless for a model with no gene-module structure."""
    d = _report(
        "sourceA", 0.8, model="prevalence", evaluated=True,
        preprocessing_fingerprint=_HASH_A, gene_list_fingerprint=_HASH_A,
        model_fingerprint=_HASH_B, source_policy_fingerprint=_HASH_A,
        source_split_manifest_fingerprint=_HASH_A, environment_fingerprint=_HASH_A,
        dataset_manifest_fingerprint=_HASH_A,
    )
    validate_robustness_report(d)  # module_fingerprint defaults to not_applicable — passes


def test_evaluated_classical_baseline_with_real_module_hash_rejected():
    d = _report(
        "sourceA", 0.8, model="prevalence", evaluated=True,
        preprocessing_fingerprint=_HASH_A, gene_list_fingerprint=_HASH_A,
        module_fingerprint=_HASH_A,  # a classical baseline has no module structure to hash
        model_fingerprint=_HASH_B, source_policy_fingerprint=_HASH_A,
        source_split_manifest_fingerprint=_HASH_A, environment_fingerprint=_HASH_A,
        dataset_manifest_fingerprint=_HASH_A,
    )
    with pytest.raises(RobustnessReportValidationError):
        validate_robustness_report(d)


def test_evaluated_module_based_candidate_missing_module_fingerprint_rejected():
    d = _report(
        "sourceA", 0.8, evaluated=True, is_module_based_candidate=True,
        preprocessing_fingerprint=_HASH_A, gene_list_fingerprint=_HASH_A,
        model_fingerprint=_HASH_B, source_policy_fingerprint=_HASH_A,
        source_split_manifest_fingerprint=_HASH_A, environment_fingerprint=_HASH_A,
        dataset_manifest_fingerprint=_HASH_A,
    )
    with pytest.raises(RobustnessReportValidationError):
        validate_robustness_report(d)


def test_evaluated_report_declared_kind_disagreeing_with_registry_rejected():
    """model="prevalence" (a registry-derived classical baseline) declaring
    is_module_based_candidate=True must be rejected — a candidate cannot
    falsely declare a kind the model registry does not derive for it."""
    d = _report(
        "sourceA", 0.8, model="prevalence", evaluated=True, is_module_based_candidate=True,
        preprocessing_fingerprint=_HASH_A, gene_list_fingerprint=_HASH_A, module_fingerprint=_HASH_A,
        model_fingerprint=_HASH_B, source_policy_fingerprint=_HASH_A,
        source_split_manifest_fingerprint=_HASH_A, environment_fingerprint=_HASH_A,
        dataset_manifest_fingerprint=_HASH_A,
    )
    with pytest.raises(RobustnessReportValidationError):
        validate_robustness_report(d)


def test_evaluated_report_pathway_candidate_falsely_declaring_non_module_rejected():
    """model="pathway_hierarchical_mil" (a registry-derived module-based
    candidate) declaring is_module_based_candidate=False must be
    rejected — a pathway candidate cannot hide its module structure."""
    d = _report(
        "sourceA", 0.8, evaluated=True, is_module_based_candidate=False,
        preprocessing_fingerprint=_HASH_A, gene_list_fingerprint=_HASH_A,
        model_fingerprint=_HASH_B, source_policy_fingerprint=_HASH_A,
        source_split_manifest_fingerprint=_HASH_A, environment_fingerprint=_HASH_A,
        dataset_manifest_fingerprint=_HASH_A,
    )
    with pytest.raises(RobustnessReportValidationError):
        validate_robustness_report(d)


def test_evaluated_report_unknown_model_name_raises_typed_registry_error():
    """A model name absent from every canonical registry must raise a
    typed error rather than silently default to any candidate kind."""
    from benchmarks.candidate_registry import UnknownCandidateNameError

    d = _report(
        "sourceA", 0.8, model="some_never_registered_model", evaluated=True,
        preprocessing_fingerprint=_HASH_A, gene_list_fingerprint=_HASH_A, module_fingerprint=_HASH_A,
        model_fingerprint=_HASH_B, source_policy_fingerprint=_HASH_A,
        source_split_manifest_fingerprint=_HASH_A, environment_fingerprint=_HASH_A,
        dataset_manifest_fingerprint=_HASH_A,
    )
    with pytest.raises(UnknownCandidateNameError):
        validate_robustness_report(d)


def test_evaluated_report_missing_dataset_manifest_fingerprint_rejected():
    """An evaluated report must never record a not_applicable dataset
    manifest identity — a dataset was actually loaded and a model actually
    fit against it, so the manifest fingerprint must be a real hash."""
    d = _report(
        "sourceA", 0.8, model="prevalence", evaluated=True,
        preprocessing_fingerprint=_HASH_A, gene_list_fingerprint=_HASH_A,
        model_fingerprint=_HASH_B, source_policy_fingerprint=_HASH_A,
        source_split_manifest_fingerprint=_HASH_A, environment_fingerprint=_HASH_A,
        # dataset_manifest_fingerprint left at its not_applicable default
    )
    with pytest.raises(RobustnessReportValidationError):
        validate_robustness_report(d)


def test_adversarial_report_missing_domain_head_identity_rejected():
    d = _report(
        "sourceA", 0.8, strategy="domain_adversarial", evaluated=True, is_module_based_candidate=True,
        preprocessing_fingerprint=_HASH_A, gene_list_fingerprint=_HASH_A, module_fingerprint=_HASH_A,
        model_fingerprint=_HASH_B, source_policy_fingerprint=_HASH_A,
        source_split_manifest_fingerprint=_HASH_A, environment_fingerprint=_HASH_A,
        dataset_manifest_fingerprint=_HASH_A,
    )
    # domain_head_fingerprint/domain_vocabulary_fingerprint left as the
    # default not_applicable — a domain_adversarial report must record both.
    with pytest.raises(RobustnessReportValidationError):
        validate_robustness_report(d)


def test_adversarial_strategy_with_classical_baseline_winner_accepts_not_applicable_domain_head():
    """strategy='domain_adversarial' can still legitimately be won by a
    classical baseline (that source's OOF sweep simply favored logistic
    over pathway_hierarchical_mil) — only pathway_hierarchical_mil has a
    domain-adversarial head at all, so a not_applicable domain_head/
    vocabulary identity must be ACCEPTED when the winning candidate is not
    module-based, even though the requested strategy is domain_adversarial."""
    d = _report(
        "sourceA", 0.8, model="logistic", strategy="domain_adversarial", evaluated=True,
        preprocessing_fingerprint=_HASH_A, gene_list_fingerprint=_HASH_A,
        model_fingerprint=_HASH_B, source_policy_fingerprint=_HASH_A,
        source_split_manifest_fingerprint=_HASH_A, environment_fingerprint=_HASH_A,
        dataset_manifest_fingerprint=_HASH_A,
        # module_fingerprint, domain_head_fingerprint, domain_vocabulary_fingerprint
        # all left at their not_applicable defaults.
    )
    validate_robustness_report(d)


def test_calibrated_report_missing_calibration_identity_rejected():
    d = _report(
        "sourceA", 0.8, evaluated=True, is_module_based_candidate=True, calibration={"threshold": 0.5},
        preprocessing_fingerprint=_HASH_A, gene_list_fingerprint=_HASH_A, module_fingerprint=_HASH_A,
        model_fingerprint=_HASH_B, source_policy_fingerprint=_HASH_A,
        source_split_manifest_fingerprint=_HASH_A, environment_fingerprint=_HASH_A,
        dataset_manifest_fingerprint=_HASH_A,
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


def test_file_bytes_corruption_detected_independently_of_report_fingerprint():
    """Corrupting the FILE's raw bytes (disk-level corruption, not a field
    edit) must be detectable via file_sha256 — a check entirely separate
    from validate_report_fingerprint_unchanged, which only ever looks at
    already-parsed JSON content."""
    import hashlib

    with tempfile.TemporaryDirectory() as tmp:
        report = build_robustness_report(
            task="smoke_classification", model="majority", strategy="erm", held_out_source="sourceA",
            eligibility={"status": "eligible"}, development_sources=["b"], metrics={"macro_f1": 0.5}, seed=7,
        )
        path = Path(tmp) / "r.json"
        file_sha256 = write_robustness_report(path, report)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == file_sha256

        with open(path, "ab") as f:
            f.write(b" ")  # append a byte — corrupts the file without touching any JSON field value
        assert hashlib.sha256(path.read_bytes()).hexdigest() != file_sha256


def test_legitimately_rewritten_report_recomputes_a_consistent_fingerprint():
    """A NEW report built with a genuinely different field (not tampering
    with an already-written one) gets its own, internally-consistent
    report_fingerprint — this is the legitimate-rewrite case
    validate_report_fingerprint_unchanged must NOT reject."""
    with tempfile.TemporaryDirectory() as tmp:
        report_v1 = build_robustness_report(
            task="smoke_classification", model="majority", strategy="erm", held_out_source="sourceA",
            eligibility={"status": "eligible"}, development_sources=["b"], metrics={"macro_f1": 0.5}, seed=7,
        )
        path = Path(tmp) / "r.json"
        write_robustness_report(path, report_v1)

        report_v2 = build_robustness_report(
            task="smoke_classification", model="majority", strategy="erm", held_out_source="sourceA",
            eligibility={"status": "eligible"}, development_sources=["b"], metrics={"macro_f1": 0.7}, seed=7,
        )
        write_robustness_report(path, report_v2)  # legitimate overwrite, not tampering
        import json
        reloaded = json.loads(path.read_text())
        validate_report_fingerprint_unchanged(reloaded)  # must pass — this is a fresh, self-consistent report
        assert reloaded["report_fingerprint"] != report_v1.fingerprint()


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
    validate_aggregate_report(agg)  # must already validate cleanly by construction


def test_build_aggregate_report_rejects_a_malformed_per_source_report():
    """build_aggregate_report must validate every child BEFORE it enters
    the aggregate — this is the production enforcement point, not
    something only a test remembers to call."""
    good = _report("a", 0.6)
    bad = _report("b", 0.7)
    bad["module_fingerprint"] = None  # a bare None is never valid
    with pytest.raises(RobustnessReportValidationError):
        build_aggregate_report("cancer_prediction", "prevalence", "erm", [good, bad], "auroc")


def test_build_aggregate_report_rejects_a_tampered_per_source_report():
    """A child report whose report_fingerprint no longer matches its own
    content (post-hoc tampering) must be rejected before it enters an
    aggregate."""
    good = _report("a", 0.6)
    tampered = _report("b", 0.7)
    tampered["metrics"]["auroc"] = 0.99  # tamper without recomputing report_fingerprint
    with pytest.raises(RobustnessReportValidationError):
        build_aggregate_report("cancer_prediction", "prevalence", "erm", [good, tampered], "auroc")


def test_write_aggregate_report_atomic_round_trip_and_corruption_detection():
    with tempfile.TemporaryDirectory() as tmp:
        reports = [_report("a", 0.6), _report("b", 0.7)]
        agg = build_aggregate_report("cancer_prediction", "prevalence", "erm", reports, "auroc")
        path = Path(tmp) / "agg.json"
        from benchmarks.robustness_report import write_aggregate_report
        file_sha = write_aggregate_report(path, agg)
        import hashlib
        assert file_sha == hashlib.sha256(path.read_bytes()).hexdigest()

        import json
        corrupted = json.loads(path.read_text())
        corrupted["per_source_reports"][0]["module_fingerprint"] = None
        bad_path = Path(tmp) / "agg_bad.json"
        bad_path.write_text(json.dumps(corrupted))
        with pytest.raises(RobustnessReportValidationError):
            validate_aggregate_report(json.loads(bad_path.read_text()))


def test_validate_aggregate_report_rejects_n_sources_considered_mismatch():
    reports = [_report("a", 0.6), _report("b", 0.7)]
    agg = build_aggregate_report("cancer_prediction", "prevalence", "erm", reports, "auroc")
    agg["n_sources_considered"] = 99
    with pytest.raises(RobustnessReportValidationError):
        validate_aggregate_report(agg)
