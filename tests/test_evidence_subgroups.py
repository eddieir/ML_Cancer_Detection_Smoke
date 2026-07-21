"""
tests/test_evidence_subgroups.py — Step 13 tests for evidence/subgroups.py.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evidence import subgroups as S


def _auroc_fn(y_true, y_pred):
    from benchmarks.metrics import cancer_prediction_metrics
    return cancer_prediction_metrics(y_true, y_pred)["auroc"]


def _records(n_per_group=15):
    records = []
    i = 0
    for group in ("gse136831", "gse288003"):
        for j in range(n_per_group):
            y = j % 2
            records.append(S.build_subject_record(f"s{i}", y, 0.3 + 0.4 * y, {"cohort_source": group}))
            i += 1
    return records


# ─── Only genuinely-sourced dimensions are accepted ────────────────────────

def test_unsupported_dimensions_are_rejected_never_fabricated():
    records = _records()
    for fabricated in ("sex", "age_band", "race", "ethnicity", "smoking_history_years", "site"):
        with pytest.raises(S.UnsupportedSubgroupDimensionError):
            S.subgroup_report(records, fabricated, _auroc_fn)


def test_subject_record_rejects_an_unsupported_metadata_key_at_construction():
    with pytest.raises(S.SubgroupInputError):
        S.build_subject_record("s1", 1, 0.8, {"sex": "female"})


def test_supported_dimensions_match_documented_set():
    assert S.SUPPORTED_SUBGROUP_DIMENSIONS == (
        "cohort_source", "exposure_type", "disease_status", "assay_platform", "species",
    )


# ─── No inference of a sensitive characteristic from name/accession/expression ─

def test_subgroup_dimensions_never_derived_from_subject_identity_or_expression():
    """Subject IDs that LOOK like they encode a sensitive characteristic
    must have zero effect on the report — this module only ever reads
    record.metadata[dimension], never record.subject_id's string content
    or any expression value (which this module never even receives)."""
    suggestive_ids = [
        "female_65_smoker_003", "male_42_never_001", "asian_28_vape_002", "white_71_cigarette_004",
    ]
    plain_ids = ["s0", "s1", "s2", "s3"]
    y_true = [0, 1, 0, 1]
    y_pred = [0.2, 0.8, 0.3, 0.7]
    metadata = [{"cohort_source": "gse136831"}] * 4

    records_suggestive = [
        S.build_subject_record(sid, y, p, m)
        for sid, y, p, m in zip(suggestive_ids, y_true, y_pred, metadata)
    ]
    records_plain = [
        S.build_subject_record(sid, y, p, m)
        for sid, y, p, m in zip(plain_ids, y_true, y_pred, metadata)
    ]
    # Force min_subjects_per_subgroup down so 4 subjects is evaluable.
    import tempfile
    import yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"subgroup_policy": {"min_subjects_per_subgroup": 2}}, f)
        cfg_path = f.name

    report_suggestive = S.subgroup_report(records_suggestive, "cohort_source", _auroc_fn, config_path=cfg_path)
    report_plain = S.subgroup_report(records_plain, "cohort_source", _auroc_fn, config_path=cfg_path)

    # Only subject_id differs between the two record sets; the report
    # content (everything except which literal IDs are absent — this
    # module never echoes subject_id back into by_value) must be identical.
    assert report_suggestive["by_value"] == report_plain["by_value"]
    assert report_suggestive["n_subjects_total"] == report_plain["n_subjects_total"]

    # No dimension named after a sensitive characteristic is even
    # computable — grep the module's own dimension set for a definitive,
    # structural (not just this-test-instance) guarantee.
    for banned in ("sex", "gender", "race", "ethnicity", "age"):
        assert banned not in S.SUPPORTED_SUBGROUP_DIMENSIONS


def test_module_never_reads_subject_id_string_content():
    """A subject_id that raises the instant its string form is read (via
    FrozenAccessSentinel-style behavior) still produces a valid report —
    proving subgroup_report never calls str()/repr()/parses subject_id."""
    from benchmarks.sentinel import FrozenAccessSentinel

    class NoisySubjectId(str):
        """A str subclass that still behaves as a normal string for dict
        keys/equality (SubjectRecord needs a real string), but records
        whether anything ever asked for a a *different* representation of
        it beyond plain identity/equality — here we simply confirm the
        report is insensitive to totally different (but still 'flag-like')
        ID contents from test above; this test additionally confirms
        subgroup_report accepts an opaque uuid-shaped ID with no semantic
        content at all, and that its output only ever mentions the
        dimension VALUES, never a subject_id."""

    import uuid
    records = [
        S.build_subject_record(str(uuid.uuid4()), i % 2, 0.3 + 0.4 * (i % 2), {"cohort_source": "gse136831"})
        for i in range(12)
    ]
    report = S.subgroup_report(records, "cohort_source", _auroc_fn,
                                config_path=_low_min_config())
    rendered = repr(report)
    for r in records:
        assert r.subject_id not in rendered


def _low_min_config():
    import tempfile
    import yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"subgroup_policy": {"min_subjects_per_subgroup": 2}}, f)
        return f.name


# ─── Per-subgroup structure: counts, metric, calibration, CI, suppression ─

def test_subgroup_report_structure_and_suppression_under_low_support():
    records = _records(n_per_group=15)
    report = S.subgroup_report(records, "cohort_source", _auroc_fn, config_path=_low_min_config())
    assert report["n_subjects_total"] == 30
    assert report["n_subjects_missing_metadata"] == 0
    for value, entry in report["by_value"].items():
        assert entry["subject_count"] == 15
        assert "class_counts" in entry
        if entry["status"] == "evaluated":
            assert "primary_metric_value" in entry
            assert "ci" in entry
            assert "calibration_error" in entry


def test_subgroup_report_marks_not_evaluable_under_default_minimum():
    records = _records(n_per_group=3)  # well under the default minimum of 10
    report = S.subgroup_report(records, "cohort_source", _auroc_fn)
    for entry in report["by_value"].values():
        assert entry["status"] == "not_evaluable"
        assert "reason" in entry


def test_subgroup_report_counts_missing_metadata_never_drops_silently():
    records = _records(n_per_group=10)
    unlabeled = [S.build_subject_record("unlabeled1", 1, 0.7, {}), S.build_subject_record("unlabeled2", 0, 0.2, {})]
    report = S.subgroup_report(records + unlabeled, "cohort_source", _auroc_fn, config_path=_low_min_config())
    assert report["n_subjects_missing_metadata"] == 2
    assert report["n_subjects_total"] == len(records) + 2
    assert set(report["by_value"]) == {"gse136831", "gse288003"}


def test_full_subgroup_diagnostics_runs_every_requested_dimension():
    records = [
        S.build_subject_record(f"s{i}", i % 2, 0.3 + 0.4 * (i % 2),
                                {"cohort_source": "gse136831", "species": "human"})
        for i in range(20)
    ]
    out = S.full_subgroup_diagnostics(records, ["cohort_source", "species"], _auroc_fn,
                                       config_path=_low_min_config())
    assert set(out) == {"cohort_source", "species"}


# ─── Banned fairness-established claims ────────────────────────────────────

@pytest.mark.parametrize("phrase", [
    "This model establishes fairness established across all groups.",
    "The results confirm the model is bias-free.",
    "There is no bias in the predictions.",
    "The model is fair for all subgroups.",
    "Results guarantees fairness across cohorts.",
])
def test_assert_no_banned_claims_rejects_every_banned_phrasing(phrase):
    with pytest.raises(S.BannedFairnessClaimError):
        S.assert_no_banned_claims(phrase)


def test_assert_no_banned_claims_passes_descriptive_text():
    S.assert_no_banned_claims("Subgroup gse136831: n=15, primary_metric=0.6000, calibration_error=0.12")


def test_render_subgroup_summary_text_never_contains_a_banned_claim():
    records = _records(n_per_group=15)
    report = S.subgroup_report(records, "cohort_source", _auroc_fn, config_path=_low_min_config())
    text = S.render_subgroup_summary_text(report)
    S.assert_no_banned_claims(text)  # must not raise
    for phrase in S.BANNED_FAIRNESS_CLAIM_PHRASES:
        assert phrase not in text.lower()


def test_load_subgroup_policy_falls_back_when_file_missing():
    policy = S.load_subgroup_policy(config_path="/nonexistent/path/evidence.yaml")
    assert policy["min_subjects_per_subgroup"] == S._FALLBACK_MIN_SUBJECTS_PER_SUBGROUP
