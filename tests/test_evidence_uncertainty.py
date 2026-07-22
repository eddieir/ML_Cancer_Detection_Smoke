"""
tests/test_evidence_uncertainty.py — Step 10 tests for
evidence/uncertainty.py.
"""
import dataclasses
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evidence import uncertainty as U


def _auroc_fn(y_true, y_pred):
    from benchmarks.metrics import cancer_prediction_metrics
    return cancer_prediction_metrics(y_true, y_pred)["auroc"]


def _record(seed, subject_ids, y_true, y_pred, cohort_source=None):
    return U.build_repeat_record(seed, subject_ids, y_true, y_pred, cohort_source)


# ─── Type-level "development only" guarantee ───────────────────────────────

def test_role_must_be_the_literal_development_string():
    rec = _record(1, ["a", "b", "c", "d"], [0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7])
    with pytest.raises(U.DevelopmentOnlyError):
        U.build_development_repeated_oof([rec], role="internal_test")
    with pytest.raises(U.DevelopmentOnlyError):
        U.build_development_repeated_oof([rec], role="frozen_test")
    with pytest.raises(U.DevelopmentOnlyError):
        U.build_development_repeated_oof([rec], role="")
    # The one accepted literal.
    oof = U.build_development_repeated_oof([rec], role="development")
    assert oof.role == "development"


def test_module_has_no_test_data_parameter_anywhere():
    """Structural sweep: no public function in evidence.uncertainty accepts
    a parameter whose name suggests test/held-out data."""
    import inspect

    forbidden_substrings = ("test_subject", "test_bag", "held_out", "frozen_test")
    for name, fn in vars(U).items():
        if not inspect.isfunction(fn) or fn.__module__ != U.__name__:
            continue
        for param in inspect.signature(fn).parameters:
            assert not any(f in param for f in forbidden_substrings), f"{name}({param}) looks test-data-shaped"


def test_repeated_development_oof_for_cancer_track_never_reads_test_bags_or_test_cell_dataset():
    """End-to-end proof using the sentinel pattern: the one function in
    this module that trains anything (repeated_development_oof_for_cancer_track)
    completes successfully even when the context's test_bags/test_cell_dataset
    are replaced with FrozenAccessSentinel — if it ever touched them, the
    sentinel would raise immediately."""
    from benchmarks.runner import build_synthetic_context
    from benchmarks.sentinel import FrozenAccessSentinel

    ctx = build_synthetic_context(seed=3, fast=True)
    ctx = dataclasses.replace(
        ctx, test_bags=FrozenAccessSentinel("test_bags"), test_cell_dataset=FrozenAccessSentinel("test_cell_dataset"),
    )
    dev_subjects = sorted(set(ctx.subjects_for("train")) | set(ctx.subjects_for("val")))
    outcomes = {
        str(b["subject_id"]): b["cancer_label"]
        for b in list(ctx.train_bags) + list(ctx.val_bags) if b.get("cancer_label_known")
    }
    num_ct = ctx.config.get("model", {}).get("num_cell_types", 4)
    min_cells = ctx.config.get("data", {}).get("min_cells_per_subject", 5)
    n_hvgs = ctx.preprocessing_artifact.n_hvgs
    try:
        oof = U.repeated_development_oof_for_cancer_track(
            ctx, "prevalence", dev_subjects, outcomes, num_ct, min_cells, n_hvgs,
            n_repeats=2, base_seed=42, n_folds=2,
        )
        assert len(oof.repeats) == 2
    finally:
        shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


# ─── RepeatRecord: subject-level only, never cell-level ────────────────────

def test_repeat_record_rejects_duplicate_subject_ids():
    """A duplicate subject_id within one repeat is exactly what a
    cell-level (many rows per subject) input looks like — this must never
    be silently accepted as if it were subject-level."""
    with pytest.raises(U.UncertaintyInputError):
        _record(1, ["a", "a", "b"], [0, 1, 0], [0.1, 0.9, 0.2])


def test_repeat_record_rejects_mismatched_lengths():
    with pytest.raises(U.UncertaintyInputError):
        _record(1, ["a", "b"], [0, 1, 1], [0.1, 0.9])


def test_repeat_record_rejects_empty():
    with pytest.raises(U.UncertaintyInputError):
        _record(1, [], [], [])


def test_development_repeated_oof_rejects_empty_repeats():
    with pytest.raises(U.UncertaintyInputError):
        U.build_development_repeated_oof([], role="development")


# ─── Per-repeat summary and subject-level bootstrap CI ─────────────────────

def _synthetic_oof(n_repeats=3, n_subjects=30, cohort=False):
    import numpy as np
    rng = np.random.RandomState(0)
    records = []
    for r in range(n_repeats):
        subject_ids = [f"s{r}_{i}" for i in range(n_subjects)]
        y_true = (rng.rand(n_subjects) > 0.5).astype(int).tolist()
        y_pred = [min(1.0, max(0.0, y * 0.6 + rng.rand() * 0.3)) for y in y_true]
        cohort_source = [f"cohort{i % 4}" for i in range(n_subjects)] if cohort else None
        records.append(_record(100 + r, subject_ids, y_true, y_pred, cohort_source))
    return U.build_development_repeated_oof(records, role="development")


def test_repeated_metric_summary_reports_mean_median_std_iqr():
    oof = _synthetic_oof()
    summary = U.repeated_metric_summary(oof, _auroc_fn)
    assert summary["n_repeats"] == 3
    assert summary["mean"] is not None
    assert summary["median"] is not None
    assert summary["std"] is not None
    assert "seed_level_bootstrap" in summary
    assert summary["seed_level_bootstrap"]["resampling_unit"] == "seed"


def test_subject_level_bootstrap_ci_resamples_subjects_not_folds_or_cells():
    oof = _synthetic_oof(n_repeats=1, n_subjects=40)
    ci = U.subject_level_bootstrap_ci(oof.repeats[0], _auroc_fn, n_boot=200)
    assert ci is not None
    assert ci["resampling_unit"] == "subject"
    assert ci["lo"] <= ci["point_estimate"] <= ci["hi"] or ci["lo"] is None


def test_subject_level_bootstrap_ci_undefined_with_fewer_than_two_subjects():
    rec = _record(1, ["only_one"], [1], [0.9])
    assert U.subject_level_bootstrap_ci(rec, _auroc_fn) is None


def test_ci_overlap_is_labeled_descriptive_only_never_an_equivalence_verdict():
    ci_a = {"lo": 0.4, "hi": 0.6}
    ci_b = {"lo": 0.5, "hi": 0.7}
    result = U.confidence_intervals_overlap(ci_a, ci_b)
    assert result["overlaps"] is True
    assert result["interpretation"] == "descriptive_only"
    assert "not a formal equivalence test" in result["disclaimer"]

    ci_c = {"lo": 0.9, "hi": 0.95}
    result_no_overlap = U.confidence_intervals_overlap(ci_a, ci_c)
    assert result_no_overlap["overlaps"] is False
    assert result_no_overlap["interpretation"] == "descriptive_only"


def test_ci_overlap_undefined_when_either_ci_missing():
    assert U.confidence_intervals_overlap(None, {"lo": 0.1, "hi": 0.2}) is None
    assert U.confidence_intervals_overlap({"lo": None, "hi": None}, {"lo": 0.1, "hi": 0.2}) is None


# ─── Paired candidate comparison ────────────────────────────────────────────

def test_paired_candidate_comparison_win_tie_loss_counts():
    oof_a = _synthetic_oof(n_repeats=3, n_subjects=25)
    oof_b = _synthetic_oof(n_repeats=3, n_subjects=25)
    result = U.paired_candidate_comparison(oof_a, oof_b, _auroc_fn)
    assert result["wins_a_over_b"] + result["ties"] + result["losses_a_over_b"] + result["n_undefined"] == 3
    assert result["common_seeds"] == [100, 101, 102]
    assert result["unmatched_seeds"] == []


def test_paired_candidate_comparison_reports_unmatched_seeds_never_drops_silently():
    rec_a1 = _record(1, ["a", "b", "c", "d"], [0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7])
    rec_a2 = _record(2, ["a", "b", "c", "d"], [0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7])
    rec_b1 = _record(1, ["a", "b", "c", "d"], [0, 1, 0, 1], [0.6, 0.4, 0.9, 0.1])
    oof_a = U.build_development_repeated_oof([rec_a1, rec_a2], role="development")
    oof_b = U.build_development_repeated_oof([rec_b1], role="development")
    result = U.paired_candidate_comparison(oof_a, oof_b, _auroc_fn)
    assert result["common_seeds"] == [1]
    assert result["unmatched_seeds"] == [2]


# ─── Independent subject/cohort counts and status gating ──────────────────

def test_independent_subject_count_deduplicates_across_repeats():
    rec1 = _record(1, ["a", "b"], [0, 1], [0.2, 0.8])
    rec2 = _record(2, ["a", "b"], [0, 1], [0.3, 0.7])  # same subjects, different seed/predictions
    oof = U.build_development_repeated_oof([rec1, rec2], role="development")
    assert oof.independent_subject_count() == 2


def test_independent_cohort_count_none_when_never_recorded():
    oof = _synthetic_oof(cohort=False)
    assert oof.independent_cohort_count() is None


def test_uncertainty_status_insufficient_when_cohorts_below_minimum(tmp_path):
    import yaml
    cfg = tmp_path / "evidence.yaml"
    cfg.write_text(yaml.safe_dump({"uncertainty_policy": {"min_independent_cohorts": 5}}))

    oof = _synthetic_oof(cohort=True)  # only 4 distinct cohort labels
    status = U.uncertainty_status(oof, config_path=str(cfg))
    assert status["status"] == "insufficient_evidence_for_reliable_source_level_uncertainty"
    assert status["independent_cohort_count"] == 4
    assert status["min_independent_cohorts_required"] == 5


def test_uncertainty_status_adequate_when_cohorts_meet_minimum(tmp_path):
    import yaml
    cfg = tmp_path / "evidence.yaml"
    cfg.write_text(yaml.safe_dump({"uncertainty_policy": {"min_independent_cohorts": 3}}))

    oof = _synthetic_oof(cohort=True)  # 4 distinct cohort labels
    status = U.uncertainty_status(oof, config_path=str(cfg))
    assert status["status"] == "adequate"


def test_uncertainty_status_cohort_source_not_recorded_status():
    oof = _synthetic_oof(cohort=False)
    status = U.uncertainty_status(oof)
    assert status["status"] == "cohort_source_not_recorded"


def test_cohort_source_from_bags_reads_the_already_computed_source_field_only():
    bags = [{"subject_id": "s1", "source": "gse1"}, {"subject_id": "s2", "source": None}]
    mapping = U.cohort_source_from_bags(bags)
    assert mapping == {"s1": "gse1", "s2": None}


def test_load_uncertainty_policy_falls_back_when_file_missing():
    policy = U.load_uncertainty_policy(config_path="/nonexistent/path/evidence.yaml")
    assert policy["min_independent_cohorts"] == U._FALLBACK_MIN_INDEPENDENT_COHORTS
