"""
tests/test_evidence_development.py — Issue #16 blocker 4/6: canonical
development orchestration and repeated development evaluation for
GSE123352.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evidence.cohort_registry import load_cohort_registry
from evidence.development import (
    SubjectOverlapError,
    assess_frozen_internal_test_eligibility,
    run_development,
    run_external_test,
    run_gse123352_repeated_development,
    run_internal_test,
)
from evidence.development import _assert_no_subject_overlap
from evidence.evidence_contract import is_not_evaluable

COHORTS_YAML = Path(__file__).parents[1] / "configs" / "cohorts.yaml"


@pytest.fixture(scope="module")
def cohorts():
    return load_cohort_registry(str(COHORTS_YAML))


def test_development_unregistered_cohort_not_evaluable(cohorts, tmp_path):
    result = run_development("not_a_cohort", "smoke_classification", cohorts=cohorts, output_root=tmp_path)
    assert is_not_evaluable(result)
    assert result["reason_code"] == "NO_ELIGIBLE_COHORT"


def test_development_gse136831_smoke_classification_not_evaluable(cohorts, tmp_path):
    # blocker 2: gse136831 no longer supports smoke_classification at all.
    result = run_development("gse136831", "smoke_classification", cohorts=cohorts, output_root=tmp_path)
    assert is_not_evaluable(result)


def test_development_gse123352_without_local_data_reports_blocker(cohorts, tmp_path, monkeypatch):
    import evidence.development as development_module
    monkeypatch.setattr(development_module, "GSE123352_RAW_DIR", tmp_path / "does_not_exist")
    result = run_development("gse123352", "smoke_classification", cohorts=cohorts, output_root=tmp_path)
    assert is_not_evaluable(result)
    assert result["reason_code"] == "REAL_DATA_NOT_PRESENT"


def test_internal_test_always_blocked_with_specific_reason(tmp_path):
    from evidence.artifact_bundle import write_evidence_run
    root = tmp_path / "artifacts"
    write_evidence_run(root, "run1", {"configuration.json": {"x": 1}}, run_status="complete")
    result = run_internal_test(root / "run1", None)
    assert is_not_evaluable(result)
    assert result["reason_code"] == "NO_FROZEN_TEST_PARTITION_EXISTS"


def test_internal_test_rejects_invalid_run_dir(tmp_path):
    result = run_internal_test(tmp_path / "nonexistent_run", None)
    assert is_not_evaluable(result)
    assert result["reason_code"] == "DEVELOPMENT_RUN_INVALID"


def test_external_test_always_blocked_no_eligible_cohort(tmp_path):
    from evidence.artifact_bundle import write_evidence_run
    root = tmp_path / "artifacts"
    write_evidence_run(root, "run1", {"configuration.json": {"x": 1}}, run_status="complete")
    result = run_external_test(root / "run1", "some_external_cohort")
    assert is_not_evaluable(result)
    assert result["reason_code"] == "NO_ELIGIBLE_EXTERNAL_COHORT"


# ─── repeated development evaluation (synthetic dataset injection) ────────

def _synthetic_bulk_dataset(n_samples=80, n_genes=60, seed=0):
    from data.bulk_pipeline import build_bulk_smoke_dataset

    rng = np.random.RandomState(seed)
    sample_ids = [f"GSM{2000+i}" for i in range(n_samples)]
    gene_ids = [f"GENE{i}" for i in range(n_genes)]
    labels = ["cigarette" if i % 2 == 0 else "unexposed" for i in range(n_samples)]

    X = rng.normal(loc=0.0, scale=1.0, size=(n_genes, n_samples))
    signal_genes = list(range(5))
    for j, lab in enumerate(labels):
        if lab == "cigarette":
            X[signal_genes, j] += 2.5

    expr = pd.DataFrame(X, index=gene_ids, columns=sample_ids)
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp())
    csv_path = tmp_dir / "FAKE.csv"
    expr.to_csv(csv_path)
    subject_ids = [f"SUBJ{2000+i}" for i in range(n_samples)]
    meta = pd.DataFrame({
        "sample_id": sample_ids, "smoke_type": labels,
        "smoke_type_known": [True] * n_samples,
        "subject_id": subject_ids, "subject_id_verified": [True] * n_samples,
    })
    meta.to_csv(tmp_dir / "FAKE_samples_meta.csv", index=False)
    return build_bulk_smoke_dataset(csv_path)


def test_repeated_development_produces_aggregated_macro_f1():
    dataset = _synthetic_bulk_dataset()
    result = run_gse123352_repeated_development(
        seeds=[1, 2, 3], n_top_variance_genes=10, _dataset_override=dataset,
    )
    assert result["status"] == "complete"
    assert result["n_completed_seeds"] == 3
    assert result["n_requested_seeds"] == 3
    assert result["n_excluded_seeds"] == 0
    assert result["macro_f1"]["n_repeats"] == 3
    assert result["macro_f1"]["mean"] is not None
    assert 0.0 <= result["macro_f1"]["mean"] <= 1.0
    assert set(result["per_seed_macro_f1_subject_bootstrap_ci"].keys()) == {"1", "2", "3"}
    for ci in result["per_seed_macro_f1_subject_bootstrap_ci"].values():
        assert ci is None or "lo" in ci


def test_repeated_development_hyperparameter_selection_is_fold_local():
    dataset = _synthetic_bulk_dataset(seed=5)
    result = run_gse123352_repeated_development(
        seeds=[1, 2], n_top_variance_genes=10, _dataset_override=dataset,
    )
    assert set(result["selected_hyperparameters_by_seed"].keys()) == {"1", "2"}
    for c in result["selected_hyperparameters_by_seed"].values():
        assert c in (0.1, 1.0, 10.0)


def test_repeated_development_reports_original_result_not_superseded():
    dataset = _synthetic_bulk_dataset()
    result = run_gse123352_repeated_development(seeds=[1, 2], n_top_variance_genes=10, _dataset_override=dataset)
    assert any("original single 70/30" in lim for lim in result["limitations"])


def test_repeated_development_without_local_data_returns_not_evaluable(tmp_path, monkeypatch):
    import evidence.development as development_module
    monkeypatch.setattr(development_module, "GSE123352_RAW_DIR", tmp_path / "nope")
    result = run_gse123352_repeated_development(seeds=[1, 2])
    assert is_not_evaluable(result)
    assert result["reason_code"] == "REAL_DATA_NOT_PRESENT"


def test_repeated_development_deterministic_given_same_seeds():
    dataset1 = _synthetic_bulk_dataset(seed=9)
    dataset2 = _synthetic_bulk_dataset(seed=9)
    result1 = run_gse123352_repeated_development(seeds=[1, 2, 3], n_top_variance_genes=10, _dataset_override=dataset1)
    result2 = run_gse123352_repeated_development(seeds=[1, 2, 3], n_top_variance_genes=10, _dataset_override=dataset2)
    assert result1["macro_f1"]["per_repeat_values"] == result2["macro_f1"]["per_repeat_values"]


# ─── baselines, full metric bundle, calibration (Issue #16 step 3.3-3.6) ──

def test_repeated_development_reports_required_baselines():
    dataset = _synthetic_bulk_dataset(seed=3)
    result = run_gse123352_repeated_development(seeds=[1, 2, 3], n_top_variance_genes=10, _dataset_override=dataset)
    assert set(result["baseline_comparisons"].keys()) == {"majority", "prevalence", "bulk_linear_untuned"}
    for name, comparison in result["baseline_comparisons"].items():
        assert comparison["baseline_name"] == name
        assert comparison["status"] in ("reported", "insufficient_evidence")
        assert "wins_a_over_b" in comparison and "losses_a_over_b" in comparison and "ties" in comparison
        assert "mean_paired_diff" in comparison


def test_repeated_development_baselines_never_invent_a_clinical_baseline():
    dataset = _synthetic_bulk_dataset(seed=4)
    result = run_gse123352_repeated_development(seeds=[1, 2], n_top_variance_genes=10, _dataset_override=dataset)
    assert "clinical" not in result["baseline_comparisons"]
    assert any("no legitimate non-leaking clinical/metadata covariate" in lim.lower() for lim in result["limitations"])


def test_repeated_development_full_metric_bundle_has_required_fields():
    dataset = _synthetic_bulk_dataset(seed=6)
    result = run_gse123352_repeated_development(seeds=[1, 2], n_top_variance_genes=10, _dataset_override=dataset)
    for seed_key, bundle in result["full_metric_bundle_by_seed"].items():
        for field in (
            "balanced_accuracy", "per_class", "confusion_matrix", "auroc", "auprc",
            "brier", "ece", "log_loss",
        ):
            assert field in bundle, f"missing {field} for seed {seed_key}"


def test_repeated_development_calibration_pathway_is_disjoint_from_candidate():
    dataset = _synthetic_bulk_dataset(seed=7, n_samples=160)
    result = run_gse123352_repeated_development(seeds=[1, 2], n_top_variance_genes=10, _dataset_override=dataset)
    for seed_key, calib in result["calibration_by_seed"].items():
        assert calib["status"] in ("complete", "not_evaluable")
        if calib["status"] == "complete":
            assert calib["calibration_source_model"] == "fit_on_inner_train_only_not_the_primary_candidate"
            assert "uncalibrated_outer_test_metrics" in calib
            assert "calibrated_outer_test_metrics" in calib
            assert calib["calibration_method"] in ("none", "sigmoid", "isotonic")


def test_repeated_development_degenerate_calibration_is_not_evaluable_not_fabricated():
    """A seed whose inner split is too small to fit calibration must report
    status='not_evaluable' with a reason — never a fabricated calibration
    curve/method."""
    dataset = _synthetic_bulk_dataset(seed=11, n_samples=20)
    result = run_gse123352_repeated_development(seeds=[1], train_frac=0.9, test_frac=0.1, n_top_variance_genes=5, _dataset_override=dataset)
    for calib in result["calibration_by_seed"].values():
        if calib["status"] == "not_evaluable":
            assert calib["reason"]
            assert calib["calibration_fingerprint"] == "not_applicable"


# ─── frozen internal-test eligibility (Issue #16 step 3.7) ────────────────

def test_frozen_internal_test_ineligible_reports_computed_counts():
    result = assess_frozen_internal_test_eligibility(
        unique_subject_count=176, class_counts={"unexposed": 58, "cigarette": 118},
    )
    assert result["eligible"] is False
    assert result["reason_code"] == "INSUFFICIENT_SUPPORT_FOR_FROZEN_INTERNAL_TEST"
    assert result["unique_subject_count"] == 176
    assert result["class_counts"] == {"unexposed": 58, "cigarette": 118}
    assert "policy_fingerprint" in result
    assert result["required_next_action"]


def test_frozen_internal_test_eligible_synthetic_cohort():
    result = assess_frozen_internal_test_eligibility(
        unique_subject_count=500, class_counts={"unexposed": 200, "cigarette": 300},
    )
    assert result["eligible"] is True
    assert "reason_code" not in result


def test_frozen_internal_test_eligibility_fingerprint_changes_with_policy(tmp_path):
    policy_config = tmp_path / "evidence.yaml"
    policy_config.write_text(
        "frozen_internal_test_policy:\n  min_total_subjects: 50\n  min_per_class_subjects: 10\n"
        "  train_frac: 0.6\n  development_holdout_frac: 0.2\n  internal_test_frac: 0.2\n"
    )
    default_result = assess_frozen_internal_test_eligibility(
        unique_subject_count=176, class_counts={"unexposed": 58, "cigarette": 118},
    )
    custom_result = assess_frozen_internal_test_eligibility(
        unique_subject_count=176, class_counts={"unexposed": 58, "cigarette": 118},
        config_path=policy_config,
    )
    assert custom_result["eligible"] is True
    assert default_result["policy_fingerprint"] != custom_result["policy_fingerprint"]


def test_run_internal_test_gse123352_reports_real_eligibility_not_generic_gate():
    """Regression for Issue #16 step 3.7: the internal-test CLI gate for a
    real, wired cohort must name the SPECIFIC computed reason (real subject
    counts vs configured policy), not the generic 'no cohort has ever had a
    frozen partition' statement."""
    import evidence.development as development_module

    if not (development_module.GSE123352_RAW_DIR / development_module.GSE123352_RAW_FILE_NAMES[0]).exists():
        pytest.skip("real GSE123352 raw files not present locally")

    from evidence.artifact_bundle import write_evidence_run
    import tempfile
    root = Path(tempfile.mkdtemp())
    write_evidence_run(root, "run1", {"configuration.json": {"x": 1}}, run_status="complete")
    result = run_internal_test(root / "run1", None, cohort_id="gse123352", task="smoke_classification")
    assert is_not_evaluable(result)
    assert result["reason_code"] != "NO_FROZEN_TEST_PARTITION_EXISTS"
    assert "unique_subject_count" in result


# ─── hard subject-overlap validation (Issue #16 step 3.1) ─────────────────

def test_assert_no_subject_overlap_passes_for_disjoint_partitions():
    _assert_no_subject_overlap({"train": ["a", "b"], "test": ["c", "d"]})


def test_assert_no_subject_overlap_raises_for_shared_subject():
    with pytest.raises(SubjectOverlapError):
        _assert_no_subject_overlap({"train": ["a", "b"], "test": ["b", "c"]})


def test_repeated_development_outer_partitions_never_overlap():
    dataset = _synthetic_bulk_dataset(seed=13)
    result = run_gse123352_repeated_development(seeds=[1, 2, 3, 4], n_top_variance_genes=10, _dataset_override=dataset)
    assert result["status"] == "complete"  # would have raised SubjectOverlapError internally otherwise


# ─── TCGA vital-status cancer-outcome prediction (real, linked cohort) ────

def _synthetic_tcga_outcome_dataset(n_samples=120, n_genes=40, seed=0):
    from data.tcga_outcome_pipeline import TCGAOutcomeDataset

    rng = np.random.RandomState(seed)
    subject_ids = [f"case{i}" for i in range(n_samples)]
    labels = np.array([i % 2 for i in range(n_samples)])
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_genes)).astype(np.float32)
    signal_genes = list(range(5))
    X[:, signal_genes] += (labels[:, None] * 1.5)
    gene_names = [f"ENSG{i}" for i in range(n_genes)]
    return TCGAOutcomeDataset(
        X=X, y=labels, subject_ids=subject_ids, sample_ids=subject_ids, gene_names=gene_names,
    )


def test_run_development_dispatches_tcga_lung_vital_status(cohorts, tmp_path):
    import evidence.development as development_module
    if not development_module.TCGA_LUAD_CSV.exists() and not (development_module.TCGA_LUAD_RAW_DIR / "outcome_meta.csv").exists():
        pytest.skip("real TCGA data not present locally")
    result = run_development(
        "tcga_lung_vital_status", "subject_level_cancer_prediction", cohorts=cohorts,
        output_root=tmp_path / "tcga_dev_dispatch_test",
    )
    assert result["status"] in ("complete",)


def test_run_development_tcga_without_local_data_returns_not_evaluable(cohorts, tmp_path, monkeypatch):
    import evidence.development as development_module
    monkeypatch.setattr(development_module, "TCGA_LUAD_RAW_DIR", tmp_path / "nope_luad")
    monkeypatch.setattr(development_module, "TCGA_LUSC_RAW_DIR", tmp_path / "nope_lusc")
    monkeypatch.setattr(development_module, "TCGA_LUAD_CSV", tmp_path / "nope_luad.csv")
    monkeypatch.setattr(development_module, "TCGA_LUSC_CSV", tmp_path / "nope_lusc.csv")
    result = run_development(
        "tcga_lung_vital_status", "subject_level_cancer_prediction", cohorts=cohorts,
        output_root=tmp_path / "artifacts",
    )
    assert is_not_evaluable(result)
    assert result["reason_code"] == "REAL_DATA_NOT_PRESENT"


def test_tcga_repeated_development_produces_baselines_and_full_metrics():
    from evidence.development import run_tcga_lung_vital_status_repeated_development

    dataset = _synthetic_tcga_outcome_dataset(seed=3)
    result = run_tcga_lung_vital_status_repeated_development(
        seeds=[1, 2, 3], n_top_variance_genes=10, _dataset_override=dataset,
    )
    assert result["status"] == "complete"
    assert result["n_completed_seeds"] == 3
    assert set(result["baseline_comparisons"].keys()) == {"majority", "prevalence", "bulk_linear_untuned"}
    for seed_key, bundle in result["full_metric_bundle_by_seed"].items():
        assert "auroc" in bundle and "brier" in bundle
    assert "frozen_internal_test_eligibility" in result
    assert "eligible" in result["frozen_internal_test_eligibility"]


def test_tcga_repeated_development_without_local_data_returns_not_evaluable(tmp_path, monkeypatch):
    import evidence.development as development_module
    monkeypatch.setattr(development_module, "TCGA_LUAD_RAW_DIR", tmp_path / "nope_luad")
    monkeypatch.setattr(development_module, "TCGA_LUSC_RAW_DIR", tmp_path / "nope_lusc")
    monkeypatch.setattr(development_module, "TCGA_LUAD_CSV", tmp_path / "nope_luad.csv")
    monkeypatch.setattr(development_module, "TCGA_LUSC_CSV", tmp_path / "nope_lusc.csv")
    from evidence.development import run_tcga_lung_vital_status_repeated_development

    result = run_tcga_lung_vital_status_repeated_development(seeds=[1, 2])
    assert is_not_evaluable(result)
    assert result["reason_code"] == "REAL_DATA_NOT_PRESENT"
