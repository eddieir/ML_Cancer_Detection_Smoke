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
    run_development,
    run_external_test,
    run_gse123352_repeated_development,
    run_internal_test,
)
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
    assert result["n_repeats"] == 3
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
