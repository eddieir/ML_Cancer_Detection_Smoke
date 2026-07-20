"""
tests/test_evidence_audit.py — Phase 7 read-only real-data audit CLI.

These tests run against the actual repository configs (no real datasets
downloaded in this environment) and confirm the audit reports an honest,
structured "blocked" result rather than fabricating a subject/label count
or pretending a controlled-access cohort (NLST) was evaluated.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evidence.audit import main, run_audit
from evidence.evidence_contract import is_not_evaluable

REPO_ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.delenv("NLST_DATA_ROOT", raising=False)


def test_audit_runs_without_real_data_and_reports_blocked():
    report = run_audit("configs/default.yaml", "configs/cohorts.yaml")
    assert report["any_local_real_data_present"] is False
    assert report["overall_status"] == "blocked_no_real_data"
    assert is_not_evaluable(report["subject_counts"])
    assert is_not_evaluable(report["row_cell_counts"])
    assert is_not_evaluable(report["cancer_outcome_counts"])


def test_audit_never_reports_controlled_access_as_authorized_without_env_var():
    report = run_audit("configs/default.yaml", "configs/cohorts.yaml")
    assert report["controlled_access_cohorts_authorized_in_this_environment"] is False
    nlst_status = next(s for s in report["access_status_by_cohort"] if s["cohort_id"] == "nlst")
    assert nlst_status["controlled_access_authorized_locally"] is False
    assert "blocker" in nlst_status
    assert is_not_evaluable(nlst_status["blocker"])


def test_audit_reports_every_task_impossible_when_no_local_data_and_no_verified_support():
    report = run_audit("configs/default.yaml", "configs/cohorts.yaml")
    for row in report["task_eligibility"]:
        assert row["local_data_present"] is False
        # not one row claims eligibility for evaluation without real local data
        assert row["status"] != "ELIGIBLE_INTERNAL_HELD_OUT"
        assert row["status"] != "ELIGIBLE_EXTERNAL_VALIDATION"


def test_audit_config_cohort_cross_check_is_clean():
    report = run_audit("configs/default.yaml", "configs/cohorts.yaml")
    assert report["config_cohort_contradictions"] == []


def test_audit_output_contains_no_absolute_local_paths():
    report = run_audit("configs/default.yaml", "configs/cohorts.yaml")
    blob = json.dumps(report)
    assert str(Path.home()) not in blob
    assert "/Users/" not in blob


def test_audit_never_produces_a_fake_metric_dict_for_absent_cohorts():
    report = run_audit("configs/default.yaml", "configs/cohorts.yaml")
    for s in report["access_status_by_cohort"]:
        if not s["local_files_present"] and s["access_level"] == "public":
            assert "blocker" in s
            assert is_not_evaluable(s["blocker"])


def test_audit_cli_writes_output_and_exits_zero(tmp_path):
    out_path = tmp_path / "data_audit.json"
    rc = main(["--config", "configs/default.yaml", "--cohort-config", "configs/cohorts.yaml",
               "--output", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    with open(out_path) as f:
        report = json.load(f)
    assert report["overall_status"] == "blocked_no_real_data"


def test_audit_cli_validate_only_does_not_write_output(tmp_path, capsys):
    out_path = tmp_path / "should_not_exist.json"
    rc = main(["--config", "configs/default.yaml", "--cohort-config", "configs/cohorts.yaml",
               "--output", str(out_path), "--validate-only"])
    assert rc == 0
    assert not out_path.exists()
    captured = capsys.readouterr()
    assert "blocked_no_real_data" in captured.out


def test_audit_content_fingerprint_is_deterministic_given_same_inputs():
    r1 = run_audit("configs/default.yaml", "configs/cohorts.yaml")
    r2 = run_audit("configs/default.yaml", "configs/cohorts.yaml")
    fp1, fp2 = r1.pop("content_fingerprint"), r2.pop("content_fingerprint")
    assert r1 == r2
    assert fp1 == fp2
