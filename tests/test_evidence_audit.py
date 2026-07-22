"""
tests/test_evidence_audit.py — Phase 7 read-only real-data audit CLI.

These tests run against the actual repository configs. Some environments
have real GSE136831/GSE123352 files downloaded under data/raw/ (gitignored,
never committed) for the real-data evidence runs in evidence/tracks.py and
scripts/run_gse136831_weak_label_evidence.py; others have no local files at
all. Either way, the audit must report an honest, structured result — never
a fabricated subject/label count, and never a controlled-access cohort
(NLST) treated as evaluated without an authorized local copy.
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


def test_audit_reports_blocked_or_partial_consistent_with_local_files():
    report = run_audit("configs/default.yaml", "configs/cohorts.yaml")
    # overall_status must agree with any_local_real_data_present regardless
    # of which environment this runs in (no local files at all vs. one or
    # more real cohort files present under data/raw/).
    if report["any_local_real_data_present"]:
        assert report["overall_status"] == "partial_local_data_present"
    else:
        assert report["overall_status"] == "blocked_no_real_data"
    # The audit CLI is read-only and never processes a dataset itself — it
    # only detects file presence. With no local files it reports a
    # structured not_evaluable(); with some local files present it reports
    # an explicit "not implemented in the audit CLI itself" string — either
    # way, never a fabricated count.
    for key in ("subject_counts", "row_cell_counts", "cancer_outcome_counts"):
        value = report[key]
        if report["any_local_real_data_present"]:
            assert isinstance(value, str) and "not_implemented_pending_local_data" in value
        else:
            assert is_not_evaluable(value)


def test_audit_never_reports_controlled_access_as_authorized_without_env_var():
    report = run_audit("configs/default.yaml", "configs/cohorts.yaml")
    assert report["controlled_access_cohorts_authorized_in_this_environment"] is False
    nlst_status = next(s for s in report["access_status_by_cohort"] if s["cohort_id"] == "nlst")
    assert nlst_status["controlled_access_authorized_locally"] is False
    assert "blocker" in nlst_status
    assert is_not_evaluable(nlst_status["blocker"])


def test_audit_reports_every_task_impossible_absent_verified_registry_support():
    # local_data_present may legitimately be True or False per row depending
    # on which files this environment happens to have under data/raw/, but
    # no row may claim internal/external eligibility from the audit alone —
    # that requires a registry entry with task_support='yes' AND the
    # matching role in role_eligibility, which no cohort in
    # configs/cohorts.yaml carries for these three tasks.
    report = run_audit("configs/default.yaml", "configs/cohorts.yaml")
    for row in report["task_eligibility"]:
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
    assert report["overall_status"] in ("blocked_no_real_data", "partial_local_data_present")


def test_audit_cli_validate_only_does_not_write_output(tmp_path, capsys):
    out_path = tmp_path / "should_not_exist.json"
    rc = main(["--config", "configs/default.yaml", "--cohort-config", "configs/cohorts.yaml",
               "--output", str(out_path), "--validate-only"])
    assert rc == 0
    assert not out_path.exists()
    captured = capsys.readouterr()
    assert ("blocked_no_real_data" in captured.out) or ("partial_local_data_present" in captured.out)


def test_audit_content_fingerprint_is_deterministic_given_same_inputs():
    r1 = run_audit("configs/default.yaml", "configs/cohorts.yaml")
    r2 = run_audit("configs/default.yaml", "configs/cohorts.yaml")
    fp1, fp2 = r1.pop("content_fingerprint"), r2.pop("content_fingerprint")
    assert r1 == r2
    assert fp1 == fp2
