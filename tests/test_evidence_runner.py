"""
tests/test_evidence_runner.py — Phase 7 Step 16 tests: the evidence.runner
umbrella CLI.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evidence.artifact_bundle import write_evidence_run
from evidence.errors import EvidenceRunnerUsageError
from evidence.evidence_contract import is_not_evaluable
from evidence.runner import (
    build_parser,
    cmd_audit,
    cmd_clinical_readiness,
    cmd_inspect,
    cmd_track_stub,
    cmd_validate,
    main,
)


@pytest.fixture
def run_dir(tmp_path):
    root = tmp_path / "artifacts" / "evidence"
    d = write_evidence_run(root, "fixture_run", {"configuration.json": {"x": 1}}, run_status="complete")
    return d


def _args(**kwargs):
    parser = build_parser()
    argv = []
    for k, v in kwargs.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv.extend([flag, str(v)])
    return argv


# ─── audit — thin wrapper ───────────────────────────────────────────────────

def test_audit_subcommand_runs_and_matches_module(capsys):
    code = main(["audit"])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["generated_by"] == "evidence.audit"


def test_audit_accepts_diagnostic_mode_as_noop(capsys):
    code = main(["audit", "--diagnostic-mode"])
    assert code == 0


# ─── inspect — read-only ───────────────────────────────────────────────────

def test_inspect_reports_run_summary(run_dir, capsys):
    code = main(["inspect", "--run-dir", str(run_dir)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_status"] == "complete"
    assert payload["files"] == ["configuration.json"]


def test_inspect_never_writes(run_dir, capsys):
    before = {p: p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file()}
    main(["inspect", "--run-dir", str(run_dir)])
    after = {p: p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file()}
    assert before == after
    assert set(before) == set(after)


def test_inspect_missing_run_reports_not_found(tmp_path, capsys):
    code = main(["inspect", "--run-dir", str(tmp_path / "nope")])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_found"


def test_inspect_rejects_diagnostic_mode(run_dir):
    code = main(["inspect", "--run-dir", str(run_dir), "--diagnostic-mode"])
    assert code == 2


# ─── validate — read-only checksum verification ────────────────────────────

def test_validate_passes_on_intact_run(run_dir, capsys):
    code = main(["validate", "--run-dir", str(run_dir)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass"


def test_validate_fails_on_tampered_run(run_dir, capsys):
    (run_dir / "configuration.json").write_text('{"x": 999}')
    code = main(["validate", "--run-dir", str(run_dir)])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fail"


def test_validate_never_writes(run_dir):
    before = {p: p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file()}
    main(["validate", "--run-dir", str(run_dir)])
    after = {p: p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file()}
    assert before == after


def test_validate_rejects_diagnostic_mode(run_dir):
    code = main(["validate", "--run-dir", str(run_dir), "--diagnostic-mode"])
    assert code == 2


# ─── clinical-readiness — read-only, never trains ──────────────────────────

def test_clinical_readiness_never_claims_ready(run_dir, capsys):
    code = main(["clinical-readiness", "--run-dir", str(run_dir)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_status"] == "clinically_not_ready"
    assert all(d["status"] == "not_started" for d in payload["dimensions"])


def test_clinical_readiness_never_writes(run_dir):
    before = {p: p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file()}
    main(["clinical-readiness", "--run-dir", str(run_dir)])
    after = {p: p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file()}
    assert before == after


def test_clinical_readiness_fails_on_invalid_run(tmp_path, capsys):
    code = main(["clinical-readiness", "--run-dir", str(tmp_path / "nope")])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fail"


def test_clinical_readiness_rejects_diagnostic_mode(run_dir):
    code = main(["clinical-readiness", "--run-dir", str(run_dir), "--diagnostic-mode"])
    assert code == 2


def test_clinical_readiness_disclaimers_present(run_dir, capsys):
    main(["clinical-readiness", "--run-dir", str(run_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert any("not a medical device" in d.lower() for d in payload["disclaimers"])


# ─── development / internal-test / external-test — honest stubs ───────────

@pytest.mark.parametrize("subcommand", ["development", "internal-test", "external-test"])
def test_track_stub_returns_not_evaluable(subcommand, capsys):
    code = main([subcommand])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert is_not_evaluable(payload)
    assert payload["reason_code"] == "TRACK_RUNNER_NOT_YET_INTEGRATED"
    assert payload["subcommand"] == subcommand


@pytest.mark.parametrize("subcommand", ["development", "internal-test", "external-test"])
def test_track_stub_rejects_diagnostic_mode(subcommand):
    code = main([subcommand, "--diagnostic-mode"])
    assert code == 2


@pytest.mark.parametrize("subcommand", ["development", "internal-test", "external-test"])
def test_track_stub_rejects_run_dir_flag(subcommand, tmp_path):
    code = main([subcommand, "--run-dir", str(tmp_path)])
    assert code == 2


def test_track_stub_never_fabricates_success_via_direct_call():
    parser = build_parser()
    args = parser.parse_args(["development"])
    result_code = cmd_track_stub(args, "development")
    assert result_code == 0


# ─── CLI-level plumbing ─────────────────────────────────────────────────────

def test_unknown_subcommand_exits_nonzero():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["not-a-real-subcommand"])


def test_no_subcommand_exits_nonzero():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_diagnostic_mode_error_is_typed_error_class(run_dir):
    parser = build_parser()
    args = parser.parse_args(["inspect", "--run-dir", str(run_dir), "--diagnostic-mode"])
    with pytest.raises(EvidenceRunnerUsageError):
        cmd_inspect(args)
