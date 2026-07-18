"""benchmarks/runner.py — full fast synthetic end-to-end CLI, both tasks + OOD."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.runner import DomainRobustnessCLIConfigurationError, main
from benchmarks.source_held_out import UnsupportedSmokeDomainStrategyError


def _run(args):
    rc = main(args)
    assert rc == 0


def test_synthetic_fast_smoke_task_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        _run(["--synthetic", "--fast", "--task", "smoke", "--output", tmp, "--run-id", "smoke_run"])
        run_dir = Path(tmp) / "smoke_run"
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        assert manifest["synthetic"] is True
        report = (run_dir / "report.md").read_text()
        assert "SYNTHETIC RUN" in report
        assert (run_dir / "metrics" / "smoke_classification_folds.csv").exists()
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_synthetic_fast_cancer_task_end_to_end_writes_frozen_calibration():
    with tempfile.TemporaryDirectory() as tmp:
        _run(["--synthetic", "--fast", "--task", "cancer", "--output", tmp, "--run-id", "cancer_run"])
        run_dir = Path(tmp) / "cancer_run"
        calib = json.loads((run_dir / "calibration" / "frozen_policy.json").read_text())
        assert "test_result" in calib
        assert "policy" in calib["test_result"]
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_synthetic_smoke_task_with_leave_one_source_out():
    with tempfile.TemporaryDirectory() as tmp:
        _run(["--synthetic", "--fast", "--task", "smoke", "--leave-one-source-out",
              "--output", tmp, "--run-id", "ood_run"])
        run_dir = Path(tmp) / "ood_run"
        ood = json.loads((run_dir / "metrics" / "leave_one_source_out.json").read_text())
        assert "sourceA" in ood or "sourceB" in ood
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_run_id_collision_raises_not_overwrites():
    with tempfile.TemporaryDirectory() as tmp:
        _run(["--synthetic", "--fast", "--task", "smoke", "--output", tmp, "--run-id", "dup"])
        with pytest.raises(FileExistsError):
            main(["--synthetic", "--fast", "--task", "smoke", "--output", tmp, "--run-id", "dup"])
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)


def test_coral_weight_without_coral_strategy_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(DomainRobustnessCLIConfigurationError):
            main(["--synthetic", "--fast", "--task", "cancer", "--output", tmp,
                  "--coral-weight", "0.1"])


def test_mmd_weight_with_non_mmd_strategy_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(DomainRobustnessCLIConfigurationError):
            main(["--synthetic", "--fast", "--task", "cancer", "--output", tmp,
                  "--domain-strategy", "coral", "--mmd-weight", "0.1"])


def test_gradient_reversal_lambda_without_adversarial_strategy_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(DomainRobustnessCLIConfigurationError):
            main(["--synthetic", "--fast", "--task", "cancer", "--output", tmp,
                  "--gradient-reversal-lambda", "1.0"])


def test_source_balanced_without_source_balanced_strategy_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(DomainRobustnessCLIConfigurationError):
            main(["--synthetic", "--fast", "--task", "cancer", "--output", tmp,
                  "--domain-strategy", "coral", "--coral-weight", "0.1", "--source-balanced"])


def test_robustness_report_without_active_workflow_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(DomainRobustnessCLIConfigurationError):
            main(["--synthetic", "--fast", "--task", "cancer", "--output", tmp,
                  "--robustness-report", str(Path(tmp) / "r.json")])


def test_smoke_task_rejects_non_erm_domain_strategy():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(UnsupportedSmokeDomainStrategyError):
            main(["--synthetic", "--fast", "--task", "smoke", "--output", tmp,
                  "--domain-strategy", "coral", "--coral-weight", "0.1"])
    shutil.rmtree("checkpoints/benchmarks_synthetic", ignore_errors=True)
