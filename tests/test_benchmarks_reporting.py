"""benchmarks/reporting.py — statistical comparison, immutable artifacts, schema."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.reporting import compare_models, new_run_dir, summarize_comparison, write_json


def test_new_run_dir_creates_immutable_layout():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = new_run_dir(tmp, run_id="fixed_id")
        assert (run_dir / "predictions").is_dir()
        assert (run_dir / "metrics").is_dir()
        assert (run_dir / "calibration").is_dir()
        with pytest.raises(FileExistsError):
            new_run_dir(tmp, run_id="fixed_id")


def test_compare_models_only_uses_paired_folds():
    results = {
        "a": {"folds": [{"seed": 42, "fold": 0, "m": 0.9}, {"seed": 42, "fold": 1, "m": 0.8}]},
        "b": {"folds": [{"seed": 42, "fold": 0, "m": 0.5}, {"seed": 42, "fold": 1, "m": None}]},
    }
    comparison = compare_models(results, "m", "a", "b")
    assert comparison["n_pairs"] == 1  # fold 1 excluded: b's value is None


def test_summarize_comparison_requires_more_than_a_higher_mean():
    # Only 2 paired folds — must not claim "meaningfully better" from that alone.
    results = {
        "a": {"folds": [{"seed": 1, "fold": 0, "m": 0.9}, {"seed": 1, "fold": 1, "m": 0.8}]},
        "b": {"folds": [{"seed": 1, "fold": 0, "m": 0.5}, {"seed": 1, "fold": 1, "m": 0.5}]},
    }
    comparison = compare_models(results, "m", "a", "b")
    summary = summarize_comparison(comparison)
    assert not summary["meaningfully_better"]
    assert "fewer than 3" in summary["reason"]


def test_write_json_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x" / "y.json"
        write_json(path, {"a": 1})
        assert json.loads(path.read_text()) == {"a": 1}
