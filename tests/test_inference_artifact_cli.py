"""
Phase 4 — inference.py CLI extensions: standalone artifact inspect/validate
commands (Step 15/16), no checkpoint or --h5ad required.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.preprocessing import fit_preprocessing


def _artifact_path(tmp_path):
    subject_ids = [f"s{i // 10}" for i in range(60)]
    adata = ad.AnnData(
        X=np.random.default_rng(0).random((60, 10)).astype("float32"),
        obs=pd.DataFrame({"subject_id": subject_ids, "is_pseudo_bulk": [False] * 60}),
        var=pd.DataFrame(index=[f"G{i}" for i in range(10)]),
    )
    artifact = fit_preprocessing(adata, {f"s{i}" for i in range(6)}, n_hvgs=10)
    path = tmp_path / "artifact.json"
    artifact.save(path)
    return path


def _run_cli(args):
    src_dir = Path(__file__).parents[1] / "src"
    return subprocess.run(
        [sys.executable, "inference.py", *args],
        cwd=str(src_dir), capture_output=True, text=True,
    )


def test_inspect_artifact_cli_prints_json_report(tmp_path):
    path = _artifact_path(tmp_path)
    result = _run_cli(["--inspect-artifact", str(path)])
    assert result.returncode == 0
    assert '"artifact_fingerprint"' in result.stdout
    assert '"selected_gene_count": 10' in result.stdout


def test_validate_artifact_cli_ok_for_valid_artifact(tmp_path):
    path = _artifact_path(tmp_path)
    result = _run_cli(["--validate-artifact", str(path)])
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_validate_artifact_cli_fails_for_corrupt_artifact(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not valid json")
    result = _run_cli(["--validate-artifact", str(path)])
    assert result.returncode == 1
    assert "INVALID" in result.stdout
