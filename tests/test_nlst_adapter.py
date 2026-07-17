"""
tests/test_nlst_adapter.py — controlled-access NLST adapter. Uses small
synthetic, schema-compatible fixture rows only; never real participant
data, never attempts real network/DUA-gated access.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.nlst_adapter import (
    NLSTAccessError,
    check_nlst_availability,
    require_nlst_available,
)


def test_unavailable_when_env_var_not_set(monkeypatch):
    monkeypatch.delenv("NLST_DATA_ROOT_TEST", raising=False)
    status = check_nlst_availability("NLST_DATA_ROOT_TEST")
    assert status.available is False
    assert status.local_root is None


def test_unavailable_when_root_set_but_files_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("NLST_DATA_ROOT_TEST", str(tmp_path))
    status = check_nlst_availability("NLST_DATA_ROOT_TEST")
    assert status.available is False
    assert status.screen_csv_present is False


def test_unavailable_when_columns_missing(monkeypatch, tmp_path):
    pd.DataFrame({"pid": [1, 2]}).to_csv(tmp_path / "screen.csv", index=False)  # missing CIGSMOK/CIGAR
    pd.DataFrame({"pid": [1, 2], "candx": [0, 1]}).to_csv(tmp_path / "prsn.csv", index=False)
    monkeypatch.setenv("NLST_DATA_ROOT_TEST", str(tmp_path))
    status = check_nlst_availability("NLST_DATA_ROOT_TEST")
    assert status.available is False
    assert status.screen_columns_valid is False


def test_available_with_synthetic_schema_compatible_fixture(monkeypatch, tmp_path):
    """Synthetic fixture rows only — not real NLST participant data."""
    pd.DataFrame({
        "pid": [1001, 1002, 1003], "CIGSMOK": [1, 2, 0], "CIGAR": [0, 0, 1],
    }).to_csv(tmp_path / "screen.csv", index=False)
    pd.DataFrame({
        "pid": [1001, 1002, 1003], "candx": [0, 1, 0],
    }).to_csv(tmp_path / "prsn.csv", index=False)
    monkeypatch.setenv("NLST_DATA_ROOT_TEST", str(tmp_path))

    status = check_nlst_availability("NLST_DATA_ROOT_TEST")
    assert status.available is True
    require_nlst_available("NLST_DATA_ROOT_TEST")  # must not raise


def test_require_nlst_available_raises_actionable_error_when_missing(monkeypatch):
    monkeypatch.delenv("NLST_DATA_ROOT_TEST", raising=False)
    with pytest.raises(NLSTAccessError, match="Data Use Agreement"):
        require_nlst_available("NLST_DATA_ROOT_TEST")


def test_no_participant_data_logged_or_returned_by_availability_check(monkeypatch, tmp_path):
    """The availability check must only report booleans/paths — never the
    actual row contents — so nothing participant-level leaks into a log
    or a CI artifact via this adapter."""
    pd.DataFrame({
        "pid": [9999], "CIGSMOK": [1], "CIGAR": [0],
    }).to_csv(tmp_path / "screen.csv", index=False)
    pd.DataFrame({"pid": [9999], "candx": [1]}).to_csv(tmp_path / "prsn.csv", index=False)
    monkeypatch.setenv("NLST_DATA_ROOT_TEST", str(tmp_path))

    status = check_nlst_availability("NLST_DATA_ROOT_TEST")
    d = status.to_dict()
    assert "9999" not in str(d)
    assert all(isinstance(v, (bool, str, type(None))) for v in d.values())
