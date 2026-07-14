"""
Regression tests for benchmarks/atomic_io.py — genuinely atomic JSON/CSV
writes (issue 5): a reader must always see either the previous complete file
or the new complete file, never a partial write, and a failed write must
never leave a stray temp file or a corrupted destination behind.
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from benchmarks.atomic_io import (
    atomic_write_csv_rows,
    atomic_write_json,
    read_and_verify_csv_rows,
    read_and_verify_json,
)


def test_atomic_write_json_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        atomic_write_json(path, {"a": 1, "b": [1, 2, 3]})
        assert json.loads(path.read_text()) == {"a": 1, "b": [1, 2, 3]}


def test_atomic_write_leaves_no_temp_file_behind():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        atomic_write_json(path, {"a": 1})
        leftovers = [p for p in Path(tmp).iterdir() if p.name != "out.json"]
        assert leftovers == []


def test_atomic_write_replaces_existing_file_completely():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        atomic_write_json(path, {"version": 1})
        atomic_write_json(path, {"version": 2})
        assert json.loads(path.read_text()) == {"version": 2}


def test_existing_destination_survives_a_failed_serialization():
    """If json.dumps() itself fails (unserializable object), the previous
    complete file must remain intact — atomic_write_json builds the full
    payload before ever touching the filesystem."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        atomic_write_json(path, {"version": 1})

        class Unserializable:
            def __repr__(self):
                raise RuntimeError("cannot even repr this")

        with pytest.raises(RuntimeError):
            atomic_write_json(path, {"bad": Unserializable()})
        assert json.loads(path.read_text()) == {"version": 1}
        leftovers = [p for p in Path(tmp).iterdir() if p.name != "out.json"]
        assert leftovers == []


def test_atomic_write_csv_rows_round_trips_and_verifies():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.csv"
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        atomic_write_csv_rows(path, rows)
        read_and_verify_csv_rows(path, expected_row_count=2)


def test_read_and_verify_csv_rows_rejects_corrupted_row_count():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.csv"
        atomic_write_csv_rows(path, [{"a": 1}])
        with pytest.raises(RuntimeError):
            read_and_verify_csv_rows(path, expected_row_count=5)


def test_read_and_verify_json_rejects_mismatched_content():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        atomic_write_json(path, {"a": 1})
        with pytest.raises(RuntimeError):
            read_and_verify_json(path, {"a": 2})


def test_atomic_write_creates_parent_directories():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested" / "dirs" / "out.json"
        atomic_write_json(path, {"a": 1})
        assert path.exists()
