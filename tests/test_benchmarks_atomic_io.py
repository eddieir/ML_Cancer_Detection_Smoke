"""
Regression tests for benchmarks/atomic_io.py — genuinely atomic JSON/CSV
writes (issue 5): a reader must always see either the previous complete file
or the new complete file, never a partial write, and a failed write must
never leave a stray temp file or a corrupted destination behind.
"""
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import benchmarks.atomic_io as atomic_io
from benchmarks.atomic_io import (
    atomic_write_bytes,
    atomic_write_csv_rows,
    atomic_write_json,
    read_and_verify_csv_rows,
    read_and_verify_json,
)


def _leftovers(tmp: str, keep: str) -> list:
    return [p for p in Path(tmp).iterdir() if p.name != keep]


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


def test_atomic_write_handles_a_short_os_write_and_writes_every_byte(monkeypatch):
    """os.write() is only guaranteed to write *up to* the requested number of
    bytes. A real short write must be retried with the remaining slice, not
    silently accepted as "done"."""
    real_write = os.write
    calls = {"n": 0}

    def short_write(fd, data):
        calls["n"] += 1
        if calls["n"] == 1 and len(data) > 4:
            return real_write(fd, data[:4])
        return real_write(fd, data)

    monkeypatch.setattr(atomic_io.os, "write", short_write)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        payload = {"x": list(range(500))}
        atomic_write_json(path, payload)
        assert json.loads(path.read_text()) == payload
    assert calls["n"] >= 2


def test_atomic_write_retries_interrupted_error(monkeypatch):
    real_write = os.write
    calls = {"n": 0}

    def interrupted_once(fd, data):
        calls["n"] += 1
        if calls["n"] == 1:
            raise InterruptedError()
        return real_write(fd, data)

    monkeypatch.setattr(atomic_io.os, "write", interrupted_once)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        atomic_write_json(path, {"a": 1})
        assert json.loads(path.read_text()) == {"a": 1}
    assert calls["n"] >= 2


def test_atomic_write_raises_on_zero_byte_progress_and_leaves_no_temp_file(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        atomic_write_json(path, {"version": 1})
        original = path.read_bytes()

        monkeypatch.setattr(atomic_io.os, "write", lambda fd, data: 0)
        with pytest.raises(OSError):
            atomic_write_json(path, {"version": 2})

        assert path.read_bytes() == original
        assert _leftovers(tmp, "out.json") == []


def test_atomic_write_cleans_up_temp_file_on_write_failure(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        atomic_write_json(path, {"version": 1})
        original = path.read_bytes()

        def raising_write(fd, data):
            raise OSError("simulated write failure")

        monkeypatch.setattr(atomic_io.os, "write", raising_write)
        with pytest.raises(OSError):
            atomic_write_json(path, {"version": 2})

        assert path.read_bytes() == original
        assert _leftovers(tmp, "out.json") == []


def test_atomic_write_cleans_up_temp_file_on_fsync_failure(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        atomic_write_json(path, {"version": 1})
        original = path.read_bytes()

        real_fsync = os.fsync
        seen_fds = set()

        def maybe_failing_fsync(fd):
            # Only the temp-file fd triggers the simulated failure — the
            # best-effort parent-directory fsync must be left alone so this
            # test isolates the write-path failure specifically.
            if fd not in seen_fds:
                seen_fds.add(fd)
                raise OSError("simulated fsync failure")
            return real_fsync(fd)

        monkeypatch.setattr(atomic_io.os, "fsync", maybe_failing_fsync)
        with pytest.raises(OSError):
            atomic_write_json(path, {"version": 2})

        assert path.read_bytes() == original
        assert _leftovers(tmp, "out.json") == []


def test_atomic_write_cleans_up_temp_file_on_replace_failure(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        atomic_write_json(path, {"version": 1})
        original = path.read_bytes()

        def raising_replace(src, dst):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(atomic_io.os, "replace", raising_replace)
        with pytest.raises(OSError):
            atomic_write_json(path, {"version": 2})

        assert path.read_bytes() == original
        assert _leftovers(tmp, "out.json") == []


def test_atomic_write_large_payload_is_complete():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "big.bin"
        data = os.urandom(4 * 1024 * 1024)  # 4 MiB, several times a typical pipe buffer
        atomic_write_bytes(path, data)
        assert path.read_bytes() == data


def test_concurrent_readers_never_see_a_partial_or_corrupted_file():
    """A background thread reads the destination continuously while the main
    thread repeatedly overwrites it — every read must be valid JSON matching
    one of the versions actually written, never a truncated/corrupted blob."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        atomic_write_json(path, {"version": 0})

        stop = threading.Event()
        errors = []

        def reader():
            while not stop.is_set():
                try:
                    with open(path) as f:
                        content = f.read()
                    obj = json.loads(content)
                    assert "version" in obj
                except Exception as exc:  # pragma: no cover - failure path
                    errors.append(exc)

        t = threading.Thread(target=reader)
        t.start()
        try:
            for version in range(1, 60):
                atomic_write_json(path, {"version": version})
        finally:
            stop.set()
            t.join(timeout=5)

        assert errors == []
