"""
benchmarks/atomic_io.py — genuinely atomic file writes for every JSON/CSV
artifact this framework persists.

`open(path, "w")` followed by writes is NOT atomic: a reader (or a crash
mid-write) can observe a truncated/partial file. The pattern here — write a
uniquely named temp file in the SAME directory, flush, fsync, close, then
`os.replace()` onto the destination — guarantees a concurrent reader always
sees either the previous complete file or the new complete file, never a
partial one, because `os.replace()` is a single atomic rename on every POSIX
filesystem (and on Windows, `os.replace` provides the same guarantee, unlike
`os.rename`).
"""

import json
import os
import uuid
from pathlib import Path
from typing import List, Union


def _temp_path(path: Path) -> Path:
    return path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(path)
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(str(tmp), str(path))
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    # Best-effort parent-directory fsync so the rename itself is durable —
    # not supported on every platform (e.g. Windows), so failure here is not
    # a correctness problem, only a durability-under-power-loss one.
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except (OSError, AttributeError):
        pass


def atomic_write_json(path: Union[str, Path], obj) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, default=str).encode("utf-8"))


def atomic_write_csv_rows(path: Union[str, Path], rows: List[dict]) -> None:
    import csv
    import io

    path = Path(path)
    if not rows:
        atomic_write_bytes(path, b"")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, restval="")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, buf.getvalue().encode("utf-8"))


def read_and_verify_json(path: Union[str, Path], expected: object) -> None:
    """Reload a just-written JSON artifact and confirm it round-trips to the
    same object the caller intended to persist — catches truncation,
    encoding, or serialization bugs that a "the write call didn't raise"
    check alone would miss."""
    path = Path(path)
    with open(path) as f:
        reloaded = json.load(f)
    reexpected = json.loads(json.dumps(expected, default=str))
    if reloaded != reexpected:
        raise RuntimeError(f"atomic_io: reload verification failed for {path} — "
                            "persisted content does not match what was written.")


def read_and_verify_csv_rows(path: Union[str, Path], expected_row_count: int) -> None:
    import csv

    path = Path(path)
    with open(path, newline="") as f:
        reloaded = list(csv.DictReader(f))
    if len(reloaded) != expected_row_count:
        raise RuntimeError(
            f"atomic_io: reload verification failed for {path} — expected "
            f"{expected_row_count} rows, found {len(reloaded)}."
        )
