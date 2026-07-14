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


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte of `data` to `fd`. `os.write()` is only guaranteed to
    write *up to* the requested number of bytes — a short write (partial
    progress) is normal, expected OS behavior, not an error, and must be
    retried with the remaining slice. A return of 0 with bytes still pending
    is treated as a hard error rather than retried forever (it does not
    happen in practice for a regular file, but looping on it would hang).
    `InterruptedError` (EINTR) is retried explicitly as defense in depth,
    even though CPython's os.write() already retries EINTR internally per
    PEP 475."""
    view = memoryview(data)
    total = len(view)
    written = 0
    while written < total:
        try:
            n = os.write(fd, view[written:])
        except InterruptedError:
            continue
        if n == 0:
            raise OSError(
                f"os.write() made no progress with {total - written} of "
                f"{total} bytes remaining"
            )
        written += n


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(path)
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        try:
            _write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
    except BaseException:
        # Any failure before the file is fully written, flushed and closed
        # must never leave the (unfinished, unreferenced) temp file behind,
        # and must never touch the destination — it is still exactly what
        # it was before this call started.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
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


def exclusive_create_bytes(path: Union[str, Path], data: bytes, mode: int = 0o644) -> None:
    """Atomically CREATE `path` (fails with FileExistsError if it already
    exists — this is create-if-absent ownership acquisition, not a
    replace-if-present write) and durably persist every byte of `data`
    before it becomes visible under `path` at all.

    Writing `data` directly into an `O_CREAT | O_EXCL`-opened `path` would
    leave a window — between the open() succeeding and the write/fsync
    completing — during which `path` exists but holds truncated content; a
    concurrent reader checking "does this exist" in that window would see a
    corrupt file, not a clean not-yet-created one. To avoid that window
    entirely, the full payload is first written and fsynced to a uniquely
    named temp file in the same directory, and only then linked into `path`
    via `os.link()` — a hard link is a single atomic filesystem operation
    that fails with `FileExistsError` if `path` already exists (giving the
    same race-free create-if-absent guarantee as `O_CREAT | O_EXCL`), and
    the instant it succeeds, `path` refers to content that was already
    complete and durable. The temp file is always removed afterward — since
    it is a second hard link to the same inode as `path` once linking
    succeeds, deleting it never touches `path`'s content.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temp_path(path)
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    try:
        try:
            _write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    try:
        os.link(str(tmp), str(path))
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
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
