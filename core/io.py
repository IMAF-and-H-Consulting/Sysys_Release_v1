"""Shared atomic file I/O utilities for the SysS pipeline.

All writers follow the same pattern: write to a temp file in the same
directory, flush and fsync it, then atomically rename into place and fsync
the directory (where the platform allows). A process crash mid-write leaves
a .tmp file behind but never corrupts the target; the fsyncs also make the
rename durable across an OS crash or power loss.

JSON Lines appends use standard file append with fsync for O(1) per-record
I/O; a crash may leave a partial final line.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)


def json_default(obj: Any) -> Any:
    """JSON fallback that keeps numpy scalars and arrays as numbers/bools/lists.

    ``default=str`` turned ``np.int64(5)`` into ``"5"`` and ``np.bool_(True)``
    into ``"True"``. numpy is detected by module name so this module stays
    stdlib-only. Anything else still falls back to ``str`` (paths, dates,
    enums), as before.
    """
    if type(obj).__module__ == "numpy":
        if hasattr(obj, "tolist"):
            return obj.tolist()  # arrays and scalars (np.generic.tolist == item)
        if hasattr(obj, "item"):
            return obj.item()
    return str(obj)


class _JsonlWriter:
    """Append-only JSON Lines writer.

    Each record is appended in O(1) I/O. A crash mid-write may leave a
    partial final line; readers should skip malformed trailing lines.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mode = _target_file_mode(path)

    def append(self, record: Any) -> None:
        line = json.dumps(record, ensure_ascii=False, default=json_default) + "\n"
        separator = ""
        try:
            if self.path.stat().st_size:
                with open(self.path, "rb") as existing:
                    existing.seek(-1, os.SEEK_END)
                    if existing.read(1) != b"\n":
                        # Isolate a crash-truncated final record so the new
                        # complete append remains independently readable.
                        separator = "\n"
        except OSError:
            pass
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(separator + line)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(self.path, self._mode)

    def close(self) -> None:
        pass

    def __enter__(self) -> _JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _default_file_mode() -> int:
    current_umask = os.umask(0)
    os.umask(current_umask)
    return 0o666 & ~current_umask


def _target_file_mode(path: Path) -> int:
    try:
        return path.stat().st_mode & 0o777
    except OSError:
        return _default_file_mode()


def _replace_preserving_mode(tmp_path: str, path: Path, mode: int) -> None:
    os.chmod(tmp_path, mode)
    Path(tmp_path).replace(path)


def _fsync_dir(directory: Path) -> None:
    """Make a rename in ``directory`` durable. Best effort by design.

    POSIX needs this after ``replace``; Windows cannot open a directory, and
    some filesystems refuse a directory fsync. The data file itself is
    already fsynced, so failing here only weakens power-loss durability.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError as exc:
        logger.debug("Directory fsync not supported for %s: %s", directory, exc)
    finally:
        os.close(fd)


@contextmanager
def _atomic_file(path: Path, binary: bool = False) -> Iterator[IO]:
    """Yield a temp file beside ``path``; on success fsync it and rename it in.

    On any error the temp file is removed and the target is left untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = _target_file_mode(path)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=path.stem
    )
    try:
        with open(tmp_fd, "wb" if binary else "w",
                  encoding=None if binary else "utf-8") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        _replace_preserving_mode(tmp_path, path, mode)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def atomic_write_json(
    path: Path,
    data: Any,
    *,
    default=json_default,
    ensure_ascii: bool = False,
) -> None:
    """Write JSON atomically via a temp file — safe against mid-write crashes."""
    with _atomic_file(path) as f:
        json.dump(data, f, indent=2, ensure_ascii=ensure_ascii, default=default)


def atomic_write_json_array(
    path: Path,
    records: Iterable[Any],
    *,
    default=json_default,
    ensure_ascii: bool = False,
) -> int:
    """Stream an iterable of records into an atomic JSON array.

    Use this instead of ``atomic_write_json(path, list(...))`` when the caller
    can produce records lazily and each one is large — building the list first
    holds every record in memory at once. Returns the number of records
    written.

    The array is still written atomically: a crash mid-stream leaves the
    previous file untouched and only a stray .tmp behind.
    """
    count = 0
    with _atomic_file(path) as out:
        out.write("[\n")
        for record in records:
            if count:
                out.write(",\n")
            out.write(
                "  " + json.dumps(record, ensure_ascii=ensure_ascii,
                                  default=default)
            )
            count += 1
        out.write("\n]\n" if count else "]\n")
    return count


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically via a temp file — safe against mid-write crashes."""
    with _atomic_file(path) as f:
        f.write(text)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically via a temp file — safe against mid-write crashes."""
    with _atomic_file(path, binary=True) as f:
        f.write(data)


def atomic_append_jsonl(path: Path, record: Any) -> None:
    """Append a single JSON object to a JSON Lines file atomically."""
    writer = _JsonlWriter(path)
    writer.append(record)


def load_json(path: Path, default: Any = None) -> Any:
    """Load a JSON file, returning *default* when it is missing or unreadable.

    A missing file is silent. A file that exists but cannot be read or
    parsed (truncated, corrupt, wrong encoding) is logged as a WARNING so
    it is not mistaken for "no output yet".
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as exc:  # ValueError covers JSON and UTF-8 decode errors
        logger.warning("Could not read JSON file %s (%s: %s); treating it as absent.",
                       path, type(exc).__name__, exc)
        return default


def iter_jsonl_records(path: Path) -> Iterator[Any]:
    """Yield valid JSON Lines records, skipping malformed lines.

    A malformed final line is expected after a process crash and is skipped
    quietly (DEBUG). Malformed lines elsewhere mean real corruption and are
    counted and logged as a WARNING once the file has been read.
    """
    if not path.exists():
        return
    skipped_inner = 0
    last_was_bad = False
    with open(path, "rb") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            if last_was_bad:
                skipped_inner += 1
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # JSONL is append-only and fsynced per record, but a process
                # crash can still leave the last write incomplete.
                last_was_bad = True
                continue
            last_was_bad = False
            yield record
    _log_skipped_jsonl(path, skipped_inner, last_was_bad)


def _log_skipped_jsonl(path: Path, skipped_inner: int, trailing_bad: bool) -> None:
    if skipped_inner:
        logger.warning("Skipped %d malformed line(s) inside %s (not a crash-truncated tail).",
                       skipped_inner, path)
    if trailing_bad:
        logger.debug("Skipped a crash-truncated final line in %s.", path)


def jsonl_to_json_array(
    jsonl_path: Path,
    json_path: Path,
    *,
    record_key: Callable[[Any], str] | None = None,
    include_keys: set[str] | None = None,
) -> None:
    """Stream valid JSON Lines records into an atomic JSON array.

    When ``record_key`` is supplied, only the latest record for each key is
    materialized. ``include_keys`` can further restrict the final array to the
    records relevant to the current run while leaving the append-only JSONL
    checkpoint intact for later resume attempts.
    """
    last_index: dict[str, int] = {}
    if record_key is not None:
        for index, record in enumerate(iter_jsonl_records(jsonl_path)):
            key = str(record_key(record) or "")
            if key and (include_keys is None or key in include_keys):
                last_index[key] = index

    with _atomic_file(json_path) as out:
        out.write("[\n")
        first = True
        for index, record in enumerate(iter_jsonl_records(jsonl_path)):
            if record_key is not None:
                key = str(record_key(record) or "")
                if not key or (include_keys is not None and key not in include_keys):
                    continue
                if last_index.get(key) != index:
                    continue
            if not first:
                out.write(",\n")
            first = False
            out.write(
                "  " + json.dumps(record, ensure_ascii=False, default=json_default)
            )
        out.write("\n]\n")
