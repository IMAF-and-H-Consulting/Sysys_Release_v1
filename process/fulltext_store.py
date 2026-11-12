"""Storage helpers for downloaded full-text artifacts."""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import logging
import os
import pickle
import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from acquire.fulltext import write_json_atomic

logger = logging.getLogger(__name__)


def _is_cancelled(cancel_event: Any) -> bool:
    if cancel_event is None:
        return False
    if hasattr(cancel_event, "is_set"):
        return cancel_event.is_set()
    return bool(cancel_event)


def _row_key(row: dict) -> str:
    return str(
        row.get("doi")
        or row.get("pmc")
        or row.get("key")
        or row.get("title")
        or row.get("path")
        or ""
    ).strip()


_HASH_CHUNK_BYTES = 1 << 20


def _atomic_write_file(path: Path, write: Callable[[Any], None]) -> None:
    """Call ``write(binary_file)`` on a temp file, then move it onto ``path``.

    A crash or error mid-write leaves any previous ``path`` untouched and no
    partial file under the final name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _sha256_of(open_binary: Callable[[], Any]) -> str:
    digest = hashlib.sha256()
    with open_binary() as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_matches_source(archive_path: Path, source_path: Path) -> bool:
    """True when the gzip archive decompresses to exactly the source bytes."""
    try:
        return _sha256_of(lambda: gzip.open(archive_path, "rb")) == _sha256_of(
            lambda: open(source_path, "rb"))
    except (OSError, EOFError) as exc:
        logger.warning("Could not verify archive %s: %s", archive_path, exc)
        return False


def _write_gzip_archive(source_path: Path, archive_path: Path) -> None:
    def _compress(raw) -> None:
        with open(source_path, "rb") as src, gzip.GzipFile(
            filename=source_path.name, mode="wb", fileobj=raw,
        ) as dst:
            shutil.copyfileobj(src, dst)

    _atomic_write_file(archive_path, _compress)


def _archive_one_source(row: dict, archive_dir: Path) -> dict:
    """Gzip a loose original next to its text and delete it once verified.

    The archive is always rebuilt from the source that is present: a freshly
    downloaded original replaces a stale archive, and an archive left by an
    interrupted run is never trusted over the intact source.
    """
    updated = dict(row)
    source_value = str(row.get("source_path") or "").strip()
    if not source_value:
        return updated
    source_path = Path(source_value)
    if not source_path.is_file() or source_path.suffix.lower() == ".txt":
        return updated

    archive_path = archive_dir / f"{source_path.name}.gz"
    _write_gzip_archive(source_path, archive_path)
    if not _archive_matches_source(archive_path, source_path):
        logger.warning("Archive %s does not match %s; keeping the original",
                       archive_path, source_path)
        return updated

    source_path.unlink(missing_ok=True)
    updated["source_archive_path"] = str(archive_path)
    updated["source_archive_format"] = "gzip"
    updated["source_path"] = ""
    return updated


def archive_original_sources(
    rows: list[dict],
    archive_dir: Path,
    *,
    workers: int = 3,
) -> list[dict]:
    """Compress loose PDF/HTML originals and keep extracted text in place."""
    if not rows:
        return []

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        return list(pool.map(lambda row: _archive_one_source(row, archive_dir), rows))


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"indexed": []}
    try:
        import json

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("indexed", [])
            return data
    except Exception:
        logger.warning("Unable to read PaperQA manifest; rebuilding index", exc_info=True)
    return {"indexed": []}


# Globals a PaperQA ``Docs`` pickle legitimately references (checked against
# real indexes). Anything else, e.g. ``os.system`` in a crafted backup, is
# refused before it can run. (PaperQA's pydantic JSON is not used instead:
# loading it re-runs DocDetails validators that rebuild every BibTeX citation,
# roughly 50 s for a 3,000-chunk index versus about 1 s for the pickle.)
_DOCS_PICKLE_ALLOWED_GLOBALS = frozenset({
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("datetime", "date"),
    ("datetime", "datetime"),
    ("datetime", "timedelta"),
    ("datetime", "timezone"),
    ("uuid", "UUID"),
    ("paperqa.docs", "Docs"),
    ("paperqa.llms", "NumpyVectorStore"),
    ("paperqa.types", "Doc"),
    ("paperqa.types", "DocDetails"),
    ("paperqa.types", "Text"),
})
_PAPERQA_EMBEDDING = "st-multi-qa-MiniLM-L6-cos-v1"


class _DocsUnpickler(pickle.Unpickler):
    """Unpickler that only resolves the classes a PaperQA index needs."""

    source = ""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _DOCS_PICKLE_ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle
                f"Refusing to load {self.source}: it references {module}.{name}, "
                "which a PaperQA index never contains. The file may come from an "
                "untrusted backup; delete it to rebuild the index from the "
                "extracted text."
            )
        return super().find_class(module, name)


def _load_docs(docs_path: Path) -> Any:
    """Load a pickled PaperQA ``Docs`` without executing arbitrary code."""
    with open(docs_path, "rb") as f:
        unpickler = _DocsUnpickler(f)
        unpickler.source = str(docs_path)
        return unpickler.load()


def _write_docs_atomic(docs: Any, docs_path: Path) -> None:
    # Pickle is accepted here: loads go through the _DocsUnpickler allow-list (audit 06-M3).
    _atomic_write_file(docs_path, lambda f: pickle.dump(docs, f))  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle


def _paperqa_settings() -> Any:
    """Settings that only embed: no per-document metadata lookups over the network."""
    from paperqa import Settings

    # PaperQA defaults to unprefixed env reads, colliding with host flags like AGENT=1.
    return Settings(
        _env_prefix="PAPERQA_",
        embedding=_PAPERQA_EMBEDDING,
        parsing={"use_doc_details": False},
    )


def _progress_emitter(progress_callback: Any, total: int) -> Callable[[int], None]:
    """Best-effort progress, like ``stages._common.deliver_progress``.

    Not imported from there: importing the stages package loads every stage
    module, which a storage helper should not depend on.
    """
    def _emit(current: int) -> None:
        if not progress_callback:
            return
        try:
            progress_callback(f"PaperQA indexed {current}/{total} texts")
        except Exception as exc:
            logger.warning("Progress callback failed (update skipped): %s", exc, exc_info=True)

    return _emit


def _indexable_text_path(row: dict, key: str, indexed: set[str]) -> Path | None:
    text_value = str(row.get("path") or "").strip()
    if not key or key in indexed or not text_value:
        return None
    text_path = Path(text_value)
    return text_path if text_path.is_file() else None


async def _add_rows_to_paperqa(
    rows: list[dict],
    db_dir: Path,
    manifest: dict,
    *,
    progress_callback: Any = None,
    cancel_event: Any = None,
    batch_size: int = 100,
) -> dict:
    from paperqa import Docs

    docs_path = db_dir / "paperqa_docs.pkl"
    indexed = set(str(item) for item in manifest.get("indexed", []))
    if indexed and not docs_path.exists():
        logger.info(
            "PaperQA pickle missing; clearing stale manifest (%d entries)",
            len(indexed),
        )
        indexed = set()
        manifest["indexed"] = []
    # Skip the expensive pickle load entirely when there is no new work:
    # unpickling a multi-hundred-MB Docs object just to discover that
    # every row is already indexed wastes both RAM and time.
    pending_keys = {_row_key(row) for row in rows} - indexed
    pending_keys.discard("")
    if not pending_keys:
        return {
            "available": True,
            "docs_path": str(docs_path),
            "added": 0,
            "skipped": len(rows),
            "failed": [],
            "indexed_total": len(indexed),
            "cancelled": _is_cancelled(cancel_event),
        }
    if docs_path.exists():
        docs = _load_docs(docs_path)
    else:
        docs = Docs()
        manifest["indexed"] = []

    settings = _paperqa_settings()
    added = 0
    skipped = 0
    failed: list[dict[str, str]] = []
    total = len(rows)
    _emit = _progress_emitter(progress_callback, total)

    processed = 0
    for i, row in enumerate(rows):
        if _is_cancelled(cancel_event):
            logger.info("PaperQA indexing cancelled at %d/%d", i, total)
            break

        processed = i + 1
        key = _row_key(row)
        text_path = _indexable_text_path(row, key, indexed)
        if text_path is None:
            skipped += 1
            continue
        try:
            await docs.aadd(
                str(text_path),
                citation=str(row.get("title") or key),
                docname=key,
                title=str(row.get("title") or ""),
                doi=str(row.get("doi") or ""),
                settings=settings,
            )
        except Exception as exc:
            logger.warning("PaperQA could not index a text; continuing", exc_info=True)
            failed.append({"key": key, "error": str(exc)})
            continue
        indexed.add(key)
        added += 1

        # Checkpoint on added rows, not positions, so skipped rows at batch
        # boundaries never postpone it; a crash loses at most one batch.
        if added % batch_size == 0:
            _emit(processed)
            try:
                _write_docs_atomic(docs, docs_path)
                manifest["indexed"] = sorted(indexed)
            except Exception as exc:
                logger.warning("PaperQA incremental checkpoint failed: %s", exc, exc_info=True)

    _emit(processed)
    _write_docs_atomic(docs, docs_path)
    manifest["indexed"] = sorted(indexed)
    return {
        "available": True,
        "docs_path": str(docs_path),
        "added": added,
        "skipped": skipped,
        "failed": failed,
        "indexed_total": len(indexed),
        "cancelled": _is_cancelled(cancel_event),
    }


def update_fulltext_store(
    rows: list[dict],
    store_dir: Path,
    *,
    workers: int = 3,
    paperqa_enabled: bool = True,
    paperqa_batch_size: int = 100,
    progress_callback: Any = None,
    cancel_event: Any = None,
) -> tuple[list[dict], dict]:
    """Archive originals and optionally index extracted text with PaperQA."""
    store_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = store_dir / "source_archive"
    paperqa_dir = store_dir / "paperqa"
    manifest_path = paperqa_dir / "manifest.json"

    archived_rows = archive_original_sources(rows, archive_dir, workers=workers)
    archived_count = sum(1 for row in archived_rows if row.get("source_archive_path"))

    if _is_cancelled(cancel_event):
        return archived_rows, {
            "archive_dir": str(archive_dir),
            "archived_originals": archived_count,
            "paperqa": {"available": False, "error": "cancelled"},
        }

    paperqa_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)

    if not paperqa_enabled:
        write_json_atomic(manifest_path, manifest)
        return archived_rows, {
            "archive_dir": str(archive_dir),
            "archived_originals": archived_count,
            "paperqa": {
                "available": False,
                "skipped": True,
                "reason": "disabled",
                "indexed_total": len(manifest.get("indexed", [])),
            },
        }

    try:
        paperqa_result = asyncio.run(_add_rows_to_paperqa(
            archived_rows,
            paperqa_dir,
            manifest,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            batch_size=max(1, int(paperqa_batch_size)),
        ))
    except Exception as exc:
        logger.warning("PaperQA indexing failed; retaining stored text", exc_info=True)
        paperqa_result = {
            "available": False,
            "error": str(exc),
            "docs_path": "",
            "added": 0,
            "skipped": 0,
            "failed": [],
            "indexed_total": 0,
        }

    write_json_atomic(manifest_path, manifest)
    metadata = {
        "archive_dir": str(archive_dir),
        "archived_originals": archived_count,
        "paperqa": paperqa_result,
    }
    return archived_rows, metadata
