"""On-disk LLM response cache keyed by a hash of the full request."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from core.io import atomic_write_json

logger = logging.getLogger("process.llm")


def _llm_cache_key(
    route: str,
    model: str,
    temperature: float,
    max_tokens: int,
    think: bool | str | None,
    messages: list[dict[str, str]],
    base_url: str = "",
) -> str:
    """Stable content hash identifying one exact LLM request."""
    payload = {
        "route": route,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "think": think,
        "messages": messages,
        "base_url": base_url,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _llm_cache_read(cache_dir: Path, key: str) -> str | None:
    """Return the cached response text for key, or None on miss/corruption."""
    path = cache_dir / f"{key}.json"
    try:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        text = record.get("text") if isinstance(record, dict) else None
        if isinstance(text, str) and text.strip():
            return text
    except (OSError, ValueError):
        pass
    return None


def _llm_cache_delete(cache_dir: Path, key: str) -> None:
    """Remove one stored response, if present."""
    try:
        (cache_dir / f"{key}.json").unlink(missing_ok=True)
    except OSError as e:
        logger.warning("LLM cache delete failed for %s: %s", key[:12], e)


def _llm_cache_write(cache_dir: Path, key: str, text: str) -> None:
    """Store one response. File-per-hash + atomic write: concurrent workers
    never need a lock, and a simultaneous write of the same key is benign
    (identical content)."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(cache_dir / f"{key}.json", {"text": text})
    except OSError as e:
        logger.warning("LLM cache write failed for %s: %s", key[:12], e)
