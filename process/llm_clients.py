"""Shared LLM runtime state: the active config, lazily built provider clients,
and the debug/thinking log writers.

Kept in one module so every reader sees the same ``_config`` that
``set_config`` assigns."""
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any

from process.custom_endpoint import anthropic_base_url, openai_base_url, server_root

if TYPE_CHECKING:
    from core.config import Config

logger = logging.getLogger("process.llm")


_config: Config | None = None


# The config captured by the chat() call running in this thread, so a
# set_config() from the next stage cannot redirect its logs mid-request.
_call_config: ContextVar[Config | None] = ContextVar("llm_call_config", default=None)


_openai_client: Any = None


_custom_openai_client: Any = None


_custom_anthropic_client: Any = None


_ollama_client: Any = None


_lmstudio_client: Any = None


_anthropic_client: Any = None


# The OpenAI/Anthropic SDKs refuse an empty key; local servers ignore it.
_NO_KEY_PLACEHOLDER = "not-needed"


_lmstudio_models_without_reasoning: set[str] = set()


# Ollama's client defaults to no timeout, so a stalled server would block a
# worker forever. Matches the LM Studio REST client's limits.
OLLAMA_TIMEOUT_SECONDS = {"connect": 30.0, "read": 600.0, "write": 60.0, "pool": 600.0}

# The parsed-answer copy in the JSONL debug log is intentionally a preview.
# Provider-specific raw response fields retain the complete reply, while this
# boundary keeps the audit stream scannable. Adjacent metadata prevents the
# preview from being mistaken for a model or stage-output limit.
LLM_DEBUG_TEXT_PREVIEW_CHARS = 3000


_client_lock = threading.Lock()


def _close_quietly(client: Any) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as e:  # best effort: a failed close must not block a reconfigure
        logger.debug("Closing LLM client %r failed: %s", client, e)


def set_config(config: Config) -> None:
    """Set the active pipeline config so chat() uses the right provider and settings.

    Holds ``_client_lock`` so a lazy getter cannot store a client built from
    the old config after the reset.
    """
    global _config, _openai_client, _custom_openai_client, _custom_anthropic_client
    global _ollama_client, _lmstudio_client, _anthropic_client
    with _client_lock:
        old_clients = (_openai_client, _custom_openai_client, _custom_anthropic_client,
                       _ollama_client, _lmstudio_client, _anthropic_client)
        _config = config
        _openai_client = None
        _custom_openai_client = None
        _custom_anthropic_client = None
        _ollama_client = None
        _lmstudio_client = None
        _anthropic_client = None
        _lmstudio_models_without_reasoning.clear()
    for client in old_clients:
        _close_quietly(client)


def _active_config() -> Config | None:
    """The config of the chat() call in progress, else the global one."""
    return _call_config.get() or _config


@contextmanager
def config_snapshot() -> Iterator[Config | None]:
    """Pin the current config for the duration of one chat() call."""
    token = _call_config.set(_config)
    try:
        yield _config
    finally:
        _call_config.reset(token)


def _append_llm_debug(record: dict) -> None:
    """Append one LLM request/response to data/llm_responses.jsonl for inspection."""
    config = _active_config()
    if config is None:
        return
    path = config.data_dir / "llm_responses.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Could not write llm_responses.jsonl: %s", e)


def _response_debug_fields(text: str | None) -> dict[str, Any]:
    complete = str(text or "")
    return {
        # Kept for compatibility with existing JSONL readers.
        "extracted_text": complete[:LLM_DEBUG_TEXT_PREVIEW_CHARS],
        "extracted_text_chars": len(complete),
        "extracted_text_is_preview": True,
        "extracted_text_preview_truncated": (
            len(complete) > LLM_DEBUG_TEXT_PREVIEW_CHARS),
        "extracted_text_preview_chars": LLM_DEBUG_TEXT_PREVIEW_CHARS,
        "extracted_text_preview_reason": (
            "Debug-log readability only; the complete provider response is retained in "
            "the raw response field and complete parsed text is returned to the stage."
        ),
    }


def _get_openai_client():
    """Lazy-init OpenAI/OpenAI-compatible client (thread-safe)."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    with _client_lock:
        # Double-check under lock
        if _openai_client is not None:
            return _openai_client
        config = _config
        if config is None or config.llm_provider != "openai":
            return None
        try:
            from openai import OpenAI
            _openai_client = OpenAI(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url.rstrip("/"),
            )
            return _openai_client
        except Exception as e:
            logger.error("OpenAI client init failed: %s", e)
            return None


def _custom_format(config: Config | None = None) -> str | None:
    """API format of the active custom endpoint, or None when not custom."""
    config = config if config is not None else _active_config()
    if config is None or config.llm_provider != "custom":
        return None
    return config.custom_api_format


def _get_custom_openai_client():
    """Lazy-init the OpenAI-compatible client for a custom endpoint."""
    global _custom_openai_client
    if _custom_openai_client is not None:
        return _custom_openai_client
    with _client_lock:
        if _custom_openai_client is not None:
            return _custom_openai_client
        config = _config
        if _custom_format(config) != "openai":
            return None
        try:
            from openai import OpenAI

            _custom_openai_client = OpenAI(
                api_key=config.custom_api_key or _NO_KEY_PLACEHOLDER,
                base_url=openai_base_url(config.custom_base_url),
            )
            return _custom_openai_client
        except Exception as e:
            logger.error("Custom OpenAI-compatible client init failed: %s", e)
            return None


def _get_custom_anthropic_client():
    """Lazy-init the Anthropic-compatible client for a custom endpoint."""
    global _custom_anthropic_client
    if _custom_anthropic_client is not None:
        return _custom_anthropic_client
    with _client_lock:
        if _custom_anthropic_client is not None:
            return _custom_anthropic_client
        config = _config
        if _custom_format(config) != "anthropic":
            return None
        try:
            import anthropic

            _custom_anthropic_client = anthropic.Anthropic(
                api_key=config.custom_api_key or _NO_KEY_PLACEHOLDER,
                base_url=anthropic_base_url(config.custom_base_url),
            )
            return _custom_anthropic_client
        except Exception as e:
            logger.error("Custom Anthropic-compatible client init failed: %s", e)
            return None


def _get_ollama_client():
    """Lazy-init Ollama client bound to the configured host."""
    global _ollama_client
    if _ollama_client is not None:
        return _ollama_client
    with _client_lock:
        if _ollama_client is not None:
            return _ollama_client
        config = _config
        if config is None or config.llm_provider != "ollama":
            return None
        try:
            import httpx
            import ollama

            host = config.ollama_base_url.rstrip("/")
            _ollama_client = ollama.Client(
                host=host, timeout=httpx.Timeout(**OLLAMA_TIMEOUT_SECONDS),
            )
            return _ollama_client
        except Exception as e:
            logger.error("Ollama client init failed: %s", e)
            return None


def _get_lmstudio_client():
    """Lazy-init the LM Studio native REST client for a custom endpoint."""
    global _lmstudio_client
    if _lmstudio_client is not None:
        return _lmstudio_client
    with _client_lock:
        if _lmstudio_client is not None:
            return _lmstudio_client
        config = _config
        if _custom_format(config) != "lmstudio":
            return None
        try:
            import httpx

            base_url = server_root(config.custom_base_url)
            headers = {"Content-Type": "application/json"}
            if config.custom_api_key:
                headers["Authorization"] = f"Bearer {config.custom_api_key}"
            _lmstudio_client = httpx.Client(
                base_url=base_url,
                headers=headers,
                timeout=httpx.Timeout(connect=30, read=600, write=60, pool=600),
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                    keepalive_expiry=120,
                ),
            )
            return _lmstudio_client
        except Exception as e:
            logger.error("LM Studio client init failed: %s", e)
            return None


def _get_anthropic_client():
    """Lazy-init Anthropic client (thread-safe)."""
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    with _client_lock:
        if _anthropic_client is not None:
            return _anthropic_client
        config = _config
        if config is None or config.llm_provider != "anthropic":
            return None
        try:
            import anthropic

            _anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
            return _anthropic_client
        except Exception as e:
            logger.error("Anthropic client init failed: %s", e)
            return None


def _llm_cache_dir() -> Path | None:
    """Resolve the on-disk LLM response cache directory, or None if disabled."""
    config = _active_config()
    if config is None:
        return None
    if not getattr(config, "llm_cache_enabled", False):
        return None
    custom = getattr(config, "llm_cache_dir", None)
    if custom:
        return Path(custom)
    output_dir = getattr(config, "output_dir", None)
    if output_dir is None:
        return None
    return Path(output_dir) / "llm_cache"


def _provider_label() -> str:
    """Log label such as ``openai`` or ``custom:lmstudio``."""
    fmt = _custom_format()
    if fmt:
        return f"custom:{fmt}"
    config = _active_config()
    return config.llm_provider if config is not None else "openai"


def _effective_base_url() -> str:
    """Return the base URL that will be used for the current provider config."""
    config = _active_config()
    if config is None:
        return ""
    if config.llm_provider == "custom":
        return getattr(config, "custom_base_url", "") or ""
    if config.llm_provider == "openai":
        return getattr(config, "openai_base_url", "") or ""
    if config.llm_provider == "ollama":
        return getattr(config, "ollama_base_url", "") or ""
    return ""


def _append_thinking_log(record: dict) -> None:
    """Append reasoning chain to data/llm_thinking.jsonl for hallucination auditing."""
    config = _active_config()
    if config is None:
        return
    path = config.data_dir / "llm_thinking.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Could not write llm_thinking.jsonl: %s", e)
