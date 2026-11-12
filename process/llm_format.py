"""Helpers that normalise provider responses and reasoning/thinking settings."""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("process.llm")

# Anthropic models with the extended-thinking API: claude-3-7-*, the
# legacy-style claude-4-* and the current family-first IDs such as
# claude-sonnet-4-5-* / claude-haiku-4-5-* / claude-opus-4-1-*.
_CLAUDE_THINKING_RE = re.compile(r"^claude-(?:3-7|[4-9]|(?:opus|sonnet|haiku)-[4-9])\b")

# OpenAI models that take max_completion_tokens and a fixed temperature.
_OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _extract_reasoning(message: Any) -> str:
    """Extract the reasoning/thinking chain from a thinking model's response.

    Different providers expose this differently:
    - OpenAI o-series: message.reasoning_content
    - Some providers: message.reasoning or message.thought
    - Content blocks: list with type="thinking" blocks
    Returns empty string if no reasoning found.
    """
    # Direct attributes (most common for thinking models)
    for attr in ("reasoning_content", "reasoning", "thought", "thinking"):
        val = getattr(message, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Content as list of blocks — look for "thinking" type blocks
    content = getattr(message, "content", None)
    if isinstance(content, list):
        thinking_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                thinking_parts.append((block.get("thinking") or block.get("text") or "").strip())
            elif hasattr(block, "type") and getattr(block, "type", None) == "thinking":
                thinking_parts.append((getattr(block, "thinking", "") or getattr(block, "text", "") or "").strip())
        if thinking_parts:
            return "\n".join(p for p in thinking_parts if p)

    # model_dump() fallback
    if hasattr(message, "model_dump"):
        d = message.model_dump()
        for key in ("reasoning_content", "reasoning", "thought", "thinking"):
            if key in d and isinstance(d[key], str) and d[key].strip():
                return d[key].strip()

    return ""


def _is_reasoning_model(model: str) -> bool:
    """Return True for models known to support explicit reasoning/thinking."""
    m = (model or "").lower()
    return (
        m.startswith(_OPENAI_REASONING_PREFIXES)
        or "-reasoning" in m
        or _supports_anthropic_thinking(m)
    )


def _openai_reasoning_effort(think: bool | str | None) -> str | None:
    """Convert pipeline think hint to OpenAI reasoning_effort value."""
    if not think:
        return None
    if isinstance(think, str):
        effort = think.strip().lower()
        if effort in {"low", "medium", "high"}:
            return effort
        if effort in {"true", "on"}:
            return "medium"
        if effort in {"false", "off"}:
            return None
    return "medium"


def _anthropic_thinking(
    think: bool | str | None,
    max_tokens: int,
) -> dict[str, Any] | None:
    """Build Anthropic thinking parameter when requested and feasible."""
    if not think:
        return None
    # budget_tokens must be > 1024 AND < max_tokens, so thinking is only
    # possible when max_tokens > 1024.
    if max_tokens <= 1024:
        logger.warning(
            "Anthropic thinking requires max_tokens > 1024 (got %d); disabling thinking",
            max_tokens,
        )
        return None
    budget = max(1024, min(int(max_tokens * 0.8), max_tokens - 1))
    return {"type": "enabled", "budget_tokens": budget}


def _supports_anthropic_thinking(model: str) -> bool:
    """Return True for Anthropic models that expose the thinking API."""
    return bool(_CLAUDE_THINKING_RE.match((model or "").lower()))


def _extract_message_text(message: Any) -> str:
    """Get plain answer text from a message; handle content as str or list of parts.

    Only the answer ``content`` is read. Reasoning fields (``reasoning``,
    ``reasoning_content``, ``thought`` ...) are never returned: when content
    is ``None`` or empty the result is ``""`` so callers treat the call as
    failed and ``chat()`` does not cache it. Use ``_extract_reasoning`` for
    the thinking chain.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append((block.get("text") or "").strip())
            elif hasattr(block, "text"):
                parts.append((getattr(block, "text", "") or "").strip())
        return " ".join(p for p in parts if p).strip()
    # Raw dict or other shape (e.g. from model_dump())
    if hasattr(message, "model_dump"):
        d = message.model_dump()
        c = d.get("content")
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):
            return " ".join(
                (x.get("text", "") if isinstance(x, dict) else str(x)).strip()
                for x in c
            ).strip()
    return ""


def _status_icon(finish_reason: str | None, raw_content) -> str:
    """Single-char icon for quick visual scanning of responses."""
    # "length" is the OpenAI truncation signal, "max_tokens" is Anthropic's.
    truncated = finish_reason in ("length", "max_tokens")
    if truncated and raw_content is None:
        return "⏱"  # thinking ran out of tokens
    if truncated:
        return "✂"  # truncated but has content
    if raw_content is None or raw_content == "":
        return "∅"  # empty answer
    return "✓"  # good response
