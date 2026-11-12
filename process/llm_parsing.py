"""Parsing of LLM JSON replies and trimming of boilerplate from long prompts."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("process.llm")


# Kept for importers of ``process.llm``; parse_llm_json no longer uses it
# because it matches the first (or an inner) object, not the final answer.
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)

_THINK_BLOCK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
_FENCED_BLOCK_RE = re.compile(r"```(?:json)?[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)

_OPENERS = "{["
_CLOSERS = "}]"


def _strip_think_blocks(text: str) -> str:
    """Remove inline reasoning so draft verdicts are never parsed as answers.

    Drops complete ``<think>…</think>`` blocks, everything before an orphan
    closing tag (servers that strip only the opening tag), and everything
    after an unterminated opening tag (reasoning that never reached an answer).
    """
    cleaned = _THINK_BLOCK_RE.sub("", text)
    closes = list(_THINK_CLOSE_RE.finditer(cleaned))
    if closes:
        cleaned = cleaned[closes[-1].end():]
    opening = _THINK_OPEN_RE.search(cleaned)
    if opening:
        cleaned = cleaned[:opening.start()]
    return cleaned


def _bracket_span_end(text: str, start: int) -> int:
    """Index just past the bracket span opened at ``start`` (len if unclosed).

    Brackets inside JSON strings are ignored.
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS:
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def _top_level_json_values(text: str) -> list[Any]:
    """Parse every complete top-level JSON object/array in ``text``, in order.

    Values nested inside another bracket span (complete or truncated) are
    never yielded on their own, so an inner fragment cannot become the result.
    """
    decoder = json.JSONDecoder()
    values: list[Any] = []
    i = 0
    while i < len(text):
        if text[i] not in _OPENERS:
            i += 1
            continue
        try:
            value, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = _bracket_span_end(text, i)
            continue
        values.append(value)
        i = end
    return values


def _final_json_payload(text: str) -> Any:
    """Return the last top-level object, or array of objects, in ``text``."""
    for value in reversed(_top_level_json_values(text)):
        if isinstance(value, dict):
            return value
        if isinstance(value, list) and any(isinstance(v, dict) for v in value):
            return value
    return None


def parse_llm_json(text: str) -> dict[str, Any] | None:
    """Extract the answer JSON object from an LLM response.

    ``<think>`` reasoning is removed first. A fenced ```json block is
    preferred (last one wins); otherwise the last complete top-level JSON
    value in the reply is used, so echoed examples and draft verdicts that
    precede the answer are ignored. Returns ``None`` when the answer is not
    an object — including an array payload (callers that accept arrays scan
    for them separately) and a truncated object, whose nested fragments are
    never returned.
    """
    if not text or not text.strip():
        return None
    cleaned = _strip_think_blocks(text)
    fenced = [m.group(1) for m in _FENCED_BLOCK_RE.finditer(cleaned)]
    for candidate in [*reversed(fenced), cleaned]:
        payload = _final_json_payload(candidate)
        if payload is not None:
            return payload if isinstance(payload, dict) else None
    return None


# ---------------------------------------------------------------------------
# Prompt compression — strip low-signal boilerplate before sending long texts
# to an LLM. This reduces transfer bytes and lets fixed char budgets cover
# more substantive content. It is deliberately conservative:
#  * no-op on texts under _COMPRESS_MIN_CHARS (short abstracts pass through
#    byte-identical);
#  * only strips sections in the trailing part of the text, where
#    references / acknowledgments / funding boilerplate live;
#  * never reorders or rewrites the remaining prose.
# ---------------------------------------------------------------------------


_COMPRESS_MIN_CHARS = 2000


_REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*(references|bibliography|works?\s+cited|literature\s+cited)\s*$"
)


# Boilerplate sections that sit between the body and the references.
_BOILERPLATE_HEADING_RE = re.compile(
    r"(?im)^\s*(acknowledg(e)?ments?|funding|author\s+contributions?"
    r"|conflicts?\s+of\s+interest|decla?rations?|data\s+availability"
    r"|supplementary\s+(material|information))\s*:?\s*$"
)


def compress_prompt_text(text: str) -> str:
    """Remove trailing reference/bibliography and boilerplate sections.

    Returns the text unchanged when it is short or has no such sections, so
    clean abstracts are never altered. Whitespace runs are collapsed to bound
    the byte size of long inputs.
    """
    if not text or len(text) < _COMPRESS_MIN_CHARS:
        return text
    cut = len(text)
    # References live in the back half of a document; only cut there so a
    # mid-text mention of the word can never truncate the body.
    for m in _REFERENCE_HEADING_RE.finditer(text):
        if m.start() > len(text) * 0.5:
            cut = min(cut, m.start())
    # Acknowledgments/funding/etc. sit just ahead of the references.
    for m in _BOILERPLATE_HEADING_RE.finditer(text):
        if m.start() > len(text) * 0.6:
            cut = min(cut, m.start())
    text = text[:cut]
    # Collapse 3+ newlines to a paragraph break; collapse doubled-up
    # spaces/tabs left by PDF extraction. Single spacing is preserved.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
