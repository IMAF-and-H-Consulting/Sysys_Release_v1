"""Parsing of legacy LLM relevance replies and provider errors."""
from __future__ import annotations

import json
import re
from typing import Any

from process.llm import parse_llm_json


def _is_provider_request_validation_error(exc: Exception) -> bool:
    """Return True when a provider rejects one specific request payload/prompt."""
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)

    payload = {}
    if response is not None:
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                payload = parsed.get("error") or parsed
        except Exception:
            payload = {}

    haystack = " ".join(
        str(part or "")
        for part in (
            status,
            payload.get("code") if isinstance(payload, dict) else "",
            payload.get("type") if isinstance(payload, dict) else "",
            payload.get("message") if isinstance(payload, dict) else "",
            exc,
        )
    ).lower()
    return (
        status == 400
        or "invalid_parameter" in haystack
        or "invalid_request_error" in haystack
        or "invalid_request" in haystack
    )


def _parse_batched_relevance_response(text: str) -> dict[str, dict[str, int | str]]:
    """Parse batched JSON results keyed by paper id."""
    if not text or not text.strip():
        return {}

    # Shared helper handles fences/prose around a JSON object; the legacy
    # array-shaped payload needs its own scan (parse_llm_json is object-only).
    payload = parse_llm_json(text)
    if payload is None:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group())
            except json.JSONDecodeError:
                payload = None

    if payload is None:
        return {}

    if isinstance(payload, dict):
        payload = payload.get("results")
    if not isinstance(payload, list):
        return {}

    parsed: dict[str, dict[str, int | str]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("id", "")).strip()
        score = item.get("score")
        quote = item.get("quoted_span") or item.get("quote") or ""
        score = _coerce_score(score)
        if paper_id and score is not None:
            parsed[paper_id] = {
                "relevance_score": score,
                "relevance_quote": quote.strip() if isinstance(quote, str) else "",
            }
    return parsed


# Explicitly labelled scores only; bare digits (years, decimals, quoted
# numbers) are never read as a score.
_EXPLICIT_SCORE_PATTERNS = (
    r"(?:score|rating|relevance)\s*[:\s]+\s*([0-5])\b",
    r"\*\*score\*\*\s*[:\s]*\s*([0-5])\b",
    r"\b([0-5])\s*/\s*5\b",
    r"\b([0-5])\s+out\s+of\s+5\b",
    r"(?:rate|assign|give)\s+(?:this\s+)?(?:a\s+)?([0-5])\b",
    r"=\s*([0-5])\b",
)


def _coerce_score(value: Any) -> int | None:
    """Return a 0-5 score from an int, integral float or digit string, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        if not re.fullmatch(r"\s*[0-5](?:\.0+)?\s*", value):
            return None
        value = float(value)
    if isinstance(value, float):
        if not value.is_integer():
            return None
        value = int(value)
    if isinstance(value, int) and 0 <= value <= 5:
        return value
    return None


def _last_explicit_score(text: str) -> int | None:
    """The last explicitly labelled score near the end of a free-text reply."""
    tail = text.strip()[-600:]
    found = [
        (match.start(), int(match.group(1)))
        for pattern in _EXPLICIT_SCORE_PATTERNS
        for match in re.finditer(pattern, tail, re.IGNORECASE)
    ]
    return max(found)[1] if found else None


def _parse_relevance_response(text: str) -> dict[str, int | str | None]:
    """Extract score and quote from the model response.

    A JSON reply with a ``score`` key is authoritative; otherwise only an
    explicitly labelled score counts. Anything else is ``None`` (a failure).
    """
    if not text or not text.strip():
        return {"relevance_score": None, "relevance_quote": ""}

    json_data = parse_llm_json(text)
    if json_data is not None and "score" in json_data:
        quote = json_data.get("quoted_span") or json_data.get("quote") or ""
        return {
            "relevance_score": _coerce_score(json_data.get("score")),
            "relevance_quote": quote.strip() if isinstance(quote, str) else "",
        }

    score = _last_explicit_score(text)
    return {
        "relevance_score": score,
        "relevance_quote": _parse_relevance_quote(text) if score is not None else "",
    }


def _parse_relevance_quote(text: str) -> str:
    """Best-effort quote extraction when JSON parsing fails."""
    for pattern in (
        r'(?:quoted_span|quote)\s*[:=]\s*"([^"]+)"',
        r"(?:quoted_span|quote)\s*[:=]\s*'([^']+)'",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""
