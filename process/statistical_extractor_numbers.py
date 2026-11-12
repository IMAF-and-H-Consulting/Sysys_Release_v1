"""Numeric parsing and decimal-tail helpers."""
from __future__ import annotations

import re

# Typographic characters that PDFs and journals use inside numbers: Unicode
# minus, en dash as minus, Lancet-style middle-dot decimals and superscripts.
_NUMBER_CHARS = str.maketrans({
    "\u2212": "-", "\u2013": "-", "\u00b7": ".", "\u207b": "-", "\u207a": "+",
    "\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4",
    "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9",
})
_TIMES_TEN_RE = re.compile(r"[\u00d7xX]10\^?")
_DECIMAL_POINTS = (".", "\u00b7")


def normalize_number(raw: str) -> float | None:
    """Parse a reported number, including ``0·45``, ``1.2e-5`` and ``1.2 × 10−5``."""
    raw = str(raw or "").strip().translate(_NUMBER_CHARS)
    raw = re.sub(r"\s+", "", raw)
    raw = _TIMES_TEN_RE.sub("e", raw)
    if raw.startswith("."):
        raw = "0" + raw
    if raw.startswith("-."):
        raw = raw.replace("-.", "-0.", 1)
    if raw.startswith("+."):
        raw = raw.replace("+.", "+0.", 1)
    try:
        return float(raw)
    except ValueError:
        return None


def _has_spaced_decimal(text: str) -> bool:
    return bool(re.search(r"[-+\u2212]?(?:\d+\s*\.\s+\d+|\.\s+\d+)", text))


def _closest_context_match(matches: list[re.Match[str]], context: str) -> re.Match[str] | None:
    """Pick the likely source match in an old centered context window."""
    if not matches:
        return None
    center = len(context) / 2
    return min(matches, key=lambda match: abs(((match.start() + match.end()) / 2) - center))


def _is_incomplete_decimal_tail(text: str, end: int, raw_value: str | None = None) -> bool:
    if raw_value and any(point in raw_value for point in _DECIMAL_POINTS):
        return False
    tail = text[end:]
    return (bool(re.match(r"\s*[.\u00b7](?:\s*)?$", tail))
            or bool(re.match(r"\s*[.\u00b7]\s+($|[^\d])", tail)))


def parse_count(raw: str) -> int | None:
    """Parse a whole-number count such as ``1,234`` or ``12 345`` (grouped thousands)."""
    digits = re.sub(r"[,\s\u2009\u202f\u00a0]", "", str(raw or ""))
    return int(digits) if digits.isdigit() else None
