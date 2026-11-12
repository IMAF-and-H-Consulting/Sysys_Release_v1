"""Read extracted claim fields regardless of where a stage stored them.

Stage 3 nests extracted fields under ``paper["claims"]``; older outputs and
some fixtures store them at the top level. Field names also changed over
time (``brain_regions`` became ``key_entities``). Every reader goes through
these helpers so renamed or nested fields never silently read as empty.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PRIMARY_OUTCOME_FIELD = "supports_primary_question"

# Current field name -> older names still found in saved runs and configs.
LEGACY_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "key_entities": ("brain_regions",),
}

_TRUE_STRINGS = {"true", "yes", "1"}


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _lookup(container: Mapping[str, Any], names: tuple[str, ...]) -> tuple[bool, Any]:
    """First present name, preferring a non-empty legacy alias over an empty value."""
    found, first = False, None
    for name in names:
        if name not in container:
            continue
        if not _is_empty(container[name]):
            return True, container[name]
        if not found:
            found, first = True, container[name]
    return found, first


def claim_value(paper: Mapping[str, Any], name: str, default: Any = None) -> Any:
    """Return a claim field, checking nested claims first, then the paper itself."""
    names = (name, *LEGACY_FIELD_ALIASES.get(name, ()))
    claims = paper.get("claims")
    for container in (claims, paper):
        if isinstance(container, Mapping):
            found, value = _lookup(container, names)
            if found:
                return value
    return default


def claim_list(paper: Mapping[str, Any], name: str) -> list[str]:
    """A list-typed claim field as clean strings (scalars become one item)."""
    value = claim_value(paper, name)
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def claim_flag(paper: Mapping[str, Any], name: str = PRIMARY_OUTCOME_FIELD) -> bool:
    """A boolean claim field; "yes"/"true"/"1" and non-zero numbers count as true."""
    value = claim_value(paper, name, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_STRINGS
    if isinstance(value, (int, float)):
        return bool(value)  # an LLM that emits 1/0 instead of true/false
    return False


# Tri-state (plus "unknown") direction of a paper's evidence relative to the
# primary question. Stage 3 writes it under ``claims``; older records and
# other producers store it at the top level under one of DIRECTION_KEYS.
DIRECTION_FIELD = "evidence_direction"
DIRECTION_KEYS = (DIRECTION_FIELD, "direction", "llm_direction")
SUPPORTS = "supports"
CONTRADICTS = "contradicts"
NEUTRAL = "neutral"
UNKNOWN_DIRECTION = "unknown"

_DIRECTION_ALIASES: dict[str, tuple[str, ...]] = {
    SUPPORTS: ("supports", "support", "supporting", "supportive", "for", "positive"),
    CONTRADICTS: ("contradicts", "contradict", "contradicting", "contradictory",
                  "against", "opposes", "refutes", "negative"),
    NEUTRAL: ("neutral", "mixed", "null", "inconclusive", "tangential"),
    UNKNOWN_DIRECTION: ("unknown", "unclear", "not_assessed", "not assessed"),
}
_DIRECTION_LOOKUP = {
    alias: canonical
    for canonical, aliases in _DIRECTION_ALIASES.items()
    for alias in aliases
}


def normalize_direction(value: Any) -> str | None:
    """Canonical direction for a recorded value, or None when unrecognised."""
    raw = getattr(value, "value", value)  # EvidenceDirection enums
    text = " ".join(str(raw or "").strip().lower().replace("-", " ").split())
    return _DIRECTION_LOOKUP.get(text) or _DIRECTION_LOOKUP.get(text.replace(" ", "_"))


def claim_direction(paper: Mapping[str, Any]) -> str:
    """Evidence direction: supports, contradicts, neutral or unknown.

    An explicit direction (nested under ``claims`` or at the top level) wins.
    Without one, only the boolean outcome flag is available: True means
    "supports", while False cannot separate contradicting from neutral
    evidence and is reported as "unknown".
    """
    for key in DIRECTION_KEYS:
        direction = normalize_direction(claim_value(paper, key))
        if direction:
            return direction
    return SUPPORTS if claim_flag(paper) else UNKNOWN_DIRECTION
