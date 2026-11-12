"""Regex-based parsers for p-values, CIs, effect sizes, subjects, and number repair."""
from __future__ import annotations

import re
from typing import Any

from process.statistical_extractor_numbers import (
    _closest_context_match,
    _has_spaced_decimal,
    _is_incomplete_decimal_tail,
    normalize_number,
    parse_count,
)
from process.statistical_extractor_patterns import (
    AGE_RE,
    CI_RE,
    COMPARISON_RE,
    N_EQUALS_RE,
    NUMBER_RE,
    P_VALUE_RE,
    RATIO_RE,
    SAMPLE_RE,
    SUBJECT_WORDS,
    TEST_PATTERNS,
    UNLABELLED_CI_RE,
    TextBlock,
)

_UNICODE_OPERATORS = {"\u2264": "<=", "\u2265": ">="}


def repair_decimal_spaced_result_numbers(result: dict[str, Any]) -> dict[str, Any]:
    """Repair legacy rows where PDF spacing truncated decimals to integers.

    Earlier extraction patterns saw text such as ``p = 0. 001`` as ``p = 0``.
    The original context is preserved in the artifact, so the GUI can use this
    best-effort repair when loading older review files.
    """
    if not isinstance(result, dict):
        return result
    context = str(result.get("context") or "")
    if not context or not _has_spaced_decimal(context):
        return result

    repaired = dict(result)
    test_type = str(result.get("test_type") or "")
    selected_match = None
    for candidate_type, pattern in TEST_PATTERNS:
        if candidate_type != test_type:
            continue
        matches = list(pattern.finditer(context))
        match = _closest_context_match(matches, context)
        if match is None:
            break
        selected_match = match
        if _is_incomplete_decimal_tail(context, match.end(), match.groupdict().get("stat")):
            if result.get("statistic") == 0.0:
                repaired["statistic"] = ""
            break
        stat_value = normalize_number(match.groupdict().get("stat", ""))
        if stat_value is not None:
            repaired["statistic"] = stat_value
            repaired["test_text"] = match.group(0)
        break

    p_values = (
        parse_p_values_near(context, selected_match.start(), selected_match.end())
        if selected_match is not None
        else parse_p_values(context)
    )
    if p_values:
        repaired["p_values"] = p_values
    elif any(
        isinstance(item, dict) and item.get("value") == 0.0
        for item in (result.get("p_values") or [])
    ):
        repaired["p_values"] = []
    return repaired


def _p_value_matches(text: str) -> list[dict[str, Any]]:
    values = []
    for match in P_VALUE_RE.finditer(text):
        if _is_incomplete_decimal_tail(text, match.end(), match.group("value")):
            continue
        value = normalize_number(match.group("value"))
        if value is None or not 0 <= value <= 1:
            continue
        operator = match.group("operator")
        values.append({
            "operator": _UNICODE_OPERATORS.get(operator, operator),
            "value": value,
            "text": match.group(0),
            "_span": match.span(),
        })
    return values


def _without_internal_spans(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in item.items() if key != "_span"} for item in values]


def parse_p_values(text: str) -> list[dict[str, Any]]:
    return _without_internal_spans(_p_value_matches(text))


def parse_p_values_near(
    text: str,
    anchor_start: int,
    anchor_end: int,
    *,
    max_distance: int = 180,
) -> list[dict[str, Any]]:
    """Return p-values nearest a parsed statistic span."""
    values = _p_value_matches(text)
    if not values:
        return []

    def distance(item: dict[str, Any]) -> int:
        p_start, p_end = item["_span"]
        if p_start <= anchor_end and anchor_start <= p_end:
            return 0
        return min(abs(p_start - anchor_end), abs(anchor_start - p_end))

    ordered = sorted(values, key=distance)
    nearby = [item for item in ordered if distance(item) <= max_distance]
    if nearby:
        return _without_internal_spans(nearby)
    return _without_internal_spans(ordered[:1])


def parse_confidence_intervals(text: str) -> list[dict[str, Any]]:
    intervals = []
    for match in CI_RE.finditer(text):
        lower = normalize_number(match.group("lower"))
        upper = normalize_number(match.group("upper"))
        if lower is None or upper is None or lower >= upper:
            continue
        intervals.append({
            "level": int(match.group("level")),
            "lower": lower,
            "upper": upper,
            "text": match.group(0),
        })
    return intervals


def parse_unlabelled_interval(text: str, level: int) -> list[dict[str, Any]]:
    """Parse ``lo to hi`` at the start of ``text`` when its CI label came earlier."""
    match = UNLABELLED_CI_RE.match(text)
    if match is None:
        return []
    lower = normalize_number(match.group("lower"))
    upper = normalize_number(match.group("upper"))
    if lower is None or upper is None or lower >= upper:
        return []
    return [{"level": level, "lower": lower, "upper": upper, "text": match.group(0).strip()}]


def parse_effect_sizes(text: str) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    patterns = (
        re.compile(
            rf"\b(?:Cohen['']?s\s+)?(?P<type>d|g)\s*=\s*(?P<value>{NUMBER_RE})",
            re.IGNORECASE,
        ),
        RATIO_RE,
        re.compile(rf"\b(?P<type>r|rho|rs)\s*=\s*(?P<value>{NUMBER_RE})", re.IGNORECASE),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            groups = match.groupdict()
            value = normalize_number(groups.get("value") or groups.get("stat"))
            if value is None:
                continue
            effect_type = normalize_effect_type(groups.get("type") or groups.get("effect_type"))
            key = (effect_type, value)
            if key in seen:
                continue
            seen.add(key)
            effects.append({
                "type": effect_type,
                "value": value,
                "text": match.group(0),
            })
    return effects


def normalize_effect_type(value: str) -> str:
    name = str(value or "").lower().replace("-", " ")
    return {"odds ratio": "or", "risk ratio": "rr", "hazard ratio": "hr"}.get(name, name)


def parse_subjects(text: str) -> list[dict[str, Any]]:
    subjects: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    for match in SAMPLE_RE.finditer(text):
        n = parse_count(match.group("n"))
        label = match.group("label").lower()
        if n is None or not valid_sample_size(n):
            continue
        key = (n, label)
        if key in seen:
            continue
        seen.add(key)
        subjects.append({"n": n, "label": label, "text": match.group(0)})

    for match in N_EQUALS_RE.finditer(text):
        n = parse_count(match.group("n"))
        if n is None or not valid_sample_size(n):
            continue
        key = (n, "sample")
        if key in seen:
            continue
        seen.add(key)
        subjects.append({"n": n, "label": "sample", "text": match.group(0)})

    age_match = AGE_RE.search(text)
    if age_match:
        age = normalize_number(age_match.group("age"))
        sd = normalize_number(age_match.group("sd") or "")
        if age is not None:
            subjects.append({"age_mean": age, "age_sd": sd, "text": age_match.group(0)})

    return subjects


def valid_sample_size(n: int) -> bool:
    return 2 <= n <= 999999 and not (1900 <= n <= 2099)


def parse_comparisons(text: str) -> list[str]:
    comparisons = []
    for match in COMPARISON_RE.finditer(text):
        value = f"{match.group('a').strip()} vs {match.group('b').strip()}"
        if len(value) <= 100:
            comparisons.append(value)
    return comparisons[:3]


def infer_subject_terms(text: str) -> list[str]:
    lower = text.lower()
    return [word for word in SUBJECT_WORDS if word in lower]


def nearby_text(block: TextBlock, start: int, end: int, radius: int = 280) -> str:
    left = max(0, start - radius)
    right = min(len(block.text), end + radius)
    return block.text[left:right].strip()


def result_missing_fields(result: dict[str, Any], block_subjects: list[dict[str, Any]]) -> list[str]:
    missing = []
    if not result.get("p_values") and not result.get("confidence_intervals"):
        missing.append("p_value_or_ci")
    if not result.get("effect_sizes") and result.get("test_type") not in {"effect_size", "odds_or_risk_ratio", "correlation"}:
        missing.append("effect_size")
    if not block_subjects:
        missing.append("subjects")
    if not result.get("comparisons"):
        missing.append("comparison")
    return missing
