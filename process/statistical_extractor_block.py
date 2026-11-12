"""Per-block statistical result extraction."""
from __future__ import annotations

import re
from typing import Any

from process.statistical_extractor_numbers import (
    _is_incomplete_decimal_tail,
    normalize_number,
)
from process.statistical_extractor_parsers import (
    infer_subject_terms,
    nearby_text,
    normalize_effect_type,
    parse_comparisons,
    parse_confidence_intervals,
    parse_p_values_near,
    parse_subjects,
    parse_unlabelled_interval,
    result_missing_fields,
)
from process.statistical_extractor_patterns import (
    TEST_PATTERNS,
    NUMBER_RE,
    TextBlock,
)
from process.statistical_extractor_text import sentence_spans


def extract_from_block(block: TextBlock, article: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract result rows and human-review flags from one paragraph/window."""
    results: list[dict[str, Any]] = []
    review_flags: list[dict[str, Any]] = []
    block_subjects = parse_subjects(block.text)
    subject_terms = infer_subject_terms(block.text)
    comparisons = parse_comparisons(block.text)
    # Numerical fields must belong to this result, not a neighbouring effect.
    anchors = sorted({m.start() for _, pattern in TEST_PATTERNS for m in pattern.finditer(block.text)})
    sentences = sentence_spans(block.text)

    occupied: list[tuple[int, int]] = []
    for test_type, pattern in TEST_PATTERNS:
        for match in pattern.finditer(block.text):
            span = match.span()
            if _is_incomplete_decimal_tail(block.text, span[1], match.groupdict().get("stat")):
                continue
            if any(max(span[0], lo) < min(span[1], hi) for lo, hi in occupied):
                continue
            occupied.append(span)
            context = nearby_text(block, span[0], span[1])
            p_values = parse_p_values_near(block.text, span[0], span[1])
            sentence_end = next((end for start, end in sentences if start <= span[0] < end), len(block.text))
            next_anchor = next((start for start in anchors if start >= span[1]), len(block.text))
            metric_tail = block.text[span[1]:min(sentence_end, next_anchor)]
            intervals = parse_confidence_intervals(metric_tail) or _labelled_ratio_interval(match, metric_tail)
            local_subjects = parse_subjects(context) or block_subjects
            stat_value = normalize_number(match.groupdict().get("stat", ""))
            effect_type = match.groupdict().get("effect_type") or ""
            effect_sizes = []
            if test_type in {"effect_size", "odds_or_risk_ratio", "correlation"} and stat_value is not None:
                current_effect = {
                    "type": normalize_effect_type(effect_type) or ("r" if test_type == "correlation" else test_type),
                    "value": stat_value,
                    "text": match.group(0),
                }
                if not any(
                    effect.get("type") == current_effect["type"]
                    and effect.get("value") == current_effect["value"]
                    for effect in effect_sizes
                ):
                    effect_sizes.append(current_effect)

            dfs = {
                key: normalize_number(value)
                for key, value in match.groupdict().items()
                if key.startswith("df") and value is not None
            }
            result = {
                "test_type": test_type,
                "test_text": match.group(0),
                "statistic": stat_value,
                "degrees_of_freedom": dfs,
                "effect_sizes": effect_sizes,
                "p_values": p_values,
                "confidence_intervals": intervals,
                "sample_sizes": local_subjects,
                "subject_terms": subject_terms,
                "comparisons": parse_comparisons(context) or comparisons,
                "context": context,
                "location": {
                    "text_source": block.source,
                    "paragraph_index": block.paragraph_index,
                    "section": block.section,
                    "char_start": block.char_start + span[0],
                    "char_end": block.char_start + span[1],
                },
            }
            missing = result_missing_fields(result, local_subjects)
            range_match = re.match(rf"\s*(?:[-–—]|\bto\b)\s*({NUMBER_RE})", metric_tail)
            if effect_sizes and range_match:
                result["statistic_range"] = [stat_value, normalize_number(range_match.group(1))]
                result["statistic"] = None
                result["effect_sizes"] = []
                missing.append("single_effect_estimate")
            result["needs_human_review"] = bool(missing)
            result["missing_fields"] = missing
            results.append(result)
            if missing:
                review_flags.append({
                    "reason": "partial_statistical_result",
                    "missing_fields": missing,
                    "test_text": result["test_text"],
                    "context": block.text,
                    "location": result["location"],
                })

    if not results and likely_result_context(block.text):
        review_flags.append({
            "reason": "result_context_without_parseable_statistic",
            "missing_fields": ["statistical_test", "effect_size", "p_value_or_ci"],
            "test_text": "",
            "context": block.text,
            "location": {
                "text_source": block.source,
                "paragraph_index": block.paragraph_index,
                "section": block.section,
                "char_start": block.char_start,
                "char_end": block.char_end,
            },
        })

    return results, review_flags


def _labelled_ratio_interval(match: re.Match[str], metric_tail: str) -> list[dict[str, Any]]:
    """Interval for "HR, 95% CI: est, lo to hi", whose CI label precedes the estimate."""
    level = match.groupdict().get("ci_level")
    return parse_unlabelled_interval(metric_tail, int(level)) if level else []


def likely_result_context(text: str) -> bool:
    lower = text.lower()
    return any(
        phrase in lower
        for phrase in (
            "significant", "non-significant", "nonsignificant", "associated with",
            "correlated with", "increased", "decreased", "no difference",
            "main effect", "interaction", "predicted", "odds", "risk",
        )
    )
