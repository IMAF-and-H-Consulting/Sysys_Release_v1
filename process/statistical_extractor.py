"""Programmatic statistical-result extraction for post-score review.

Compatibility facade for the statistical_extractor_* implementation modules.
Keep these exports available for stage runners and other existing callers;
optional LLM context reviews live in statistical_extractor_llm.
"""
from __future__ import annotations

from process.statistical_extractor_block import (
    extract_from_block,
    likely_result_context,
)
from process.statistical_extractor_llm import (
    LLM_CONTEXT_PROMPT,
    _coerce_float,
    _coerce_int,
    _jsonish,
    _normalize_corrected_confidence_intervals,
    _normalize_corrected_effect_sizes,
    _normalize_corrected_p_values,
    _normalize_corrected_sample_sizes,
    _review_targets,
    apply_llm_corrected_fields,
    build_llm_context_prompt,
    has_llm_corrected_fields,
    is_failed_llm_review,
    normalize_llm_corrected_fields,
    parse_llm_context_response,
)
from process.statistical_extractor_llm_run import apply_llm_context_reviews
from process.statistical_extractor_numbers import (
    _closest_context_match,
    _has_spaced_decimal,
    _is_incomplete_decimal_tail,
    normalize_number,
)
from process.statistical_extractor_parsers import (
    _p_value_matches,
    _without_internal_spans,
    infer_subject_terms,
    nearby_text,
    parse_comparisons,
    parse_confidence_intervals,
    parse_effect_sizes,
    parse_p_values,
    parse_p_values_near,
    parse_subjects,
    repair_decimal_spaced_result_numbers,
    result_missing_fields,
    valid_sample_size,
)
from process.statistical_extractor_patterns import (
    AGE_RE,
    CI_RE,
    COMPARISON_RE,
    N_EQUALS_RE,
    NUMBER_RE,
    P_VALUE_RE,
    SAMPLE_RE,
    SUBJECT_WORDS,
    TEST_PATTERNS,
    TextBlock,
)
from process.statistical_extractor_pipeline import (
    collect_human_review,
    dedupe_subjects,
    extract_article,
    extract_statistical_evidence,
    metrics_for_meta_analysis,
    summarize,
)
from process.statistical_extractor_text import (
    article_id,
    paragraph_blocks,
    sentence_window_blocks,
    text_candidates,
)

# Explicit exports preserve the compatibility API during unused-import cleanup.
__all__ = [
    "AGE_RE",
    "CI_RE",
    "COMPARISON_RE",
    "LLM_CONTEXT_PROMPT",
    "NUMBER_RE",
    "N_EQUALS_RE",
    "P_VALUE_RE",
    "SAMPLE_RE",
    "SUBJECT_WORDS",
    "TEST_PATTERNS",
    "TextBlock",
    "_closest_context_match",
    "_coerce_float",
    "_coerce_int",
    "_has_spaced_decimal",
    "_is_incomplete_decimal_tail",
    "_jsonish",
    "_normalize_corrected_confidence_intervals",
    "_normalize_corrected_effect_sizes",
    "_normalize_corrected_p_values",
    "_normalize_corrected_sample_sizes",
    "_p_value_matches",
    "_review_targets",
    "_without_internal_spans",
    "apply_llm_context_reviews",
    "apply_llm_corrected_fields",
    "article_id",
    "build_llm_context_prompt",
    "collect_human_review",
    "dedupe_subjects",
    "extract_article",
    "extract_from_block",
    "extract_statistical_evidence",
    "has_llm_corrected_fields",
    "infer_subject_terms",
    "is_failed_llm_review",
    "likely_result_context",
    "metrics_for_meta_analysis",
    "nearby_text",
    "normalize_llm_corrected_fields",
    "normalize_number",
    "paragraph_blocks",
    "parse_comparisons",
    "parse_confidence_intervals",
    "parse_effect_sizes",
    "parse_llm_context_response",
    "parse_p_values",
    "parse_p_values_near",
    "parse_subjects",
    "repair_decimal_spaced_result_numbers",
    "result_missing_fields",
    "sentence_window_blocks",
    "summarize",
    "text_candidates",
    "valid_sample_size",
]
