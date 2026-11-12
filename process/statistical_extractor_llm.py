"""LLM context review prompt, parsing, and correction application."""
from __future__ import annotations

import json
import re
from typing import Any

from process.statistical_extractor_numbers import (
    normalize_number,
)
from process.statistical_extractor_parsers import (
    repair_decimal_spaced_result_numbers,
    valid_sample_size,
)

# Prompt pieces shared with the bundled prompt (statistical_extractor_llm_bundle).
# They are str.format templates, so literal braces are doubled.
SOURCE_RULES = (
    "Use the supplied source text only. When a complete abstract is supplied, use its "
    "methods and sample information to interpret the target statistic, but do not transfer "
    "a sample size or confidence interval from a different comparison. Do not infer sample "
    "size from degrees of freedom without knowing the study design. Distinguish usable "
    "statistical evidence from sufficient data for calculating a sampling variance."
)

VERDICT_FIELDS = """{{
  "represents_study_data": true/false,
  "usable_for_meta_analysis": true/false,
  "data_role": "primary_result|secondary_result|descriptive_sample|model_statistic|background_or_citation|not_result|unclear",
  "population": "brief population/subject description or empty string",
  "outcome": "brief outcome/measure description or empty string",
  "comparison": "groups/conditions being compared or empty string",
  "effect_direction": "positive|negative|mixed|none|unclear",
  "corrected_fields": {{
    "test_type": "corrected statistical test type or empty string",
    "test_text": "corrected source text span or empty string",
    "statistic": null,
    "effect_sizes": [],
    "sample_sizes": [],
    "confidence_intervals": [],
    "p_values": []
  }},
  "confidence": 0.0,
  "reason": "one concise sentence explaining the judgment",
  "needs_human_review": true/false
}}"""

VERDICT_RULES = """If the deterministic candidate is wrong but the local context contains the correct study-data statistic, fill corrected_fields with the updated test/statistic/p/effect/N/CI values and set usable_for_meta_analysis=true when those corrected values are synthesis-ready.
If the statistic is from background literature, a citation being discussed, a model-fit-only number, or context is insufficient and no corrected study-data statistic is available in the local window, set represents_study_data=false, usable_for_meta_analysis=false, needs_human_review=true, and leave corrected_fields empty.
"""

LLM_CONTEXT_PROMPT = (
    "You are validating one local evidence window from a scientific paper.\n\n"
    "You must decide whether the extracted statistic represents actual study data that can "
    "be used in evidence synthesis or meta-analysis. " + SOURCE_RULES + """

Paper title:
{title}

Deterministic extraction candidate:
{candidate_json}

Local context window:
{context}

Additional source context (complete abstract when available):
{article_context}

Return ONLY a JSON object with these fields:
""" + VERDICT_FIELDS + "\n\n" + VERDICT_RULES
)


def _jsonish(value: Any, limit: int | None = None) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str, indent=2)
    if limit is None or limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "\n..."


def candidate_json(candidate: dict[str, Any]) -> str:
    """The complete deterministic extraction candidate shown to the LLM.

    Request packing is handled against the resolved model context by the
    Stage 2c bundle planner.  Clipping this JSON here would silently remove
    effect sizes or source locations before that planner could split work.
    """
    compact_candidate = {
        "test_type": candidate.get("test_type") or candidate.get("reason"),
        "test_text": candidate.get("test_text", ""),
        "statistic": candidate.get("statistic"),
        "effect_sizes": candidate.get("effect_sizes", []),
        "p_values": candidate.get("p_values", []),
        "confidence_intervals": candidate.get("confidence_intervals", []),
        "sample_sizes": candidate.get("sample_sizes", []),
        "missing_fields": candidate.get("missing_fields", []),
        "location": candidate.get("location", {}),
    }
    return _jsonish(compact_candidate)


def build_llm_context_prompt(
    title: str, candidate: dict[str, Any], context: str, *, article_context: str = "",
) -> str:
    """Build a complete prompt; the context-aware planner owns any splitting."""
    return LLM_CONTEXT_PROMPT.format(
        title=title or "",
        candidate_json=candidate_json(candidate),
        context=context or "",
        article_context=article_context or "Not supplied.",
    )


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return normalize_number(value)
    return None


def _coerce_int(value: Any) -> int | None:
    number = _coerce_float(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def _normalize_corrected_effect_sizes(items: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        effect_type = ""
        text = "LLM corrected"
        if isinstance(item, dict):
            value = _coerce_float(item.get("value"))
            effect_type = str(item.get("type") or item.get("effect_type") or "")
            text = str(item.get("text") or item.get("raw") or text)
        else:
            value = _coerce_float(item)
        if value is not None:
            normalized.append({"type": effect_type, "value": value, "text": text})
    return normalized


def _normalize_corrected_sample_sizes(items: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        label = "sample"
        text = "LLM corrected"
        if isinstance(item, dict):
            n = _coerce_int(item.get("n", item.get("value", item.get("sample_size"))))
            label = str(item.get("label") or item.get("group") or label)
            text = str(item.get("text") or item.get("raw") or text)
        else:
            n = _coerce_int(item)
        if n is not None and valid_sample_size(n):
            normalized.append({"n": n, "label": label, "text": text})
    return normalized


def _normalize_corrected_confidence_intervals(items: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        level = 95
        text = "LLM corrected"
        if isinstance(item, dict):
            lower = _coerce_float(item.get("lower"))
            upper = _coerce_float(item.get("upper"))
            level = _coerce_int(item.get("level")) or level
            text = str(item.get("text") or item.get("raw") or text)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            lower = _coerce_float(item[0])
            upper = _coerce_float(item[1])
        else:
            lower = upper = None
        if lower is not None and upper is not None and lower < upper:
            normalized.append({"level": level, "lower": lower, "upper": upper, "text": text})
    return normalized


def _normalize_corrected_p_values(items: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return normalized
    for item in items:
        operator = "="
        text = "LLM corrected"
        if isinstance(item, dict):
            value = _coerce_float(item.get("value", item.get("p")))
            operator = str(item.get("operator") or item.get("op") or operator)
            text = str(item.get("text") or item.get("raw") or text)
        else:
            value = _coerce_float(item)
        if operator not in {"=", "<", ">", "<=", ">="}:
            operator = "="
        if value is not None and 0 <= value <= 1:
            normalized.append({"operator": operator, "value": value, "text": text})
    return normalized


def normalize_llm_corrected_fields(corrected: dict[str, Any]) -> dict[str, Any]:
    """Normalize model-proposed corrections into the deterministic row shape."""
    statistic = _coerce_float(corrected.get("statistic", corrected.get("statistic_value")))
    return {
        "test_type": str(corrected.get("test_type") or ""),
        "test_text": str(corrected.get("test_text") or corrected.get("text") or ""),
        "statistic": statistic,
        "effect_sizes": _normalize_corrected_effect_sizes(corrected.get("effect_sizes")),
        "sample_sizes": _normalize_corrected_sample_sizes(corrected.get("sample_sizes")),
        "confidence_intervals": _normalize_corrected_confidence_intervals(
            corrected.get("confidence_intervals")
        ),
        "p_values": _normalize_corrected_p_values(corrected.get("p_values")),
    }


def has_llm_corrected_fields(review: Any) -> bool:
    if not isinstance(review, dict):
        return False
    corrected = review.get("corrected_fields")
    if not isinstance(corrected, dict):
        return False
    return (
        corrected.get("statistic") is not None
        or bool(corrected.get("test_text"))
        or any(
            corrected.get(key)
            for key in ("effect_sizes", "sample_sizes", "confidence_intervals", "p_values")
        )
    )


def apply_llm_corrected_fields(result: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of an extracted result with LLM-proposed fields applied."""
    repaired = repair_decimal_spaced_result_numbers(result)
    review = repaired.get("llm_context_review")
    if not has_llm_corrected_fields(review):
        return repaired

    corrected_fields = review.get("corrected_fields", {}) if isinstance(review, dict) else {}
    corrected_fields = (
        normalize_llm_corrected_fields(corrected_fields)
        if isinstance(corrected_fields, dict)
        else normalize_llm_corrected_fields({})
    )
    corrected = dict(repaired)
    if corrected_fields.get("test_type"):
        corrected["test_type"] = corrected_fields["test_type"]
    if corrected_fields.get("test_text"):
        corrected["test_text"] = corrected_fields["test_text"]
    if corrected_fields.get("statistic") is not None:
        corrected["statistic"] = corrected_fields["statistic"]
    for key in ("effect_sizes", "sample_sizes", "confidence_intervals", "p_values"):
        if corrected_fields.get(key):
            corrected[key] = corrected_fields[key]
    corrected["llm_correction_applied"] = True
    return corrected


def reply_json_candidates(text: str) -> list[str]:
    """Texts that may hold the reply's JSON: the reply without code fences, then
    its outermost ``{...}`` span (for replies wrapped in prose)."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    return candidates


def _parse_review_json(text: str) -> dict[str, Any] | None:
    """Return the JSON object in an LLM reply, or None when there is none."""
    for candidate in reply_json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_bool(value: Any) -> bool | None:
    """Strict boolean: JSON booleans or "true"/"false" strings; anything else is None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return {"true": True, "false": False}.get(value.strip().lower())
    return None


def review_verdict(data: dict[str, Any]) -> tuple[bool, bool] | None:
    """Return (represents, usable), or None when the reply gives no verdict.

    An explicit ``represents_study_data: false`` is a rejection on its own; a
    positive answer also needs an explicit usability decision.
    """
    represents = _parse_bool(data.get("represents_study_data"))
    if represents is None:
        return None
    usable = _parse_bool(data.get("usable_for_meta_analysis"))
    if usable is None:
        return None if represents else (False, False)
    return represents, usable


def parse_llm_context_response(text: str) -> dict[str, Any]:
    """Parse and normalize the context-validation JSON returned by an LLM.

    An empty, refused, truncated or verdict-less reply is a failed check (no
    verdict, statistic kept) rather than a rejection.
    """
    if not (text or "").strip():
        return failed_review("empty LLM response")
    data = _parse_review_json(text)
    if data is None:
        return failed_review("unparseable LLM response")
    if review_verdict(data) is None:
        return failed_review("LLM response has no valid verdict")
    return review_from_data(data)


def review_from_data(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize one parsed verdict object; ``review_verdict(data)`` must not be None."""
    represents, usable = review_verdict(data)

    corrected = data.get("corrected_fields")
    if not isinstance(corrected, dict):
        corrected = {}
    corrected = normalize_llm_corrected_fields(corrected)

    confidence = data.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0

    needs_review = _parse_bool(data.get("needs_human_review"))
    if needs_review is None:
        needs_review = not (represents and usable)

    return {
        "represents_study_data": represents,
        "usable_for_meta_analysis": usable,
        "data_role": str(data.get("data_role") or "unclear"),
        "population": str(data.get("population") or ""),
        "outcome": str(data.get("outcome") or ""),
        "comparison": str(data.get("comparison") or ""),
        "effect_direction": str(data.get("effect_direction") or "unclear"),
        "corrected_fields": corrected,
        "confidence": confidence,
        "reason": str(data.get("reason") or "No parseable LLM reason returned."),
        "needs_human_review": needs_review,
    }


def _review_target_groups(records, include_human_flags):
    """Group redundant flags with their evidence, within each article only.

    A partial-result flag describes the result at the same source span. The two
    no-statistic flag types can also describe exactly the same passage. Keep
    distinct statistics and different passages as separate model decisions.
    """
    groups = []
    for record in records:
        seen = {}
        result_spans = {}
        for kind, candidates in (
            ("result", record.get("results", []) or []),
            ("flag", (record.get("human_review", []) or []) if include_human_flags else []),
        ):
            for candidate in candidates:
                context = str(candidate.get("context") or "")
                if not context.strip():
                    continue
                location = candidate.get("location") or {}
                span = (json.dumps(location, sort_keys=True), candidate.get("test_text", ""))
                group = None
                if kind == "flag" and candidate.get("reason") == "partial_statistical_result" and location:
                    group = result_spans.get(span)
                if group is None:
                    if kind == "flag" and candidate.get("reason") in {
                        "no_statistical_result_found", "result_context_without_parseable_statistic",
                    }:
                        key = ("unparsed", span, context)
                    else:
                        content = {k: v for k, v in candidate.items() if k != "llm_context_review"}
                        key = (kind, json.dumps(content, sort_keys=True))
                    group = seen.get(key)
                    if group is None:
                        group = []
                        seen[key] = group
                        groups.append(group)
                group.append((record, candidate, context))
                if kind == "result" and location:
                    result_spans.setdefault(span, group)
    return groups


def _review_targets(records: list[dict[str, Any]], include_human_flags: bool) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    return [group[0] for group in _review_target_groups(records, include_human_flags)]


def review_location(candidate: dict[str, Any]) -> str:
    location = candidate.get("location") or {}
    source = location.get("text_source") or "text"
    paragraph = location.get("paragraph_index")
    passage = f", passage {paragraph + 1}" if isinstance(paragraph, int) and paragraph >= 0 else ""
    statistic = candidate.get("test_text") or "no parsed statistic"
    return f"{source}{passage}; {statistic}"


_FAILED_REASON_PREFIX = "LLM context check failed: "


def is_failed_llm_review(review: Any) -> bool:
    """True when the LLM never answered (error or timeout), so there is no verdict.

    Also recognises failures saved before ``check_failed`` existed, which were
    stored as a rejection with an ``error`` field and the failure reason prefix.
    """
    if not isinstance(review, dict):
        return False
    if review.get("check_failed"):
        return True
    return "error" in review and str(review.get("reason", "")).startswith(_FAILED_REASON_PREFIX)


def as_failed_review(review: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a failed review in the current no-verdict form."""
    return {
        **review,
        "check_failed": True,
        "represents_study_data": None,
        "usable_for_meta_analysis": None,
    }


def failed_review(reason: str) -> dict[str, Any]:
    # No verdict: the statistic is treated as if the check had not run (kept for
    # meta-analysis) and listed for human review with the error.
    return {
        "check_failed": True,
        "represents_study_data": None,
        "usable_for_meta_analysis": None,
        "data_role": "unclear",
        "population": "",
        "outcome": "",
        "comparison": "",
        "effect_direction": "unclear",
        "corrected_fields": normalize_llm_corrected_fields({}),
        "confidence": 0.0,
        "reason": f"{_FAILED_REASON_PREFIX}{reason}",
        "needs_human_review": True,
        "error": reason,
    }
