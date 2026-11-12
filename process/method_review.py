"""Study design classification and span validation."""
import logging
import math
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Prompt order; ALLOWED_METHOD_TYPES is the same set for membership checks.
METHOD_TYPES: tuple[str, ...] = (
    "randomized_controlled_trial",
    "cohort_observational",
    "case_control",
    "cross_sectional",
    "longitudinal",
    "systematic_review_meta_analysis",
    "narrative_review",
    "case_report_case_series",
    "preclinical_animal",
    "in_vitro_cellular",
    "imaging_ml_classifier",
    "methods_protocol",
    "genetic_association",
    "mixed_methods",
    "unknown",
)
ALLOWED_METHOD_TYPES = set(METHOD_TYPES)

# A span that is not found verbatim is accepted only when a leading run of at
# least this share of its words appears in the abstract (truncated quotes).
PARTIAL_SPAN_MIN_COVERAGE = 0.8
_MIN_PARTIAL_SPAN_WORDS = 4

# LLMs sometimes report confidence on a 0-100 scale; such values are rescaled.
_PERCENT_CONFIDENCE_MAX = 100.0

_TRUE_STRINGS = {"true", "yes", "y", "1"}
_FALSE_STRINGS = {"false", "no", "n", "0", "none", ""}


@dataclass
class SpanValidation:
    field: str
    quote: str
    start_char: int = -1
    end_char: int = -1
    valid: bool = False
    reason: str = ""


def extraction_prompt(title: str, abstract: str) -> str:
    """Build the LLM prompt for study design classification."""
    return f"""Classify the study design and extract method details from this paper.

Title: {title}
Abstract: {abstract}

Respond with ONLY a JSON object:
{{
    "study_method_type": "<one of: {", ".join(METHOD_TYPES)}>",
    "confidence": <0.0-1.0>,
    "sample_size": <integer or -1 if unknown>,
    "has_control_group": <true/false>,
    "is_longitudinal": <true/false>,
    "design_notes": "<brief note about the design>",
    "evidence_spans": ["<key quote 1>", "<key quote 2>"]
}}"""


def parse_json_object(text: str) -> dict | None:
    """Parse JSON from LLM response, stripping markdown fences if needed."""
    from process.llm import parse_llm_json

    result = parse_llm_json(text)
    if result is None:
        logger.debug("Could not parse JSON from LLM response: %s",
                     (text or "")[:200])
    return result


def _matched_prefix_words(words: list[str], norm_abstract: str) -> int:
    """Length of the longest leading run of ``words`` found in the abstract,
    or 0 when it covers less than PARTIAL_SPAN_MIN_COVERAGE of the span."""
    if len(words) < _MIN_PARTIAL_SPAN_WORDS:
        return 0
    min_words = math.ceil(len(words) * PARTIAL_SPAN_MIN_COVERAGE)
    for end in range(len(words), min_words - 1, -1):
        if " ".join(words[:end]) in norm_abstract:
            return end
    return 0


def validate_spans(abstract: str, spans: list) -> list[SpanValidation]:
    """Check each quote span is a real substring of the abstract.

    For each span: try exact match, then case-insensitive, then fuzzy.
    Record start_char/end_char if found.
    """
    results = []
    if not abstract or not spans:
        return results

    abstract_lower = abstract.lower()

    for span_text in spans:
        if not isinstance(span_text, str) or not span_text.strip():
            continue

        sv = SpanValidation(field="evidence_span", quote=span_text)

        # 1. Exact match
        idx = abstract.find(span_text)
        if idx >= 0:
            sv.start_char = idx
            sv.end_char = idx + len(span_text)
            sv.valid = True
            sv.reason = "exact_match"
            results.append(sv)
            continue

        # 2. Case-insensitive match
        span_lower = span_text.lower()
        idx = abstract_lower.find(span_lower)
        if idx >= 0:
            sv.start_char = idx
            sv.end_char = idx + len(span_text)
            sv.valid = True
            sv.reason = "case_insensitive_match"
            results.append(sv)
            continue

        # 3. Fuzzy match — normalize whitespace and try again. Offsets stay
        # -1 because normalized positions do not map back to the original.
        norm_abstract = re.sub(r"\s+", " ", abstract_lower)
        norm_span = re.sub(r"\s+", " ", span_lower.strip())
        if norm_span in norm_abstract:
            sv.valid = True
            sv.reason = "fuzzy_whitespace_match"
            results.append(sv)
            continue

        # 4. A truncated quote: a long leading part of the span is verbatim.
        matched = _matched_prefix_words(norm_span.split(), norm_abstract)
        if matched:
            sv.valid = True
            sv.reason = f"partial_match ({matched}/{len(norm_span.split())} words)"

        if not sv.valid:
            sv.reason = "not_found"

        results.append(sv)

    return results


def validate_method_profile(
    abstract: str,
    parsed: dict | None,
    confidence_threshold: float = 0.5,
) -> dict:
    """Normalize parsed LLM output: enforce allowed method types, validate spans.

    Args:
        abstract: The paper abstract used for span validation.
        parsed: Raw LLM output dict, or None if the LLM returned nothing.
        confidence_threshold: Minimum confidence below which a record is flagged
            for manual review.  Defaults to 0.5; callers should pass the value
            from ``Config.get_method_review_config().get("confidence_threshold")``
            so the threshold stays consistent with ``research_config.yaml``.
    """
    default = {
        "study_method_type": "unknown",
        "confidence": 0.0,
        "sample_size": -1,
        "has_control_group": False,
        "is_longitudinal": False,
        "design_notes": "",
        "evidence_spans": [],
        "span_validations": [],
        "needs_manual_review": True,
        "manual_review_reason": "no_llm_response",
    }

    if parsed is None:
        return default

    profile = dict(default)
    issues: list[str] = []
    profile["study_method_type"] = _method_type(parsed.get("study_method_type", "unknown"))
    profile["confidence"], confidence_issue = _coerce_confidence(parsed.get("confidence", 0.0))
    if confidence_issue:
        issues.append(confidence_issue)
    profile["sample_size"] = _coerce_sample_size(parsed.get("sample_size", -1))
    for name in ("has_control_group", "is_longitudinal"):
        value = _coerce_bool(parsed.get(name, False))
        if value is None:
            issues.append(f"unparsed_{name}")
        profile[name] = bool(value)
    profile["design_notes"] = str(parsed.get("design_notes", ""))

    raw_spans = parsed.get("evidence_spans", [])
    profile["evidence_spans"] = ([s for s in raw_spans if isinstance(s, str)]
                                 if isinstance(raw_spans, list) else [])
    if abstract and profile["evidence_spans"]:
        profile["span_validations"] = [
            {"quote": sv.quote, "start_char": sv.start_char, "end_char": sv.end_char,
             "valid": sv.valid, "reason": sv.reason}
            for sv in validate_spans(abstract, profile["evidence_spans"])
        ]

    reasons = _review_reasons(profile, confidence_threshold) + issues
    profile["needs_manual_review"] = bool(reasons)
    profile["manual_review_reason"] = "; ".join(reasons)
    return profile


def _method_type(value) -> str:
    raw_type = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if raw_type in ALLOWED_METHOD_TYPES:
        return raw_type
    logger.debug("Unrecognized method type '%s', defaulting to unknown", raw_type)
    return "unknown"


def _coerce_confidence(value) -> tuple[float, str | None]:
    """Confidence in [0, 1] plus an issue label when the value was not usable as is.

    Non-numeric, non-finite and out-of-range values become 0.0; values on a
    0-100 scale are rescaled. Either way the profile is sent to review.
    """
    if isinstance(value, bool):
        return 0.0, "invalid_confidence"
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0, "invalid_confidence"
    if not math.isfinite(confidence) or not 0 <= confidence <= _PERCENT_CONFIDENCE_MAX:
        return 0.0, "invalid_confidence"
    if confidence > 1:
        return confidence / _PERCENT_CONFIDENCE_MAX, "confidence_rescaled"
    return confidence, None


def _coerce_sample_size(value) -> int:
    """A non-negative integer sample size, or -1 when unknown or not an integer."""
    if isinstance(value, bool):
        return -1
    if isinstance(value, float) and value.is_integer():  # False for inf and NaN
        value = int(value)
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        value = int(value.strip())
    return value if isinstance(value, int) and value >= 0 else -1


def _coerce_bool(value) -> bool | None:
    """Parse an LLM boolean ("false" is False); None when it cannot be read."""
    if value is None or isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
    return None


def _review_reasons(profile: dict, confidence_threshold: float) -> list[str]:
    reasons = []
    if profile["confidence"] < confidence_threshold:
        reasons.append(f"low_confidence ({profile['confidence']:.2f})")
    if profile["study_method_type"] == "unknown":
        reasons.append("unknown_type")
    n_invalid = sum(1 for sv in profile["span_validations"] if not sv["valid"])
    if n_invalid:
        reasons.append(f"invalid_spans ({n_invalid}/{len(profile['span_validations'])})")
    return reasons


def summarize_method_types(records: list[dict]) -> dict:
    """Produce summary: count by method type, % flagged, total."""
    type_counts: dict[str, int] = {}
    flagged = 0
    total = len(records)

    for rec in records:
        mp = rec.get("method_profile", {})
        mtype = mp.get("study_method_type", "unknown")
        type_counts[mtype] = type_counts.get(mtype, 0) + 1
        if mp.get("needs_manual_review", False):
            flagged += 1

    return {
        "total_papers": total,
        "flagged_for_review": flagged,
        "flagged_pct": round(flagged / total * 100, 1) if total else 0.0,
        "type_counts": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
    }
