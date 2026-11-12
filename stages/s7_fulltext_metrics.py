"""Stage 7 Phase 4: quantitative metric extraction via regex.

Split out of ``stages.s7_fulltext``, which re-exports every name here.
"""
import logging
import re

from process.statistical_extractor_numbers import normalize_number, parse_count

# Shares the Stage 7 logger so log records keep their original name.
logger = logging.getLogger("stages.s7_fulltext")

# ---------------------------------------------------------------------------
# Phase 4: Metric extraction via regex
# ---------------------------------------------------------------------------
# Single-character substitutions keep match offsets aligned with the text:
# Unicode minus and dashes become "-", Lancet-style middle dots become ".".
_METRIC_TEXT_CHARS = str.maketrans({
    "\u2212": "-", "\u2010": "-", "\u2011": "-", "\u2012": "-",
    "\u2013": "-", "\u2014": "-", "\u00b7": ".",
})
_SIGNED_NUM = r"-?(?:\d+(?:\.\d+)?|\.\d+)"
_DECIMAL = r"-?\d*\.\d{1,4}(?!\d)"
# The look-behinds give d/g/N/p a word boundary, so "SD", "WMD", "and", "fold",
# "mean" and "median" are not read as effect sizes or sample sizes.
_D_G_RE = re.compile(rf"(?<![\w.])([dg])\s*=\s*({_DECIMAL})", re.IGNORECASE)
_ETA_SQUARED_RE = re.compile(
    r"(?:partial\s+)?(?:\u03b7\u00b2|\u03b72|\u03b7-squared|eta[- ]?squared)"
    rf"\s*=\s*({_DECIMAL})",
    re.IGNORECASE,
)
_RATIO_RE = re.compile(r"\b(OR|RR)\s*=\s*(\d+\.\d{1,4})")
_N_EQUALS_RE = re.compile(
    r"(?<![\w.])[Nn]\s*=\s*(\d{1,3}(?:,\d{3})+|\d{2,6})(?![\d,.]?\d)"
)
_TOTAL_N_RE = re.compile(
    r"(?:total\s+(?:of\s+)?|included\s+|enrolled\s+)"
    r"(\d{1,3}(?:,\d{3})*)\s+"
    r"(?:participants|patients|subjects|individuals)",
    re.IGNORECASE,
)
# "95% CI [0.21, 0.68]", "95% CI: 0.15-0.55", "CI 0.3 to 0.8", "[-0.52, -0.11]"
_CI_RE = re.compile(
    r"(\d{2,3})%?\s*CI\s*[:=]?\s*[\[\(]?\s*"
    rf"({_SIGNED_NUM})(?:\s*(?:[,;]|-|to\b)\s*|\s+)({_SIGNED_NUM})\s*[\]\)]?",
    re.IGNORECASE,
)
# "p < 0.001", "p<.05", "P = .001", "p = 1e-5", "p = 1.2 x 10-5"
_P_VALUE_RE = re.compile(
    r"(?<![\w.])p\s*([<>=\u2264\u2265])\s*"
    r"((?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+|\s*[\u00d7xX]\s*10\^?[-+]?\d+)?)",
    re.IGNORECASE,
)
_CI_LEVELS = (90, 95, 99)
_MAX_ABS_STANDARDISED_EFFECT = 5.0
_RATIO_RANGE = (0.01, 100.0)
_SAMPLE_SIZE_RANGE = (10, 999999)


def _append_unique(rows: list[dict], row: dict, *identity: str) -> None:
    """Append ``row`` unless one with the same ``identity`` fields is present."""
    key = tuple(row[name] for name in identity)
    if not any(tuple(existing[name] for name in identity) == key for existing in rows):
        rows.append(row)


def _extract_effect_sizes(text: str) -> list[dict]:
    """Cohen's d / Hedges' g, eta-squared and OR/RR, each value listed once."""
    effects: list[dict] = []

    def _add(kind: str, raw: str, low: float, high: float, pos: int) -> None:
        value = normalize_number(raw)
        if value is not None and low <= value <= high:
            _append_unique(effects, {
                "type": kind, "value": value, "context": _get_context(text, pos, 40),
            }, "type", "value")

    bound = _MAX_ABS_STANDARDISED_EFFECT
    for m in _D_G_RE.finditer(text):
        _add(m.group(1).lower(), m.group(2), -bound, bound, m.start())
    for m in _ETA_SQUARED_RE.finditer(text):
        _add("eta_squared", m.group(1), 0.0, 1.0, m.start())
    for m in _RATIO_RE.finditer(text):
        _add(m.group(1).lower(), m.group(2), *_RATIO_RANGE, m.start())
    return effects


def _extract_sample_sizes(text: str) -> list[dict]:
    """``N = 150`` / ``n = 1,234`` and "total of 1,234 participants", deduplicated."""
    sizes: list[dict] = []
    low, high = _SAMPLE_SIZE_RANGE
    for pattern in (_N_EQUALS_RE, _TOTAL_N_RE):
        for m in pattern.finditer(text):
            value = parse_count(m.group(1))
            if value is not None and low <= value <= high:
                _append_unique(sizes, {
                    "value": value, "context": _get_context(text, m.start(), 40),
                }, "value")
    return sizes


def _extract_confidence_intervals(text: str) -> list[dict]:
    """90/95/99% CIs with comma, dash, "to" or space separators."""
    intervals: list[dict] = []
    for m in _CI_RE.finditer(text):
        level = int(m.group(1))
        low, high = normalize_number(m.group(2)), normalize_number(m.group(3))
        if level in _CI_LEVELS and low is not None and high is not None and low < high:
            _append_unique(intervals, {
                "level": level, "lower": low, "upper": high,
                "context": _get_context(text, m.start(), 50),
            }, "level", "lower", "upper")
    return intervals


def _extract_p_values(text: str) -> list[dict]:
    """p-values in (0, 1], including APA ``.001`` and scientific notation."""
    p_values: list[dict] = []
    for m in _P_VALUE_RE.finditer(text):
        value = normalize_number(m.group(2))
        if value is not None and 0.0 < value <= 1.0:
            p_values.append({
                "operator": m.group(1),
                "value": value,
                "context": _get_context(text, m.start(), 40),
            })
    return p_values


def extract_metrics_from_text(text: str, paper_id: str = "") -> dict:
    """Extract quantitative metrics using regex.

    Returns dict with effect_sizes, sample_sizes, confidence_intervals, p_values.
    """
    text = (text or "").translate(_METRIC_TEXT_CHARS)
    metrics = {
        "effect_sizes": _extract_effect_sizes(text) if text else [],
        "sample_sizes": _extract_sample_sizes(text) if text else [],
        "confidence_intervals": _extract_confidence_intervals(text) if text else [],
        "p_values": _extract_p_values(text) if text else [],
    }
    if any(metrics.values()):
        logger.debug(
            "  Metrics for %s: %d effect sizes, %d sample sizes, "
            "%d CIs, %d p-values",
            paper_id, len(metrics["effect_sizes"]), len(metrics["sample_sizes"]),
            len(metrics["confidence_intervals"]), len(metrics["p_values"]),
        )
    return metrics


def _get_context(text: str, pos: int, window: int = 40) -> str:
    """Extract a context snippet around a match position."""
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    snippet = text[start:end].replace("\n", " ").strip()
    return snippet
