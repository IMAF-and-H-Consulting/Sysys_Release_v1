"""Extract and check abstract statistics (κ, CI, p, r) for Stage 9 verification.

Each metric has its own tolerance: a single loose tolerance would let any
r, p or CI in [-1, 1] pass. CIs, p-values and correlations are compared with a
Stage 8 robustness statistic only when the wording around the claim names that
analysis; otherwise they are reported as unverifiable rather than checked
against an unrelated number.
"""
from __future__ import annotations

import logging
import operator
import re

logger = logging.getLogger(__name__)

# ── Tolerances ──────────────────────────────────────────────────────────
# Absolute tolerance per metric. The claim's own rounding (half a unit in its
# last reported decimal place) is also allowed, so "r = 0.8" matches 0.83.
COUNT_ABS_TOL = 0.0
PERCENT_ABS_TOL_PP = 0.5
KAPPA_ABS_TOL = 0.01
CORRELATION_ABS_TOL = 0.01
P_VALUE_ABS_TOL = 0.001
CI_ABS_TOL = 0.01
FLOAT_EPS = 1e-9

METRIC_TOLERANCES = {
    "count": COUNT_ABS_TOL,
    "percentage": PERCENT_ABS_TOL_PP,
    "kappa": KAPPA_ABS_TOL,
    "correlation": CORRELATION_ABS_TOL,
    "p_value": P_VALUE_ABS_TOL,
    "ci": CI_ABS_TOL,
}


def number_decimals(number_text: str) -> int:
    """Number of digits after the decimal point in a reported number."""
    return len(number_text.split(".", 1)[1]) if "." in number_text else 0


def _rounding_tolerance(decimals: int) -> float:
    return 0.5 * 10 ** -decimals if decimals > 0 else 0.0


def within_tolerance(claimed: float, actual: float, metric: str,
                     decimals: int = 0) -> bool:
    """True when ``actual`` matches ``claimed`` within the metric's tolerance."""
    tolerance = max(METRIC_TOLERANCES[metric], _rounding_tolerance(decimals))
    return abs(actual - claimed) <= tolerance + FLOAT_EPS


# ── Extraction ──────────────────────────────────────────────────────────

_NUMBER = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
_UNICODE_MINUS = "−"

_KAPPA_RE = re.compile(
    rf"(?:κ|\bkappa\b)\s*(?:[:=]|\bof\b|\bwas\b|\bis\b)?\s*(?P<value>{_NUMBER})",
    re.IGNORECASE,
)
_CI_RE = re.compile(
    rf"\bCI\b\s*[:=]?\s*[\[(]?\s*(?P<lower>{_NUMBER})"
    rf"\s*(?:,|;|–|—|-|\bto\b)\s*(?P<upper>{_NUMBER})\s*[\])]?",
    re.IGNORECASE,
)
_P_VALUE_RE = re.compile(
    rf"\bp(?:[\s-]*values?)?\s*(?P<op><=|>=|[<>=≤≥])\s*(?P<value>{_NUMBER})",
    re.IGNORECASE,
)
_CORRELATION_RE = re.compile(
    rf"\br(?:\s*\(\s*\d+\s*\))?\s*=\s*(?P<value>{_NUMBER})",
    re.IGNORECASE,
)
_OPERATOR_ALIASES = {"<=": "≤", ">=": "≥"}


def _normalise(text: str) -> str:
    # Same-length replacement keeps match spans valid for the original text.
    return text.replace(_UNICODE_MINUS, "-")


def _scalar_item(match: re.Match, text: str) -> dict:
    value = match.group("value")
    return {"value": float(value), "decimals": number_decimals(value),
            "raw": text[match.start():match.end()], "span": match.span()}


def extract_kappas(text: str) -> list[dict]:
    """"κ = 0.72", "Cohen's kappa of 0.7", "kappa: .61"."""
    return [_scalar_item(m, text) for m in _KAPPA_RE.finditer(_normalise(text))]


def extract_cis(text: str) -> list[dict]:
    """"CI [0.55, 0.89]", "95% CI: 0.55-0.89", "CI [−0.20, 0.40]", "CI 0.1 to 0.3"."""
    items = []
    for m in _CI_RE.finditer(_normalise(text)):
        lower, upper = m.group("lower"), m.group("upper")
        items.append({
            "lower": float(lower), "upper": float(upper),
            "decimals": [number_decimals(lower), number_decimals(upper)],
            "raw": text[m.start():m.end()], "span": m.span(),
        })
    return items


def extract_p_values(text: str) -> list[dict]:
    """"p < 0.001", "p<.05", "p ≥ 0.1", "p-value = 0.03" (not "group = 2")."""
    items = []
    for m in _P_VALUE_RE.finditer(_normalise(text)):
        item = _scalar_item(m, text)
        op = m.group("op")
        item["operator"] = _OPERATOR_ALIASES.get(op, op)
        items.append(item)
    return items


def extract_correlations(text: str) -> list[dict]:
    """"r = 0.83", "R = 0.5", "r(48) = .45", "r = −0.3"."""
    return [_scalar_item(m, text) for m in _CORRELATION_RE.finditer(_normalise(text))]


# ── Referent resolution ─────────────────────────────────────────────────

# Stage 8 analyses an abstract statistic may refer to: (label, wording).
_REFERENTS = {
    "bootstrap": ("bootstrap CI of the mean relevance score",
                  re.compile(r"\bbootstrap|\bmean\s+relevance", re.IGNORECASE)),
    "permutation": ("permutation test of the condition-support gradient",
                    re.compile(r"\bpermutation|\bcondition[\s\-–—]+support\s+gradient",
                               re.IGNORECASE)),
    "split_half": ("split-half reliability",
                   re.compile(r"\bsplit[\s\-–]*half|\bspearman[\s\-–]*brown",
                              re.IGNORECASE)),
}
# Which analyses can supply each statistic, and the robustness keys to read.
_REFERENCE_KEYS = {
    "ci": {"bootstrap": ("ci_lower", "ci_upper")},
    "p_value": {"permutation": ("p_value",)},
    "correlation": {"permutation": ("observed_correlation",),
                    "split_half": ("mean_correlation",)},
}
_CLAUSE_END = re.compile(
    r"[.!?;](?=\s|$)|\n|\b(?:while|whereas|whilst|but|although)\b", re.IGNORECASE)


def clause_around(text: str, span: tuple[int, int]) -> str:
    """The sentence/clause containing ``span``; decimal points do not split it."""
    start, end = span
    left = max((m.end() for m in _CLAUSE_END.finditer(text, 0, start)), default=0)
    right = _CLAUSE_END.search(text, end)
    return text[left:right.start() if right else len(text)]


def _reference_values(robustness: dict, name: str, keys: tuple[str, ...]) -> list[float] | None:
    section = robustness.get(name) or {}
    if section.get("skipped"):
        return None
    if name == "split_half" and not section.get("estimable", False):
        # A non-estimable split-half stores 0.0 as a placeholder.
        return None
    values = [section.get(key) for key in keys]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return None
    return [float(v) for v in values]


def resolve_referent(text: str, span: tuple[int, int], stat: str,
                     robustness: dict | None) -> tuple[list[float] | None, str]:
    """Return ``(reference values, label)`` or ``(None, unverifiable note)``."""
    clause = clause_around(text, span)
    candidates = [name for name in _REFERENCE_KEYS[stat]
                  if _REFERENTS[name][1].search(clause)]
    if not candidates:
        return None, ("unverifiable: wording does not identify a Stage 8 analysis "
                      "(bootstrap CI, permutation test, split-half); not compared "
                      "with unrelated robustness statistics")
    if len(candidates) > 1:
        labels = ", ".join(_REFERENTS[name][0] for name in candidates)
        return None, f"unverifiable: wording names several analyses ({labels})"
    name = candidates[0]
    label = _REFERENTS[name][0]
    if not robustness:
        return None, f"unverifiable: {label} — no robustness results available"
    values = _reference_values(robustness, name, _REFERENCE_KEYS[stat][name])
    if values is None:
        return None, f"unverifiable: {label} not available in robustness results"
    return values, label


# ── Checks ──────────────────────────────────────────────────────────────

_P_COMPARATORS = {"<": operator.lt, "≤": operator.le,
                  ">": operator.gt, "≥": operator.ge}


def _warn(item: dict, expected, note: str) -> dict:
    logger.info("CHECK %s → WARN (%s)", item["raw"], note)
    return {"claim": item["raw"], "expected": expected, "actual": None,
            "status": "WARN", "note": note}


def _result(item: dict, expected, actual, ok: bool, note: str) -> dict:
    status = "PASS" if ok else "FAIL"
    logger.info("CHECK %s: actual=%s → %s", item["raw"], actual, status)
    return {"claim": item["raw"], "expected": expected, "actual": actual,
            "status": status, "note": note}


def check_ci(item: dict, text: str, robustness: dict | None) -> dict:
    expected = [item["lower"], item["upper"]]
    values, label = resolve_referent(text, item["span"], "ci", robustness)
    if values is None:
        return _warn(item, expected, label)
    lo_dec, hi_dec = item.get("decimals", [0, 0])
    ok_lo = within_tolerance(item["lower"], values[0], "ci", lo_dec)
    ok_hi = within_tolerance(item["upper"], values[1], "ci", hi_dec)
    note = (f"{label}: within tolerance" if ok_lo and ok_hi
            else f"{label}: lo_ok={ok_lo} hi_ok={ok_hi}")
    return _result(item, expected, [round(v, 4) for v in values], ok_lo and ok_hi, note)


def check_p_value(item: dict, text: str, robustness: dict | None) -> dict:
    claimed, op = item["value"], item.get("operator", "=")
    if not 0.0 <= claimed <= 1.0:
        return _warn(item, claimed, "not a valid p-value (outside [0, 1]); not checked")
    values, label = resolve_referent(text, item["span"], "p_value", robustness)
    if values is None:
        return _warn(item, claimed, label)
    actual = values[0]
    if op in _P_COMPARATORS:
        ok = _P_COMPARATORS[op](actual, claimed)
    else:
        ok = within_tolerance(claimed, actual, "p_value", item.get("decimals", 0))
    note = (f"{label}: consistent" if ok
            else f"{label}: actual p={actual:.6f} does not satisfy p {op} {claimed}")
    return _result(item, claimed, round(actual, 6), ok, note)


def check_correlation(item: dict, text: str, robustness: dict | None) -> dict:
    claimed = item["value"]
    values, label = resolve_referent(text, item["span"], "correlation", robustness)
    if values is None:
        return _warn(item, claimed, label)
    actual = values[0]
    ok = within_tolerance(claimed, actual, "correlation", item.get("decimals", 0))
    note = (f"{label}: within tolerance" if ok
            else f"{label}: diff={actual - claimed:.4f}")
    return _result(item, claimed, round(actual, 4), ok, note)
