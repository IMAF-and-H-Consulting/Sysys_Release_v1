"""Turn one extracted statistic into a calculable effect, or say exactly why not.

Every reported piece that can carry the uncertainty is tried in turn: a
confidence interval, a reported standard error, an exact p-value, the result's
own sample size(s) and, only when the researcher allows it, the paper-level N.
One sufficient piece is enough: the rule is "this OR that", never "this AND
that". A statistic that still cannot be used is returned with what was found
and what was missing, so nothing is passed over silently.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace

from scipy.stats import norm

from process.meta_analysis_conversions import convert_test_statistic
from process.meta_analysis_models import StudyEffect
from process.statistical_extractor_numbers import normalize_number

SUPPORTED_EFFECTS = frozenset({"d", "g", "r", "or", "aor", "rr", "hr"})
RATIO_EFFECTS = frozenset({"or", "aor", "rr", "hr"})
MIN_CORRELATION_N = 4   # SE of Fisher's z is 1/sqrt(n - 3)
MIN_SMD_N = 3
DEFAULT_CI_LEVEL = 95.0
_EFFECT_ALIASES = {"odds_ratio": "or", "risk_ratio": "rr", "hazard_ratio": "hr",
                   "correlation": "r", "pearson_r": "r", "cohens_d": "d",
                   "hedges_g": "g", "cohen_d": "d"}
# Test statistics that standard formulas can turn into d or r (not enabled yet).
_CONVERTIBLE_TESTS = frozenset({"t_test", "f_test", "anova", "chi_square", "z_test"})

ADJACENT_P_WINDOW = 80  # characters after a statistic searched for its own p-value
_P_TOKEN = re.compile(
    r"(?<![A-Za-z])p(?:\s*-?\s*value)?\s*(?P<op><=|>=|[<>=≤≥])\s*"
    r"(?P<value>[-−]?(?:\d+(?:[.·]\d+)?|[.·]\d+)(?:\s*[eE][-−]?\d+|\s*[×x]\s*10\s*[-−⁻]?\s*\d+)?)",
    re.IGNORECASE)
# Another statistic or a clause/sentence break between a statistic and a p-value
# means that p-value belongs to something else.
_OTHER_STATISTIC = re.compile(
    r"(?<![A-Za-z])(?:r|t|F|d|g|z|β|OR|RR|HR|χ2?|chi2?)\s*(?:\(|=)", re.IGNORECASE)
_CLAUSE_END = re.compile(r";|\n|\.\s+[A-Z]")

STATUS_CALCULABLE = "calculable"
STATUS_CONVERTED = "converted"  # derived from a test statistic; awaits the researcher
STATUS_PASSED_UP = "passed_up"
PAPER_N_SETTING = "use paper-level N"

REASON_LABELS = {
    "context_review_excluded": "Context review excluded the statistic",
    "no_effect_size": "No effect size reported",
    "awaiting_decision": "Converted from a test statistic; awaiting the researcher's decision",
    "direction_needed": "Included conversion needs a direction (the statistic carries no sign)",
    "not_convertible": "Test statistic that cannot be converted to an effect size",
    "excluded_by_researcher": "Excluded by the researcher",
    "invalid_effect": "Missing, unsupported or out-of-range effect estimate",
    "no_uncertainty": "Effect found but no CI, SE, exact p or sample size",
    "paper_n_only": f"Only a paper-level N is available ('{PAPER_N_SETTING}' is off)",
    "ambiguous_uncertainty": "Several CIs, p-values or sample sizes; none can be matched",
    "legacy_linkage": "Legacy statistics lack effect-to-uncertainty linkage",
    "missing_article_id": "Missing article identifier",
}


def number(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def effect_type(value) -> str:
    name = str(value or "").lower().replace("'", "").strip()
    return _EFFECT_ALIASES.get(name, name)


@dataclass(frozen=True)
class _Uncertainty:
    se: float
    source: str
    n: int = 0
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Interval:
    lower: float
    upper: float
    level: float


def _scale(kind: str):
    if kind in RATIO_EFFECTS:
        return math.log, f"Log {kind.upper()}"
    if kind == "r":
        return math.atanh, "Fisher's z"
    return float, "Hedges' g" if kind == "g" else "Cohen's d"


def _invalid_effect(kind: str, value: float | None) -> str:
    if value is None or kind not in SUPPORTED_EFFECTS:
        return f"effect '{kind or '?'}' is missing, unsupported or non-finite"
    if kind in RATIO_EFFECTS and value <= 0:
        return f"ratio estimate {value:g} is not positive"
    if kind == "r" and not -1 < value < 1:
        return f"correlation {value:g} is outside (-1, 1)"
    return ""


def _intervals(result: dict) -> list[_Interval]:
    intervals = []
    for ci in result.get("confidence_intervals") or []:
        if isinstance(ci, dict):
            lower, upper = number(ci.get("lower")), number(ci.get("upper"))
            level = number(ci.get("level", DEFAULT_CI_LEVEL))
        elif isinstance(ci, (list, tuple)) and len(ci) == 2:
            (lower, upper), level = map(number, ci), DEFAULT_CI_LEVEL
        else:
            continue
        if lower is not None and upper is not None and level is not None and 0 < level < 100:
            intervals.append(_Interval(lower, upper, level))
    return intervals


def _interval_fits(kind: str, value: float, ci: _Interval) -> bool:
    if not ci.lower < ci.upper or not ci.lower <= value <= ci.upper:
        return False
    if kind in RATIO_EFFECTS:
        return ci.lower > 0
    if kind == "r":
        return -1 < ci.lower and ci.upper < 1
    return True


def _from_interval(kind, value, transform, result) -> tuple[_Uncertainty | None, str]:
    intervals = _intervals(result)
    fitting = [ci for ci in intervals if _interval_fits(kind, value, ci)]
    if len(fitting) == 1:
        ci = fitting[0]
        zcrit = float(norm.ppf(.5 + ci.level / 200))
        se = (transform(ci.upper) - transform(ci.lower)) / (2 * zcrit)
        return _Uncertainty(se, "confidence_interval"), ""
    if len(fitting) > 1:
        return None, f"{len(fitting)} confidence intervals contain the estimate"
    if intervals:
        return None, "no reported confidence interval contains the estimate"
    return None, "no CI"


def _from_reported_se(kind, result) -> _Uncertainty | None:
    se = number(result.get("standard_error"))
    scale = result.get("standard_error_scale")
    if se is None or se <= 0:
        return None
    if (kind in {"d", "g"} or (kind in RATIO_EFFECTS and scale == "log")
            or (kind == "r" and scale == "fisher_z")):
        return _Uncertainty(se, "standard_error")
    return None


def _adjacent_p(result) -> tuple[str, float] | None:
    """The p-value written directly after the statistic, as ``(operator, value)``.

    Extraction links every p-value in the surrounding text to a statistic; the
    one that belongs to it is the next p in the same parenthetical or clause,
    with no other statistic in between ("r = 0.56, p = 0.008 r = 0.46, p = 0.08").
    """
    context, text = result.get("context") or "", result.get("test_text") or ""
    starts = [m.end() for m in re.finditer(re.escape(text), context)] if text else []
    if len(starts) != 1:
        return None
    tail = context[starts[0]:starts[0] + ADJACENT_P_WINDOW]
    match = _P_TOKEN.search(tail)
    if not match:
        return None
    gap = tail[:match.start()]
    if (_OTHER_STATISTIC.search(gap) or _CLAUSE_END.search(gap)
            or gap.count(")") > gap.count("(")):
        return None
    value = normalize_number(match.group("value"))
    if value is None:
        return None
    op = {"≤": "<=", "≥": ">="}.get(match.group("op"), match.group("op"))
    return op, value


def _from_p_value(estimate, result) -> tuple[_Uncertainty | None, str]:
    adjacent = _adjacent_p(result)
    if adjacent is not None:
        op, p = adjacent
        if op != "=" or not 0 < p < 1:
            return None, f"the p-value next to the statistic is a bound (p {op} {p:g})"
        if estimate == 0:
            return None, "exact p-value cannot give an SE for a zero estimate"
        return _Uncertainty(abs(estimate) / float(norm.isf(p / 2)), "p_value"), ""
    unc, note = _from_linked_p_values(estimate, result)
    if unc is None:
        return None, note
    return _Uncertainty(unc.se, unc.source, unc.n,
                        ("p-value not directly after the statistic; linked from the surrounding text",)), ""


def _from_linked_p_values(estimate, result) -> tuple[_Uncertainty | None, str]:
    p_values = [p for p in result.get("p_values") or [] if isinstance(p, dict)]
    exact = {number(p.get("value")) for p in p_values
             if str(p.get("operator", "")).strip() == "="}
    exact = {p for p in exact if p is not None and 0 < p < 1}
    if len(exact) > 1:
        return None, f"{len(exact)} different exact p-values"
    if not exact:
        if p_values:
            return None, "no exact p-value (only '<' or '>' bounds)"
        return None, "no exact p-value"
    if estimate == 0:
        return None, "exact p-value cannot give an SE for a zero estimate"
    z = float(norm.isf(next(iter(exact)) / 2))
    return _Uncertainty(abs(estimate) / z, "p_value"), ""


def sample_sizes(values) -> list[int]:
    """Distinct positive whole-number Ns, in reported order."""
    seen: list[int] = []
    for item in values or []:
        n = number(item.get("n", item.get("value")) if isinstance(item, dict) else item)
        if n is not None and n > 0 and n.is_integer() and int(n) not in seen:
            seen.append(int(n))
    return seen


def _smd_se(d: float, n1: float, n2: float) -> float:
    return math.sqrt((n1 + n2) / (n1 * n2) + d * d / (2 * (n1 + n2)))


def _from_sample_sizes(kind, value, ns, source) -> tuple[_Uncertainty | None, str]:
    if kind in RATIO_EFFECTS:
        return None, "a ratio needs a CI, SE or exact p (a sample size alone is not enough)"
    if not ns:
        return None, "no sample size"
    if kind == "r":
        if len(ns) > 1:
            return None, f"several sample sizes ({', '.join(map(str, ns))}); which one applies is unclear"
        if ns[0] < MIN_CORRELATION_N:
            return None, f"sample size {ns[0]} is too small for a correlation SE"
        return _Uncertainty(1 / math.sqrt(ns[0] - 3), source, ns[0]), ""
    if len(ns) == 2:
        n1, n2 = ns
        return _Uncertainty(_smd_se(value, n1, n2), source, n1 + n2), ""
    if len(ns) > 2:
        return None, f"several sample sizes ({', '.join(map(str, ns))}); groups are unclear"
    if ns[0] < MIN_SMD_N:
        return None, f"sample size {ns[0]} is too small for a standardized mean difference"
    half = ns[0] / 2
    return _Uncertainty(_smd_se(value, half, half), source, ns[0],
                        ("equal group sizes assumed",)), ""


def _from_correlation_df(kind, result) -> _Uncertainty | None:
    """r(df) reports df = n - 2, so the correlation's own n is df + 2."""
    df = number((result.get("degrees_of_freedom") or {}).get("df"))
    if kind != "r" or df is None or not df.is_integer() or df + 2 < MIN_CORRELATION_N:
        return None
    n = int(df) + 2
    return _Uncertainty(1 / math.sqrt(n - 3), "degrees_of_freedom", n, ("n = df + 2 from r(df)",))


def _from_paper_n(kind, value, entry, allow_paper_n) -> tuple[_Uncertainty | None, str, str]:
    """Return (uncertainty, note, reason_code) for the paper-level N route."""
    paper_ns = sample_sizes(entry.get("paper_sample_sizes"))
    if not paper_ns or kind in RATIO_EFFECTS:
        return None, "", ""
    if len(paper_ns) > 1:
        return None, (f"several paper-level sample sizes ({', '.join(map(str, paper_ns))}); "
                      "none used"), "ambiguous_uncertainty"
    if not allow_paper_n:
        return None, (f"only a paper-level N ({paper_ns[0]}) is available; enable "
                      f"'{PAPER_N_SETTING}' to use it"), "paper_n_only"
    unc, note = _from_sample_sizes(kind, value, paper_ns, "paper_sample_size")
    if unc is None:
        return None, note, "no_uncertainty"
    return (_Uncertainty(unc.se, unc.source, unc.n,
                         unc.assumptions + ("paper-level N applied to this result",)),
            "", "")


def _noting(unc: _Uncertainty, ci_note: str) -> _Uncertainty:
    """Keep a rejected CI visible when another source supplied the SE."""
    if ci_note != "no reported confidence interval contains the estimate":
        return unc
    return _Uncertainty(unc.se, unc.source, unc.n,
                        unc.assumptions + ("a reported CI did not contain the estimate and was ignored",))


def _uncertainty(kind, value, estimate, transform, entry, result, allow_paper_n):
    """First sufficient source of uncertainty, or (None, notes, reason_code)."""
    unc, ci_note = _from_interval(kind, value, transform, result)
    if unc:
        return unc, [], ""
    notes = [ci_note]
    unc = _from_reported_se(kind, result)
    if unc:
        return _noting(unc, ci_note), [], ""
    notes.append("no usable SE")
    unc, note = _from_p_value(estimate, result)
    if unc:
        return _noting(unc, ci_note), [], ""
    notes.append(note)
    unc, note = _from_sample_sizes(kind, value, sample_sizes(result.get("sample_sizes")), "sample_size")
    if unc:
        return _noting(unc, ci_note), [], ""
    notes.append(note)
    unc = _from_correlation_df(kind, result)
    if unc:
        return _noting(unc, ci_note), [], ""
    unc, note, code = _from_paper_n(kind, value, entry, allow_paper_n)
    if unc:
        return _noting(unc, ci_note), [], ""
    if note:
        notes.append(note)
    ambiguous = any(word in n for n in notes for word in ("several", "different exact", "contain the estimate"))
    return None, notes, code or ("ambiguous_uncertainty" if ambiguous else "no_uncertainty")


def _with_n(unc: _Uncertainty, result: dict) -> _Uncertainty:
    """Fill in the sample size for routes that did not need one, when unique."""
    if unc.n:
        return unc
    ns = sample_sizes(result.get("sample_sizes"))
    return _Uncertainty(unc.se, unc.source, ns[0] if len(ns) == 1 else 0, unc.assumptions)


def _study(entry, result, review, kind, value, unc) -> StudyEffect:
    transform, scale = _scale(kind)
    estimate, se, n = transform(value), unc.se, unc.n
    if kind == "d" and n > 2:
        correction = 1 - 3 / (4 * (n - 2) - 1)
        estimate, se = estimate * correction, se * correction
        kind, scale = "g", "Hedges' g"
    return StudyEffect(
        article_id=str(entry.get("article_id") or entry.get("doi") or ""),
        title=entry.get("title", ""), effect_size=estimate,
        standard_error=se, variance=se * se, sample_size=n,
        ci_lower=estimate - 1.96 * se, ci_upper=estimate + 1.96 * se,
        effect_type=kind, analysis_scale=scale,
        analysis_group=str(result.get("analysis_group") or entry.get("analysis_group") or ""),
        outcome=str(result.get("outcome") or review.get("outcome") or entry.get("outcome") or ""),
        comparison=str(result.get("comparison") or review.get("comparison") or entry.get("comparison") or ""),
        population=str(result.get("population") or review.get("population") or entry.get("population") or ""),
        source_location=result.get("location") or {},
        condition=entry.get("condition", ""), component_id=entry.get("component_id", ""),
        uncertainty_source=unc.source, assumptions=unc.assumptions,
    )


def _llm_verdict(review: dict) -> str:
    if not review:
        return ""
    if review.get("check_failed"):
        return "check failed"
    if review.get("usable_for_meta_analysis") is False:
        return f"not usable: {review.get('reason') or 'no reason given'}"
    return "usable"


def _fmt_effects(result) -> str:
    return "; ".join(f"{effect_type(e.get('type')) or '?'}={e.get('value')}" if isinstance(e, dict)
                     else f"?={e}" for e in result.get("effect_sizes") or [] if e is not None)


def _fmt_intervals(result) -> str:
    return "; ".join(f"{ci.lower:g} to {ci.upper:g} ({ci.level:g}%)" for ci in _intervals(result))


def _fmt_p(result) -> str:
    return "; ".join(f"{p.get('operator', '=')}{p.get('value')}"
                     for p in result.get("p_values") or [] if isinstance(p, dict))


def effect_id(entry: dict, result: dict) -> str:
    """Stable id for one extracted statistic, used to store the researcher's decisions."""
    location = result.get("location") or {}
    key = "|".join(str(part) for part in (
        entry.get("article_id") or entry.get("doi") or "", result.get("test_text") or "",
        location.get("section", ""), location.get("char_start", ""), result.get("test_type") or ""))
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()


def _row(entry, result, review, status, code, reason, study=None, used="", route="",
         decision=None) -> dict:
    location = result.get("location") or {}
    decision = decision or {}
    return {
        "effect_id": effect_id(entry, result),
        "article_id": str(entry.get("article_id") or entry.get("doi") or ""),
        "title": entry.get("title", ""),
        "test_type": result.get("test_type") or "",
        "test_text": result.get("test_text") or "",
        "effects": _fmt_effects(result),
        "confidence_intervals": _fmt_intervals(result),
        "standard_error": result.get("standard_error") if result.get("standard_error") is not None else "",
        "p_values": _fmt_p(result),
        "sample_sizes": "; ".join(map(str, sample_sizes(result.get("sample_sizes")))),
        "paper_sample_sizes": "; ".join(map(str, sample_sizes(entry.get("paper_sample_sizes")))),
        "degrees_of_freedom": json.dumps(result.get("degrees_of_freedom") or {}, sort_keys=True)
        if result.get("degrees_of_freedom") else "",
        "llm_verdict": _llm_verdict(review),
        "status": status,
        "reason_code": code,
        "reason": reason,
        "effect_used": used,
        "route": route,
        "analysis_scale": study.analysis_scale if study else "",
        "analysis_estimate": study.effect_size if study else "",
        "analysis_se": study.standard_error if study else "",
        "analysis_group": study.analysis_group if study else str(decision.get("group") or ""),
        "include": "" if decision.get("include") is None else bool(decision.get("include")),
        "direction": str(decision.get("direction") or ""),
        "uncertainty_source": study.uncertainty_source if study else "",
        "assumptions": "; ".join(study.assumptions) if study else "",
        "outcome": study.outcome if study else (result.get("outcome") or review.get("outcome") or ""),
        "comparison": study.comparison if study else (result.get("comparison") or review.get("comparison") or ""),
        "population": study.population if study else (result.get("population") or review.get("population") or ""),
        "section": location.get("section", ""),
        "char_start": location.get("char_start", ""),
    }


def _with_decision(study: StudyEffect, decision: dict) -> StudyEffect:
    """Apply the decided group; a reversed comparison is turned to match the group.

    Every analysis scale (SMD, log ratio, Fisher's z) reverses by negation.
    """
    group = str(decision.get("group") or "").strip()
    if not group:
        return study
    if decision.get("reversed"):
        study = replace(study, effect_size=-study.effect_size, ci_lower=-study.ci_upper,
                        ci_upper=-study.ci_lower,
                        assumptions=(*study.assumptions, "comparison reversed to match the group"))
    return replace(study, analysis_group=group,
                   group_source=str(decision.get("group_source") or "researcher"))


def _paragraph(result: dict) -> tuple:
    location = result.get("location") or {}
    return location.get("text_source"), location.get("paragraph_index")


def _beside_reported_effect(entry: dict, result: dict) -> bool:
    """A reported effect size in the same paragraph likely restates this test's result."""
    location = _paragraph(result)
    if not all(value is not None for value in location):
        return False
    return any(other.get("effect_sizes") and _paragraph(other) == location
               for other in entry.get("statistical_tests") or [])


_REVIEW_DIRECTIONS = {"positive": "+", "negative": "-"}


def _direction(decision: dict, review: dict) -> tuple[str, str]:
    """("+" / "-" / "", its source): the researcher's choice, else the context review's.

    The review's direction refers to its own comparison label, the one model
    grouping marks as reversed or not, so the two compose.
    """
    chosen = str(decision.get("direction") or "")
    if chosen in {"+", "-"}:
        return chosen, "researcher"
    reviewed = _REVIEW_DIRECTIONS.get(str(review.get("effect_direction") or "").casefold(), "")
    return reviewed, "context review" if reviewed else ""


def _converted(entry, result, review, decision, allow_paper_n):
    """Effect derived from a test statistic.

    It is pooled by default when its direction is known (see ``_direction``)
    and no reported effect size in the same paragraph may be the same result;
    a printed t or z is often a magnitude, so its own sign is not enough.
    Otherwise it waits for the researcher, who must also give a direction
    when the statistic carries no sign (F, χ², p only).
    """
    conversion, why = convert_test_statistic(result, entry, allow_paper_n=allow_paper_n)
    text = result.get("test_text") or result.get("test_type") or "statistic"
    if conversion is None:
        code = "not_convertible" if result.get("test_type") in _CONVERTIBLE_TESTS else "no_effect_size"
        return None, _row(entry, result, review, STATUS_PASSED_UP, code,
                          f"{result.get('test_type') or 'statistic'} ({text}): {why or 'no effect size reported'}",
                          decision=decision)
    used = f"{conversion.effect_type}={conversion.value:.3g} (from {text})"
    note = (f"{result.get('test_type')} ({text}) is convertible to {conversion.effect_type} via "
            f"{conversion.formula}")
    direction, source = _direction(decision, review)
    include = decision.get("include")
    if include is None:
        include = bool(direction) and not _beside_reported_effect(entry, result)
    if not include:
        return None, _row(entry, result, review, STATUS_CONVERTED, "awaiting_decision",
                          note + "; not pooled until the researcher includes it",
                          used=used, route=conversion.route, decision=decision)
    if not conversion.sign_known and not direction:
        return None, _row(entry, result, review, STATUS_CONVERTED, "direction_needed",
                          note + "; included, but the statistic carries no sign: choose a direction",
                          used=used, route=conversion.route, decision=decision)
    value = conversion.value
    assumptions = conversion.assumptions + (f"converted: {conversion.formula}",)
    if direction:
        value = abs(value) * (-1 if direction == "-" else 1)
        assumptions += (f"direction from the {source}",)
    unc = _Uncertainty(conversion.se, f"conversion:{conversion.route}", conversion.n, assumptions)
    study = _with_decision(_study(entry, result, review, conversion.effect_type, value, unc), decision)
    return study, _row(entry, result, review, STATUS_CALCULABLE, "", "", study,
                       used=used, route=conversion.route, decision=decision)


def evaluate_result(entry: dict, result: dict, *, allow_paper_n: bool = False,
                    decision: dict | None = None):
    """Return ``(StudyEffect | None, audit row)`` for one extracted statistic.

    *decision* is the researcher's stored choice for this statistic
    (``include``, ``group``, ``direction``; see ``effect_id``).
    """
    review = result.get("llm_context_review") or {}
    decision = decision or {}

    def passed(code: str, reason: str):
        return None, _row(entry, result, review, STATUS_PASSED_UP, code, reason, decision=decision)

    if result.get("linkage_missing"):
        return passed("legacy_linkage", REASON_LABELS["legacy_linkage"])
    if review and not review.get("check_failed") and review.get("usable_for_meta_analysis") is False:
        return passed("context_review_excluded",
                      f"Context review excluded this result: {review.get('reason') or 'no reason given'}")
    # Legacy inputs list bare numbers; treat each as an untyped effect value.
    effects = [e if isinstance(e, dict) else {"value": e}
               for e in result.get("effect_sizes") or [] if e is not None]
    if not effects:
        return _converted(entry, result, review, decision, allow_paper_n)
    failures: list[tuple[str, str]] = []
    for effect in effects:
        kind = effect_type(effect.get("type")) or effect_type(result.get("effect_type"))
        if not kind and result.get("legacy_input"):
            kind = "d"  # Historical API accepted standardized mean differences.
        value = number(effect.get("value"))
        invalid = _invalid_effect(kind, value)
        if invalid or value is None:
            failures.append(("invalid_effect", f"{kind or '?'}={effect.get('value')}: {invalid}"))
            continue
        transform, _ = _scale(kind)
        unc, notes, code = _uncertainty(kind, value, transform(value), transform,
                                        entry, result, allow_paper_n)
        if unc is None:
            failures.append((code, f"{kind}={value:g}: " + "; ".join(n for n in notes if n)))
            continue
        unc = _with_n(unc, result)
        if not math.isfinite(unc.se) or unc.se <= 0:
            failures.append(("no_uncertainty", f"{kind}={value:g}: invalid standard error"))
            continue
        if decision.get("include") is False:
            return passed("excluded_by_researcher", REASON_LABELS["excluded_by_researcher"])
        study = _with_decision(_study(entry, result, review, kind, value, unc), decision)
        return study, _row(entry, result, review, STATUS_CALCULABLE, "",
                           "", study, used=f"{kind}={value:g}", decision=decision)
    code = failures[0][0]
    return passed(code, "; ".join(reason for _, reason in failures))
