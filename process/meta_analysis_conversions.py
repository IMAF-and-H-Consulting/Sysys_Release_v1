"""Convert reported test statistics to effect sizes for the researcher to review.

Formulas follow Borenstein, Hedges, Higgins & Rothstein (2009), *Introduction
to Meta-Analysis*, ch. 7, and Lipsey & Wilson (2001):

- t(df) → d = t·√(1/n1 + 1/n2); without group sizes n1 = n2 = (df + 2)/2.
- F(1, df2) → t = √F, then as above. F carries no sign.
- χ²(1) with N → φ = √(χ²/N) on the r scale. χ² carries no sign.
- z with N → r = z/√N.
- an exact p with N and no statistic → z = Φ⁻¹(1 − p/2), r = z/√N (no sign).

A conversion is never pooled by default: the researcher includes it, and gives
a direction when the statistic carries no sign.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

MIN_CORRELATION_N = 4
MIN_SMD_N = 3


@dataclass(frozen=True)
class Conversion:
    effect_type: str          # "d" or "r" (natural scale)
    value: float
    se: float                 # d scale for "d"; Fisher's z scale for "r"
    n: int
    route: str
    formula: str
    assumptions: tuple[str, ...]
    sign_known: bool


def _number(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _ns(values) -> list[int]:
    seen: list[int] = []
    for item in values or []:
        n = _number(item.get("n", item.get("value")) if isinstance(item, dict) else item)
        if n is not None and n > 0 and n.is_integer() and int(n) not in seen:
            seen.append(int(n))
    return seen


def _total_n(result: dict, entry: dict, allow_paper_n: bool) -> tuple[int | None, tuple[str, ...], str]:
    """Single N for z / χ² / p conversions: the result's own, else the paper's (if allowed)."""
    own = _ns(result.get("sample_sizes"))
    if len(own) == 1:
        return own[0], (), ""
    if len(own) > 1:
        return None, (), f"several sample sizes ({', '.join(map(str, own))})"
    paper = _ns(entry.get("paper_sample_sizes"))
    if len(paper) == 1 and allow_paper_n:
        return paper[0], ("paper-level N applied to this result",), ""
    if len(paper) == 1:
        return None, (), f"only a paper-level N ({paper[0]}) is available"
    return None, (), "no sample size"


def _smd_se(d: float, n1: float, n2: float) -> float:
    return math.sqrt((n1 + n2) / (n1 * n2) + d * d / (2 * (n1 + n2)))


def _t_to_d(t: float, df: float, result: dict, route: str, sign_known: bool,
            formula: str) -> tuple[Conversion | None, str]:
    groups = _ns(result.get("sample_sizes"))
    if len(groups) == 2:
        n1, n2 = groups
        assumptions: tuple[str, ...] = ("independent groups assumed",)
    else:
        if df + 2 < MIN_SMD_N:
            return None, "too few degrees of freedom"
        n1 = n2 = (df + 2) / 2
        assumptions = ("independent groups assumed", "equal group sizes assumed")
    d = t * math.sqrt(1 / n1 + 1 / n2)
    return Conversion("d", d, _smd_se(d, n1, n2), int(round(n1 + n2)), route, formula,
                      assumptions, sign_known), ""


def _from_t(result: dict) -> tuple[Conversion | None, str]:
    t, df = _number(result.get("statistic")), _number((result.get("degrees_of_freedom") or {}).get("df"))
    if t is None or df is None:
        return None, "t statistic needs its df"
    return _t_to_d(t, df, result, "t_to_d", True, "d = t·√(1/n1 + 1/n2)")


def _from_f(result: dict) -> tuple[Conversion | None, str]:
    f = _number(result.get("statistic"))
    dfs = result.get("degrees_of_freedom") or {}
    df1, df2 = _number(dfs.get("df1")), _number(dfs.get("df2"))
    if f is None or df2 is None or f < 0:
        return None, "F statistic needs its df"
    if df1 != 1:
        return None, (f"F with {df1:g} numerator df compares more than two groups; "
                      "it has no single two-group effect" if df1 else "F needs its numerator df")
    return _t_to_d(math.sqrt(f), df2, result, "f_to_d", False, "t = √F; d = t·√(1/n1 + 1/n2)")


def _r_conversion(r: float, n: int, route: str, formula: str, assumptions, sign_known):
    if n < MIN_CORRELATION_N:
        return None, f"sample size {n} is too small"
    if not -1 < r < 1:
        return None, f"converted r = {r:.3g} is outside (-1, 1)"
    return Conversion("r", r, 1 / math.sqrt(n - 3), n, route, formula,
                      tuple(assumptions), sign_known), ""


def _from_chi_square(result, entry, allow_paper_n):
    chi2 = _number(result.get("statistic"))
    df = _number((result.get("degrees_of_freedom") or {}).get("df"))
    if chi2 is None or chi2 < 0:
        return None, "χ² statistic missing"
    if df != 1:
        return None, ("χ² needs df = 1 (a 2×2 table) to give φ" if df is None
                      else f"χ² with {df:g} df has no single two-group effect")
    n, assumptions, note = _total_n(result, entry, allow_paper_n)
    if n is None:
        return None, f"χ² needs N: {note}"
    return _r_conversion(math.sqrt(chi2 / n), n, "chi2_to_phi", "φ = √(χ²/N)", assumptions, False)


def _from_z(result, entry, allow_paper_n):
    z = _number(result.get("statistic"))
    if z is None:
        return None, "z statistic missing"
    n, assumptions, note = _total_n(result, entry, allow_paper_n)
    if n is None:
        return None, f"z needs N: {note}"
    return _r_conversion(z / math.sqrt(n), n, "z_to_r", "r = z/√N", assumptions, True)


def _from_p(result, entry, allow_paper_n):
    exact = {_number(p.get("value")) for p in result.get("p_values") or []
             if isinstance(p, dict) and str(p.get("operator", "")).strip() == "="}
    exact = {p for p in exact if p is not None and 0 < p < 1}
    if len(exact) != 1:
        return None, "no single exact p-value"
    n, assumptions, note = _total_n(result, entry, allow_paper_n)
    if n is None:
        return None, f"p-value conversion needs N: {note}"
    z = float(norm.isf(next(iter(exact)) / 2))
    return _r_conversion(z / math.sqrt(n), n, "p_to_r", "z = Φ⁻¹(1 − p/2); r = z/√N",
                         assumptions + ("effect inferred from the p-value alone",), False)


_BY_TEST = {
    "t_test": lambda r, e, a: _from_t(r),
    "f_test": lambda r, e, a: _from_f(r),
    "anova": lambda r, e, a: _from_f(r),
    "chi_square": _from_chi_square,
    "z_test": _from_z,
}


def convert_test_statistic(result: dict, entry: dict, *,
                           allow_paper_n: bool = False) -> tuple[Conversion | None, str]:
    """Return ``(Conversion, "")`` or ``(None, why it cannot be converted)``."""
    converter = _BY_TEST.get(str(result.get("test_type") or ""))
    if converter is not None:
        conversion, reason = converter(result, entry, allow_paper_n)
        if conversion is not None:
            return conversion, ""
        fallback, _ = _from_p(result, entry, allow_paper_n)
        return (fallback, "") if fallback is not None else (None, reason)
    return _from_p(result, entry, allow_paper_n)


# ── putting a researcher-defined group on one scale ─────────────────────

SMD_SCALE = "Standardized mean difference (harmonized)"
_SMD_SCALES = {"Hedges' g", "Cohen's d", SMD_SCALE}


def harmonize_to_smd(study):
    """Return ``(study on the SMD scale, "")`` or ``(None, why it cannot be)``.

    Borenstein et al. (2009) ch. 7: r → d = 2r/√(1−r²), var_d = 4·var_r/(1−r²)³;
    log OR → d = ln(OR)·√3/π, var_d = var·3/π². RR and HR have no standard
    conversion to d and are left out of a mixed group.
    """
    from dataclasses import replace

    if study.analysis_scale in _SMD_SCALES:
        return study, ""
    if study.analysis_scale == "Fisher's z":
        r = math.tanh(study.effect_size)
        var_r = (1 - r * r) ** 2 * study.variance
        d = 2 * r / math.sqrt(1 - r * r)
        var_d = 4 * var_r / (1 - r * r) ** 3
        note = "r converted to d = 2r/√(1−r²) to share a group's scale"
    elif study.analysis_scale in {"Log OR", "Log AOR"}:
        d = study.effect_size * math.sqrt(3) / math.pi
        var_d = study.variance * 3 / math.pi ** 2
        note = "log odds ratio converted to d = ln(OR)·√3/π to share a group's scale"
    else:
        return None, f"{study.analysis_scale} has no standard conversion to a standardized mean difference"
    se = math.sqrt(var_d)
    return replace(study, effect_size=d, standard_error=se, variance=var_d,
                   ci_lower=d - 1.96 * se, ci_upper=d + 1.96 * se, effect_type="d",
                   analysis_scale=SMD_SCALE, assumptions=study.assumptions + (note,)), ""
