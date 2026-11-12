"""Stage 8 robustness: condition configuration and outcome definitions.

Split out of stages/s9_robustness.py, which re-exports these names.
"""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.claims import PRIMARY_OUTCOME_FIELD, claim_flag, claim_list, claim_value
from process.domain_patterns import (
    condition_patterns,
    contains_term,
    literal_pattern,
    research_config,
    terms,
)

# ── Condition configuration ───────────────────────────────────────────────
# Condition *matching* follows the documented precedence shared with the
# network analysis (``process.domain_patterns.condition_patterns``): explicit
# ``graph_analysis.conditions`` regexes, else whole-term literals for
# ``co_occurring_conditions``, else the selected pattern pack (audit 12-M5).
# The condition *ranking* that the gradient permutation test correlates
# against is only the explicitly ordered ``co_occurring_conditions`` list: a
# pack's dict order is not a scientific ordering.
#
# Condition patterns come in two shapes, both accepted by the matchers below:
# ``{canonical: [regex, ...]}`` (from the config) and the legacy
# ``{literal_term: canonical}`` used by direct callers.

ConditionPatterns = Mapping[str, str | list[str]]


def _load_condition_config(config) -> tuple[dict[str, Any], list[str]]:
    """Condition regexes (canonical name -> list of regexes) and the explicit ranking.

    The values are typed ``Any`` only so callers that still annotate the
    legacy ``dict[str, str]`` shape (Stage 10) keep type-checking.
    """
    patterns = dict(condition_patterns(config))
    ranked = [str(c).strip() for c in getattr(config, "co_occurring_conditions", None) or []]
    by_lower = {name.lower(): name for name in patterns}
    ranking: list[str] = []
    for name in filter(None, ranked):
        canonical = by_lower.get(name.lower())
        if canonical is None:
            # A ranked condition without a graph regex keeps its literal match.
            canonical = name
            patterns[canonical] = [literal_pattern(name)]
            by_lower[name.lower()] = canonical
        if canonical not in ranking:
            ranking.append(canonical)
    return patterns, ranking


def condition_names(patterns: ConditionPatterns) -> list[str]:
    """Canonical condition names, in configuration order, for either shape."""
    return list(dict.fromkeys(canonical for canonical, _ in _condition_matchers(patterns)))


def _condition_matchers(patterns: ConditionPatterns) -> list[tuple[str, list[str]]]:
    """``(canonical, regexes)`` pairs for either pattern shape."""
    matchers: list[tuple[str, list[str]]] = []
    for key, value in patterns.items():
        if isinstance(value, str):
            matchers.append((value, [literal_pattern(key)]))
        else:
            matchers.append((key, list(value)))
    return matchers


def _matches(text: str, regexes: list[str]) -> bool:
    return any(re.search(regex, text, re.IGNORECASE) for regex in regexes)


# ── Paper field helpers ───────────────────────────────────────────────────

def _is_missing_score(paper: dict) -> bool:
    """True when a paper carries no relevance score at all (absent, null, blank)."""
    score = paper.get("relevance_score")
    return score is None or (isinstance(score, str) and not score.strip())


def _parse_relevance_score(paper: dict) -> float | None:
    """A finite numeric relevance score, or None when missing or unparseable.

    Numeric strings ("3.5") are parsed and fractions are kept (audit 12-L8).
    """
    score = paper.get("relevance_score")
    if _is_missing_score(paper) or isinstance(score, bool):
        return None
    try:
        value = float(score)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def _get_relevance_score(paper: dict) -> float:
    """Relevance score for threshold comparisons; missing or unparseable is 0."""
    value = _parse_relevance_score(paper)
    return 0.0 if value is None else value


def _usable_outcome_value(value) -> bool:
    """True for an extracted flag claim_flag can read: bool, number or text."""
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class RobustnessOutcome:
    """The extracted claim flag the robustness analyses count as "support".

    ``field`` names a boolean claim field (nested under ``claims`` or at the
    top level); 1/0 numbers and "yes"/"no" text are read as flags too. When a
    paper has no value for it, ``fallback_terms`` are matched against its
    findings; with no terms there is no keyword fallback and the paper has
    no outcome (``has_value`` is False). Such papers still count as
    non-supporting in the denominators; ``run`` records how many papers have
    an outcome and refuses to report outcome analyses when none do.
    """

    field: str = PRIMARY_OUTCOME_FIELD
    fallback_terms: tuple[str, ...] = ()

    def has_value(self, paper: dict) -> bool:
        """Whether this paper's outcome was determined (flag or keyword fallback)."""
        if _usable_outcome_value(claim_value(paper, self.field)):
            return True
        return bool(self.fallback_terms) and bool(claim_list(paper, "findings"))

    def supports(self, paper: dict) -> bool:
        if _usable_outcome_value(claim_value(paper, self.field)):
            return claim_flag(paper, self.field)
        if not self.fallback_terms:
            return False
        findings = " ".join(claim_list(paper, "findings"))
        return any(contains_term(findings, term) for term in self.fallback_terms)


DEFAULT_OUTCOME = RobustnessOutcome()


def _load_outcome(config) -> RobustnessOutcome:
    """Resolve the robustness outcome from ``robustness`` in the research config.

    ``robustness.outcome_field`` picks the claim flag (default
    ``supports_primary_question``) and ``robustness.outcome_terms`` the
    optional keyword fallback. For the legacy ``supports_criticality`` field
    the active pattern pack's ``criticality_terms`` serve as the fallback.
    """
    if config is None:
        return DEFAULT_OUTCOME
    section = research_config(config).get("robustness") or {}
    if not isinstance(section, dict):
        raise ValueError("robustness must be a mapping")
    field = str(section.get("outcome_field") or PRIMARY_OUTCOME_FIELD).strip()
    if "outcome_terms" in section:
        fallback = terms(section["outcome_terms"], "robustness.outcome_terms") \
            if section["outcome_terms"] else []
    elif field == "supports_criticality":
        from process.domain_patterns import extraction_patterns
        fallback = extraction_patterns(config).get("criticality_terms", [])
    else:
        fallback = []
    return RobustnessOutcome(field=field, fallback_terms=tuple(fallback))


def _normalize_condition(raw: str, patterns: ConditionPatterns) -> str | None:
    """Map raw text to a canonical condition name from the given patterns.

    Literal terms match whole terms only, so "rett" never matches "Barrett".
    """
    for canonical, regexes in _condition_matchers(patterns):
        if _matches(raw, regexes):
            return canonical
    return None


def _extract_conditions(paper: dict, patterns: ConditionPatterns) -> list[str]:
    """Extract normalized conditions from a paper.

    Checks the ``conditions`` field (list of strings), ``title``,
    and ``abstract`` for condition pattern matches. ``patterns`` is required.
    """
    found: set[str] = set()
    matchers = _condition_matchers(patterns)

    # From the extracted conditions field (nested under claims or top level)
    for raw in claim_list(paper, "conditions"):
        found.update(c for c, regexes in matchers if _matches(raw, regexes))

    # From title + abstract text
    text = " ".join(filter(None, [
        paper.get("title", ""),
        paper.get("abstract", ""),
    ]))
    found.update(c for c, regexes in matchers if _matches(text, regexes))

    return list(found)
