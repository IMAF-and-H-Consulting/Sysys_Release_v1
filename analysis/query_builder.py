"""Search components and weighted Boolean queries.

Split out of ``analysis.scoring`` (audit 03-L10); every public name is still
importable from ``analysis.scoring``.  ``core.research_components`` builds the
Search-tab components through ``build_hypothesis_components`` too, so the
preview and the search use one builder and one ID scheme (03-M2).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Any

from core.models import HypothesisComponent

logger = logging.getLogger(__name__)

# Words that do not carry useful search meaning by themselves.
QUERY_STOPWORDS = {
    "a",
    "an",
    "about",
    "affect",
    "affected",
    "affecting",
    "affects",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "current",
    "does",
    "evidence",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "question",
    "research",
    "say",
    "scholarly",
    "that",
    "this",
    "the",
    "their",
    "to",
    "through",
    "user",
    "what",
    "vs",
    "with",
    # Negations and function words: they never name a concept, and leaving
    # them in produced exact n-grams such as "imbalance not explain" (03-H3).
    "all",
    "among",
    "any",
    "been",
    "between",
    "did",
    "do",
    "either",
    "had",
    "has",
    "have",
    "its",
    "neither",
    "never",
    "no",
    "nor",
    "not",
    "than",
    "these",
    "those",
    "was",
    "were",
    "when",
    "where",
    "whether",
    "which",
    "who",
    "why",
    "without",
    # Proximity operators in WoS / Ovid (03-M4).
    "adj",
    "near",
    "same",
}

# Words that providers may read as Boolean or proximity operators (03-M4).
BOOLEAN_OPERATOR_WORDS = frozenset({"and", "or", "not", "near", "same", "adj"})
# A term that needs no quotes: letters/digits, optionally a trailing wildcard.
_BARE_TERM_RE = re.compile(r"[A-Za-z0-9]+\*?")

# Component-weight thresholds for the weighted Boolean query (03-H2).
REQUIRED_CONCEPT_WEIGHT = 1.35  # the component becomes its own AND block
INCLUDED_CONCEPT_WEIGHT = 0.35  # below this the component is left out

# Components built by ``build_hypothesis_components`` carry a kind prefix.
# Components of one facet are alternatives (ORed); facets are ANDed.
# Unprefixed components are treated as topic alternatives.
_FACET_BY_ID_PREFIX = (
    ("subject_", "subject"),
    ("domain_", "topic"),
    ("condition_", "topic"),
    ("term_", "question"),
)
_FACET_ORDER = ("subject", "topic", "question")
_DEFAULT_FACET = "topic"


# ---------------------------------------------------------------------------
# Term helpers
# ---------------------------------------------------------------------------
def _unique_terms(terms: list[Any]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        cleaned = " ".join(str(term).split())
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def normalize_terms(value: Any) -> list[str]:
    """A YAML term field as unique strings (03-M7).

    ``None`` is no terms, a scalar is one term (never split per character),
    and ``None`` / mapping items inside a list are skipped.
    """
    return _unique_terms(_term_items(value))


def _term_items(value: Any) -> list[str]:
    """Non-empty strings of a term field, in order, duplicates kept."""
    if value is None or isinstance(value, dict):
        return []
    items = list(value) if isinstance(value, (list, tuple)) else [value]
    cleaned = (
        " ".join(str(item).split())
        for item in items if item is not None and not isinstance(item, dict)
    )
    return [item for item in cleaned if item]


def _mapping(value: Any, name: str) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    logger.warning(
        "Ignoring research config %r: expected a mapping, got %s",
        name, type(value).__name__,
    )
    return {}


def search_domains(search: dict) -> list[dict]:
    """The ``search.domains`` mappings; malformed entries are logged and skipped."""
    raw = search.get("domains")
    if raw is None:
        return []
    if not isinstance(raw, list):
        logger.warning(
            "Ignoring research config 'search.domains': expected a list, got %s",
            type(raw).__name__,
        )
        return []
    domains: list[dict] = []
    for index, domain in enumerate(raw, start=1):
        if isinstance(domain, dict):
            domains.append(domain)
        else:
            logger.warning(
                "Ignoring search domain %d: expected a mapping with 'name' and 'terms', got %r",
                index, domain,
            )
    return domains


def _search_term_id(term: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", term.lower()).strip("_")
    return f"term_{slug or 'query'}"


def _component_id(prefix: str, label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"{prefix}_{slug or 'component'}"


def quote_boolean_term(term: str) -> str:
    """A term safe to place in a Boolean query (03-M4).

    Embedded double quotes are removed.  Only plain letters/digits (with an
    optional trailing ``*`` wildcard) that are not operator words stay bare;
    everything else is quoted as a phrase.
    """
    cleaned = " ".join(str(term).replace('"', " ").split())
    if not cleaned:
        return ""
    if _BARE_TERM_RE.fullmatch(cleaned) and cleaned.lower() not in BOOLEAN_OPERATOR_WORDS:
        return cleaned
    return f'"{cleaned}"'


# ---------------------------------------------------------------------------
# Question terms
# ---------------------------------------------------------------------------
def _query_tokens(query: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9+\-/]*", query.lower()):
        token = raw.strip("-/")
        if len(token) < 3 or token in QUERY_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _query_token_segments(query: str) -> list[list[str]]:
    segments: list[list[str]] = []
    current: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9+\-/]*", query.lower()):
        token = raw.strip("-/")
        if len(token) < 3 or token in QUERY_STOPWORDS:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def _query_phrase_candidates(query: str, *, min_size: int = 2, max_size: int = 4) -> list[str]:
    phrases: list[str] = []
    for tokens in _query_token_segments(query):
        for size in range(min(max_size, len(tokens)), min_size - 1, -1):
            for idx in range(max(0, len(tokens) - size + 1)):
                phrase = " ".join(tokens[idx:idx + size])
                if phrase:
                    phrases.append(phrase)
    return _unique_terms(phrases)


def build_search_term_components(query: str) -> list[HypothesisComponent]:
    """Build weighted components from meaningful query phrases and words.

    The Search tab uses these components as the editable model for keyword
    analysis. Phrase components let the system learn that a compound concept
    retrieved useful papers, while word components keep the query tunable.
    """
    seen: set[str] = set()
    components: list[HypothesisComponent] = []

    for phrase in _query_phrase_candidates(query):
        if phrase in seen:
            continue
        seen.add(phrase)
        components.append(HypothesisComponent(
            id=_search_term_id(f"phrase_{phrase}"),
            label=phrase,
            description=f"Phrase from question: {phrase}",
            keywords=[phrase],
            weight=1.0,
        ))

    for term in _query_tokens(query):
        if term in seen:
            continue
        seen.add(term)
        components.append(HypothesisComponent(
            id=_search_term_id(term),
            label=term,
            description=f"Search term from question: {term}",
            keywords=[term],
            weight=1.0,
        ))
    return components


# ---------------------------------------------------------------------------
# Weighted Boolean query
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BooleanQueryPlan:
    """A weighted Boolean query plus the terms it leaves out."""

    query: str
    blocks: tuple[tuple[str, ...], ...]
    dropped_terms: tuple[str, ...]


@dataclass(frozen=True)
class _ConceptGroup:
    terms: tuple[str, ...]
    is_facet: bool


def _component_weight(component: HypothesisComponent) -> float:
    return float(getattr(component, "weight", 1.0) or 0.0)


def _component_boolean_terms(component: HypothesisComponent) -> tuple[str, ...]:
    raw_terms = getattr(component, "keywords", []) or [component.label]
    quoted = (quote_boolean_term(term) for term in raw_terms)
    return tuple(term for term in quoted if term)


def _component_facet(component: HypothesisComponent) -> str:
    for prefix, facet in _FACET_BY_ID_PREFIX:
        if component.id.startswith(prefix):
            return facet
    return _DEFAULT_FACET


def _concept_groups(
    components: list[HypothesisComponent],
) -> tuple[list[_ConceptGroup], tuple[str, ...]]:
    """Group component terms into AND-ed concepts; return disabled terms too."""
    enabled = [c for c in components if _component_weight(c) >= INCLUDED_CONCEPT_WEIGHT]
    disabled = [c for c in components if _component_weight(c) < INCLUDED_CONCEPT_WEIGHT]
    required = [
        _ConceptGroup(_component_boolean_terms(c), is_facet=False)
        for c in enabled if _component_weight(c) >= REQUIRED_CONCEPT_WEIGHT
    ]
    alternatives = [c for c in enabled if _component_weight(c) < REQUIRED_CONCEPT_WEIGHT]
    facets = [
        _ConceptGroup(
            tuple(
                term
                for c in alternatives if _component_facet(c) == facet
                for term in _component_boolean_terms(c)
            ),
            is_facet=True,
        )
        for facet in _FACET_ORDER
    ]
    dropped = tuple(term for c in disabled for term in _component_boolean_terms(c))
    return [*required, *facets], dropped


def _dedupe_concept_groups(groups: list[_ConceptGroup]) -> list[_ConceptGroup]:
    """Keep each term only in its first group, so one term cannot satisfy two ANDs."""
    seen: set[str] = set()
    deduped: list[_ConceptGroup] = []
    for group in groups:
        terms = tuple(_unique_terms([t for t in group.terms if t.lower() not in seen]))
        seen.update(term.lower() for term in terms)
        if terms:
            deduped.append(_ConceptGroup(terms, group.is_facet))
    return deduped


def _format_concept_group(group: _ConceptGroup) -> str:
    if len(group.terms) == 1 and not group.is_facet:
        return group.terms[0]
    return f"({' OR '.join(group.terms)})"


def build_boolean_query_plan(components: list[HypothesisComponent]) -> BooleanQueryPlan:
    """Build a concept-structured Boolean query from weighted components.

    Synonyms inside a component are ORed.  A component weighted at
    ``REQUIRED_CONCEPT_WEIGHT`` or more is its own required concept.  Other
    enabled components are ORed with the rest of their facet (subject, topic
    = domains and conditions, question terms) and the facets are ANDed.
    Components below ``INCLUDED_CONCEPT_WEIGHT`` are left out and reported in
    ``dropped_terms``.  No term is truncated.
    """
    groups, disabled_terms = _concept_groups(components)
    blocks = _dedupe_concept_groups(groups)
    included = {term.lower() for block in blocks for term in block.terms}
    dropped = tuple(_unique_terms([t for t in disabled_terms if t.lower() not in included]))
    return BooleanQueryPlan(
        query=" AND ".join(_format_concept_group(block) for block in blocks),
        blocks=tuple(block.terms for block in blocks),
        dropped_terms=dropped,
    )


def describe_boolean_query_plan(plan: BooleanQueryPlan) -> str:
    """Preview text: the query, then the terms it leaves out (03-M2)."""
    if not plan.dropped_terms:
        return plan.query
    return (
        f"{plan.query}\n\nLeft out (weight below {INCLUDED_CONCEPT_WEIGHT}): "
        f"{', '.join(plan.dropped_terms)}"
    )


def build_weighted_boolean_query(components: list[HypothesisComponent]) -> str:
    """Build a provider-neutral weighted Boolean query from component weights."""
    plan = build_boolean_query_plan(components)
    if plan.dropped_terms:
        logger.info(
            "Weighted Boolean query leaves out low-weight terms: %s",
            ", ".join(plan.dropped_terms),
        )
    return plan.query


def build_weighted_component_query_terms(
    components: list[HypothesisComponent],
) -> list[str]:
    """Return query terms ordered by current component weight.

    Components below ``INCLUDED_CONCEPT_WEIGHT`` are left out, as they are
    from the Boolean query (03-L8); zero weight means explicitly disabled.
    """
    weighted: list[tuple[float, int, int, str]] = []
    for idx, component in enumerate(components):
        weight = _component_weight(component)
        if weight < INCLUDED_CONCEPT_WEIGHT:
            continue
        keywords = getattr(component, "keywords", []) or []
        terms = _unique_terms(list(keywords) if keywords else [component.label])
        weighted.extend((-weight, idx, term_idx, term) for term_idx, term in enumerate(terms))

    weighted.sort()
    return _unique_terms([term for _, _, _, term in weighted])


# ---------------------------------------------------------------------------
# Component builder (the one builder for Search and the config preview)
# ---------------------------------------------------------------------------
def _subject_component(search: dict) -> list[HypothesisComponent]:
    base_terms = normalize_terms(search.get("base_terms"))
    if not base_terms:
        return []
    return [HypothesisComponent(
        id=_component_id("subject", " ".join(base_terms[:3])),
        label=f"Subject: {' / '.join(base_terms[:3])}",
        description="Primary subject terms from the research configuration.",
        keywords=base_terms,
        weight=1.0,
    )]


def _domain_components(search: dict) -> list[HypothesisComponent]:
    components: list[HypothesisComponent] = []
    for index, domain in enumerate(search_domains(search), start=1):
        name = " ".join(str(domain.get("name") or "").split()) or f"Domain {index}"
        terms = _unique_terms([
            *normalize_terms(domain.get("terms")),
            *normalize_terms(domain.get("require_also")),
        ])
        if not terms:
            logger.warning("Search domain %r has no terms; it is left out", name)
            continue
        components.append(HypothesisComponent(
            id=_component_id("domain", name),
            label=name,
            description=f"Search domain from config: {name}",
            keywords=terms,
            weight=1.0,
        ))
    return components


def _condition_components(conditions: Any) -> list[HypothesisComponent]:
    return [
        HypothesisComponent(
            id=_component_id("condition", condition),
            label=condition,
            description=f"Co-occurring term from the research configuration: {condition}",
            # Only the configured term: relationship words ("comorbid") would
            # match every condition and leak clinical vocabulary into searches.
            keywords=[condition],
            weight=1.0,
        )
        # Not de-duplicated here: repeats are merged (and logged) by ID.
        for condition in _term_items(conditions)
    ]


def _same_keywords(a: HypothesisComponent, b: HypothesisComponent) -> bool:
    return [k.lower() for k in a.keywords] == [k.lower() for k in b.keywords]


def _free_id(base: str, used: set[str]) -> str:
    suffix = 2
    while f"{base}_{suffix}" in used:
        suffix += 1
    return f"{base}_{suffix}"


def _dedupe_component_ids(components: list[HypothesisComponent]) -> list[HypothesisComponent]:
    """Merge true duplicates and rename colliding IDs, logging both (03-L7)."""
    kept: list[HypothesisComponent] = []
    by_id: dict[str, HypothesisComponent] = {}
    for comp in components:
        existing = by_id.get(comp.id)
        if existing is not None and _same_keywords(existing, comp):
            logger.info("Merged duplicate search component %s (%r)", comp.id, comp.label)
            continue
        if existing is not None:
            new_id = _free_id(comp.id, set(by_id))
            logger.warning(
                "Component ID %s is used by %r; %r is kept as %s",
                comp.id, existing.label, comp.label, new_id,
            )
            comp = replace(comp, id=new_id)
        by_id[comp.id] = comp
        kept.append(comp)
    return kept


def build_hypothesis_components(
    *,
    hypothesis: str = "",
    research_config: dict | None = None,
    co_occurring_conditions: list[str] | None = None,
    query: str = "",
) -> list[HypothesisComponent]:
    """Build weighted components from hypothesis parts, search domains, and query terms."""
    research = _mapping(research_config, "research_config")
    search = _mapping(research.get("search"), "search")
    relevance = _mapping(research.get("relevance"), "relevance")
    research_question = " ".join(
        str(relevance.get("research_question") or research.get("description") or "").split()
    )

    components = [
        *_subject_component(search),
        *_domain_components(search),
        *_condition_components(co_occurring_conditions),
        *build_search_term_components(query or research_question or hypothesis),
    ]
    return _dedupe_component_ids(components)


def merge_component_edits(
    fresh: list[HypothesisComponent],
    previous: list[HypothesisComponent],
) -> list[HypothesisComponent]:
    """New copies of *fresh* carrying the user's edits from *previous* (03-M2).

    Weight, keywords and description of a component with the same ID survive
    a rebuild; components that are no longer built are dropped.  Neither
    input list nor its components are changed.
    """
    previous_by_id = {comp.id: comp for comp in previous}
    merged: list[HypothesisComponent] = []
    for comp in fresh:
        old = previous_by_id.get(comp.id)
        if old is None:
            merged.append(replace(comp, keywords=list(comp.keywords)))
            continue
        merged.append(replace(
            comp,
            weight=old.weight,
            description=old.description,
            keywords=list(old.keywords or comp.keywords),
        ))
    return merged
