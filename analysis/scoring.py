"""Keyword-based scoring of articles against hypothesis components.

Scores each article against each hypothesis component using keyword frequency,
co-occurrence, and directional language analysis.  No LLM calls — pure text.

Query building lives in ``analysis.query_builder`` and calibration in
``analysis.calibration`` (03-L10); their public names are re-exported here.
"""
from __future__ import annotations

import functools
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any

from analysis.calibration import calibrate_confidence  # noqa: F401  (re-export)
from analysis.query_builder import (  # noqa: F401  (re-exports)
    BOOLEAN_OPERATOR_WORDS,
    INCLUDED_CONCEPT_WEIGHT,
    QUERY_STOPWORDS,
    REQUIRED_CONCEPT_WEIGHT,
    BooleanQueryPlan,
    build_boolean_query_plan,
    build_hypothesis_components,
    build_search_term_components,
    build_weighted_boolean_query,
    build_weighted_component_query_terms,
    describe_boolean_query_plan,
    merge_component_edits,
    normalize_terms,
    quote_boolean_term,
)
from core.models import (
    AlignmentResult,
    AnalysisSession,
    Article,
    EvidenceDirection,
    EvidenceLink,
    HypothesisComponent,
    TrendPoint,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BOOLEAN_OPERATOR_WORDS",
    "INCLUDED_CONCEPT_WEIGHT",
    "QUERY_STOPWORDS",
    "REQUIRED_CONCEPT_WEIGHT",
    "AnalysisCancelled",
    "BooleanQueryPlan",
    "build_boolean_query_plan",
    "build_hypothesis_components",
    "build_search_term_components",
    "build_weighted_boolean_query",
    "build_weighted_component_query_terms",
    "calibrate_confidence",
    "compute_alignment",
    "compute_trends",
    "describe_boolean_query_plan",
    "merge_component_edits",
    "normalize_terms",
    "quote_boolean_term",
    "run_full_analysis",
    "score_article_against_component",
    "update_component_weights_from_keyword_analysis",
]

# ---------------------------------------------------------------------------
# Language patterns
# ---------------------------------------------------------------------------
NEGATION_PATTERNS = [
    r"\bnot\b", r"\bno\b", r"\bnor\b", r"\bnever\b", r"\bneither\b",
    r"\bwithout\b", r"\black\b", r"\bfail\w*\b", r"\binsufficient\b",
    r"\bunrelated\b", r"\bcontrary\b", r"\bunlike\b", r"\babsen\w*\b",
]

SUPPORT_PATTERNS = [
    r"\bsupport\w*\b", r"\bconfirm\w*\b", r"\bconsistent\b", r"\bdemonstrat\w*\b",
    r"\bcorrelat\w*\b", r"\bassociat\w*\b", r"\bsignificant\w*\b",
    r"\bincreas\w*\b", r"\benhance\w*\b", r"\bimplicat\w*\b", r"\bcontribut\w*\b",
]

# Relevance blends keyword frequency and co-occurrence only; tone is not
# evidence of relevance (03-H1).  The weights sum to 1.0.
KEYWORD_RELEVANCE_WEIGHT = 0.625
COOCCURRENCE_RELEVANCE_WEIGHT = 0.375
NEUTRAL_RELEVANCE_CUTOFF = 0.1
DIRECTION_CUTOFF = 0.2

# Keyword-analysis weight learning (03-L8): utility blend and weight mapping.
UTILITY_RELEVANCE_WEIGHT = 0.35
UTILITY_COVERAGE_WEIGHT = 0.30
UTILITY_CONFIDENCE_WEIGHT = 0.20
WEIGHT_BASE = 0.5
WEIGHT_PER_UTILITY = 1.5
MIN_LEARNED_WEIGHT = 0.2
MAX_LEARNED_WEIGHT = 3.0

# Plausible publication years (03-L5).
MIN_PUBLICATION_YEAR = 1800
MAX_PUBLICATION_YEAR = 2100
_YEAR_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")

# A sentence ends at . ! ? followed by whitespace and a capital, digit or
# bracket — not inside "0.05" and not after common abbreviations (03-L9).
_SENTENCE_BREAK_RE = re.compile(
    r"(?<=[.!?])(?<!\bal\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)(?<!\bFig\.)(?<!\bfig\.)"
    r"\s+(?=[A-Z0-9(\[\"'])"
)

_PROGRESS_LOG_EVERY_ARTICLES = 50


class AnalysisCancelled(Exception):
    """Raised by ``run_full_analysis`` when ``should_cancel`` returns True."""


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _coerce_year(value: Any) -> int | None:
    """A plausible 4-digit year from *value* ('2024-01-15', 'May 2020', 2019)."""
    if value is None or isinstance(value, bool):
        return None
    match = _YEAR_RE.search(str(value).strip())
    if not match:
        return None
    year = int(match.group(1))
    return year if MIN_PUBLICATION_YEAR <= year <= MAX_PUBLICATION_YEAR else None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _normalize_keyword(keyword: str) -> str:
    return " ".join(keyword.lower().split())


@functools.lru_cache(maxsize=4096)
def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Compiled pattern for a normalised keyword (03-L2).

    Lookarounds instead of ``\\b`` so keywords with non-word edges such as
    "GABA(A)" still match; a trailing ``*`` is a word-continuation wildcard.
    """
    is_wildcard = keyword.endswith("*") and len(keyword) > 1
    stem = keyword[:-1] if is_wildcard else keyword
    body = re.escape(stem) + (r"\w*" if is_wildcard else "")
    if re.search(r"\w", stem):
        return re.compile(rf"(?<!\w){body}(?!\w)")
    return re.compile(body)


def _scan_keyword(text_lower: str, keyword: str) -> int:
    """Occurrences of a normalised *keyword* in normalised *text_lower*."""
    return len(_keyword_pattern(keyword).findall(text_lower))


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BREAK_RE.split(text) if s.strip()]


class _TextHits:
    """Keyword hits in one text: normalised once, each keyword scanned once (03-M6)."""

    __slots__ = ("text", "normalized", "word_count", "_counts", "_sentences")

    def __init__(self, text: str) -> None:
        self.text = text
        self.normalized = _normalize_text(text)
        self.word_count = len(self.normalized.split())
        self._counts: dict[str, int] = {}
        self._sentences: list[_TextHits] | None = None

    def count(self, keyword: str) -> int:
        key = _normalize_keyword(keyword)
        if key not in self._counts:
            self._counts[key] = _scan_keyword(self.normalized, key) if key else 0
        return self._counts[key]

    def hits(self, keywords: Iterable[str]) -> tuple[int, list[str]]:
        matched: list[str] = []
        total = 0
        for keyword in keywords:
            count = self.count(keyword)
            if count:
                matched.append(keyword)
                total += count
        return total, matched

    @property
    def sentences(self) -> list[_TextHits]:
        if self._sentences is None:
            self._sentences = [_TextHits(s) for s in _split_sentences(self.text)]
        return self._sentences


def _count_keyword_hits(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    return _TextHits(text).hits(keywords)


def _keyword_frequency_score(hits: _TextHits, keywords: list[str]) -> float:
    """0-1 score based on how many keywords appear and how often."""
    if not keywords:
        return 0.0
    total, matched = hits.hits(keywords)
    coverage = len(matched) / len(keywords)
    density = min(total / max(hits.word_count, 1) * 50, 1.0)
    return 0.6 * coverage + 0.4 * density


def _cooccurrence_score(hits: _TextHits, keywords_a: list[str], keywords_b: list[str]) -> float:
    """Score based on co-occurrence of two keyword sets in the same text."""
    _, matched_a = hits.hits(keywords_a)
    _, matched_b = hits.hits(keywords_b)
    if not matched_a or not matched_b:
        return 0.0
    return min(len(matched_a), len(matched_b)) / max(len(keywords_a), len(keywords_b), 1)


def _directional_score(text: str) -> float:
    """Return -1 to +1 based on supportive vs contradictory language."""
    text_lower = _normalize_text(text)
    support = sum(1 for p in SUPPORT_PATTERNS if re.search(p, text_lower))
    negate = sum(1 for p in NEGATION_PATTERNS if re.search(p, text_lower))
    total = support + negate
    if total == 0:
        return 0.0
    return (support - negate) / total


def _keyword_windows(hits: _TextHits, keywords: list[str]) -> str:
    """Join the sentences of the text that mention at least one of *keywords*."""
    return " ".join(s.text for s in hits.sentences if s.hits(keywords)[0])


def _classify_direction(relevance: float, dir_score: float) -> EvidenceDirection:
    if relevance < NEUTRAL_RELEVANCE_CUTOFF:
        return EvidenceDirection.NEUTRAL
    if dir_score > DIRECTION_CUTOFF:
        return EvidenceDirection.SUPPORTS
    if dir_score < -DIRECTION_CUTOFF:
        return EvidenceDirection.CONTRADICTS
    return EvidenceDirection.TANGENTIAL


def _extract_key_excerpts(hits: _TextHits, keywords: list[str], max_excerpts: int = 3) -> list[str]:
    """Find sentences containing the most keywords."""
    scored = []
    for sentence in hits.sentences:
        if len(sentence.text) < 20:
            continue
        count, _ = sentence.hits(keywords)
        if count:
            scored.append((count, sentence.text))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_excerpts]]


# ---------------------------------------------------------------------------
# Weight learning
# ---------------------------------------------------------------------------
def _link_stats(links: list[EvidenceLink]) -> dict[str, float]:
    if not links:
        return {"coverage": 0.0, "mean_relevance": 0.0, "mean_confidence": 0.0,
                "direction_balance": 0.0}
    matched = [link for link in links if link.matched_keywords or link.relevance_score > 0]
    support = sum(1 for link in links if link.direction == EvidenceDirection.SUPPORTS)
    contradict = sum(1 for link in links if link.direction == EvidenceDirection.CONTRADICTS)
    return {
        "coverage": len(matched) / len(links),
        "mean_relevance": sum(link.relevance_score for link in links) / len(links),
        "mean_confidence": sum(link.confidence for link in links) / len(links),
        "direction_balance": (support - contradict) / len(links),
    }


def update_component_weights_from_keyword_analysis(
    session: AnalysisSession,
    *,
    min_weight: float = MIN_LEARNED_WEIGHT,
    max_weight: float = MAX_LEARNED_WEIGHT,
) -> dict[str, dict[str, float]]:
    """Update component weights from keyword-analysis evidence.

    Weight semantics:
    - matched, relevant terms rise above 1.0;
    - terms that do not appear in useful records decay below 1.0.

    The direction of the evidence never changes a weight: a search that
    favoured terms tied to supporting findings would bias the next retrieval
    towards confirmation (03-H5).  ``direction_balance`` is reported only.
    The session's components are updated in place (the GUI relies on it).
    """
    links_by_component: dict[str, list[EvidenceLink]] = {}
    for link in session.evidence_links:
        links_by_component.setdefault(link.component_id, []).append(link)

    stats: dict[str, dict[str, float]] = {}
    for component in session.hypothesis_components:
        link_stats = _link_stats(links_by_component.get(component.id, []))
        utility = (
            UTILITY_RELEVANCE_WEIGHT * link_stats["mean_relevance"]
            + UTILITY_COVERAGE_WEIGHT * link_stats["coverage"]
            + UTILITY_CONFIDENCE_WEIGHT * link_stats["mean_confidence"]
        )
        new_weight = max(min_weight, min(max_weight, WEIGHT_BASE + WEIGHT_PER_UTILITY * utility))
        component.weight = round(new_weight, 3)
        component.description = (
            f"{component.description.split(' | Keyword analysis:')[0]} | "
            f"Keyword analysis: utility={utility:.3f}, "
            f"coverage={link_stats['coverage']:.2f}, "
            f"relevance={link_stats['mean_relevance']:.2f}, "
            f"confidence={link_stats['mean_confidence']:.2f}, "
            f"direction={link_stats['direction_balance']:.2f}"
        )
        stats[component.id] = {
            "utility": round(utility, 6),
            **{key: round(value, 6) for key, value in link_stats.items()},
            "weight": component.weight,
        }
    return stats


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _other_keywords(
    component: HypothesisComponent, all_components: list[HypothesisComponent] | None,
) -> list[str]:
    return [kw for c in all_components or [] if c.id != component.id for kw in c.keywords]


def _score_text(
    article_id: str,
    component: HypothesisComponent,
    hits: _TextHits,
    other_keywords: list[str],
) -> EvidenceLink:
    kw_score = _keyword_frequency_score(hits, component.keywords) if hits.normalized else 0.0
    if kw_score <= 0:
        return EvidenceLink(
            article_id=article_id,
            component_id=component.id,
            direction=EvidenceDirection.NEUTRAL,
            relevance_score=0.0,
            confidence=0.0,
        )

    cooc = _cooccurrence_score(hits, component.keywords, other_keywords) if other_keywords else 0.0
    relevance = KEYWORD_RELEVANCE_WEIGHT * kw_score + COOCCURRENCE_RELEVANCE_WEIGHT * cooc
    dir_score = _directional_score(_keyword_windows(hits, component.keywords))

    return EvidenceLink(
        article_id=article_id,
        component_id=component.id,
        direction=_classify_direction(relevance, dir_score),
        relevance_score=relevance,
        confidence=min(kw_score + cooc, 1.0),
        key_excerpts=_extract_key_excerpts(hits, component.keywords),
        # Same word-boundary matching as the hit counts (03-L1).
        matched_keywords=hits.hits(component.keywords)[1],
    )


def _article_text(article: Article) -> str:
    return article.text_content or article.abstract or article.title


def score_article_against_component(
    article: Article,
    component: HypothesisComponent,
    all_components: list[HypothesisComponent] | None = None,
) -> EvidenceLink:
    """Score one article against one hypothesis component.

    Without a hit on one of the component's keywords the link is NEUTRAL with
    zero relevance.  Direction is read only from the sentences that mention
    the component's keywords, so each component gets its own direction.
    """
    return _score_text(
        article.id, component, _TextHits(_article_text(article)),
        _other_keywords(component, all_components),
    )


def compute_alignment(
    evidence_links: list[EvidenceLink],
    component: HypothesisComponent,
    articles: dict[str, Article],
) -> AlignmentResult:
    """Aggregate evidence links into an alignment score for one component."""
    comp_links = [el for el in evidence_links if el.component_id == component.id]
    if not comp_links:
        return AlignmentResult(
            component_id=component.id,
            component_label=component.label,
        )

    supporting = [el for el in comp_links if el.direction == EvidenceDirection.SUPPORTS]
    contradicting = [el for el in comp_links if el.direction == EvidenceDirection.CONTRADICTS]

    n = len(comp_links)
    support_score = (len(supporting) - len(contradicting)) / n if n else 0.0
    confidence = sum(el.confidence for el in comp_links) / n if n else 0.0

    return AlignmentResult(
        component_id=component.id,
        component_label=component.label,
        support_score=support_score,
        evidence_count=n,
        supporting_count=len(supporting),
        contradicting_count=len(contradicting),
        neutral_count=n - len(supporting) - len(contradicting),
        confidence=confidence,
        top_supporting=[el.article_id for el in sorted(
            supporting, key=lambda x: x.relevance_score, reverse=True)[:5]],
        top_contradicting=[el.article_id for el in sorted(
            contradicting, key=lambda x: x.relevance_score, reverse=True)[:3]],
    )


def _direction_balance(links: list[EvidenceLink]) -> float:
    """(supporting - contradicting) / n, the same measure as ``compute_alignment``."""
    if not links:
        return 0.0
    support = sum(1 for el in links if el.direction == EvidenceDirection.SUPPORTS)
    contradict = sum(1 for el in links if el.direction == EvidenceDirection.CONTRADICTS)
    return (support - contradict) / len(links)


def compute_trends(
    evidence_links: list[EvidenceLink],
    articles: dict[str, Article],
    component: HypothesisComponent,
) -> list[TrendPoint]:
    """Compute publication trend data for one component.

    ``avg_support_score`` is the year's direction balance (03-L3); years are
    coerced, so string years from saved corpora do not crash.
    """
    year_data: dict[int, list[EvidenceLink]] = {}
    for el in evidence_links:
        if el.component_id != component.id:
            continue
        art = articles.get(el.article_id)
        year = _coerce_year(art.year) if art else None
        if year is not None:
            year_data.setdefault(year, []).append(el)

    points = []
    cum_citations = 0
    for year in sorted(year_data):
        links = year_data[year]
        cum_citations += sum(
            int(getattr(articles.get(el.article_id), "citations", 0) or 0) for el in links
        )
        points.append(TrendPoint(
            year=year,
            article_count=len(links),
            avg_support_score=_direction_balance(links),
            cumulative_citations=cum_citations,
            component_id=component.id,
        ))
    return points


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------
def _article_from_dict(data: dict, art_id: str) -> Article:
    return Article(
        id=art_id,
        title=data.get("title", "") or "",
        authors=data.get("authors", []) or [],
        abstract=data.get("abstract", "") or "",
        full_text=data.get("full_text", "") or "",
        year=_coerce_year(data.get("year")),
        journal=data.get("journal", "") or "",
        doi=data.get("doi", "") or "",
        citations=data.get("citationCount", 0) or data.get("citations", 0) or 0,
        url=data.get("url", "") or "",
        pmid=data.get("pmid", "") or "",
        keywords=data.get("keywords", []) or [],
        has_full_text=bool(data.get("has_full_text", False)),
        open_access=bool(data.get("open_access", False)),
        full_text_url=data.get("full_text_url", "") or "",
    )


def _list_item_article(item: Any, idx: int) -> Article:
    if isinstance(item, Article):
        return item
    if isinstance(item, dict):
        raw_id = item.get("id") or item.get("doi") or item.get("pmid") or ""
        return _article_from_dict(item, str(raw_id))
    raise TypeError(
        "run_full_analysis expected articles to be a dict[str, Article] or list[Article]; "
        f"got list item {idx} of type {type(item).__name__}"
    )


def _unique_article_id(article: Article, idx: int, used: dict[str, Article]) -> str:
    art_id = article.id or f"article_{idx}"
    if art_id not in used:
        return art_id
    suffix = 2
    while f"{art_id}_{suffix}" in used:
        suffix += 1
    logger.warning(
        "Duplicate article ID %r (list item %d, %r) is kept as %s_%d",
        art_id, idx, article.title[:60], art_id, suffix,
    )
    return f"{art_id}_{suffix}"


def _articles_by_id(articles: Any) -> dict[str, Article]:
    """Coerce the GUI's dict or list input into ``dict[str, Article]`` (03-L4).

    Missing IDs fall back to the list index and duplicates get a suffix; the
    caller's Article objects are never modified.
    """
    if isinstance(articles, dict):
        return articles
    if not isinstance(articles, list):
        raise TypeError(
            "run_full_analysis expected articles to be dict[str, Article] or list[Article]; "
            f"got {type(articles).__name__}"
        )
    by_id: dict[str, Article] = {}
    for idx, item in enumerate(articles):
        article = _list_item_article(item, idx)
        art_id = _unique_article_id(article, idx, by_id)
        by_id[art_id] = article if art_id == article.id else replace(article, id=art_id)
    return by_id


def _score_all(
    articles: dict[str, Article],
    components: list[HypothesisComponent],
    progress_cb: Callable[[int, int], None] | None,
    should_cancel: Callable[[], bool] | None,
) -> list[EvidenceLink]:
    """Score every article against every component, one text scan per article."""
    other_keywords = {c.id: _other_keywords(c, components) for c in components}
    total = len(articles) * len(components)
    links: list[EvidenceLink] = []
    for index, (art_id, article) in enumerate(articles.items(), start=1):
        if should_cancel is not None and should_cancel():
            raise AnalysisCancelled(
                f"Keyword analysis cancelled after {index - 1} of {len(articles)} articles"
            )
        hits = _TextHits(_article_text(article))
        links.extend(_score_text(art_id, c, hits, other_keywords[c.id]) for c in components)
        if index % _PROGRESS_LOG_EVERY_ARTICLES == 0:
            logger.info("[%d/%d] Scoring article×component pairs...", len(links), total)
        if progress_cb:
            progress_cb(len(links), total)
    return links


def _log_alignment(alignment: AlignmentResult, label: str) -> None:
    logger.info("  %s: support=%.2f (%d supporting, %d contradicting, %d neutral)",
                label, alignment.support_score, alignment.supporting_count,
                alignment.contradicting_count, alignment.neutral_count)


def run_full_analysis(
    articles: dict[str, Article] | list[Any],
    components: list[HypothesisComponent],
    progress_cb: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> AnalysisSession:
    """Run keyword analysis on all articles × all components.

    ``progress_cb(done_pairs, total_pairs)`` is called after every article.
    ``should_cancel()`` is checked before every article; when it returns True
    ``AnalysisCancelled`` is raised (03-M6).  Returns a populated AnalysisSession.
    """
    articles = _articles_by_id(articles)
    session = AnalysisSession()
    session.hypothesis_components = components
    session.articles = articles
    session.total_articles_fetched = len(articles)

    logger.info("═" * 60)
    logger.info("KEYWORD ANALYSIS: %d articles × %d components", len(articles), len(components))
    logger.info("═" * 60)

    evidence_links = _score_all(articles, components, progress_cb, should_cancel)
    session.evidence_links = evidence_links
    session.total_articles_analyzed = len(articles)
    logger.info("Scored %d article×component pairs", len(evidence_links))

    for comp in components:
        alignment = compute_alignment(evidence_links, comp, articles)
        session.alignment_results.append(alignment)
        _log_alignment(alignment, comp.label)

    for comp in components:
        session.trend_data.extend(compute_trends(evidence_links, articles, comp))

    supporting_total = sum(a.supporting_count for a in session.alignment_results)
    contradicting_total = sum(a.contradicting_count for a in session.alignment_results)
    logger.info("═" * 60)
    logger.info("ANALYSIS COMPLETE: %d supporting, %d contradicting",
                supporting_total, contradicting_total)
    logger.info("═" * 60)

    return session
