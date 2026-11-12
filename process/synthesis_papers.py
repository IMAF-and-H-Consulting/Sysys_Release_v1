"""Shared paper accessors for Stage 6 synthesis artifacts.

Every Stage 6 module reads paper identity, score, direction and claim terms
through these helpers, so the explanation, the evidence network, the source
citations and the narrative prompts classify a paper the same way.
"""
from __future__ import annotations

import re
from collections import Counter

from core.claims import claim_direction, claim_value
from core.doi import normalize_doi

_claim_value = claim_value

# Identifier fields in priority order (shared by source keys and joins).
IDENTITY_FIELDS = (
    "doi", "pmid", "pmc", "pmcid", "s2_id", "oa_id", "wos_id", "scopus_id", "id", "article_id",
)


def _as_list(value) -> list:
    """Normalize common scalar/list claim values to a clean list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _paper_id(paper: dict, fallback: str = "") -> str:
    for key in ("id", "article_id", "doi", "pmid", "pmcid", "paperId", "title"):
        value = paper.get(key)
        if value:
            return str(value)
    return fallback


def _paper_title(paper: dict) -> str:
    return str(paper.get("title") or paper.get("article_title") or "Untitled")


def _paper_score(paper: dict) -> float:
    for key in (
        "relevance_score",
        "score",
        "llm_relevance",
        "support_score",
        "combined_score",
    ):
        value = paper.get(key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    if _claim_value(paper, "supports_primary_question", False):
        return 3.0
    return 0.0


def _paper_direction(paper: dict) -> str:
    """supports / contradicts / neutral / unknown, from Stage 3's nested claims."""
    return claim_direction(paper)


def _paper_quote(paper: dict) -> str:
    for key in ("key_quote", "quoted_span", "quote", "excerpt"):
        value = _claim_value(paper, key, "")
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    spans = _as_list(paper.get("evidence_spans") or _claim_value(paper, "evidence_spans", []))
    return spans[0] if spans else ""


def _top_terms(papers: list[dict], field: str, limit: int = 12) -> list[dict]:
    """Count producer-supplied canonical labels exactly.

    Stage 3 owns categorical normalization.  Keeping this as an exact counter is
    intentional: a stale/legacy artifact remains visibly stale instead of being
    silently repaired in a report.
    """
    counter: Counter[str] = Counter()
    for paper in papers:
        for value in _as_list(_claim_value(paper, field, [])):
            counter[value] += 1
    return [
        {"term": term, "count": count}
        for term, count in counter.most_common(limit)
    ]


def _paper_rank_key(p: dict) -> tuple:
    """Shared sort key: supporting papers first, then score, citations, year, title."""
    return (
        _paper_direction(p) == "supports",
        _paper_score(p),
        int(p.get("citations") or p.get("citation_count") or 0),
        str(p.get("year") or ""),
        _paper_title(p).lower(),
    )


def _rank_papers(papers: list[dict], limit: int = 8) -> list[dict]:
    ranked = sorted(papers, key=_paper_rank_key, reverse=True)
    rows = []
    for paper in ranked[:limit]:
        rows.append({
            "id": _paper_id(paper),
            "title": _paper_title(paper),
            "year": paper.get("year"),
            "score": _paper_score(paper),
            "direction": _paper_direction(paper),
            "citations": paper.get("citations") or paper.get("citation_count") or 0,
            "quote": _paper_quote(paper),
            "findings": _as_list(_claim_value(paper, "findings", []))[:3],
            "mechanisms": _as_list(_claim_value(paper, "mechanisms", []))[:5],
        })
    return rows


def _source_identity_key(paper: dict) -> str:
    """Return a stable key for collapsing source citations."""
    for key in IDENTITY_FIELDS:
        value = paper.get(key)
        if value:
            text = normalize_doi(value) if key == "doi" else str(value).strip().lower()
            if text:
                return f"{key}:{text}"
    title = re.sub(r"\W+", " ", _paper_title(paper).lower()).strip()
    year = str(paper.get("year") or "nd")[:4]
    return f"title:{title}|year:{year}"
