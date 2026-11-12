"""Build reviewable Evidence Table rows from live analysis or claim artifacts."""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from core.claims import claim_list, claim_value
from process.human_relevance_review import article_id as _stable_article_id

logger = logging.getLogger(__name__)


def _field(value: Any, name: str, default: Any = "") -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


_claim_value = claim_value


def _direction_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    text = str(raw or "neutral").strip().lower()
    if "contradict" in text or text in {"against", "opposes", "negative"}:
        return "contradicts"
    if "support" in text or text in {"for", "positive"}:
        return "supports"
    if "tangential" in text:
        return "tangential"
    return "neutral"


def _paper_direction(paper: Mapping[str, Any]) -> str:
    for key in ("direction", "llm_direction", "evidence_direction"):
        value = _claim_value(paper, key)
        if value not in (None, ""):
            return _direction_value(value)
    if _claim_value(paper, "supports_primary_question", False) is True:
        return "supports"
    return "neutral"


def _authors_text(authors: Any) -> str:
    if isinstance(authors, list):
        return ", ".join(str(author) for author in authors if author)
    return str(authors or "")


def _claim_excerpt(paper: Mapping[str, Any]) -> str:
    claims = paper.get("claims")
    claims = claims if isinstance(claims, Mapping) else {}
    key_quote = claims.get("key_quote") or paper.get("relevance_quote")
    if key_quote:
        return str(key_quote).strip()
    findings = claims.get("findings")
    if isinstance(findings, list) and findings:
        return str(findings[0]).strip()
    if isinstance(findings, str) and findings.strip():
        return findings.strip()
    return str(paper.get("abstract") or paper.get("tldr") or "").strip()[:800]


def build_claim_evidence_rows(papers: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Convert extracted paper claims into rows suitable for the GUI table."""
    rows: list[dict] = []
    for index, paper in enumerate(papers):
        if not isinstance(paper, Mapping):
            continue
        claims = paper.get("claims")
        claims = claims if isinstance(claims, Mapping) else {}
        title = str(paper.get("title") or f"Paper {index + 1}")
        article_id = _stable_article_id(dict(paper), index)
        mechanisms = claims.get("mechanisms")
        mechanisms = mechanisms if isinstance(mechanisms, list) else []
        conditions = claims.get("conditions")
        conditions = conditions if isinstance(conditions, list) else []
        score = paper.get("relevance_score")
        if score in (None, ""):
            score = paper.get("relevance_confidence", 0.0)
        try:
            numeric_score = float(score or 0.0)
        except (TypeError, ValueError):
            numeric_score = 0.0

        rows.append({
            "article_id": article_id,
            "title": title,
            "component": "Primary question",
            "score": numeric_score,
            "direction": _paper_direction(paper),
            "excerpt": _claim_excerpt(paper),
            "year": paper.get("year"),
            "doi": str(paper.get("doi") or ""),
            "pmid": str(paper.get("pmid") or ""),
            "authors": _authors_text(paper.get("authors")),
            "source": str(paper.get("source") or "claims artifact"),
            "mechanisms": [str(value) for value in mechanisms if value],
            "conditions": [str(value) for value in conditions if value],
            "key_entities": claim_list(paper, "key_entities"),
            "matched_keywords": [],
        })
    return rows


def build_session_evidence_rows(
    evidence_links: Iterable[Any],
    *,
    articles: Mapping[str, Any] | None = None,
    components: Iterable[Any] | None = None,
) -> list[dict]:
    """Convert live keyword-analysis EvidenceLink objects into table rows."""
    articles = articles or {}
    component_labels = {
        str(_field(component, "id", "")): str(
            _field(component, "label", "") or _field(component, "id", "")
        )
        for component in (components or [])
    }
    rows: list[dict] = []
    for link in evidence_links or []:
        article_id = str(_field(link, "article_id", ""))
        component_id = str(_field(link, "component_id", ""))
        article = articles.get(article_id, {})
        excerpts = _field(link, "key_excerpts", []) or []
        if isinstance(excerpts, str):
            excerpts = [excerpts]
        keywords = _field(link, "matched_keywords", []) or []
        if isinstance(keywords, str):
            keywords = [keywords]
        try:
            score = float(getattr(link, "combined_score", _field(link, "relevance_score", 0.0)))
        except (TypeError, ValueError):
            score = 0.0
        direction = getattr(link, "combined_direction", _field(link, "direction", "neutral"))
        rows.append({
            "article_id": article_id,
            "title": str(_field(article, "title", "") or article_id),
            "component": component_labels.get(component_id, component_id or "Primary question"),
            "score": score,
            "direction": _direction_value(direction),
            "excerpt": str(excerpts[0]).strip() if excerpts else "",
            "year": _field(article, "year", None),
            "doi": str(_field(article, "doi", "") or ""),
            "pmid": str(_field(article, "pmid", "") or ""),
            "authors": _authors_text(_field(article, "authors", [])),
            "source": "live keyword analysis",
            "mechanisms": [],
            "conditions": [],
            "key_entities": [],
            "matched_keywords": [str(value) for value in keywords if value],
        })
    return rows


def _load_json_rows(path: Path) -> list[Mapping[str, Any]]:
    """Rows of a claims artifact; [] (logged) when it cannot be read (15-L9)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("Evidence table: cannot read %s (%s); trying the next source.", path, exc)
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, Mapping):
        for key in ("papers", "claims", "results", "articles"):
            rows = data.get(key)
            if isinstance(rows, list):
                return rows
    return []


def load_evidence_rows(
    config: Any,
    *,
    session: Any = None,
    articles: Mapping[str, Any] | None = None,
    components: Iterable[Any] | None = None,
) -> tuple[list[dict], str]:
    """Load the best available evidence source and report its provenance."""
    live_links = getattr(session, "evidence_links", None) if session is not None else None
    if live_links:
        rows = build_session_evidence_rows(
            live_links,
            articles=articles,
            components=components,
        )
        return rows, "live keyword analysis"

    claims_dir = Path(config.claims_dir)
    for name in ("claims_filtered.json", "claims.json"):
        path = claims_dir / name
        if not path.exists():
            continue
        rows = build_claim_evidence_rows(_load_json_rows(path))
        if rows:
            return rows, str(path)
    return [], ""
