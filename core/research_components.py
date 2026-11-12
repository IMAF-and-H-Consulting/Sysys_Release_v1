"""Build hypothesis components from the loaded research configuration.

The config-loaded components (startup, project load, Reset) and the
components rebuilt at Search time come from one builder,
``analysis.query_builder.build_hypothesis_components``: same prefixed IDs and
the same keywords, so the Search-tab preview shows the query that is searched
and weights carry over by ID (03-M1, 03-M2).
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from analysis.query_builder import build_hypothesis_components
from core.models import HypothesisComponent


def _clean_text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _research_config(config: Any) -> dict[str, Any]:
    research = getattr(config, "research_config", {})
    return research if isinstance(research, dict) else {}


def _search_config(config: Any) -> Any:
    """``config.get_search_config()`` when the config has one, else research["search"]."""
    getter = getattr(config, "get_search_config", None)
    if callable(getter):
        return getter()
    return _research_config(config).get("search")


def _hypothesis_placeholder() -> str:
    """The ``Config.hypothesis_text`` default, which is not a real question."""
    from core.config import Config

    default = next(f.default for f in fields(Config) if f.name == "hypothesis_text")
    return _clean_text(default)


def resolve_research_question(research: dict[str, Any], hypothesis_text: Any = "") -> str:
    """The configured research question, or "" when none is set (03-L6).

    Order: ``natural_language_question``, ``relevance.research_question``,
    the config's ``hypothesis_text``, then the research-config keys
    ``hypothesis_text``, ``hypothesis``, ``question`` and ``description``.
    The ``Config`` placeholder hypothesis is skipped wherever it appears.
    """
    research = research if isinstance(research, dict) else {}
    relevance_raw = research.get("relevance")
    relevance = relevance_raw if isinstance(relevance_raw, dict) else {}
    placeholder = _hypothesis_placeholder()
    for value in (
        research.get("natural_language_question"),
        relevance.get("research_question"),
        hypothesis_text,
        research.get("hypothesis_text"),
        research.get("hypothesis"),
        research.get("question"),
        research.get("description"),
    ):
        text = _clean_text(value)
        if text and text != placeholder:
            return text
    return ""


def build_hypothesis_components_from_config(config: Any) -> list[HypothesisComponent]:
    """Create GUI/search components from current-schema config values.

    Uses the same builder and inputs as the Search-time rebuild, with the
    configured research question as the question text.
    """
    research = _research_config(config)
    question = resolve_research_question(research, getattr(config, "hypothesis_text", ""))
    return build_hypothesis_components(
        hypothesis=question,
        research_config={**research, "search": _search_config(config)},
        co_occurring_conditions=getattr(config, "co_occurring_conditions", []) or [],
        query=question,
    )
