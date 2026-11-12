"""Stage 1 helpers shared with ``gui.search_runner.run_gui_search``.

Calls async provider clients, normalizes results to a common schema
(``acquire.normalizers``), deduplicates, optionally runs citation snowballing,
and saves raw JSON.

Stage 1 runs live through ``gui.search_runner.run_gui_search`` (one query per
provider, additive corpus, cancel/progress hooks), which imports ``save_raw``,
``SNOWBALL_SOURCE``, ``_manifest_provider_entry`` and ``_snowball_expand`` from
this module. (A second, multi-query-per-provider orchestration path used to
live here as ``run_acquisition``; it had no production caller and was removed
— see ``reviews/stage1/agreed.md`` item 9.)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from acquire.normalizers import (
    normalize_elicit_paper,
    normalize_oa_paper,
    normalize_pm_paper,
    normalize_s2_paper,
    normalize_scopus_paper,
    normalize_wos_paper,
)
from acquire.semantic_scholar import SemanticScholarClient
from core.config import Config
from core.io import atomic_write_json

logger = logging.getLogger(__name__)

# Legacy underscore names: archive/hard_mode imports these from this module.
_normalize_elicit_paper = normalize_elicit_paper
_normalize_s2_paper = normalize_s2_paper
_normalize_oa_paper = normalize_oa_paper
_normalize_pm_paper = normalize_pm_paper
_normalize_wos_paper = normalize_wos_paper
_normalize_scopus_paper = normalize_scopus_paper


# ---------------------------------------------------------------------------
# Save raw results (atomic write)
# ---------------------------------------------------------------------------
def save_raw(path: Path, data: list[dict]) -> None:
    """Write JSON atomically and log the record count."""
    atomic_write_json(path, data)
    logger.info("Saved %d records to %s", len(data), path)



# ---------------------------------------------------------------------------
# Citation snowballing
# ---------------------------------------------------------------------------
def _select_snowball_seeds(corpus: list[dict], seed_limit: int) -> list[dict]:
    """Pick the top-cited papers with an S2 ID as snowball seeds."""
    seeds = [
        paper for paper in corpus
        if paper.get("s2_id") and paper.get("abstract") and paper.get("title")
    ]
    seeds.sort(
        key=lambda paper: (
            int(paper.get("citation_count") or 0),
            int(paper.get("year") or 0),
            len(paper.get("abstract") or ""),
        ),
        reverse=True,
    )
    return seeds[:max(0, int(seed_limit))]


async def _run_citation_snowball(config: Config, seed_papers: list[dict]) -> list[dict]:
    """Fetch citations + references for seed papers via S2 (parallel per-seed)."""
    if not seed_papers:
        return []

    relation_limit = max(1, int(config.citation_snowball_relation_limit))
    logger.info(
        "S2 snowball: expanding around %d seeds with %d citations and %d references each",
        len(seed_papers),
        relation_limit,
        relation_limit,
    )
    s2 = SemanticScholarClient(api_key=config.s2_api_key)
    expanded: list[dict] = []
    try:
        for index, seed in enumerate(seed_papers, start=1):
            paper_id = seed.get("s2_id")
            if not paper_id:
                continue
            try:
                citations, references = await asyncio.gather(
                    s2.get_citations(paper_id, limit=relation_limit),
                    s2.get_references(paper_id, limit=relation_limit),
                )
            except Exception as e:
                logger.warning(
                    "S2 snowball failed for %s (%d/%d): %s",
                    paper_id,
                    index,
                    len(seed_papers),
                    e,
                )
                continue

            expanded.extend(normalize_s2_paper(paper) for paper in citations)
            expanded.extend(normalize_s2_paper(paper) for paper in references)
            logger.info(
                "S2 snowball %d/%d: %s -> %d citations, %d references",
                index,
                len(seed_papers),
                paper_id,
                len(citations),
                len(references),
            )
            await asyncio.sleep(s2._delay)
    finally:
        await s2.close()

    return expanded


# ---------------------------------------------------------------------------
# Search manifest
# ---------------------------------------------------------------------------
SNOWBALL_SOURCE = "citation_snowball"


def _manifest_provider_entry(source: str, retrieved: int) -> dict:
    """Manifest entry for a source with no single query (the citation snowball)."""
    return {"source": source, "query": None, "hits": None, "retrieved": retrieved,
            "error": None, "truncated": None}


async def _snowball_expand(config: Config, corpus: list[dict]) -> list[dict] | None:
    """Snowball records (tagged ``citation_snowball``); None when there are no seeds."""
    seed_limit = max(0, int(config.citation_snowball_seed_limit))
    seed_papers = _select_snowball_seeds(corpus, seed_limit)
    if not seed_papers:
        return None
    logger.info("S2 snowball: selected %d seed papers from the initial corpus",
                len(seed_papers))
    return [
        {**paper, "search_method": SNOWBALL_SOURCE}
        for paper in await _run_citation_snowball(config, seed_papers)
    ]
