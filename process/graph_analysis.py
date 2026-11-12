"""Configurable condition–entity co-occurrence analysis.

Patterns come from graph_analysis in the project research configuration.
The default has no domain-specific vocabulary. Select defaults.pattern_pack:
neuroscience to opt into the shipped legacy vocabulary. Legacy structural_regions
and functional_networks categories remain supported alongside neutral entities.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Called periodically inside long loops; the caller's check raises to stop.
CancelCheck = Callable[[], None]
# Permutations between cancellation checks.
CANCEL_CHECK_INTERVAL = 50


# Category names used by older saved projects remain valid. New domains can
# use the neutral `entities` target category.
from process.domain_patterns import (
    condition_patterns,
    load_pack,
    mapping,
    pattern_map,
    research_config,
)


def load_patterns(config=None) -> dict[str, dict[str, list[str]]]:
    """Load validated project patterns; neuroscience requires explicit opt-in."""
    custom = mapping(research_config(config).get("graph_analysis", {}), "graph_analysis")
    defaults = load_pack(config).get("graph_analysis", {})
    result = {"conditions": condition_patterns(config)}
    for category in ("entities", "structural_regions", "functional_networks"):
        result[category] = pattern_map(custom.get(category, defaults.get(category, {})),
                                       f"graph_analysis.{category}")
    return result


def network_artifact_path(directory):
    """Prefer the neutral filename, while still opening older saved reviews."""
    from pathlib import Path
    directory = Path(directory)
    current = directory / "condition_networks.json"
    return current if current.exists() else directory / "autism_condition_networks.json"


def _targets(entities):
    return list(dict.fromkeys(
        entity for category, names in entities.items() if category != "conditions"
        for entity in names
    ))


def _target_type(name, patterns):
    for category, kind in (("entities", "entity"), ("structural_regions", "structural"),
                           ("functional_networks", "functional")):
        if name in patterns.get(category, {}):
            return kind
    return "entity"


# ── Entity extraction ─────────────────────────────────────────────────────


def extract_entities(
    text: str,
    patterns: dict[str, dict[str, list[str]]],
) -> dict[str, list[str]]:
    """Extract named entities from text using the pattern dictionaries.

    Returns a dict with keys ``conditions``, ``structural_regions``,
    ``functional_networks``, each mapping to a list of matched entity
    names found in the text.
    """
    results: dict[str, list[str]] = {category: [] for category in patterns}
    results.setdefault("conditions", [])
    for category, entity_dict in patterns.items():
        if not isinstance(entity_dict, dict):
            continue
        for entity_name, regexes in entity_dict.items():
            if not isinstance(regexes, list):
                continue
            for pat in regexes:
                try:
                    if re.search(pat, text, re.IGNORECASE):
                        results[category].append(entity_name)
                        break
                except (re.error, TypeError) as exc:
                    raise ValueError(f"Invalid pattern for {category}.{entity_name}: {exc}") from exc
    return results


def _paper_text(paper: dict) -> str:
    return " ".join(filter(None, [
        paper.get("title", ""),
        paper.get("abstract", ""),
    ]))


def _no_entities(patterns: dict) -> dict[str, list[str]]:
    return {category: [] for category in ("conditions", *patterns)}


def extract_corpus_entities(
    corpus: list[dict],
    patterns: dict[str, dict[str, list[str]]],
) -> list[dict[str, list[str]]]:
    """Entities of every paper, extracted once and shared by the edge builder
    and the permutation threshold (audit 12-L5). Papers without text get no
    entities."""
    return [
        extract_entities(text, patterns) if text.strip() else _no_entities(patterns)
        for text in (_paper_text(paper) for paper in corpus)
    ]


def build_cooccurrence_edges(
    corpus: list[dict],
    patterns: dict[str, dict[str, list[str]]],
    entities: list[dict[str, list[str]]] | None = None,
) -> list[dict]:
    """Build condition–entity co-occurrence edges from a corpus.

    For each article, extract entities and create edges between every
    condition and every configured target entity that
    co-occur in the same text.  Returns a list of edge dicts with
    ``source``, ``target``, ``weight`` (co-occurrence count), and
    ``source_type`` / ``target_type``.  ``entities`` may carry the output of
    ``extract_corpus_entities`` to avoid a second regex pass.
    """
    edge_counts: dict[tuple[str, str], int] = {}
    edge_types: dict[tuple[str, str], tuple[str, str]] = {}
    if entities is None:
        entities = extract_corpus_entities(corpus, patterns)

    for paper_entities in entities:
        conditions = paper_entities["conditions"]
        regions = _targets(paper_entities)

        for cond in conditions:
            for region in regions:
                key = (cond, region)
                edge_counts[key] = edge_counts.get(key, 0) + 1
                if key not in edge_types:
                    edge_types[key] = ("condition", _target_type(region, patterns))

    edges = []
    for (source, target), weight in edge_counts.items():
        src_type, tgt_type = edge_types.get((source, target), ("condition", "region"))
        edges.append({
            "source": source,
            "target": target,
            "weight": weight,
            "source_type": src_type,
            "target_type": tgt_type,
        })

    return sorted(edges, key=lambda e: e["weight"], reverse=True)


# ── Data-driven edge thresholding ────────────────────────────────────────

def compute_edge_threshold(
    corpus: list[dict],
    patterns: dict[str, dict[str, list[str]]],
    n_permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    entities: list[dict[str, list[str]]] | None = None,
    check_cancel: CancelCheck | None = None,
) -> float:
    """Compute a significance threshold for co-occurrence edge weights.

    Builds a null distribution by permuting condition labels across papers,
    then returns the (1−α) percentile of the max-weight distribution.
    Edges with weight > threshold are statistically significant.

    Parameters
    ----------
    corpus : list of paper dicts (must have 'title' and/or 'abstract')
    patterns : entity pattern dictionary from load_patterns()
    n_permutations : number of shuffles for the null distribution
    alpha : significance level (default 0.05 → 95th percentile)
    seed : random seed for reproducibility
    entities : optional per-paper entities from ``extract_corpus_entities``
    check_cancel : optional callable, run every ``CANCEL_CHECK_INTERVAL``
        permutations; it stops the loop by raising

    Returns
    -------
    float : weight threshold (edges above this are significant at level α)
    """
    import random as _random

    if not corpus:
        return 0

    if entities is None:
        entities = extract_corpus_entities(corpus, patterns)
    paper_conditions = [paper["conditions"] for paper in entities]
    paper_regions = [_targets(paper) for paper in entities]

    if not any(paper_conditions) or not any(paper_regions):
        return 0

    rng = _random.Random(seed)
    null_max_weights: list[int] = []

    for iteration in range(n_permutations):
        if check_cancel is not None and iteration % CANCEL_CHECK_INTERVAL == 0:
            check_cancel()
        # Shuffle condition labels across papers
        shuffled_conditions = list(paper_conditions)
        rng.shuffle(shuffled_conditions)

        # Rebuild co-occurrence counts with shuffled conditions
        edge_counts: dict[tuple[str, str], int] = {}
        for conds, regions in zip(shuffled_conditions, paper_regions):
            for c in conds:
                for r in regions:
                    key = (c, r)
                    edge_counts[key] = edge_counts.get(key, 0) + 1

        max_w = max(edge_counts.values()) if edge_counts else 0
        null_max_weights.append(max_w)

    if not null_max_weights:
        return 0

    # Threshold at (1 - alpha) percentile of null max-weight distribution
    null_max_weights.sort()
    idx = int(len(null_max_weights) * (1 - alpha))
    idx = min(idx, len(null_max_weights) - 1)
    return null_max_weights[idx]


# ── Community detection ──────────────────────────────────────────────────

LOUVAIN_UNAVAILABLE = "networkx or python-louvain not installed"


def detect_communities(edges: list[dict], seed: int | None = None) -> dict:
    """Louvain community detection on the co-occurrence graph.

    Parameters
    ----------
    edges : list of dicts with 'source', 'target', 'weight'
    seed : Louvain ``random_state``; pass the stage seed so communities and
        modularity are reproducible (audit 12-M3). None draws from numpy's
        global RNG.

    Returns
    -------
    dict with:
        communities : {node_name: community_id}
        modularity : float (None when unavailable)
        n_communities : int (None when unavailable)
        unavailable : reason, only when the libraries are missing — so a
            missing install is not mistaken for "no structure" (audit 12-M7)
    """
    if not edges:
        return {"communities": {}, "modularity": 0.0, "n_communities": 0}

    try:
        import community as community_louvain
        import networkx as nx
    except ImportError:
        logger.warning("%s — community detection unavailable", LOUVAIN_UNAVAILABLE)
        return {"communities": {}, "modularity": None, "n_communities": None,
                "unavailable": LOUVAIN_UNAVAILABLE}

    G = nx.Graph()
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        w = edge.get("weight", 1)
        if src and tgt:
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] += w
            else:
                G.add_edge(src, tgt, weight=w)

    if G.number_of_nodes() == 0:
        return {"communities": {}, "modularity": 0.0, "n_communities": 0}

    partition = community_louvain.best_partition(G, weight="weight", random_state=seed)
    modularity = community_louvain.modularity(partition, G, weight="weight")
    n_communities = len(set(partition.values()))

    return {
        "communities": partition,
        "modularity": round(float(modularity), 6),
        "n_communities": n_communities,
    }


# ── Temporal network analysis ────────────────────────────────────────────

def temporal_network_analysis(
    corpus: list[dict],
    patterns: dict[str, dict[str, list[str]]],
    window_years: int = 5,
) -> list[dict]:
    """Build co-occurrence networks for successive time windows.

    Parameters
    ----------
    corpus : list of paper dicts (must have 'year', 'title', 'abstract')
    patterns : entity pattern dictionary from load_patterns()
    window_years : width of each time bin in years

    Returns
    -------
    list of dicts, one per window, each with:
        window_start, window_end, n_papers, n_edges, top_edges,
        edge_set (frozenset for Jaccard), jaccard_vs_prev
    """
    if not corpus or window_years < 1:
        return []

    # Extract years and find range
    papers_with_year = []
    for p in corpus:
        y = p.get("year")
        if y is not None:
            try:
                y = int(y)
                if 1900 <= y <= 2100:
                    papers_with_year.append((y, p))
            except (ValueError, TypeError):
                pass

    if not papers_with_year:
        return []

    min_year = min(y for y, _ in papers_with_year)
    max_year = max(y for y, _ in papers_with_year)

    # Build windows
    windows = []
    start = min_year
    while start <= max_year:
        end = start + window_years - 1
        window_papers = [p for y, p in papers_with_year if start <= y <= end]

        if window_papers:
            edges = build_cooccurrence_edges(window_papers, patterns)
            edge_set = frozenset((e["source"], e["target"]) for e in edges)

            windows.append({
                "window_start": start,
                "window_end": end,
                "n_papers": len(window_papers),
                "n_edges": len(edges),
                "top_edges": edges[:10],
                "edge_set": edge_set,
            })

        start += window_years

    # Compute Jaccard similarity between consecutive windows
    for i, w in enumerate(windows):
        if i == 0:
            w["jaccard_vs_prev"] = None
        else:
            prev_set = windows[i - 1]["edge_set"]
            curr_set = w["edge_set"]
            if prev_set or curr_set:
                jaccard = len(prev_set & curr_set) / len(prev_set | curr_set)
            else:
                jaccard = 1.0
            w["jaccard_vs_prev"] = round(jaccard, 4)

    # Remove frozensets before returning (not JSON-serializable)
    for w in windows:
        w.pop("edge_set", None)

    return windows
