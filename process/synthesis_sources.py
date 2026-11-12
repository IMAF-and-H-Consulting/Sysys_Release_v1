"""Source-citation map and evidence network for Stage 6 synthesis.

``build_synthesis_source_citations`` assigns the citation keys used in the
narratives, ``synthesis_sources.md/json`` and ``synthesis_sources.bib``;
``build_evidence_network`` links the top papers of each cluster to their
terms and methods.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from process.synthesis_papers import (
    _as_list,
    _claim_value,
    _paper_direction,
    _paper_id,
    _paper_rank_key,
    _paper_score,
    _paper_title,
    _source_identity_key,
)


def _format_authors_short(authors) -> str:
    if isinstance(authors, list):
        clean = [str(author).strip() for author in authors if str(author).strip()]
    elif authors:
        clean = [part.strip() for part in str(authors).split(";") if part.strip()]
    else:
        clean = []
    if not clean:
        return "Unknown authors"
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{clean[0]} et al."


def _format_source_reference(source: dict) -> str:
    authors = _format_authors_short(source.get("authors"))
    year = source.get("year") or "n.d."
    title = source.get("title") or "Untitled"
    venue = source.get("journal") or source.get("venue") or ""
    doi = source.get("doi") or ""
    parts = [f"{authors} ({year}). {title}."]
    if venue:
        parts.append(str(venue).rstrip(".") + ".")
    if doi:
        parts.append(f"doi:{doi}")
    return " ".join(parts)


def build_synthesis_source_citations(clusters: list[dict]) -> dict:
    """Build a source-citation map for the exact papers used in synthesis."""
    from process.export import assign_citation_keys

    sources_by_identity: dict[str, dict] = {}
    cluster_source_identities: dict[str, list[str]] = defaultdict(list)

    for cluster in clusters:
        cluster_id = str(cluster.get("cluster_id"))
        for paper in cluster.get("papers", []) or []:
            identity = _source_identity_key(paper)
            if identity not in cluster_source_identities[cluster_id]:
                cluster_source_identities[cluster_id].append(identity)

            if identity not in sources_by_identity:
                sources_by_identity[identity] = {
                    "identity": identity,
                    "id": _paper_id(paper),
                    "article_id": paper.get("article_id") or paper.get("id") or "",
                    "title": _paper_title(paper),
                    "authors": paper.get("authors") or [],
                    "year": paper.get("year"),
                    "journal": paper.get("journal") or paper.get("venue") or "",
                    "venue": paper.get("venue") or paper.get("journal") or "",
                    "doi": paper.get("doi") or "",
                    "pmid": paper.get("pmid") or "",
                    "pmc": paper.get("pmc") or paper.get("pmcid") or "",
                    "s2_id": paper.get("s2_id") or paper.get("paperId") or "",
                    "source": paper.get("source") or "",
                    "url": paper.get("url") or "",
                    "abstract": paper.get("abstract") or "",
                    "relevance_score": _paper_score(paper),
                    "direction": _paper_direction(paper),
                    "cluster_ids": [cluster_id],
                }
            elif cluster_id not in sources_by_identity[identity]["cluster_ids"]:
                sources_by_identity[identity]["cluster_ids"].append(cluster_id)

    source_records = list(sources_by_identity.values())

    keyed_sources = []
    citation_key_by_identity: dict[str, str] = {}
    for citation_key, source in assign_citation_keys(source_records):
        row = dict(source)
        row["citation_key"] = citation_key
        row["reference"] = _format_source_reference(row)
        keyed_sources.append(row)
        citation_key_by_identity[row["identity"]] = citation_key

    cluster_rows = []
    for cluster in clusters:
        cluster_id = str(cluster.get("cluster_id"))
        source_keys = [
            citation_key_by_identity[identity]
            for identity in cluster_source_identities.get(cluster_id, [])
            if identity in citation_key_by_identity
        ]
        cluster_rows.append({
            "cluster_id": cluster.get("cluster_id"),
            "paper_count": cluster.get("paper_count", len(cluster.get("papers", []) or [])),
            "mechanisms": cluster.get("mechanisms", [])[:12],
            "conditions": cluster.get("conditions", [])[:12],
            "source_citation_keys": source_keys,
        })

    return {
        "schema_version": 1,
        "kind": "synthesis_source_citations",
        "source_count": len(keyed_sources),
        "cluster_count": len(cluster_rows),
        "clusters": cluster_rows,
        "sources": keyed_sources,
    }


def render_synthesis_source_citations_markdown(citations: dict) -> str:
    """Render synthesis source citations as Markdown grouped by cluster."""
    lines = [
        "# Sources Used for Synthesis",
        "",
        "These are the source papers that directly entered Stage 6 clustering, "
        "cluster narratives, and the synthesis abstract.",
        "",
        "Use the bracketed citation keys with `synthesis_sources.bib` when "
        "writing or revising the synthesis.",
        "",
        "## Summary",
        "",
        f"- Unique cited sources: {citations.get('source_count', 0)}",
        f"- Clusters represented: {citations.get('cluster_count', 0)}",
        "",
        "## Cluster Source Map",
        "",
    ]

    source_by_key = {
        source.get("citation_key"): source
        for source in citations.get("sources", [])
        if source.get("citation_key")
    }

    for cluster in citations.get("clusters", []):
        cid = cluster.get("cluster_id")
        lines.append(f"### Cluster {cid}")
        terms = []
        terms.extend(cluster.get("mechanisms") or [])
        terms.extend(cluster.get("conditions") or [])
        if terms:
            lines.append(f"Key terms: {', '.join(terms[:10])}")
        keys = cluster.get("source_citation_keys") or []
        if not keys:
            lines.append("- No source papers recorded for this cluster.")
        for key in keys:
            source = source_by_key.get(key, {})
            title = source.get("title") or "Untitled"
            year = source.get("year") or "n.d."
            doi = source.get("doi") or ""
            suffix = f" doi:{doi}" if doi else ""
            lines.append(f"- [{key}] {title} ({year}).{suffix}")
        lines.append("")

    lines.extend(["## Reference List", ""])
    for source in citations.get("sources", []):
        key = source.get("citation_key") or "unknown"
        lines.append(f"- [{key}] {source.get('reference') or _format_source_reference(source)}")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_evidence_network(
    clusters: list[dict],
    *,
    top_papers_per_cluster: int = 8,
    max_terms_per_type: int = 80,
) -> dict:
    """Build a generalized evidence network from clustered papers."""
    nodes: dict[str, dict] = {}
    edge_weights: dict[tuple[str, str, str], float] = defaultdict(float)
    edge_counts: Counter[tuple[str, str, str]] = Counter()
    term_counts: dict[str, Counter[str]] = {
        "mechanism": Counter(),
        "condition": Counter(),
        "entity": Counter(),
        "finding": Counter(),
        "method": Counter(),
    }

    def norm_id(kind: str, label: str) -> str:
        clean = " ".join(str(label).split()).lower()
        clean = "".join(ch if ch.isalnum() else "_" for ch in clean).strip("_")
        return f"{kind}:{clean or 'unknown'}"

    def add_node(node_id: str, label: str, node_type: str, **metadata) -> None:
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "label": label,
                "type": node_type,
                "count": 0,
                "metadata": {},
            }
        nodes[node_id]["count"] += int(metadata.pop("count", 1) or 1)
        nodes[node_id]["metadata"].update({k: v for k, v in metadata.items() if v not in (None, "")})

    def add_edge(source: str, target: str, edge_type: str, weight: float = 1.0) -> None:
        if source == target:
            return
        key = (source, target, edge_type)
        edge_weights[key] += float(weight or 1.0)
        edge_counts[key] += 1

    ranked_per_cluster = [
        sorted(
            cluster.get("papers", []) or [],
            key=_paper_rank_key,
            reverse=True,
        )[:top_papers_per_cluster]
        for cluster in clusters
    ]

    for cluster, ranked_originals in zip(clusters, ranked_per_cluster):
        cid = str(cluster.get("cluster_id"))
        cluster_id = f"cluster:{cid}"
        add_node(cluster_id, f"Cluster {cid}", "cluster", count=cluster.get("paper_count", 1))
        papers = cluster.get("papers", []) or []
        for paper in papers:
            for kind, field in (
                ("mechanism", "mechanisms"),
                ("condition", "conditions"),
                ("entity", "key_entities"),
                ("finding", "findings"),
            ):
                for value in _as_list(_claim_value(paper, field, [])):
                    term_counts[kind][value] += 1
            methodology = _claim_value(paper, "methodology", "")
            if methodology:
                for token in str(methodology).replace("/", " ").split(",")[:3]:
                    token = token.strip()
                    if token:
                        term_counts["method"][token] += 1

        for paper in ranked_originals:
            paper_id = norm_id("paper", _paper_id(paper))
            add_node(
                paper_id,
                _paper_title(paper),
                "paper",
                count=1,
                year=paper.get("year"),
                score=_paper_score(paper),
                direction=_paper_direction(paper),
            )
            add_edge(paper_id, cluster_id, "belongs_to", 1.0)

    allowed_terms = {
        kind: {term for term, _ in counter.most_common(max_terms_per_type)}
        for kind, counter in term_counts.items()
    }

    for cluster, ranked_originals in zip(clusters, ranked_per_cluster):
        cid = str(cluster.get("cluster_id"))
        cluster_id = f"cluster:{cid}"
        for paper in ranked_originals:
            pid = norm_id("paper", _paper_id(paper))
            direction = _paper_direction(paper)
            if direction in ("supports", "contradicts"):
                add_edge(pid, cluster_id, direction, max(_paper_score(paper), 1.0))
            for kind, field, edge_type in (
                ("mechanism", "mechanisms", "mentions"),
                ("condition", "conditions", "mentions"),
                ("entity", "key_entities", "mentions"),
                ("finding", "findings", "reports"),
            ):
                for value in _as_list(_claim_value(paper, field, [])):
                    if value not in allowed_terms[kind]:
                        continue
                    term_id = norm_id(kind, value)
                    if term_id not in nodes:
                        add_node(term_id, value, kind, count=term_counts[kind][value])
                    add_edge(pid, term_id, edge_type, 1.0)
                    add_edge(cluster_id, term_id, "cluster_theme", 1.0)
            methodology = _claim_value(paper, "methodology", "")
            for value in [v.strip() for v in str(methodology).replace("/", " ").split(",")[:3] if v.strip()]:
                if value not in allowed_terms["method"]:
                    continue
                method_id = norm_id("method", value)
                if method_id not in nodes:
                    add_node(method_id, value, "method", count=term_counts["method"][value])
                add_edge(pid, method_id, "uses_method", 1.0)

    edges = [
        {
            "source": source,
            "target": target,
            "type": edge_type,
            "weight": round(weight, 3),
            "evidence_count": edge_counts[(source, target, edge_type)],
        }
        for (source, target, edge_type), weight in edge_weights.items()
    ]
    edges.sort(key=lambda e: (e["source"], e["target"], e["type"]))
    node_list = sorted(nodes.values(), key=lambda n: (n["type"], n["label"].lower()))
    return {
        "schema_version": 1,
        "kind": "general_evidence_network",
        "nodes": node_list,
        "edges": edges,
        "summary": {
            "node_count": len(node_list),
            "edge_count": len(edges),
            "included_cluster_count": len(clusters),
            "top_papers_per_cluster": top_papers_per_cluster,
        },
    }
