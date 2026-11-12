"""Deterministic hypothesis-explanation artifact for Stage 6.

Builds ``hypothesis_explanation.json``/``.md``: evidence counts by direction,
frequent terms, the study-design mix, a GRADE snapshot and caveats. No LLM
is involved; every number traces back to the clustered papers.
"""
from __future__ import annotations

from collections import Counter

from core.claims import UNKNOWN_DIRECTION, claim_list
from core.doi import normalize_doi
from process.claim_extractor import _research_question_from_config
from process.synthesis_papers import IDENTITY_FIELDS as _IDENTITY_FIELDS
from process.synthesis_papers import (
    _paper_direction,
    _paper_score,
    _rank_papers,
    _top_terms,
)

# Fields that hold the same identifier under different names.
_IDENTITY_NAMESPACE = {"pmcid": "pmc", "article_id": "id"}


def _identity_keys(record: dict) -> list[str]:
    """Every namespaced identifier of a paper or method-profile record."""
    keys: list[str] = []
    for field in _IDENTITY_FIELDS:
        value = record.get(field)
        if not value:
            continue
        text = normalize_doi(value) if field == "doi" else str(value).strip().lower()
        if text:
            keys.append(f"{_IDENTITY_NAMESPACE.get(field, field)}:{text}")
    return keys


def _method_lookup(method_profiles: list[dict] | dict | None) -> dict[str, dict]:
    """Method-profile rows indexed by every identifier they carry."""
    if not method_profiles:
        return {}
    if isinstance(method_profiles, dict):
        # {article_id: profile} maps carry the id only as the key.
        rows = [{"article_id": aid, **row} for aid, row in method_profiles.items()
                if isinstance(row, dict)]
    else:
        rows = [row for row in method_profiles if isinstance(row, dict)]
    return {key: row for row in rows for key in _identity_keys(row)}


def _method_type(profile: dict) -> str:
    """Study design from a Stage 4 row (nested ``method_profile``) or a flat profile."""
    nested = profile.get("method_profile")
    if isinstance(nested, dict) and nested.get("study_method_type"):
        return str(nested["study_method_type"])
    return str(profile.get("study_method_type") or "unknown")


def _match_method_profile(paper: dict, method_by_id: dict[str, dict]) -> dict | None:
    return next((method_by_id[k] for k in _identity_keys(paper) if k in method_by_id), None)


def _method_caveat(method_profiles, method_counts: Counter) -> str | None:
    if not method_profiles:
        return "Method validation output was not available for study-design weighting."
    if not method_counts:
        return (f"Method validation output ({len(method_profiles)} profile(s)) matched none "
                "of the synthesized papers, so the method mix is empty.")
    return None


def _direction_caveats(contradict_total: int, unassessed_total: int) -> list[str]:
    caveats: list[str] = []
    if contradict_total:
        caveats.append(
            f"{contradict_total} paper(s) were marked as contradicting or directionally opposed."
        )
    if unassessed_total:
        caveats.append(
            f"Contradiction was not assessed for {unassessed_total} paper(s): their claims "
            "record only that they do not support the question (or come from a heuristic "
            "fallback), so the contradicting count covers only papers with an explicit direction."
        )
    return caveats


def _grade_summary(grade_results: list[dict] | dict | None) -> dict:
    if not grade_results:
        return {}
    rows = grade_results.values() if isinstance(grade_results, dict) else grade_results
    counts: Counter[str] = Counter()
    examples: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cert = (
            row.get("overall_certainty")
            or row.get("certainty")
            or row.get("overall")
            or ""
        )
        if isinstance(cert, dict):
            cert = cert.get("value") or cert.get("label") or ""
        cert_text = str(cert).strip()
        if cert_text:
            counts[cert_text] += 1
        label = row.get("component_label") or row.get("component") or row.get("label")
        if label and cert_text and len(examples) < 8:
            examples.append({"component": str(label), "certainty": cert_text})
    return {"certainty_counts": dict(sorted(counts.items())), "examples": examples}


def _cluster_row(cluster: dict, narrative_by_cluster: dict) -> dict:
    papers = cluster.get("papers", []) or []
    directions = [_paper_direction(p) for p in papers]
    support_count = directions.count("supports")
    contradict_count = directions.count("contradicts")
    avg_score = sum(_paper_score(p) for p in papers) / len(papers) if papers else 0.0
    narrative = narrative_by_cluster.get(cluster.get("cluster_id"), {}).get("narrative", "")
    return {
        "cluster_id": cluster.get("cluster_id"),
        "paper_count": len(papers),
        "avg_relevance_score": round(avg_score, 3),
        "supporting_count": support_count,
        "contradicting_count": contradict_count,
        "neutral_count": len(papers) - support_count - contradict_count,
        "direction_unassessed_count": directions.count(UNKNOWN_DIRECTION),
        "mechanisms": cluster.get("mechanisms", [])[:12],
        "conditions": cluster.get("conditions", [])[:12],
        "key_entities": claim_list(cluster, "key_entities")[:12],
        "top_papers": _rank_papers(papers, limit=6),
        "narrative_excerpt": (narrative or "")[:600],
    }


def _context_caveats(method_profiles, method_counts: Counter, grade_results,
                     unreadable: dict[str, str]) -> list[str]:
    """Caveats about the optional method-review and GRADE inputs.

    An input that exists but could not be read is reported as such, never
    as "not available".
    """
    caveats: list[str] = []
    if "method_profiles" in unreadable:
        caveats.append("Method validation output exists but could not be read "
                       f"({unreadable['method_profiles']}); the method mix is empty.")
    else:
        method_caveat = _method_caveat(method_profiles, method_counts)
        if method_caveat:
            caveats.append(method_caveat)
    if "grade_results" in unreadable:
        caveats.append("GRADE output exists but could not be read "
                       f"({unreadable['grade_results']}); GRADE context was omitted.")
    elif not grade_results:
        caveats.append("GRADE output was not available when this explanation was generated.")
    return caveats


def build_hypothesis_explanation(
    clusters: list[dict],
    narratives: list[dict],
    abstract: str,
    *,
    research_config: dict | None = None,
    method_profiles: list[dict] | dict | None = None,
    grade_results: list[dict] | dict | None = None,
    unreadable_inputs: dict[str, str] | None = None,
    extra_caveats: list[str] | None = None,
) -> dict:
    """Build a deterministic provenance explanation for the synthesis.

    *unreadable_inputs* maps ``"method_profiles"``/``"grade_results"`` to the
    read error of an optional input that exists but could not be loaded.
    *extra_caveats* (e.g. a weak-cluster-structure warning) are appended.
    """
    all_papers: list[dict] = []
    for cluster in clusters:
        all_papers.extend(cluster.get("papers", []) or [])

    method_by_id = _method_lookup(method_profiles)
    method_counts: Counter[str] = Counter()
    for paper in all_papers:
        profile = _match_method_profile(paper, method_by_id)
        if profile:
            method_counts[_method_type(profile)] += 1

    narrative_by_cluster = {
        n.get("cluster_id"): n
        for n in narratives
        if isinstance(n, dict)
    }
    cluster_rows = [_cluster_row(cluster, narrative_by_cluster) for cluster in clusters]
    cluster_rows.sort(
        key=lambda row: (
            row["supporting_count"],
            row["avg_relevance_score"],
            row["paper_count"],
        ),
        reverse=True,
    )

    support_total = sum(row["supporting_count"] for row in cluster_rows)
    contradict_total = sum(row["contradicting_count"] for row in cluster_rows)
    neutral_total = sum(row["neutral_count"] for row in cluster_rows)
    unassessed_total = sum(row["direction_unassessed_count"] for row in cluster_rows)

    caveats: list[str] = []
    if not all_papers:
        caveats.append("No papers were available for deterministic provenance.")
    caveats.extend(_direction_caveats(contradict_total, unassessed_total))
    if support_total and neutral_total > support_total:
        caveats.append("Neutral or weakly directional evidence outnumbers direct support.")
    caveats.extend(_context_caveats(method_profiles, method_counts, grade_results,
                                    unreadable_inputs or {}))
    caveats.extend(extra_caveats or [])

    return {
        "schema_version": 1,
        "research_question": _research_question_from_config(research_config),
        "total_papers": len(all_papers),
        "cluster_count": len(clusters),
        "supporting_count": support_total,
        "contradicting_count": contradict_total,
        "neutral_count": neutral_total,
        "direction_unassessed_count": unassessed_total,
        "top_mechanisms": _top_terms(all_papers, "mechanisms"),
        "top_conditions": _top_terms(all_papers, "conditions"),
        "top_key_entities": _top_terms(all_papers, "key_entities"),
        "top_findings": _top_terms(all_papers, "findings"),
        "method_mix": dict(sorted(method_counts.items())),
        "grade_summary": _grade_summary(grade_results),
        "clusters": cluster_rows,
        "top_papers_overall": _rank_papers(all_papers, limit=10),
        "caveats": caveats,
        "abstract_excerpt": (abstract or "")[:1200],
    }


def _unassessed_note(explanation: dict) -> str:
    unassessed = explanation.get("direction_unassessed_count") or 0
    return f" (direction not assessed for {unassessed} paper(s))" if unassessed else ""


def render_hypothesis_explanation_markdown(explanation: dict) -> str:
    """Render the deterministic explanation artifact as readable Markdown."""
    lines = [
        "# Why This Hypothesis?",
        "",
        "## Research Question",
        "",
        str(explanation.get("research_question") or "Not specified."),
        "",
        "## Evidence Basis",
        "",
        f"- Papers in synthesis: {explanation.get('total_papers', 0)}",
        f"- Clusters: {explanation.get('cluster_count', 0)}",
        f"- Supporting signals: {explanation.get('supporting_count', 0)}",
        f"- Contradicting signals: {explanation.get('contradicting_count', 0)}"
        + _unassessed_note(explanation),
        f"- Neutral or unclear signals: {explanation.get('neutral_count', 0)}",
        "",
    ]

    for title, key in (
        ("Top Mechanisms", "top_mechanisms"),
        ("Top Conditions or Contexts", "top_conditions"),
        ("Top Key Entities", "top_key_entities"),
    ):
        rows = explanation.get(key) or []
        if rows:
            lines.extend([f"## {title}", ""])
            for row in rows[:10]:
                lines.append(f"- {row['term']} (n={row['count']})")
            lines.append("")

    method_mix = explanation.get("method_mix") or {}
    if method_mix:
        lines.extend(["## Method Mix", ""])
        for method, count in method_mix.items():
            lines.append(f"- {method}: {count}")
        lines.append("")

    grade_summary = explanation.get("grade_summary") or {}
    if grade_summary.get("certainty_counts"):
        lines.extend(["## GRADE Snapshot", ""])
        for certainty, count in grade_summary["certainty_counts"].items():
            lines.append(f"- {certainty}: {count}")
        lines.append("")

    lines.extend(["## Top Contributing Clusters", ""])
    for cluster in explanation.get("clusters", [])[:6]:
        cid = cluster.get("cluster_id")
        lines.append(
            f"### Cluster {cid} - {cluster.get('paper_count', 0)} papers"
        )
        lines.append(
            f"Average score: {cluster.get('avg_relevance_score', 0)}; "
            f"supporting: {cluster.get('supporting_count', 0)}; "
            f"contradicting: {cluster.get('contradicting_count', 0)}"
        )
        terms = []
        for key in ("mechanisms", "conditions", "key_entities"):
            terms.extend(claim_list(cluster, key))
        if terms:
            lines.append(f"Key terms: {', '.join(terms[:10])}")
        for paper in cluster.get("top_papers", [])[:3]:
            title = paper.get("title") or "Untitled"
            year = paper.get("year") or "n.d."
            score = paper.get("score", 0)
            direction = paper.get("direction", "neutral")
            lines.append(f"- {title} ({year}) - {direction}, score {score:g}")
            quote = paper.get("quote")
            if quote:
                lines.append(f"  Quote: {quote[:280]}")
        lines.append("")

    caveats = explanation.get("caveats") or []
    if caveats:
        lines.extend(["## Deterministic Caveats", ""])
        for caveat in caveats:
            lines.append(f"- {caveat}")
        lines.append("")

    if explanation.get("abstract_excerpt"):
        lines.extend(["## Synthesis Excerpt", "", str(explanation["abstract_excerpt"]).strip(), ""])

    return "\n".join(lines).rstrip() + "\n"
