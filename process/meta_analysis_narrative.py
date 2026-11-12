"""Source-linked evidence reporting, including when no statistical pool is possible."""
from __future__ import annotations

from collections import Counter, defaultdict
import json

from core.claims import claim_direction, claim_list
from core.io import atomic_write_json
from process.meta_analysis_groups import review_rows
from process.paths import load_paper_list
from process.statistical_extractor_text import article_id


def build_evidence_report(results, config):
    """Join only the assessed corpus to its saved claims and thematic clusters.

    Thematic membership explains the literature; it never authorizes pooling.
    Title-only cluster files are joined only when titles uniquely identify papers.
    """
    corpus_path = results.get("coverage", {}).get("corpus_path")
    papers = load_paper_list(corpus_path) if corpus_path else []
    by_id = {article_id(p, i): p for i, p in enumerate(papers)}
    titles = defaultdict(list)
    for aid, paper in by_id.items():
        titles[str(paper.get("title") or "")].append(aid)
    memberships = defaultdict(list)
    warnings = []
    cluster_path = config.clusters_dir / "clusters.json"
    if corpus_path and cluster_path.exists():
        try:
            clusters = json.loads(cluster_path.read_text(encoding="utf-8"))
            if not isinstance(clusters, list):
                raise ValueError("expected a list of clusters")
            unresolved = 0
            for cluster in clusters:
                label = f"Thematic cluster {cluster.get('cluster_id', '?')}"
                members = cluster.get("paper_titles", [])
                if not members:
                    members = [p.get("title", "") for p in cluster.get("papers", [])]
                for title in members:
                    ids = titles.get(title, []) if title else []
                    if len(ids) == 1:
                        if label not in memberships[ids[0]]:
                            memberships[ids[0]].append(label)
                    else:
                        unresolved += 1
            if unresolved:
                warnings.append(f"{unresolved} cluster memberships could not be uniquely matched to the assessed corpus.")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            warnings.append(f"Saved thematic clusters could not be read: {exc}")

    stats = defaultdict(list)
    for row in results.get("statistic_rows", []):
        stats[row["article_id"]].append(row)
    sources = []
    for i, audit in enumerate(results["paper_audit"], 1):
        aid = audit["article_id"]
        paper = by_id.get(aid, {})
        sources.append({
            "source_key": f"P{i}", "article_id": aid,
            "title": audit["title"] or paper.get("title") or aid,
            "doi": paper.get("doi", ""), "year": paper.get("year"),
            "groups": memberships.get(aid, ["Papers without matched thematic membership"]),
            "direction": claim_direction(paper), "findings": claim_list(paper, "findings"),
            "abstract": str(paper.get("abstract") or ""),
            "statistics": stats.get(aid, []), "pooled": audit["pooled"],
            "pooling_notes": audit["reasons"],
        })
    suggestions = defaultdict(list)
    for row in review_rows(results.get("statistic_rows", []), results.get("decisions", {}),
                           results.get("model_groups")):
        if row["suggested_group"]:
            suggestions[row["suggested_group"]].append(row)
    proposed = [{"label": label, "n_effects": len(rows),
                 "n_papers": len({r['article_id'] for r in rows}),
                 "effect_ids": [r["effect_id"] for r in rows]}
                for label, rows in suggestions.items()]
    return {"sources": sources, "suggested_groups": proposed, "warnings": warnings,
            "research_question": config.get_natural_language_question()}


def evidence_sections(evidence):
    """Complete, individually attributable sections, also used for model input."""
    sections = []
    grouped = defaultdict(list)
    for paper in evidence["sources"]:
        # Overlapping thematic memberships are reported on the paper, but each
        # paper is written once so it cannot inflate overall evidence counts.
        grouped[paper["groups"][0]].append(paper)
    for label, papers in grouped.items():
        directions = Counter(p["direction"] for p in papers)
        sections.append((label, f"{len(papers)} assessed papers. Recorded claim directions: "
                         + ", ".join(f"{k}: {v}" for k, v in sorted(directions.items()))
                         + ". These labels are descriptive; they are not a pooled effect or a vote on validity."))
        for paper in papers:
            key = paper["source_key"]
            lines = [f"[{key}] {paper['title']}", f"Article identifier: {paper['article_id']}",
                     f"Thematic membership: {'; '.join(paper['groups'])}",
                     f"Recorded claim direction: {paper['direction']}"]
            if paper["doi"]:
                lines.append(f"DOI: {paper['doi']}")
            if paper["findings"]:
                lines += ["Saved extracted findings:", *[f"- {f}" for f in paper["findings"]]]
            if paper["abstract"]:
                lines += ["Source abstract:", paper["abstract"]]
            if not paper["findings"] and not paper["abstract"]:
                lines.append("No saved findings or abstract are available for narrative interpretation.")
            for row in paper["statistics"]:
                lines += [f"Statistic [{row.get('effect_id', '')}]: {row.get('test_text') or row.get('effect_used') or 'No statistic text'}",
                          f"Outcome: {row.get('outcome') or 'unassigned'}; comparison: {row.get('comparison') or 'unassigned'}; population: {row.get('population') or 'unassigned'}.",
                          f"Reported effects: {row.get('effects', '')}; confidence intervals: {row.get('confidence_intervals', '')}; p values: {row.get('p_values', '')}; sample sizes: {row.get('sample_sizes', '')}.",
                          f"Assessment: {row.get('status', '')}. {row.get('reason', '')}",
                          f"Uncertainty source: {row.get('uncertainty_source', '')}; assumptions: {row.get('assumptions', '')}.",
                          f"Analysis estimate: {row.get('analysis_estimate', '')}; SE: {row.get('analysis_se', '')}; scale: {row.get('analysis_scale', '')}."]
            lines += [f"Contributes to a pooled estimate: {'yes' if paper['pooled'] else 'no'}.",
                      paper["pooling_notes"]]
            sections.append((f"{label} / [{key}]", "\n\n".join(lines)))
    return sections


def render_evidence_report(evidence):
    lines = ["## Evidence synthesis and source findings", "",
             "The following account uses saved extracted claims, abstracts and assessed statistics. "
             "Thematic clusters organize the papers; statistical pooling requires compatible, independent estimates. "
             "An absence of pooled estimates does not mean an absence of findings.", ""]
    lines += ["Research question: " + (evidence.get("research_question") or "Not specified."), ""]
    lines += [f"Warning: {w}" for w in evidence["warnings"]]
    lines += ["### Suggested statistical groups", "",
              f"{len(evidence['suggested_groups'])} suggested groups. A model group is pooled only after "
              "its paper-title/research-question gate and every recorded scientific compatibility "
              "characteristic pass. Outcome-label suggestions remain unpooled until assigned. The full "
              "model grouping audit records the paper titles, extracted statistics, matched traits, "
              "differences, and rejection reasons.", ""]
    for group in evidence["suggested_groups"]:
        lines.append(f"- {group['label']}: {group['n_papers']} papers, {group['n_effects']} candidate effects.")
    for label, body in evidence_sections(evidence):
        lines += ["", f"### {label}", "", body]
    if not evidence["sources"]:
        lines += ["", "No papers were available to assess."]
    return "\n".join(lines) + "\n"


def save_evidence_report(results, config):
    evidence = build_evidence_report(results, config)
    path = config.output_dir / "meta_analysis_evidence" / "narrative_sources.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, evidence)
    results["evidence_synthesis"] = evidence
    results["narrative_sources_path"] = str(path)
    return render_evidence_report(evidence)
