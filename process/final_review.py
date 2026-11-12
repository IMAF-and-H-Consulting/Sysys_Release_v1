"""Final, source-linked review from saved evidence; no endpoint or extraction rerun."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from core.claims import claim_direction, claim_list
from core.io import atomic_write_json, atomic_write_text
from process.paths import resolve_claims_corpus, resolve_validation_dir
from stages._common import is_cancelled, make_progress


def _load(path, warnings, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.append(f"Could not read {path.name}: {exc}")
        return default


def _text(value):
    # Saved evidence is prose, not report Markdown or embedded HTML.
    import html
    return html.escape(str(value or "")).replace("\n", " ").replace("|", "\\|")


def _cite(keys):
    return ", ".join(f"[{key}](#source-{key.lower()})" for key in keys)


METRIC_LABELS = {
    "papers": "Papers accounted for", "findings": "Papers with extracted findings",
    "abstract_only": "Papers with abstract evidence only", "missing": "Papers without findings or abstract",
    "fulltext": "Papers with saved full text",
}


def check_report(text, sources, robustness=None, meta=None):
    """Check owned coverage declarations, not statistics attributed to sources.

    This deliberately does not treat every subgroup count or number in quoted
    study evidence as a claim about the entire corpus.
    """
    expected = {
        "papers": len(sources), "findings": sum(bool(s["findings"]) for s in sources),
        "abstract_only": sum(not s["findings"] and bool(s["abstract"]) for s in sources),
        "missing": sum(not s["findings"] and not s["abstract"] for s in sources),
        "fulltext": sum(s.get("fulltext_available", False) for s in sources),
    }
    checks = []
    for key, label in METRIC_LABELS.items():
        matches = re.findall(rf"^- {re.escape(label)}: (\d+)\s*$", text, re.MULTILINE)
        actual = int(matches[0]) if len(matches) == 1 else None
        checks.append({"claim": label, "expected": expected[key], "actual": actual,
                       "status": "PASS" if actual == expected[key] else "FAIL",
                       "note": "Report coverage checked against source ledger"})
    anchors = re.findall(r'<a id="source-(p\d+)"></a>', text)
    keys = [s["source_key"].lower() for s in sources]
    links = re.findall(r'\]\(#source-(p\d+)\)', text)
    ok = Counter(anchors) == Counter(keys) and set(links) <= set(keys)
    checks.append({"claim": "Source references resolve to the complete ledger", "expected": len(keys),
                   "actual": len(anchors), "status": "PASS" if ok else "FAIL",
                   "note": "Every paper has one reference; every citation resolves"})
    # Cross-check displayed analysis values against the saved result objects.
    # Study-quoted numbers are left attributed, rather than compared to unrelated metrics.
    bootstrap = (robustness or {}).get("bootstrap") or {}
    if bootstrap and not bootstrap.get("skipped"):
        match = re.search(r"Saved bootstrap of mean relevance score: ([^;]+); interval \[([^,]+), ([^\]]+)\]", text)
        expected_values = [bootstrap.get(k) for k in ("mean", "ci_lower", "ci_upper")]
        try:
            observed = [float(v) for v in match.groups()] if match else None
        except ValueError:
            observed = None
        checks.append({"claim": "Displayed bootstrap estimate and interval", "expected": expected_values,
                       "actual": observed, "status": "PASS" if observed == expected_values else "FAIL",
                       "note": "Compared with saved robustness values"})
    expected_pools = [[g["pooled"].get(k) for k in ("pooled_effect", "ci_lower", "ci_upper", "i_squared")]
                      for g in (meta or {}).get("groups", []) if g.get("pooled")]
    matches = re.findall(r"Saved estimate on [^\n]+?: ([^;]+); interval \[([^,]+), ([^\]]+)\]; I²=([^ .]+(?:\.[0-9]+)?)\.", text)
    try:
        observed_pools = [[float(v) for v in row] for row in matches]
    except ValueError:
        observed_pools = None
    if expected_pools:
        checks.append({"claim": "Displayed pooled estimates, intervals and heterogeneity", "expected": expected_pools,
                       "actual": observed_pools, "status": "PASS" if observed_pools == expected_pools else "FAIL",
                       "note": "Compared with saved meta-analysis values; scientific validity is separate"})
    return checks


def _render(question, sources, groups, robustness, meta, warnings):
    findings = sum(bool(s["findings"]) for s in sources)
    abstract_only = sum(not s["findings"] and bool(s["abstract"]) for s in sources)
    lines = ["# Final systematic review", "", "## Research question", "", _text(question), "",
             "## Evidence coverage", "", f"- Papers accounted for: {len(sources)}",
             f"- Papers with extracted findings: {findings}",
             f"- Papers with abstract evidence only: {abstract_only}",
             f"- Papers without findings or abstract: {len(sources)-findings-abstract_only}", "",
             f"- Papers with saved full text: {sum(s.get('fulltext_available', False) for s in sources)}", "",
             "Full-text availability records saved coverage. This stage uses the findings and statistical "
             "reviews already extracted; it does not run a new interpretation of the raw full texts.", "",
             "## Findings in relation to the question", "",
             "The sections below bring together the saved findings by thematic membership and recorded evidence "
             "direction. Findings remain attributed to their sources. Direction labels are extraction results; "
             "their frequency is not an effect estimate or a judgment of study quality.", ""]
    for theme, members in groups.items():
        lines += [f"### {_text(theme)}", ""]
        for direction in ("supports", "contradicts", "neutral", "unknown"):
            selected = [s for s in members if s["direction"] == direction]
            if not selected:
                continue
            lines += [f"**Recorded direction: {direction}** — {_cite([s['source_key'] for s in selected])}", ""]
            shared = defaultdict(list)
            for source in selected:
                for finding in source["findings"] or (["Abstract evidence: " + source["abstract"]] if source["abstract"] else ["No saved findings or abstract available."]):
                    shared[finding].append(source["source_key"])
            for finding, keys in shared.items():
                lines.append(f"- {_text(finding)} {_cite(keys)}")
            lines.append("")
        if len({s["direction"] for s in members} & {"supports", "contradicts"}) == 2:
            lines += ["Supporting and contradicting findings coexist in this theme. Their study-specific "
                      "comparisons must be examined before drawing a common conclusion.", ""]
    lines += ["## Robustness of the saved evidence", ""]
    if robustness:
        from stages.s9_robustness_run import skipped_analyses
        bs = robustness.get("bootstrap") or {}
        if bs and not bs.get("skipped"):
            lines += [f"Saved bootstrap of mean relevance score: {bs.get('mean')}; "
                      f"interval [{bs.get('ci_lower')}, {bs.get('ci_upper')}]. "
                      "This measures screening-score precision, not the magnitude of a research effect.", ""]
        sh = robustness.get("split_half") or {}
        if sh.get("estimable"):
            lines += [f"Saved split-half condition-support reliability: {sh.get('mean_correlation')}.", ""]
        sa = robustness.get("sensitivity") or {}
        lines += [f"Saved relevance-threshold sensitivity stability: {sa.get('stable', 'unavailable')}.", ""]
        for label, reason in skipped_analyses(robustness):
            lines.append(f"- {_text(label)} unavailable: {_text(reason)}")
        for name in ("permutation", "condition_sensitivity", "network_analysis"):
            section = robustness.get(name)
            if section:
                lines += [f"### Saved {_text(name.replace('_', ' '))}", "", "```json",
                          json.dumps(section, ensure_ascii=False, indent=2), "```", ""]
    else:
        lines += ["No matching saved robustness result is available. Robustness has not been established by this report.", ""]
    lines += ["## Statistical synthesis", ""]
    pooled = [g for g in (meta or {}).get("groups", []) if g.get("pooled")]
    if pooled:
        for group in pooled:
            p = group["pooled"]
            lines += [f"### {_text(group.get('label', 'Saved analysis group'))}", "",
                      f"Saved estimate on {_text(group.get('analysis_scale'))}: {p.get('pooled_effect')}; "
                      f"interval [{p.get('ci_lower')}, {p.get('ci_upper')}]; I²={p.get('i_squared')}. "
                      f"Independent studies: {group.get('n_studies')}.", ""]
            ids = {str(s.get("article_id")) for s in group.get("studies", [])}
            lines += ["Contributing saved sources: " + _cite([s["source_key"] for s in sources if s["article_id"] in ids]), ""]
    else:
        lines += ["No matching saved pooled estimate is available. The source-linked findings above remain "
                  "available for narrative interpretation; this report does not manufacture a combined effect.", ""]
    lines += ["## Interpretation and limitations", "",
              "The research question is addressed through the attributed findings above. A single corpus-wide "
              "effect, causal conclusion, or certainty rating cannot be inferred from claim counts. "
              "Read supporting, contradicting, neutral and unknown findings together, preserving each paper's "
              "population and comparison. Saved extraction can contain errors and requires source review.", "",
              "Statistics reproduced from saved analyses and study text are attributed evidence. The internal "
              "numerical check verifies this report's coverage and references; it does not independently "
              "validate every published statistic or the scientific conclusions.", ""]
    lines += [f"- {_text(w)}" for w in warnings]
    lines += ["", "## Source references", ""]
    for source in sources:
        lines += [f'<a id="source-{source["source_key"].lower()}"></a>',
                  f"**{source['source_key']}. {_text(source['title'])}** ({_text(source['year'])}).", "",
                  f"Saved identifier: `{_text(source['article_id'])}`. DOI: {_text(source['doi']) or 'not recorded'}.", ""]
        if source.get("method_profile"):
            lines += ["Saved study methods and validation:", "", "```json",
                      json.dumps(source["method_profile"], ensure_ascii=False, indent=2), "```", ""]
        if source["doi"]:
            from urllib.parse import quote
            lines += [f"[Open source DOI](https://doi.org/{quote(str(source['doi']), safe='/')})", ""]
        if source["statistics"]:
            lines += ["Saved statistical evidence (context and review retained in final_review_sources.json):", ""]
            for stat in source["statistics"]:
                lines.append(f"- {_text(stat.get('test_text') or stat.get('effect_used') or 'Statistic without text')}")
            lines.append("")
    return "\n".join(lines)


def _primary_report(full, sources, groups, narratives, citation_map):
    """Readable synthesis first; complete evidence remains in a separate appendix."""
    start = full.split("## Findings in relation to the question", 1)[0]
    lines = [start, "## Findings in relation to the question", "",
             "[Complete findings, study methods and statistical evidence](final_review_evidence.md). "
             "Every paper is accounted for in that appendix and the JSON source ledger.", ""]
    for theme, members in groups.items():
        lines += [f"### {_text(theme)}", ""]
        narrative = narratives.get(theme)
        if narrative:
            def citations(match):
                keys = [k.strip() for k in match.group(1).split(";")]
                resolved = [citation_map[k] for k in keys if k in citation_map]
                return _cite(resolved) if len(resolved) == len(keys) else match.group(0)
            lines += ["Saved thematic interpretation (generated during synthesis; source claims remain available for review):", "",
                      re.sub(r"\[([^]\n]+)\]", citations, "\n\n".join(_text(part) for part in narrative.split("\n\n"))), ""]
        else:
            lines += ["No current saved thematic narrative is available. The following attributed findings "
                      "provide a descriptive account; they do not establish a common effect.", ""]
        for direction in ("supports", "contradicts", "neutral", "unknown"):
            selected = [p for p in members if p["direction"] == direction]
            if not selected:
                continue
            lines += [f"**Recorded direction: {direction}.**", ""]
            examples = [(f, p["source_key"]) for p in selected for f in
                        (p["findings"] or (["Abstract evidence: " + p["abstract"]] if p["abstract"] else []))]
            # Explicit examples, with the complete uncropped account in the appendix.
            chosen = examples if len(sources) <= 10 else examples[:3]
            lines += [f"- {_text(f)} {_cite([key])}" for f, key in chosen]
            if len(chosen) < len(examples):
                lines += ["These are illustrative saved findings. All findings in this category are retained "
                          "in the [complete evidence appendix](final_review_evidence.md)."]
            if not examples:
                lines += ["No findings or abstracts were saved for these sources: " + _cite([p["source_key"] for p in selected])]
            lines.append("")
        if {p["direction"] for p in members} >= {"supports", "contradicts"}:
            lines += ["Supporting and contradicting findings coexist in this theme; populations, methods "
                      "and comparisons need to be considered before drawing a shared conclusion.", ""]
    middle = full.split("## Robustness of the saved evidence", 1)[1].split("## Source references", 1)[0]
    lines += ["## Robustness of the saved evidence" + middle, "## Source references", ""]
    from urllib.parse import quote
    for p in sources:
        lines += [f'<a id="source-{p["source_key"].lower()}"></a>',
                  f"**{p['source_key']}. {_text(p['title'])}** ({_text(p['year'])}). "
                  + (f"[DOI](https://doi.org/{quote(str(p['doi']), safe='/')})" if p['doi'] else "DOI not recorded."), ""]
    return "\n".join(lines)


def run_review(config, claims_path=None, clusters_path=None, output_dir=None,
               cancel_event=None, progress_callback=None, **kw):
    """Desktop Stage 9: compile saved evidence, check it, then publish the report."""
    import logging
    from process.statistical_extractor_text import article_id
    from stages.s10_verify_counts import load_clusters
    progress = make_progress(logging.getLogger(__name__), progress_callback)
    progress("STAGE 9: FINAL SOURCE-LINKED REVIEW (local saved evidence)")
    corpus = resolve_claims_corpus(config, claims_path)
    if corpus is None:
        return {"error": "No saved claims corpus found"}
    warnings = []
    papers = _load(corpus, warnings)
    if isinstance(papers, dict):
        papers = papers.get("articles", list(papers.values()))
    if not isinstance(papers, list) or not papers or not all(isinstance(p, dict) for p in papers):
        return {"error": "Final review needs a nonempty list of saved paper records"}
    cluster_path = Path(clusters_path) if clusters_path else config.clusters_dir / "clusters.json"
    clusters, note = load_clusters(cluster_path, papers)
    if clusters is None:
        warnings.append(f"Thematic membership unavailable: {note}. Findings are grouped as unassigned evidence.")
    title_counts = Counter(p.get("title", "") for p in papers)
    themes = {}
    for cluster in clusters or []:
        for title in cluster.get("paper_titles", []):
            if title_counts[title] == 1:
                themes[title] = f"Thematic cluster {cluster['cluster_id']}"
    inputs = [corpus, cluster_path]
    def saved(path):
        inputs.append(path)
        return _load(path, warnings, {})
    extraction = saved(config.claims_dir / "statistical_extractions.json")
    stats = {str(r.get("article_id")): r.get("results", []) for r in extraction.get("records", [])} if isinstance(extraction, dict) else {}
    profiles = saved(config.data_dir / "method_review" / "method_profiles_validated.json")
    methods = {str(article_id(p, i)): p.get("method_profile", {}) for i, p in enumerate(profiles)
               if isinstance(p, dict)} if isinstance(profiles, list) else {}
    from process.statistical_extractor_text import enrich_from_saved_fulltext
    fulltext_path = config.corpus_dir / "corpus_fulltext.json"
    inputs.append(fulltext_path)
    try:
        enriched, _ = enrich_from_saved_fulltext(papers, fulltext_path)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        warnings.append(f"Saved full-text coverage could not be read: {exc}")
        enriched = papers
    sources, groups = [], defaultdict(list)
    for i, paper in enumerate(papers):
        if is_cancelled(cancel_event):
            return {"cancelled": True}
        aid = str(article_id(paper, i))
        source = {"source_key": f"P{i+1}", "article_id": aid, "title": paper.get("title") or aid,
                  "year": paper.get("year", "not recorded"), "doi": paper.get("doi", ""),
                  "direction": claim_direction(paper), "findings": claim_list(paper, "findings"),
                  "abstract": paper.get("abstract") or "", "statistics": stats.get(aid, []),
                  "method_profile": paper.get("method_profile") or methods.get(aid, {}),
                  "fulltext_available": any(isinstance(enriched[i].get(k), str) and bool(enriched[i][k].strip())
                                            for k in ("full_text", "fulltext", "extracted_text"))}
        sources.append(source)
        groups[themes.get(paper.get("title"), "Evidence without unambiguous thematic membership")].append(source)
    progress(f"Accounted for {len(sources)} papers; assembling findings and conflicting evidence")
    robust_path = config.data_dir / "validation" / "robustness_results.json"
    robustness = saved(robust_path)
    if not isinstance(robustness, dict) or (robustness and (robustness.get("n_papers") != len(papers) or robust_path.stat().st_mtime_ns < corpus.stat().st_mtime_ns)):
        warnings.append("Saved robustness is stale or does not match corpus size; excluded.")
        robustness = {}
    meta_path = config.output_dir / "meta_analysis_results.json"
    meta = saved(meta_path)
    if meta:
        meta_corpus = (meta.get("coverage") or {}).get("corpus_path") if isinstance(meta, dict) else None
        if not meta_corpus or Path(meta_corpus).resolve() != corpus.resolve() or meta_path.stat().st_mtime_ns < corpus.stat().st_mtime_ns:
            warnings.append("Saved meta-analysis cannot be matched to this corpus version; excluded.")
            meta = {}
    warnings.append("Saved analysis freshness uses recorded corpus path, size and modification time; historical files have no common input hash.")
    question = config.get_natural_language_question() or "Research question not recorded in this project."
    narratives_path = cluster_path.with_name("narratives.json")
    narratives_raw = saved(narratives_path)
    narrative_map = {}
    cluster_sizes = {f"Thematic cluster {c['cluster_id']}": c['paper_count'] for c in clusters or []}
    if isinstance(narratives_raw, list) and narratives_path.stat().st_mtime_ns >= corpus.stat().st_mtime_ns:
        for n in narratives_raw:
            if isinstance(n, dict):
                label = f"Thematic cluster {n.get('cluster_id')}"
                if n.get('paper_count') == cluster_sizes.get(label) and n.get('narrative'):
                    narrative_map[label] = n['narrative']
    citations = saved(cluster_path.with_name("synthesis_sources.json"))
    citation_map = {}
    title_to_key = {s['title']: s['source_key'] for s in sources if title_counts[s['title']] == 1}
    for record in citations.get('sources', []) if isinstance(citations, dict) else []:
        if isinstance(record, dict) and record.get('title') in title_to_key:
            citation_map[record.get('citation_key')] = title_to_key[record['title']]
    full = _render(question, sources, groups, robustness, meta, warnings)
    text = _primary_report(full, sources, groups, narrative_map, citation_map)
    checks = check_report(text, sources, robustness, meta)
    summary = {"total_checks": len(checks), "passed": sum(c["status"] == "PASS" for c in checks),
               "failed": sum(c["status"] == "FAIL" for c in checks), "warnings": len(warnings)}
    if summary["failed"]:
        return {"error": "Final report failed internal coverage/reference checks", "checks": checks, "summary": summary}
    if is_cancelled(cancel_event):
        return {"cancelled": True}
    validation = resolve_validation_dir(config, output_dir)
    output = config.output_dir
    report = output / "final_review_report.md"
    ledger = output / "final_review_sources.json"
    appendix = output / "final_review_evidence.md"
    provenance = [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in inputs if p.is_file()]
    result = {"review_report": True, "n_papers": len(sources), "n_findings": sum(len(s['findings']) for s in sources),
              "n_themes": len(groups), "summary": summary, "checks": checks, "warnings": warnings,
              "report_path": str(report), "sources_path": str(ledger), "evidence_path": str(appendix), "inputs": provenance}
    text += "\n\n## Internal numerical and source-reference checks\n\n" + "\n".join(
        f"- {c['status']}: {c['claim']} (expected {c['expected']}, observed {c['actual']})" for c in checks) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(ledger, {"research_question": question, "sources": sources, "inputs": provenance})
    atomic_write_text(appendix, full)
    atomic_write_text(report, text)
    atomic_write_json(validation / "verification_results.json", result)
    progress(f"Final review saved: {len(sources)} papers, {result['n_findings']} findings, {len(groups)} themes; internal checks passed")
    return result
