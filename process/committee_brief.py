"""Committee-facing evidence brief exports."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from core.io import atomic_write_text
from process.export import export_bibtex, export_ris, load_corpus_for_export
from process.prisma import (
    collect_prisma_data,
    generate_checklist,
    generate_flow_diagram,
    save_checklist_csv,
    save_checklist_markdown,
)

logger = logging.getLogger(__name__)


# Built-in defaults are topic-neutral. Projects set their own audience,
# title, and questions in the ``committee_brief`` / ``abct_tech_ai_infosheet``
# sections of the research config (see the templates/abct_*.yaml examples).
DEFAULT_COMMITTEE_BRIEF = {
    "audience": "Review committee",
    "meeting_date": "",
    "title": "Evidence Brief",
    "filename": "committee_evidence_brief.md",
    "decision_questions": [
        "Where is the evidence strongest for the research question?",
        "Which conclusions are not yet supported well enough to act on?",
        "What risks, gaps, or safeguards should the committee discuss?",
    ],
}


DEFAULT_INFOSHEET = {
    "audience": "Review committee",
    "meeting_date": "",
    "title": "Evidence Info-Sheet",
    "subtitle": "One-page summary of the systematic review",
    "filename": "evidence_infosheet.md",
    "decision_questions": [
        "What does the evidence support, and with what certainty?",
        "Which uses or conclusions need caution labels?",
        "What oversight, transparency, or follow-up does the evidence call for?",
    ],
    "key_messages": [
        "Separate well-supported findings from preliminary or conflicting ones.",
        "State the certainty of evidence alongside every recommendation.",
        "Name the populations, settings, and conditions the evidence covers, and those it does not.",
    ],
    "practical_guidance": [
        "Check that cited evidence matches the population and setting it is applied to.",
        "Prefer narrow, reviewable recommendations over broad endorsements.",
        "Record open questions and the evidence that would resolve them.",
    ],
}

# Former names, kept for callers that imported them.
DEFAULT_ABCT_BRIEF = DEFAULT_COMMITTEE_BRIEF
DEFAULT_ABCT_TECH_AI_INFOSHEET = DEFAULT_INFOSHEET


def export_abct_committee_brief(config, output_dir: Path) -> Path:
    """Export a committee evidence brief pack to *output_dir*.

    The pack is deliberately useful from partial pipeline state: it always
    writes a Markdown brief and PRISMA-trAIce checklist; it adds flow diagrams,
    bibliography, and copied synthesis/validation artifacts when available.
    """
    return _export_abct_pack(
        config,
        output_dir,
        defaults=DEFAULT_COMMITTEE_BRIEF,
        config_section="committee_brief",
        renderer=render_abct_committee_brief,
        corpus_stem="abct_brief_corpus",
        readme_title="Committee Brief Pack",
        log_label="committee brief",
    )


def export_abct_tech_ai_infosheet(config, output_dir: Path) -> Path:
    """Export a one-page committee info-sheet pack."""
    return _export_abct_pack(
        config,
        output_dir,
        defaults=DEFAULT_INFOSHEET,
        config_section="abct_tech_ai_infosheet",
        renderer=render_abct_tech_ai_infosheet,
        corpus_stem="abct_infosheet_corpus",
        readme_title="Committee Info-Sheet Pack",
        log_label="committee info-sheet",
    )


def _export_abct_pack(
    config,
    output_dir: Path,
    *,
    defaults: dict,
    config_section: str,
    renderer,
    corpus_stem: str,
    readme_title: str,
    log_label: str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    research = getattr(config, "research_config", {}) or {}
    brief_cfg = _brief_config(research, config_section=config_section, defaults=defaults)
    prisma_data = collect_prisma_data(config)
    checklist = generate_checklist(prisma_data)

    files: list[str] = []

    brief_path = output_dir / brief_cfg["filename"]
    atomic_write_text(brief_path, renderer(config, prisma_data, checklist, brief_cfg))
    files.append(brief_path.name)

    save_checklist_markdown(checklist, output_dir / "prisma_traice_checklist.md")
    save_checklist_csv(checklist, output_dir / "prisma_traice_checklist.csv")
    files.extend(["prisma_traice_checklist.md", "prisma_traice_checklist.csv"])

    try:
        generate_flow_diagram(prisma_data, output_dir / "prisma_flow_diagram", fmt="png")
        generate_flow_diagram(prisma_data, output_dir / "prisma_flow_diagram", fmt="svg")
        files.extend(["prisma_flow_diagram.png", "prisma_flow_diagram.svg"])
    except Exception as exc:
        logger.warning("Could not generate PRISMA flow diagram for committee brief: %s", exc)

    corpus = load_corpus_for_export(config)
    if corpus:
        try:
            export_bibtex(corpus, output_dir / f"{corpus_stem}.bib")
            files.append(f"{corpus_stem}.bib")
        except Exception as exc:
            logger.warning("Committee brief BibTeX export failed: %s", exc)
        try:
            export_ris(corpus, output_dir / f"{corpus_stem}.ris")
            files.append(f"{corpus_stem}.ris")
        except Exception as exc:
            logger.warning("Committee brief RIS export failed: %s", exc)

    files.extend(_copy_supporting_artifacts(config, output_dir))
    _write_readme(output_dir, brief_cfg, files, title=readme_title)
    files.append("README.md")

    logger.info("%s exported to %s (%d files)", log_label, output_dir, len(files))
    return output_dir


def render_abct_committee_brief(
    config, prisma_data, checklist: list[dict], brief_cfg: dict | None = None
) -> str:
    """Render the committee evidence brief as Markdown."""
    research = getattr(config, "research_config", {}) or {}
    brief_cfg = brief_cfg or _brief_config(
        research, config_section="committee_brief", defaults=DEFAULT_COMMITTEE_BRIEF
    )
    artifacts = _collect_artifacts(config)
    counts = _counts(artifacts)
    meeting = _format_date(brief_cfg["meeting_date"])
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# {brief_cfg['title']}",
        "",
        f"**Audience:** {brief_cfg['audience']}",
        f"**Target meeting:** {meeting}",
        f"**Generated:** {generated}",
        "",
        "## Review Focus",
        "",
        _clean_text(
            research.get("natural_language_question")
            or (research.get("relevance") or {}).get("research_question")
            or getattr(config, "hypothesis_text", "")
            or "Evidence summary for the configured research question."
        ),
        "",
        "## Decision Questions",
        "",
    ]

    for question in brief_cfg["decision_questions"]:
        lines.append(f"- {question}")

    lines.extend([
        "",
        "## Evidence Snapshot",
        "",
        f"- Records identified: {prisma_data.records_identified}",
        f"- Records screened: {prisma_data.records_screened}",
        f"- Records retained after relevance screening: {counts['relevant']}",
        f"- Records with extracted claims: {counts['claims']}",
        f"- Records included in synthesis: {prisma_data.studies_in_synthesis}",
        f"- Relevance threshold: >= {getattr(config, 'relevance_threshold', 3)} on the configured 0-5 scale",
        "",
        "## Current Synthesis",
        "",
        _synthesis_summary(artifacts),
        "",
        "## Certainty of Evidence (GRADE)",
        "",
        *_certainty_lines(artifacts),
        "",
        "## Committee-Ready Themes",
        "",
    ])

    themes = _theme_lines(artifacts, research)
    lines.extend(themes or ["- Run synthesis to populate evidence themes from extracted claims."])

    lines.extend([
        "",
        "## Safety, Privacy, Equity, and Oversight",
        "",
    ])
    lines.extend(_risk_lines(artifacts) or [
        "- Extracted safety, privacy, equity, and oversight notes will appear here after Stage 3 extraction.",
    ])

    lines.extend([
        "",
        "## Top Relevant Papers",
        "",
    ])
    lines.extend(_top_paper_lines(artifacts["relevant"] or artifacts["corpus"]))

    lines.extend([
        "",
        "## PRISMA-trAIce Notes",
        "",
        f"- Information sources: {_source_list(prisma_data.records_per_source)}",
        f"- AI tools: {_ai_tool_list(prisma_data.ai_tools)}",
        f"- Human oversight points: {_list_or_pending(prisma_data.human_oversight_points)}",
        f"- AI performance metrics: {_performance_summary(prisma_data.ai_performance)}",
        "- Full PRISMA-trAIce checklist exported as `prisma_traice_checklist.md` and `prisma_traice_checklist.csv`.",
        "",
        "## Action-Oriented Gaps",
        "",
    ])
    lines.extend(_gap_lines(artifacts, checklist))

    lines.extend([
        "",
        "## Suggested Meeting Use",
        "",
        "- Use the Evidence Snapshot to decide whether the brief is ready for committee discussion or needs another search/screening pass.",
        "- Use Committee-Ready Themes as the agenda backbone.",
        "- Use Safety, Privacy, Equity, and Oversight notes to separate evidence-backed recommendations from governance questions.",
        "",
        "---",
        f"*Generated by Sysys for {brief_cfg['audience']}.*",
    ])

    return "\n".join(lines) + "\n"


def render_abct_tech_ai_infosheet(
    config, prisma_data, checklist: list[dict], brief_cfg: dict | None = None
) -> str:
    """Render a concise one-page committee info-sheet handout."""
    research = getattr(config, "research_config", {}) or {}
    brief_cfg = brief_cfg or _brief_config(
        research,
        config_section="abct_tech_ai_infosheet",
        defaults=DEFAULT_INFOSHEET,
    )
    artifacts = _collect_artifacts(config)
    counts = _counts(artifacts)
    meeting = _format_date(brief_cfg["meeting_date"])
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    focus = _clean_text(
        research.get("natural_language_question")
        or (research.get("relevance") or {}).get("research_question")
        or getattr(config, "hypothesis_text", "")
        or "Practical guidance from the evidence for the configured research question."
    )

    lines = [
        f"# {brief_cfg['title']}",
        "",
        f"**Audience:** {brief_cfg['audience']}  ",
        f"**Target meeting:** {meeting}  ",
        f"**Generated:** {generated}",
        "",
    ]
    subtitle = brief_cfg.get("subtitle")
    if subtitle:
        lines.extend([f"*{_clean_text(str(subtitle))}*", ""])

    lines.extend([
        "## Purpose",
        "",
        _truncate(focus, 420),
        "",
        f"## {_takeaways_heading(brief_cfg['meeting_date'])}",
        "",
    ])
    lines.extend(f"- {item}" for item in _infosheet_config_lines(brief_cfg, "key_messages", 5))

    lines.extend([
        "",
        "## Practical Guidance Checks",
        "",
    ])
    lines.extend(f"- {item}" for item in _infosheet_config_lines(brief_cfg, "practical_guidance", 5))

    lines.extend([
        "",
        "## Evidence Status",
        "",
        f"- Identified/screened: {prisma_data.records_identified} / {prisma_data.records_screened}",
        f"- Retained/extracted/synthesized: {counts['relevant']} / {counts['claims']} / {prisma_data.studies_in_synthesis}",
        f"- Current threshold: >= {getattr(config, 'relevance_threshold', 3)} on the configured 0-5 relevance scale",
        *_certainty_lines(artifacts),
        "",
        "## Committee Discussion Prompts",
        "",
    ])
    for question in brief_cfg["decision_questions"]:
        lines.append(f"- {question}")

    risk_lines = _risk_lines(artifacts)
    lines.extend([
        "",
        "## Evidence-Backed Cautions",
        "",
    ])
    lines.extend(risk_lines[:5] if risk_lines else [
        "- Run extraction to populate source-backed notes on safety, privacy, equity, oversight, and implementation risk.",
    ])

    lines.extend([
        "",
        "## Starter Reading List",
        "",
    ])
    lines.extend(_top_paper_lines(artifacts["relevant"] or artifacts["corpus"])[:5])

    lines.extend([
        "",
        "## Export Notes",
        "",
        "- Full PRISMA-trAIce checklist, bibliography, flow diagram, and available synthesis artifacts are included in this export pack.",
        "- Use this page as the handout; use the supporting files when the committee asks for source traceability.",
        "",
        "---",
        f"*Generated by Sysys for {brief_cfg['audience']}.*",
    ])

    return "\n".join(lines) + "\n"


def _brief_config(research: dict, *, config_section: str, defaults: dict) -> dict:
    config = dict(defaults)
    custom = research.get(config_section) if isinstance(research, dict) else None
    if isinstance(custom, dict):
        config.update({k: v for k, v in custom.items() if v not in (None, "")})
    if not isinstance(config.get("decision_questions"), list):
        config["decision_questions"] = defaults["decision_questions"]
    config["filename"] = _safe_brief_filename(config.get("filename"), defaults["filename"])
    return config


def _safe_brief_filename(name: Any, default: str) -> str:
    """Bare ``.md`` file name: no directories, so the brief stays in the pack (15-L8)."""
    stem = Path(str(name or "")).stem
    return f"{stem}.md" if stem and stem not in {".", ".."} else default


def _takeaways_heading(meeting_date: Any) -> str:
    """"June 22 Takeaways" for a dated meeting, "Key Takeaways" otherwise (15-L7)."""
    try:
        date = datetime.fromisoformat(str(meeting_date or "").strip())
    except ValueError:
        return "Key Takeaways"
    return f"{date:%B} {date.day} Takeaways"


def _certainty_lines(artifacts: dict[str, Any]) -> list[str]:
    """One line per GRADE-assessed component from ``grade_results.json``."""
    rows = artifacts.get("grade")
    lines = [
        f"- {row.get('component_label') or row.get('component_id')}: "
        f"{row['overall_certainty']} certainty ({row.get('n_studies', 0)} studies)"
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, dict) and row.get("overall_certainty")
    ]
    return lines or ["- GRADE has not been run; certainty of evidence is not yet rated."]


def _collect_artifacts(config) -> dict[str, Any]:
    return {
        "corpus": _load_json_list(config.corpus_dir / "corpus.json"),
        "relevant": _load_json_list(config.claims_dir / "relevant.json"),
        "claims": _load_json_list(config.claims_dir / "claims_filtered.json")
        or _load_json_list(config.claims_dir / "claims.json"),
        "clusters": _load_json_list(config.clusters_dir / "clusters.json"),
        "narratives": _load_json_list(config.clusters_dir / "narratives.json"),
        "abstract": _read_first_text(
            config.clusters_dir / "abstract_draft.md",
            config.output_dir / "synthesis_abstract.md",
            config.output_dir / "abstract_draft.md",
        ),
        "grade": _load_json(config.output_dir / "grade_results.json"),
    }


def _copy_supporting_artifacts(config, output_dir: Path) -> list[str]:
    copied: list[str] = []
    file_map = [
        (config.clusters_dir / "abstract_draft.md", "synthesis_abstract_draft.md"),
        (config.clusters_dir / "narratives.json", "synthesis_narratives.json"),
        (config.clusters_dir / "clusters.json", "synthesis_clusters.json"),
        (config.output_dir / "grade_sof_table.md", "grade_summary_of_findings.md"),
        (config.output_dir / "grade_results.json", "grade_results.json"),
        (config.data_dir / "validation" / "robustness_results.json", "robustness_results.json"),
        (config.data_dir / "validation" / "verification_results.json", "verification_results.json"),
    ]
    for src, dest in file_map:
        if src.exists():
            shutil.copy2(src, output_dir / dest)
            copied.append(dest)
    return copied


def _write_readme(output_dir: Path, brief_cfg: dict, files: list[str], *, title: str) -> None:
    descriptions = {
        brief_cfg["filename"]: "Primary committee-facing Markdown output",
        "prisma_flow_diagram.png": "PRISMA flow diagram",
        "prisma_flow_diagram.svg": "Editable PRISMA flow diagram",
        "prisma_traice_checklist.md": "PRISMA 2020 checklist with trAIce extension",
        "prisma_traice_checklist.csv": "Editable PRISMA-trAIce checklist",
        "abct_brief_corpus.bib": "Brief bibliography in BibTeX",
        "abct_brief_corpus.ris": "Brief bibliography in RIS",
        "abct_infosheet_corpus.bib": "Info-sheet bibliography in BibTeX",
        "abct_infosheet_corpus.ris": "Info-sheet bibliography in RIS",
        "synthesis_abstract_draft.md": "Synthesis abstract draft copied from Stage 6",
        "synthesis_narratives.json": "Cluster narratives copied from Stage 6",
        "synthesis_clusters.json": "Cluster metadata copied from Stage 6",
        "grade_summary_of_findings.md": "GRADE summary of findings",
        "grade_results.json": "GRADE assessment results",
        "robustness_results.json": "Robustness checks",
        "verification_results.json": "Abstract verification checks",
    }
    lines = [
        f"# {title}",
        "",
        f"Audience: {brief_cfg['audience']}",
        f"Target meeting: {_format_date(brief_cfg['meeting_date'])}",
        "",
        "| File | Description |",
        "|------|-------------|",
    ]
    for name in sorted(set(files)):
        lines.append(f"| `{name}` | {descriptions.get(name, 'Supporting pipeline artifact')} |")
    atomic_write_text(output_dir / "README.md", "\n".join(lines) + "\n")


def _counts(artifacts: dict[str, Any]) -> dict[str, int]:
    return {
        "corpus": len(artifacts.get("corpus") or []),
        "relevant": len(artifacts.get("relevant") or []),
        "claims": len(artifacts.get("claims") or []),
    }


def _synthesis_summary(artifacts: dict[str, Any]) -> str:
    abstract = artifacts.get("abstract") or ""
    if abstract:
        return _truncate(_clean_text(abstract), 1400)

    narratives = artifacts.get("narratives") or []
    if narratives:
        snippets = []
        for item in narratives[:3]:
            text = item.get("narrative") or item.get("summary") or str(item)
            snippets.append(f"- {_truncate(_clean_text(text), 360)}")
        return "\n".join(snippets)

    return "Run Audit & Synthesize to populate a narrative synthesis for the committee brief."


def _theme_lines(artifacts: dict[str, Any], research: dict) -> list[str]:
    narratives = artifacts.get("narratives") or []
    if narratives:
        lines = []
        for item in narratives[:6]:
            label = item.get("cluster_label") or item.get("label") or item.get("title") or "Evidence theme"
            text = item.get("narrative") or item.get("summary") or ""
            lines.append(f"- **{_clean_text(str(label))}:** {_truncate(_clean_text(text), 260)}")
        return lines

    axes = research.get("axes") if isinstance(research, dict) else []
    if isinstance(axes, list) and axes:
        return [
            f"- **{axis.get('display_name', axis.get('id', 'Review axis'))}:** pending synthesis."
            for axis in axes[:6]
            if isinstance(axis, dict)
        ]
    return []


def _risk_lines(artifacts: dict[str, Any]) -> list[str]:
    claims = artifacts.get("claims") or []
    notes: list[str] = []
    for paper in claims:
        claim_data = paper.get("claims") if isinstance(paper, dict) else None
        if not isinstance(claim_data, dict):
            continue
        for key in ("safety_privacy_equity", "implementation_notes", "clinical_readiness"):
            value = claim_data.get(key)
            for item in _iter_text_values(value):
                if item and item.lower() not in {"none", "not reported", "not specified"}:
                    notes.append(item)
    return [f"- {_truncate(_clean_text(note), 220)}" for note in _unique(notes, limit=8)]


def _infosheet_config_lines(config: dict, key: str, limit: int) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list):
        return []
    lines = [_clean_text(str(item)) for item in value if _clean_text(str(item))]
    return lines[:limit]


def _top_paper_lines(papers: list[dict]) -> list[str]:
    if not papers:
        return ["- Run Search and Score & Filter to populate top relevant papers."]

    scored = sorted(
        papers,
        key=lambda p: (
            _score_value(p),
            str(p.get("year") or ""),
            str(p.get("title") or ""),
        ),
        reverse=True,
    )
    lines = []
    for paper in scored[:8]:
        title = _clean_text(str(paper.get("title") or "Untitled"))
        year = paper.get("year") or "n.d."
        score = _score_value(paper)
        score_text = f", score {score:g}" if score >= 0 else ""
        journal = paper.get("journal") or paper.get("venue") or ""
        suffix = f" ({journal})" if journal else ""
        lines.append(f"- {title} ({year}{score_text}){suffix}")
    return lines


def _gap_lines(artifacts: dict[str, Any], checklist: list[dict]) -> list[str]:
    gaps: list[str] = []
    if not artifacts.get("relevant"):
        gaps.append("Complete relevance screening before using the brief for recommendations.")
    if not artifacts.get("claims"):
        gaps.append("Run extraction/validation to populate readiness and safety notes.")
    if not artifacts.get("abstract") and not artifacts.get("narratives"):
        gaps.append("Run synthesis to convert extracted evidence into committee-ready themes.")
    manual_items = [item["item"] for item in checklist if item.get("status") == "manual"]
    if manual_items:
        gaps.append(f"Complete manual PRISMA checklist items: {', '.join(map(str, manual_items[:10]))}.")
    return [f"- {gap}" for gap in gaps] or ["- No immediate export gaps detected from available pipeline artifacts."]


def _source_list(records_per_source: dict) -> str:
    if not records_per_source:
        return "configured databases; source counts pending"
    return ", ".join(f"{source} ({count})" for source, count in records_per_source.items())


def _ai_tool_list(ai_tools: list[dict]) -> str:
    if not ai_tools:
        return "pending"
    return "; ".join(f"{tool.get('name', 'unknown')} for {tool.get('role', 'unspecified role')}" for tool in ai_tools)


def _performance_summary(metrics: dict) -> str:
    if not metrics:
        return "not evaluated against human labels"
    return "; ".join(f"{key}: {value}" for key, value in metrics.items())


def _list_or_pending(values: list[str]) -> str:
    return "; ".join(values) if values else "pending documentation"


def _score_value(paper: dict) -> float:
    for key in ("relevance_score", "score", "mean_score"):
        value = paper.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    llm = paper.get("llm_result")
    if isinstance(llm, dict) and isinstance(llm.get("llm_relevance"), (int, float)):
        return float(llm["llm_relevance"])
    return -1.0


def _load_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _load_json_list(path: Path) -> list[dict]:
    data = _load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        articles = data.get("articles")
        if isinstance(articles, list):
            return articles
        return [data]
    return []


def _read_first_text(*paths: Path) -> str:
    for path in paths:
        try:
            if path.exists():
                return path.read_text(encoding="utf-8")
        except OSError:
            continue
    return ""


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _truncate(text: str, limit: int) -> str:
    text = _clean_text(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _format_date(value: str) -> str:
    if not str(value or "").strip():
        return "not scheduled"
    try:
        return datetime.fromisoformat(str(value)).strftime("%B %d, %Y")
    except ValueError:
        return str(value)


def _iter_text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, dict):
        return [f"{key}: {val}" for key, val in value.items() if val not in (None, "")]
    return []


def _unique(values: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = _clean_text(value).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= limit:
            break
    return out
