"""
PRISMA 2020 reporting module for the Sysys pipeline.

Collects pipeline output data into PRISMA-required counts, generates a
publication-quality PRISMA 2020 flow diagram using matplotlib, and produces
a PRISMA 2020 checklist with PRISMA-trAIce AI-transparency extensions.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from core.io import load_json as _load_json
from core.models import (
    REMOVAL_LOG_FILENAME,
    SEARCH_MANIFEST_FILENAME,
    summarize_search_manifest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PRISMAData:
    """All counts and metadata needed for PRISMA 2020 reporting."""

    # Identification
    records_identified: int = 0
    records_per_source: dict = field(default_factory=dict)
    records_removed_duplicates: int = 0
    records_removed_no_abstract: int = 0
    records_removed_by_filters: int = 0
    # "search_manifest", "analysis_session" (legacy) or "corpus_fallback"
    # (no search record: identification shows the deduplicated corpus size).
    identification_source: str = ""
    failed_sources: list = field(default_factory=list)
    search_queries: dict = field(default_factory=dict)

    # Screening
    records_screened: int = 0
    records_excluded_screening: int = 0
    screening_threshold: int = 3
    screening_method: str = ""

    # Eligibility
    reports_sought_fulltext: int = 0
    reports_not_retrieved: int = 0
    reports_assessed_eligibility: int = 0
    reports_excluded_reasons: dict = field(default_factory=dict)

    # Included
    studies_in_review: int = 0
    studies_in_synthesis: int = 0

    # AI-specific (PRISMA-trAIce)
    ai_tools: list = field(default_factory=list)
    human_oversight_points: list = field(default_factory=list)
    ai_performance: dict = field(default_factory=dict)

    # Box-arithmetic checks that failed (15-M5); empty when the flow adds up.
    reconciliation_warnings: list = field(default_factory=list)

    @property
    def records_removed_before_screening(self) -> int:
        return (self.records_removed_duplicates + self.records_removed_no_abstract
                + self.records_removed_by_filters)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _download_log_rows(config) -> list[dict]:
    """Load full-text download records from the stage logs, if any exist."""
    candidates = [
        getattr(config, "data_dir", None) and config.data_dir / "fulltext" / "download_log.json",
        getattr(config, "data_dir", None) and config.data_dir / "fulltext_corpus" / "download_log.json",
    ]
    for path in candidates:
        if path is None:
            continue
        payload = _load_json(path)
        rows: list[dict] = []
        if isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict)]
        elif isinstance(payload, dict):
            for key in ("downloads", "papers", "results", "records"):
                inner = payload.get(key)
                if isinstance(inner, list):
                    rows = [row for row in inner if isinstance(row, dict)]
                    break
        if rows:
            return rows
    return []


def _report_was_retrieved(row: dict) -> bool:
    """True when a download-log row says the full text was obtained."""
    if row.get("success") is True:
        return True
    if row.get("success") is False:
        return False
    status = str(row.get("status") or row.get("download_status") or "").strip().lower()
    if status in {"ok", "success", "downloaded", "cached"}:
        return True
    if row.get("file") or row.get("saved_file") or row.get("text"):
        return status not in {"failed", "no-text", "error", "not_found", "missing"}
    return False


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _apply_search_manifest(data: PRISMAData, manifest: Any) -> bool:
    """Fill identification counts from ``search_manifest.json``; False if unusable."""
    summary = summarize_search_manifest(manifest)
    if summary is None:
        return False
    data.records_identified = summary.records_identified
    data.records_per_source = dict(summary.records_per_source)
    data.records_removed_duplicates = summary.duplicates_removed
    data.records_removed_no_abstract = summary.no_abstract_removed
    data.records_removed_by_filters = summary.removed_by_filters
    data.failed_sources = list(summary.failed_sources)
    data.search_queries = {k: list(v) for k, v in summary.queries.items()}
    data.identification_source = "search_manifest"
    return True


def _apply_session_counts(data: PRISMAData, session: Any, corpus_count: int) -> bool:
    """Legacy ``analysis_session.json`` ``raw_search_counts``; False if absent."""
    counts = session.get("raw_search_counts") if isinstance(session, dict) else None
    if not isinstance(counts, dict) or not counts:
        return False
    data.records_per_source = {str(k): _as_int(v) for k, v in counts.items()}
    data.records_identified = sum(data.records_per_source.values())
    data.records_removed_duplicates = max(data.records_identified - corpus_count, 0)
    data.identification_source = "analysis_session"
    return True


def _apply_search_provenance(data: PRISMAData, config, corpus_count: int) -> None:
    """Identification counts: search manifest, legacy session, else corpus size."""
    manifest = _load_json(config.corpus_dir / SEARCH_MANIFEST_FILENAME)
    if _apply_search_manifest(data, manifest):
        return
    session = _load_json(config.output_dir / "analysis_session.json")
    if _apply_session_counts(data, session, corpus_count):
        return
    # No record of the search: the deduplicated corpus is all that is known.
    data.records_identified = corpus_count
    data.identification_source = "corpus_fallback"


def _load_removal_log(config) -> Any:
    """Stage 5's removal log (claims dir), falling back to the legacy clusters dir."""
    for directory in (config.claims_dir, config.clusters_dir):
        removal_log = _load_json(directory / REMOVAL_LOG_FILENAME)
        if removal_log is not None:
            return removal_log
    return None


def _exclusion_reasons(removal_log: Any) -> dict[str, int]:
    """Count exclusions by reason, ignoring entries for papers that were kept."""
    reasons: dict[str, int] = {}
    if isinstance(removal_log, list):
        entries = [e for e in removal_log if isinstance(e, dict) and e.get("removed", True)]
        labels = [str(e.get("reason", "unspecified")) for e in entries]
    elif isinstance(removal_log, dict):
        labels = [
            str(r.get("reason", "unspecified")) if isinstance(r, dict) else str(r)
            for r in removal_log.values()
        ]
    else:
        labels = []
    for label in labels:
        reasons[label] = reasons.get(label, 0) + 1
    return reasons


def _reconciliation_warnings(data: PRISMAData, relevant_count: int) -> list[str]:
    """Return a message for every PRISMA box that does not follow from the one above."""
    warnings: list[str] = []
    expected_screened = data.records_identified - data.records_removed_before_screening
    if data.identification_source == "search_manifest" and expected_screened != data.records_screened:
        warnings.append(
            f"the search manifest accounts for {expected_screened} records but "
            f"corpus.json has {data.records_screened}; the corpus changed after "
            "the last recorded search"
        )
    if relevant_count > data.records_screened:
        warnings.append(
            f"{relevant_count} records passed screening but only "
            f"{data.records_screened} were screened"
        )
    excluded = sum(data.reports_excluded_reasons.values())
    if data.reports_assessed_eligibility - excluded != data.studies_in_review:
        warnings.append(
            f"reports assessed ({data.reports_assessed_eligibility}) minus excluded "
            f"({excluded}) does not equal studies included ({data.studies_in_review})"
        )
    return warnings


def collect_prisma_data(
    config,
) -> PRISMAData:
    """Walk all pipeline output files and collect PRISMA-required counts.

    Handles missing files gracefully — returns 0 for stages that have not run.
    """
    data = PRISMAData()

    # -- Screening: corpus → relevant --------------------------------------
    corpus = _load_json(config.corpus_dir / "corpus.json")
    relevant = _load_json(config.claims_dir / "relevant.json")

    if isinstance(corpus, list):
        corpus_count = len(corpus)
    elif isinstance(corpus, dict):
        corpus_count = len(corpus.get("articles", corpus))
    else:
        corpus_count = 0

    if isinstance(relevant, list):
        relevant_count = len(relevant)
    elif isinstance(relevant, dict):
        relevant_count = len(relevant.get("articles", relevant))
    else:
        relevant_count = 0

    # -- Identification: Stage 1 search manifest -------------------------
    data.records_screened = corpus_count
    _apply_search_provenance(data, config, corpus_count)
    data.records_excluded_screening = max(corpus_count - relevant_count, 0)
    data.screening_threshold = getattr(config, "relevance_threshold", 3)

    # -- Eligibility: claims_filtered + removal_log ------------------------
    claims_filtered = _load_json(config.claims_dir / "claims_filtered.json")
    removal_log = _load_removal_log(config)

    if isinstance(claims_filtered, list) or isinstance(claims_filtered, dict):
        claims_count = len(claims_filtered)
    else:
        claims_count = 0

    download_rows = _download_log_rows(config)
    retrieved = sum(1 for row in download_rows if _report_was_retrieved(row))
    if download_rows:
        data.reports_sought_fulltext = len(download_rows)
        data.reports_not_retrieved = len(download_rows) - retrieved
        data.reports_assessed_eligibility = retrieved
    else:
        data.reports_sought_fulltext = relevant_count
        data.reports_not_retrieved = 0
        data.reports_assessed_eligibility = relevant_count

    data.reports_excluded_reasons = _exclusion_reasons(removal_log)

    # -- Included: clusters / synthesis ------------------------------------
    clusters = _load_json(config.clusters_dir / "clusters.json")

    if isinstance(clusters, list):
        # Count unique article ids across clusters
        article_ids = set()
        for cl in clusters:
            if isinstance(cl, dict):
                for member in cl.get("members", cl.get("articles", [])):
                    if isinstance(member, str):
                        article_ids.add(member)
                    elif isinstance(member, dict):
                        article_ids.add(member.get("article_id", ""))
        data.studies_in_synthesis = len(article_ids) if article_ids else len(clusters)
    elif isinstance(clusters, dict):
        data.studies_in_synthesis = len(clusters)

    data.studies_in_review = claims_count if claims_count > 0 else relevant_count
    if data.studies_in_synthesis == 0:
        data.studies_in_synthesis = data.studies_in_review
    data.reconciliation_warnings = _reconciliation_warnings(data, relevant_count)
    for warning in data.reconciliation_warnings:
        logger.warning("PRISMA counts do not reconcile: %s", warning)

    # -- AI tools ----------------------------------------------------------
    provider = getattr(config, "llm_provider", "unknown")
    model = getattr(config, "llm_model", "unknown")
    data.ai_tools = [
        {
            "name": f"{provider} / {model}",
            "role": "relevance scoring, claim extraction, synthesis",
            "version": model,
        }
    ]

    # Describe the saved screening run when available, rather than claiming
    # a newly selected strategy produced an older corpus.
    screening = _load_json(config.claims_dir / "screening_report.json")
    scored = _load_json(config.claims_dir / "scored_all.json")
    first_screening = {}
    if isinstance(scored, list) and scored and isinstance(scored[0], dict):
        first_screening = scored[0].get("screening") or {}
        mode = first_screening.get("mode", "off")
    else:
        mode = (screening.get("mode") if isinstance(screening, dict) else None)
        mode = mode or getattr(config, "relevance_screening_mode", "laya-bert")
    opts = screening.get("settings", {}) if isinstance(screening, dict) else {}
    # Per-paper provenance wins over current settings or a different run report.
    opts = {**opts, **first_screening}
    if mode != "off":
        cut = opts.get("cut_bin", getattr(config, "laya_cut_bin", 4))
        threshold = opts.get("brt_threshold", getattr(config, "laya_brt_threshold", 0.236))
        gates = []
        if mode != "packaged-bert":
            gates.append(f"Laya bin >= {cut}")
            data.ai_tools.append({"name": "Laya", "role": "relevance filtering", "version": "pw4"})
        if mode != "laya-only":
            gates.append(f"BRT-BERT P >= {threshold:g}")
            data.ai_tools.append({"name": "BRT-BERT", "role": "relevance filtering", "version": "pw4"})
        data.screening_method = f"{mode}: " + " AND ".join(gates) + "."
        data.ai_tools[0]["role"] = "claim extraction, synthesis"
    else:
        data.screening_method = f"Relevance threshold >= {data.screening_threshold} on 0-5 scale."

    data.human_oversight_points = [
        "Hypothesis and search strategy defined by researcher",
        "Screening strategy and cutoffs configurable by researcher",
        "Stage 2 human review supports rating and correction of model decisions",
        "Span-verification provides auditable evidence trail",
        "Audit stage flags duplicate / suspicious spans for review",
        "Robustness analysis tests conclusion stability",
    ]
    # Cascade decisions and invocation counts are not independent agreement or
    # effectiveness metrics. Quality evaluation requires human-labelled data.

    return data


# ---------------------------------------------------------------------------
# Flow diagram
# ---------------------------------------------------------------------------

def _draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    header: str = "",
    fontsize: float = 8,
    header_fontsize: float = 9,
    facecolor: str = "#ffffff",
    header_color: str = "#000000",
):
    """Draw a rounded box with optional bold header line."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=facecolor,
        edgecolor="#000000",
        linewidth=1.0,
    )
    ax.add_patch(box)
    if header:
        ax.text(
            x + w / 2, y + h - 0.025, header,
            ha="center", va="top",
            fontsize=header_fontsize, fontweight="bold", color=header_color,
        )
        ax.text(
            x + w / 2, y + h / 2 - 0.015, text,
            ha="center", va="center",
            fontsize=fontsize, color="#333333",
            wrap=True,
        )
    else:
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color="#333333",
        )


def _draw_arrow(ax, x1, y1, x2, y2):
    """Draw a simple arrow between two points."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#000000",
            lw=1.2,
            connectionstyle="arc3,rad=0",
        ),
    )


def _source_breakdown(data: PRISMAData) -> str:
    if not data.records_per_source:
        return ""
    parts = [f"{src}: {n}" for src, n in sorted(data.records_per_source.items())]
    return "\n".join(parts)


def _draw_reconciliation_note(ax, data: PRISMAData) -> None:
    """Print failed box-arithmetic checks under the diagram."""
    if not data.reconciliation_warnings:
        return
    note = "Count check: " + "; ".join(data.reconciliation_warnings)
    ax.text(0.5, 0.21, note, ha="center", va="top", fontsize=6,
            color="#b00000", wrap=True)


def _render_flow_diagram(
    data: PRISMAData,
    figsize: tuple[float, float],
    dpi: int,
    title_fontsize: int,
) -> Figure:
    """Shared PRISMA 2020 flow-diagram renderer (single layout source)."""
    fig = Figure(figsize=figsize, dpi=dpi, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(
        0.5, 0.97, "PRISMA 2020 Flow Diagram",
        ha="center", va="top", fontsize=title_fontsize, fontweight="bold",
    )

    # Layout constants
    lx, lw = 0.08, 0.48   # left column
    rx, rw = 0.62, 0.32   # right column (exclusion boxes)
    bh = 0.10              # box height
    bh_small = 0.08

    # ── IDENTIFICATION ────────────────────────────────────────────────
    y_id = 0.84
    source_text = _source_breakdown(data)
    id_body = f"Records identified (n = {data.records_identified})"
    if data.identification_source == "corpus_fallback":
        id_body += "\n(per-source search counts not recorded)"
    if source_text:
        id_body += "\n" + source_text
    _draw_box(ax, lx, y_id, lw, bh, id_body,
              header="IDENTIFICATION", facecolor="#f0f8ff")

    # ── Duplicates removed ────────────────────────────────────────────
    y_dedup = 0.72
    _draw_box(
        ax, lx, y_dedup, lw, bh_small,
        f"Records removed before screening (n = {data.records_removed_before_screening})\n"
        f"duplicates: {data.records_removed_duplicates}; "
        f"no abstract: {data.records_removed_no_abstract}; "
        f"search filters: {data.records_removed_by_filters}",
    )
    _draw_arrow(ax, lx + lw / 2, y_id, lx + lw / 2, y_dedup + bh_small)

    # ── SCREENING ─────────────────────────────────────────────────────
    y_scr = 0.58
    _draw_box(
        ax, lx, y_scr, lw, bh,
        f"Records screened\n(n = {data.records_screened})",
        header="SCREENING", facecolor="#f0fff0",
    )
    _draw_arrow(ax, lx + lw / 2, y_dedup, lx + lw / 2, y_scr + bh)

    # Exclusion box (right)
    _draw_box(
        ax, rx, y_scr + 0.01, rw, bh_small,
        f"Records excluded\n(n = {data.records_excluded_screening}, "
        "failed relevance screening)",
    )
    _draw_arrow(ax, lx + lw, y_scr + bh / 2, rx, y_scr + 0.01 + bh_small / 2)

    # ── ELIGIBILITY ───────────────────────────────────────────────────
    y_el = 0.42
    _draw_box(
        ax, lx, y_el, lw, bh,
        f"Reports sought for retrieval (n = {data.reports_sought_fulltext})\n"
        f"Reports not retrieved (n = {data.reports_not_retrieved})\n"
        f"Reports assessed for eligibility (n = {data.reports_assessed_eligibility})",
        header="ELIGIBILITY", facecolor="#fffff0",
    )
    _draw_arrow(ax, lx + lw / 2, y_scr, lx + lw / 2, y_el + bh)

    # Exclusion reasons box (right)
    total_excl = sum(data.reports_excluded_reasons.values())
    reason_lines = [f"Reports excluded (n = {total_excl})"]
    for reason, count in sorted(
        data.reports_excluded_reasons.items(), key=lambda x: -x[1]
    )[:5]:
        reason_lines.append(f"  {reason}: {count}")
    excl_text = "\n".join(reason_lines) if reason_lines else "Reports excluded (n = 0)"
    excl_h = max(bh_small, 0.02 * len(reason_lines) + 0.04)
    _draw_box(ax, rx, y_el + 0.01, rw, excl_h, excl_text, fontsize=7)
    _draw_arrow(
        ax, lx + lw, y_el + bh / 2,
        rx, y_el + 0.01 + excl_h / 2,
    )

    # ── INCLUDED ──────────────────────────────────────────────────────
    y_inc = 0.24
    _draw_box(
        ax, lx, y_inc, lw, bh + 0.02,
        f"Studies included in review (n = {data.studies_in_review})\n"
        f"Studies included in synthesis (n = {data.studies_in_synthesis})",
        header="INCLUDED", facecolor="#fff0f5",
    )
    _draw_arrow(ax, lx + lw / 2, y_el, lx + lw / 2, y_inc + bh + 0.02)

    _draw_reconciliation_note(ax, data)

    # ── AI transparency note ──────────────────────────────────────────
    if data.ai_tools:
        tool_names = ", ".join(t.get("name", "?") for t in data.ai_tools)
        ai_note = f"AI tools used: {tool_names}"
        ax.text(
            0.5, 0.17, ai_note,
            ha="center", va="top", fontsize=7, fontstyle="italic",
            color="#666666",
        )

    return fig


def generate_flow_diagram(
    data: PRISMAData,
    output_path: Path,
    fmt: str = "png",
) -> Path:
    """Generate a PRISMA 2020 flow diagram using matplotlib.

    Returns the *Path* to the saved file.
    """
    fig = _render_flow_diagram(data, figsize=(10, 12), dpi=150,
                               title_fontsize=14)
    # ── Save ──────────────────────────────────────────────────────────
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{fmt}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), format=fmt, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("PRISMA flow diagram saved to %s", output_path)
    return output_path


def generate_flow_diagram_figure(data: PRISMAData) -> Figure:
    """Generate the PRISMA flow diagram and return the matplotlib Figure.

    Used by the GUI to embed in FigureCanvasQTAgg without saving to disk.
    """
    return _render_flow_diagram(data, figsize=(8, 10), dpi=100,
                                title_fontsize=13)


# ---------------------------------------------------------------------------
# Checklist
# ---------------------------------------------------------------------------

_PRISMA_ITEMS = [
    ("Title", 1, "Identify the report as a systematic review"),
    ("Title", 2, "Provide a structured summary including background, objectives, methods, results, and conclusions"),
    ("Introduction", 3, "Describe the rationale for the review in the context of existing knowledge"),
    ("Introduction", 4, "Provide an explicit statement of the question being addressed"),
    ("Methods", 5, "Specify the eligibility criteria and information sources"),
    ("Methods", 6, "Describe all information sources searched"),
    ("Methods", 7, "Present the full search strategy for at least one database"),
    ("Methods", 8, "Describe the process for selecting studies"),
    ("Methods", 9, "Describe the process for data extraction"),
    ("Methods", 10, "List and define all variables for which data were sought"),
    ("Methods", 11, "Describe methods used to assess risk of bias"),
    ("Methods", 12, "Describe methods for handling data and combining results"),
    ("Methods", 13, "Describe the methods of synthesis"),
    ("Methods", 14, "Describe any methods used to explore publication bias"),
    ("Methods", 15, "Describe methods of additional analyses (sensitivity, subgroup)"),
    ("Results", 16, "Give numbers of studies screened, assessed, and included, with reasons for exclusions"),
    ("Results", 17, "For each study, present characteristics and cite the study"),
    ("Results", 18, "Present data on risk of bias for each study"),
    ("Results", 19, "Present all collected data for each outcome considered"),
    ("Results", 20, "Present results of each synthesis undertaken"),
    ("Results", 21, "Present results of additional analyses"),
    ("Results", 22, "Present results of risk of bias across studies"),
    ("Discussion", 23, "Summarise main findings including strength of evidence"),
    ("Discussion", 24, "Discuss limitations at study and outcome level"),
    ("Discussion", 25, "Provide a general interpretation in the context of other evidence"),
    ("Other", 26, "Describe sources of funding and role of funders"),
    ("Other", 27, "Indicate registration and where the protocol can be accessed"),
]

_AUTO_ITEMS = {1, 5, 6, 7, 8, 13, 16, 17, 20}


def _information_sources_text(data: PRISMAData) -> str:
    if not data.records_per_source:
        return ("Information sources: not recorded (no search manifest; "
                "re-run the Stage 1 search to record per-source counts)")
    text = f"Information sources: {', '.join(data.records_per_source)}"
    if data.failed_sources:
        text += f" (search failed: {', '.join(data.failed_sources)})"
    return text


def _search_strategy_text(data: PRISMAData) -> str:
    if not data.search_queries:
        return "Search queries auto-generated from hypothesis components and co-occurring conditions"
    parts = [f"{source}: {' | '.join(queries)}"
             for source, queries in data.search_queries.items() if queries]
    return "Search queries as run — " + "; ".join(parts)


def generate_checklist(data: PRISMAData) -> list[dict]:
    """Generate a PRISMA 2020 checklist (27 items + PRISMA-trAIce extension)."""
    checklist: list[dict] = []

    for section, item, description in _PRISMA_ITEMS:
        entry: dict[str, Any] = {
            "section": section,
            "item": item,
            "description": description,
            "status": "auto" if item in _AUTO_ITEMS else "manual",
            "content": "",
            "prisma_traice": False,
        }

        # Auto-fill where pipeline data is available
        if item == 1:
            entry["content"] = "Systematic review of evidence (AI-assisted)"
        elif item == 5:
            entry["content"] = (
                data.screening_method
                or f"Relevance threshold >= {data.screening_threshold} on 0-5 scale."
            )
        elif item == 6:
            entry["content"] = _information_sources_text(data)
        elif item == 7:
            entry["content"] = _search_strategy_text(data)
        elif item == 8:
            entry["content"] = (
                (data.screening_method or "Relevance screening.")
                + " Span verification; claim extraction and audit."
            )
        elif item == 13:
            entry["content"] = (
                "Narrative synthesis via LLM-generated cluster summaries, "
                "grouped by hypothesis component"
            )
        elif item == 16:
            entry["content"] = (
                f"Identified: {data.records_identified}; "
                f"Duplicates removed: {data.records_removed_duplicates}; "
                f"Removed (no abstract): {data.records_removed_no_abstract}; "
                f"Removed (search filters): {data.records_removed_by_filters}; "
                f"Screened: {data.records_screened}; "
                f"Excluded (screening): {data.records_excluded_screening}; "
                f"Assessed (eligibility): {data.reports_assessed_eligibility}; "
                f"Included in review: {data.studies_in_review}; "
                f"Included in synthesis: {data.studies_in_synthesis}"
            )
        elif item == 17:
            entry["content"] = (
                f"{data.studies_in_review} studies with extracted claims, "
                f"study designs classified via LLM"
            )
        elif item == 20:
            entry["content"] = (
                f"{data.studies_in_synthesis} studies included in synthesis "
                f"across hypothesis components"
            )

        checklist.append(entry)

    # PRISMA-trAIce extension items
    traice_items = [
        {
            "section": "AI Transparency",
            "item": 28,
            "description": "AI tool identification — name, version, and provider of each AI tool used",
            "status": "auto",
            "content": "; ".join(
                f"{t['name']} ({t['role']})" for t in data.ai_tools
            ) if data.ai_tools else "",
            "prisma_traice": True,
        },
        {
            "section": "AI Transparency",
            "item": 29,
            "description": "AI role description — what each AI tool did at each pipeline stage",
            "status": "auto",
            "content": (
                "Relevance scoring (abstract/fulltext), claim extraction, "
                "study design classification, narrative synthesis"
            ),
            "prisma_traice": True,
        },
        {
            "section": "AI Transparency",
            "item": 30,
            "description": "Human-AI interaction points — where human oversight occurred",
            "status": "auto",
            "content": "; ".join(data.human_oversight_points) if data.human_oversight_points else "",
            "prisma_traice": True,
        },
        {
            "section": "AI Transparency",
            "item": 31,
            "description": "AI performance metrics — agreement scores and verification rates",
            "status": "auto" if data.ai_performance else "manual",
            "content": "; ".join(
                f"{k}: {v}" for k, v in data.ai_performance.items()
            ) if data.ai_performance else "",
            "prisma_traice": True,
        },
    ]
    checklist.extend(traice_items)

    return checklist


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def save_checklist_markdown(checklist: list[dict], output_path: Path) -> Path:
    """Save checklist as a nicely formatted markdown file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# PRISMA 2020 Checklist",
        "",
        "| # | Section | Description | Status | Content |",
        "|---|---------|-------------|--------|---------|",
    ]
    for item in checklist:
        traice = " (trAIce)" if item.get("prisma_traice") else ""
        content = item["content"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {item['item']} | {item['section']}{traice} | "
            f"{item['description']} | {item['status']} | {content} |"
        )

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("PRISMA checklist (Markdown) saved to %s", output_path)
    return output_path


def save_checklist_csv(checklist: list[dict], output_path: Path) -> Path:
    """Save checklist as CSV for easy editing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Item", "Section", "Description", "Status", "Content", "PRISMA-trAIce"])
        for item in checklist:
            writer.writerow([
                item["item"],
                item["section"],
                item["description"],
                item["status"],
                item["content"],
                "Yes" if item.get("prisma_traice") else "No",
            ])

    logger.info("PRISMA checklist (CSV) saved to %s", output_path)
    return output_path
