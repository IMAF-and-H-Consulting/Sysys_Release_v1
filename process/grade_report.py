"""GRADE Summary of Findings table and the saved GRADE outputs.

``save_grade_outputs`` writes the two files that the synthesis stage, the
committee brief and the export bundle read:

- ``grade_results.json``: one record per component (domains included);
- ``grade_sof_table.md``: the Summary of Findings table in Markdown.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.io import atomic_write_json, atomic_write_text
from process.grade_models import ComponentGrade, DomainRating, GradeConcern

GRADE_RESULTS_FILENAME = "grade_results.json"
GRADE_SOF_FILENAME = "grade_sof_table.md"

SOF_DOMAINS = ("risk_of_bias", "inconsistency", "indirectness",
               "imprecision", "publication_bias")
_SOF_HEADERS = ("Component", "Studies", "Risk of bias", "Inconsistency",
                "Indirectness", "Imprecision", "Publication bias", "Certainty")


def _concern_label(concern: GradeConcern) -> str:
    if concern == GradeConcern.NO_CONCERN:
        return "No concern"
    if concern == GradeConcern.SERIOUS:
        return "Serious"
    return "Very serious"


def _domain_cell(rating: DomainRating | None) -> dict[str, Any]:
    if rating is None:
        return {"concern": "no_concern", "label": "Not assessed",
                "rationale": "", "downgrade": 0}
    return {"concern": rating.concern.value, "label": _concern_label(rating.concern),
            "rationale": rating.rationale, "downgrade": rating.downgrade_levels}


def generate_sof_table(grades: list[ComponentGrade]) -> list[dict]:
    """Generate a GRADE Summary of Findings table.

    Returns a list of row dicts with columns: component, n_studies,
    risk_of_bias, inconsistency, indirectness, imprecision,
    publication_bias, overall_certainty.  Each domain cell contains
    the concern level and a brief rationale.
    """
    rows: list[dict] = []
    for g in grades:
        domain_map = {d.domain: d for d in g.domains}
        rows.append({
            "component": g.component_label,
            "component_id": g.component_id,
            "n_studies": g.n_studies,
            **{key: _domain_cell(domain_map.get(key)) for key in SOF_DOMAINS},
            "overall_certainty": g.overall_certainty.value,
            "summary": g.summary,
        })
    return rows


def grade_result_records(grades: list[ComponentGrade]) -> list[dict]:
    """JSON-ready records, one per component, with every domain rating."""
    return [
        {
            "component_id": g.component_id,
            "component_label": g.component_label,
            "n_studies": g.n_studies,
            "starting_certainty": g.starting_certainty,
            "overall_certainty": g.overall_certainty.value,
            "summary": g.summary,
            "domains": [
                {"domain": d.domain, "concern": d.concern.value,
                 "downgrade_levels": d.downgrade_levels, "rationale": d.rationale,
                 "data_points": d.data_points}
                for d in g.domains
            ],
        }
        for g in grades
    ]


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def sof_table_markdown(rows: list[dict]) -> str:
    """Render ``generate_sof_table`` rows as a Markdown table."""
    lines = [
        "# GRADE Summary of Findings",
        "",
        "| " + " | ".join(_SOF_HEADERS) + " |",
        "|" + "---|" * len(_SOF_HEADERS),
    ]
    for row in rows:
        cells = [row["component"], row["n_studies"],
                 *(row[key]["label"] for key in SOF_DOMAINS), row["overall_certainty"]]
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    lines += ["", *(f"- **{_cell(r['component'])}:** {_cell(r['summary'])}" for r in rows), ""]
    return "\n".join(lines)


def save_grade_outputs(output_dir: Path, grades: list[ComponentGrade]) -> tuple[Path, Path]:
    """Write ``grade_results.json`` and ``grade_sof_table.md`` atomically."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / GRADE_RESULTS_FILENAME
    sof_path = output_dir / GRADE_SOF_FILENAME
    atomic_write_json(results_path, grade_result_records(grades))
    atomic_write_text(sof_path, sof_table_markdown(generate_sof_table(grades)))
    return results_path, sof_path
