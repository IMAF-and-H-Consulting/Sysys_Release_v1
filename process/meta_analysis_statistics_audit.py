"""Write and summarise the per-statistic audit (what was found, used or passed up)."""
from __future__ import annotations

import csv
from collections import Counter
import io
from pathlib import Path

from core.io import atomic_write_text

STATISTICS_AUDIT_FILE = "statistics_audit.csv"
STATISTICS_AUDIT_FIELDS = (
    "effect_id", "article_id", "title", "test_type", "test_text", "effects", "confidence_intervals",
    "standard_error", "p_values", "sample_sizes", "paper_sample_sizes",
    "degrees_of_freedom", "llm_verdict", "status", "reason_code", "reason",
    "effect_used", "route", "analysis_scale", "analysis_estimate", "analysis_se",
    "analysis_group", "include", "direction", "uncertainty_source", "assumptions", "outcome", "comparison",
    "population", "section", "char_start",
)
_ROUTE_LABELS = {
    "t_to_d": "t → d", "f_to_d": "F(1, df) → d", "chi2_to_phi": "χ²(1) → φ",
    "z_to_r": "z → r", "p_to_r": "p-value only → r",
}
_SOURCE_LABELS = {
    "confidence_interval": "confidence interval",
    "standard_error": "reported SE",
    "p_value": "exact p-value",
    "sample_size": "result sample size",
    "degrees_of_freedom": "r(df), n = df + 2",
    "paper_sample_size": "paper-level N",
    **{f"conversion:{route}": f"included conversion {label}" for route, label in _ROUTE_LABELS.items()},
}


def write_statistics_audit(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=STATISTICS_AUDIT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(Path(path), buffer.getvalue())


def found_lines(rows: list[dict]) -> list[str]:
    """What kinds of statistic were found, e.g. 'Found by type: f_test 771, t_test 258'."""
    counts = Counter(row.get("test_type") or "unknown" for row in rows)
    if not counts:
        return ["Found by type: none"]
    return ["Found by type: " + ", ".join(f"{kind} {n}" for kind, n in counts.most_common())]


def funnel_lines(funnel: dict | None) -> list[str]:
    """Readable lines for logs and reports from ``statistics_funnel`` output."""
    if not funnel:
        return []
    lines = [f"Statistics extracted: {funnel['statistics']}; calculable effects: "
             f"{funnel['calculable']}; converted (awaiting your decision): "
             f"{funnel.get('converted', 0)}; passed up: {funnel['passed_up']}"]
    routes = funnel.get("converted_by_route") or {}
    if routes:
        lines.append("  Converted from test statistics: " + ", ".join(
            f"{_ROUTE_LABELS.get(k, k)} {v}" for k, v in sorted(routes.items(), key=lambda x: -x[1])))
    sources = funnel.get("by_uncertainty_source") or {}
    if sources:
        lines.append("  Uncertainty taken from: " + ", ".join(
            f"{_SOURCE_LABELS.get(k, k)} {v}" for k, v in sorted(sources.items(), key=lambda x: -x[1])))
    for reason, count in (funnel.get("passed_up_by_reason") or {}).items():
        lines.append(f"  Passed up — {reason}: {count}")
    return lines
