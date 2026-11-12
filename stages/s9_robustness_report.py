"""Stage 8 robustness: Markdown report writers.

Split out of stages/s9_robustness.py, which re-exports these names.

Every section renders a failed or skipped analysis as an explicit
"Failed"/"Skipped" block, never as zero counts that read like a real empty
result (audit 12-M7).
"""

import logging
from pathlib import Path

from core.claims import PRIMARY_OUTCOME_FIELD
from core.io import atomic_write_text

# Same logger name as the original single module (stages.s9_robustness).
logger = logging.getLogger("stages.s9_robustness")

# Share of papers with an extracted outcome below which the report warns
# that missing outcomes (counted as non-supporting) dominate the denominators.
LOW_OUTCOME_COVERAGE = 0.5
# Rows of the network report's top co-occurrence table.
NETWORK_REPORT_TOP_EDGES = 30


# ── Shared blocks ─────────────────────────────────────────────────────────

def _blocked(reason: str, label: str = "Skipped") -> list[str]:
    return [f"> **{label}.** {reason}", ""]


def _section_status(section: dict) -> list[str]:
    """A Failed/Skipped block for an analysis that produced no result, else []."""
    if section.get("error"):
        return _blocked(f"{section['error']} — no result was produced.", "Failed")
    if section.get("skipped"):
        return _blocked(section.get("reason") or "No result was produced.")
    return []


def _outcome_lines(results: dict, n_papers: int) -> list[str]:
    """How many papers carry the outcome the support analyses count."""
    n_with = results.get("n_with_outcome")
    if n_with is None:
        return []
    lines = [f"**Papers with a usable outcome value:** {n_with} of {n_papers}"]
    if n_with == 0:
        lines.append("> **Warning.** No paper has a value for the outcome field, so "
                     "the outcome analyses (2, 3, 4 and 6) were skipped. Check the "
                     "field name and that Stage 3 extraction ran.")
    elif n_papers and n_with / n_papers < LOW_OUTCOME_COVERAGE:
        lines.append(f"> **Warning.** {n_papers - n_with} paper(s) have no outcome value "
                     "and count as non-supporting, which deflates every support rate.")
    return lines


# ── Section 1: bootstrap ──────────────────────────────────────────────────

def _bootstrap_lines(bs: dict, n_papers: int) -> list[str]:
    lines = ["## 1. Bootstrap Confidence Intervals", ""]
    if not bs or bs.get("skipped"):
        return lines + _blocked(bs.get("reason", "No result was produced."))
    excluded = bs.get("n_unscored", 0) + bs.get("n_unparseable", 0)
    lines += [
        f"- **Mean relevance score:** {bs['mean']:.3f}",
        f"- **95% CI:** [{bs['ci_lower']:.3f}, {bs['ci_upper']:.3f}]",
        f"- **Std:** {bs['std']:.3f}",
        f"- **Iterations:** {bs.get('n_iterations', 0):,}",
        (f"- **Papers with a relevance score:** "
        f"{bs.get('n_scored', n_papers)} of {n_papers}"),
    ]
    if excluded:
        lines.append(
            f"- **Excluded (not imputed):** {bs.get('n_unscored', 0)} without a score, "
            f"{bs.get('n_unparseable', 0)} with an unparseable score"
        )
    return lines + [""]


# ── Section 2: split-half ─────────────────────────────────────────────────

def _interval_text(section: dict, key: str) -> str:
    """``[lo, hi]`` for a split-half statistic, or a note when unavailable."""
    lower = section.get(f"{key}_ci_lower")
    upper = section.get(f"{key}_ci_upper")
    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
        return f"[{lower:.3f}, {upper:.3f}]"
    return "[interval unavailable]"


def _number_text(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "not estimable"


def _negative_agreement_note(sh: dict) -> list[str]:
    """Explain a negative median tau/ICC (a gradient-free corpus gives these).

    The two halves partition the corpus, so within one condition a high rate
    in one half implies a low rate in the other. Without real between-
    condition differences that pushes agreement below zero; a negative ICC is
    conventionally read as zero reliability, not as a reversed gradient.
    """
    values = [sh.get("kendall_tau"), sh.get("icc")]
    if not any(isinstance(v, (int, float)) and v < 0 for v in values):
        return []
    return [(
        "  - **Negative agreement** means the half-to-half sampling noise within "
        "each condition outweighs the differences between conditions: there is "
        "no reproducible difference between conditions (read as zero reliability, "
        "not as a reversed gradient)."
    )]


def _split_half_lines(sh: dict) -> list[str]:
    lines = ["## 2. Split-Half Reliability", ""]
    if not sh.get("estimable", False):
        return lines + _blocked(sh.get("note", "No result was produced."), "Not estimable")
    return lines + [
        ("Random halves of the corpus are compared on their per-condition "
        "support rates; the half-length correlation is projected to "
        "full corpus length with the Spearman-Brown formula."),
        "",
        f"- **Reliability (Spearman-Brown corrected):** {sh.get('mean_correlation', 0):.3f}",
        f"- **Half-length correlation (uncorrected):** {sh.get('mean_half_correlation', 0):.3f}",
        (f"- **Kendall τ (median per split):** {_number_text(sh.get('kendall_tau'))} "
        f"{_interval_text(sh, 'kendall_tau')}"),
        (f"- **ICC(1,1) (median per split):** {_number_text(sh.get('icc'))} "
        f"{_interval_text(sh, 'icc')}"),
        (
            "  - Computed within each split over its conditions; the interval is "
            f"the {sh.get('agreement_interval', 'split-to-split percentile')} "
            "range. No pooled p-value is given: pooling splits would count the "
            "same conditions as independent subjects."
        ),
        *_negative_agreement_note(sh),
        f"- **Std:** {sh.get('std_correlation', 0):.3f}",
        f"- **Range:** [{sh.get('min_correlation', 0):.3f}, {sh.get('max_correlation', 0):.3f}]",
        f"- **Interpretation:** {sh.get('interpretation', 'N/A')}",
        f"- **Splits contributing:** {sh.get('n_splits', 0):,}",
        f"- **Conditions in profile (mean):** {sh.get('profile_categories', 0)}",
        "",
    ]


# ── Section 3: sensitivity ────────────────────────────────────────────────

def _sensitivity_row(threshold: int, info: dict) -> str:
    """One threshold row; thresholds no paper passes are marked, not 0%."""
    n_pass = info.get("n_pass", 0)
    if info.get("pct_supporting") is None:
        return f"| {threshold} | {n_pass} | no papers | — | — | — |"
    es = info.get("effect_size", {})
    d_str = f"{es.get('d', 0):.2f}" if es else "—"
    mag_str = es.get("magnitude", "—") if es else "—"
    return (
        f"| {threshold} | {n_pass} | "
        f"{info['pct_supporting']:.1f}% | "
        f"{info.get('mean_score', 0):.3f} | "
        f"{d_str} | {mag_str} |"
    )


def _stability_text(sensitivity: dict) -> str:
    if "stable" not in sensitivity:
        return "N/A"
    stable = sensitivity["stable"]
    return "not evaluable (no threshold has papers)" if stable is None else str(stable)


def _sensitivity_lines(sa: dict) -> list[str]:
    lines = ["## 3. Sensitivity Analysis", ""]
    status = _section_status(sa)
    if status:
        return lines + status
    lines += [
        f"- **Stable across thresholds:** {_stability_text(sa)}",
        ("- The relevance-score d compares papers above and below each threshold "
        "on that same score, so it is large by construction; it describes the "
        "split, not the robustness of the finding."),
        "",
        ("| Threshold | Papers | Supporting % | Mean Score "
        "| Relevance-score d (pass vs fail) | Magnitude |"),
        ("|-----------|--------|--------------|------------"
        "|----------------------------------|-----------|"),
    ]
    thresholds = sa.get("thresholds", {})
    for t in range(1, 6):
        lines.append(_sensitivity_row(t, thresholds.get(str(t), thresholds.get(t, {}))))
    return lines + [""]


# ── Section 4: permutation ────────────────────────────────────────────────

def _underpowered_note(pt: dict) -> list[str]:
    if not pt.get("underpowered"):
        return []
    return [(
        f"- **Underpowered:** with {len(pt.get('condition_gradient', {}))} "
        f"conditions the smallest attainable p is {pt.get('min_attainable_p', 1):.4f}, "
        "so this test cannot reach significance; a non-significant result is "
        "uninformative."
    )]


def _permutation_lines(pt: dict, ranking: list[str]) -> list[str]:
    heading = ["## 4. Permutation Test (Condition–Support Gradient)", ""]
    if pt.get("skipped"):
        return heading + _blocked(pt.get("reason", "No result was produced."))
    if not ranking:
        return []
    lines = heading + [
        f"- **Observed Spearman r:** {pt.get('observed_correlation', 0):.4f}",
        f"- **p-value:** {pt.get('p_value', 1):.4f}",
        f"- **Significant (p < 0.05):** {pt.get('significant', False)}",
        f"- **Permutations:** {pt.get('n_permutations', 0):,}",
        *_underpowered_note(pt),
        "",
        "### Condition Gradient",
        "",
        "| Condition | Papers | Support Rate |",
        "|-----------|--------|--------------|",
    ]
    gradient = pt.get("condition_gradient", {})
    for cond in ranking:
        if cond in gradient:
            info = gradient[cond]
            lines.append(f"| {cond} | {info['n_papers']} | {info['support_rate']:.2%} |")
    return lines + [""]


# ── Section 5: network ────────────────────────────────────────────────────

def _communities_text(comm: dict) -> str:
    if comm.get("unavailable"):
        return f"unavailable ({comm['unavailable']})"
    modularity = comm.get("modularity")
    modularity_text = f"{modularity:.3f}" if isinstance(modularity, (int, float)) else "—"
    return f"{comm.get('n_communities', 0)} (modularity = {modularity_text})"


def _network_lines(net: dict) -> list[str]:
    if not net:
        return []
    lines = ["## 5. Condition–Entity Co-occurrence Networks", ""]
    status = _section_status(net)
    if status:
        return lines + status
    return lines + [
        (f"- **Edges:** {net.get('n_edges', 0)} "
        f"({net.get('n_significant_edges', '?')} significant)"),
        (f"- **Edge threshold:** {net.get('edge_threshold', '—')} "
        f"(permutation null, α=0.05; significant when weight > threshold)"),
        f"- **Conditions found:** {net.get('n_conditions', 0)}",
        f"- **Target entities found:** {net.get('n_regions', 0)}",
        f"- **Communities:** {_communities_text(net.get('communities', {}))}",
        "",
    ]


# ── Section 6: co-occurring condition sensitivity ─────────────────────────

def _condition_row(name: str, res: dict) -> str:
    if res.get("skipped"):
        return (f"| {name} | {res.get('n_with', 0)} | {res.get('n_without', 0)} "
                f"| — | — | — | — | {res.get('reason', '')} |")
    return (
        f"| {name} | {res['n_with']} | {res['n_without']} | "
        f"{res['support_rate_with']:.1%} | {res['support_rate_without']:.1%} | "
        f"{res['rate_difference']:+.1%} | {res['hedges_g']:.2f} | |"
    )


def _condition_sensitivity_report_lines(sens: dict) -> list[str]:
    """Markdown for analysis 6 (a Skipped block when it did not run)."""
    if not sens:
        return []
    lines = ["## 6. Co-occurring Condition Sensitivity", ""]
    if sens.get("skipped"):
        return lines + _blocked(sens.get("reason", "No result was produced."))
    lines += [
        "Each configured condition's papers are compared with the rest of the corpus.",
        "Hedges' g compares the two groups' relevance scores, not their support.",
        "",
        ("| Condition | With | Without | Support (with) | Support (without) "
        "| Difference | Relevance-score g | Note |"),
        ("|-----------|------|---------|----------------|-------------------"
        "|------------|-------------------|------|"),
    ]
    lines += [_condition_row(name, res) for name, res in sens.get("conditions", {}).items()]
    largest = sens.get("largest_shift")
    if largest:
        lines += ["", (f"- **Largest shift:** {largest} — "
                      f"{sens['conditions'][largest]['interpretation']}")]
    return lines + [""]


# ── Report writers ────────────────────────────────────────────────────────

def _write_report(results: dict, path: Path, n_papers: int,
                  ranking: list[str] | None = None) -> None:
    """Write a markdown robustness report."""
    lines = [
        "# Robustness Analysis Report",
        "",
        f"**Papers analysed:** {n_papers}",
        (f"**Outcome measured:** `{results.get('outcome_field', PRIMARY_OUTCOME_FIELD)}` "
        "(share of papers whose extracted claims support it)"),
        *_outcome_lines(results, n_papers),
        "",
        *_bootstrap_lines(results.get("bootstrap", {}), n_papers),
        *_split_half_lines(results.get("split_half", {})),
        *_sensitivity_lines(results.get("sensitivity", {})),
        *_permutation_lines(results.get("permutation", {}), ranking or []),
        *_network_lines(results.get("network_analysis", {})),
        *_condition_sensitivity_report_lines(results.get("condition_sensitivity", {})),
        "---",
        "*Generated by SystS Stage 8: Robustness Analyses*",
    ]
    atomic_write_text(path, "\n".join(lines))
    logger.info("Robustness report written: %s", path)


def _significance_text(edge: dict) -> str:
    if "significant" not in edge:
        return "—"
    return "yes" if edge["significant"] else "no"


def _write_network_report(edges: list[dict], path: Path,
                          n_papers: int, note: str = "") -> None:
    """Write a markdown report for the condition–entity networks.

    A skip note goes before the table (inside it, it broke the Markdown), and
    each row says whether the edge passed the permutation threshold.
    """
    lines = [
        "# Condition–Entity Co-occurrence Networks",
        "",
        f"**Papers analysed:** {n_papers}",
        f"**Edges detected:** {len(edges)}",
        "",
    ]
    if note:
        lines += [f"Network analysis skipped: {note}", ""]
    lines += [
        "## Top Co-occurrences",
        "",
        "| Condition | Entity | Weight | Entity Type | Significant |",
        "|-----------|--------|--------|-------------|-------------|",
    ]
    lines += [
        f"| {edge['source']} | {edge['target']} | {edge['weight']} | "
        f"{edge.get('target_type', 'entity')} | {_significance_text(edge)} |"
        for edge in edges[:NETWORK_REPORT_TOP_EDGES]
    ]
    lines += [
        "",
        "---",
        "*Generated by SystS Stage 8: Network Analysis*",
    ]

    atomic_write_text(path, "\n".join(lines))
    logger.info("Network report written: %s", path)
