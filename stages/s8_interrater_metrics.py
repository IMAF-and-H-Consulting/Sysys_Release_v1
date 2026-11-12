"""Stage 8 agreement metrics and markdown report.

Split from ``stages.s8_interrater``, which re-exports these names. Weighted κ
is the chance-corrected score agreement and the only value banded as κ. Claim
overlap is a descriptive exact-string Jaccard reported beside it, never
blended into it. Undefined statistics are None and render as N/A.
"""
import math
import re

from core.agreement import (
    adjacent_agreement,
    cohens_kappa,
    format_adjacent_agreement,
    interpret_kappa,
    jaccard_similarity,
    weighted_kappa,
)
from core.claims import LEGACY_FIELD_ALIASES

# Summary key -> claim field compared between the two raters.
CLAIM_JACCARD_FIELDS = {
    "claim_jaccard_findings": "findings",
    "claim_jaccard_mechanisms": "mechanisms",
    "claim_jaccard_conditions": "conditions",
    "claim_jaccard_entities": "key_entities",
}
_SCORE_METRICS = ("score_kappa", "score_weighted_kappa", "score_mean_diff",
                  "score_correlation")
_REPORT_TITLE_CHARS = 50

# Kept under its historical name; banding lives in core.agreement.
_interpret_kappa = interpret_kappa


def normalize_label(text: str) -> str:
    """Normalize text for comparison (lowercase, strip, collapse whitespace)."""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def _fleiss_insufficient(n_subjects: int, n_raters: int) -> dict:
    return {"kappa": None, "p_observed": None, "p_expected": None,
            "interpretation": "insufficient data",
            "n_subjects": n_subjects, "n_raters": n_raters}


def fleiss_kappa(ratings_matrix: list[list[int]]) -> dict:
    """Compute Fleiss' kappa for multi-rater agreement.

    Parameters
    ----------
    ratings_matrix : list of lists, shape (n_subjects, n_categories)
        Each row is one subject; each column is a category.
        Entry [i][j] = number of raters who assigned subject i to category j.

    Returns
    -------
    dict with kappa, p_observed, p_expected, interpretation, n_subjects, n_raters.
    With fewer than two subjects or raters the statistics are None
    ("insufficient data"), and when every rating falls in one category
    (expected agreement 1) kappa is None ("undefined") -- never a fake 0
    or 1 (audit 22-L5). Rows must all sum to the same number of raters.
    """
    if not ratings_matrix or not ratings_matrix[0]:
        return _fleiss_insufficient(0, 0)

    n = len(ratings_matrix)          # number of subjects
    k = len(ratings_matrix[0])       # number of categories
    n_raters = sum(ratings_matrix[0])  # total raters per subject
    if any(len(row) != k or sum(row) != n_raters for row in ratings_matrix):
        raise ValueError(
            "Fleiss' kappa needs the same categories and number of raters "
            "in every row")

    if n < 2 or n_raters < 2:
        return _fleiss_insufficient(n, n_raters)

    # P_i for each subject: proportion of agreeing rater pairs
    p_i_list = []
    for row in ratings_matrix:
        sum_sq = sum(r * r for r in row)
        p_i = (sum_sq - n_raters) / (n_raters * (n_raters - 1)) if n_raters > 1 else 0
        p_i_list.append(p_i)

    p_bar = sum(p_i_list) / n  # mean observed agreement

    # p_j for each category: proportion of all assignments in that category
    p_e = 0.0
    total_assignments = n * n_raters
    for j in range(k):
        col_sum = sum(ratings_matrix[i][j] for i in range(n))
        p_j = col_sum / total_assignments
        p_e += p_j * p_j

    kappa = None if abs(1.0 - p_e) < 1e-10 else (p_bar - p_e) / (1.0 - p_e)

    return {
        "kappa": None if kappa is None else round(kappa, 6),
        "p_observed": round(p_bar, 6),
        "p_expected": round(p_e, 6),
        "interpretation": interpret_kappa(kappa),
        "n_subjects": n,
        "n_raters": n_raters,
    }


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r; None when undefined (fewer than 2 pairs or zero variance)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-10:
        return None
    return cov / denom


def _record_values(record: dict, side: str, field_name: str) -> list:
    """``original_*``/``second_*`` list from a comparison record.

    Checkpoints written before the key_entities rename store
    ``*_brain_regions`` instead.
    """
    for name in (field_name, *LEGACY_FIELD_ALIASES.get(field_name, ())):
        values = record.get(f"{side}_{name}")
        if values:
            return list(values)
    return []


def _is_rated(record: dict) -> bool:
    """True when both raters produced a score (failed rows carry ``error``)."""
    return (
        not record.get("error")
        and record.get("original_score") is not None
        and record.get("second_score") is not None
    )


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _score_bin(score: int) -> str:
    """Nominal bins for Cohen's κ: low (0-2), mid (3-4), high (5)."""
    if score <= 2:
        return "low"
    if score <= 4:
        return "mid"
    return "high"


def _score_metrics(results: list[dict]) -> dict:
    """Adjacent agreement, κ, weighted κ, mean |Δ| and r over rated rows."""
    orig = [r["original_score"] for r in results]
    second = [r["second_score"] for r in results]
    return {
        **adjacent_agreement(orig, second),
        "score_kappa": _round(cohens_kappa([_score_bin(s) for s in orig],
                                           [_score_bin(s) for s in second])),
        "score_weighted_kappa": _round(weighted_kappa(orig, second)),
        "score_mean_diff": _round(_mean([abs(a - b) for a, b in zip(orig, second)])),
        "score_correlation": _round(pearson_correlation(
            [float(s) for s in orig], [float(s) for s in second])),
    }


def _field_jaccard(results: list[dict], field_name: str) -> float | None:
    """Mean Jaccard for one claim field.

    Rows whose second extraction failed, and pairs where neither rater
    listed anything (overlap undefined), are left out.
    """
    sims = []
    for r in results:
        if r.get("extract_error"):
            continue
        orig = {normalize_label(x) for x in _record_values(r, "original", field_name)}
        second = {normalize_label(x) for x in _record_values(r, "second", field_name)}
        if orig or second:
            sims.append(jaccard_similarity(orig, second))
    return _mean(sims)


def _claim_metrics(results: list[dict], score_only: bool) -> dict:
    if score_only:
        jaccards = dict.fromkeys(CLAIM_JACCARD_FIELDS)
    else:
        jaccards = {key: _field_jaccard(results, name)
                    for key, name in CLAIM_JACCARD_FIELDS.items()}
    defined = [v for v in jaccards.values() if v is not None]
    return {**{k: _round(v) for k, v in jaccards.items()},
            "claim_jaccard_mean": _round(_mean(defined))}


def compute_agreement(results: list[dict], score_only: bool = False) -> dict:
    """Compute agreement metrics from per-paper comparison results.

    Rows without both ratings (a failed, timed-out or unparseable second
    rating) are excluded and counted in ``n_failed``; ``n_papers`` is the
    effective n. ``interpretation`` bands the weighted κ alone. The claim
    Jaccards (and their mean, ``claim_jaccard_mean``) are descriptive and
    reported separately; they are None when ``score_only`` (a cross-encoder
    rater extracts no claims) or when no pair is comparable.
    """
    rated = [r for r in results if _is_rated(r)]
    base = {"n_papers": len(rated), "n_failed": len(results) - len(rated),
            "score_only": score_only}
    if not rated:
        return {**base, **adjacent_agreement([], []),
                **dict.fromkeys(_SCORE_METRICS),
                **_claim_metrics([], score_only=True),
                "interpretation": "insufficient data"}
    scores = _score_metrics(rated)
    return {**base, **scores, **_claim_metrics(rated, score_only),
            "interpretation": interpret_kappa(scores["score_weighted_kappa"])}


# ── Markdown report ──────────────────────────────────────────────────────

def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _escape_cell(text: object) -> str:
    """A markdown table cell: pipes escaped, newlines flattened."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _summary_section(summary: dict) -> list[str]:
    claim_overlap = ("N/A (score-only rater)" if summary.get("score_only")
                     else _fmt(summary.get("claim_jaccard_mean")))
    return [
        "## Summary",
        "",
        f"- **Papers evaluated**: {summary.get('n_papers', 0)}",
        f"- **Failed second ratings (excluded)**: {summary.get('n_failed', 0)}",
        f"- **Duplicate papers skipped**: {summary.get('n_duplicates_skipped', 0)}",
        (f"- **Score agreement (weighted κ)**: {_fmt(summary.get('score_weighted_kappa'))}"
         f" ({summary.get('interpretation', 'N/A')})"),
        f"- **Mean claim overlap (Jaccard, not chance-corrected)**: {claim_overlap}",
        "",
    ]


def _score_section(summary: dict) -> list[str]:
    return [
        "## Score Agreement",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Percent Adjacent Agreement (within 1 point) | {format_adjacent_agreement(summary)} |",
        f"| Cohen's κ (binned) | {_fmt(summary.get('score_kappa'))} |",
        f"| Weighted κ (linear, 0-5) | {_fmt(summary.get('score_weighted_kappa'))} |",
        f"| Mean absolute difference | {_fmt(summary.get('score_mean_diff'))} |",
        f"| Pearson correlation | {_fmt(summary.get('score_correlation'))} |",
        "",
        ("Percent Adjacent Agreement = 100 × pairs with |original − second| ≤ 1 / "
         "rated pairs. Exact matches are included; comparison uses the original "
         "rating scale, without binning. This is descriptive agreement, not chance-corrected "
         "agreement. N/A marks a statistic that is undefined for these ratings (for example "
         "κ when both raters give every paper the same score)."),
        "",
    ]


def _claim_section(summary: dict) -> list[str]:
    if summary.get("score_only"):
        return [
            "## Claim Extraction Agreement",
            "",
            ("_Not applicable — this rater scores relevance only "
             "(cross-encoder) and does not extract claims._"),
            "",
        ]
    rows = [("Findings", "claim_jaccard_findings"), ("Mechanisms", "claim_jaccard_mechanisms"),
            ("Conditions", "claim_jaccard_conditions"), ("Key entities", "claim_jaccard_entities")]
    return [
        "## Claim Extraction Agreement (Jaccard Similarity)",
        "",
        "| Field | Jaccard |",
        "|-------|---------|",
        *(f"| {label} | {_fmt(summary.get(key))} |" for label, key in rows),
        "",
        ("Exact overlap of normalised free-text claims: descriptive and not "
         "chance-corrected, so it is not banded as κ. Papers where neither "
         "rater listed a value for a field are excluded from that field."),
        "",
    ]


def _interpretation_section(summary: dict) -> list[str]:
    return [
        "## Interpretation",
        "",
        (f"Score agreement is **{summary.get('interpretation', 'N/A')}**"
         f" (weighted κ = {_fmt(summary.get('score_weighted_kappa'))}). "
         "The bands below (Landis & Koch, 1977) apply to weighted κ only."),
        "",
        "| Range | Interpretation |",
        "|-------|---------------|",
        "| ≤ 0.00 | Poor |",
        "| 0.00 – 0.20 | Slight |",
        "| 0.20 – 0.40 | Fair |",
        "| 0.40 – 0.60 | Moderate |",
        "| 0.60 – 0.80 | Substantial |",
        "| > 0.80 | Almost perfect |",
        "",
    ]


def _per_paper_section(results: list[dict]) -> list[str]:
    if not results:
        return []
    lines = [
        "## Per-Paper Detail",
        "",
        "| # | Title | Original | Second | Δ |",
        "|---|-------|----------|--------|---|",
    ]
    for i, r in enumerate(results, 1):
        title = _escape_cell(str(r.get("title") or "?")[:_REPORT_TITLE_CHARS])
        orig = r.get("original_score")
        sec = r.get("second_score")
        diff = abs(int(orig) - int(sec)) if isinstance(orig, int) and isinstance(sec, int) else "?"
        lines.append(f"| {i} | {title} | {'?' if orig is None else orig} "
                     f"| {'?' if sec is None else sec} | {diff} |")
    lines.append("")
    return lines


def generate_report(summary: dict, results: list[dict]) -> str:
    """Generate a markdown report with tables and interpretation."""
    # Also support regenerating reports from summaries saved before this metric.
    if "adjacent_agreement_pct" not in summary:
        rated = [r for r in results if _is_rated(r)]
        summary = {**summary, **adjacent_agreement(
            [r["original_score"] for r in rated], [r["second_score"] for r in rated])}
    lines = [
        "# Inter-Rater Reliability Report",
        "",
        *_summary_section(summary),
        *_score_section(summary),
        *_claim_section(summary),
        *_interpretation_section(summary),
        *_per_paper_section(results),
    ]
    note = summary.get("independence_note")
    if note:
        lines.extend(["## Independence", "", note, ""])
    return "\n".join(lines)
