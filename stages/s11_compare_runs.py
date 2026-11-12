"""Stage 10: Compare pipeline runs.

Compares 2+ completed pipeline runs to measure pipeline-level stability.
Uses shared agreement metrics from core.agreement — kappa, Jaccard, etc.
Pairs with a missing relevance score, and fields empty in both runs, are left
out of the agreement metrics (the number of pairs used is reported) so that
missing data never reads as perfect agreement.

No LLM calls — pure comparison statistics.
"""

import json
import logging
import math
import re
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from core.claims import claim_list
from core.doi import normalize_doi
from core.io import atomic_write_json, atomic_write_text
from stages._common import is_cancelled, make_progress
from stages.s9_robustness_config import DEFAULT_OUTCOME, RobustnessOutcome

logger = logging.getLogger(__name__)

DEFAULT_COMPARISON_NAME = "run_compare"
# Comparison names whose "<slug>_results.json" would overwrite another
# stage's validation output; they get a "compare_" prefix instead.
RESERVED_COMPARISON_SLUGS = frozenset({"robustness", "verification", "interrater",
                                       "meta_analysis"})
# Composite stability bands (lower bound, label), highest first. The score is
# the mean of paper Jaccard, max(0, score κw), max(0, support κ) and content
# Jaccard — chance-level κ contributes 0, not 0.5.
STABILITY_BANDS = ((0.9, "excellent stability"), (0.75, "good stability"),
                   (0.5, "moderate stability"), (0.0, "poor stability"))
CONTENT_FIELDS = (("conditions", "conditions_jaccard"), ("mechanisms", "mechanisms_jaccard"),
                  ("findings", "findings_jaccard"), ("key_entities", "entities_jaccard"))
_NULL_IDS = frozenset({"", "none", "null", "nan"})
MAX_SCORE = 5


# ── Record helpers ──────────────────────────────────────────────────────────

def _clean_id(value: Any) -> str:
    """An identifier as text; None, "None", "null" and blanks become ""."""
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in _NULL_IDS else text


def _record_key(record: dict) -> str:
    """Generate unique key for a paper. Priority: doi > pmid > s2_id > title|year."""
    doi = normalize_doi(record.get("doi"))
    if doi:
        return f"doi:{doi}"
    pmid = _clean_id(record.get("pmid"))
    if pmid:
        return f"pmid:{pmid}"
    s2 = _clean_id(record.get("s2_id"))
    if s2:
        return f"s2:{s2}"
    title = str(record.get("title") or "").strip().lower()[:80]
    year = _clean_id(record.get("year"))
    return f"title:{title}|{year}"


def _list_values(record: dict, field: str) -> set:
    """Extract a set of normalized string values from a (possibly nested) list field."""
    return {v.lower() for v in claim_list(record, field)}


def _as_outcome(outcome: RobustnessOutcome | str | None) -> RobustnessOutcome:
    if isinstance(outcome, RobustnessOutcome):
        return outcome
    return RobustnessOutcome(field=outcome) if outcome else DEFAULT_OUTCOME


def _support(record: dict, outcome: RobustnessOutcome | str | None = None) -> bool | None:
    """Stage 8's support decision (flag, else fallback terms); None when undetermined."""
    outcome = _as_outcome(outcome)
    if not outcome.has_value(record):
        return None
    return outcome.supports(record)


def _relevance_score(record: dict) -> float | None:
    """Finite relevance score, or None when missing/unparseable (never 0, never truncated)."""
    score = record.get("relevance_score")
    if score is None or isinstance(score, bool):
        return None
    try:
        value = float(score)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def _mean(values: list) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _weighted_mean(pairs: list[tuple[float, float]]) -> float | None:
    """Mean of (value, weight) pairs; None when there is no positive weight."""
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    return sum(v * w for v, w in pairs) / total


def _slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def _comparison_slug(name: str | None) -> str:
    """A file-name slug that cannot be empty or collide with another stage's output."""
    slug = _slugify(name or "")
    if not slug:
        return DEFAULT_COMPARISON_NAME
    return f"compare_{slug}" if slug in RESERVED_COMPARISON_SLUGS else slug


# ── Load & summarize ────────────────────────────────────────────────────────

def _load_claims(path: Path) -> list:
    """Load claims JSON from path."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    return data


def _keyed_records(records: list, path: Path) -> tuple[dict, int]:
    """Records by paper key (last wins) and the number of duplicates dropped."""
    keyed: dict[str, dict] = {}
    duplicates = 0
    for index, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"{path}: record {index} is not a JSON object "
                             f"(got {type(rec).__name__})")
        key = _record_key(rec)
        duplicates += key in keyed
        keyed[key] = rec  # last wins on duplicates within a run
    return keyed, duplicates


def _score_label(score: float) -> int | float:
    return int(score) if score.is_integer() else score


def _run_summary(path: Path, records: list,
                 outcome: RobustnessOutcome | str | None = None) -> dict:
    """Build summary for one run (pass the deduplicated records)."""
    outcome = _as_outcome(outcome)
    scores = [s for s in (_relevance_score(r) for r in records) if s is not None]
    score_dist: dict = {}
    for s in scores:
        score_dist[_score_label(s)] = score_dist.get(_score_label(s), 0) + 1

    conditions = set()
    mechanisms = set()
    entities = set()
    support_count = 0
    support_total = 0
    for r in records:
        conditions |= _list_values(r, "conditions")
        mechanisms |= _list_values(r, "mechanisms")
        entities |= _list_values(r, "key_entities")
        supported = _support(r, outcome)
        if supported is not None:
            support_total += 1
            if supported:
                support_count += 1

    return {
        "path": str(path),
        "paper_count": len(records),
        "scored_count": len(scores),
        "avg_relevance": _mean(scores),
        "score_distribution": dict(sorted(score_dist.items())),
        "conditions_found": sorted(conditions),
        "mechanisms_found": sorted(mechanisms),
        "entities_found": sorted(entities),
        "outcome_field": outcome.field,
        "support_rate": support_count / support_total if support_total > 0 else None,
        "support_count": support_count,
        "support_total": support_total,
    }


# ── Pairwise comparison ────────────────────────────────────────────────────

def _score_agreement(pairs: list[tuple[dict, dict]]) -> dict:
    """Score metrics over pairs where both runs have a score."""
    from core.agreement import weighted_kappa
    scored = [(a, b) for a, b in ((_relevance_score(x), _relevance_score(y)) for x, y in pairs)
              if a is not None and b is not None]
    if not scored:
        return {"scored_pairs": 0, "score_weighted_kappa": None,
                "exact_agreement": None, "within_1_agreement": None}
    integral = all(a.is_integer() and b.is_integer() for a, b in scored)
    wk = (weighted_kappa([int(a) for a, _ in scored], [int(b) for _, b in scored],
                         max_score=MAX_SCORE) if integral else None)
    return {
        "scored_pairs": len(scored),
        "score_weighted_kappa": wk,
        "exact_agreement": sum(a == b for a, b in scored) / len(scored),
        "within_1_agreement": sum(abs(a - b) <= 1 for a, b in scored) / len(scored),
    }


def _support_agreement(pairs: list[tuple[dict, dict]], outcome: RobustnessOutcome) -> dict:
    from core.agreement import cohens_kappa
    labels = [(_support(a, outcome), _support(b, outcome)) for a, b in pairs]
    labels = [(str(a), str(b)) for a, b in labels if a is not None and b is not None]
    kappa = cohens_kappa([a for a, _ in labels], [b for _, b in labels]) if labels else None
    return {"support_pairs": len(labels), "support_kappa": kappa}


def _content_agreement(pairs: list[tuple[dict, dict]]) -> dict:
    """Per-field mean Jaccard, skipping papers where the field is empty in both runs."""
    from core.agreement import jaccard_similarity
    result: dict[str, Any] = {}
    papers_with_content = 0
    for rec_a, rec_b in pairs:
        papers_with_content += any(_list_values(rec_a, f) or _list_values(rec_b, f)
                                   for f, _ in CONTENT_FIELDS)
    for field, key in CONTENT_FIELDS:
        values = [jaccard_similarity(sa, sb) for sa, sb in
                  ((_list_values(a, field), _list_values(b, field)) for a, b in pairs)
                  if sa or sb]
        result[key] = _mean(values)
    defined = [result[key] for _, key in CONTENT_FIELDS if result[key] is not None]
    result["content_jaccard_avg"] = _mean(defined)
    result["content_pairs"] = papers_with_content
    return result


def _pairwise_compare(run_a: dict, run_b: dict,
                      outcome: RobustnessOutcome | str | None = None) -> dict:
    """Compare two runs.

    run_a / run_b each have: {"summary": ..., "records": {...key: record...}}
    """
    from core.agreement import jaccard_similarity
    outcome = _as_outcome(outcome)
    keys_a = set(run_a["records"].keys())
    keys_b = set(run_b["records"].keys())
    shared_keys = sorted(keys_a & keys_b)
    pairs = [(run_a["records"][k], run_b["records"][k]) for k in shared_keys]

    return {
        "run_a": run_a["summary"]["path"],
        "run_b": run_b["summary"]["path"],
        "papers_a": len(keys_a),
        "papers_b": len(keys_b),
        "shared_papers": len(shared_keys),
        "only_in_a": len(keys_a - keys_b),
        "only_in_b": len(keys_b - keys_a),
        "paper_jaccard": jaccard_similarity(keys_a, keys_b),
        **_score_agreement(pairs),
        **_support_agreement(pairs, outcome),
        **_content_agreement(pairs),
    }


# ── Overall summary ─────────────────────────────────────────────────────────

def _avg_weighted(pairwise: list, field: str, weight_field: str) -> float | None:
    """Pair average of ``field`` weighted by the pairs' ``weight_field`` counts."""
    return _weighted_mean([(p[field], p.get(weight_field, p.get("shared_papers", 1)) or 0)
                           for p in pairwise if p.get(field) is not None])


def _interpret_stability(stability: float | None) -> str:
    if stability is None:
        return "insufficient data"
    return next(label for bound, label in STABILITY_BANDS if stability >= bound)


def _overall_summary(pairwise: list, run_summaries: list) -> dict:
    """Pair averages (weighted by the pairs each metric used) and composite stability.

    The composite needs at least one agreement metric besides paper overlap;
    exact/within-1 agreement are reported but not chance-corrected, so they
    are not part of it.
    """
    if not pairwise:
        return {"n_pairs": 0, "interpretation": "insufficient data"}

    avg_paper_j = _mean([p["paper_jaccard"] for p in pairwise
                         if p.get("paper_jaccard") is not None])
    avg_wk = _avg_weighted(pairwise, "score_weighted_kappa", "scored_pairs")
    avg_ck = _avg_weighted(pairwise, "support_kappa", "support_pairs")
    avg_content = _avg_weighted(pairwise, "content_jaccard_avg", "content_pairs")

    agreement = [v for v in (None if avg_wk is None else max(0.0, avg_wk),
                             None if avg_ck is None else max(0.0, avg_ck),
                             avg_content) if v is not None]
    components = ([avg_paper_j] if avg_paper_j is not None else []) + agreement
    stability = _mean(components) if agreement else None

    return {
        "n_runs": len(run_summaries),
        "n_pairs": len(pairwise),
        "avg_paper_jaccard": avg_paper_j,
        "avg_score_weighted_kappa": avg_wk,
        "avg_exact_agreement": _avg_weighted(pairwise, "exact_agreement", "scored_pairs"),
        "avg_within_1_agreement": _avg_weighted(pairwise, "within_1_agreement", "scored_pairs"),
        "avg_support_kappa": avg_ck,
        "avg_content_jaccard": avg_content,
        "stability_score": stability,
        "interpretation": _interpret_stability(stability),
    }


# ── Report generation ───────────────────────────────────────────────────────

def _fmt(val: Any, decimals: int = 3) -> str:
    """Format a value for the report."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _report_text(results: dict) -> str:
    """Generate markdown report with tables."""
    lines = [
        "# Stage 10: Compare Runs Report",
        "",
        f"Generated: {results.get('timestamp', 'unknown')}",
        f"Number of runs compared: {results.get('n_runs', 0)}",
        "",
    ]

    # Per-run summaries
    lines.append("## Per-Run Summaries")
    lines.append("")
    lines.append("| Run | Papers | Avg Relevance | Support Rate |")
    lines.append("|-----|--------|---------------|------------------|")
    for i, s in enumerate(results.get("run_summaries", []), 1):
        lines.append(
            f"| Run {i} | {s['paper_count']} "
            f"| {_fmt(s['avg_relevance'])} "
            f"| {_fmt(s['support_rate'])} |"
        )
    lines.append("")

    # Pairwise comparisons
    pairwise = results.get("pairwise_comparisons", [])
    if pairwise:
        lines.append("## Pairwise Comparisons")
        lines.append("")
        lines.append(
            "| Pair | Shared | Scored | Jaccard | wKappa | Exact% | Within-1% "
            "| Support-κ | Content-J |"
        )
        lines.append(
            "|------|--------|--------|---------|--------|--------|---------- "
            "|--------|-----------|"
        )
        for i, p in enumerate(pairwise, 1):
            lines.append(
                f"| {i} "
                f"| {p['shared_papers']}/{p['papers_a']}+{p['papers_b']} "
                f"| {p.get('scored_pairs', 'N/A')} "
                f"| {_fmt(p['paper_jaccard'])} "
                f"| {_fmt(p['score_weighted_kappa'])} "
                f"| {_fmt(p['exact_agreement'])} "
                f"| {_fmt(p['within_1_agreement'])} "
                f"| {_fmt(p['support_kappa'])} "
                f"| {_fmt(p['content_jaccard_avg'])} |"
            )
        lines.append("")

    # Overall stability
    overall = results.get("overall_summary", {})
    lines.append("## Overall Stability")
    lines.append("")
    lines.append(f"- **Stability score**: {_fmt(overall.get('stability_score'))}")
    lines.append(f"- **Interpretation**: {overall.get('interpretation', 'N/A')}")
    lines.append(f"- **Avg paper Jaccard**: {_fmt(overall.get('avg_paper_jaccard'))}")
    lines.append(
        f"- **Avg score weighted kappa**: "
        f"{_fmt(overall.get('avg_score_weighted_kappa'))}"
    )
    lines.append(
        f"- **Avg exact agreement**: {_fmt(overall.get('avg_exact_agreement'))}"
    )
    lines.append(
        f"- **Avg within-1 agreement**: "
        f"{_fmt(overall.get('avg_within_1_agreement'))}"
    )
    lines.append(
        f"- **Avg support kappa**: "
        f"{_fmt(overall.get('avg_support_kappa'))}"
    )
    lines.append(
        f"- **Avg content Jaccard**: {_fmt(overall.get('avg_content_jaccard'))}"
    )
    lines.append("")
    lines.append(
        "Stability score = mean of paper Jaccard, max(0, score κw), max(0, support κ) "
        "and content Jaccard; it needs at least one agreement metric besides paper "
        "overlap. Pair averages are weighted by the pairs each metric used; pairs "
        "without scores and fields empty in both runs are excluded. Bands: "
        + ", ".join(f"≥{bound:g} {label}" for bound, label in STABILITY_BANDS) + "."
    )
    lines.append("")

    return "\n".join(lines)


# ── Output paths ────────────────────────────────────────────────────────────

def _output_paths(config, comparison_name: str | None,
                  report_path: str | Path | None) -> tuple[Path, Path]:
    """Results JSON in the run's validation dir; report in the run's output dir.

    Uses ``config.output_dir`` (run-scoped for named runs), not the project
    root, and honours an explicit ``report_path`` (audit 13-M9).
    """
    data_dir = Path(config.data_dir) if config and hasattr(config, "data_dir") \
        else Path.cwd() / "data"
    output_dir = Path(config.output_dir) if config and hasattr(config, "output_dir") \
        else Path.cwd() / "output"
    slug = _comparison_slug(comparison_name)
    results_path = data_dir / "validation" / f"{slug}_results.json"
    report = Path(report_path) if report_path else output_dir / f"{slug}_report.md"
    return results_path, report


# ── Main entry point ────────────────────────────────────────────────────────

def run(
    config=None,
    claims_paths=None,
    comparison_name: str = DEFAULT_COMPARISON_NAME,
    cancel_event=None,
    progress_callback=None,
    report_path: str | Path | None = None,
    **kw,
) -> dict:
    """Compare 2+ pipeline runs.

    Parameters
    ----------
    config : Config, optional
        Pipeline configuration object.
    claims_paths : list[str|Path]
        Paths to 2+ claims JSON files to compare.
    comparison_name : str
        Name for this comparison (used in output filenames). Empty names and
        names that would overwrite another stage's results get a safe slug.
    report_path : str or Path, optional
        Exact markdown report path chosen by the user; default
        ``config.output_dir / "<slug>_report.md"``.
    cancel_event : threading.Event or similar, optional
        Set to cancel the run.
    progress_callback : callable, optional
        Called with progress message strings.
    """

    _cancelled = partial(is_cancelled, cancel_event)

    _progress = make_progress(logger, progress_callback)

    _progress("=" * 60)
    _progress("STAGE 10: COMPARE RUNS")
    _progress("=" * 60)

    if not claims_paths or len(claims_paths) < 2:
        raise ValueError("Need at least 2 claims files to compare")

    claims_paths = [Path(p) for p in claims_paths]
    for p in claims_paths:
        if not p.exists():
            raise FileNotFoundError(f"Claims file not found: {p}")

    _progress(f"Comparing {len(claims_paths)} runs:")
    for i, p in enumerate(claims_paths):
        _progress(f"  Run {i + 1}: {p}")

    results_path, report_path = _output_paths(config, comparison_name, report_path)

    if _cancelled():
        return {"cancelled": True}

    # ── Step 1: Load each claims file ────────────────────────────────
    from stages.s9_robustness import _load_outcome
    outcome = _load_outcome(config)
    _progress(f"Outcome compared: {outcome.field}")
    _progress("Loading claims files...")
    run_data = []
    for i, p in enumerate(claims_paths):
        records = _load_claims(p)
        keyed, duplicates = _keyed_records(records, p)
        summary = {**_run_summary(p, list(keyed.values()), outcome),
                   "raw_record_count": len(records), "duplicates": duplicates}
        run_data.append({
            "summary": summary,
            "records": keyed,
        })
        _progress(
            f"  Run {i + 1}: {len(records)} papers, "
            f"{len(keyed)} unique keys"
            + (f" ({duplicates} duplicates)" if duplicates else "")
        )

    if _cancelled():
        return {"cancelled": True}

    # ── Step 2: Log per-run summaries ────────────────────────────────
    _progress("-" * 40)
    _progress("Per-run summaries:")
    for i, rd in enumerate(run_data):
        s = rd["summary"]
        _progress(
            f"  Run {i + 1}: {s['paper_count']} papers, "
            f"avg relevance={_fmt(s['avg_relevance'])}, "
            f"support rate={_fmt(s['support_rate'])}, "
            f"conditions={len(s['conditions_found'])}, "
            f"mechanisms={len(s['mechanisms_found'])}, "
            f"entities={len(s['entities_found'])}"
        )

    if _cancelled():
        return {"cancelled": True}

    # ── Step 3: Pairwise comparisons ─────────────────────────────────
    _progress("-" * 40)
    _progress("Computing pairwise comparisons...")
    pairwise = []
    n_runs = len(run_data)
    pair_idx = 0
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            if _cancelled():
                return {"cancelled": True}
            pair_idx += 1
            _progress(f"  Pair {pair_idx}: Run {i + 1} vs Run {j + 1}")
            comparison = _pairwise_compare(run_data[i], run_data[j], outcome)
            pairwise.append(comparison)

            _progress(
                f"    Overlap: {comparison['shared_papers']} shared, "
                f"Jaccard={_fmt(comparison['paper_jaccard'])}"
            )
            if comparison.get("exact_agreement") is not None:
                _progress(
                    f"    Scores ({comparison['scored_pairs']} scored pairs): "
                    f"wκ={_fmt(comparison['score_weighted_kappa'])}, "
                    f"exact={_fmt(comparison['exact_agreement'])}, "
                    f"within-1={_fmt(comparison['within_1_agreement'])}"
                )
            if comparison.get("support_kappa") is not None:
                _progress(
                    f"    Support κ={_fmt(comparison['support_kappa'])}"
                )
            if comparison.get("content_jaccard_avg") is not None:
                _progress(
                    f"    Content Jaccard avg={_fmt(comparison['content_jaccard_avg'])}"
                )

    if _cancelled():
        return {"cancelled": True}

    # ── Step 4: Overall stability ────────────────────────────────────
    _progress("-" * 40)
    _progress("Computing overall stability...")
    summaries = [rd["summary"] for rd in run_data]
    overall = _overall_summary(pairwise, summaries)

    _progress(
        f"  Stability score: {_fmt(overall.get('stability_score'))} "
        f"({overall.get('interpretation', 'N/A')})"
    )

    # ── Step 5: Build results ────────────────────────────────────────
    results = {
        "stage": "s11_compare_runs",
        "timestamp": datetime.now(UTC).isoformat(),
        "n_runs": len(run_data),
        "claims_paths": [str(p) for p in claims_paths],
        "comparison_name": comparison_name,
        "run_summaries": summaries,
        "pairwise_comparisons": pairwise,
        "overall_summary": overall,
    }

    # ── Step 6: Save outputs ─────────────────────────────────────────
    _progress("-" * 40)
    _progress("Saving results...")
    atomic_write_json(results_path, results)
    _progress(f"  Results JSON: {results_path}")

    report = _report_text(results)
    atomic_write_text(report_path, report)
    _progress(f"  Report: {report_path}")

    results["results_path"] = str(results_path)
    results["report_path"] = str(report_path)

    _progress("=" * 60)
    _progress(
        f"Stage 10 complete: {overall.get('interpretation', 'N/A')} "
        f"(score={_fmt(overall.get('stability_score'))})"
    )
    _progress("=" * 60)

    return results


# ── Check output ────────────────────────────────────────────────────────────
