"""Stage 9: Abstract number verification.

Extract every number and statistic from the synthesised abstract and
cross-check each against the actual data (claims_filtered.json) and,
optionally, robustness_results.json. Explicit historical callers of verify()
can still supply an agreement file; pipeline runs do not discover retired outputs.

No LLM calls — pure regex extraction + numerical comparison.
Tolerances are per metric (counts exact, percentages ±0.5 pp, κ/r/CI ±0.01,
p-values by comparator); see ``stages.s10_verify_stats``. CIs, p-values and
correlations are compared only with the robustness statistic they name.
"""

import json
import logging
import re
from functools import partial
from pathlib import Path

from core.claims import claim_list
from core.io import atomic_write_json, atomic_write_text
from process.domain_patterns import contains_term
from process.paths import load_paper_list, resolve_claims_corpus, resolve_validation_dir
from stages._common import announce_no_llm, is_cancelled, make_progress
from stages.s9_robustness import (
    DEFAULT_OUTCOME,
    RobustnessOutcome,
    _extract_conditions,
    _load_condition_config,
    _load_outcome,
)
from stages.s10_verify_claims import (
    check_cluster_sizes,
    drop_overlapping,
    extract_cluster_size_lists,
    extract_generic_counts,
    extract_unclaimed,
    unclaimed_rows,
)
from stages.s10_verify_counts import load_clusters, paper_count_reference
from stages.s10_verify_stats import (
    check_ci,
    check_correlation,
    check_p_value,
    clause_around,
    extract_cis,
    extract_correlations,
    extract_kappas,
    extract_p_values,
    number_decimals,
    within_tolerance,
)

logger = logging.getLogger(__name__)

# ── Verification vocabulary ─────────────────────────────────────────────
# Domain vocabulary comes from the research config: mechanism terms from
# ``process.domain_patterns.extraction_patterns``, conditions and the support
# outcome from the same loaders Stage 8 uses. Without a config nothing
# domain-specific is assumed; only generic "support" wording is recognized.

# Words in a percentage's clause that mark it as the share of supporting papers.
SUPPORT_CONTEXT = ["support", "inform"]
AGREEMENT_CONTEXT = ["agree", "agreed", "agreement", "concordance", "concordant"]
# A percentage directly followed by these is a confidence/credible level, not a claim.
_CONFIDENCE_LEVEL_RE = re.compile(r"\s*(?:CI\b|CrI\b|confidence\b|credible\b)", re.IGNORECASE)

# Stage 8 interrater summary keys (``s8_interrater`` summary), newest first;
# ``weighted_kappa``/``cohens_kappa``/``overall_agreement_pct`` are legacy.
_WEIGHTED_KAPPA_KEYS = ("score_weighted_kappa", "weighted_kappa")
_COHENS_KAPPA_KEYS = ("score_kappa", "cohens_kappa")
_ADJACENT_AGREEMENT_KEYS = ("adjacent_agreement_pct",)
_LEGACY_AGREEMENT_KEYS = ("overall_agreement_pct",)
_WEIGHTED_RE = re.compile(r"\bweighted\b", re.IGNORECASE)
_COHEN_RE = re.compile(r"\bcohen", re.IGNORECASE)
_ADJACENT_RE = re.compile(r"\badjacent\b|\bwithin[\s-]+(?:one|1)\b|±\s*1\b", re.IGNORECASE)


def _matched_term(clause: str, terms: list[str]) -> str | None:
    """First clause word that is one of ``terms`` or a simple inflection of it.

    "support" matches supports/supported/supporting; "inform" does not match
    "informative" (audit 13-M2).
    """
    for term in terms:
        m = re.search(rf"(?<!\w){re.escape(term)}(?:s|es|ed|ing)?(?!\w)", clause, re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def _verify_terms(config) -> tuple[list[str], RobustnessOutcome, list[str],
                                   list[str], dict[str, str]]:
    """Resolve stage-10 vocabulary from the research config.

    Returns ``(mechanism_terms, outcome, mechanism_context, support_context,
    condition_patterns)``.
    """
    if config is None:
        return ([], DEFAULT_OUTCOME, [], list(SUPPORT_CONTEXT), {})
    from process.domain_patterns import extraction_patterns
    mechanism_terms = list(extraction_patterns(config).get("mechanisms", []))
    outcome = _load_outcome(config)
    mechanism_context = ["mechanism"] + [t.lower() for t in mechanism_terms]
    support_context = SUPPORT_CONTEXT + [t.lower() for t in outcome.fallback_terms]
    condition_patterns, _ = _load_condition_config(config)
    return (mechanism_terms, outcome, mechanism_context, support_context,
            condition_patterns)


# ── Paper helpers ───────────────────────────────────────────────────────

def _has_mechanism(paper: dict, keywords: list[str]) -> bool:
    text = " ".join([paper.get("abstract", "") or "", *claim_list(paper, "mechanisms")])
    return any(contains_term(text, kw) for kw in keywords)


def _corpus_stats(papers: list[dict], config) -> dict:
    """Reference numbers computed from the claims corpus."""
    (mechanism_terms, outcome, mechanism_context,
     support_context, cond_patterns) = _verify_terms(config)
    n_papers = len(papers)
    n_supporting = sum(1 for p in papers if outcome.supports(p))
    n_mech = sum(1 for p in papers if _has_mechanism(p, mechanism_terms))
    conditions = {cond for p in papers for cond in _extract_conditions(p, cond_patterns)}
    return {
        "n_papers": n_papers,
        "pct_supporting": (n_supporting / n_papers * 100) if n_papers else 0.0,
        "pct_mech": (n_mech / n_papers * 100) if n_papers else 0.0,
        "has_mechanism_terms": bool(mechanism_terms),
        "outcome": outcome,
        "mechanism_context": mechanism_context,
        "support_context": support_context,
        "has_condition_patterns": bool(cond_patterns),
        "n_conditions": len(conditions),
    }


def _load_json_dict(path: Path | None, label: str) -> dict | None:
    """A JSON object from ``path``, or None when absent, unreadable or not an object."""
    if not path or not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load %s data: %s", label, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Ignoring %s data in %s: expected a JSON object", label, path)
        return None
    logger.info("Loaded %s data for cross-check: %s", label, path)
    return data


# ── Number extraction from abstract ────────────────────────────────────

def _extract_paper_counts(text: str) -> list[dict]:
    """"245 papers", "120 studies", "80 articles"."""
    return [{"value": int(m.group(1).replace(",", "")), "raw": m.group(0), "span": m.span()}
            for m in re.finditer(r'(\d[\d,]*)\s+(?:papers?|studies|articles|publications?)',
                                 text, re.IGNORECASE)]


def _extract_percentages(text: str) -> list[dict]:
    """"78.5%" or "78.5 %", except confidence levels ("95% CI")."""
    items = []
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*%', text):
        if _CONFIDENCE_LEVEL_RE.match(text, m.end()):
            continue
        items.append({"value": float(m.group(1)), "decimals": number_decimals(m.group(1)),
                      "raw": m.group(0), "span": m.span()})
    return items


def _extract_abstract_numbers(abstract_text: str) -> dict:
    """Extract all numbers and statistics from abstract text.

    Returns dict with keys: paper_counts, percentages, kappa_values,
    ci_values, p_values, correlations, generic_counts, cluster_sizes and
    unclaimed (every other number that is not an obvious non-claim).
    """
    sizes = extract_cluster_size_lists(abstract_text)
    numbers: dict[str, list] = {
        "paper_counts": drop_overlapping(_extract_paper_counts(abstract_text),
                                         [i["span"] for i in sizes]),
        "percentages": _extract_percentages(abstract_text),
        "kappa_values": extract_kappas(abstract_text),
        "ci_values": extract_cis(abstract_text),
        "p_values": extract_p_values(abstract_text),
        "correlations": extract_correlations(abstract_text),
        "generic_counts": extract_generic_counts(abstract_text),
        "cluster_sizes": sizes,
    }
    claimed = [item["span"] for items in numbers.values() for item in items]
    numbers["unclaimed"] = extract_unclaimed(abstract_text, claimed)
    return numbers


# ── Check rows ──────────────────────────────────────────────────────────

def _compare(claim: str, claimed: float, actual: float, metric: str, label: str,
             decimals: int = 0, diff_fmt: str = "off by {diff:g}") -> dict:
    """PASS/FAIL row for a claim with a known reference value."""
    ok = within_tolerance(claimed, actual, metric, decimals)
    status = "PASS" if ok else "FAIL"
    detail = "within tolerance" if ok else diff_fmt.format(diff=actual - claimed)
    logger.info("CHECK %s: claim=%s actual=%s → %s", label, claimed, actual, status)
    return {"claim": claim, "expected": claimed, "actual": actual,
            "status": status, "note": f"{label}: {detail}"}


def _unverified(claim: str, claimed, note: str) -> dict:
    logger.info("CHECK %s → WARN (%s)", claim, note)
    return {"claim": claim, "expected": claimed, "actual": None,
            "status": "WARN", "note": note}


def _check_paper_counts(items: list[dict], text: str, n_papers: int,
                        clusters: list[dict] | None, cluster_note: str) -> list[dict]:
    rows = []
    for item in items:
        reference = paper_count_reference(text, item, clusters, cluster_note)
        actual, label = reference if reference is not None else (n_papers, "corpus total")
        if actual is None:
            rows.append(_unverified(item["raw"], item["value"], label))
        else:
            rows.append(_compare(item["raw"], item["value"], actual, "count", label))
    return rows


def _first_value(data: dict | None, keys: tuple[str, ...]) -> tuple[float | None, str | None]:
    for key in keys:
        value = (data or {}).get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), key
    return None, None


def _agreement_reference(clause: str, interrater: dict | None) -> tuple[float | None, str]:
    """Stage 8 records adjacent (within-one) agreement; other % are not checkable."""
    legacy, key = _first_value(interrater, _LEGACY_AGREEMENT_KEYS)
    if legacy is not None:
        return legacy, f"agreement ({key})"
    if not _ADJACENT_RE.search(clause):
        return None, ("agreement: wording does not say which agreement; Stage 8 "
                      "records only adjacent (within-one) agreement, not verified")
    value, key = _first_value(interrater, _ADJACENT_AGREEMENT_KEYS)
    if value is None:
        return None, "agreement: no interrater data available for cross-check"
    return value, f"agreement ({key})"


def _check_percentage(item: dict, text: str, stats: dict, interrater: dict | None) -> dict:
    # Attribute the percentage only from words in its own clause.
    clause = clause_around(text, item["span"])
    if keyword := _matched_term(clause, stats["support_context"]):
        actual, label = stats["pct_supporting"], f"support ({stats['outcome'].field})"
    elif stats["has_mechanism_terms"] and (
            keyword := _matched_term(clause, stats["mechanism_context"])):
        actual, label = stats["pct_mech"], "mechanism"
    elif keyword := _matched_term(clause, AGREEMENT_CONTEXT):
        actual, label = _agreement_reference(clause, interrater)
        if actual is None:
            return _unverified(item["raw"], item["value"], label)
    else:
        return _unverified(item["raw"], item["value"],
                           "could not match percentage to a known metric")
    row = _compare(item["raw"], item["value"], actual, "percentage", label,
                   item["decimals"], diff_fmt="diff={diff:.2f}pp")
    return {**row, "actual": round(actual, 2), "note": f"{row['note']} [matched '{keyword}']"}


def _check_kappa(item: dict, text: str, interrater: dict | None) -> dict:
    clause = clause_around(text, item["span"])
    is_cohen = _COHEN_RE.search(clause) and not _WEIGHTED_RE.search(clause)
    keys = _COHENS_KAPPA_KEYS if is_cohen else _WEIGHTED_KAPPA_KEYS
    actual, key = _first_value(interrater, keys)
    if actual is None:
        return _unverified(item["raw"], item["value"],
                           "no interrater data available for cross-check")
    row = _compare(item["raw"], item["value"], actual, "kappa",
                   f"interrater {key}", item["decimals"], diff_fmt="diff={diff:.4f}")
    return {**row, "actual": round(actual, 4)}


def _check_generic_count(item: dict, stats: dict, clusters: list[dict] | None,
                         cluster_note: str) -> list[dict]:
    noun, raw = item["noun"], item["raw"]
    rows = []
    if "subset" in item:
        rows.append(_unverified(raw, item["subset"],
                                f"subset of {noun}: no reference for the subgroup, not verified"))
    if noun.startswith("condition"):
        if not stats["has_condition_patterns"]:
            rows.append(_unverified(raw, item["value"], "condition count: no condition "
                                    "patterns configured, cannot count conditions"))
        else:
            rows.append(_compare(raw, item["value"], stats["n_conditions"], "count",
                                 "condition count"))
    elif noun.startswith("cluster"):
        if clusters is None:
            rows.append(_unverified(raw, item["value"], f"cluster count: {cluster_note}"))
        else:
            rows.append(_compare(raw, item["value"], len(clusters), "count",
                                 "cluster count (clusters.json)"))
    else:
        rows.append(_unverified(raw, item["value"], f"{noun} count: no reference data available"))
    return rows


# ── Core verification logic ─────────────────────────────────────────────

def verify(papers: list[dict], abstract_text: str,
           robustness_path: Path | None = None,
           interrater_path: Path | None = None,
           config=None,
           clusters_path: Path | None = None) -> dict:
    """Core verification: extract numbers from abstract, check each.

    Pass the pipeline ``config`` so condition patterns, mechanism terms, and
    the support outcome come from the research config instead of hardcoded
    domain vocabulary. Without a config, term lists are empty and support is
    read from the extracted ``supports_primary_question`` flag.
    ``clusters_path`` supplies Stage 6 cluster sizes and membership for subgroup
    counts; corpus totals and individual clusters are checked separately.
    ``interrater_path`` (explicit callers only) is a Stage 8 interrater summary.
    Numbers no check covers are reported as WARN rows, never dropped.

    Returns
    -------
    dict with keys: summary, checks, extracted_numbers
    """
    extracted = _extract_abstract_numbers(abstract_text)
    stats = _corpus_stats(papers, config)
    robustness_data = _load_json_dict(robustness_path, "robustness")
    interrater_data = _load_json_dict(interrater_path, "interrater")
    clusters, cluster_note = load_clusters(clusters_path, papers)

    checks = _check_paper_counts(extracted["paper_counts"], abstract_text,
                                 stats["n_papers"], clusters, cluster_note)
    checks += [_check_percentage(i, abstract_text, stats, interrater_data)
               for i in extracted["percentages"]]
    checks += [_check_kappa(i, abstract_text, interrater_data)
               for i in extracted["kappa_values"]]
    # CIs, p-values, correlations: only against their referent.
    checks += [check_ci(i, abstract_text, robustness_data) for i in extracted["ci_values"]]
    checks += [check_p_value(i, abstract_text, robustness_data) for i in extracted["p_values"]]
    checks += [check_correlation(i, abstract_text, robustness_data)
               for i in extracted["correlations"]]
    for item in extracted["generic_counts"]:
        checks += _check_generic_count(item, stats, clusters, cluster_note)
    for item in extracted["cluster_sizes"]:
        checks += check_cluster_sizes(item, clusters, cluster_note)
    checks += unclaimed_rows(extracted["unclaimed"])

    statuses = [c["status"] for c in checks]
    return {
        "summary": {
            "total_checks": len(checks),
            "passed": statuses.count("PASS"),
            "failed": statuses.count("FAIL"),
            "warnings": statuses.count("WARN"),
        },
        "checks": checks,
        "extracted_numbers": extracted,
    }


def _verification_alerts(results: dict) -> list[str]:
    """Plain-language warnings about what a verification run did not prove."""
    summary = results.get("summary", {})
    total = summary.get("total_checks", 0)
    alerts: list[str] = []
    if total == 0:
        alerts.append(
            "WARNING: The abstract contains no numbers this stage recognises "
            "(paper counts, %, κ, CI, p, r). NOTHING WAS VERIFIED; a clean "
            "report here does not mean the abstract is correct."
        )
        return alerts
    for check in results.get("checks", []):
        if check.get("status") == "FAIL":
            alerts.append(
                f"FAIL: abstract says '{check.get('claim')}' but the data "
                f"gives {check.get('actual')} ({check.get('note', '')})"
            )
    if summary.get("passed", 0) == 0 and summary.get("failed", 0) == 0:
        alerts.append(
            f"WARNING: All {total} number(s) are unverified (WARN): no "
            f"reference data matched them. NOTHING WAS CONFIRMED."
        )
    elif summary.get("warnings", 0):
        alerts.append(
            f"NOTE: {summary['warnings']} of {total} number(s) could not be "
            f"checked against any data; see the WARN rows in the report."
        )
    return alerts


# ── Markdown report writer ──────────────────────────────────────────────

def _write_report(results: dict, path: Path) -> None:
    """Write markdown verification report with pass/fail table."""
    lines: list[str] = []
    summary = results.get("summary", {})

    lines.append("# Abstract Number Verification Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total checks:** {summary.get('total_checks', 0)}")
    lines.append(f"- **Passed:** {summary.get('passed', 0)}")
    lines.append(f"- **Failed:** {summary.get('failed', 0)}")
    lines.append(f"- **Warnings:** {summary.get('warnings', 0)}")
    lines.append("")

    checks = results.get("checks", [])
    if checks:
        lines.append("## Detailed Results")
        lines.append("")
        lines.append("| # | Claim | Expected | Actual | Status | Note |")
        lines.append("|---|-------|----------|--------|--------|------|")
        for i, c in enumerate(checks, 1):
            claim = c.get("claim", "").replace("|", "\\|")
            expected = c.get("expected", "")
            actual = c.get("actual", "N/A")
            if actual is None:
                actual = "N/A"
            status = c.get("status", "")
            note = c.get("note", "").replace("|", "\\|")
            icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(status, "")
            lines.append(
                f"| {i} | {claim} | {expected} | {actual} "
                f"| {icon} {status} | {note} |"
            )
        lines.append("")

    # Numbers found
    extracted = results.get("extracted_numbers", {})
    if extracted:
        lines.append("## Extracted Numbers")
        lines.append("")
        for category, items in extracted.items():
            if items:
                lines.append(f"### {category.replace('_', ' ').title()}")
                for item in items:
                    lines.append(f"- `{item.get('raw', item)}`")
                lines.append("")

    lines.append("---")
    lines.append("*Generated by Stage 9: Abstract Number Verification*")
    lines.append("")

    atomic_write_text(path, "\n".join(lines))


def _nothing_verified(validation_dir: Path, warning: str, progress) -> dict:
    """Save and return a run that checked nothing, naming why."""
    results = {
        "summary": {"total_checks": 0, "passed": 0, "failed": 0, "warnings": 0},
        "checks": [],
        "extracted_numbers": {},
        "warning": warning,
    }
    results_path = validation_dir / "verification_results.json"
    atomic_write_json(results_path, results)
    _remove_stale_report(validation_dir / "verification_report.md", progress)
    results["results_path"] = str(results_path)
    return results


def _remove_stale_report(path: Path, progress) -> None:
    """A previous report would contradict a run that verified nothing (13-M5)."""
    if not path.exists():
        return
    try:
        path.unlink()
        progress(f"Removed stale report from an earlier run: {path}")
    except OSError as exc:
        progress(f"WARNING: could not remove stale report {path}: {exc}")


# ── Main entry point ────────────────────────────────────────────────────

def run(
    claims_path: str | None = None,
    abstract_path: str | None = None,
    output_dir: str | None = None,
    config=None,
    cancel_event=None,
    progress_callback=None,
    clusters_path: str | None = None,
    **kw,
) -> dict:
    """Run Stage 9: abstract number verification.

    1. Load claims data
    2. Load abstract text
    3. Extract numbers/statistics from abstract
    4. Verify each against the actual data
    5. Optionally cross-check against robustness_results.json
    6. Save verification_results.json + verification_report.md
    """

    _cancelled = partial(is_cancelled, cancel_event)

    _progress = make_progress(logger, progress_callback)

    _progress("═" * 60)
    _progress("STAGE 9: ABSTRACT NUMBER VERIFICATION")
    _progress("═" * 60)
    announce_no_llm(_progress, "Stage 9 (Verify)", kw)

    # ── Resolve paths ─────────────────────────────────────────────────
    validation_dir = resolve_validation_dir(config, output_dir)
    run_dir = config.data_dir if config else Path.cwd() / "data"
    run_output_dir = config.output_dir if config else Path.cwd() / "output"
    corpus_path = resolve_claims_corpus(config, claims_path)
    if corpus_path is None:
        _progress("ERROR: No claims corpus found. Run earlier stages first.")
        return {"error": "No claims corpus found"}

    papers = load_paper_list(corpus_path)

    n_papers = len(papers)
    _progress(f"Loaded {n_papers} papers from {corpus_path}")

    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    # ── Load abstract text ───────────────────────────────────────────
    abs_path: Path | None = None
    if abstract_path:
        abs_path = Path(abstract_path)
    else:
        # Only this run's dirs — never another run's abstract.
        candidates_abs = [
            run_dir / "clusters" / "abstract_draft.md",
            run_output_dir / "abstract_draft.md",
        ]
        for cand in candidates_abs:
            if cand.exists():
                abs_path = cand
                break

    if abs_path is None or not abs_path.exists():
        searched = [str(abs_path)] if abs_path else [
            str(run_dir / "clusters" / "abstract_draft.md"),
            str(run_output_dir / "abstract_draft.md"),
        ]
        _progress(
            "WARNING: No abstract_draft.md found. NOTHING WAS VERIFIED. "
            "Run Stage 6 (Synthesize) to produce the abstract first. "
            f"Looked in: {', '.join(searched)}"
        )
        return _nothing_verified(validation_dir, "No abstract_draft.md found", _progress)

    abstract_text = abs_path.read_text(encoding="utf-8")
    _progress(f"Loaded abstract ({len(abstract_text)} chars) from {abs_path}")
    if not abstract_text.strip():
        _progress(
            f"WARNING: {abs_path} is empty. NOTHING WAS VERIFIED. Stage 6 (Synthesize) "
            "produced no abstract; re-run it."
        )
        return _nothing_verified(
            validation_dir, "abstract_draft.md is empty (Stage 6 produced no abstract)",
            _progress)

    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    # ── Resolve optional cross-check paths ───────────────────────────
    robustness_path = validation_dir / "robustness_results.json"
    if not robustness_path.exists():
        robustness_path = resolve_validation_dir(config) / "robustness_results.json"
    if not robustness_path.exists():
        _progress(
            "WARNING: No robustness_results.json found. CIs, p-values and "
            "correlations in the abstract cannot be checked (run Stage 8)."
        )
        robustness_path = None

    # ── Run verification ─────────────────────────────────────────────
    # Use the abstract's sibling artifact when explicitly selected; otherwise
    # use only this run's clusters, never a global/other-run fallback.
    if clusters_path:
        cluster_source = Path(clusters_path)
    elif abstract_path:
        cluster_source = abs_path.with_name("clusters.json")
        if not cluster_source.exists() and abs_path.parent.resolve() == run_output_dir.resolve():
            cluster_source = run_dir / "clusters" / "clusters.json"
    else:
        cluster_source = run_dir / "clusters" / "clusters.json"
    _progress("Extracting numbers from abstract and verifying...")
    results = verify(
        papers, abstract_text,
        robustness_path=robustness_path,
        config=config,
        clusters_path=cluster_source,
    )

    summary = results["summary"]
    _progress(
        f"Verification complete: {summary['total_checks']} checks — "
        f"{summary['passed']} passed, {summary['failed']} failed, "
        f"{summary['warnings']} warnings"
    )
    for line in _verification_alerts(results):
        _progress(line)

    if _cancelled():
        _progress("Cancelled.")
        return {"cancelled": True}

    # ── Save outputs ─────────────────────────────────────────────────
    results_path = validation_dir / "verification_results.json"
    atomic_write_json(results_path, results)
    _progress(f"Results saved: {results_path}")
    results["results_path"] = str(results_path)

    report_path = validation_dir / "verification_report.md"
    _write_report(results, report_path)
    _progress(f"Report saved: {report_path}")
    results["report_path"] = str(report_path)

    _progress("═" * 60)
    _progress("STAGE 9 COMPLETE")
    _progress("═" * 60)

    return results


# ── Output checker ──────────────────────────────────────────────────────
