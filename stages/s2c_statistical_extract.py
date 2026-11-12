"""Stage 2c: deterministic statistical evidence extraction.

Runs after relevance scoring to surface statistical tests, subjects, numerical
results, source locations, and human-review cases before LLM claim extraction.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.io import atomic_write_json
from process.meta_analysis import STAGE2C_META_ANALYSIS_INPUT
from process.meta_analysis_groups import REVIEW_FILE, load_decisions, write_review
from process.meta_analysis_inputs import evaluate_metrics, statistics_funnel
from process.meta_analysis_statistics_audit import (
    STATISTICS_AUDIT_FILE,
    found_lines,
    funnel_lines,
    write_statistics_audit,
)
from process.statistical_extractor import (
    apply_llm_context_reviews,
    collect_human_review,
    extract_statistical_evidence,
    metrics_for_meta_analysis,
    summarize,
)
from process.statistical_extractor_pipeline import (
    STAGE2C_EXTRACTION_ARTIFACT,
    STAGE2C_REVIEW_ARTIFACT,
)
from process.statistical_extractor_text import enrich_from_saved_fulltext
from process.term_normalization import (
    normalize_statistical_records,
    normalization_audit_path,
)
from stages._common import is_cancelled, make_progress

logger = logging.getLogger(__name__)


def run(
    config=None,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    cancel_event: Any = None,
    progress_callback: Callable[[str], None] | None = None,
    **kwargs,
) -> dict:
    """Extract statistical evidence from Stage 2 relevant papers."""
    if config is None:
        raise ValueError("config is required")

    input_path = Path(input_path) if input_path else config.claims_dir / "relevant.json"
    output_path = (
        Path(output_path)
        if output_path
        else config.claims_dir / STAGE2C_EXTRACTION_ARTIFACT
    )
    review_path = output_path.with_name(STAGE2C_REVIEW_ARTIFACT)
    # Default metrics mirror goes to this stage's own input file, not Stage 7's
    # auto_extracted_metrics.json (different schema; sharing the path made the
    # file's schema depend on which stage ran last).
    metrics_path = (
        Path(metrics_path)
        if metrics_path
        else config.claims_dir / STAGE2C_META_ANALYSIS_INPUT
    )
    meta_analysis_input_path = output_path.with_name(STAGE2C_META_ANALYSIS_INPUT)

    _progress = make_progress(logger, progress_callback, log_with_callback=False)

    _progress("Stage 2c: Statistical Evidence Extraction")
    _progress(f"  Input: {input_path}")
    _progress(f"  Output: {output_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open(encoding="utf-8") as f:
        papers = json.load(f)
    if not isinstance(papers, list):
        raise ValueError(f"Expected a list of papers in {input_path}")

    corpus_path = config.corpus_dir / "corpus_fulltext.json"
    papers, enriched_count = enrich_from_saved_fulltext(papers, corpus_path)
    _progress(f"Attached saved full text to {enriched_count}/{len(papers)} retained papers")

    result = extract_statistical_evidence(
        papers,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    if result.get("cancelled"):
        return {
            "cancelled": True,
            "summary": result.get("summary", {}),
            "output_path": str(output_path),
        }

    records = result.get("records", [])
    bundle_tokens = max(0, int(kwargs.get("llm_bundle_tokens") or 0))
    llm_counts: dict[str, Any] = {}
    if kwargs.get("use_llm_context_check"):
        llm_counts = _run_llm_context_check(
            config, records, kwargs, bundle_tokens, cancel_event, progress_callback, _progress,
        )
        if is_cancelled(cancel_event):
            return {
                "cancelled": True,
                "summary": result.get("summary", {}),
                "output_path": str(output_path),
            }
        _progress(
            "LLM context check complete: "
            f"{llm_counts.get('llm_supported', 0)} supported, "
            f"{llm_counts.get('llm_corrected', 0)} corrected, "
            f"{llm_counts.get('llm_rejected', 0)} rejected/unsafe"
        )
        bundles = llm_counts.get("bundles") or {}
        rejections = bundles.get("rejections", [])
        failed = llm_counts.get("llm_failed", 0)
        terminal = int(bundles.get("terminal_failures") or 0)
        if failed:
            _progress(
                f"WARNING: LLM context check FAILED on {failed} text window(s) "
                f"({terminal} terminal review failure(s) after local recovery). "
                "Their statistics are kept in "
                "the meta-analysis input as unchecked and listed for human review."
            )
        if bundles.get("splits"):
            _progress(
                "LLM token/reply recovery: "
                f"{bundles['splits']} split action(s), {bundles.get('recoveries', 0)} "
                f"recovery event(s), {bundles.get('recovered_statistics', 0)} recovered "
                "statistic verdict(s). Later paper bundles continued normally."
            )
        if rejections:
            status = (
                f"{terminal} statistic(s) remain for human review"
                if terminal else
                "all affected statistics were recovered in smaller same-run retries"
            )
            _progress(
                f"LLM reply audit: {len(rejections)} incomplete/malformed attempt(s); {status}. "
                "Valid member verdicts were retained, and only invalid IDs were retried. Every "
                f"raw attempt is under llm_context_check.bundles.attempts in {output_path.name}."
            )

    # Population/outcome/comparison are structured labels, not report prose.
    # Canonicalize them once here, while raw model values are still attached to
    # their source reviews.  Meta-analysis consumes the saved canonical labels.
    normalization_audit = normalize_statistical_records(records, config)
    term_audit_path = normalization_audit_path(output_path)
    for field, details in normalization_audit.get("fields", {}).items():
        _progress(
            "Structured-term action: "
            f"{field}: {details.get('input_variants', 0)} raw variants -> "
            f"{details.get('canonical_terms', 0)} canonical labels; "
            f"{details.get('changed_occurrences', 0)} occurrence(s) changed. "
            "Raw mappings are retained."
        )
    human_review = collect_human_review(records)
    summary = summarize(records, human_review)
    metrics = metrics_for_meta_analysis(records)
    decisions = load_decisions(config)
    _, _, statistic_rows = evaluate_metrics(
        metrics, allow_paper_n=bool(getattr(config, "meta_analysis_use_paper_sample_size", False)),
        decisions=decisions)
    statistics_audit_path = review_path.with_name(STATISTICS_AUDIT_FILE)

    artifact = {
        "stage": "statistical_evidence_extraction",
        "input_path": str(input_path),
        "fulltext_source_path": str(corpus_path),
        "fulltext_records_attached": enriched_count,
        "llm_context_check": {
            "enabled": bool(kwargs.get("use_llm_context_check")),
            "provider": getattr(config, "llm_provider", ""),
            "model": str(kwargs.get("llm_model") or getattr(config, "llm_model", "") or ""),
            "max_contexts": int(kwargs.get("max_llm_contexts", 0)),
            "bundle_tokens": bundle_tokens,
            "scope": llm_counts.get("scope"),
            "budget_planning": (llm_counts.get("bundles") or {}).get("planning"),
            "bundles": llm_counts.get("bundles"),
        },
        "summary": summary,
        "term_normalization": normalization_audit,
        "records": records,
        "human_review": human_review,
    }
    atomic_write_json(output_path, artifact)
    atomic_write_json(term_audit_path, normalization_audit)
    atomic_write_json(review_path, human_review)
    atomic_write_json(meta_analysis_input_path, metrics)
    if metrics_path != meta_analysis_input_path:
        atomic_write_json(metrics_path, metrics)
    write_statistics_audit(statistics_audit_path, statistic_rows)
    effects_review_path = write_review(review_path.with_name(REVIEW_FILE), statistic_rows, decisions)

    _progress(
        "Statistical extraction complete: "
        f"{summary.get('results', 0)} results, "
        f"{summary.get('human_review_items', 0)} review items"
    )
    _progress(
        f"  Text coverage: {summary['papers_with_fulltext']} full text, "
        f"{summary['papers_with_abstract_only']} abstract only; "
        f"{summary['papers_with_results']}/{summary['papers']} papers with statistical candidates"
    )
    for line in found_lines(statistic_rows) + funnel_lines(statistics_funnel(statistic_rows)):
        _progress(f"  {line}")
    _progress(f"  Statistics audit (every statistic, used or passed up, and why): "
              f"{statistics_audit_path}")
    _progress(f"  Effects to review (include / group / direction): {effects_review_path}")
    _progress(f"  Review required: {review_path}")
    _progress(f"  Meta-analysis input: {meta_analysis_input_path}")
    _progress(f"  Structured-term normalization audit: {term_audit_path}")
    if metrics_path != meta_analysis_input_path:
        _progress(f"  Legacy metrics mirror: {metrics_path}")

    return {
        "statistics_audit_path": str(statistics_audit_path),
        "effects_review_path": str(effects_review_path),
        "summary": summary,
        "records": records,
        "human_review": human_review,
        "output_path": str(output_path),
        "review_path": str(review_path),
        "term_normalization_path": str(term_audit_path),
        "meta_analysis_input_path": str(meta_analysis_input_path),
        "metrics_path": str(metrics_path),
    }


def _run_llm_context_check(
    config,
    records: list[dict],
    kwargs: dict[str, Any],
    bundle_tokens: int,
    cancel_event: Any,
    progress_callback: Callable[[str], None] | None,
    _progress: Callable[[str], None],
) -> dict[str, Any]:
    """Phase 2: LLM review of each statistic's text window."""
    from process.llm import set_config

    llm_model = str(kwargs.get("llm_model") or getattr(config, "llm_model", "") or "").strip()
    workers = max(1, int(getattr(config, "concurrent_workers", None) or 1))
    stagger = kwargs.get("stagger_delay")
    if stagger is None:
        stagger = getattr(config, "stagger_delay", 0.5)
    bundling = (
        f"explicit synthetic total-request limit {bundle_tokens:,} tokens"
        if bundle_tokens else
        "automatic complete-request packing from resolved model context"
    )
    _progress(
        "Phase 2: LLM context check on local statistic windows "
        f"(model={llm_model or 'not configured'}, workers={workers}, {bundling})"
    )
    set_config(config)
    return apply_llm_context_reviews(
        records,
        model=llm_model,
        max_contexts=int(kwargs.get("max_llm_contexts", 0)),
        include_human_flags=True,
        use_thinking=bool(kwargs.get("llm_use_thinking", False)),
        max_tokens=(int(kwargs["llm_max_tokens"])
                    if int(kwargs.get("llm_max_tokens") or 0) > 0 else None),
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        concurrent_workers=workers,
        stagger_delay=float(stagger or 0.0),
        bundle_tokens=bundle_tokens,
        config=config,
    )
