"""Prepare corpus-wide, result-linked evidence for meta-analysis."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import re
from pathlib import Path

from core.io import atomic_write_json
from process.meta_analysis_effects import (
    REASON_LABELS, STATUS_CALCULABLE, STATUS_CONVERTED, STATUS_PASSED_UP, effect_id,
    evaluate_result,
)
from process.paths import load_paper_list, resolve_claims_corpus
from process.statistical_extractor_llm import is_failed_llm_review
from process.statistical_extractor_llm_run import apply_llm_context_reviews
from process.statistical_extractor_pipeline import (
    collect_human_review, extract_statistical_evidence, metrics_for_meta_analysis, summarize,
)
from process.statistical_extractor_text import (
    article_id, enrich_from_saved_fulltext, text_candidates,
)
from process.term_normalization import normalization_audit_path


def _read(path, default):
    if not Path(path).exists():
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _unverified(review) -> bool:
    return not isinstance(review, dict) or is_failed_llm_review(review)


def _carry_reviews(records, papers, old_records):
    """Reuse a saved verdict only when its original source span still matches.

    Abstract results remain valid evidence when full text becomes available;
    they are kept alongside the new full-text candidates, never used to approve
    other statistics merely because the paper is the same. A verdict already
    attached is replaced only when it is a failed check.
    """
    old_by_id = {r.get("article_id"): r for r in old_records}
    for record, paper in zip(records, papers):
        sources = dict(text_candidates(paper))
        for old in old_by_id.get(record["article_id"], {}).get("results", []):
            review = old.get("llm_context_review")
            if not isinstance(review, dict):
                continue
            loc = old.get("location", {})
            source = sources.get(loc.get("text_source"), "")
            start, end = loc.get("char_start"), loc.get("char_end")
            if (not isinstance(start, int) or not isinstance(end, int)
                    or not 0 <= start < end <= len(source)):
                continue
            # Old extraction inserted spaces into decimals and could include
            # the opening parenthesis in its span. Require the same complete
            # statistic at that location after only whitespace/edge cleanup.
            normalize = lambda value: re.sub(r"\s+", "", value.strip(" ()"))
            if normalize(source[start:end]) != normalize(old.get("test_text", "")):
                continue
            while start < end and source[start] in " (":
                start += 1
            matches = [r for r in record["results"]
                       if r.get("location", {}).get("text_source") == loc.get("text_source")
                       and r.get("location", {}).get("char_start") == start]
            if len(matches) == 1:
                if _unverified(matches[0].get("llm_context_review")):
                    matches[0]["llm_context_review"] = deepcopy(review)
            elif not matches:
                retained = deepcopy(old)
                retained["location"]["char_start"] = start
                retained["test_text"] = source[start:end]
                record["results"].append(retained)


def recheck_unverified(config, records, *, context_tokens: int | None, progress=None,
                       cancel_event=None) -> dict:
    """Ask the configured model about statistics that still have no context verdict.

    Stage 2c leaves a statistic unverified when its request failed (for
    example the model was not loaded) or its bundle reply gave no verdict.
    Verdicts attach to the given records' results; returns the review counts.
    ``context_tokens`` is the already resolved model capacity, not a synthetic
    Stage 2c bundle boundary, so automatic bundle planning can use it directly.
    """
    pending = []
    for record in records:
        results = [r for r in record.get("results", []) if _unverified(r.get("llm_context_review"))]
        if results:
            pending.append({**record, "results": results, "human_review": []})
    if not pending:
        return {"llm_reviewed": 0, "llm_failed": 0}
    from process.llm import set_config

    set_config(config)
    return apply_llm_context_reviews(
        pending, model=config.llm_model, config=config,
        max_contexts=0, include_human_flags=False,
        progress_callback=progress, cancel_event=cancel_event,
        concurrent_workers=max(1, int(getattr(config, "concurrent_workers", None) or 1)),
        stagger_delay=float(getattr(config, "stagger_delay", 0.0) or 0.0),
        context_tokens=context_tokens if context_tokens and context_tokens > 0 else None,
        bundle_tokens=0,
        # Model calls target only unresolved statistics, while canonical label
        # selection sees the complete extraction corpus. Only the pending result
        # objects are mutated; prior labels are not repaired here.
        normalization_scope_records=records,
    )


def prepare_corpus_metrics(config, metrics_path: Path, *, recheck: bool = False,
                           context_tokens: int | None = 0, progress=None,
                           cancel_event=None) -> tuple[list[dict], dict]:
    """Recompute candidates for every retained paper using saved source text.

    Context verdicts come from Stage 2c, then from this analysis's earlier
    re-checks. With ``recheck`` the configured model is asked about the
    statistics still unverified (see ``recheck_unverified``); otherwise no LLM
    request is made. The Stage 2c files are never modified.
    """
    corpus_path = resolve_claims_corpus(config)
    if corpus_path is None:
        return _read(metrics_path, []), {}
    papers = load_paper_list(corpus_path)
    enriched, _ = enrich_from_saved_fulltext(
        papers, config.corpus_dir / "corpus_fulltext.json")
    extraction = extract_statistical_evidence(enriched)
    records = extraction["records"]
    evidence_dir = config.output_dir / "meta_analysis_evidence"
    for prior_path in (config.claims_dir / "statistical_extractions.json",
                       evidence_dir / "statistical_extractions.json"):
        _carry_reviews(records, enriched, _read(prior_path, {}).get("records", []))
    counts = (recheck_unverified(config, records, context_tokens=context_tokens,
                                 progress=progress, cancel_event=cancel_event)
              if recheck else {})
    metrics = metrics_for_meta_analysis(records)
    extraction["summary"] = summarize(records, collect_human_review(records))
    evidence_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = evidence_dir / "statistical_meta_analysis_input.json"
    atomic_write_json(prepared_path, metrics)
    extraction_path = evidence_dir / "statistical_extractions.json"
    normalization_audit = counts.get("term_normalization")
    atomic_write_json(extraction_path, {
        "input_path": str(corpus_path), "records": records,
        "summary": extraction["summary"], "llm_rerun": bool(counts.get("llm_reviewed")),
        "term_normalization": normalization_audit,
    })
    term_audit_path = None
    if normalization_audit:
        term_audit_path = normalization_audit_path(extraction_path)
        atomic_write_json(term_audit_path, normalization_audit)
    unverified = sum(_unverified(r.get("llm_context_review"))
                     for record in records for r in record.get("results", []))
    return metrics, {
        "n_unverified_statistics": unverified,
        **({"n_rechecked": counts.get("llm_reviewed", 0),
            "n_recheck_failed": counts.get("llm_failed", 0),
            "model_recheck": {
                "scope": counts.get("scope", {}),
                "bundles": counts.get("bundles", {}),
            }} if recheck else {}),
        "corpus_path": str(corpus_path), "prepared_input_path": str(prepared_path),
        **({"term_normalization_path": str(term_audit_path)} if term_audit_path else {}),
        "n_corpus_papers": len(papers),
        "n_fulltext_papers": extraction["summary"]["papers_with_fulltext"],
        "n_abstract_only_papers": extraction["summary"]["papers_with_abstract_only"],
        "n_papers_with_statistics": extraction["summary"]["papers_with_results"],
        "n_statistical_results": extraction["summary"]["results"],
        "papers": [{"article_id": article_id(p, i), "title": p.get("title", ""),
                    "n_statistical_results": len(records[i]["results"])}
                   for i, p in enumerate(papers)],
    }


def _linked_results(entry):
    tests = entry.get("statistical_tests", [])
    if tests:
        for test in tests:
            if "effect_sizes" in test:
                yield test
            else:
                # Older exports retained the corrected fields in the review.
                review = test.get("llm_context_review") or {}
                corrected = review.get("corrected_fields") or {}
                if corrected.get("effect_sizes"):
                    yield {**test, **corrected, "llm_context_review": review}
                else:
                    yield {**test, "effect_sizes": [], "linkage_missing": True}
        return
    # Explicit legacy input: support one effect and one uncertainty per row.
    metrics = entry.get("metrics") or entry
    yield {**metrics, "legacy_input": True}


def evaluate_metrics(raw, *, allow_paper_n: bool = False, decisions: dict | None = None):
    """Return ``(studies, paper_audit, statistic_rows)`` for meta-analysis input.

    ``statistic_rows`` has one row per extracted statistic: what was found,
    whether it became a calculable effect, and why not when it did not.
    *decisions* maps ``effect_id`` to the researcher's include/group/direction.
    """
    decisions = decisions or {}
    if not isinstance(raw, list):
        raise ValueError("Meta-analysis input must be a list of paper records")
    studies, audit, rows = [], [], []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        aid = str(entry.get("article_id") or entry.get("doi") or "")
        reasons = Counter()
        count = 0
        for result in _linked_results(entry):
            study, row = evaluate_result(entry, result, allow_paper_n=allow_paper_n,
                                         decision=decisions.get(effect_id(entry, result)))
            if study is not None and not aid:
                study, row = None, {**row, "status": STATUS_PASSED_UP,
                                    "reason_code": "missing_article_id",
                                    "reason": REASON_LABELS["missing_article_id"]}
            rows.append(row)
            if study is not None:
                studies.append(study)
                count += 1
            else:
                reasons[row["reason"]] += 1
        audit.append({"article_id": aid, "title": entry.get("title", ""),
                      "n_calculable_effects": count, "exclusion_reasons": dict(reasons)})
    return studies, audit, rows


def parse_metrics(raw, *, allow_paper_n: bool = False, decisions: dict | None = None):
    """Return all calculable linked effects and a reason for every omitted row."""
    studies, audit, _ = evaluate_metrics(raw, allow_paper_n=allow_paper_n, decisions=decisions)
    return studies, audit


def statistics_funnel(rows: list[dict]) -> dict:
    """Counts of extracted statistics by outcome, uncertainty source and reason."""
    status = Counter(row["status"] for row in rows)
    return {
        "statistics": len(rows),
        STATUS_CALCULABLE: status.get(STATUS_CALCULABLE, 0),
        STATUS_CONVERTED: status.get(STATUS_CONVERTED, 0),
        STATUS_PASSED_UP: status.get(STATUS_PASSED_UP, 0),
        "by_uncertainty_source": dict(Counter(
            row["uncertainty_source"] for row in rows if row["status"] == STATUS_CALCULABLE)),
        # Pooled or awaiting a decision: every statistic converted to an effect.
        "converted_by_route": dict(Counter(row["route"] for row in rows if row.get("route"))),
        "passed_up_by_reason": {
            REASON_LABELS.get(code, code): n for code, n in Counter(
                row["reason_code"] for row in rows if row["status"] == STATUS_PASSED_UP
            ).most_common()},
    }
