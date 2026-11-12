"""Article-level extraction, summaries, and meta-analysis export."""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from process.statistical_extractor_block import (
    extract_from_block,
)
from process.statistical_extractor_llm import (
    apply_llm_corrected_fields,
    as_failed_review,
    has_llm_corrected_fields,
    is_failed_llm_review,
)
from process.statistical_extractor_parsers import (
    parse_subjects,
    repair_decimal_spaced_result_numbers,
    result_missing_fields,
)
from process.statistical_extractor_text import (
    article_id,
    paragraph_blocks,
    text_candidates,
)
from process.term_normalization import TERM_NORMALIZATION_KEY

logger = logging.getLogger(__name__)

STAGE2C_EXTRACTION_ARTIFACT = "statistical_extractions.json"
STAGE2C_REVIEW_ARTIFACT = "statistical_review_required.json"


def extract_article(paper: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Extract statistical evidence for one article."""
    aid = article_id(paper, index)
    candidates = text_candidates(paper)
    selected_source = candidates[0][0] if candidates else "none"
    selected_text = candidates[0][1] if candidates else ""
    blocks = paragraph_blocks(selected_text, selected_source)

    results: list[dict[str, Any]] = []
    review_flags: list[dict[str, Any]] = []
    subjects: list[dict[str, Any]] = []
    for block in blocks:
        block_results, block_flags = extract_from_block(block, paper)
        results.extend(block_results)
        review_flags.extend(block_flags)
        subjects.extend(parse_subjects(block.text))

    if not candidates:
        review_flags.append({
            "reason": "no_text_available",
            "missing_fields": ["text", "statistical_test", "subjects", "effect_size"],
            "test_text": "",
            "context": "",
            "location": {
                "text_source": "none",
                "paragraph_index": -1,
                "section": "",
                "char_start": -1,
                "char_end": -1,
            },
        })
    elif not results:
        # Keep the complete selected text when paragraph recovery produced no
        # block. The model-facing bundle planner, not this deterministic stage,
        # decides whether a request must be split for the loaded context.
        context = blocks[0].text if blocks else selected_text
        review_flags.append({
            "reason": "no_statistical_result_found",
            "missing_fields": ["statistical_test", "effect_size", "p_value_or_ci"],
            "test_text": "",
            "context": context,
            "location": {
                "text_source": selected_source,
                "paragraph_index": 0 if blocks else -1,
                "section": blocks[0].section if blocks else "",
                "char_start": blocks[0].char_start if blocks else 0,
                "char_end": blocks[0].char_end if blocks else len(selected_text),
            },
        })

    return {
        "article_id": aid,
        "title": paper.get("title", ""),
        "doi": paper.get("doi", ""),
        "pmid": paper.get("pmid", ""),
        "year": paper.get("year", ""),
        "text_source": selected_source,
        "text_length": len(selected_text),
        # The model needs methods/sample information as well as the target
        # statistic. Keep complete abstracts rather than a clipped local window.
        "review_context": selected_text if selected_source == "abstract" else str(paper.get("abstract") or ""),
        "text_source_path": paper.get("statistical_text_source_path", ""),
        "subject_summary": dedupe_subjects(subjects),
        "results": results,
        "human_review": review_flags,
        "needs_human_review": bool(review_flags),
    }


def dedupe_subjects(subjects: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, Any]] = set()
    out = []
    for item in subjects:
        key = (item.get("n"), item.get("label"), item.get("age_mean"))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def extract_statistical_evidence(
    papers: list[dict[str, Any]],
    *,
    progress_callback=None,
    cancel_event=None,
) -> dict[str, Any]:
    """Extract statistical evidence from a scored/relevant-paper list."""
    records = []
    human_review = []
    for index, paper in enumerate(papers):
        if cancel_event is not None:
            is_set = cancel_event.is_set() if hasattr(cancel_event, "is_set") else bool(cancel_event)
            if is_set:
                return {
                    "cancelled": True,
                    "records": records,
                    "human_review": human_review,
                    "summary": summarize(records, human_review),
                }
        if progress_callback:
            title = str(paper.get("title", "") or f"paper {index + 1}")[:80]
            progress_callback(f"[{index + 1}/{len(papers)}] Extracting statistics: {title}")
        record = extract_article(paper, index)
        records.append(record)
        for flag in record["human_review"]:
            human_review.append({
                "article_id": record["article_id"],
                "title": record["title"],
                **flag,
            })

    return {
        "cancelled": False,
        "records": records,
        "human_review": collect_human_review(records),
        "summary": summarize(records, human_review),
    }


def collect_human_review(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    human_review = []
    for record in records:
        for flag in record.get("human_review", []) or []:
            human_review.append({
                "article_id": record.get("article_id", ""),
                "title": record.get("title", ""),
                **flag,
            })
        for result in record.get("results", []) or []:
            review = result.get("llm_context_review")
            if isinstance(review, dict) and review.get("needs_human_review"):
                human_review.append({
                    "article_id": record.get("article_id", ""),
                    "title": record.get("title", ""),
                    "reason": "llm_context_review",
                    "missing_fields": result.get("missing_fields", []),
                    "test_text": result.get("test_text", ""),
                    "context": result.get("context", ""),
                    "location": result.get("location", {}),
                    "llm_context_review": review,
                })
    return human_review


def summarize(records: list[dict[str, Any]], human_review: list[dict[str, Any]]) -> dict[str, int]:
    result_count = sum(len(record.get("results", [])) for record in records)
    complete_count = sum(
        1
        for record in records
        for result in record.get("results", [])
        if not result.get("needs_human_review")
    )
    all_reviews = [
        result.get("llm_context_review")
        for record in records
        for result in record.get("results", [])
        if isinstance(result.get("llm_context_review"), dict)
    ]
    llm_reviews = [review for review in all_reviews if not is_failed_llm_review(review)]
    return {
        "papers": len(records),
        "papers_with_fulltext": sum(record.get("text_source") in {"full_text", "full_text_file"} for record in records),
        "papers_with_abstract_only": sum(record.get("text_source") == "abstract" for record in records),
        "papers_with_results": sum(1 for record in records if record.get("results")),
        "results": result_count,
        "complete_results": complete_count,
        "human_review_items": len(human_review),
        "llm_reviewed": len(all_reviews),
        "llm_supported": sum(
            1 for review in llm_reviews
            if review.get("represents_study_data") and review.get("usable_for_meta_analysis")
        ),
        "llm_rejected": sum(1 for review in llm_reviews if review.get("represents_study_data") is False),
        "llm_corrected": sum(1 for review in llm_reviews if has_llm_corrected_fields(review)),
        "llm_failed": len(all_reviews) - len(llm_reviews),
    }


def _excluded_by_review(review: Any) -> bool:
    # A failed check has no verdict, so the statistic is kept as if unchecked.
    return (isinstance(review, dict) and not is_failed_llm_review(review)
            and not review.get("usable_for_meta_analysis", False))


def _exported_test(metric_result: dict[str, Any], review: Any) -> dict[str, Any]:
    review_normalization = (
        review.get(TERM_NORMALIZATION_KEY)
        if isinstance(review, dict) else None
    )
    return {
        "test_type": metric_result.get("test_type"),
        "test_text": metric_result.get("test_text"),
        "statistic": metric_result.get("statistic"),
        # Keep each effect with its own uncertainty, N, p-values and context.
        # The article-wide arrays below are a legacy export only.
        "effect_sizes": metric_result.get("effect_sizes", []),
        "sample_sizes": metric_result.get("sample_sizes", []),
        "confidence_intervals": metric_result.get("confidence_intervals", []),
        "p_values": metric_result.get("p_values", []),
        "degrees_of_freedom": metric_result.get("degrees_of_freedom") or {},
        "standard_error": metric_result.get("standard_error"),
        "standard_error_scale": metric_result.get("standard_error_scale", ""),
        "context": metric_result.get("context", ""),
        "analysis_group": metric_result.get("analysis_group", ""),
        "outcome": metric_result.get("outcome", ""),
        "comparison": metric_result.get("comparison", ""),
        "population": metric_result.get("population", ""),
        "location": metric_result.get("location"),
        "needs_human_review": metric_result.get("needs_human_review", False),
        "llm_correction_applied": bool(metric_result.get("llm_correction_applied")),
        "llm_context_review": review if isinstance(review, dict) else None,
        # The canonical labels remain in their established fields/review.  This
        # marker lets meta-analysis verify provenance without re-normalizing.
        TERM_NORMALIZATION_KEY: (
            metric_result.get(TERM_NORMALIZATION_KEY) or review_normalization
        ),
    }


def _legacy_arrays(results: list[dict[str, Any]]) -> dict[str, list]:
    """Article-wide arrays kept for older loaders (usable results only)."""
    effect_sizes = [float(e["value"]) for r in results for e in r.get("effect_sizes", [])
                    if isinstance(e.get("value"), (int, float))]
    sample_sizes = {s["n"] for r in results for s in r.get("sample_sizes", [])
                    if isinstance(s.get("n"), int)}
    intervals = [[float(ci["lower"]), float(ci["upper"])] for r in results
                 for ci in r.get("confidence_intervals", [])
                 if isinstance(ci.get("lower"), (int, float))
                 and isinstance(ci.get("upper"), (int, float))]
    p_values = [float(p["value"]) for r in results for p in r.get("p_values", [])
                if isinstance(p.get("value"), (int, float))]
    return {"effect_sizes": effect_sizes, "sample_sizes": sorted(sample_sizes, reverse=True),
            "confidence_intervals": intervals, "p_values": p_values}


def _paper_sample_sizes(record: dict[str, Any]) -> list[int]:
    return sorted({s["n"] for s in record.get("subject_summary") or []
                   if isinstance(s, dict) and isinstance(s.get("n"), int) and s["n"] > 0},
                  reverse=True)


def metrics_for_meta_analysis(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten extracted rows into the shape consumed by meta-analysis loaders.

    Every extracted statistic is exported, including ones the context review
    excluded, so the meta-analysis audit can show what was passed over and why.
    """
    promote_reviewed_results(records)
    metrics = []
    for record in records:
        tests: list[dict[str, Any]] = []
        usable: list[dict[str, Any]] = []
        for result in record.get("results", []):
            metric_result = repair_decimal_spaced_result_numbers(result)
            review = metric_result.get("llm_context_review")
            if isinstance(review, dict) and has_llm_corrected_fields(review):
                metric_result = apply_llm_corrected_fields(metric_result)
            tests.append(_exported_test(metric_result, review))
            if not _excluded_by_review(review):
                usable.append(metric_result)
        if tests:
            metrics.append({
                "article_id": record.get("article_id", ""),
                "doi": record.get("doi", ""),
                "pmid": record.get("pmid", ""),
                "title": record.get("title", ""),
                **_legacy_arrays(usable),
                "paper_sample_sizes": _paper_sample_sizes(record),
                "statistical_tests": tests,
                "needs_human_review": record.get("needs_human_review", False),
            })
    return metrics


def promote_reviewed_results(records: list[dict[str, Any]]) -> int:
    """Retain corrected numerical results discovered while reviewing flags.

    A flag and an existing result at the same span are aliases, not two studies.
    Original flags stay available for audit. Calling this repeatedly is safe.
    """
    promoted = 0
    for record in records:
        results = record.setdefault("results", [])
        spans = {(json.dumps(r.get("location", {}), sort_keys=True), r.get("test_text", "")) for r in results}
        texts = {r.get("test_text", "") for r in results if r.get("test_text")}
        for flag in record.get("human_review", []):
            review = flag.get("llm_context_review")
            if (not isinstance(review, dict) or is_failed_llm_review(review)
                    or not review.get("usable_for_meta_analysis") or not has_llm_corrected_fields(review)):
                continue
            span = (json.dumps(flag.get("location", {}), sort_keys=True), flag.get("test_text", ""))
            if span in spans:
                continue
            corrected = apply_llm_corrected_fields(flag)
            text = corrected.get("test_text", "")
            if not text or text in texts:
                continue
            if not any(corrected.get(key) for key in ("effect_sizes", "p_values", "confidence_intervals")):
                continue
            corrected.pop("reason", None)
            corrected["recovered_from_review_flag"] = True
            corrected["missing_fields"] = result_missing_fields(corrected, corrected.get("sample_sizes", []))
            corrected["needs_human_review"] = bool(corrected["missing_fields"] or review.get("needs_human_review"))
            results.append(corrected)
            spans.add(span)
            texts.add(text)
            promoted += 1
    return promoted


def _upgrade_items(items: list[Any]) -> tuple[list[Any], int]:
    upgraded: list[Any] = []
    count = 0
    for item in items:
        review = item.get("llm_context_review") if isinstance(item, dict) else None
        if (
            isinstance(review, dict)
            and is_failed_llm_review(review)
            and not review.get("check_failed")
        ):
            upgraded.append({**item, "llm_context_review": as_failed_review(review)})
            count += 1
        else:
            upgraded.append(item)
    return upgraded, count


def upgrade_legacy_failed_reviews(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Return records with old-format failed LLM checks converted to no-verdict form.

    Before ``check_failed`` existed, a failed LLM call was saved as a rejection,
    which dropped the statistic from the meta-analysis input.
    """
    upgraded_records: list[dict[str, Any]] = []
    total = 0
    for record in records:
        results, n_results = _upgrade_items(record.get("results", []) or [])
        flags, n_flags = _upgrade_items(record.get("human_review", []) or [])
        if n_results or n_flags:
            record = {**record, "results": results, "human_review": flags}
            total += n_results + n_flags
        upgraded_records.append(record)
    return upgraded_records, total


def refresh_stage2c_outputs_for_legacy_failures(meta_input_path: Path) -> int:
    """Rewrite Stage 2c outputs saved with old-format failed LLM checks.

    Looks for the extraction artifact next to ``meta_input_path``; if it holds
    failures recorded as rejections, rewrites the artifact, the review list and
    the meta-analysis input so those statistics are kept as unchecked. Returns
    the number of reviews converted (0 when nothing needed changing).
    """
    from core.io import atomic_write_json

    artifact_path = meta_input_path.with_name(STAGE2C_EXTRACTION_ARTIFACT)
    if not artifact_path.exists():
        return 0
    with open(artifact_path, encoding="utf-8") as f:
        artifact = json.load(f)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("records"), list):
        return 0

    records, converted = upgrade_legacy_failed_reviews(artifact["records"])
    if not converted:
        return 0

    human_review = collect_human_review(records)
    atomic_write_json(artifact_path, {
        **artifact,
        "summary": summarize(records, human_review),
        "records": records,
        "human_review": human_review,
    })
    atomic_write_json(artifact_path.with_name(STAGE2C_REVIEW_ARTIFACT), human_review)
    atomic_write_json(meta_input_path, metrics_for_meta_analysis(records))
    logger.info(
        "Converted %d failed LLM context check(s) saved as rejections in %s; "
        "their statistics are now kept in %s as unchecked.",
        converted, artifact_path, meta_input_path,
    )
    return converted
