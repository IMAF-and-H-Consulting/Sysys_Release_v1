"""Stage 4: Method review & span validation.

Reads relevant.json (or claims.json) from earlier stages, classifies each
paper's study design via LLM, validates evidence spans against the abstract,
and outputs method_profiles_validated.json + method_review_manual.csv +
method_type_summary.json.
"""
import csv
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.config import Config

from core.io import atomic_write_json
from process.domain_patterns import contains_term, method_rules, paper_text
from process.llm_budget import (
    ESTIMATOR_CHARS_PER_TOKEN,
    BudgetPlanningError,
    ModelCapacity,
    TokenBudget,
    estimate_messages_tokens,
    plan_token_budget,
    resolve_model_capacity,
)
from process.llm_context import ContextWindowExceededError, OutputLimitExceededError

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/claims/relevant.json")

# Fallback confidence threshold used when research_config.yaml does not specify one.
_DEFAULT_CONFIDENCE_THRESHOLD = 0.5

LLM_METHOD = "llm"
HEURISTIC_REVIEW_REASON = "heuristic_classification"
_EXPLICIT_OUTPUT_REASON = (
    "The user selected this Stage 4 response-token override. It replaces automatic "
    "context-relative method-review output scaling for this request."
)
_CONTEXT_REDUCTION_REASON = (
    "The complete abstract did not fit after reserving the method-review JSON response and "
    "safety allowance. A context-derived opening prefix was used and its coverage is saved."
)


def _clip_utf8_prefix(text: str, byte_budget: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= max(0, byte_budget):
        return text
    return raw[:max(0, byte_budget)].decode("utf-8", errors="ignore").rstrip()


def _fit_method_prompt(
    title: str,
    abstract: str,
    extraction_prompt,
    capacity: ModelCapacity,
    max_tokens: int | None,
) -> tuple[str, dict[str, Any], TokenBudget]:
    """Use the complete abstract when possible, otherwise make reduction auditable."""
    explicit = int(max_tokens) if max_tokens and int(max_tokens) > 0 else None
    full_prompt = extraction_prompt(title, abstract)
    try:
        budget = plan_token_budget(
            "method_review", capacity,
            messages=[{"role": "user", "content": full_prompt}],
            expected_items=1,
            explicit_output_tokens=explicit,
            explicit_reason=_EXPLICIT_OUTPUT_REASON,
        )
        fitted = abstract
        prompt = full_prompt
    except BudgetPlanningError:
        empty_prompt = extraction_prompt(title, "")
        base_budget = plan_token_budget(
            "method_review", capacity,
            messages=[{"role": "user", "content": empty_prompt}],
            expected_items=1,
            explicit_output_tokens=explicit,
            explicit_reason=_EXPLICIT_OUTPUT_REASON,
        )
        available_tokens = max(
            0,
            base_budget.input_capacity_tokens
            - estimate_messages_tokens([{"role": "user", "content": empty_prompt}]),
        )
        fitted = _clip_utf8_prefix(
            abstract, int(available_tokens * ESTIMATOR_CHARS_PER_TOKEN))
        while fitted:
            prompt = extraction_prompt(title, fitted)
            try:
                budget = plan_token_budget(
                    "method_review", capacity,
                    messages=[{"role": "user", "content": prompt}],
                    expected_items=1,
                    explicit_output_tokens=explicit,
                    explicit_reason=_EXPLICIT_OUTPUT_REASON,
                )
                break
            except BudgetPlanningError:
                fitted = fitted[:max(0, int(len(fitted) * 0.95))].rstrip()
        else:
            prompt = empty_prompt
            budget = base_budget
    source_chars = len(abstract)
    submitted_chars = len(fitted)
    truncated = submitted_chars < source_chars
    coverage = {
        "source_chars": source_chars,
        "submitted_chars": submitted_chars,
        "coverage_ratio": round(submitted_chars / source_chars, 6) if source_chars else 1.0,
        "truncated": truncated,
        "policy": "context-derived-prefix" if truncated else "complete-abstract",
        "reason": _CONTEXT_REDUCTION_REASON if truncated
                  else "The complete abstract fit the model context.",
        "context_window_tokens": capacity.context_window_tokens,
    }
    return prompt, coverage, budget


# ---------------------------------------------------------------------------
# Heuristic classification (keyword-based fallback)
# ---------------------------------------------------------------------------

def _heuristic_classify(paper: dict, config=None, *, rules=None) -> dict:
    """Apply ordered, configurable study-design rules to whole terms."""
    text = paper_text(paper, "abstract") + " " + paper_text(paper, "title")
    rules = method_rules(config) if rules is None else rules
    for rule in rules:
        any_terms, all_terms = rule["any_terms"], rule["all_terms"]
        if ((not any_terms or any(contains_term(text, t) for t in any_terms))
                and all(contains_term(text, t) for t in all_terms)):
            return {"study_method_type": rule["study_method_type"],
                    "confidence": rule["confidence"]}
    return {"study_method_type": "unknown", "confidence": 0.1}


def _heuristic_parsed(paper: dict, config=None, *, rules=None) -> dict:
    """A heuristic classification in the shape of a parsed LLM reply."""
    heuristic = _heuristic_classify(paper, config, rules=rules)
    return {
        "study_method_type": heuristic["study_method_type"],
        "confidence": heuristic["confidence"],
        "sample_size": -1,
        "has_control_group": False,
        "is_longitudinal": False,
        "design_notes": "heuristic classification",
        "evidence_spans": [],
    }


def _mark_classification(profile: dict, method: str) -> dict:
    """Record how a profile was classified; keyword labels always go to review."""
    marked = {**profile, "classification_method": method}
    if method == LLM_METHOD:
        return marked
    reasons = [r for r in (profile.get("manual_review_reason") or "").split("; ") if r]
    marked["needs_manual_review"] = True
    marked["manual_review_reason"] = "; ".join([*reasons, HEURISTIC_REVIEW_REASON])
    return marked


def _method_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        method = (rec.get("method_profile") or {}).get("classification_method", "unknown")
        counts[method] = counts.get(method, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run(
    config: "Config | None" = None,
    cancel_event: Any = None,
    progress_callback: Callable[[str], None] | None = None,
    **kwargs,
) -> dict:
    """Run Stage 4: method review & span validation.

    For each paper in relevant.json:
      1. Send abstract to LLM with method classification prompt
      2. Parse response
      3. Validate evidence spans against abstract text
      4. Save results

    Returns a summary dict.
    """
    from process.llm import chat as llm_chat
    from process.llm import set_config
    from process.method_review import (
        extraction_prompt,
        parse_json_object,
        summarize_method_types,
        validate_method_profile,
    )

    rules = method_rules(config)  # Reject invalid rules before doing any work.

    if config:
        set_config(config)

    # Resolve paths
    input_path = kwargs.get("input_path") or (
        config.claims_dir / "relevant.json" if config else DEFAULT_INPUT)
    input_path = Path(input_path)

    out_dir = kwargs.get("output_dir") or (
        config.data_dir / "method_review" if config else Path("data/method_review"))
    out_dir = Path(out_dir)

    model = kwargs.get("model") or (
        config.llm_model if config else "")

    logger.info("Stage 4 — Method Review & Span Validation")
    logger.info("  Input:  %s", input_path)
    logger.info("  Output dir: %s", out_dir)
    logger.info("  Model: %s", model)

    # Load papers
    if not input_path.exists():
        msg = f"Input file not found: {input_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    with open(input_path, encoding="utf-8") as f:
        papers = json.load(f)

    total = len(papers)
    logger.info("  Loaded %d papers", total)
    if progress_callback:
        progress_callback(f"Loaded {total} papers for method review")

    if total == 0:
        logger.warning("No papers to validate — writing empty outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_dir / "method_profiles_validated.json", [])
        atomic_write_json(out_dir / "method_type_summary.json",
                    {"total_papers": 0, "flagged_for_review": 0,
                     "flagged_pct": 0.0, "type_counts": {}})
        return {"total": 0, "validated": 0,
                "output_path": str(out_dir / "method_profiles_validated.json")}

    # Determine if LLM is available
    use_llm = bool(config and getattr(config, "llm_provider", "") and model)
    if use_llm:
        logger.info("  LLM classification: provider=%s, model=%s",
                    getattr(config, "llm_provider", ""), model)
        capacity = resolve_model_capacity(
            config,
            requested_context_tokens=kwargs.get("context_tokens"),
            model=model,
        )
        max_tokens = int(kwargs.get("max_tokens", 0) or 0)
        logger.info(
            "  Token action: context=%d (%s), response=%s; complete abstracts are "
            "preferred and any reduction is saved in method_profile.llm_input_coverage",
            capacity.context_window_tokens, capacity.source,
            max_tokens if max_tokens > 0 else "auto",
        )
    else:
        logger.info("  No LLM configured — heuristic-only mode")
        capacity = None
        max_tokens = 0

    # Configured confidence threshold (loop-invariant)
    mr_cfg = (config.get_method_review_config() if config else None) or {}
    confidence_threshold = float(mr_cfg.get("confidence_threshold", _DEFAULT_CONFIDENCE_THRESHOLD))

    # Process each paper concurrently. run_sliding_window collects results
    # in input order, so validated_records is identical to a sequential run.
    # Set concurrent_workers=1 for local inference servers (Ollama / LM
    # Studio) to avoid model contention.
    from core.checkpoint import run_sliding_window

    _workers = kwargs.get("concurrent_workers")
    if _workers is None and config is not None:
        _workers = getattr(config, "concurrent_workers", None)
    _workers = max(1, int(_workers or 4))
    _stagger = max(0.0, float(kwargs.get("stagger_delay", 0.5)))
    if "stagger_delay" not in kwargs and config is not None:
        _stagger = max(0.0, float(getattr(config, "stagger_delay", _stagger)))
    _timeout = float(getattr(config, "worker_timeout_seconds", None) or 300.0)
    logger.info("Method review concurrency: workers=%d", _workers)

    def _validated(paper: dict, parsed: dict, method: str,
                   coverage: dict[str, Any] | None = None) -> dict:
        profile = validate_method_profile(paper_text(paper, "abstract"), parsed,
                                          confidence_threshold=confidence_threshold)
        if coverage:
            profile["llm_input_coverage"] = coverage
        return {**paper, "method_profile": _mark_classification(profile, method)}

    def _classify_one(paper: dict) -> dict:
        title = (paper_text(paper, "title") or "?")[:80]
        abstract = paper_text(paper, "abstract")
        logger.info("Classifying: %s", title)

        parsed = None
        coverage = None
        if use_llm and abstract:
            try:
                assert capacity is not None
                prompt, coverage, budget = _fit_method_prompt(
                    title, abstract, extraction_prompt, capacity, max_tokens)
                if coverage["truncated"]:
                    logger.warning(
                        "Stage 4 input coverage for %r: %d/%d chars (%.1f%%). %s",
                        title, coverage["submitted_chars"], coverage["source_chars"],
                        100.0 * coverage["coverage_ratio"], coverage["reason"],
                    )
                response = llm_chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=max_tokens if max_tokens > 0 else None,
                    task="method_review",
                    expected_items=1,
                    context_tokens=capacity.context_window_tokens,
                    token_budget=budget,
                    explicit_limit_reason=_EXPLICIT_OUTPUT_REASON,
                )
                parsed = parse_json_object(response)
                if not parsed:
                    logger.warning(
                        "  LLM returned unparseable response for '%s', "
                        "falling back to heuristic", title)
            except (OutputLimitExceededError, ContextWindowExceededError,
                    BudgetPlanningError) as exc:
                logger.warning(
                    "  LLM method review is unresolved for '%s' because the complete "
                    "response did not fit (%s). A heuristic label is shown only as a "
                    "manual-review aid, not as completed model validation.",
                    title, exc,
                )
                unresolved = _validated(
                    paper,
                    _heuristic_parsed(paper, config, rules=rules),
                    "unresolved_model_capacity",
                    coverage,
                )
                profile = unresolved["method_profile"]
                profile.update({
                    "model_review_status": "unresolved",
                    "model_review_error": str(exc),
                    "model_review_error_type": type(exc).__name__,
                    "heuristic_label_is_review_aid": True,
                })
                return unresolved
            except Exception as exc:
                logger.warning(
                    "  LLM call failed for '%s' (%s), "
                    "falling back to heuristic", title, exc)

        if parsed:
            return _validated(paper, parsed, LLM_METHOD, coverage)
        method = "heuristic_fallback" if use_llm else "heuristic_only"
        return _validated(
            paper, _heuristic_parsed(paper, config, rules=rules), method, coverage)

    def _build_result(idx: int, record: dict | None) -> dict:
        if record is None:
            logger.error("Worker timed out or failed for paper %d/%d, using heuristic fallback", idx + 1, total)
            record = _validated(papers[idx], _heuristic_parsed(papers[idx], config, rules=rules),
                                "heuristic_fallback_timeout")
        title = (paper_text(record, "title") or "?")[:80]
        mp = record.get("method_profile", {})
        result_msg = (f"[{idx + 1}/{total}] {title} → "
                      f"{mp.get('study_method_type')} "
                      f"(conf={mp.get('confidence', 0.0):.2f}) "
                      f"[{mp.get('classification_method')}]")
        logger.info(result_msg)
        if progress_callback:
            progress_callback(result_msg)
        return record

    validated_records, was_cancelled = run_sliding_window(
        items=papers,
        worker_fn=_classify_one,
        result_fn=_build_result,
        workers=_workers,
        stagger=_stagger,
        cancel_event=cancel_event,
        future_timeout=_timeout,
    )
    if was_cancelled:
        return {"total": total, "validated": len(validated_records),
                "cancelled": True,
                "output_path": str(out_dir / "method_profiles_validated.json")}

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. method_profiles_validated.json
    validated_path = out_dir / "method_profiles_validated.json"
    atomic_write_json(validated_path, validated_records)
    logger.info("Saved %d validated profiles to %s",
                len(validated_records), validated_path)

    # 2. method_review_manual.csv — papers flagged for manual review
    manual_path = out_dir / "method_review_manual.csv"
    flagged = [r for r in validated_records
               if r.get("method_profile", {}).get("needs_manual_review", False)]
    _write_manual_csv(manual_path, flagged)
    logger.info("Flagged %d papers for manual review → %s",
                len(flagged), manual_path)

    # 3. method_type_summary.json
    summary_data = summarize_method_types(validated_records)
    summary_path = out_dir / "method_type_summary.json"
    atomic_write_json(summary_path, summary_data)
    logger.info("Method type summary → %s", summary_path)

    methods = _method_counts(validated_records)
    n_heuristic = len(validated_records) - methods.get(LLM_METHOD, 0)
    n_unresolved = methods.get("unresolved_model_capacity", 0)
    summary = {
        "total": total,
        "validated": len(validated_records),
        "flagged": len(flagged),
        "classification_methods": methods,
        "heuristic_fallback": n_heuristic if use_llm else 0,
        "unresolved_model_reviews": n_unresolved,
        "status": "partial" if n_unresolved else "complete",
        "output_path": str(validated_path),
        "summary_path": str(summary_path),
        "manual_csv_path": str(manual_path),
    }
    logger.info("Stage 4 summary: %s", summary)
    if progress_callback:
        progress_callback(
            f"Method review complete: {len(validated_records)} papers, "
            f"{len(flagged)} flagged for manual review")
    if summary["heuristic_fallback"]:
        msg = (f"WARNING: LLM classification failed for {n_heuristic} paper(s); "
               "their keyword-based design labels are flagged for manual review")
        logger.warning(msg)
        if progress_callback:
            progress_callback(msg)
    if n_unresolved:
        msg = (
            f"PARTIAL: {n_unresolved} method review(s) hit a context/output boundary. "
            "Their heuristic labels are manual-review aids and were not counted as "
            "completed model validation."
        )
        logger.warning(msg)
        if progress_callback:
            progress_callback(msg)

    return summary


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _write_manual_csv(path: Path, records: list[dict]) -> None:
    """Write flagged papers to CSV for manual review."""
    fieldnames = [
        "paper_id", "title", "study_method_type", "confidence",
        "classification_method", "manual_review_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            mp = rec.get("method_profile", {})
            writer.writerow({
                "paper_id": rec.get("paper_id", rec.get("id", "")),
                "title": rec.get("title", ""),
                "study_method_type": mp.get("study_method_type", ""),
                "confidence": mp.get("confidence", 0.0),
                "classification_method": mp.get("classification_method", ""),
                "manual_review_reason": mp.get("manual_review_reason", ""),
            })
