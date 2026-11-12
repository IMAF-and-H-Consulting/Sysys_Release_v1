"""Context fitting and coverage accounting for Stage 2 relevance prompts."""
from __future__ import annotations

import logging
from typing import Any

from process.llm_budget import (
    COMPATIBILITY_FALLBACK_CONTEXT_TOKENS,
    ESTIMATOR_CHARS_PER_TOKEN,
    BudgetPlanningError,
    ModelCapacity,
    TokenBudget,
    estimate_messages_tokens,
    plan_token_budget,
    resolve_model_capacity,
)

logger = logging.getLogger(__name__)

_AUTO_LIMIT_REASON = (
    "The input was fitted to the loaded model context after reserving the task-scaled "
    "JSON response and safety allowance. The opening text is retained because relevance "
    "screening depends most heavily on the title, abstract, introduction, and methods."
)
_EXPLICIT_OUTPUT_REASON = (
    "The user selected this Stage 2 response-token override. It replaces automatic "
    "context-relative output scaling for this request."
)


def _paper_text_for_scoring(
    paper: dict,
    *,
    abstract_chars: int | None = None,
    fulltext_chars: int | None = None,
) -> tuple[str, str]:
    """Complete candidate text, except for an explicit positive caller limit."""
    from process.llm import compress_prompt_text

    full_text = (paper.get("full_text") or "").strip()
    use_full_text = fulltext_chars is None or int(fulltext_chars) > 0
    if full_text and use_full_text:
        text = compress_prompt_text(full_text)
        if fulltext_chars is not None and int(fulltext_chars) > 0:
            text = text[:int(fulltext_chars)]
        return text, "full_text"
    abstract = compress_prompt_text((paper.get("abstract") or "").strip())
    if abstract_chars is not None and int(abstract_chars) > 0:
        abstract = abstract[:int(abstract_chars)]
    return abstract, "abstract"


def _capacity(context_tokens: int | None) -> ModelCapacity:
    value = int(context_tokens or 0)
    from process import llm_clients

    if llm_clients._config is not None:
        # Preserve the independently configured provider/model completion
        # ceiling even when Stage 2 already resolved the context length.
        return resolve_model_capacity(
            llm_clients._config,
            requested_context_tokens=value if value > 0 else None,
        )
    if value > 0:
        return ModelCapacity(value, source="Stage 2 resolved model context")
    logger.warning(
        "Stage 2 token action: no active LLM config or explicit context; using the "
        "unconfirmed %d-token compatibility fallback for this legacy call",
        COMPATIBILITY_FALLBACK_CONTEXT_TOKENS,
    )
    return ModelCapacity(
        COMPATIBILITY_FALLBACK_CONTEXT_TOKENS,
        source="unconfirmed compatibility fallback",
        confirmed=False,
        detail="No Stage 2 context value was supplied by this legacy caller.",
    )


def _clip_utf8_prefix(text: str, byte_budget: int) -> str:
    if byte_budget <= 0:
        return ""
    raw = text.encode("utf-8")
    if len(raw) <= byte_budget:
        return text
    return raw[:byte_budget].decode("utf-8", errors="ignore").rstrip()


def _single_prompt(paper: dict, paper_text: str, text_source: str, hypothesis: str) -> str:
    from process.relevance_filter import RELEVANCE_PROMPT, _require_question

    return RELEVANCE_PROMPT.format(
        hypothesis=_require_question(hypothesis), title=paper.get("title", ""),
        text_source=text_source, paper_text=paper_text,
    )


def _fit_single_relevance_input(
    paper: dict,
    candidate_text: str,
    text_source: str,
    *,
    hypothesis: str,
    context_tokens: int | None,
    max_tokens: int | None,
    task: str = "relevance_single",
) -> tuple[str, dict[str, Any], TokenBudget]:
    """Fit one complete candidate text to context and expose every reduction."""
    from process.llm import compress_prompt_text

    capacity = _capacity(context_tokens)
    raw_value = (paper.get("full_text") if text_source == "full_text"
                 else paper.get("abstract"))
    raw_source = str(raw_value or "").strip()
    compressed_source = compress_prompt_text(raw_source)
    explicit = int(max_tokens) if max_tokens and int(max_tokens) > 0 else None
    full_prompt = _single_prompt(paper, candidate_text, text_source, hypothesis)
    try:
        budget = plan_token_budget(
            task, capacity, messages=[{"role": "user", "content": full_prompt}],
            expected_items=1, explicit_output_tokens=explicit,
            explicit_reason=_EXPLICIT_OUTPUT_REASON)
        fitted = candidate_text
    except BudgetPlanningError:
        empty_prompt = _single_prompt(paper, "", text_source, hypothesis)
        base_budget = plan_token_budget(
            task, capacity, messages=[{"role": "user", "content": empty_prompt}],
            expected_items=1, explicit_output_tokens=explicit,
            explicit_reason=_EXPLICIT_OUTPUT_REASON)
        available_tokens = max(
            0, base_budget.input_capacity_tokens
            - estimate_messages_tokens([{"role": "user", "content": empty_prompt}]))
        fitted = _clip_utf8_prefix(
            candidate_text, int(available_tokens * ESTIMATOR_CHARS_PER_TOKEN))
        while fitted:
            try:
                budget = plan_token_budget(
                    task, capacity,
                    messages=[{"role": "user", "content": _single_prompt(
                        paper, fitted, text_source, hypothesis)}],
                    expected_items=1, explicit_output_tokens=explicit,
                    explicit_reason=_EXPLICIT_OUTPUT_REASON)
                break
            except BudgetPlanningError:
                fitted = fitted[:max(0, int(len(fitted) * 0.95))].rstrip()
        else:
            budget = base_budget

    source_chars, compressed_chars = len(raw_source), len(compressed_source)
    selected_chars, submitted_chars = len(candidate_text), len(fitted)
    compression_applied = compressed_source != raw_source
    application_limit_applied = selected_chars < compressed_chars
    context_truncated = submitted_chars < selected_chars
    truncated = compression_applied or application_limit_applied or context_truncated
    reasons = []
    if compression_applied:
        reasons.append(
            "Prompt compression removed trailing references/boilerplate or redundant PDF whitespace.")
    if application_limit_applied:
        reasons.append("A positive caller-selected source character limit was applied.")
    if context_truncated:
        reasons.append(_AUTO_LIMIT_REASON)
    if not reasons:
        reasons.append("The complete selected text fit the model context.")
    coverage = {
        "source": text_source, "source_chars": source_chars,
        "compressed_chars": compressed_chars,
        "selected_chars_before_context_fit": selected_chars,
        "submitted_chars": submitted_chars,
        "coverage_ratio": round(submitted_chars / source_chars, 6) if source_chars else 1.0,
        "truncated": truncated, "compression_applied": compression_applied,
        "application_limit_applied": application_limit_applied,
        "context_truncated": context_truncated,
        "policy": "context-derived-prefix" if context_truncated else (
            "compressed-or-explicitly-limited-input" if truncated else "complete-input"),
        "reason": " ".join(reasons),
        "context_window_tokens": capacity.context_window_tokens,
    }
    if truncated:
        logger.warning(
            "Stage 2 input coverage for %r: %d/%d chars (%.1f%%). %s",
            str(paper.get("title") or "")[:80], submitted_chars, source_chars,
            100.0 * coverage["coverage_ratio"], coverage["reason"])
    return fitted, coverage, budget


def _score_batch_payload(
    papers: list[dict], *, abstract_chars: int | None = None,
    fulltext_chars: int | None = None,
) -> list[dict]:
    from process.llm import compress_prompt_text

    payload = []
    for idx, paper in enumerate(papers, start=1):
        paper_text, text_source = _paper_text_for_scoring(
            paper, abstract_chars=abstract_chars, fulltext_chars=fulltext_chars)
        raw_value = (paper.get("full_text") if text_source == "full_text"
                     else paper.get("abstract"))
        raw_text = str(raw_value or "").strip()
        compressed_text = compress_prompt_text(raw_text)
        payload.append({
            "id": f"paper_{idx}", "title": paper.get("title", ""),
            "paper_text": paper_text, "text_source": text_source,
            "source_chars": len(raw_text), "compressed_chars": len(compressed_text),
            "application_char_limit": (
                int(fulltext_chars) if text_source == "full_text" and fulltext_chars
                else int(abstract_chars) if text_source == "abstract" and abstract_chars
                else None),
        })
    return payload


def _batch_prompt(payload: list[dict], hypothesis: str) -> str:
    from process.relevance_filter import BATCH_RELEVANCE_PROMPT, _require_question

    blocks = ["\n".join([
        f"ID: {item['id']}", f"TITLE: {item['title']}",
        f"TEXT SOURCE: {item['text_source']}", f"PAPER TEXT: {item['paper_text']}",
        "END PAPER",
    ]) for item in payload]
    return BATCH_RELEVANCE_PROMPT.format(
        hypothesis=_require_question(hypothesis), paper_blocks="\n\n".join(blocks))


def _fair_prefixes(texts: list[str], total_bytes: int) -> list[str]:
    if not texts:
        return []
    remaining, pending = max(0, total_bytes), list(range(len(texts)))
    allocations = [0] * len(texts)
    sizes = [len(text.encode("utf-8")) for text in texts]
    while pending and remaining > 0:
        share = max(1, remaining // len(pending))
        next_pending, spent = [], 0
        for idx in pending:
            grant = min(sizes[idx] - allocations[idx], share)
            allocations[idx] += grant
            spent += grant
            if allocations[idx] < sizes[idx]:
                next_pending.append(idx)
        if spent <= 0:
            break
        remaining -= spent
        pending = next_pending
    return [_clip_utf8_prefix(text, allowance)
            for text, allowance in zip(texts, allocations)]


def _fit_batch_payload(
    payload: list[dict], *, hypothesis: str, context_tokens: int | None,
    max_tokens: int | None,
) -> tuple[list[dict], str, TokenBudget]:
    capacity = _capacity(context_tokens)
    explicit = int(max_tokens) if max_tokens and int(max_tokens) > 0 else None
    full_prompt = _batch_prompt(payload, hypothesis)
    try:
        budget = plan_token_budget(
            "relevance_batch", capacity,
            messages=[{"role": "user", "content": full_prompt}],
            expected_items=len(payload), explicit_output_tokens=explicit,
            explicit_reason=_EXPLICIT_OUTPUT_REASON)
        fitted_texts = [item["paper_text"] for item in payload]
    except BudgetPlanningError:
        empty_payload = [{**item, "paper_text": ""} for item in payload]
        empty_prompt = _batch_prompt(empty_payload, hypothesis)
        base_budget = plan_token_budget(
            "relevance_batch", capacity,
            messages=[{"role": "user", "content": empty_prompt}],
            expected_items=len(payload), explicit_output_tokens=explicit,
            explicit_reason=_EXPLICIT_OUTPUT_REASON)
        available_tokens = max(
            0, base_budget.input_capacity_tokens
            - estimate_messages_tokens([{"role": "user", "content": empty_prompt}]))
        fitted_texts = _fair_prefixes(
            [item["paper_text"] for item in payload],
            int(available_tokens * ESTIMATOR_CHARS_PER_TOKEN))
        while True:
            fitted_payload = [{**item, "paper_text": text}
                              for item, text in zip(payload, fitted_texts)]
            prompt = _batch_prompt(fitted_payload, hypothesis)
            try:
                budget = plan_token_budget(
                    "relevance_batch", capacity,
                    messages=[{"role": "user", "content": prompt}],
                    expected_items=len(payload), explicit_output_tokens=explicit,
                    explicit_reason=_EXPLICIT_OUTPUT_REASON)
                break
            except BudgetPlanningError:
                if not any(fitted_texts):
                    raise
                fitted_texts = [text[:max(0, int(len(text) * 0.95))].rstrip()
                                for text in fitted_texts]

    fitted_payload = []
    for item, fitted in zip(payload, fitted_texts):
        source_chars = int(item.get("source_chars", len(item["paper_text"])))
        compressed_chars = int(item.get("compressed_chars", len(item["paper_text"])))
        selected_chars, submitted_chars = len(item["paper_text"]), len(fitted)
        compression_applied = compressed_chars < source_chars
        context_truncated = submitted_chars < selected_chars
        explicit_char_limit = item.get("application_char_limit")
        reasons = []
        if explicit_char_limit:
            reasons.append(
                f"A user-configured {explicit_char_limit:,}-character source limit was applied first.")
        if compression_applied:
            reasons.append(
                "Prompt compression removed trailing references/boilerplate or redundant PDF whitespace.")
        if context_truncated:
            reasons.append(_AUTO_LIMIT_REASON)
        if not reasons:
            reasons.append("The complete selected text fit the model context.")
        fitted_payload.append({
            **item, "paper_text": fitted,
            "input_coverage": {
                "source": item["text_source"], "source_chars": source_chars,
                "compressed_chars": compressed_chars,
                "selected_chars_before_context_fit": selected_chars,
                "submitted_chars": submitted_chars,
                "coverage_ratio": round(submitted_chars / source_chars, 6)
                    if source_chars else 1.0,
                "truncated": bool(compression_applied or context_truncated or explicit_char_limit),
                "compression_applied": compression_applied,
                "context_truncated": context_truncated,
                "policy": "context-derived-fair-prefix" if context_truncated else (
                    "compressed-or-explicitly-limited-input"
                    if compression_applied or explicit_char_limit else "complete-input"),
                "reason": " ".join(reasons),
                "context_window_tokens": capacity.context_window_tokens,
                "application_char_limit": explicit_char_limit,
            },
        })
    return fitted_payload, _batch_prompt(fitted_payload, hypothesis), budget


def _build_relevance_batches(
    papers: list[dict], start_index: int, batch_size: int,
    max_chars: int | None = None, *, abstract_chars: int | None = None,
    fulltext_chars: int | None = None, context_tokens: int | None = None,
    hypothesis: str = "", max_tokens: int | None = None,
) -> list[list[int]]:
    """Pack indices using the task profile and resolved context, not a char cap."""
    from process.relevance_filter import _require_question

    batches, current, current_chars = [], [], 0
    capacity = _capacity(context_tokens)
    explicit = int(max_tokens) if max_tokens and int(max_tokens) > 0 else None
    for idx in range(start_index, len(papers)):
        paper_text, _ = _paper_text_for_scoring(
            papers[idx], abstract_chars=abstract_chars, fulltext_chars=fulltext_chars)
        paper_chars = len(paper_text)
        would_overflow = bool(
            current and max_chars and max_chars > 0
            and current_chars + paper_chars > max_chars)
        would_hit_limit = bool(current and len(current) >= batch_size)
        would_exceed_profile = False
        if current and not would_overflow and not would_hit_limit:
            prospective = [*current, idx]
            empty_payload = [
                {"id": f"paper_{pos + 1}",
                 "title": papers[paper_idx].get("title", ""),
                 "paper_text": "", "text_source": "abstract"}
                for pos, paper_idx in enumerate(prospective)]
            try:
                plan_token_budget(
                    "relevance_batch", capacity,
                    messages=[{"role": "user", "content": _batch_prompt(
                        empty_payload, _require_question(hypothesis))}],
                    expected_items=len(prospective), explicit_output_tokens=explicit,
                    explicit_reason=_EXPLICIT_OUTPUT_REASON)
            except BudgetPlanningError:
                would_exceed_profile = True
        if would_overflow or would_hit_limit or would_exceed_profile:
            batches.append(current)
            current, current_chars = [], 0
        current.append(idx)
        current_chars += paper_chars
    if current:
        batches.append(current)
    return batches
