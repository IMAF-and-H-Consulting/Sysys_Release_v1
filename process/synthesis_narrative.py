"""Context-scaled, resumable Stage 6 cluster-narrative generation.

This module owns the model-facing cluster synthesis path.  It deliberately
keeps context fitting separate from clustering so Stage 6 can checkpoint each
complete context batch and resume it without repeating successful model work.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from itertools import zip_longest
from typing import Any, Callable

from core.claims import claim_list
from process.claim_extractor import _claim_think_mode as _synthesis_think_mode
from process.claim_extractor import _research_question_from_config
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
from process.synthesis_papers import (
    _paper_direction,
    _paper_id,
    _paper_rank_key,
    _paper_title,
    _source_identity_key,
)

logger = logging.getLogger(__name__)

_EXPLICIT_NARRATIVE_OUTPUT_REASON = (
    "The user selected this Stage 6 cluster-response override. It replaces automatic "
    "context-relative narrative output scaling for this request."
)
_NARRATIVE_SELECTION_REASON = (
    "The complete cluster evidence did not fit after reserving the context-relative 2-3 "
    "paragraph response and safety allowance. Papers were relevance-ranked and interleaved "
    "across evidence directions, then the largest context-feasible breadth was retained; "
    "remaining evidence was fairly allocated across the selected papers."
)

NARRATIVE_PROMPT = """You are synthesizing research findings from a cluster of {paper_count} academic papers for a systematic evidence review.

Research question:
{research_question}

Most frequent mechanisms: {mechanisms}
Most frequent populations, constructs, or contextual factors: {conditions}
Most frequent entities or systems: {entities}
Papers directly informing the research question: {primary_support_pct}%

Paper summaries:
{paper_summaries}

Write a cohesive 2-3 paragraph narrative synthesizing these findings. Focus on:
1. What the papers collectively show about the configured research question and described mechanisms
2. Areas of agreement and disagreement
3. Strength of the evidence

Grounding rules:
- Use only the paper summaries and terms above; do not add studies, results, or numbers from elsewhere.
- Support each statement with the bracketed citation key of the paper it comes from, e.g. [smith2020].
- The summaries are abstracts and extracted claims, not appraised full texts; describe the strength of evidence cautiously.

Write in academic style suitable for a review paper."""

NARRATIVE_COMBINE_PROMPT = """You are combining grounded partial syntheses for one cluster in a systematic evidence review.

Research question:
{research_question}

The partial syntheses below collectively cover every paper selected for this cluster. Each partial was generated from a disjoint, context-fit batch; none may be ignored.

{partials}

Write one cohesive 2-3 paragraph cluster narrative. Preserve the bracketed citation keys and retain disagreements, neutral evidence, and uncertainty represented in any partial. Use only the supplied partial syntheses; do not introduce studies, results, or numbers."""

CheckpointCallback = Callable[[dict[str, Any]], None]


def _local_citation_keys(papers: list[dict]) -> dict[str, str]:
    """Citation keys for a standalone cluster (Stage 6 passes the global map)."""
    from process.export import assign_citation_keys

    return {_source_identity_key(p): key for key, p in assign_citation_keys(papers)}


def _narrative_sample(papers: list[dict], limit: int | None) -> list[dict]:
    """Top-ranked papers, interleaved across evidence directions."""
    by_direction: dict[str, list[dict]] = defaultdict(list)
    for paper in sorted(papers, key=_paper_rank_key, reverse=True):
        by_direction[_paper_direction(paper)].append(paper)
    interleaved = [p for row in zip_longest(*by_direction.values()) for p in row if p]
    return interleaved if limit is None else interleaved[:limit]


def _summary_parts(paper: dict, citation_key: str) -> tuple[str, str]:
    """Stable citation/title prefix and complete evidence text for one paper."""
    key = f"[{citation_key}] " if citation_key else ""
    year = paper.get("year") or "n.d."
    prefix = f"- {key}{_paper_title(paper)}: ({year})"
    abstract = " ".join(str(paper.get("abstract") or "").split())
    findings = "; ".join(claim_list(paper, "findings"))
    evidence = f" Abstract: {abstract} Findings: {findings}".rstrip()
    return prefix, evidence


def _clip_utf8_prefix(text: str, byte_budget: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= max(0, byte_budget):
        return text
    return raw[:max(0, byte_budget)].decode("utf-8", errors="ignore").rstrip()


def _fair_clip_texts(texts: list[str], total_bytes: int) -> list[str]:
    """Water-fill a shared input budget so every selected source contributes."""
    if not texts:
        return []
    sizes = [len(text.encode("utf-8")) for text in texts]
    allocations = [0] * len(texts)
    pending = list(range(len(texts)))
    remaining = max(0, total_bytes)
    while pending and remaining > 0:
        share = max(1, remaining // len(pending))
        spent = 0
        next_pending: list[int] = []
        for idx in pending:
            grant = min(share, sizes[idx] - allocations[idx])
            allocations[idx] += grant
            spent += grant
            if allocations[idx] < sizes[idx]:
                next_pending.append(idx)
        if spent <= 0:
            break
        remaining -= spent
        pending = next_pending
    return [_clip_utf8_prefix(text, allowance) for text, allowance in zip(texts, allocations)]


def _summary_line(paper: dict, citation_key: str, evidence: str | None = None) -> str:
    prefix, complete = _summary_parts(paper, citation_key)
    return prefix + (complete if evidence is None else evidence)


def _terms_text(cluster: dict, field: str) -> str:
    return ", ".join(claim_list(cluster, field)) or "not specified"


def _synthesis_capacity(capacity: ModelCapacity | int | None) -> ModelCapacity:
    if isinstance(capacity, ModelCapacity):
        return capacity
    value = int(capacity or 0)
    from process import llm_clients

    if llm_clients._config is not None:
        return resolve_model_capacity(
            llm_clients._config,
            requested_context_tokens=value if value > 0 else None,
        )
    if value <= 0:
        logger.warning(
            "Stage 6 token action: no active LLM config or explicit context; using the "
            "unconfirmed %d-token compatibility fallback for this legacy call",
            COMPATIBILITY_FALLBACK_CONTEXT_TOKENS,
        )
    return ModelCapacity(
        value if value > 0 else COMPATIBILITY_FALLBACK_CONTEXT_TOKENS,
        source=("Stage 6 resolved model context" if value > 0
                else "unconfirmed compatibility fallback"),
        confirmed=value > 0,
    )


def _narrative_prompt(
    cluster: dict,
    research_config: dict | None,
    paper_count: int,
    primary_count: int,
    terms: dict[str, str],
    lines: list[str],
    selected_count: int,
) -> str:
    header = ([
        f"(showing {selected_count} of {paper_count} papers, ranked by relevance and "
        "taken in turn from each evidence direction; see input_coverage for the context policy)"
    ] if selected_count < paper_count else [])
    summaries = "\n".join([*header, *lines]) if lines else "No summaries available."
    return NARRATIVE_PROMPT.format(
        research_question=_research_question_from_config(research_config),
        paper_count=paper_count,
        mechanisms=terms["mechanisms"],
        conditions=terms["conditions"],
        entities=terms["key_entities"],
        primary_support_pct=round(100 * primary_count / paper_count) if paper_count else 0,
        paper_summaries=summaries,
    )


def _plan_narrative(
    capacity: ModelCapacity,
    prompt: str,
    expected_items: int,
    max_tokens: int | None,
) -> TokenBudget:
    return plan_token_budget(
        "synthesis_cluster", capacity,
        messages=[{"role": "user", "content": prompt}],
        expected_items=max(1, expected_items),
        explicit_output_tokens=(int(max_tokens) if max_tokens and int(max_tokens) > 0 else None),
        explicit_reason=_EXPLICIT_NARRATIVE_OUTPUT_REASON,
    )


def _fit_narrative_prompt(
    cluster: dict,
    sample: list[dict],
    keys: dict[str, str],
    research_config: dict | None,
    capacity: ModelCapacity,
    max_tokens: int | None,
) -> tuple[list[dict], str, dict[str, Any], TokenBudget]:
    """Prefer complete cluster evidence, otherwise fit breadth before detail."""
    paper_count = int(cluster.get("paper_count", len(cluster.get("papers", []) or [])))
    primary_count = int(cluster.get("primary_support_count", 0) or 0)
    complete_terms = {
        field: _terms_text(cluster, field)
        for field in ("mechanisms", "conditions", "key_entities")
    }

    def complete_lines(selected: list[dict]) -> list[str]:
        return [_summary_line(p, keys.get(_source_identity_key(p), "")) for p in selected]

    full_prompt = _narrative_prompt(
        cluster, research_config, paper_count, primary_count,
        complete_terms, complete_lines(sample), len(sample))
    try:
        budget = _plan_narrative(capacity, full_prompt, len(sample), max_tokens)
        total_chars = sum(len(_summary_parts(p, "")[1]) for p in sample)
        coverage = {
            "context_window_tokens": capacity.context_window_tokens,
            "papers_available": len(cluster.get("papers", []) or []),
            "papers_selected": len(sample),
            "papers_omitted": max(len(cluster.get("papers", []) or []) - len(sample), 0),
            "evidence_source_chars": total_chars,
            "evidence_submitted_chars": total_chars,
            "evidence_coverage_ratio": 1.0,
            "terms_source_chars": sum(len(v) for v in complete_terms.values()),
            "terms_submitted_chars": sum(len(v) for v in complete_terms.values()),
            "truncated": len(sample) < len(cluster.get("papers", []) or []),
            "policy": "complete-selected-input",
            "reason": ("The complete cluster evidence fit the model context."
                       if len(sample) == len(cluster.get("papers", []) or [])
                       else "The caller explicitly limited papers; omissions are listed."),
        }
        return sample, full_prompt, coverage, budget
    except BudgetPlanningError:
        pass

    low, high, feasible = 1, len(sample), 0
    empty_terms = {field: "not specified" for field in complete_terms}
    while low <= high:
        mid = (low + high) // 2
        selected = sample[:mid]
        prefix_lines = [_summary_parts(p, keys.get(_source_identity_key(p), ""))[0]
                        for p in selected]
        prompt = _narrative_prompt(
            cluster, research_config, paper_count, primary_count,
            empty_terms, prefix_lines, len(selected))
        try:
            _plan_narrative(capacity, prompt, len(selected), max_tokens)
        except BudgetPlanningError:
            high = mid - 1
        else:
            feasible = mid
            low = mid + 1
    if feasible < 1:
        selected = sample[:1]
        impossible = _narrative_prompt(
            cluster, research_config, paper_count, primary_count, empty_terms,
            [_summary_parts(selected[0], keys.get(_source_identity_key(selected[0]), ""))[0]],
            1)
        _plan_narrative(capacity, impossible, 1, max_tokens)
        raise AssertionError("unreachable")

    selected = sample[:feasible]
    prefixes = [_summary_parts(p, keys.get(_source_identity_key(p), ""))[0]
                for p in selected]
    evidence = [_summary_parts(p, keys.get(_source_identity_key(p), ""))[1]
                for p in selected]
    term_values = [complete_terms[field] for field in complete_terms]
    base_prompt = _narrative_prompt(
        cluster, research_config, paper_count, primary_count, empty_terms,
        prefixes, len(selected))
    base_budget = _plan_narrative(capacity, base_prompt, len(selected), max_tokens)
    available_tokens = max(
        0,
        base_budget.input_capacity_tokens
        - estimate_messages_tokens([{"role": "user", "content": base_prompt}]),
    )
    available_bytes = int(available_tokens * ESTIMATOR_CHARS_PER_TOKEN)
    all_variable = [*term_values, *evidence]
    fitted = _fair_clip_texts(all_variable, available_bytes)
    fitted_terms_values = fitted[:len(term_values)]
    fitted_evidence = fitted[len(term_values):]

    while True:
        fitted_terms = {
            field: value or "not specified"
            for field, value in zip(complete_terms, fitted_terms_values)
        }
        lines = [prefix + value for prefix, value in zip(prefixes, fitted_evidence)]
        prompt = _narrative_prompt(
            cluster, research_config, paper_count, primary_count,
            fitted_terms, lines, len(selected))
        try:
            budget = _plan_narrative(capacity, prompt, len(selected), max_tokens)
            break
        except BudgetPlanningError:
            if not any(fitted_terms_values) and not any(fitted_evidence):
                raise
            fitted_terms_values = [v[:max(0, int(len(v) * 0.95))].rstrip()
                                   for v in fitted_terms_values]
            fitted_evidence = [v[:max(0, int(len(v) * 0.95))].rstrip()
                               for v in fitted_evidence]

    source_evidence_chars = sum(len(value) for value in evidence)
    submitted_evidence_chars = sum(len(value) for value in fitted_evidence)
    source_term_chars = sum(len(value) for value in term_values)
    submitted_term_chars = sum(len(value) for value in fitted_terms_values)
    available_papers = len(cluster.get("papers", []) or [])
    coverage = {
        "context_window_tokens": capacity.context_window_tokens,
        "papers_available": available_papers,
        "papers_selected": len(selected),
        "papers_omitted": max(available_papers - len(selected), 0),
        "evidence_source_chars": source_evidence_chars,
        "evidence_submitted_chars": submitted_evidence_chars,
        "evidence_coverage_ratio": round(
            submitted_evidence_chars / source_evidence_chars, 6)
            if source_evidence_chars else 1.0,
        "terms_source_chars": source_term_chars,
        "terms_submitted_chars": submitted_term_chars,
        "truncated": True,
        "policy": "context-derived-direction-balanced-selection-and-fair-allocation",
        "reason": _NARRATIVE_SELECTION_REASON,
    }
    logger.warning(
        "Stage 6 cluster %s input coverage: %d/%d papers; %d/%d evidence chars. %s",
        cluster.get("cluster_id", "?"), len(selected), available_papers,
        submitted_evidence_chars, source_evidence_chars, coverage["reason"],
    )
    return selected, prompt, coverage, budget


def _checkpoint(
    callback: CheckpointCallback | None,
    *,
    cluster_id: Any,
    phase: str,
    partials: list[dict[str, Any]],
    batch_records: list[dict[str, Any]],
    combine_records: list[dict[str, Any]] | None = None,
) -> None:
    if callback is None:
        return
    callback({
        "cluster_id": cluster_id,
        "phase": phase,
        "partials": list(partials),
        "batch_requests": list(batch_records),
        "combine_requests": list(combine_records or []),
    })


def _combine_partial_narratives(
    model: str,
    partials: list[dict[str, Any]],
    research_config: dict | None,
    capacity: ModelCapacity,
    max_tokens: int | None,
    use_thinking: bool,
    cancel_event: Any,
    *,
    cluster_id: Any,
    batch_records: list[dict[str, Any]],
    checkpoint_callback: CheckpointCallback | None,
) -> tuple[str, list[dict[str, Any]], bool]:
    """Hierarchically combine disjoint batch syntheses without dropping one."""
    from process.llm import chat as llm_chat

    explicit = int(max_tokens) if max_tokens and int(max_tokens) > 0 else None
    records: list[dict[str, Any]] = []
    current = list(partials)
    layer = 0

    def prompt_for(group: list[dict[str, Any]]) -> str:
        blocks = "\n\n".join(
            f"PARTIAL {i + 1} (paper keys: {', '.join(part['paper_ids'])}):\n{part['text']}"
            for i, part in enumerate(group)
        )
        return NARRATIVE_COMBINE_PROMPT.format(
            research_question=_research_question_from_config(research_config),
            partials=blocks,
        )

    def plan(group: list[dict[str, Any]]) -> tuple[str, TokenBudget]:
        prompt = prompt_for(group)
        budget = plan_token_budget(
            "synthesis_cluster", capacity,
            messages=[{"role": "user", "content": prompt}],
            expected_items=max(1, len(group)),
            explicit_output_tokens=explicit,
            explicit_reason=_EXPLICIT_NARRATIVE_OUTPUT_REASON,
        )
        return prompt, budget

    while len(current) > 1:
        groups: list[list[dict[str, Any]]] = []
        active: list[dict[str, Any]] = []
        for partial in current:
            candidate = [*active, partial]
            try:
                plan(candidate)
            except BudgetPlanningError:
                if active:
                    groups.append(active)
                    active = [partial]
                    plan(active)
                else:
                    raise
            else:
                active = candidate
        if active:
            groups.append(active)

        if all(len(group) == 1 for group in groups):
            sections = "\n\n".join(
                f"### Context-fit synthesis section {i + 1}\n{part['text']}"
                for i, part in enumerate(current)
            )
            records.append({
                "layer": layer,
                "action": "transparent-multisection-fallback",
                "reason": (
                    "No pair of retained partial syntheses fit a combine request; every "
                    "partial is preserved as a labeled section instead of being dropped."
                ),
                "paper_ids": [paper_id for part in current for paper_id in part["paper_ids"]],
            })
            _checkpoint(
                checkpoint_callback, cluster_id=cluster_id, phase="combine_fallback",
                partials=partials, batch_records=batch_records, combine_records=records)
            return sections, records, True

        next_level: list[dict[str, Any]] = []
        for group_index, group in enumerate(groups):
            if len(group) == 1:
                next_level.append(group[0])
                continue
            prompt, budget = plan(group)
            combined = llm_chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens if max_tokens and max_tokens > 0 else None,
                think=_synthesis_think_mode(model, enabled=use_thinking),
                cancel_event=cancel_event,
                task="synthesis_cluster",
                expected_items=len(group),
                context_tokens=capacity.context_window_tokens,
                token_budget=budget,
                explicit_limit_reason=_EXPLICIT_NARRATIVE_OUTPUT_REASON,
            )
            paper_ids = [paper_id for part in group for paper_id in part["paper_ids"]]
            next_level.append({"text": combined, "paper_ids": paper_ids})
            records.append({
                "layer": layer,
                "group": group_index,
                "partial_count": len(group),
                "paper_ids": paper_ids,
                "token_budget": budget.as_dict(),
            })
            _checkpoint(
                checkpoint_callback, cluster_id=cluster_id, phase="combine_complete",
                partials=partials, batch_records=batch_records, combine_records=records)
        current = next_level
        layer += 1
    return current[0]["text"], records, False


def generate_cluster_narrative(
    model: str,
    cluster: dict,
    *,
    research_config: dict | None = None,
    use_thinking: bool = False,
    max_tokens: int | None = None,
    citation_keys: dict[str, str] | None = None,
    max_papers: int | None = None,
    model_capacity: ModelCapacity | int | None = None,
    cancel_event: Any = None,
    checkpoint_callback: CheckpointCallback | None = None,
    resume_state: dict[str, Any] | None = None,
) -> dict:
    """Generate one cluster narrative with context batching and durable resume hooks."""
    from process.llm import chat as llm_chat

    papers = cluster.get("papers", []) or []
    paper_count = cluster.get("paper_count", len(papers))
    primary_count = cluster.get("primary_support_count", 0)
    keys = citation_keys if citation_keys is not None else _local_citation_keys(papers)
    sample = _narrative_sample(papers, max_papers)
    capacity = _synthesis_capacity(model_capacity)

    resumed = resume_state or {}
    partials = [dict(item) for item in resumed.get("partials", []) if isinstance(item, dict)]
    batch_records = [dict(item) for item in resumed.get("batch_requests", [])
                     if isinstance(item, dict)]
    completed_source_ids = {
        str(source_id)
        for record in batch_records
        for source_id in (record.get("source_ids") or [])
    }
    remaining = [p for p in sample if _paper_id(p) not in completed_source_ids]
    shown = [paper_id for part in partials for paper_id in (part.get("paper_ids") or [])]
    if partials:
        logger.info(
            "Resuming cluster %s from %d completed context batch(es); %d paper(s) remain",
            cluster.get("cluster_id", "?"), len(partials), len(remaining),
        )

    while remaining:
        selected, prompt_text, batch_coverage, budget = _fit_narrative_prompt(
            cluster, remaining, keys, research_config, capacity, max_tokens)
        selected_ids = [keys.get(_source_identity_key(p), "") or _paper_id(p)
                        for p in selected]
        selected_source_ids = [_paper_id(p) for p in selected]
        selected_titles = [_paper_title(p) for p in selected]
        logger.info(
            "Generating cluster %s context batch %d (%d papers, %d remaining, model=%s)",
            cluster.get("cluster_id", "?"), len(partials) + 1, len(selected),
            len(remaining) - len(selected), model,
        )
        partial = llm_chat(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=max_tokens if max_tokens and max_tokens > 0 else None,
            think=_synthesis_think_mode(model, enabled=use_thinking),
            cancel_event=cancel_event,
            task="synthesis_cluster",
            expected_items=max(1, len(selected)),
            context_tokens=capacity.context_window_tokens,
            token_budget=budget,
            explicit_limit_reason=_EXPLICIT_NARRATIVE_OUTPUT_REASON,
        )
        partials.append({"text": partial, "paper_ids": selected_ids})
        shown.extend(selected_ids)
        batch_records.append({
            "batch": len(batch_records) + 1,
            "paper_ids": selected_ids,
            "source_ids": selected_source_ids,
            "paper_titles": selected_titles,
            "coverage": batch_coverage,
            "token_budget": budget.as_dict(),
        })
        remaining = remaining[len(selected):]
        _checkpoint(
            checkpoint_callback, cluster_id=cluster.get("cluster_id"),
            phase="context_batch_complete", partials=partials,
            batch_records=batch_records)

    if len(partials) == 1:
        narrative = partials[0]["text"]
        combine_records: list[dict[str, Any]] = []
        multisection = False
    elif partials:
        narrative, combine_records, multisection = _combine_partial_narratives(
            model, partials, research_config, capacity, max_tokens,
            use_thinking, cancel_event,
            cluster_id=cluster.get("cluster_id"),
            batch_records=batch_records,
            checkpoint_callback=checkpoint_callback,
        )
    else:
        narrative = ""
        combine_records = []
        multisection = False

    explicit_omitted = max(len(papers) - len(sample), 0)
    source_chars = sum(
        record.get("coverage", {}).get("evidence_source_chars", 0)
        for record in batch_records
    )
    submitted_chars = sum(
        record.get("coverage", {}).get("evidence_submitted_chars", 0)
        for record in batch_records
    )
    coverage = {
        "context_window_tokens": capacity.context_window_tokens,
        "papers_available": len(papers),
        "papers_selected": len(sample),
        "papers_omitted": explicit_omitted,
        "fit_only_omissions": 0,
        "evidence_source_chars": source_chars,
        "evidence_submitted_chars": submitted_chars,
        "evidence_coverage_ratio": round(submitted_chars / source_chars, 6)
            if source_chars else 1.0,
        "truncated": any(
            record.get("coverage", {}).get("truncated") for record in batch_records),
        "policy": "context-fit-batches-covering-every-selected-paper",
        "reason": (
            "Every selected paper was sent in exactly one context-fit evidence batch. "
            "Grounded partial syntheses were retained and explicitly combined; no paper "
            "was omitted solely because of context size."
        ),
        "batch_requests": batch_records,
        "combine_requests": combine_records,
        "transparent_multisection_fallback": multisection,
    }
    if explicit_omitted:
        coverage["explicit_paper_limit"] = int(max_papers or len(sample))
        coverage["reason"] += (
            f" The caller explicitly limited synthesis to {len(sample)} of {len(papers)} papers."
        )
    logger.info("Cluster %s narrative: %d chars", cluster.get("cluster_id", "?"), len(narrative))

    result = {
        "cluster_id": cluster.get("cluster_id"),
        "narrative": narrative,
        "paper_count": paper_count,
        "papers_shown": shown,
        "papers_omitted": explicit_omitted,
        "input_coverage": coverage,
        "token_budget": batch_records[-1]["token_budget"] if batch_records else {},
        "partial_narratives": partials,
        "mechanisms": cluster.get("mechanisms", []),
        "conditions": cluster.get("conditions", []),
        "key_entities": claim_list(cluster, "key_entities"),
        "primary_support_count": primary_count,
    }
    _checkpoint(
        checkpoint_callback, cluster_id=cluster.get("cluster_id"),
        phase="cluster_complete", partials=partials,
        batch_records=batch_records, combine_records=combine_records)
    return result
