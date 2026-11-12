"""LLM-based relevance filtering of paper abstracts/full text."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, overload

from core.checkpoint import CheckpointManager, input_fingerprint, run_sliding_window
from process import llm as _llm
from process.claim_extractor import _claim_think_mode as _stage2_think_mode
from process.relevance_bert import (  # noqa: F401 - re-exported legacy names
    _bert_overrides,
    _bert_relevant_label_index,
    _get_cached_bert_scorer,
    _is_bert_relevance_model,
    _resolve_bert_model_name,
    _resolve_local_bert_relevance_dir,
    _score_relevance_with_bert,
    bert_scoring_identity,
    reset_bert_load_errors,
)
from process.relevance_parsing import (  # noqa: F401 - re-exported legacy names
    _coerce_score,
    _is_provider_request_validation_error,
    _last_explicit_score,
    _parse_batched_relevance_response,
    _parse_relevance_quote,
    _parse_relevance_response,
)
from process.llm_budget import TokenBudget
from process.relevance_input_budget import (
    _EXPLICIT_OUTPUT_REASON,
    _batch_prompt,
    _build_relevance_batches,
    _capacity,
    _fit_batch_payload,
    _fit_single_relevance_input,
    _paper_text_for_scoring,
    _score_batch_payload,
)

logger = logging.getLogger(__name__)


def llm_chat(*args: Any, **kwargs: Any) -> str:
    """Call ``process.llm.chat`` through the module attribute at call time.

    Binding the function at import would keep whatever ``chat`` was in place
    when this module was first imported (e.g. a test's patch) for good.
    """
    return _llm.chat(*args, **kwargs)


DEFAULT_CHECKPOINT_EVERY = 1
DEFAULT_BATCH_SIZE = 1
# Fallback defaults if no config passed; prefer config.concurrent_workers
_DEFAULT_WORKERS = 2
_DEFAULT_STAGGER = 2.0
# Legacy checkpoint filenames (for cleanup of old-format checkpoints)
_LEGACY_FILES = ("relevance_partial.json", "relevance_progress.json")


RELEVANCE_PROMPT = """You are a research assistant. Rate this paper's relevance to the following research question on a scale of 0-5:

RESEARCH QUESTION: {hypothesis}

SCORING:
0 = Completely irrelevant
1 = Tangentially related (shares topic area but does not answer the question)
2 = Somewhat related (touches one relevant concept but weakly)
3 = Moderately relevant (addresses at least one core concept from the question)
4 = Highly relevant (directly addresses multiple key concepts in the question)
5 = Essential (centrally and explicitly answers the research question)

PAPER TITLE: {title}

TEXT SOURCE: {text_source}

PAPER TEXT: {paper_text}

Respond with ONLY valid JSON:
{{
  "score": 0,
  "quoted_span": "exact verbatim quote copied from PAPER TEXT"
}}

Rules:
- score must be a single integer 0-5
- quoted_span must be copied exactly from PAPER TEXT
- quoted_span should be short (one or two sentences)
- if no supporting quote is available, quoted_span should be an empty string"""

_UNSCORED_EXAMPLES = 5  # titles named in the "could not be scored" warning


def _require_question(hypothesis: str) -> str:
    """The research question, or ValueError: never score against a placeholder."""
    question = (hypothesis or "").strip()
    if not question:
        raise ValueError(
            "Relevance scoring needs a research question; set the natural-language "
            "question or hypothesis before running Stage 2."
        )
    return question


def _failed_relevance(text_source: str, error: str) -> dict:
    """Record for a paper the scorer could not rate.

    ``relevance_score`` is None (never 0): a failed call or unreadable reply
    is not a judgement that the paper is irrelevant.
    """
    return {
        "relevance_score": None,
        "relevance_quote": "",
        "relevance_quote_source": text_source,
        "relevance_quote_valid": False,
        "score_error": error,
    }


BATCH_RELEVANCE_PROMPT = """You are a research assistant. Rate each paper's relevance to the following research question on a scale of 0-5.

RESEARCH QUESTION: {hypothesis}

SCORING:
0 = Completely irrelevant
1 = Tangentially related (shares topic area but does not answer the question)
2 = Somewhat related (touches one relevant concept but weakly)
3 = Moderately relevant (addresses at least one core concept from the question)
4 = Highly relevant (directly addresses multiple key concepts in the question)
5 = Essential (centrally and explicitly answers the research question)

Return ONLY valid JSON:
{{
  "results": [
    {{
      "id": "paper_1",
      "score": 0,
      "quoted_span": "exact verbatim quote copied from that paper's PAPER TEXT"
    }}
  ]
}}

Rules:
- return exactly one result object per paper id
- score must be a single integer 0-5
- quoted_span must be copied exactly from the matching PAPER TEXT
- quoted_span should be short (one or two sentences)
- if no supporting quote is available, quoted_span should be an empty string
- judge each paper independently

PAPERS:
{paper_blocks}"""


def _score_relevance_text(
    model: str,
    paper: dict,
    paper_text: str,
    text_source: str,
    *,
    use_thinking: bool = True,
    max_tokens: int | None = None,
    context_tokens: int | None = None,
    token_budget: TokenBudget | None = None,
    input_coverage: dict[str, Any] | None = None,
    hypothesis: str = "",
) -> dict:
    """Score one paper against a specific text payload."""
    prompt = RELEVANCE_PROMPT.format(
        hypothesis=_require_question(hypothesis),
        title=paper.get("title", ""),
        text_source=text_source,
        paper_text=paper_text,
    )

    text = llm_chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens if max_tokens and max_tokens > 0 else None,
        think=_stage2_think_mode(model, enabled=use_thinking),
        task="relevance_single",
        expected_items=1,
        context_tokens=context_tokens,
        token_budget=token_budget,
        explicit_limit_reason=_EXPLICIT_OUTPUT_REASON,
    )

    parsed = _parse_relevance_response(text)
    if parsed["relevance_score"] is None:
        logger.warning(f"Could not parse relevance score from: {text[:200]}...")
        return {
            **_failed_relevance(text_source, "unparseable_reply"),
            "relevance_input_coverage": input_coverage or {},
        }

    quote = parsed["relevance_quote"]
    quote_valid = bool(isinstance(quote, str) and quote and quote in paper_text)
    if quote and not quote_valid:
        logger.warning("Relevance quote was not an exact substring for '%s'", paper.get("title", "?")[:80])

    return {
        "relevance_score": parsed["relevance_score"],
        "relevance_quote": quote if quote_valid else "",
        "relevance_quote_source": text_source,
        "relevance_quote_valid": quote_valid,
        "relevance_input_coverage": input_coverage or {},
    }


def score_relevance(
    model: str,
    paper: dict,
    *,
    use_thinking: bool = True,
    max_tokens: int | None = None,
    abstract_chars: int | None = None,
    fulltext_chars: int | None = None,
    context_tokens: int | None = None,
    hypothesis: str = "",
) -> dict:
    """Score a single paper's relevance and capture an evidence quote."""
    _require_question(hypothesis)
    paper_text, text_source = _paper_text_for_scoring(
        paper,
        abstract_chars=abstract_chars,
        fulltext_chars=fulltext_chars,
    )
    if _is_bert_relevance_model(model):
        bert_result = _score_relevance_with_bert(
            model,
            paper_text,
            hypothesis=hypothesis,
        )
        # Keep source trace aligned with actual text source used for pair scoring.
        bert_result["relevance_quote_source"] = text_source
        return bert_result

    paper_text, coverage, budget = _fit_single_relevance_input(
        paper,
        paper_text,
        text_source,
        hypothesis=hypothesis,
        context_tokens=context_tokens,
        max_tokens=max_tokens,
    )

    try:
        return _score_relevance_text(
            model,
            paper,
            paper_text,
            text_source,
            use_thinking=use_thinking,
            max_tokens=max_tokens,
            context_tokens=budget.context_window_tokens,
            token_budget=budget,
            input_coverage=coverage,
            hypothesis=hypothesis,
        )
    except Exception as exc:
        abstract_text, _ = _paper_text_for_scoring(
            {**paper, "full_text": ""},
            abstract_chars=abstract_chars,
            fulltext_chars=0,
        )
        if (
            text_source == "full_text"
            and abstract_text
            and _is_provider_request_validation_error(exc)
        ):
            logger.warning(
                "Provider rejected full-text relevance request for '%s'; retrying with abstract only: %s",
                paper.get("title", "?")[:80],
                exc,
            )
            try:
                abstract_text, abstract_coverage, abstract_budget = _fit_single_relevance_input(
                    paper,
                    abstract_text,
                    "abstract",
                    hypothesis=hypothesis,
                    context_tokens=context_tokens,
                    max_tokens=max_tokens,
                )
                return _score_relevance_text(
                    model,
                    paper,
                    abstract_text,
                    "abstract",
                    use_thinking=use_thinking,
                    max_tokens=max_tokens,
                    context_tokens=abstract_budget.context_window_tokens,
                    token_budget=abstract_budget,
                    input_coverage=abstract_coverage,
                    hypothesis=hypothesis,
                )
            except Exception as retry_exc:
                if _is_provider_request_validation_error(retry_exc):
                    logger.warning(
                        "Provider still rejected abstract fallback for '%s'; skipping paper: %s",
                        paper.get("title", "?")[:80],
                        retry_exc,
                    )
                    return {
                        **_failed_relevance("abstract", f"provider_rejected: {retry_exc}"),
                        "relevance_input_coverage": abstract_coverage,
                    }
                raise

        if _is_provider_request_validation_error(exc):
            logger.warning(
                "Provider rejected relevance request for '%s'; skipping paper: %s",
                paper.get("title", "?")[:80],
                exc,
            )
            return {
                **_failed_relevance(text_source, f"provider_rejected: {exc}"),
                "relevance_input_coverage": coverage,
            }
        raise


def score_relevance_batch(
    model: str,
    papers: list[dict],
    *,
    use_thinking: bool = True,
    max_tokens: int | None = None,
    single_max_tokens: int | None = None,
    abstract_chars: int | None = None,
    fulltext_chars: int | None = None,
    context_tokens: int | None = None,
    hypothesis: str = "",
) -> list[dict]:
    """Score a batch of papers in one prompt; fall back per-paper if needed."""
    _require_question(hypothesis)
    if _is_bert_relevance_model(model):
        return [
            score_relevance(
                model,
                paper,
                use_thinking=use_thinking,
                max_tokens=single_max_tokens,
                abstract_chars=abstract_chars,
                fulltext_chars=fulltext_chars,
                context_tokens=context_tokens,
                hypothesis=hypothesis,
            )
            for paper in papers
        ]

    if len(papers) <= 1:
        return [
            score_relevance(
                model,
                papers[0],
                use_thinking=use_thinking,
                max_tokens=single_max_tokens,
                abstract_chars=abstract_chars,
                fulltext_chars=fulltext_chars,
                context_tokens=context_tokens,
                hypothesis=hypothesis,
            )
        ] if papers else []

    payload = _score_batch_payload(
        papers,
        abstract_chars=abstract_chars,
        fulltext_chars=fulltext_chars,
    )
    payload, prompt, budget = _fit_batch_payload(
        payload,
        hypothesis=hypothesis,
        context_tokens=context_tokens,
        max_tokens=max_tokens,
    )

    try:
        text = llm_chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens if max_tokens and max_tokens > 0 else None,
            think=_stage2_think_mode(model, enabled=use_thinking),
            task="relevance_batch",
            expected_items=len(payload),
            context_tokens=budget.context_window_tokens,
            token_budget=budget,
            explicit_limit_reason=_EXPLICIT_OUTPUT_REASON,
        )
        parsed = _parse_batched_relevance_response(text)
    except Exception as e:
        logger.warning(
            "Batch relevance scoring failed for %d papers; retrying individually: %s",
            len(papers),
            e,
        )
        parsed = {}

    results: list[dict] = []
    for item, paper in zip(payload, papers):
        parsed_item = parsed.get(item["id"])
        if parsed_item is None:
            logger.warning(
                "Batch scoring returned no usable result for %s; retrying individually",
                paper.get("title", "?")[:80],
            )
            results.append(
                score_relevance(
                    model,
                    paper,
                    use_thinking=use_thinking,
                    max_tokens=single_max_tokens,
                    abstract_chars=abstract_chars,
                    fulltext_chars=fulltext_chars,
                    context_tokens=context_tokens,
                    hypothesis=hypothesis,
                )
            )
            continue

        quote = parsed_item["relevance_quote"]
        quote_valid = bool(quote and quote in item["paper_text"])
        if quote and not quote_valid:
            logger.warning(
                "Batch relevance quote was not an exact substring for '%s'; retrying individually",
                paper.get("title", "?")[:80],
            )
            results.append(
                score_relevance(
                    model,
                    paper,
                    use_thinking=use_thinking,
                    max_tokens=single_max_tokens,
                    abstract_chars=abstract_chars,
                    fulltext_chars=fulltext_chars,
                    context_tokens=context_tokens,
                    hypothesis=hypothesis,
                )
            )
            continue

        results.append({
            "relevance_score": parsed_item["relevance_score"],
            "relevance_quote": quote if quote_valid else "",
            "relevance_quote_source": item["text_source"],
            "relevance_quote_valid": quote_valid,
            "relevance_input_coverage": item["input_coverage"],
        })

    return results


@overload
def filter_corpus(
    papers: list[dict],
    model: str,
    threshold: int = 3,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    stagger_delay: float | None = None,
    cancel_event: Any = None,
    progress_callback: Callable[[str], None] | None = None,
    hypothesis: str = "",
    concurrent_workers: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    future_timeout: float | None = None,
    use_thinking: bool = True,
    score_max_tokens: int | None = None,
    score_batch_max_tokens: int | None = None,
    abstract_chars: int | None = None,
    fulltext_chars: int | None = None,
    context_tokens: int | None = None,
    *,
    return_scored: Literal[False] = False,
    **kwargs,
) -> tuple[list[dict], bool]: ...


@overload
def filter_corpus(
    papers: list[dict],
    model: str,
    threshold: int = 3,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    stagger_delay: float | None = None,
    cancel_event: Any = None,
    progress_callback: Callable[[str], None] | None = None,
    hypothesis: str = "",
    concurrent_workers: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    future_timeout: float | None = None,
    use_thinking: bool = True,
    score_max_tokens: int | None = None,
    score_batch_max_tokens: int | None = None,
    abstract_chars: int | None = None,
    fulltext_chars: int | None = None,
    context_tokens: int | None = None,
    *,
    return_scored: Literal[True],
    **kwargs,
) -> tuple[list[dict], bool, list[dict]]: ...


def filter_corpus(
    papers: list[dict],
    model: str,
    threshold: int = 3,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    stagger_delay: float | None = None,
    cancel_event: Any = None,
    progress_callback: Callable[[str], None] | None = None,
    hypothesis: str = "",
    concurrent_workers: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    future_timeout: float | None = None,
    use_thinking: bool = True,
    score_max_tokens: int | None = None,
    score_batch_max_tokens: int | None = None,
    abstract_chars: int | None = None,
    fulltext_chars: int | None = None,
    context_tokens: int | None = None,
    return_scored: bool = False,
    **kwargs,
) -> tuple[list[dict], bool] | tuple[list[dict], bool, list[dict]]:
    """Filter papers by relevance score. Saves progress to checkpoint_dir so you can resume.

    Papers the scorer could not rate get ``relevance_score=None`` and a
    ``score_error``. They are never counted as relevant, are reported through
    ``progress_callback``, and are kept out of the checkpoint so a re-run
    retries them. A blank ``hypothesis`` raises ValueError.
    """
    _require_question(hypothesis)
    _workers = concurrent_workers if concurrent_workers is not None else _DEFAULT_WORKERS
    _stagger = stagger_delay if stagger_delay is not None else _DEFAULT_STAGGER
    _batch_size = max(1, int(batch_size or DEFAULT_BATCH_SIZE))
    _future_timeout = max(1.0, float(future_timeout or 120.0))
    _score_max_tokens = max(0, int(score_max_tokens or 0))
    _score_batch_max_tokens = max(0, int(score_batch_max_tokens or 0))
    _abstract_chars = max(0, int(abstract_chars or 0)) or None
    _fulltext_chars = None if fulltext_chars is None else max(0, int(fulltext_chars))
    _context_tokens = _capacity(context_tokens).context_window_tokens
    scorer_identity: dict = {}
    if _is_bert_relevance_model(model):
        _workers, _batch_size = _bert_overrides(
            use_thinking, _workers, _batch_size, _score_max_tokens, _score_batch_max_tokens)
        reset_bert_load_errors()  # a fixed install is retried without a restart
        scorer_identity = bert_scoring_identity(model)

    text_kwargs = {
        "use_thinking": use_thinking, "abstract_chars": _abstract_chars,
        "fulltext_chars": _fulltext_chars, "hypothesis": hypothesis,
        "context_tokens": _context_tokens,
    }
    fingerprint = relevance_checkpoint_fingerprint(
        papers, model, batch_size=_batch_size, max_tokens=_score_max_tokens,
        batch_max_tokens=_score_batch_max_tokens, context_policy="context-aware/v1",
        **scorer_identity, **text_kwargs)
    ckpt = CheckpointManager(
        checkpoint_dir, "relevance_checkpoint.json", checkpoint_every, fingerprint=fingerprint)
    scored, start_index = _load_scored(ckpt, len(papers))

    if _batch_size > 1:
        new_records, cancelled = _score_in_batches(
            model, papers, start_index, _batch_size, prior=scored, ckpt=ckpt,
            cancel_event=cancel_event, workers=_workers,
            single_max_tokens=_score_max_tokens, batch_max_tokens=_score_batch_max_tokens,
            text_kwargs=text_kwargs)
    else:
        new_records, cancelled = _score_in_window(
            model, papers, start_index, prior=scored, ckpt=ckpt, cancel_event=cancel_event,
            workers=_workers, stagger=_stagger, future_timeout=_future_timeout,
            max_tokens=_score_max_tokens, text_kwargs=text_kwargs)
    all_scored = scored + new_records

    filtered = _finish_filter(
        ckpt, len(papers), all_scored, threshold, cancelled, progress_callback)
    if return_scored:
        return filtered, cancelled, all_scored
    return filtered, cancelled


def relevance_checkpoint_fingerprint(papers: list[dict], model: str, **settings: Any) -> str:
    """Identify a legacy scoring run: ordered papers, model, prompts and settings.

    The relevance checkpoint resumes only when this matches, so a checkpoint
    left by a cancelled run is never merged into a run over other papers, a
    different question, model or prompt (audit 05-H4).
    """
    return input_fingerprint(
        papers, model=model, prompt=RELEVANCE_PROMPT,
        batch_prompt=BATCH_RELEVANCE_PROMPT, **settings)



def _load_scored(ckpt: CheckpointManager, total: int) -> tuple[list[dict], int]:
    """Resume state from a matching checkpoint (legacy two-file ones are removed)."""
    scored, start_index = ckpt.load()
    scored = [
        {"relevance_quote": "", "relevance_quote_source": "abstract",
         "relevance_quote_valid": False, **record}
        for record in scored
    ]
    ckpt.cleanup_legacy(*_LEGACY_FILES)
    if 0 < start_index < total:
        logger.info(
            "Resuming relevance filter: %d already scored, %d remaining",
            start_index, total - start_index,
        )
    return scored, start_index


def _is_relevant(record: dict, threshold: int) -> bool:
    score = record.get("relevance_score")
    return score is not None and score >= threshold


def _resumable_prefix(records: list[dict]) -> list[dict]:
    """Records before the first scoring failure, so a resume retries it."""
    for index, record in enumerate(records):
        if record.get("score_error"):
            return records[:index]
    return records


def _score_in_batches(
    model: str, papers: list[dict], start_index: int, batch_size: int, *,
    prior: list[dict], ckpt: CheckpointManager, cancel_event: Any, workers: int,
    single_max_tokens: int | None, batch_max_tokens: int | None, text_kwargs: dict,
) -> tuple[list[dict], bool]:
    """Score papers[start_index:] in multi-paper prompts; returns (records, cancelled)."""
    batches = _build_relevance_batches(
        papers, start_index, batch_size,
        abstract_chars=text_kwargs["abstract_chars"],
        fulltext_chars=text_kwargs["fulltext_chars"],
        context_tokens=text_kwargs["context_tokens"],
        hypothesis=text_kwargs["hypothesis"],
        max_tokens=batch_max_tokens,
    )
    logger.info(
        "Batch scoring enabled: up to %d papers/request, %d batches; input size is "
        "planned from the %d-token model context for each request",
        batch_size, len(batches), text_kwargs["context_tokens"],
    )
    if workers > 1:
        logger.info(
            "Batch mode runs sequentially to reduce repeated model offload churn; "
            "ignoring concurrent_workers=%d",
            workers,
        )

    new_records: list[dict] = []
    for batch_num, batch_indices in enumerate(batches, start=1):
        if cancel_event and cancel_event.is_set():
            logger.info(
                "Relevance filter: stop requested after %d/%d papers",
                start_index + len(new_records), len(papers),
            )
            return new_records, True
        batch_results = score_relevance_batch(
            model, [papers[idx] for idx in batch_indices],
            max_tokens=batch_max_tokens, single_max_tokens=single_max_tokens, **text_kwargs,
        )
        for idx, result in zip(batch_indices, batch_results):
            new_records.append({**papers[idx], **result})
            ckpt.save(_resumable_prefix(prior + new_records), len(papers))
            logger.info(
                "Scored %d/%d  (batch %d/%d, score=%s)",
                start_index + len(new_records), len(papers), batch_num, len(batches),
                new_records[-1].get("relevance_score"),
            )
    return new_records, False


def _score_in_window(
    model: str, papers: list[dict], start_index: int, *,
    prior: list[dict], ckpt: CheckpointManager, cancel_event: Any, workers: int,
    stagger: float, future_timeout: float, max_tokens: int | None, text_kwargs: dict,
) -> tuple[list[dict], bool]:
    """Score papers[start_index:] one per request; returns (records, cancelled)."""
    new_records: list[dict] = []
    failures: dict[int, str] = {}  # paper index -> "failed" | "timed_out"

    def _on_failure(idx: int, kind: str, _error: BaseException | None) -> None:
        failures[idx] = kind

    def _worker(paper: dict) -> dict:
        return score_relevance(model, paper, max_tokens=max_tokens, **text_kwargs)

    def _build_result(idx: int, result: dict | None) -> dict:
        if result is None:
            _, source = _paper_text_for_scoring(
                papers[idx], abstract_chars=text_kwargs["abstract_chars"],
                fulltext_chars=text_kwargs["fulltext_chars"])
            error = "worker_timeout" if failures.get(idx) == "timed_out" else "worker_failed"
            result = _failed_relevance(source, error)
        record = {**papers[idx], **result}
        new_records.append(record)
        return record

    def _on_checkpoint() -> None:
        ckpt.save(_resumable_prefix(prior + new_records), len(papers))

    def _on_progress(done: int, total: int) -> None:
        score = new_records[-1].get("relevance_score") if new_records else "?"
        logger.info("Scored %d/%d  (score=%s)", done + start_index, total, score)

    return run_sliding_window(
        items=papers,
        worker_fn=_worker,
        result_fn=_build_result,
        start_index=start_index,
        workers=workers,
        stagger=stagger,
        checkpoint_fn=_on_checkpoint,
        log_fn=_on_progress,
        cancel_event=cancel_event,
        future_timeout=future_timeout,
        on_failure=_on_failure,
    )


def _report_unscored(unscored: list[dict], progress_callback: Callable[[str], None] | None) -> None:
    if not unscored:
        return
    label = "paper" if len(unscored) == 1 else "papers"
    titles = "; ".join(str(p.get("title") or "?")[:60] for p in unscored[:_UNSCORED_EXAMPLES])
    message = (
        f"WARNING: {len(unscored)} {label} could not be scored (model call failed or "
        f"reply unreadable) and were NOT counted as relevant. Screen them manually "
        f"or re-run Stage 2 to retry them: {titles}"
    )
    logger.warning(message)
    if progress_callback:
        progress_callback(message)


def _finish_filter(
    ckpt: CheckpointManager, total: int, scored: list[dict], threshold: int,
    cancelled: bool, progress_callback: Callable[[str], None] | None,
) -> list[dict]:
    """Apply the threshold, report failures, and keep or clear the checkpoint."""
    filtered = [p for p in scored if _is_relevant(p, threshold)]
    logger.info(f"Relevance filter: {total} -> {len(filtered)} papers (threshold={threshold})")
    unscored = [p for p in scored if p.get("score_error")]
    _report_unscored(unscored, progress_callback)
    if cancelled:
        logger.info("Relevance filter stopped early: %d/%d papers processed", len(scored), total)
    if cancelled or unscored:
        # Keep the progress before the first failure; a re-run retries the rest.
        ckpt.force_save(_resumable_prefix(scored), total)
    else:
        ckpt.cleanup()
    return filtered
