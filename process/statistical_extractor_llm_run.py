"""Run the Stage 2c LLM context check: bundle, ask, attach, report."""
from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from process.llm_budget import ModelCapacity, resolve_model_capacity
from process.llm_context import ContextWindowExceededError, OutputLimitExceededError
from process.statistical_extractor_llm import (
    _review_target_groups,
    build_llm_context_prompt,
    failed_review,
    has_llm_corrected_fields,
    is_failed_llm_review,
    parse_llm_context_response,
    review_location,
)
from process.statistical_extractor_llm_bundle import (
    Bundle,
    BundleStats,
    Target,
    bundle_reviews,
    make_bundle,
    plan_bundles,
)

_LABEL_CHARS = 80


def empty_counts() -> dict[str, Any]:
    return {
        "llm_reviewed": 0, "llm_supported": 0, "llm_rejected": 0,
        "llm_unclear": 0, "llm_corrected": 0, "llm_failed": 0,
    }


def apply_llm_context_reviews(
    records: list[dict[str, Any]],
    *,
    model: str,
    max_contexts: int = 0,
    include_human_flags: bool = True,
    use_thinking: bool = False,
    max_tokens: int | None = None,
    progress_callback=None,
    cancel_event=None,
    chat_fn=None,
    concurrent_workers: int = 1,
    stagger_delay: float = 0.0,
    future_timeout: float | None = None,
    bundle_tokens: int = 0,
    forget_reply_fn=None,
    config: Any = None,
    context_tokens: int | None = None,
    normalization_scope_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ask an LLM to validate local context windows around extracted statistics.

    Redundant flags share one request and receive independent copies of its
    verdict. Different statistics or source passages remain separate decisions.
    Requests are sent ``concurrent_workers`` at a time; reviews are attached
    in target order, so the output matches a sequential run. ``future_timeout``
    of ``None`` applies no stage-level limit (the LLM client's own HTTP timeouts
    still apply), matching the sequential behaviour.

    A window whose LLM call errors or times out gets no verdict: it is counted
    in ``llm_reviewed`` and ``llm_failed`` (never as rejected), reported through
    ``progress_callback`` with the error, and flagged for human review.

    ``bundle_tokens == 0`` packs each paper automatically against the resolved
    loaded-model context. A positive value is an explicit synthetic
    total-request override, not a model-capability claim. Each fit includes the
    complete prompt, a schema/item-derived response allowance, and a safety
    reserve. A positive ``max_tokens`` is retained only as an explicit response
    override and is recorded as such; otherwise no 900-token magic allowance is
    used. Oversized, truncated, or malformed multi-item calls are split and
    retried locally without stopping later papers. Only a terminal singleton
    failure is left for human review.

    ``forget_reply_fn`` (``chat``'s signature) drops an unusable reply from the
    response cache so a rerun asks the model again; it defaults to
    ``process.llm.forget_cached_reply`` when ``chat_fn`` is the default. Usable
    replies stay cached, so a rerun after a cancel only asks the rest.

    ``normalization_scope_records`` is used when ``records`` is a producer's
    pending subset. Its labels participate in corpus-wide canonical selection,
    but only the newly produced ``records`` are changed. This keeps
    canonicalization at the model-output boundary rather than repairing prior
    saved labels downstream.
    """
    if not model:
        return empty_counts()
    if chat_fn is None:
        from process.llm import chat as chat_fn
        from process.llm import forget_cached_reply

        forget_reply_fn = forget_reply_fn or forget_cached_reply
    if cancel_event is not None and not hasattr(cancel_event, "is_set"):
        if cancel_event:
            return empty_counts()
        cancel_event = None

    target_groups = _review_target_groups(records, include_human_flags=include_human_flags)
    available_targets = len(target_groups)
    if max_contexts > 0:
        target_groups = target_groups[:max_contexts]
    capacity, planning = _review_capacity(
        config=config,
        model=model,
        context_tokens=context_tokens,
        bundle_tokens=bundle_tokens,
    )
    counts = ReviewRun(
        target_groups,
        model=model,
        chat_fn=chat_fn,
        use_thinking=use_thinking,
        max_tokens=max_tokens,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        future_timeout=future_timeout,
        bundle_tokens=bundle_tokens,
        forget_reply_fn=forget_reply_fn,
        capacity=capacity,
        planning=planning,
    ).execute(workers=max(1, int(concurrent_workers)), stagger=max(0.0, float(stagger_delay)))
    counts["scope"] = {
        "available": available_targets,
        "reviewed": len(target_groups),
        "limited": bool(max_contexts > 0 and len(target_groups) < available_targets),
        "limit": int(max_contexts) if max_contexts > 0 else 0,
        "action": "explicit-user-scope-limit" if max_contexts > 0 else "review-all-candidates",
        "reason": (
            "The caller explicitly limited the number of deterministic candidates reviewed."
            if max_contexts > 0 else
            "No synthetic candidate-count cap was applied."
        ),
    }

    from process.statistical_extractor_pipeline import promote_reviewed_results
    from process.term_normalization import normalize_statistical_records

    promote_reviewed_results(records)
    # This is the model-output boundary.  Attach canonical structured labels
    # before returning records to any caller (Stage 2c or corpus preparation),
    # so meta-analysis never performs label cleanup itself.
    counts["term_normalization"] = normalize_statistical_records(
        records,
        config,
        normalization_scope=normalization_scope_records,
    )
    return counts


def _review_capacity(
    *,
    config: Any,
    model: str,
    context_tokens: int | None,
    bundle_tokens: int,
) -> tuple[ModelCapacity, dict[str, Any]]:
    """Resolve actual capacity, then apply an explicit total-request override."""
    capacity_config = config or SimpleNamespace(
        llm_provider="",
        llm_model=model,
        llm_context_tokens=0,
        llm_max_output_tokens=None,
    )
    actual = resolve_model_capacity(
        capacity_config,
        requested_context_tokens=context_tokens,
        model=model,
    )
    override = max(0, int(bundle_tokens or 0))
    effective = min(actual.context_window_tokens, override) if override else actual.context_window_tokens
    if override and override < actual.context_window_tokens:
        source = "explicit Stage 2c total-request override"
        action = "user-total-request-limit"
        reason = (
            "The caller requested a smaller synthetic total-request boundary; it is applied "
            "to prompt, response, and safety together and is not presented as model capacity."
        )
    elif override:
        source = actual.source
        action = "model-context-limit"
        reason = (
            "The requested total-request override is not smaller than the resolved model "
            "context, so actual model capacity remains the effective boundary."
        )
    else:
        source = actual.source
        action = "automatic-context-packing"
        reason = (
            "Automatic mode uses the resolved loaded-model context and packs complete "
            "per-paper requests without a fixed application token ceiling."
        )
    planned = ModelCapacity(
        effective,
        actual.max_output_tokens,
        source,
        actual.provider,
        actual.model or model,
        actual.confirmed,
        reason,
    )
    return planned, {
        "mode": "explicit-synthetic-total-request-limit" if override else "automatic",
        "action": action,
        "reason": reason,
        "requested_total_request_tokens": override,
        "resolved_model_context_tokens": actual.context_window_tokens,
        "effective_total_request_tokens": effective,
        "capacity_source": actual.source,
        "capacity_confirmed": actual.confirmed,
        "capacity_detail": actual.detail,
    }


def _count_review(counts: dict[str, Any], review: dict[str, Any]) -> None:
    counts["llm_reviewed"] += 1
    if is_failed_llm_review(review):
        counts["llm_failed"] += 1
        return
    if has_llm_corrected_fields(review):
        counts["llm_corrected"] += 1
    if review.get("represents_study_data") and review.get("usable_for_meta_analysis"):
        counts["llm_supported"] += 1
    elif review.get("represents_study_data") is False:
        counts["llm_rejected"] += 1
    else:
        counts["llm_unclear"] += 1


def _single_prompt(target: Target) -> str:
    record, candidate, context = target
    return build_llm_context_prompt(
        str(record.get("title") or ""), candidate, context,
        article_context=str(record.get("review_context") or ""),
    )


def _label(record: dict[str, Any]) -> str:
    return str(record.get("title") or record.get("article_id") or "")[:_LABEL_CHARS]


def _read_reply(response: str, bundle: Bundle) -> list[dict[str, Any]]:
    """Reviews for ``bundle`` from the reply; a reply that cannot be read fails them."""
    try:
        if bundle.size == 1:
            return [parse_llm_context_response(response)]
        return bundle_reviews(response, bundle)
    except Exception as exc:  # noqa: BLE001 - e.g. a number too large for a float
        reason = f"could not read LLM response ({type(exc).__name__}: {exc})"
        return [failed_review(reason)] * bundle.size


@dataclass
class _ReviewOutcome:
    """Final reviews plus the complete audit trail for one initially planned bundle."""

    reviews: list[dict[str, Any]]
    attempts: list[dict[str, Any]] = field(default_factory=list)
    splits: list[dict[str, Any]] = field(default_factory=list)
    recoveries: list[dict[str, Any]] = field(default_factory=list)
    terminals: list[dict[str, Any]] = field(default_factory=list)

    def extend(self, other: "_ReviewOutcome") -> None:
        self.attempts.extend(other.attempts)
        self.splits.extend(other.splits)
        self.recoveries.extend(other.recoveries)
        self.terminals.extend(other.terminals)


def _review_problem(review: dict[str, Any]) -> str:
    bundle = review.get("bundle")
    if isinstance(bundle, dict) and bundle.get("reject_reason"):
        return str(bundle["reject_reason"])
    return str(review.get("error") or review.get("reason") or "invalid model verdict")


class ReviewRun:
    """One pass of LLM context checks over ``target_groups`` (see ``_review_target_groups``)."""

    def __init__(self, target_groups: list[list[Target]], *, model: str, chat_fn: Callable,
                 use_thinking: bool, max_tokens: int | None,
                 progress_callback: Callable[[str], None] | None, cancel_event: Any,
                 future_timeout: float | None, bundle_tokens: int,
                 forget_reply_fn: Callable | None = None,
                 capacity: ModelCapacity, planning: dict[str, Any]) -> None:
        self._groups = target_groups
        self._targets = [group[0] for group in target_groups]
        self._model = model
        self._chat = chat_fn
        self._forget_reply = forget_reply_fn
        self._think = bool(use_thinking) or None
        self._max_tokens = int(max_tokens) if max_tokens and int(max_tokens) > 0 else None
        self._progress = progress_callback
        self._cancel_event = cancel_event
        self._future_timeout = future_timeout
        self._bundle_tokens = bundle_tokens
        self._capacity = capacity
        self._planning = dict(planning)
        self._counts = empty_counts()
        self._stats = BundleStats()
        self._prompts: list[str] = []

    def execute(self, *, workers: int, stagger: float) -> dict[str, Any]:
        from core.checkpoint import run_sliding_window

        self._prompts = [_single_prompt(target) for target in self._targets]
        bundles = plan_bundles(
            self._targets,
            self._prompts,
            capacity=self._capacity,
            explicit_output_tokens=self._max_tokens,
        )
        self._stats.record_plan(bundles, self._planning)
        run_sliding_window(
            items=bundles,
            worker_fn=self._ask,
            result_fn=lambda position, outcome: self._collect(bundles[position], outcome),
            workers=workers,
            stagger=stagger,
            cancel_event=self._cancel_event,
            future_timeout=math.inf if self._future_timeout is None else float(self._future_timeout),
        )
        return {**self._counts, "bundles": self._stats.as_dict()}

    def _ask(self, bundle: Bundle) -> Any:
        """Ask one bundle, recursively retrying only the affected statistics."""
        if bundle.planning_error is not None:
            reason = str(bundle.planning_error.get("error") or "request does not fit")
            reviews = [failed_review(reason) for _ in bundle.indices]
            return _ReviewOutcome(
                reviews,
                attempts=[self._attempt(
                    bundle,
                    sent=False,
                    outcome="planning-error",
                    error=reason,
                    action="terminal-human-review",
                )],
                terminals=[self._terminal_event(bundle, "budget-planning-error", reason)],
            )

        messages = [{"role": "user", "content": bundle.prompt}]
        try:
            response = self._chat(
                self._model,
                messages,
                temperature=0,
                max_tokens=bundle.max_tokens,
                think=self._think,
                task="statistical_review",
                expected_items=bundle.size,
                token_budget=bundle.token_budget,
                **({"cancel_event": self._cancel_event} if self._cancel_event is not None else {}),
            )
        except (ContextWindowExceededError, OutputLimitExceededError) as exc:
            failure = (
                "context-window-overflow"
                if isinstance(exc, ContextWindowExceededError)
                else "output-limit-exceeded"
            )
            attempt = self._attempt(
                bundle,
                sent=True,
                outcome=failure,
                error=str(exc),
                action="split-and-retry" if bundle.size > 1 else "terminal-human-review",
            )
            if bundle.size > 1:
                children = self._halves(bundle)
                return self._retry(
                    bundle,
                    children=children,
                    retained={},
                    affected=list(bundle.indices),
                    attempt=attempt,
                    reason=failure,
                    detail=str(exc),
                )
            review = failed_review(str(exc))
            return _ReviewOutcome(
                [review],
                attempts=[attempt],
                terminals=[self._terminal_event(bundle, failure, str(exc))],
            )
        except Exception as exc:  # noqa: BLE001 - recorded as a failed check
            reviews = [failed_review(str(exc)) for _ in bundle.indices]
            return _ReviewOutcome(
                reviews,
                attempts=[self._attempt(
                    bundle,
                    sent=True,
                    outcome="request-error",
                    error=str(exc),
                    action="terminal-human-review",
                )],
                terminals=[self._terminal_event(bundle, "request-error", str(exc))],
            )

        reviews = _read_reply(response, bundle)
        invalid_positions = [
            position for position, review in enumerate(reviews)
            if is_failed_llm_review(review)
        ]
        if not invalid_positions:
            return _ReviewOutcome(
                reviews,
                attempts=[self._attempt(
                    bundle,
                    sent=True,
                    outcome="valid-reply",
                    reply=response,
                    action="apply-verdicts",
                )],
            )

        reason = _review_problem(reviews[invalid_positions[0]])
        cache_action = self._evict(bundle, messages)
        attempt = self._attempt(
            bundle,
            sent=True,
            outcome="invalid-reply",
            reply=response,
            error=reason,
            action="retry-invalid-smaller" if bundle.size > 1 else "terminal-human-review",
            rejected_statistics=len(invalid_positions),
            cache_action=cache_action,
        )
        if bundle.size == 1:
            return _ReviewOutcome(
                reviews,
                attempts=[attempt],
                terminals=[self._terminal_event(bundle, "invalid-singleton-reply", reason)],
            )

        affected = [bundle.indices[position] for position in invalid_positions]
        retained = {
            index: review
            for index, review in zip(bundle.indices, reviews, strict=True)
            if index not in affected
        }
        children = (
            self._halves(bundle)
            if len(affected) == bundle.size
            else [self._make(affected)]
        )
        return self._retry(
            bundle,
            children=children,
            retained=retained,
            affected=affected,
            attempt=attempt,
            reason="invalid-bundle-reply",
            detail=reason,
        )

    def _make(self, indices: list[int] | tuple[int, ...]) -> Bundle:
        return make_bundle(
            self._targets,
            self._prompts,
            indices,
            capacity=self._capacity,
            explicit_output_tokens=self._max_tokens,
        )

    def _halves(self, bundle: Bundle) -> list[Bundle]:
        midpoint = max(1, bundle.size // 2)
        return [self._make(bundle.indices[:midpoint]), self._make(bundle.indices[midpoint:])]

    def _retry(
        self,
        bundle: Bundle,
        *,
        children: list[Bundle],
        retained: dict[int, dict[str, Any]],
        affected: list[int],
        attempt: dict[str, Any],
        reason: str,
        detail: str,
    ) -> _ReviewOutcome:
        """Retry smaller complete requests and merge them with trusted verdicts."""
        child_indices = [list(child.indices) for child in children]
        outcome = _ReviewOutcome(
            [],
            attempts=[attempt],
            splits=[{
                "reason": reason,
                "detail": detail,
                "source_indices": [index + 1 for index in bundle.indices],
                "child_indices": [[index + 1 for index in indices] for indices in child_indices],
                "action": "split-affected-bundle-and-retry-in-same-run",
            }],
        )
        merged = dict(retained)
        for child in children:
            child_outcome = self._ask(child)
            outcome.extend(child_outcome)
            merged.update(zip(child.indices, child_outcome.reviews, strict=True))

        outcome.reviews = [merged[index] for index in bundle.indices]
        recovered = [
            index for index in affected
            if not is_failed_llm_review(merged[index])
        ]
        if recovered:
            outcome.recoveries.append({
                "reason": reason,
                "statistics": len(recovered),
                "indices": [index + 1 for index in recovered],
                "action": "applied-verdicts-from-smaller-retry",
            })
        return outcome

    def _evict(self, bundle: Bundle, messages: list[dict[str, str]]) -> str:
        if self._forget_reply is None:
            return "no-cache-evictor-configured"
        try:
            self._forget_reply(
                self._model,
                messages,
                temperature=0,
                max_tokens=bundle.max_tokens,
                think=self._think,
                task="statistical_review",
                expected_items=bundle.size,
                token_budget=bundle.token_budget,
            )
        except Exception as exc:  # noqa: BLE001 - recovery must still continue
            return f"cache-eviction-failed: {type(exc).__name__}: {exc}"
        return "evicted-invalid-reply-before-retry"

    @staticmethod
    def _attempt(
        bundle: Bundle,
        *,
        sent: bool,
        outcome: str,
        action: str,
        reply: str | None = None,
        error: str = "",
        rejected_statistics: int = 0,
        cache_action: str = "",
    ) -> dict[str, Any]:
        return {
            "indices": [index + 1 for index in bundle.indices],
            "statistics": bundle.size,
            "sent": sent,
            "outcome": outcome,
            "action": action,
            "estimated_total_tokens": bundle.estimated_tokens,
            "budget": bundle.budget_dict(),
            "reply": reply,
            "error": error,
            "reason": error,
            "rejected_statistics": rejected_statistics,
            **({"cache_action": cache_action} if cache_action else {}),
        }

    @staticmethod
    def _terminal_event(bundle: Bundle, reason: str, detail: str) -> dict[str, Any]:
        return {
            "reason": reason,
            "detail": detail,
            "statistics": bundle.size,
            "indices": [index + 1 for index in bundle.indices],
            "action": "keep-unverified-and-require-human-review",
        }

    def _collect(self, bundle: Bundle, outcome: Any) -> dict[str, Any]:
        if outcome is None:  # the worker timed out
            waited = "" if self._future_timeout is None else f" within {self._future_timeout:g}s"
            reason = f"no response{waited}"
            outcome = _ReviewOutcome(
                [failed_review(reason) for _ in bundle.indices],
                attempts=[self._attempt(
                    bundle,
                    sent=True,
                    outcome="request-timeout",
                    error=reason,
                    action="terminal-human-review",
                )],
                terminals=[self._terminal_event(bundle, "request-timeout", reason)],
            )
        record = self._targets[bundle.indices[0]][0]
        self._stats.record_attempts(outcome.attempts, record=record)
        self._stats.record_events(
            splits=outcome.splits,
            recoveries=outcome.recoveries,
            terminals=outcome.terminals,
            record=record,
        )
        for index, review in zip(bundle.indices, outcome.reviews, strict=True):
            for _record, alias, _context in self._groups[index]:
                alias["llm_context_review"] = deepcopy(review)
            _count_review(self._counts, review)
        self._report(bundle, outcome.reviews, record, outcome)
        return {}

    def _report(
        self,
        bundle: Bundle,
        reviews: list[dict[str, Any]],
        record: dict[str, Any],
        outcome: _ReviewOutcome,
    ) -> None:
        if not self._progress:
            return
        total = len(self._targets)
        first, last = bundle.indices[0] + 1, bundle.indices[-1] + 1
        span = f"{first}" if bundle.size == 1 else f"{first}-{last}"
        label = _label(record)
        if outcome.splits:
            recovered = sum(
                1 for review in reviews if not is_failed_llm_review(review)
            )
            self._progress(
                f"[{span}/{total}] LLM bundle recovery: {label}; split "
                f"{len(outcome.splits)} affected request(s), recovered {recovered}/{bundle.size} "
                "statistics, and kept every raw attempt in the Stage 2c audit."
            )
        failed = [
            (index, review)
            for index, review in zip(bundle.indices, reviews, strict=True)
            if is_failed_llm_review(review)
        ]
        if failed:
            for index, review in failed:
                detail = review_location(self._targets[index][1])
                self._progress(
                    f"[{index + 1}/{total}] LLM context check FAILED: {label} "
                    f"({review['error']}). Kept unchecked; listed for human review. [{detail}]"
                )
            if len(failed) == bundle.size:
                return
        if bundle.size == 1:
            detail = review_location(self._targets[bundle.indices[0]][1])
            self._progress(f"[{span}/{total}] LLM context check: {label} [{detail}]")
        else:
            self._progress(f"[{span}/{total}] LLM context check: {label} ({bundle.size} statistics)")
