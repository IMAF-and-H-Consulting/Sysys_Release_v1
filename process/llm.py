"""Unified LLM client for Ollama, OpenAI, Anthropic, and custom endpoints.

A custom endpoint speaks one of three formats (see ``process.custom_endpoint``)
and reuses the matching chat path: OpenAI-compatible, Anthropic-compatible,
or LM Studio's native REST API.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import CancelledError
from typing import Any

from process import llm_clients as _llm_clients
from process.llm_cache import (
    _llm_cache_delete,
    _llm_cache_key,
    _llm_cache_read,
    _llm_cache_write,
)
from process.llm_clients import (
    _append_llm_debug,
    _append_thinking_log,
    _custom_format,
    _effective_base_url,
    _get_custom_openai_client,
    _get_ollama_client,
    _get_openai_client,
    _llm_cache_dir,
    _provider_label,
    _response_debug_fields,
    config_snapshot,
    set_config,
)
from process.llm_format import (
    _anthropic_thinking,
    _extract_message_text,
    _extract_reasoning,
    _is_reasoning_model,
    _openai_reasoning_effort,
    _status_icon,
    _supports_anthropic_thinking,
)
from process.llm_context import OutputLimitExceededError, output_limit_error
from process.llm_budget import (
    TokenBudget,
    plan_token_budget,
    resolve_model_capacity,
)
from process.llm_parsing import (
    compress_prompt_text,
    parse_llm_json,
)
from process.llm_providers import (
    _chat_anthropic,
    _chat_lmstudio,
)
from process.llm_status import (
    _executor,
    _print_llm_status,
    _print_lock,
    _wait_with_heartbeat,
)

logger = logging.getLogger(__name__)


def chat(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0,
    max_tokens: int | None = None,
    think: bool | str | None = None,
    *,
    cancel_event=None,
    task: str = "general",
    expected_items: int = 1,
    context_tokens: int | None = None,
    token_budget: TokenBudget | None = None,
    explicit_limit_reason: str = "",
) -> str:
    """
    Send a chat completion request to the configured LLM provider.
    Returns the assistant message content.

    Responses are cached on disk keyed by the full request (provider, model,
    parameters, and messages), so re-running a stage with unchanged prompts
    returns instantly without hitting the provider again. The cache is safe
    to delete at any time and can be disabled with
    ``config.llm_cache_enabled = False``. Empty responses are never cached;
    provider-reported output-limit responses raise
    :class:`~process.llm_context.OutputLimitExceededError` before the cache
    write and therefore cannot become persistent partial answers.

    The config active when the call starts is used for the whole call, so a
    concurrent ``set_config`` cannot redirect its logs or cache. A set
    ``cancel_event`` raises ``CancelledError`` while the request is queued,
    backing off or in flight, on every provider route.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("LLM request cancelled before capacity discovery")
    with config_snapshot() as config:
        if config is None:
            raise RuntimeError("llm.set_config(config) must be called before chat()")
        route = _custom_format() or config.llm_provider
        budget = token_budget or plan_token_budget(
            task,
            resolve_model_capacity(
                config,
                requested_context_tokens=context_tokens,
                model=model,
            ),
            messages=messages,
            expected_items=expected_items,
            explicit_output_tokens=max_tokens if max_tokens and max_tokens > 0 else None,
            explicit_reason=explicit_limit_reason,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("LLM request cancelled after capacity discovery")
        _append_llm_debug({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": "token_budget",
            "model": model,
            "provider": route,
            **budget.as_dict(),
        })
        return _chat_with_cache(
            route, model, messages, temperature, budget.max_output_tokens, think, cancel_event,
        )


def forget_cached_reply(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0,
    max_tokens: int | None = None,
    think: bool | str | None = None,
    *,
    task: str = "general",
    expected_items: int = 1,
    context_tokens: int | None = None,
    token_budget: TokenBudget | None = None,
    explicit_limit_reason: str = "",
) -> None:
    """Drop the cached reply to this exact request (same arguments as ``chat``).

    Call it when a reply turned out unusable (cut off, malformed, incomplete),
    so asking again reaches the model instead of replaying the stored text.
    """
    with config_snapshot() as config:
        if config is None:
            return
        cache_dir = _llm_cache_dir()
        if cache_dir is None:
            return
        route = _custom_format() or config.llm_provider
        budget = token_budget or plan_token_budget(
            task,
            resolve_model_capacity(
                config,
                requested_context_tokens=context_tokens,
                model=model,
            ),
            messages=messages,
            expected_items=expected_items,
            explicit_output_tokens=max_tokens if max_tokens and max_tokens > 0 else None,
            explicit_reason=explicit_limit_reason,
        )
        _llm_cache_delete(cache_dir, _request_cache_key(route, model, messages, temperature,
                                                        budget.max_output_tokens, think))


def _request_cache_key(route: str, model: str, messages: list[dict[str, str]],
                       temperature: float, max_tokens: int, think: bool | str | None) -> str:
    """Cache key for one request; shared by ``chat`` and ``forget_cached_reply``."""
    return _llm_cache_key(route, model, temperature, max_tokens, think, messages,
                          _effective_base_url())


def _chat_with_cache(
    route: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    think: bool | str | None,
    cancel_event: Any,
) -> str:
    cache_dir = _llm_cache_dir()
    cache_key = None
    if cache_dir is not None:
        cache_key = _request_cache_key(route, model, messages, temperature, max_tokens, think)
        hit = _llm_cache_read(cache_dir, cache_key)
        if hit is not None:
            _append_llm_debug({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": model,
                "provider": route,
                "elapsed_s": 0,
                "cached": True,
                "request_preview": (messages[-1].get("content", "") or "")[-200:],
            })
            logger.debug("LLM cache hit: model=%s key=%s", model, cache_key[:12])
            return hit
    # Only pass the event when there is one, so simple stand-ins for
    # _chat_uncached (tests) keep working without a cancel parameter.
    extra = {} if cancel_event is None else {"cancel_event": cancel_event}
    text = _chat_uncached(route, model, messages, temperature, max_tokens, think, **extra)
    if cache_dir is not None and cache_key is not None and text and text.strip():
        _llm_cache_write(cache_dir, cache_key, text)
    return text


def _chat_uncached(
    route: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    think: bool | str | None,
    cancel_event: Any = None,
) -> str:
    """Dispatch one chat request to the configured provider (no cache)."""
    providers = {
        "openai": _chat_openai,
        "lmstudio": _chat_lmstudio,
        "anthropic": _chat_anthropic,
        "ollama": _chat_ollama,
    }
    provider = providers.get(route)
    if provider is None:
        raise RuntimeError(
            f"Unsupported LLM provider/format {route!r}; expected one of "
            f"{', '.join(sorted(providers))} (custom endpoints: openai, anthropic, lmstudio)."
        )
    return provider(model, messages, temperature, max_tokens, think, cancel_event=cancel_event)


def _ollama_request(client, cancel_event, **kwargs):
    """Run one blocking Ollama call on the shared pool behind the heartbeat wait."""
    t0 = time.monotonic()
    future = _executor.submit(client.chat, **kwargs)
    return _wait_with_heartbeat(future, t0, cancel_event=cancel_event)


def _chat_ollama(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    think: bool | str | None,
    cancel_event: Any = None,
) -> str:
    client = _get_ollama_client()
    if client is None:
        raise RuntimeError("Ollama is configured but client failed to initialize.")

    t0 = time.monotonic()
    options = {"temperature": temperature, "num_predict": max_tokens}
    try:
        response = _ollama_request(
            client, cancel_event, model=model, messages=messages, think=think, options=options,
        )
    except CancelledError:
        raise
    except Exception as e:
        err_text = str(e)
        if (
            "Unexpected endpoint or method" in err_text
            and "/api/chat" in err_text
        ):
            raise RuntimeError(
                "The configured host is not speaking the Ollama API at /api/chat. "
                "If this is LM Studio, switch the provider to 'LM Studio (Native API)' "
                "and set the Stage 2 abstract profile to LM Studio as well."
            ) from e
        if think is not None and "think" in str(e).lower():
            logger.warning("Ollama server rejected think=%r; retrying without exposed reasoning", think)
            response = _ollama_request(client, cancel_event, model=model, messages=messages,
                                       options=options)
            think = None
        else:
            raise
    elapsed = time.monotonic() - t0

    message = getattr(response, "message", None)
    raw_content = getattr(message, "content", None)
    text = raw_content.strip() if isinstance(raw_content, str) else _extract_message_text(message)
    finish_reason = getattr(response, "done_reason", None)
    eval_count = getattr(response, "eval_count", None)
    prompt_eval_count = getattr(response, "prompt_eval_count", None)
    reasoning = _extract_reasoning(message) if message is not None else ""
    last_msg = (messages[-1] if messages else {}).get("content", "") or ""

    _append_llm_debug({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "provider": "ollama",
        "elapsed_s": round(elapsed, 2),
        "request_preview": last_msg[-500:],
        "raw_content": raw_content if isinstance(raw_content, str) else repr(raw_content),
        **_response_debug_fields(text),
        "finish_reason": finish_reason,
        "completion_tokens": eval_count,
        "prompt_tokens": prompt_eval_count,
        "think": think,
    })

    if reasoning:
        _append_thinking_log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "provider": "ollama",
            "elapsed_s": round(elapsed, 2),
            "request_preview": last_msg[:1000],
            "answer": raw_content if isinstance(raw_content, str) else repr(raw_content),
            "reasoning": reasoning,
            "reasoning_tokens": len(reasoning.split()),
            "finish_reason": finish_reason,
            "completion_tokens": eval_count,
            "prompt_tokens": prompt_eval_count,
            "think": think,
        })
    elif think:
        _append_thinking_log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "provider": "ollama",
            "elapsed_s": round(elapsed, 2),
            "request_preview": last_msg[:1000],
            "answer": raw_content if isinstance(raw_content, str) else repr(raw_content),
            "reasoning": "(requested but not exposed by Ollama response)",
            "reasoning_tokens": 0,
            "finish_reason": finish_reason,
            "completion_tokens": eval_count,
            "prompt_tokens": prompt_eval_count,
            "think": think,
        })

    limit_error = output_limit_error(
        provider="ollama",
        model=model,
        max_output_tokens=max_tokens,
        finish_reason=finish_reason,
        output_tokens=eval_count,
        partial_text=text,
    )
    if limit_error is not None:
        logger.error("Token action: discard non-cacheable truncated reply: %s", limit_error)
        raise limit_error

    return text


def _chat_openai(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    think: bool | str | None,
    cancel_event: Any = None,
) -> str:
    provider = _provider_label()
    is_custom = _custom_format() is not None
    client = _get_custom_openai_client() if is_custom else _get_openai_client()
    if client is None:
        key_name = "custom_base_url" if is_custom else "openai_api_key"
        raise RuntimeError(
            f"{provider} is configured but client failed to initialize. Set {key_name}."
        )
    last_err = None
    # Reasoning/thinking wiring
    is_reasoning = _is_reasoning_model(model)
    effort = _openai_reasoning_effort(think)
    for attempt in range(4):
        t0 = time.monotonic()
        try:
            # Build provider-specific kwargs
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if is_reasoning:
                # OpenAI o-series requires temperature=1 and uses
                # max_completion_tokens instead of max_tokens.
                kwargs["temperature"] = 1
                kwargs["max_completion_tokens"] = max_tokens
                if effort:
                    kwargs["reasoning_effort"] = effort
            else:
                kwargs["temperature"] = temperature
                kwargs["max_tokens"] = max_tokens
                if effort:
                    logger.warning(
                        "thinking requested but model %r is not a known reasoning model; "
                        "ignoring reasoning_effort",
                        model,
                    )

            # Submit blocking call to thread so we can print heartbeats while waiting
            future = _executor.submit(
                client.chat.completions.create,
                **kwargs,
            )
            resp = _wait_with_heartbeat(future, t0, cancel_event=cancel_event)

            elapsed = time.monotonic() - t0
            choice = (resp.choices or [None])[0]
            if not choice or choice.message is None:
                logger.warning(
                    "%s returned no choices or message (%.1fs): choices=%s",
                    provider.title(),
                    elapsed,
                    [getattr(c, "message", c) for c in (resp.choices or [])],
                )
                return ""
            text = _extract_message_text(choice.message)
            raw_content = getattr(choice.message, "content", None)
            finish_reason = getattr(choice, "finish_reason", None)
            usage = getattr(resp, "usage", None)
            comp_tok = getattr(usage, "completion_tokens", None)
            prompt_tok = getattr(usage, "prompt_tokens", None)

            icon = _status_icon(finish_reason, raw_content)
            answer_preview = (raw_content or "")[:60].replace("\n", " ") if isinstance(raw_content, str) else ""

            # Compact one-liner: ✓ 12.3s 847tok → "4"  |  ⏱ 8.1s 1024tok → (thinking, no answer)
            _print_llm_status(
                icon, elapsed, comp_tok,
                f"fin={finish_reason or '?':<6}", answer_preview,
            )

            # Stream to data/llm_responses.jsonl so you can double-check returns
            last_msg = (messages[-1] if messages else {}).get("content", "") or ""
            _append_llm_debug({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": model,
                "provider": provider,
                "elapsed_s": round(elapsed, 2),
                "request_preview": last_msg[-500:],  # tail captures paper title/abstract
                "raw_content": raw_content if isinstance(raw_content, str) else repr(raw_content),
                **_response_debug_fields(text),
                "finish_reason": finish_reason,
                "completion_tokens": comp_tok,
                "prompt_tokens": prompt_tok,
            })

            # Save reasoning chain for hallucination auditing
            reasoning = _extract_reasoning(choice.message)
            if reasoning:
                _append_thinking_log({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "model": model,
                    "provider": provider,
                    "elapsed_s": round(elapsed, 2),
                    "request_preview": last_msg[:1000],
                    "answer": raw_content if isinstance(raw_content, str) else repr(raw_content),
                    "reasoning": reasoning,
                    "reasoning_tokens": len(reasoning.split()),
                    "finish_reason": finish_reason,
                })
            elif comp_tok and comp_tok > 100 and (not raw_content or len(str(raw_content)) < 20):
                # Model used lots of tokens but answer is tiny — reasoning was likely
                # consumed internally. Log a note so we know it happened.
                _append_thinking_log({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "model": model,
                    "provider": provider,
                    "elapsed_s": round(elapsed, 2),
                    "request_preview": last_msg[:1000],
                    "answer": raw_content if isinstance(raw_content, str) else repr(raw_content),
                    "reasoning": "(not exposed by API — consumed internally)",
                    "reasoning_tokens": comp_tok,
                    "finish_reason": finish_reason,
                    "note": "Model used many tokens but reasoning not in response object. "
                            "API may not expose thinking chain for this model.",
                })

            limit_error = output_limit_error(
                provider=provider,
                model=model,
                max_output_tokens=max_tokens,
                finish_reason=finish_reason,
                output_tokens=comp_tok,
                partial_text=text,
            )
            if limit_error is not None:
                logger.error("Token action: discard non-cacheable truncated reply: %s", limit_error)
                raise limit_error
            if not text:
                # No answer content (e.g. content=None with only a reasoning
                # field). The reasoning is never used as the answer; returning
                # "" makes callers treat this as a failed call and skips caching.
                logger.warning(
                    "%s returned no answer content (finish_reason=%s, %.1fs); treating as empty reply",
                    provider.title(), finish_reason, elapsed,
                )
            return text
        except (CancelledError, OutputLimitExceededError):
            raise
        except Exception as e:
            elapsed = time.monotonic() - t0
            last_err = e
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            if status == 429:
                wait = 2 ** attempt
                with _print_lock:
                    print(f"  ⚠ {elapsed:5.1f}s  429 rate-limited, retrying in {wait}s...")
                _sleep_unless_cancelled(wait, cancel_event)
                continue
            # Connection/timeout errors
            err_name = type(e).__name__
            with _print_lock:
                print(f"  ✗ {elapsed:5.1f}s  {err_name}: {str(e)[:80]}")
            raise
    raise last_err or RuntimeError(f"{provider.title()} request failed after retries")


def _sleep_unless_cancelled(seconds: float, cancel_event: Any) -> None:
    """Back off for ``seconds``; a set ``cancel_event`` ends the wait early."""
    if cancel_event is None:
        time.sleep(seconds)
    elif cancel_event.wait(seconds):
        raise CancelledError("LLM request cancelled during rate-limit backoff")


_LIVE_CLIENT_STATE = frozenset({
    '_config',
    '_openai_client',
    '_custom_openai_client',
    '_custom_anthropic_client',
    '_ollama_client',
    '_lmstudio_client',
    '_anthropic_client',
})


def __getattr__(name: str) -> Any:
    """Serve the mutable runtime state from its owner so reads stay current.

    ``set_config`` rebinds ``_config`` and the client globals in
    ``process.llm_clients``; a copied binding here would go stale.
    """
    if name in _LIVE_CLIENT_STATE:
        return getattr(_llm_clients, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Public API, plus the helpers the test suite reaches through this module.
# Other internals live in process.llm_* and are imported from there.
__all__ = [
    "_anthropic_thinking",
    "_extract_message_text",
    "_get_openai_client",
    "_is_reasoning_model",
    "_llm_cache_key",
    "_openai_reasoning_effort",
    "_status_icon",
    "_supports_anthropic_thinking",
    "chat",
    "compress_prompt_text",
    "forget_cached_reply",
    "parse_llm_json",
    "set_config",
]
