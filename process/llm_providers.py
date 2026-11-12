"""Chat implementations for LM Studio's native REST API and Anthropic-compatible endpoints."""
from __future__ import annotations

import logging
import time
from concurrent.futures import CancelledError
from typing import Any

from process.llm_backoff import MAX_ATTEMPTS, for_client, retryable
from process.llm_context import context_window_error, output_limit_error
from process.llm_clients import (
    _append_llm_debug,
    _append_thinking_log,
    _custom_format,
    _get_anthropic_client,
    _get_custom_anthropic_client,
    _get_lmstudio_client,
    _lmstudio_models_without_reasoning,
    _provider_label,
    _response_debug_fields,
)
from process.llm_format import (
    _anthropic_thinking,
    _extract_message_text,
    _extract_reasoning,
    _status_icon,
    _supports_anthropic_thinking,
)
from process.llm_status import (
    _executor,
    _print_llm_status,
    _wait_with_heartbeat,
)

logger = logging.getLogger("process.llm")


def _lmstudio_reasoning_payload(think: bool | str | None) -> str | None:
    """Translate pipeline think hints into LM Studio reasoning settings."""
    if think is None:
        return None
    if isinstance(think, str):
        effort = think.strip().lower()
        if effort in {"low", "medium", "high", "on"}:
            return effort
        if effort in {"true"}:
            return "on"
        if effort in {"false", "off"}:
            return "off"
        return "medium"
    if think:
        return "medium"
    return "off"


def _lmstudio_reasoning_unsupported(exc: Exception) -> bool:
    """Return True when LM Studio rejects the request's reasoning field."""
    response = getattr(exc, "response", None)
    error_payload = {}
    if response is not None:
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                error_payload = parsed.get("error") or {}
        except Exception:
            error_payload = {}

    if not isinstance(error_payload, dict):
        error_payload = {}

    haystack = " ".join(
        str(part or "")
        for part in (
            error_payload.get("message"),
            error_payload.get("param"),
            error_payload.get("code"),
            exc,
        )
    ).lower()

    return "reasoning" in haystack and (
        error_payload.get("param") == "reasoning"
        or "does not support reasoning" in haystack
        or "reasoning configuration" in haystack
    )


def _split_system_prompt(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Flatten simple message history into LM Studio's system_prompt + input fields."""
    system_parts = []
    input_parts = []
    for message in messages:
        role = (message.get("role") or "user").strip().lower()
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
            continue
        if role == "user" and not input_parts:
            input_parts.append(content)
            continue
        input_parts.append(f"{role.upper()}: {content}")
    return "\n\n".join(system_parts).strip(), "\n\n".join(input_parts).strip()


def _split_system_messages(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Separate system prompts from a simple role/content message list."""
    system_parts = []
    converted: list[dict[str, str]] = []
    for message in messages:
        role = (message.get("role") or "user").strip().lower()
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
            continue
        if role not in {"user", "assistant"}:
            role = "user"
            content = f"{(message.get('role') or 'user').upper()}: {content}"
        converted.append({"role": role, "content": content})
    return "\n\n".join(system_parts).strip(), converted


def _lmstudio_response(client, model, payload, think, cancel_event):
    backoff = for_client(client, model)
    with backoff.request_slot(cancel_event):
        return _request_lmstudio(client, model, payload, think, cancel_event, backoff)


def _request_lmstudio(client, model, payload, think, cancel_event, backoff):
    attempt = 0
    while True:
        generation = backoff.wait(cancel_event)
        try:
            future = _executor.submit(client.post, "/api/v1/chat", json=payload)
            response = _wait_with_heartbeat(future, time.monotonic(), cancel_event=cancel_event)
            response.raise_for_status()
            data = response.json()
        except CancelledError:
            raise
        except Exception as e:
            context_error = context_window_error(
                e, model=model, max_output_tokens=payload["max_output_tokens"],
            )
            if context_error is not None:
                logger.error("%s", context_error)
                raise context_error from e
            if "reasoning" in payload and _lmstudio_reasoning_unsupported(e):
                _lmstudio_models_without_reasoning.add(model)
                logger.warning(
                    "LM Studio model %r rejected reasoning=%r; retrying without it",
                    model, payload.get("reasoning"),
                )
                payload.pop("reasoning", None)
                think = None
                continue
            attempt += 1
            err_detail = str(e)
            err_response = getattr(e, "response", None)
            if err_response is not None:
                err_detail = f"HTTP {err_response.status_code}: {err_response.text[:500]}"
            if not err_detail.strip():
                err_detail = f"{type(e).__name__} (no message)"
            if retryable(e):
                delay = backoff.failed(e)
                if attempt < MAX_ATTEMPTS:
                    logger.warning(
                        "LM Studio %s: %s; shared cooldown %.1fs before retry %d/%d",
                        model, err_detail, delay, attempt + 1, MAX_ATTEMPTS,
                    )
                    continue
                logger.warning(
                    "LM Studio %s: retries exhausted; new requests wait for shared cooldown %.1fs",
                    model, delay,
                )
            logger.error("LM Studio request to %s failed: %s", model, err_detail)
            raise RuntimeError(f"LM Studio chat request failed: {err_detail}") from e
        else:
            backoff.succeeded(generation)
            return data, think


def _chat_lmstudio(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    think: bool | str | None,
    cancel_event=None,
) -> str:
    client = _get_lmstudio_client()
    if client is None:
        raise RuntimeError(
            "LM Studio REST endpoint is configured but client failed to initialize. "
            "Check custom_base_url."
        )

    system_prompt, input_text = _split_system_prompt(messages)
    if not input_text:
        raise RuntimeError("LM Studio request had no user input after message flattening.")

    payload = {
        "model": model,
        "input": input_text,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if system_prompt:
        payload["system_prompt"] = system_prompt
    reasoning = None
    if model not in _lmstudio_models_without_reasoning:
        reasoning = _lmstudio_reasoning_payload(think)
    if reasoning:
        payload["reasoning"] = reasoning

    t0 = time.monotonic()
    data, think = _lmstudio_response(client, model, payload, think, cancel_event)

    elapsed = time.monotonic() - t0
    output = data.get("output") or []
    if not isinstance(output, list):
        output = []

    answer_parts = []
    reasoning_parts = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = (item.get("type") or "").strip().lower()
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if item_type == "reasoning":
            reasoning_parts.append(content)
        elif item_type == "message":
            answer_parts.append(content)

    text = "\n\n".join(answer_parts).strip()
    reasoning_text = "\n\n".join(reasoning_parts).strip()
    stats = data.get("stats") or {}
    input_tokens = stats.get("input_tokens")
    total_output_tokens = stats.get("total_output_tokens")
    reasoning_output_tokens = stats.get("reasoning_output_tokens")
    ttft = stats.get("time_to_first_token_seconds")
    tokens_per_second = stats.get("tokens_per_second")
    response_id = data.get("response_id")
    incomplete = data.get("incomplete_details") or {}
    finish_reason = (
        data.get("finish_reason")
        or data.get("stop_reason")
        or (incomplete.get("reason") if isinstance(incomplete, dict) else None)
    )
    if not finish_reason:
        for item in output:
            if not isinstance(item, dict):
                continue
            details = item.get("incomplete_details") or {}
            finish_reason = (
                item.get("finish_reason")
                or item.get("stop_reason")
                or (details.get("reason") if isinstance(details, dict) else None)
            )
            if finish_reason:
                break

    icon = _status_icon(None, text)
    answer_preview = text[:60].replace("\n", " ")
    _print_llm_status(
        icon, elapsed, total_output_tokens,
        f"tps={tokens_per_second or '?':<6}", answer_preview,
    )

    _append_llm_debug({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "provider": _provider_label(),
        "elapsed_s": round(elapsed, 2),
        "request_preview": input_text[-500:],
        "system_prompt_preview": system_prompt[:500],
        "raw_output": output,
        **_response_debug_fields(text),
        "response_id": response_id,
        "input_tokens": input_tokens,
        "completion_tokens": total_output_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
        "finish_reason": finish_reason,
        "tokens_per_second": tokens_per_second,
        "time_to_first_token_seconds": ttft,
        "think": think,
    })

    if reasoning_text:
        _append_thinking_log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "provider": _provider_label(),
            "elapsed_s": round(elapsed, 2),
            "request_preview": input_text[:1000],
            "answer": text,
            "reasoning": reasoning_text,
            "reasoning_tokens": len(reasoning_text.split()),
            "response_id": response_id,
            "input_tokens": input_tokens,
            "completion_tokens": total_output_tokens,
            "reasoning_output_tokens": reasoning_output_tokens,
            "tokens_per_second": tokens_per_second,
            "time_to_first_token_seconds": ttft,
            "think": think,
        })
    elif think:
        _append_thinking_log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "provider": _provider_label(),
            "elapsed_s": round(elapsed, 2),
            "request_preview": input_text[:1000],
            "answer": text,
            "reasoning": "(requested but not exposed by LM Studio response)",
            "reasoning_tokens": 0,
            "response_id": response_id,
            "input_tokens": input_tokens,
            "completion_tokens": total_output_tokens,
            "reasoning_output_tokens": reasoning_output_tokens,
            "tokens_per_second": tokens_per_second,
            "time_to_first_token_seconds": ttft,
            "think": think,
        })

    limit_error = output_limit_error(
        provider=_provider_label(),
        model=model,
        max_output_tokens=max_tokens,
        finish_reason=finish_reason,
        output_tokens=total_output_tokens,
        partial_text=text,
        # Native LM Studio has historically omitted a finish reason.  Its
        # reported total reaching the requested allowance is the available
        # completion signal in those responses.
        infer_from_token_count=True,
    )
    if limit_error is not None:
        logger.error("Token action: discard non-cacheable truncated reply: %s", limit_error)
        raise limit_error

    return text


def _chat_anthropic(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    think: bool | str | None,
    cancel_event=None,
) -> str:
    is_custom = _custom_format() is not None
    client = _get_custom_anthropic_client() if is_custom else _get_anthropic_client()
    if client is None:
        key_name = "custom_base_url" if is_custom else "anthropic_api_key"
        raise RuntimeError(
            f"{_provider_label()} is configured but client failed to initialize. Set {key_name}."
        )

    system_prompt, anthropic_messages = _split_system_messages(messages)
    if not anthropic_messages:
        raise RuntimeError("Anthropic request had no user/assistant messages to send.")

    payload: dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
    }
    if system_prompt:
        payload["system"] = system_prompt

    thinking = None
    if _supports_anthropic_thinking(model):
        thinking = _anthropic_thinking(think, max_tokens)
        if thinking:
            payload["thinking"] = thinking
    elif think:
        logger.warning(
            "thinking requested but model %r is not a known Anthropic thinking model; "
            "ignoring thinking parameter",
            model,
        )
    # Anthropic requires temperature=1 whenever thinking is enabled.
    payload["temperature"] = 1 if thinking else temperature

    t0 = time.monotonic()
    future = _executor.submit(client.messages.create, **payload)
    resp = _wait_with_heartbeat(future, t0, cancel_event=cancel_event)

    elapsed = time.monotonic() - t0
    text = _extract_message_text(resp)
    raw_content = getattr(resp, "content", None)
    stop_reason = getattr(resp, "stop_reason", None)
    usage = getattr(resp, "usage", None)
    comp_tok = getattr(usage, "output_tokens", None)
    prompt_tok = getattr(usage, "input_tokens", None)
    reasoning = _extract_reasoning(resp)
    last_msg = (messages[-1] if messages else {}).get("content", "") or ""

    icon = _status_icon(stop_reason, raw_content)
    answer_preview = text[:60].replace("\n", " ") if text else ""
    _print_llm_status(
        icon, elapsed, comp_tok,
        f"fin={stop_reason or '?':<6}", answer_preview,
    )

    _append_llm_debug({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "provider": _provider_label(),
        "elapsed_s": round(elapsed, 2),
        "request_preview": last_msg[-500:],
        "system_prompt_preview": system_prompt[:500],
        "raw_content": repr(raw_content),
        **_response_debug_fields(text),
        "finish_reason": stop_reason,
        "completion_tokens": comp_tok,
        "prompt_tokens": prompt_tok,
        "think": think,
    })

    if reasoning:
        _append_thinking_log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "provider": _provider_label(),
            "elapsed_s": round(elapsed, 2),
            "request_preview": last_msg[:1000],
            "answer": text,
            "reasoning": reasoning,
            "reasoning_tokens": len(reasoning.split()),
            "finish_reason": stop_reason,
            "completion_tokens": comp_tok,
            "prompt_tokens": prompt_tok,
            "think": think,
        })
    elif think:
        _append_thinking_log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "provider": _provider_label(),
            "elapsed_s": round(elapsed, 2),
            "request_preview": last_msg[:1000],
            "answer": text,
            "reasoning": "(requested but not exposed by Anthropic response)",
            "reasoning_tokens": 0,
            "finish_reason": stop_reason,
            "completion_tokens": comp_tok,
            "prompt_tokens": prompt_tok,
            "think": think,
        })

    limit_error = output_limit_error(
        provider=_provider_label(),
        model=model,
        max_output_tokens=max_tokens,
        finish_reason=stop_reason,
        output_tokens=comp_tok,
        partial_text=text,
    )
    if limit_error is not None:
        logger.error("Token action: discard non-cacheable truncated reply: %s", limit_error)
        raise limit_error

    return text


# ---------------------------------------------------------------------------
# Shared LLM response parsing
# ---------------------------------------------------------------------------
