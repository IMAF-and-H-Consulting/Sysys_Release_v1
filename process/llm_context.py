"""Typed failures for requests that exceed an LLM's token capacity.

Context overflow and response truncation are deliberately different errors:
the former means the request did not fit, while the latter means the provider
started answering but stopped at the requested output allowance.  A truncated
answer is diagnostic evidence only.  It must never be returned as a normal
reply or written to the response cache.
"""
from __future__ import annotations

import hashlib
import re


class ContextWindowExceededError(RuntimeError):
    """The complete request cannot fit in the model's loaded context."""


class OutputLimitExceededError(RuntimeError):
    """The provider stopped generation at the requested response allowance.

    Only a short preview and a digest are retained.  The partial answer is not
    exposed as a usable reply, which makes accidental downstream parsing much
    harder while still leaving enough information to diagnose the request.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        max_output_tokens: int,
        output_tokens: int | None,
        finish_reason: str | None,
        partial_text: str,
    ) -> None:
        clean = str(partial_text or "")
        self.provider = str(provider or "unknown")
        self.model = str(model or "unknown")
        self.max_output_tokens = int(max_output_tokens)
        self.output_tokens = int(output_tokens) if output_tokens is not None else None
        self.finish_reason = str(finish_reason or "output_token_limit")
        self.partial_text_chars = len(clean)
        self.partial_text_sha256 = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        # This is deliberately a diagnostic preview, not the partial response.
        self.partial_text_preview = " ".join(clean.split())[:240]
        used = (
            f"{self.output_tokens:,}" if self.output_tokens is not None
            else "an unknown number of"
        )
        super().__init__(
            f"{self.provider} model '{self.model}' stopped at its requested output "
            f"allowance ({used} of {self.max_output_tokens:,} tokens; "
            f"finish_reason={self.finish_reason!r}). The partial response "
            f"({self.partial_text_chars:,} characters, sha256 "
            f"{self.partial_text_sha256[:12]}) was discarded and was not cached. "
            "Increase the context-aware response allowance or split the task into "
            "smaller complete requests before retrying."
        )


_OUTPUT_LIMIT_REASONS = frozenset({
    "length",
    "max_tokens",
    "max_token",
    "max_output_tokens",
    "token_limit",
    "output_token_limit",
    "output_tokens",
    "truncated",
})


def output_limit_error(
    *,
    provider: str,
    model: str,
    max_output_tokens: int,
    finish_reason: object = None,
    output_tokens: object = None,
    partial_text: str = "",
    infer_from_token_count: bool = False,
) -> OutputLimitExceededError | None:
    """Return a typed truncation error when provider metadata proves it.

    ``infer_from_token_count`` is intentionally opt-in.  LM Studio's native
    endpoint does not always expose a finish reason, but does report total
    output tokens; reaching the exact requested allowance is its reliable
    truncation signal.  Other providers have explicit finish/stop reasons, so
    an exact count alone is not interpreted for them.
    """
    reason = str(finish_reason or "").strip().casefold().replace("-", "_")
    try:
        used = int(output_tokens) if output_tokens is not None else None
    except (TypeError, ValueError):
        used = None
    requested = max(1, int(max_output_tokens))
    reached = reason in _OUTPUT_LIMIT_REASONS
    if infer_from_token_count and used is not None and used >= requested:
        reached = True
    if not reached:
        return None
    return OutputLimitExceededError(
        provider=provider,
        model=model,
        max_output_tokens=requested,
        output_tokens=used,
        finish_reason=reason or "output_token_count_reached_limit",
        partial_text=partial_text,
    )


def context_window_error(
    exc: Exception, *, model: str, max_output_tokens: int,
) -> ContextWindowExceededError | None:
    """Read LM Studio errors, including an engine HTTP 400 wrapped in HTTP 500.

    Inspect the complete body: the useful engine message/counts may occur after
    the provider's usual 500-character diagnostic preview. Escaped quotes arise
    when the inner engine JSON is stored in the outer error's message string.
    """
    response = getattr(exc, "response", None)
    detail = response.text if response is not None else str(exc)
    detail = detail.replace('\\"', '"').lower()
    markers = (
        "exceed_context_size_error", "context_length_exceeded",
        "exceeds the available context size", "exceeds the context window",
        "exceeds the model's maximum context length",
    )
    if not any(marker in detail for marker in markers):
        return None

    def count(*patterns: str) -> int | None:
        for pattern in patterns:
            match = re.search(pattern, detail)
            if match:
                return int(match.group(1).replace(",", ""))
        return None

    requested = count(r'"n_prompt_tokens"\s*:\s*([\d,]+)',
                      r'request\s*\(([\d,]+) tokens\)')
    available = count(r'"n_ctx"\s*:\s*([\d,]+)',
                      r'available context size\s*\(([\d,]+) tokens\)')
    parts = [f"Context window too small for LM Studio model '{model}'."]
    if requested is not None:
        parts.append(f"The request contains {requested:,} input tokens.")
    if available is not None:
        parts.append(f"The server reports {available:,} available context tokens.")
    parts.append(
        "Increase the model's loaded context length in LM Studio, or select a "
        "model with a larger context, then run this stage again. "
        f"Allow room for the requested response budget of {max_output_tokens:,} "
        "tokens as well as the input. "
        "The request was stopped without shortening the prompt or reducing the "
        "response budget. Splitting the evidence into separate responses can "
        "change the synthesis and cannot guarantee equivalent detail."
    )
    return ContextWindowExceededError(" ".join(parts))
