"""Shared HTTP retry policy and secret redaction for the provider clients.

One retry helper (audit 02-M11), used by every provider client (OpenAlex,
Semantic Scholar, PubMed, Scopus, Web of Science and Elicit — Stage 1
review round 1, item 5):

* 429 and 5xx responses are retried, honouring ``Retry-After`` but capped;
* other 4xx responses (401, 403, 400, ...) are returned at once, never retried;
* transport errors (timeouts, resets) are retried;
* back-off is measured in seconds, not fractions of the polite request delay.

``redact_secrets`` (audit 02-M6) masks API keys and contact e-mails that some
providers take as query parameters, so request URLs embedded in exception text
never reach the on-screen or exported log, the search manifest, or
``SearchResult.provider_errors``.
"""
from __future__ import annotations

import asyncio
import logging
import re
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BASE_BACKOFF = 1.0
DEFAULT_MAX_BACKOFF = 30.0
DEFAULT_MAX_RETRY_AFTER = 60.0
REDACTED = "REDACTED"

_SECRET_PARAM_RE = re.compile(
    r"(?i)\b(api_?key|apikey|access_token|token|insttoken|email|mailto)=([^&\s'\"<>]+)"
)
_MAILTO_RE = re.compile(r"(?i)\bmailto:[^\s&'\"<>]+")


@dataclass(frozen=True)
class RetryPolicy:
    """How often and how long to retry one HTTP request."""

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_backoff: float = DEFAULT_BASE_BACKOFF
    max_backoff: float = DEFAULT_MAX_BACKOFF
    max_retry_after: float = DEFAULT_MAX_RETRY_AFTER
    retry_statuses: frozenset[int] = field(default=RETRYABLE_STATUSES)

    def backoff(self, attempt: int) -> float:
        """Exponential back-off in seconds for the zero-based *attempt*."""
        return min(self.max_backoff, self.base_backoff * (2 ** attempt))

    def wait_for(self, response: httpx.Response, attempt: int) -> float:
        """Seconds to wait before retrying *response*: capped Retry-After or back-off."""
        retry_after = parse_retry_after(response.headers.get("Retry-After"))
        if retry_after is None:
            return self.backoff(attempt)
        return min(retry_after, self.max_retry_after)


DEFAULT_POLICY = RetryPolicy()


def parse_retry_after(value) -> float | None:
    """Parse a delta-seconds ``Retry-After`` value; anything else gives None."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        return None


def redact_secrets(text) -> str:
    """Return *text* with API keys, tokens and contact e-mails masked."""
    masked = _SECRET_PARAM_RE.sub(lambda m: f"{m.group(1)}={REDACTED}", str(text))
    return _MAILTO_RE.sub(f"mailto:{REDACTED}", masked)


def describe_error(exc: BaseException) -> str:
    """``"<Type>: <message>"`` for *exc*, with secrets redacted."""
    return redact_secrets(f"{type(exc).__name__}: {exc}")


def redacted_traceback(exc: BaseException) -> str:
    """The formatted traceback of *exc*, with secrets redacted."""
    return redact_secrets("".join(traceback.format_exception(exc)))


class RedactingFilter(logging.Filter):
    """Mask secrets in records of a third-party logger (httpx logs full URLs)."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = redact_secrets(message)
        if redacted != message:
            record.msg, record.args = redacted, ()
        return True


def install_httpx_log_redaction() -> None:
    """Attach one ``RedactingFilter`` to the ``httpx`` logger (idempotent)."""
    httpx_logger = logging.getLogger("httpx")
    if not any(isinstance(f, RedactingFilter) for f in httpx_logger.filters):
        httpx_logger.addFilter(RedactingFilter())


install_httpx_log_redaction()


def raise_for_status_redacted(response: httpx.Response) -> httpx.Response:
    """Like ``Response.raise_for_status`` but the message never carries secrets.

    The exception is raised outside any ``except`` block so no unredacted
    chained exception is printed with it.
    """
    if response.is_success:
        return response
    request = response.request
    message = redact_secrets(
        f"HTTP {response.status_code} {response.reason_phrase} "
        f"for {request.method} {request.url}"
    )
    raise httpx.HTTPStatusError(message, request=request, response=response)


async def _request_with_retry(
    make_request: Callable[[], Awaitable[httpx.Response]],
    *,
    policy: RetryPolicy,
    label: str,
) -> httpx.Response:
    """Call ``make_request()``, retrying per *policy*.

    Returns the first response whose status is not retryable, or the last
    retryable response once attempts run out (the caller decides whether to
    raise). The final transport error propagates.
    """
    attempts = max(1, policy.max_attempts)
    for attempt in range(attempts):
        is_last = attempt == attempts - 1
        try:
            response = await make_request()
        except httpx.TransportError as exc:
            if is_last:
                raise
            wait = policy.backoff(attempt)
            logger.warning("%s request failed (%s); retrying in %.1fs (attempt %d/%d)",
                           label, describe_error(exc), wait, attempt + 1, attempts)
            await asyncio.sleep(wait)
            continue
        if response.status_code not in policy.retry_statuses or is_last:
            return response
        wait = policy.wait_for(response, attempt)
        logger.warning("%s HTTP %d; retrying in %.1fs (attempt %d/%d)",
                       label, response.status_code, wait, attempt + 1, attempts)
        await asyncio.sleep(wait)
    raise AssertionError("unreachable: the last attempt always returns or raises")


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict | None = None,
    policy: RetryPolicy = DEFAULT_POLICY,
    label: str = "HTTP",
) -> httpx.Response:
    """GET *url*, retrying per *policy*. See ``_request_with_retry``."""
    return await _request_with_retry(
        lambda: client.get(url, params=params), policy=policy, label=label)


async def post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    json: dict | None = None,
    policy: RetryPolicy = DEFAULT_POLICY,
    label: str = "HTTP",
) -> httpx.Response:
    """POST *url*, retrying per *policy*. See ``_request_with_retry``."""
    return await _request_with_retry(
        lambda: client.post(url, json=json), policy=policy, label=label)
