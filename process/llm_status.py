"""Console status lines and the heartbeat wait used by blocking provider calls."""
from __future__ import annotations

import logging
import sys
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from typing import Any

logger = logging.getLogger("process.llm")


# Heartbeat: print a dot every HEARTBEAT_INTERVAL seconds while waiting,
# and a warning message after HEARTBEAT_WARN seconds.
HEARTBEAT_INTERVAL = 5      # seconds between "still thinking" dots


HEARTBEAT_WARN = 30         # seconds before printing a warning


HEARTBEAT_CRITICAL = 90     # seconds before printing a "likely stuck" warning


CANCEL_POLL_SECONDS = 0.25  # how often an in-flight request re-checks cancellation


def _print_llm_status(
    icon: str,
    elapsed: float,
    tokens: int | None,
    detail: str,
    preview: str,
) -> None:
    """Compact one-line LLM call status (thread-safe).

    ``detail`` is a provider-specific middle field, e.g. ``fin=stop`` or
    ``tps=42.0``; ``preview`` is the first 60 chars of the answer.
    """
    with _print_lock:
        print(
            f"  {icon} {elapsed:5.1f}s"
            f"  {tokens or '?':>5}tok"
            f"  {detail}"
            f"  → {preview or '(empty)'}"
        )


class _Heartbeat:
    """Dots and slow-response warnings printed while one request is in flight."""

    def __init__(self, t0: float) -> None:
        self._t0 = t0
        self._next_beat = time.monotonic() + HEARTBEAT_INTERVAL
        self._warned = False
        self._critical_warned = False
        self._dots_printed = 0

    def tick(self) -> None:
        """Print a dot (and a warning when due) once per HEARTBEAT_INTERVAL."""
        if time.monotonic() < self._next_beat:
            return
        self._next_beat += HEARTBEAT_INTERVAL
        elapsed = time.monotonic() - self._t0
        with _print_lock:
            sys.stdout.write("·")
            sys.stdout.flush()
        self._dots_printed += 1
        if elapsed >= HEARTBEAT_CRITICAL and not self._critical_warned:
            with _print_lock:
                print(f"\n  ⚠ {elapsed:.0f}s — response is taking unusually long, may be stuck")
            self._critical_warned = True
            self._warned = True
        elif elapsed >= HEARTBEAT_WARN and not self._warned:
            with _print_lock:
                print(f"\n  💭 {elapsed:.0f}s — still thinking (this model reasons before answering)...")
            self._warned = True

    def finish(self) -> None:
        """End the dots line, if any dots were printed."""
        with _print_lock:
            if self._dots_printed > 0:
                print()


def _raise_if_cancelled(future: Future, cancel_event: Any, heartbeat: _Heartbeat) -> None:
    if cancel_event is None or not cancel_event.is_set():
        return
    # A request that already reached the server keeps running in the pool;
    # its reply is discarded (and never cached) because nobody collects it.
    future.cancel()
    heartbeat.finish()
    raise CancelledError("LLM request cancelled while waiting for the response")


def _wait_with_heartbeat(future: Future, t0: float, cancel_event: Any = None) -> Any:
    """Wait for a Future to complete, printing heartbeat messages while it runs.

    Prints a dot every HEARTBEAT_INTERVAL seconds, a warning after HEARTBEAT_WARN,
    and a critical warning after HEARTBEAT_CRITICAL. When ``cancel_event`` is
    set, it is polled every CANCEL_POLL_SECONDS and a set event raises
    ``CancelledError`` without waiting for the response.
    """
    heartbeat = _Heartbeat(t0)
    poll = HEARTBEAT_INTERVAL if cancel_event is None else CANCEL_POLL_SECONDS
    while True:
        _raise_if_cancelled(future, cancel_event, heartbeat)
        try:
            result = future.result(timeout=poll)
        except TimeoutError:
            if future.done():  # the call itself raised TimeoutError
                heartbeat.finish()
                raise
            heartbeat.tick()
            continue
        except BaseException:
            heartbeat.finish()
            raise
        heartbeat.finish()
        return result


# Thread pool for running blocking API calls (supports concurrent LLM requests).
# Sized for the sliding-window workers used by the pipeline stages (default 4
# concurrent callers); each chat() submits exactly one blocking SDK call, so
# the pool must be at least as large as the largest worker count or callers
# queue behind each other.
_executor = ThreadPoolExecutor(max_workers=8)


# Lock to prevent concurrent status/heartbeat lines from garbling each other
_print_lock = threading.Lock()
