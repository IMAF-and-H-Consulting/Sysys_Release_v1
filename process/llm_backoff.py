"""Shared, bounded backoff for concurrent LM Studio requests."""
from __future__ import annotations

import math
import random
import threading
import time
import weakref
from collections import deque
from concurrent.futures import CancelledError
from contextlib import contextmanager
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from core.checkpoint import queued_for_shared_slot

MAX_ATTEMPTS = 4
BASE_DELAY = 2.0
MAX_DELAY = 30.0
SLOT_POLL_SECONDS = 0.1  # how often a queued request re-checks cancellation


def retryable(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError)):
        return True
    response = getattr(exc, "response", None)
    if response is None:
        return False
    if response.status_code in {408, 429, 500, 502, 503, 504}:
        return True
    # LM Studio reports a cancelled model load as invalid_request / HTTP 400.
    return response.status_code == 400 and "failed to load model" in response.text.lower()


def retry_after(exc: Exception) -> float:
    response = getattr(exc, "response", None)
    if response is None:
        return 0.0
    value = response.headers.get("Retry-After", "")
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return 0.0
    return max(0.0, delay) if math.isfinite(delay) else 0.0


class SharedBackoff:
    """New windows and retries respect the same model cooldown.

    A success from a request started before a newer failure must not clear that
    failure's cooldown. Generations distinguish those in-flight requests.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._slot = threading.Condition()
        self._slot_busy = False
        self._slot_queue: deque[object] = deque()
        self._deadline = 0.0
        self._failures = 0
        self._generation = 0

    @property
    def queued_requests(self) -> int:
        """Number of requests waiting for the slot."""
        with self._slot:
            return len(self._slot_queue)

    @contextmanager
    def request_slot(self, cancel_event=None):
        """Hold the queue until the actual response (including retries) arrives.

        Waiters are served first-come, first-served, so concurrent workers get
        the slot in submission order. The queued time is reported to
        ``run_sliding_window`` so it does not count as a worker timeout.
        """
        with queued_for_shared_slot():
            self._acquire_slot(cancel_event)
        try:
            yield
        finally:
            with self._slot:
                self._slot_busy = False
                self._slot.notify_all()

    def _acquire_slot(self, cancel_event) -> None:
        ticket = object()
        with self._slot:
            self._slot_queue.append(ticket)
            try:
                while self._slot_busy or self._slot_queue[0] is not ticket:
                    if cancel_event is not None and cancel_event.is_set():
                        raise CancelledError("LLM request cancelled while queued")
                    self._slot.wait(SLOT_POLL_SECONDS)
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError("LLM request cancelled while queued")
            except BaseException:
                self._slot_queue.remove(ticket)
                self._slot.notify_all()
                raise
            self._slot_queue.popleft()
            self._slot_busy = True

    def wait(self, cancel_event=None) -> int:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("LLM request cancelled during backoff")
            with self._lock:
                remaining = self._deadline - time.monotonic()
                if remaining <= 0:
                    return self._generation
            if cancel_event is None:
                time.sleep(remaining)
            elif cancel_event.wait(remaining):
                raise CancelledError("LLM request cancelled during backoff")

    def failed(self, exc: Exception) -> float:
        with self._lock:
            self._failures += 1
            self._generation += 1
            base = min(MAX_DELAY, BASE_DELAY * 2 ** min(self._failures - 1, 10))
            delay = max(min(MAX_DELAY, base + random.uniform(0, base * 0.25)), retry_after(exc))
            self._deadline = max(self._deadline, time.monotonic() + delay)
            return self._deadline - time.monotonic()

    def succeeded(self, generation: int) -> None:
        with self._lock:
            if generation == self._generation:
                self._failures = 0
                self._deadline = 0.0


_backoffs: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_backoffs_lock = threading.Lock()


def for_client(client, model: str) -> SharedBackoff:
    # Client ownership scopes cooldowns to an endpoint/configuration and avoids
    # retaining old clients after set_config() closes and replaces them.
    with _backoffs_lock:
        models = _backoffs.setdefault(client, {})
        return models.setdefault(model, SharedBackoff())
