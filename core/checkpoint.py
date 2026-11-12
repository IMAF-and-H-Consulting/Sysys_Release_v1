"""Shared checkpoint and sliding-window execution utilities.

CheckpointManager: single-file atomic checkpoint for long-running stages.
run_sliding_window: ordered concurrent execution with checkpoint integration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from core.io import atomic_write_json

logger = logging.getLogger(__name__)

DEFAULT_FUTURE_TIMEOUT = 120.0  # seconds before treating a worker as stuck
WINDOW_POLL_SECONDS = 0.25  # how often the collector re-checks cancel and deadlines


def input_fingerprint(items: list, **settings: Any) -> str:
    """Hash the ordered inputs plus the settings that shape each result.

    Pass it to ``CheckpointManager(fingerprint=...)`` so a checkpoint written
    for other inputs, models or prompts is discarded instead of resumed.
    """
    payload = json.dumps({"items": items, "settings": settings},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CheckpointManager:
    """Single-file atomic checkpoint for list-accumulating stages.

    Combines data + progress into one JSON file to eliminate the race condition
    where two separate files (partial + progress) can become inconsistent if
    the process crashes between writes.
    """

    def __init__(
        self,
        checkpoint_dir: Path | None,
        filename: str,
        checkpoint_every: int = 1,
        fingerprint: str | None = None,
    ):
        """``fingerprint`` (see ``input_fingerprint``) ties the checkpoint to
        its inputs: on load, a checkpoint with a different or missing
        fingerprint is discarded. ``None`` keeps the legacy positional resume.
        """
        self._dir = checkpoint_dir
        self._filename = filename
        self._every = checkpoint_every
        self._fingerprint = fingerprint

    @property
    def path(self) -> Path | None:
        return self._dir / self._filename if self._dir else None

    def load(self) -> tuple[list[dict], int]:
        """Load checkpoint and return (data, start_index).

        Returns ([], 0) if no checkpoint exists, it's corrupt, or it was
        written for different inputs (fingerprint mismatch).
        """
        if self._dir is None:
            return [], 0
        path = self._dir / self._filename
        if not path.exists():
            return [], 0
        try:
            with open(path, encoding="utf-8") as f:
                ckpt = json.load(f)
            if not self._fingerprint_matches(ckpt, path):
                return [], 0
            parsed = _parse_checkpoint(ckpt, path)
            if parsed is None:
                return [], 0
            data, count = parsed
            logger.info("Resuming from checkpoint: %d already processed", count)
            return data, count
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as e:
            logger.warning("Could not load checkpoint %s: %s", path, e)
            return [], 0

    def _fingerprint_matches(self, ckpt: Any, path: Path) -> bool:
        if self._fingerprint is None:
            return True
        stored = ckpt.get("fingerprint") if isinstance(ckpt, dict) else None
        if stored == self._fingerprint:
            return True
        logger.warning(
            "Discarding checkpoint %s: its input fingerprint does not match this "
            "run (inputs, model or settings changed). Starting from the beginning.",
            path,
        )
        return False

    def save(self, data: list[dict], total: int) -> bool:
        """Atomically save checkpoint if due (based on checkpoint_every)."""
        if self._dir is None or self._every <= 0:
            return False
        if len(data) % self._every != 0:
            return False
        return self._write(data, total)

    def force_save(self, data: list[dict], total: int) -> bool:
        """Atomically save checkpoint unconditionally."""
        if self._dir is None:
            return False
        return self._write(data, total)

    def cleanup(self) -> None:
        """Remove checkpoint file after successful completion."""
        if self._dir is None:
            return
        path = self._dir / self._filename
        if path.exists():
            try:
                path.unlink()
            except OSError as e:
                logger.warning("Could not remove checkpoint %s: %s", path, e)

    # Also clean up legacy two-file checkpoints from older runs
    def cleanup_legacy(self, *filenames: str) -> None:
        """Remove legacy checkpoint files (e.g. progress + partial from old format)."""
        if self._dir is None:
            return
        for name in filenames:
            p = self._dir / name
            if p.exists():
                try:
                    p.unlink()
                except OSError as e:
                    logger.warning("Could not remove legacy checkpoint %s: %s", p, e)

    def _write(self, data: list[dict], total: int) -> bool:
        if self._dir is None:
            return False
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / self._filename
        try:
            payload: dict[str, Any] = {
                "processed_count": len(data), "total": total, "data": data,
            }
            if self._fingerprint is not None:
                payload["fingerprint"] = self._fingerprint
            atomic_write_json(path, payload)
            return True
        except OSError as e:
            logger.warning("Checkpoint write failed for %s: %s", path, e)
            return False


def _parse_checkpoint(ckpt: Any, path: Path) -> tuple[list[dict], int] | None:
    """Return (data, count) from any supported checkpoint format, or None."""
    if isinstance(ckpt, list):
        data, count = ckpt, len(ckpt)
    elif isinstance(ckpt, dict) and "extracted" in ckpt:
        data = ckpt.get("extracted", [])
        count = ckpt.get("next_index", len(data))
    elif isinstance(ckpt, dict):
        data = ckpt.get("data", [])
        count = ckpt.get("processed_count", len(data))
    else:
        logger.warning(
            "Unsupported checkpoint format in %s: %s", path, type(ckpt).__name__,
        )
        return None
    if not isinstance(data, list):
        logger.warning(
            "Checkpoint data is not a list in %s: %s", path, type(data).__name__,
        )
        return None
    try:
        count = int(count)
    except (TypeError, ValueError):
        logger.warning(
            "Checkpoint processed count is invalid in %s: %r. Using data length.",
            path, count,
        )
        count = len(data)
    # Validate consistency: count must match data length
    if count != len(data):
        logger.warning(
            "Checkpoint inconsistent: processed_count=%d but data has %d items. "
            "Using data length.",
            count, len(data),
        )
        count = len(data)
    return data, count


# ── Sliding window ────────────────────────────────────────────────────────

_RUNNING = "running"
_QUEUED = "queued"

_worker_local = threading.local()


class _ItemClock:
    """Tracks when an item's timeout started counting.

    The clock starts when the worker thread actually begins the item (not
    when collection reaches it), and restarts when the worker leaves a
    ``queued_for_shared_slot`` wait. State is one tuple so reads from the
    collector thread are atomic.
    """

    def __init__(self) -> None:
        self._state = (_RUNNING, time.monotonic())

    def start(self) -> None:
        self._state = (_RUNNING, time.monotonic())

    def queue(self) -> None:
        self._state = (_QUEUED, time.monotonic())

    def deadline(self, timeout: float, queue_timeout: float) -> float:
        phase, since = self._state
        return since + (queue_timeout if phase == _QUEUED else timeout)


@contextmanager
def queued_for_shared_slot() -> Iterator[None]:
    """Mark the current worker as waiting for a shared serial resource.

    Inside ``run_sliding_window`` this wait does not count toward the item's
    ``future_timeout``; the clock restarts when the block exits, so the
    timeout measures the request itself. The wait is still bounded
    (``future_timeout`` per window slot), so a hung holder cannot block the
    stage forever. Outside ``run_sliding_window`` it is a no-op.
    """
    clock = getattr(_worker_local, "clock", None)
    if clock is None:
        yield
        return
    clock.queue()
    try:
        yield
    finally:
        clock.start()


class _Outcome(NamedTuple):
    status: str  # "ok", "failed", "timed_out" or "cancelled"
    value: Any = None
    error: BaseException | None = None


def _spawn(worker_fn: Callable, item: Any) -> tuple[Future, _ItemClock]:
    """Run one item on its own daemon thread.

    A dedicated thread per item means a hung worker never occupies a pool
    slot that later items need, and nothing ever joins it: an abandoned
    worker cannot block stage completion or interpreter exit.
    """
    future: Future = Future()
    clock = _ItemClock()

    def run() -> None:
        _worker_local.clock = clock
        clock.start()
        future.set_running_or_notify_cancel()
        try:
            future.set_result(worker_fn(item))
        except BaseException as exc:  # noqa: BLE001 - handed to the collector
            future.set_exception(exc)
        finally:
            _worker_local.clock = None

    threading.Thread(target=run, name="sliding-window-worker", daemon=True).start()
    return future, clock


def _is_set(cancel_event) -> bool:
    return bool(cancel_event is not None and cancel_event.is_set())


def _stagger_wait(seconds: float, cancel_event) -> None:
    """Pause between initial submissions; returns early when cancelled."""
    if cancel_event is None:
        time.sleep(seconds)
    else:
        cancel_event.wait(seconds)


def _await_item(future: Future, clock: _ItemClock, timeout: float,
                queue_timeout: float, cancel_event) -> _Outcome:
    """Wait for one item until it finishes, times out, or the stage is cancelled."""
    while True:
        if _is_set(cancel_event) and not future.done():
            return _Outcome("cancelled")
        remaining = clock.deadline(timeout, queue_timeout) - time.monotonic()
        if remaining <= 0 and not future.done():
            return _Outcome("timed_out")
        try:
            # exception() never confuses a worker's own TimeoutError with ours.
            error = future.exception(timeout=min(WINDOW_POLL_SECONDS, max(remaining, 0.0)))
        except FutureTimeout:
            continue
        if error is None:
            return _Outcome("ok", future.result())
        return _Outcome("failed", error=error)


def _report_failure(index: int, total: int, outcome: _Outcome, timeout: float,
                    on_failure: Callable[[int, str, BaseException | None], None] | None,
                    ) -> None:
    if outcome.status == "timed_out":
        logger.error(
            "Worker timed out after %.1fs on item %d/%d. Skipping.",
            timeout, index, total,
        )
    else:
        logger.error(
            "Worker failed on item %d/%d. Skipping. Error: %s: %s",
            index, total, type(outcome.error).__name__, outcome.error,
        )
    if on_failure:
        on_failure(index, outcome.status, outcome.error)


def run_sliding_window(
    items: list,
    worker_fn: Callable,
    result_fn: Callable[[int, Any], dict],
    start_index: int = 0,
    workers: int = 2,
    stagger: float = 2.0,
    checkpoint_fn: Callable[[], None] | None = None,
    log_fn: Callable[[int, int], None] | None = None,
    cancel_event=None,
    future_timeout: float = DEFAULT_FUTURE_TIMEOUT,
    on_failure: Callable[[int, str, BaseException | None], None] | None = None,
) -> tuple[list[dict], bool]:
    """Execute worker_fn over items with a sliding window of concurrent workers.

    Args:
        items: Full list of items to process.
        worker_fn: Callable(item) -> result for each item.
        result_fn: Callable(index, worker_result) -> dict to build output record.
            ``worker_result`` is None when the worker failed or timed out.
        start_index: Index to resume from (0-based).
        workers: Number of concurrent workers (1 runs items one at a time,
            with the same timeout and error handling).
        stagger: Seconds between initial submissions (cut short by cancel).
        checkpoint_fn: Called after each result is collected.
        log_fn: Called with (completed_count, total_count) after each result.
            ``completed_count`` counts items finished in this call only (it
            excludes the ``start_index`` resumed items); ``total_count`` is
            ``len(items)``. Add ``start_index`` for an overall position.
        cancel_event: Optional ``threading.Event``. When set, the call stops
            waiting, abandons in-flight workers and returns ``cancelled=True``
            with the results collected so far.
        future_timeout: Seconds an item may run, measured from when its worker
            starts (or leaves a ``queued_for_shared_slot`` wait). A timed-out
            worker is abandoned, never joined.
        on_failure: Called with (index, "failed" | "timed_out", error) before
            result_fn for an item without a result, so callers can keep it out
            of their checkpoint and retry it on resume.

    Returns:
        Tuple of (results, cancelled) where results is a list of dicts built by
        result_fn, in order, for items ``start_index`` onward. A failed or
        timed-out item still gets a result (built from ``None``). When
        ``start_index >= len(items)`` the result is ``([], False)``.
    """
    results: list[dict] = []
    window = max(1, min(workers, len(items) - start_index))
    queue_timeout = future_timeout * window
    in_flight: dict[int, tuple[Future, _ItemClock]] = {}
    next_submit = start_index

    # Seed the window with staggered submissions
    while next_submit < len(items) and len(in_flight) < window:
        if _is_set(cancel_event):
            return results, True
        if in_flight and stagger > 0:
            _stagger_wait(stagger, cancel_event)
            if _is_set(cancel_event):
                return results, True
        in_flight[next_submit] = _spawn(worker_fn, items[next_submit])
        next_submit += 1

    # Drain results in order, refilling as we go
    for index in range(start_index, len(items)):
        future, clock = in_flight.pop(index)
        outcome = _await_item(future, clock, future_timeout, queue_timeout, cancel_event)
        if outcome.status == "cancelled":
            return results, True
        if outcome.status != "ok":
            _report_failure(index, len(items), outcome, future_timeout, on_failure)

        results.append(result_fn(index, outcome.value))
        if checkpoint_fn:
            checkpoint_fn()
        if log_fn:
            log_fn(len(results), len(items))

        if _is_set(cancel_event):
            return results, True
        # Submit next item to keep the window full
        if next_submit < len(items):
            in_flight[next_submit] = _spawn(worker_fn, items[next_submit])
            next_submit += 1

    return results, False
