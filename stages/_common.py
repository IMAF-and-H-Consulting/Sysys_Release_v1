"""Helpers shared by every stage's ``run()``: progress reporting and cancellation."""
from __future__ import annotations

import logging
import weakref
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Callbacks whose failure was already logged with a traceback; later failures
# of the same callback are logged at DEBUG so a broken UI cannot flood the log.
_FAILED_CALLBACKS: weakref.WeakSet = weakref.WeakSet()

ProgressCallback = Callable[[Any], None]


def is_cancelled(cancel_event: Any) -> bool:
    """True when a stage should stop. Accepts a threading.Event, a bool, or None."""
    if cancel_event is None:
        return False
    if hasattr(cancel_event, "is_set"):
        return bool(cancel_event.is_set())
    return bool(cancel_event)


def deliver_progress(callback: ProgressCallback, payload: Any) -> None:
    """Hand one progress update to the GUI callback.

    The callback is GUI code (a Qt signal emit) running inside a long stage.
    If the callback fails, we log the error and continue — progress updates are
    best-effort UI feedback, not critical-path operations. The stage should
    complete its work even if the UI is temporarily broken or unresponsive.
    """
    try:
        callback(payload)
    except Exception as exc:
        # Log the failure but don't abort the stage. Progress updates are
        # non-critical; the primary work (data processing, LLM calls, file writes)
        # should continue regardless of UI state.
        if _first_failure(callback):
            logger.warning("Progress callback failed (update skipped; later "
                           "failures of this callback are logged at DEBUG): %s",
                           exc, exc_info=True)
        else:
            logger.debug("Progress callback failed again (update skipped): %s", exc)


def _first_failure(callback: ProgressCallback) -> bool:
    """True the first time ``callback`` fails; callbacks that cannot be tracked always are."""
    try:
        if callback in _FAILED_CALLBACKS:
            return False
        _FAILED_CALLBACKS.add(callback)
    except TypeError:  # not weak-referenceable (e.g. a builtin method)
        return True
    return True


# Substrings that mark a keyword argument as an LLM setting.
_LLM_KWARG_MARKERS = ("provider", "model", "api_key", "base_url", "llm", "endpoint")


def announce_no_llm(progress: Callable[[str], None], stage_label: str,
                    kw: dict) -> list[str]:
    """Tell the user plainly that a stage never contacts an LLM.

    Statistics-only stages used to sit beside a provider/model picker, which
    implied a model was doing the work. This states the truth up front and
    names any LLM settings the caller passed, since those are dropped.
    Returns the ignored keyword names.
    """
    progress(
        f"{stage_label} makes NO LLM calls. It runs deterministic statistics "
        f"on files already on disk; nothing is sent to Ollama, OpenAI, "
        f"Anthropic, or any custom endpoint."
    )
    ignored = sorted(
        k for k in kw if any(m in k.lower() for m in _LLM_KWARG_MARKERS)
    )
    if ignored:
        progress(
            f"WARNING: {stage_label} was given LLM settings it cannot use "
            f"({', '.join(ignored)}). They were IGNORED; no request was sent "
            f"to any provider."
        )
    return ignored


def make_progress(stage_logger: logging.Logger,
                  callback: ProgressCallback | None, *,
                  log_with_callback: bool = True) -> Callable[[str], None]:
    """Report progress, optionally letting a GUI callback own log delivery.

    Disable log_with_callback when the GUI callback already writes to the same
    console as its Python logging bridge. CLI runs still log normally.
    """
    def _progress(msg: str) -> None:
        if callback is None or log_with_callback:
            stage_logger.info(msg)
        if callback is not None:
            deliver_progress(callback, msg)
    return _progress
