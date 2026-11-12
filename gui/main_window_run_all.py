"""Pipeline board, stage progress, Run All chaining, and stage-worker wiring.

Split from gui/main_window.py; mixed into MainWindow.
"""
from __future__ import annotations

import logging

from gui.pipeline_board import (
    PENDING_SUMMARY,
    PipelineBoardDialog,
    collect_stage_progress,
    empty_stage_progress,
    merge_disk_progress,
)

logger = logging.getLogger("gui.main_window")

# Board row for each Run All stage key and each PipelineStageWorker stage_name.
STAGE_NUMBERS = {
    "search": 1, "score": 2, "extract": 3, "validate": 4, "audit": 5,
    "synthesize": 6, "fulltext": 7, "robustness": 8, "verify": 9,
    "compare_runs": 10,
}


def _error_summary(err: object) -> str:
    """One-line board summary for an error."""
    text = str(err).strip().replace("\n", " ")
    if len(text) > 80:
        text = text[:77] + "..."
    return text or "Failed"


def _halt_run_all(host, num: int, summary: str, msg: str) -> None:
    """Disarm Run All and mark the stage that broke the chain as failed."""
    host._log(msg, "ERROR")
    host._run_all_active = False
    host._run_all_queue.clear()
    host._set_pipeline_stage(num, "failed", summary=summary)


class RunAllMixin:
    """Pipeline board, stage progress, Run All chaining, and stage-worker wiring."""

    # ── Tab 13: Log ───────────────────────────────────────────────────

    # ═══════════════════════════════════════════════════════════════
    # Feature A: Pipeline Board
    # ═══════════════════════════════════════════════════════════════

    def _show_pipeline_board(self) -> None:
        """Create (or re-show) the floating Pipeline Status Board dialog."""
        self._refresh_pipeline_progress_from_disk(self.pipeline_config)
        if self._pipeline_board is None:
            self._pipeline_board = PipelineBoardDialog(parent=self)
            self._pipeline_board.run_all_requested.connect(self._run_all_stages)
            self._pipeline_board.stop_requested.connect(self._cancel_pipeline_stage)
            self._pipeline_board.reset_requested.connect(
                self._reset_pipeline_progress_snapshot
            )
        self._pipeline_board.apply_progress(self._pipeline_progress)
        self._pipeline_board.show()
        self._pipeline_board.raise_()
        self._pipeline_board.activateWindow()

    def _pipeline_error_summary(self, err: object) -> str:
        return _error_summary(err)

    def _set_pipeline_stage(
        self,
        num: int,
        status: str,
        summary: str | None = None,
    ) -> None:
        """Update the MainWindow snapshot and the board dialog if it exists."""
        if not getattr(self, "_pipeline_progress", None):
            self._pipeline_progress = empty_stage_progress()
        entry = self._pipeline_progress.setdefault(
            num, {"status": "pending", "summary": PENDING_SUMMARY}
        )
        if status == "error":
            status = "failed"
        entry["status"] = status
        if summary is not None:
            entry["summary"] = summary
        elif status == "pending":
            entry["summary"] = PENDING_SUMMARY
        elif status == "running" and not entry.get("summary"):
            entry["summary"] = "In progress…"
        board = getattr(self, "_pipeline_board", None)
        if board is not None:
            board.update_stage(num, status, summary=entry.get("summary"))

    def _refresh_pipeline_progress_from_disk(self, cfg) -> None:
        """Merge on-disk artifacts into the snapshot; skip running stages."""
        current = getattr(self, "_pipeline_progress", None) or empty_stage_progress()
        try:
            disk = collect_stage_progress(cfg)
        except Exception:
            logger.exception("Could not collect pipeline board progress from disk")
            return
        self._pipeline_progress = merge_disk_progress(current, disk)
        board = getattr(self, "_pipeline_board", None)
        if board is not None:
            board.apply_progress(self._pipeline_progress)

    def _reset_pipeline_progress_snapshot(self) -> None:
        self._pipeline_progress = empty_stage_progress()

    def _run_all_stages(self) -> None:
        """Run stages 1→9 sequentially by chaining finished signals."""
        self._log("═" * 60)
        self._log("RUN ALL STAGES requested")
        self._log("═" * 60)
        # Cancellation is cooperative, so a running worker would still hold
        # the shared slot and make the first stage "not start". Refuse instead.
        if not self._pipeline_stage_guard("Run All"):
            return
        self._run_all_queue = [
            "search", "score", "extract", "validate", "audit",
            "synthesize", "fulltext", "robustness", "verify",
        ]
        self._run_all_active = True
        self._reset_run_all_stage_paths()
        for i in range(1, 10):
            self._set_pipeline_stage(i, "pending")
        self._run_all_stages_advance()

    def _reset_run_all_stage_paths(self) -> None:
        """Drop manual stage path overrides so every stage chains to the last."""
        dropped = self._reset_stage_path_fields()
        for path in dropped:
            self._log(f"RUN ALL: ignoring custom stage path {path}", "WARNING")

    def _pipeline_stage_guard(self, stage_name: str) -> bool:
        """Block a new pipeline stage if another worker is still running.

        Prevents the single `_pipeline_worker` / `_search_worker` slots from
        being overwritten and orphaning an in-progress stage.
        """
        meta_worker = getattr(self, "_meta_worker", None)
        if meta_worker is not None and meta_worker.isRunning() and meta_worker.uses_model:
            self._log(f"Cannot start {stage_name}: the meta-analysis is using the model. "
                      "Wait for it to finish or click Stop.", "WARNING")
            return False
        if self._pipeline_worker is not None and self._pipeline_worker.isRunning():
            self._log(
                f"Cannot start {stage_name}: another pipeline stage is running. "
                "Wait for it to finish or click Cancel.",
                "WARNING",
            )
            return False
        if self._search_worker is not None and self._search_worker.isRunning():
            self._log(
                f"Cannot start {stage_name}: Stage 1 search is still running. "
                "Wait for it to finish or click Cancel.",
                "WARNING",
            )
            return False
        return True

    def _run_all_stages_advance(self) -> None:
        """Pop the next queued stage and start it; abort if it fails to start."""
        if not self._run_all_active:
            return
        if not self._run_all_queue:
            self._run_all_active = False
            self._log("═" * 60)
            self._log("RUN ALL STAGES complete")
            self._log("═" * 60)
            return

        stage = self._run_all_queue.pop(0)
        stage_start_map = {
            "search": self._start_sysys_search,
            "score": self._start_sysys_score,
            "extract": self._start_extract,
            "validate": self._start_validate,
            "audit": self._start_audit,
            "synthesize": self._start_synthesize,
            "fulltext": self._start_sysys_fulltext,
            "robustness": self._start_robustness,
            "verify": self._start_verify,
        }
        num = STAGE_NUMBERS.get(stage, 0)
        self._set_pipeline_stage(num, "running")
        self._log(f"RUN ALL: starting stage {num} ({stage})")

        start_fn = stage_start_map.get(stage)
        if start_fn is None:
            self._log(f"RUN ALL: no start method for stage {stage}; aborting.", "ERROR")
            self._run_all_active = False
            self._run_all_queue.clear()
            return

        # Point auto-filled inputs at the outputs the previous stage just wrote.
        self._sync_stage_path_fields(self.pipeline_config)
        try:
            start_fn()
        except Exception as exc:
            logger.exception("Run All: stage %s (%s) launcher raised", num, stage)
            _halt_run_all(
                self, num, _error_summary(exc),
                f"RUN ALL: stage {num} ({stage}) failed to start: {exc}; aborting chain.",
            )
            return

        # Detect whether a worker was actually started. Search uses its own
        # worker slot; all other pipeline stages share _pipeline_worker.
        worker = self._search_worker if stage == "search" else self._pipeline_worker
        if worker is None or not worker.isRunning():
            _halt_run_all(
                self, num, "Did not start",
                f"RUN ALL: stage {num} ({stage}) did not start; aborting chain.",
            )

    def _abort_run_all(self, msg: str) -> None:
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log(msg, "ERROR")

    def _cancel_pipeline_stage(self):
        if self._pipeline_worker:
            self._pipeline_worker.cancel()
        if self._search_worker:
            self._search_worker.cancel()
        # Immediately re-enable buttons so the user isn't stuck
        run_btn = getattr(self, '_active_stage_run_btn', None)
        cancel_btn = getattr(self, '_active_stage_cancel_btn', None)
        progress_bar = getattr(self, '_active_stage_progress', None)
        if run_btn:
            run_btn.setEnabled(True)
        if cancel_btn:
            cancel_btn.setEnabled(False)
        if progress_bar:
            progress_bar.setVisible(False)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL cancelled by user.")
        self._log("Stage cancelled")

    def _connect_stage_worker(self, on_done, on_error, on_cancelled=None):
        """Wire the standard worker signals for a stage launcher.

        Connects progress (to the log), finished, error, and optionally
        cancelled. Every stage launcher used to repeat this block verbatim.
        """
        worker = self._pipeline_worker
        if worker is None:
            raise RuntimeError("_connect_stage_worker called before the worker was created")
        worker.progress.connect(lambda msg: self._log(msg))
        worker.finished.connect(lambda name, result: on_done(result))
        worker.error.connect(lambda name, err: on_error(err))
        if on_cancelled is None:
            stage_name = getattr(worker, "stage_name", "")
            worker.cancelled.connect(lambda: self._on_stage_worker_cancelled(stage_name))
        else:
            worker.cancelled.connect(on_cancelled)

    def _on_stage_worker_cancelled(self, stage_name: str) -> None:
        """Default ``cancelled`` handler: free the stage UI, board row and Run All."""
        run_btn = getattr(self, "_active_stage_run_btn", None)
        cancel_btn = getattr(self, "_active_stage_cancel_btn", None)
        progress_bar = getattr(self, "_active_stage_progress", None)
        if run_btn:
            run_btn.setEnabled(True)
        if cancel_btn:
            cancel_btn.setEnabled(False)
        if progress_bar:
            progress_bar.setVisible(False)
        num = STAGE_NUMBERS.get(stage_name)
        if num is not None:
            self._set_pipeline_stage(num, "pending", summary="Cancelled")
        self._abort_run_all(f"RUN ALL cancelled at stage {num} ({stage_name}).")
        self._log(f"Stage '{stage_name}' cancelled.", "WARNING")
