"""
Pipeline Board dialog for SystS — Windows 98 style.

Shows all 10 pipeline stages with ✅/⏳/❌/⬜ status icons, elapsed time,
a one-line summary of each stage's output, and buttons to run or stop
the full pipeline.
"""
from __future__ import annotations

import logging
import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.stage_outputs import PENDING_SUMMARY, STAGE_NUMBERS, check_output
from gui.theme import _RAISED, _SUNKEN, COLORS

logger = logging.getLogger(__name__)


# Stage definitions: (stage_num, display_name)
STAGES = [
    (1, "Search"),
    (2, "Score"),
    (3, "Extract"),
    (4, "Validate"),
    (5, "Audit"),
    (6, "Synthesize"),
    (7, "Fulltext"),
    (8, "Robustness"),
    (9, "Verify"),
    (10, "Compare"),
]

STATUS_ICONS = {
    "pending":  "⬜",
    "running":  "⏳",
    "done":     "✅",
    "failed":   "❌",
}

STATUS_LABELS = {
    "pending":  "Pending",
    "running":  "Running…",
    "done":     "Done",
    "failed":   "Failed",
}


def normalize_status(status: str) -> str:
    """Map aliases (e.g. 'error') onto the board's known status keys."""
    if status == "error":
        return "failed"
    if status not in STATUS_ICONS:
        return "pending"
    return status


def empty_stage_progress() -> dict[int, dict[str, str]]:
    """Default snapshot: every stage pending with no output."""
    return {
        num: {"status": "pending", "summary": PENDING_SUMMARY}
        for num, _name in STAGES
    }


def merge_disk_progress(
    current: dict[int, dict],
    disk: dict[int, dict],
) -> dict[int, dict[str, str]]:
    """Overlay disk artifacts onto the live snapshot.

    A stage that is currently ``running`` is left untouched so a refresh
    cannot clobber an in-flight worker.
    """
    merged = empty_stage_progress()
    for num, info in (current or {}).items():
        if num not in merged:
            continue
        merged[num] = {
            "status": normalize_status(str(info.get("status", "pending"))),
            "summary": str(info.get("summary") or PENDING_SUMMARY),
        }
    for num, info in (disk or {}).items():
        if num not in merged:
            continue
        if merged[num]["status"] == "running":
            continue
        merged[num] = {
            "status": normalize_status(str(info.get("status", "pending"))),
            "summary": str(info.get("summary") or PENDING_SUMMARY),
        }
    return merged


def collect_stage_progress(cfg) -> dict[int, dict[str, str]]:
    """Inspect on-disk pipeline artifacts and return a status snapshot.

    Missing or unreadable files leave the stage pending with
    ``No output yet``. Never raises.
    """
    progress = empty_stage_progress()
    if cfg is None:
        return progress
    for num in STAGE_NUMBERS:
        try:
            progress[num] = check_output(num, cfg)
        except Exception:
            logger.debug("Could not summarize stage %s", num, exc_info=True)
            progress[num] = {"status": "pending", "summary": PENDING_SUMMARY}
    return progress


class PipelineBoardDialog(QDialog):
    """Non-modal floating dialog showing real-time pipeline stage status."""

    run_all_requested = Signal()
    stop_requested = Signal()
    reset_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowTitle("Pipeline Status Board")
        self.setMinimumSize(720, 420)
        self.setModal(False)

        # Internal state: stage_num (1-based) → {status, start_time, elapsed, summary}
        self._stage_data: dict[int, dict] = {}
        for num, name in STAGES:
            self._stage_data[num] = {
                "name": name,
                "status": "pending",
                "start_time": None,
                "elapsed": None,
                "summary": PENDING_SUMMARY,
            }

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_running_stages)
        self._timer.start(500)  # update every 500 ms

        self._setup_ui()

    # ── UI construction ────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # Title bar area
        title = QLabel("SystS Pipeline — Stage Status")
        title.setStyleSheet(
            f"background-color: {COLORS['primary']}; color: {COLORS['selection_fg']}; "
            f"font-weight: bold; padding: 4px 8px;"
        )
        layout.addWidget(title)

        # Stage table
        self.table = QTableWidget(len(STAGES), 5)
        self.table.setHorizontalHeaderLabels(
            ["Stage", "Name", "Status", "Elapsed", "Summary"]
        )
        self.table.setStyleSheet(
            f"QTableWidget {{ background-color: #ffffff; {_SUNKEN} "
            f"selection-background-color: {COLORS['selection_bg']}; "
            f"selection-color: {COLORS['selection_fg']}; }}"
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)

        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        self._populate_table()
        layout.addWidget(self.table)

        # Button row
        btn_frame = QFrame()
        btn_frame.setStyleSheet(f"QFrame {{ {_RAISED} background-color: {COLORS['bg']}; }}")
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(6, 4, 6, 4)

        self._run_btn = QPushButton("▶  Run All (S1→S9)")
        self._run_btn.setToolTip("Run Search through Verify; Compare is run separately on saved runs")
        self._run_btn.clicked.connect(self._on_run_clicked)

        self._stop_btn = QPushButton("⏹  Stop")
        self._stop_btn.setToolTip("Request cancellation of the current stage")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_clicked)

        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Reset all stages to Pending")
        reset_btn.clicked.connect(self._reset_all)

        btn_layout.addWidget(self._run_btn)
        btn_layout.addWidget(self._stop_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(reset_btn)

        layout.addWidget(btn_frame)

    def _populate_table(self) -> None:
        """Fill table rows from internal stage data."""
        for row, (num, _name) in enumerate(STAGES):
            d = self._stage_data[num]
            status: str = d["status"]
            icon = STATUS_ICONS.get(status, "⬜")
            status_label = STATUS_LABELS.get(status, status.title())

            # Stage number
            num_item = QTableWidgetItem(f"Stage {num}")
            num_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, num_item)

            # Stage name with icon prefix
            name_item = QTableWidgetItem(f"{icon} {d['name']}")
            self.table.setItem(row, 1, name_item)

            # Status
            status_item = QTableWidgetItem(status_label)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if status == "done":
                status_item.setForeground(Qt.GlobalColor.darkGreen)
            elif status == "failed":
                status_item.setForeground(Qt.GlobalColor.red)
            elif status == "running":
                status_item.setForeground(Qt.GlobalColor.darkBlue)
            self.table.setItem(row, 2, status_item)

            # Elapsed time
            elapsed = d["elapsed"]
            if elapsed is not None:
                elapsed_str = f"{elapsed:.1f}s"
            elif status == "running" and d["start_time"] is not None:
                elapsed_str = f"{time.time() - d['start_time']:.1f}s"
            else:
                elapsed_str = "—"
            elapsed_item = QTableWidgetItem(elapsed_str)
            elapsed_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 3, elapsed_item)

            summary_item = QTableWidgetItem(d.get("summary") or PENDING_SUMMARY)
            self.table.setItem(row, 4, summary_item)

    # ── Public API ────────────────────────────────────────────────────

    def update_stage(
        self,
        stage_num: int,
        status: str,
        elapsed_sec: float | None = None,
        summary: str | None = None,
    ) -> None:
        """Update a single stage row.

        Parameters
        ----------
        stage_num:
            1-based stage number (1–11).
        status:
            One of "pending", "running", "done", "failed" (or alias "error").
        elapsed_sec:
            Elapsed seconds; computed automatically for running stages if omitted.
        summary:
            Optional one-line description of the stage outcome.
        """
        if stage_num not in self._stage_data:
            return
        status = normalize_status(status)
        d = self._stage_data[stage_num]
        previous = d["status"]
        d["status"] = status
        if status == "running" and (previous != "running" or d["start_time"] is None):
            # A re-run starts a fresh clock; do not count the idle gap.
            d["start_time"] = time.time()
            d["elapsed"] = None
        if status == "pending":
            d["start_time"] = None
            d["elapsed"] = None
        if elapsed_sec is not None:
            d["elapsed"] = elapsed_sec
        elif status in ("done", "failed") and d["start_time"] is not None:
            d["elapsed"] = time.time() - d["start_time"]
        if summary is not None:
            d["summary"] = summary
        elif status == "pending":
            d["summary"] = PENDING_SUMMARY
        elif status == "running" and not d.get("summary"):
            d["summary"] = "In progress…"

        # Update run/stop button state
        any_running = any(v["status"] == "running" for v in self._stage_data.values())
        self._run_btn.setEnabled(not any_running)
        self._stop_btn.setEnabled(any_running)

        self._refresh_row(stage_num)

    def apply_progress(self, progress: dict[int, dict]) -> None:
        """Bulk-apply a snapshot without overwriting rows that are running."""
        for num, info in (progress or {}).items():
            if num not in self._stage_data:
                continue
            if self._stage_data[num]["status"] == "running":
                continue
            status = normalize_status(str(info.get("status", "pending")))
            summary = info.get("summary")
            self.update_stage(num, status, summary=summary)

    def _refresh_row(self, stage_num: int) -> None:
        """Refresh a single table row without rebuilding everything."""
        row = stage_num - 1
        if row < 0 or row >= self.table.rowCount():
            return
        d = self._stage_data[stage_num]
        status: str = d["status"]
        icon = STATUS_ICONS.get(status, "⬜")
        status_label = STATUS_LABELS.get(status, status.title())

        name_item = self.table.item(row, 1)
        if name_item:
            name_item.setText(f"{icon} {d['name']}")

        status_item = self.table.item(row, 2)
        if status_item:
            status_item.setText(status_label)
            if status == "done":
                status_item.setForeground(Qt.GlobalColor.darkGreen)
            elif status == "failed":
                status_item.setForeground(Qt.GlobalColor.red)
            elif status == "running":
                status_item.setForeground(Qt.GlobalColor.darkBlue)
            else:
                status_item.setForeground(Qt.GlobalColor.black)

        elapsed = d["elapsed"]
        if elapsed is not None:
            elapsed_str = f"{elapsed:.1f}s"
        elif status == "running" and d["start_time"] is not None:
            elapsed_str = f"{time.time() - d['start_time']:.1f}s"
        else:
            elapsed_str = "—"
        elapsed_item = self.table.item(row, 3)
        if elapsed_item:
            elapsed_item.setText(elapsed_str)

        summary_item = self.table.item(row, 4)
        if summary_item:
            summary_item.setText(d.get("summary") or PENDING_SUMMARY)

    def _tick_running_stages(self) -> None:
        """Called by QTimer to refresh elapsed time for any running stages."""
        for num, d in self._stage_data.items():
            if d["status"] == "running":
                self._refresh_row(num)

    # ── Button slots ──────────────────────────────────────────────────

    def _on_run_clicked(self) -> None:
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self.run_all_requested.emit()

    def _on_stop_clicked(self) -> None:
        self.stop_requested.emit()

    def _reset_all(self) -> None:
        for num in self._stage_data:
            self._stage_data[num]["status"] = "pending"
            self._stage_data[num]["start_time"] = None
            self._stage_data[num]["elapsed"] = None
            self._stage_data[num]["summary"] = PENDING_SUMMARY
        self._populate_table()
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self.reset_requested.emit()
