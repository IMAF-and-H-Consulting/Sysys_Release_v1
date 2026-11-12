"""
Pop-out result viewer windows for the Synthesis tab.

Provides independent top-level windows that display large synthesis outputs
(abstracts, hypothesis explanations, source citations, evidence networks)
with save/copy/export controls. Keeping these in separate windows frees up
space on the main tab while preserving the in-tab previews.

Classes:
- TextResultWindow: Markdown/plain-text viewer with Save + Copy buttons.
- NetworkResultWindow: Evidence-network graph + raw JSON with Save JSON +
  Save Image buttons.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.theme import COLORS

logger = logging.getLogger(__name__)


def _report_save_failure(parent: QWidget, what: str, path: str, exc: Exception) -> None:
    """Log a failed save and tell the user (audit 19-L3)."""
    logger.error("Could not save %s to %s: %s", what, path, exc)
    QMessageBox.warning(parent, "Save Failed", f"Could not save {what} to\n{path}\n\n{exc}")


# ── Text result window ────────────────────────────────────────────────


class TextResultWindow(QMainWindow):
    """
    Independent window displaying Markdown/plain-text synthesis output.

    Parameters
    ----------
    title : str
        Window title.
    source_path : Path | None
        File the text was loaded from (used as default save location).
    parent : QWidget | None
        Parent widget (kept weak — window is independent once shown).
    """

    def __init__(
        self,
        title: str,
        source_path: Path | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._source_path = Path(source_path) if source_path else None
        self._text: str = ""

        self.setMinimumSize(640, 520)
        self.resize(720, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Source label
        if self._source_path is not None:
            src_label = QLabel(f"Source: {self._source_path}")
        else:
            src_label = QLabel("Source: (unsaved)")
        src_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(src_label)

        # Text browser (read-only, renders Markdown)
        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(True)
        self._browser.setFont(QFont("Tahoma", 10))
        layout.addWidget(self._browser, 1)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        copy_btn = QPushButton("Copy")
        copy_btn.setMinimumWidth(80)
        copy_btn.clicked.connect(self._copy_text)
        btn_row.addWidget(copy_btn)

        save_btn = QPushButton("Save As...")
        save_btn.setMinimumWidth(90)
        save_btn.clicked.connect(self._save_as)
        btn_row.addWidget(save_btn)

        close_btn = QPushButton("Close")
        close_btn.setMinimumWidth(80)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)

    def set_text(self, text: str, render_markdown: bool = True) -> None:
        """Set the displayed text. Renders as Markdown when supported."""
        self._text = text or ""
        if render_markdown and hasattr(self._browser, "setMarkdown"):
            self._browser.setMarkdown(self._text)
        else:
            self._browser.setPlainText(self._text)

    def _copy_text(self) -> None:
        QApplication.clipboard().setText(self._text)

    def _save_as(self) -> None:
        default_name = ""
        default_dir = ""
        if self._source_path is not None:
            default_dir = str(self._source_path.parent)
            default_name = self._source_path.name
        suggested = str(Path(default_dir) / default_name) if default_name else ""
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result",
            suggested,
            "Markdown files (*.md);;Text files (*.txt);;All files (*)",
        )
        if not chosen:
            return
        try:
            Path(chosen).write_text(self._text, encoding="utf-8")
            logger.info("Saved result text to %s", chosen)
        except OSError as exc:
            _report_save_failure(self, "result text", chosen, exc)


# ── Network result window ─────────────────────────────────────────────


class NetworkResultWindow(QMainWindow):
    """
    Independent window showing the evidence network graph plus the raw JSON.

    Includes Save JSON and Save Image (PNG/SVG) buttons.
    """

    def __init__(
        self,
        title: str = "Evidence Network",
        source_path: Path | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._source_path = Path(source_path) if source_path else None
        self._network_data: dict = {}

        self.setMinimumSize(820, 600)
        self.resize(960, 720)

        from gui.network_view import NetworkGraphWidget

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Source label
        if self._source_path is not None:
            src_label = QLabel(f"Source: {self._source_path}")
        else:
            src_label = QLabel("Source: (none loaded)")
        src_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(src_label)

        # Splitter: graph on top, JSON below
        splitter = QSplitter(Qt.Orientation.Vertical)

        self._graph_widget = NetworkGraphWidget()
        splitter.addWidget(self._graph_widget)

        self._json_view = QTextEdit()
        self._json_view.setReadOnly(True)
        self._json_view.setFont(QFont("Monaco", 9))
        self._json_view.setPlaceholderText("No JSON loaded.")
        splitter.addWidget(self._json_view)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        save_json_btn = QPushButton("Save JSON As...")
        save_json_btn.setMinimumWidth(110)
        save_json_btn.clicked.connect(self._save_json)
        btn_row.addWidget(save_json_btn)

        save_img_btn = QPushButton("Save Image As...")
        save_img_btn.setMinimumWidth(110)
        save_img_btn.clicked.connect(self._save_image)
        btn_row.addWidget(save_img_btn)

        close_btn = QPushButton("Close")
        close_btn.setMinimumWidth(80)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)

    def set_network_data(self, data: dict) -> None:
        """Load network data into both the graph widget and JSON view."""
        self._network_data = data or {}
        self._graph_widget.set_network_data(self._network_data)
        try:
            pretty = json.dumps(self._network_data, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            pretty = str(self._network_data)
        self._json_view.setPlainText(pretty)

    def _save_json(self) -> None:
        if not self._network_data:
            return
        default_name = ""
        default_dir = ""
        if self._source_path is not None:
            default_dir = str(self._source_path.parent)
            default_name = self._source_path.name
        suggested = str(Path(default_dir) / default_name) if default_name else ""
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save Evidence Network JSON",
            suggested,
            "JSON files (*.json);;All files (*)",
        )
        if not chosen:
            return
        try:
            Path(chosen).write_text(
                json.dumps(self._network_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Saved evidence network JSON to %s", chosen)
        except (OSError, TypeError, ValueError) as exc:
            _report_save_failure(self, "evidence network JSON", chosen, exc)

    def _save_image(self) -> None:
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save Network Graph Image",
            "evidence_network.svg",
            "SVG files (*.svg);;PNG files (*.png);;All files (*)",
        )
        if not chosen:
            return
        path = Path(chosen)
        fmt = path.suffix.lstrip(".").lower() or "svg"
        if fmt not in ("svg", "png", "pdf", "jpg", "jpeg"):
            fmt = "svg"
        try:
            self._graph_widget.export_image(path, fmt=fmt)
            logger.info("Saved network graph image to %s", path)
        except Exception as exc:  # matplotlib raises assorted errors on export
            _report_save_failure(self, "network graph image", str(path), exc)
