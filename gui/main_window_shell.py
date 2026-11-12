"""Window chrome: menus, status bar, console, logging, shortcuts, browse dialogs.

Split from gui/main_window.py; mixed into MainWindow.
"""
from __future__ import annotations

import html
import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.main_window_common import _paths_resolve_under_root
from gui.theme import COLORS


class ShellMixin:
    """Window chrome: menus, status bar, console, logging, shortcuts, browse dialogs."""

    # ── Menu bar ──────────────────────────────────────────────────────

    def _build_menu_bar(self):
        """Create the File, Edit, View, and Help menus."""
        menu_bar = self.menuBar()

        # ── File menu ──
        file_menu = menu_bar.addMenu("&File")

        new_proj_action = QAction("&New Project...", self)
        new_proj_action.setShortcut(QKeySequence("Ctrl+N"))
        new_proj_action.setStatusTip("Create a new SystS project folder")
        new_proj_action.triggered.connect(self._new_project)
        file_menu.addAction(new_proj_action)

        open_proj_action = QAction("&Open Project...", self)
        open_proj_action.setShortcut(QKeySequence("Ctrl+O"))
        open_proj_action.setStatusTip("Open an existing SystS project folder")
        open_proj_action.triggered.connect(self._open_project)
        file_menu.addAction(open_proj_action)

        self._recent_menu = QMenu("Recent Projects", self)
        file_menu.addMenu(self._recent_menu)
        self._populate_recent_menu()

        save_proj_action = QAction("&Save Project", self)
        save_proj_action.setShortcut(QKeySequence("Ctrl+S"))
        save_proj_action.setStatusTip("Save the current project state")
        save_proj_action.triggered.connect(self._save_project)
        file_menu.addAction(save_proj_action)

        backup_proj_action = QAction("&Backup Project...", self)
        backup_proj_action.setStatusTip("Create a timestamped project backup zip")
        backup_proj_action.triggered.connect(self._backup_project)
        file_menu.addAction(backup_proj_action)

        restore_proj_action = QAction("&Restore Project Backup...", self)
        restore_proj_action.setStatusTip("Restore a SystS project backup")
        restore_proj_action.triggered.connect(self._restore_project_backup)
        file_menu.addAction(restore_proj_action)

        auto_backup_action = QAction("Auto &Backup Settings...", self)
        auto_backup_action.setStatusTip("Configure automatic Drive/local-folder backups")
        auto_backup_action.triggered.connect(
            lambda checked=False: self._configure_auto_backup()
        )
        file_menu.addAction(auto_backup_action)

        file_menu.addSeparator()

        # Log window — opened on demand as a floating window (like the console).
        self._toggle_log_action = QAction("&Log Window", self)
        self._toggle_log_action.setShortcut(QKeySequence("Ctrl+L"))
        self._toggle_log_action.setCheckable(True)
        self._toggle_log_action.setChecked(False)
        self._toggle_log_action.setStatusTip("Show/hide the application log window")
        self._toggle_log_action.triggered.connect(self._toggle_log_window)
        file_menu.addAction(self._toggle_log_action)

        file_menu.addSeparator()

        citation_action = QAction("&Citation Guide...", self)
        citation_action.setStatusTip("Show recommended citations for SystS analyses")
        citation_action.triggered.connect(self._open_citation_guide)
        file_menu.addAction(citation_action)

        settings_action = QAction("&Settings...", self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.setStatusTip("Open application settings")
        settings_action.triggered.connect(self._open_settings_dialog)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ── Edit menu ──
        edit_menu = menu_bar.addMenu("&Edit")

        center_output_action = QAction("&Show Output Console in Center", self)
        center_output_action.setShortcut(QKeySequence("Ctrl+Shift+`"))
        center_output_action.setStatusTip(
            "Show the output console and move it to the center of the screen"
        )
        center_output_action.triggered.connect(self._show_output_console_centered)
        edit_menu.addAction(center_output_action)

        # ── View menu ──
        view_menu = menu_bar.addMenu("&View")

        self._toggle_output_action = QAction("&Output Console", self)
        self._toggle_output_action.setShortcut(QKeySequence("Ctrl+`"))
        self._toggle_output_action.setCheckable(True)
        self._toggle_output_action.setChecked(True)
        self._toggle_output_action.setStatusTip("Show/hide the output console")
        self._toggle_output_action.triggered.connect(self._toggle_output_console)
        view_menu.addAction(self._toggle_output_action)

        # ── Help menu ──
        help_menu = menu_bar.addMenu("&Help")

        pipeline_board_action = QAction("&Pipeline Board...", self)
        pipeline_board_action.setShortcut(QKeySequence("Ctrl+P"))
        pipeline_board_action.setStatusTip("Show the pipeline status board")
        pipeline_board_action.triggered.connect(self._show_pipeline_board)
        help_menu.addAction(pipeline_board_action)

    def _open_settings_dialog(self):
        """Open the Settings dialog."""
        from gui.dialogs import SettingsDialog
        dlg = SettingsDialog(self, parent=self)
        dlg.exec()

    def _open_citation_guide(self):
        """Open copyable citation guidance for SystS analyses."""
        from gui.dialogs import CitationGuideDialog
        dlg = CitationGuideDialog(self)
        dlg.exec()

    # ── Status bar ────────────────────────────────────────────────────

    def _build_status_bar(self):
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("Ready")

    # ── Output console (dockable) ──────────────────────────────────────

    def _build_output_console(self):
        """Create a dockable output console at the bottom of the window."""
        self._output_dock = QDockWidget("Output", self)
        self._output_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._console_output = QTextEdit()
        self._console_output.setReadOnly(True)
        self._console_output.setAccessibleName("Output console")
        # Bound the document: long pipeline runs append unboundedly otherwise.
        self._console_output.document().setMaximumBlockCount(5000)
        self._console_output.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace; "
            "font-size: 14pt; "
            "background-color: #1a1a2e; color: #e0e0e0; "
            f"border: 1px solid {COLORS['border']};"
        )
        self._console_output.setMinimumHeight(80)
        layout.addWidget(self._console_output)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self._console_output.clear)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._output_dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._output_dock)
        # Float the output console as a separate window by default
        self._output_dock.setFloating(True)
        self._output_dock.resize(700, 350)
        self._output_dock.show()

        # Sync dock visibility with menu checkbox
        self._output_dock.visibilityChanged.connect(
            lambda visible: self._toggle_output_action.setChecked(visible)
        )

    def _toggle_output_console(self, checked: bool):
        """Show or hide the output console dock."""
        if checked:
            self._output_dock.show()
        else:
            self._output_dock.hide()

    def _show_output_console_centered(self):
        """Show the output console as a floating window centered on-screen."""
        dock = self._output_dock
        dock.setFloating(True)
        dock.showNormal()
        dock.show()

        screen = self.windowHandle().screen() if self.windowHandle() else None
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(max(dock.width(), 700), int(available.width() * 0.9))
            height = min(max(dock.height(), 350), int(available.height() * 0.9))
            x = available.x() + (available.width() - width) // 2
            y = available.y() + (available.height() - height) // 2
            dock.setGeometry(x, y, width, height)

        dock.raise_()
        dock.activateWindow()

    # ── Python logging → GUI bridge ─────────────────────────────────────

    def _install_log_handler(self):
        """Route Python logging from all Sysys modules into the GUI console.

        Uses a QObject signal to safely cross from worker threads to the GUI thread.
        """
        from PySide6.QtCore import QObject
        from PySide6.QtCore import Signal as QSignal

        class _Bridge(QObject):
            log_record = QSignal(str, str)  # (message, level)

        class _GUIHandler(logging.Handler):
            def __init__(self, bridge):
                super().__init__()
                self.bridge = bridge

            def emit(self, record):
                try:
                    msg = self.format(record)
                    self.bridge.log_record.emit(msg, record.levelname)
                except Exception:
                    self.handleError(record)

        self._log_bridge = _Bridge()
        self._log_bridge.log_record.connect(self._on_log_record)

        handler = _GUIHandler(self._log_bridge)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.DEBUG)

        # Attach to the root logger for gui.* and process.* modules
        for name in ("gui", "process", "core", "stages", "acquire"):
            log = logging.getLogger(name)
            log.addHandler(handler)
            log.setLevel(logging.DEBUG)

    def _on_log_record(self, message: str, level: str):
        """Slot receiving log records from worker threads."""
        self._log(message, level)

    # ── Help / Welcome ─────────────────────────────────────────────────

    def _add_help_button(self, tab_widget: QWidget, tab_name: str) -> QPushButton:
        """Insert a small '?' help button at the top-right of a tab's layout."""
        from gui.dialogs import HelpDialog

        btn = QPushButton("?")
        btn.setFixedSize(24, 24)
        btn.setToolTip("Learn about this stage")
        btn.setAccessibleName(f"Help for {tab_name} tab")
        btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 13pt; "
            "padding: 0px; min-height: 0px; }"
        )
        btn.clicked.connect(lambda checked, n=tab_name: HelpDialog(n, self).exec())

        help_row = QHBoxLayout()
        help_row.setContentsMargins(0, 0, 0, 0)
        help_row.addStretch()
        help_row.addWidget(btn)

        tab_layout = tab_widget.layout()
        if isinstance(tab_layout, QBoxLayout):
            tab_layout.insertLayout(0, help_row)
        return btn

    # ═══════════════════════════════════════════════════════════════
    # Feature D: At-a-Glance Panel helpers
    # ═══════════════════════════════════════════════════════════════

    def _toggle_glance_panel(self) -> None:
        """Show/hide the At-a-Glance panel."""
        if self._glance_frame.isVisible():
            self._glance_frame.hide()
            self._glance_toggle_btn.setText("▼ Show")
        else:
            self._glance_frame.show()
            self._glance_toggle_btn.setText("▲ Hide")

    def _update_glance_corpus(self, n: int) -> None:
        if hasattr(self, "_card_corpus"):
            self._card_corpus.set_value(f"{n} articles")

    def _update_glance_relevant(self, n: int) -> None:
        if hasattr(self, "_card_relevant"):
            self._card_relevant.set_value(f"{n} papers")

    def _update_glance_stage(self, stage_name: str) -> None:
        if hasattr(self, "_card_stage"):
            self._card_stage.set_value(stage_name)

    # ═══════════════════════════════════════════════════════════════
    # Feature F: Tab count badges
    # ═══════════════════════════════════════════════════════════════

    def _set_tab_badge(self, idx: int, base: str, count: int | None = None) -> None:
        """Set a tab's label with an optional count badge."""
        label = f"{base} ({count})" if count is not None else base
        self.tabs.setTabText(idx, label)

    # ═══════════════════════════════════════════════════════════════
    # Feature G: Keyboard Shortcuts
    # ═══════════════════════════════════════════════════════════════

    def _setup_shortcuts(self) -> None:
        """Install Ctrl+1–9 tab switchers and Ctrl+R run-current-tab shortcut."""
        from PySide6.QtGui import QShortcut
        # Tab navigation: Ctrl+1 through Ctrl+9
        for i in range(9):
            sc = QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self)
            sc.activated.connect(lambda idx=i: self.tabs.setCurrentIndex(idx))
        # Run current tab action
        run_sc = QShortcut(QKeySequence("Ctrl+R"), self)
        run_sc.activated.connect(self._run_current_tab_action)

    def _run_current_tab_action(self) -> None:
        """Run the primary action for the currently visible tab."""
        idx = self.tabs.currentIndex()
        actions = {
            0: self._start_sysys_search,
            1: self._start_sysys_score,
            2: self._start_extract,
            3: self._start_audit,
            4: self._start_synthesize,
            5: self._start_fulltext_retrieval,
            6: self._start_robustness,
        }
        action = actions.get(idx)
        if action:
            action()

    def _build_log_window(self):
        """Create the application Log as a floating window (opened from File menu).

        Previously the Log was a crowded tab; it now lives in its own dockable
        window (like the Output Console) toggled via File → Log Window.
        """
        self._log_dock = QDockWidget("Log", self)
        self._log_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setAccessibleName("Application log")
        # Bound the document: long pipeline runs append unboundedly otherwise.
        self.log_output.document().setMaximumBlockCount(5000)
        self.log_output.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 14pt;"
        )
        layout.addWidget(self.log_output)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_log)
        save_btn = QPushButton("Save Log...")
        save_btn.clicked.connect(self._save_log)
        self.log_autoscroll = QCheckBox("Auto-scroll")
        self.log_autoscroll.setChecked(True)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(self.log_autoscroll)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._log_dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)
        # Floating, hidden by default — the user opens it from the File menu.
        self._log_dock.setFloating(True)
        self._log_dock.resize(760, 420)
        self._log_dock.hide()

        # Keep the File-menu checkbox in sync with the window's visibility.
        self._log_dock.visibilityChanged.connect(
            lambda visible: self._toggle_log_action.setChecked(visible)
        )

    def _toggle_log_window(self, checked: bool):
        """Show or hide the log window from the File menu."""
        if checked:
            self._log_dock.show()
            self._log_dock.raise_()
            self._log_dock.activateWindow()
        else:
            self._log_dock.hide()

    # ── Log helpers ───────────────────────────────────────────────────

    def _log(self, msg: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] [{level}] {msg}"

        # Messages are text, never markup: a title or path containing "<" or
        # "&" must show as written (18-L8). The span forces rich-text mode so
        # the escaped entities render back to the original characters.
        safe = f'<span style="white-space:pre-wrap">{html.escape(line)}</span>'

        # Log window (may not exist yet during early UI construction)
        if hasattr(self, "log_output"):
            self.log_output.append(safe)
            if getattr(self, "log_autoscroll", None) and self.log_autoscroll.isChecked():
                sb = self.log_output.verticalScrollBar()
                sb.setValue(sb.maximum())

        # Output console (color-coded)
        if hasattr(self, '_console_output'):
            if level == "ERROR":
                color = "#ff6b6b"
            elif level == "WARNING":
                color = "#ffa94d"
            else:
                color = "#e0e0e0"
            self._console_output.append(
                f'<span style="color:{color}; white-space:pre-wrap">{html.escape(line)}</span>'
            )
            csb = self._console_output.verticalScrollBar()
            csb.setValue(csb.maximum())

    def _clear_log(self):
        self.log_output.clear()

    def _save_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "", "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.log_output.toPlainText())
        except OSError as exc:
            self._log(f"Could not save log to {path}: {exc}", "ERROR")
            QMessageBox.warning(self, "Save Log Failed", f"Could not save the log:\n{exc}")
            return
        self._log(f"Log saved to {path}")

    # ── File browse helpers ───────────────────────────────────────────

    def _browse_open_file(
        self,
        line_edit: QLineEdit,
        file_filter: str = "All files (*)",
        initial_dir: Path | None = None,
        fallback_dir: Path | None = None,
    ):
        start = ""
        for d in (initial_dir, fallback_dir, self.pipeline_config.project_root):
            if d is not None:
                try:
                    p = Path(d)
                    if p.is_dir():
                        start = str(p)
                        break
                except OSError:
                    continue
        cur = line_edit.text().strip()
        if cur:
            rp = _paths_resolve_under_root(cur, self.pipeline_config.project_root)
            if rp and rp.is_file():
                start = str(rp.parent)
        path, _ = QFileDialog.getOpenFileName(self, "Open File", start, file_filter)
        if path:
            line_edit.setText(path)

    def _browse_save_file(
        self,
        line_edit: QLineEdit,
        file_filter: str = "All files (*)",
        initial_dir: Path | None = None,
        fallback_dir: Path | None = None,
    ):
        start = ""
        for d in (initial_dir, fallback_dir, self.pipeline_config.project_root):
            if d is not None:
                try:
                    p = Path(d)
                    if p.is_dir():
                        start = str(p)
                        break
                except OSError:
                    continue
        cur = line_edit.text().strip()
        if cur:
            rp = _paths_resolve_under_root(cur, self.pipeline_config.project_root)
            if rp and rp.parent.is_dir():
                start = str(rp.parent)
            elif rp is None:
                p = Path(cur).expanduser()
                if not p.is_absolute():
                    p = self.pipeline_config.project_root / p
                if p.parent.is_dir():
                    start = str(p.parent)
        path, _ = QFileDialog.getSaveFileName(self, "Save File", start, file_filter)
        if path:
            line_edit.setText(path)
