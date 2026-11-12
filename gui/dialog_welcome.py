"""First-run welcome dialog for the Sysys GUI."""
from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gui.theme import COLORS

# ---------------------------------------------------------------------------
# WelcomeDialog — first-run onboarding
# ---------------------------------------------------------------------------

class WelcomeDialog(QDialog):
    """Welcome dialog shown on first launch."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from gui.help_content import WELCOME_CONTENT

        self.setWindowTitle("Welcome to Sysys")
        self.setMinimumSize(550, 600)
        self.resize(550, 600)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(10)

        # Title
        title_label = QLabel(WELCOME_CONTENT["title"])
        title_font = QFont("Tahoma", 14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Separator
        sep = QWidget()
        sep.setFixedHeight(2)
        sep.setStyleSheet(
            "border-top: 1px solid #808080; border-bottom: 1px solid #ffffff;"
        )
        layout.addWidget(sep)

        # Overview
        overview = QLabel(WELCOME_CONTENT["overview"])
        overview.setWordWrap(True)
        layout.addWidget(overview)

        # Pipeline stages
        stages_header = QLabel("Pipeline Stages")
        hdr_font = QFont("Tahoma", 10)
        hdr_font.setBold(True)
        stages_header.setFont(hdr_font)
        layout.addWidget(stages_header)

        stages_html = "<ol style='margin-left: 16px;'>"
        for stage in WELCOME_CONTENT["pipeline_stages"]:
            stages_html += f"<li>{stage}</li>"
        stages_html += "</ol>"
        stages_browser = QTextBrowser()
        stages_browser.setHtml(stages_html)
        stages_browser.setOpenExternalLinks(False)
        stages_browser.setStyleSheet(
            f"background: {COLORS['surface']}; border: none; font-family: Tahoma; font-size: 8pt;"
        )
        stages_browser.setMaximumHeight(220)
        layout.addWidget(stages_browser)

        # Getting started
        gs_header = QLabel("Getting Started")
        gs_header.setFont(hdr_font)
        layout.addWidget(gs_header)

        for tip in WELCOME_CONTENT["getting_started"]:
            tip_label = QLabel(f"\u2022  {tip}")
            tip_label.setWordWrap(True)
            layout.addWidget(tip_label)

        layout.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # Bottom row
        bottom = QHBoxLayout()
        self._dont_show = QCheckBox("Don't show again")
        bottom.addWidget(self._dont_show)
        bottom.addStretch()

        start_btn = QPushButton("Get Started")
        start_btn.setMinimumWidth(100)
        start_btn.clicked.connect(self._on_start)
        bottom.addWidget(start_btn)
        outer.addLayout(bottom)

    def _on_start(self):
        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        if self._dont_show.isChecked():
            settings.setValue("welcome/shown", True)
        self.accept()
