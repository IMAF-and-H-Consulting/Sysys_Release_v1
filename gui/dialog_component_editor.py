"""Component editor dialog for the Sysys GUI."""
from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from core.models import HypothesisComponent
from gui.theme import COLORS

# ---------------------------------------------------------------------------
# ComponentEditor
# ---------------------------------------------------------------------------

_KEYWORD_ANALYSIS_LABEL = "Keyword analysis:"


def _split_keyword_analysis_description(description: str) -> tuple[str, str]:
    """Return editable description text and preserved keyword-analysis suffix."""
    if _KEYWORD_ANALYSIS_LABEL not in description:
        return description.strip(), ""

    base, analysis = description.split(_KEYWORD_ANALYSIS_LABEL, 1)
    return base.rstrip(" |"), f" | {_KEYWORD_ANALYSIS_LABEL}{analysis.rstrip()}"


def _parse_keyword_analysis_metrics(analysis_suffix: str) -> dict[str, str]:
    """Extract displayed metric values from a stored keyword-analysis suffix."""
    metrics = re.findall(
        r"\b(utility|coverage|relevance|confidence|direction)=(-?\d+(?:\.\d+)?)",
        analysis_suffix,
    )
    return {name: value for name, value in metrics}


class ComponentEditor(QDialog):
    """Modal dialog for editing a single hypothesis component."""

    def __init__(self, component: HypothesisComponent, parent=None):
        super().__init__(parent)
        self.component = component
        self._base_description, self._analysis_suffix = (
            _split_keyword_analysis_description(component.description)
        )
        self.setWindowTitle("Edit Component")
        self.setMinimumSize(720, 360)

        layout = QVBoxLayout(self)

        content_row = QHBoxLayout()
        content_row.setSpacing(12)

        form = QFormLayout()
        self.label_edit = QLineEdit(component.label)
        form.addRow("Label:", self.label_edit)

        self.desc_edit = QTextEdit()
        self.desc_edit.setPlainText(self._base_description)
        self.desc_edit.setMaximumHeight(100)
        form.addRow("Description:", self.desc_edit)

        self.keywords_edit = QTextEdit()
        self.keywords_edit.setPlainText(", ".join(component.keywords))
        self.keywords_edit.setMaximumHeight(60)
        form.addRow("Keywords:", self.keywords_edit)

        content_row.addLayout(form, 3)
        content_row.addWidget(self._build_keyword_feedback_panel(), 2)
        layout.addLayout(content_row)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _save(self):
        label = self.label_edit.text().strip()
        if not label:
            # Saved components without a label are dropped on reload (18-L10).
            QMessageBox.warning(self, "Edit Component", "A component needs a label.")
            return
        self.component.label = label
        base_description = self.desc_edit.toPlainText().strip()
        self.component.description = (
            f"{base_description}{self._analysis_suffix}"
            if self._analysis_suffix and base_description
            else base_description or self._analysis_suffix.lstrip(" |")
        )
        kw_text = self.keywords_edit.toPlainText().strip()
        self.component.keywords = [
            k.strip() for k in kw_text.split(",") if k.strip()
        ]
        self.accept()

    def _build_keyword_feedback_panel(self) -> QGroupBox:
        group = QGroupBox("Keyword Analysis")
        layout = QVBoxLayout(group)

        metrics = _parse_keyword_analysis_metrics(self._analysis_suffix)
        if not metrics:
            note = QLabel("Run Keyword Analysis to populate feedback.")
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {COLORS['text_secondary']};")
            layout.addWidget(note)
            layout.addStretch()
            return group

        metric_form = QFormLayout()
        rows = [
            (
                "Weight",
                f"{float(getattr(self.component, 'weight', 1.0) or 0.0):.3f}",
            ),
            ("Utility", metrics.get("utility", "-")),
            ("Coverage", metrics.get("coverage", "-")),
            ("Relevance", metrics.get("relevance", "-")),
            ("Confidence", metrics.get("confidence", "-")),
            ("Direction", metrics.get("direction", "-")),
        ]
        for label, value in rows:
            value_label = QLabel(value)
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value_label.setStyleSheet("font-weight: bold;")
            metric_form.addRow(f"{label}:", value_label)
        layout.addLayout(metric_form)

        hint = QLabel("These values are read-only feedback from the last keyword analysis.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(hint)
        layout.addStretch()
        return group
