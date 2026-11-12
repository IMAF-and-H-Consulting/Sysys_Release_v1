"""
Custom widgets for the Hypothesis Consensus Analyzer GUI.

Provides Win98-styled MetricCard, AlignmentBar, SectionHeader, and
GradeBadge widgets.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from gui.theme import COLORS


class MetricCard(QWidget):
    """Compact KPI card showing a title and numeric value in Win98 style."""

    def __init__(self, title: str, value: str = "—", parent=None):
        super().__init__(parent)
        self.setFixedHeight(60)
        self.setMinimumWidth(120)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(
            f"font-size: 7pt; color: {COLORS['text_secondary']};"
        )
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._value_label = QLabel(value)
        self._value_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._title_label)
        layout.addWidget(self._value_label)

        self.setStyleSheet(
            f"MetricCard {{ background: {COLORS['surface']}; "
            f"border-top: 2px solid #ffffff; border-left: 2px solid #ffffff; "
            f"border-bottom: 2px solid #404040; border-right: 2px solid #404040; }}"
        )

    def set_value(self, value: str):
        self._value_label.setText(value)

    def set_title(self, title: str):
        self._title_label.setText(title)


class AlignmentBar(QWidget):
    """Horizontal bar showing alignment score from -1 to +1 in Win98 style."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setMinimumWidth(200)
        self._score = 0.0
        self._label = ""
        # Set by the alignment tab so the bar's click/context handlers know
        # which component it belongs to.
        self._component_id: str | None = None

    def set_score(self, score: float, label: str = ""):
        self._score = max(-1.0, min(1.0, score))
        self._label = label
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        w, h = self.width(), self.height()

        # Sunken border
        painter.setPen(QColor("#404040"))
        painter.drawLine(0, 0, w - 1, 0)
        painter.drawLine(0, 0, 0, h - 1)
        painter.setPen(QColor("#ffffff"))
        painter.drawLine(w - 1, 0, w - 1, h - 1)
        painter.drawLine(0, h - 1, w - 1, h - 1)

        # Background
        inner = 2
        painter.fillRect(inner, inner, w - 2 * inner, h - 2 * inner, QColor("#ffffff"))

        # Bar
        mid = (w - 2 * inner) / 2 + inner
        bar_w = abs(self._score) * (w - 2 * inner) / 2
        if self._score >= 0:
            painter.fillRect(int(mid), inner, int(bar_w), h - 2 * inner,
                             QColor(COLORS["success"]))
        else:
            painter.fillRect(int(mid - bar_w), inner, int(bar_w), h - 2 * inner,
                             QColor(COLORS["error"]))

        # Center line
        painter.setPen(QColor(COLORS["border"]))
        painter.drawLine(int(mid), inner, int(mid), h - inner)

        # Label
        if self._label:
            painter.setPen(QColor(COLORS["text"]))
            font = QFont("Tahoma", 7)
            painter.setFont(font)
            painter.drawText(inner + 4, inner, w - 2 * inner - 8, h - 2 * inner,
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             self._label)

        painter.end()


class SectionHeader(QWidget):
    """Section divider with bold title, optional subheading, and etched separator."""

    def __init__(self, title: str, description: str = "", parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 4)
        layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 9pt;")
        layout.addWidget(title_label)

        if description:
            desc_label = QLabel(description)
            desc_label.setStyleSheet(
                f"font-size: 7pt; color: {COLORS['text_secondary']};"
            )
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)

        sep = QWidget()
        sep.setFixedHeight(2)
        sep.setStyleSheet(
            "border-top: 1px solid #808080; border-bottom: 1px solid #ffffff;"
        )
        layout.addWidget(sep)


class GradeBadge(QWidget):
    """Small colored badge showing GRADE certainty level in Win98 style."""

    COLORS = {
        "High": "#4CAF50",
        "Moderate": "#FFC107",
        "Low": "#FF9800",
        "Very Low": "#F44336",
    }

    TEXT_COLORS = {
        "High": "#ffffff",
        "Moderate": "#000000",
        "Low": "#000000",
        "Very Low": "#ffffff",
    }

    def __init__(self, certainty: str = "Very Low", parent=None):
        super().__init__(parent)
        self._certainty = certainty
        self.setFixedHeight(20)
        self.setFixedWidth(72)

    def set_certainty(self, certainty: str):
        self._certainty = certainty
        self.update()

    @property
    def certainty(self) -> str:
        return self._certainty

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        w, h = self.width(), self.height()
        bg = QColor(self.COLORS.get(self._certainty, "#808080"))
        fg = QColor(self.TEXT_COLORS.get(self._certainty, "#ffffff"))

        # Background fill
        painter.fillRect(1, 1, w - 2, h - 2, bg)

        # Win98-ish border: black outer border
        painter.setPen(QPen(QColor("#000000"), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        # Inner highlight edges for 3D effect
        painter.setPen(QColor("#ffffff"))
        painter.drawLine(1, 1, w - 2, 1)
        painter.drawLine(1, 1, 1, h - 2)
        darker = bg.darker(130)
        painter.setPen(darker)
        painter.drawLine(w - 2, 1, w - 2, h - 2)
        painter.drawLine(1, h - 2, w - 2, h - 2)

        # Text
        painter.setPen(fg)
        font = QFont("Tahoma", 7, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(2, 2, w - 4, h - 4,
                         Qt.AlignmentFlag.AlignCenter, self._certainty)

        painter.end()
