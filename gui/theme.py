"""
Windows 98 / Classic GUI theme for Hypothesis Consensus Analyzer.

Provides COLORS dict, CHART_SEQUENCE list, and get_stylesheet() for
consistent Win98-style visual appearance across the application.
"""

COLORS = {
    # Core UI
    "bg": "#c0c0c0",
    "surface": "#d4d0c8",
    "text": "#000000",
    "text_secondary": "#404040",
    "border": "#808080",
    "border_light": "#ffffff",
    "border_dark": "#404040",
    "primary": "#000080",
    "primary_hover": "#0000a0",
    "selection_bg": "#000080",
    "selection_fg": "#ffffff",
    # Semantic
    "error": "#cc0000",
    "warning": "#cc6600",
    "success": "#008000",
    # Chart palette
    "chart_teal": "#008080",
    "chart_terra": "#cc6633",
    "dark_teal": "#006666",
    "cyan": "#00cccc",
    "mauve": "#996699",
    "gold": "#cc9900",
    "olive": "#669933",
    "brown": "#996633",
}

CHART_SEQUENCE = [
    COLORS["chart_teal"],
    COLORS["chart_terra"],
    COLORS["dark_teal"],
    COLORS["cyan"],
    COLORS["mauve"],
    COLORS["gold"],
    COLORS["olive"],
    COLORS["brown"],
]

# 3-D border effects for Win98 look
_RAISED = (
    "border-top: 2px solid #ffffff; border-left: 2px solid #ffffff; "
    "border-bottom: 2px solid #404040; border-right: 2px solid #404040;"
)
_SUNKEN = (
    "border-top: 2px solid #404040; border-left: 2px solid #404040; "
    "border-bottom: 2px solid #ffffff; border-right: 2px solid #ffffff;"
)
_ETCHED = (
    "border-top: 1px solid #808080; border-left: 1px solid #808080; "
    "border-bottom: 1px solid #ffffff; border-right: 1px solid #ffffff;"
)


def get_stylesheet(font_size: int = 8) -> str:
    """Return a complete QSS stylesheet implementing the Windows 98 visual theme."""
    return f"""
        /* ── Global ── */
        QMainWindow, QDialog {{
            background-color: {COLORS['bg']};
        }}
        QWidget {{
            font-family: Tahoma, sans-serif;
            font-size: {font_size}pt;
            color: {COLORS['text']};
        }}

        /* ── Menu Bar ── */
        QMenuBar {{
            background-color: {COLORS['bg']};
            border-bottom: 1px solid {COLORS['border']};
            padding: 1px;
        }}
        QMenuBar::item {{
            padding: 2px 8px;
            background: transparent;
        }}
        QMenuBar::item:selected {{
            background-color: {COLORS['selection_bg']};
            color: {COLORS['selection_fg']};
        }}
        QMenu {{
            background-color: {COLORS['surface']};
            border: 2px outset {COLORS['bg']};
            padding: 2px;
        }}
        QMenu::item {{
            padding: 3px 20px 3px 25px;
        }}
        QMenu::item:selected {{
            background-color: {COLORS['selection_bg']};
            color: {COLORS['selection_fg']};
        }}
        QMenu::separator {{
            height: 1px;
            background: {COLORS['border']};
            margin: 2px 5px;
        }}

        /* ── Tabs ── */
        QTabWidget::pane {{
            {_SUNKEN}
            background: {COLORS['surface']};
            padding: 6px;
        }}
        QTabBar::tab {{
            background: {COLORS['bg']};
            border: 1px solid {COLORS['border']};
            border-bottom: none;
            padding: 4px 10px;
            margin-right: 2px;
        }}
        QTabBar::tab:selected {{
            background: {COLORS['surface']};
            border-bottom: 1px solid {COLORS['surface']};
        }}
        QTabBar::tab:hover {{
            background: {COLORS['surface']};
        }}
        QTabBar::tab:focus {{
            border: 1px dashed {COLORS['text']};
        }}

        /* ── Buttons ── */
        QPushButton {{
            background-color: {COLORS['bg']};
            {_RAISED}
            padding: 3px 12px;
            min-height: 18px;
        }}
        QPushButton:hover {{
            background-color: {COLORS['surface']};
        }}
        QPushButton:pressed {{
            {_SUNKEN}
        }}
        QPushButton:disabled {{
            color: {COLORS['border']};
        }}
        QPushButton:focus {{
            border: 1px dashed {COLORS['text']};
        }}

        /* ── Line edits / Combo / Spin ── */
        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: #ffffff;
            {_SUNKEN}
            padding: 2px 4px;
        }}
        QComboBox {{
            background-color: #ffffff;
            {_SUNKEN}
            padding: 2px 4px;
        }}
        QComboBox::drop-down {{
            {_RAISED}
            width: 16px;
        }}
        QSpinBox, QDoubleSpinBox {{
            background-color: #ffffff;
            {_SUNKEN}
            padding: 2px 4px;
        }}
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px dashed {COLORS['primary']};
        }}

        /* ── Scroll bars ── */
        QScrollBar:vertical {{
            background: {COLORS['bg']};
            width: 16px;
            margin: 16px 0;
        }}
        QScrollBar::handle:vertical {{
            background: {COLORS['bg']};
            {_RAISED}
            min-height: 20px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            background: {COLORS['bg']};
            {_RAISED}
            height: 16px;
        }}
        QScrollBar:horizontal {{
            background: {COLORS['bg']};
            height: 16px;
            margin: 0 16px;
        }}
        QScrollBar::handle:horizontal {{
            background: {COLORS['bg']};
            {_RAISED}
            min-width: 20px;
        }}

        /* ── Progress bar ── */
        QProgressBar {{
            {_SUNKEN}
            text-align: center;
            background: #ffffff;
        }}
        QProgressBar::chunk {{
            background-color: {COLORS['primary']};
        }}

        /* ── Tables ── */
        QTableWidget {{
            background-color: #ffffff;
            {_SUNKEN}
            gridline-color: {COLORS['bg']};
            selection-background-color: {COLORS['selection_bg']};
            selection-color: {COLORS['selection_fg']};
        }}
        QTableWidget::item:focus {{
            border: 1px dashed {COLORS['text']};
            background-color: {COLORS['selection_bg']};
            color: {COLORS['selection_fg']};
        }}
        QTableWidget:focus {{
            border: 2px dashed {COLORS['primary']};
        }}
        QHeaderView::section {{
            background-color: {COLORS['bg']};
            {_RAISED}
            padding: 3px;
            border: none;
        }}

        /* ── Group box ── */
        QGroupBox {{
            {_ETCHED}
            margin-top: 10px;
            padding-top: 14px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }}

        /* ── Checkbox / Radio ── */
        QCheckBox, QRadioButton {{
            spacing: 5px;
        }}
        QCheckBox:focus, QRadioButton:focus {{
            color: {COLORS['primary']};
        }}

        /* ── Slider ── */
        QSlider::groove:horizontal {{
            {_SUNKEN}
            height: 6px;
        }}
        QSlider::handle:horizontal {{
            background: {COLORS['bg']};
            {_RAISED}
            width: 12px;
            margin: -4px 0;
        }}

        /* ── Scroll area ── */
        QScrollArea {{
            border: none;
            background: transparent;
        }}

        /* ── Status bar ── */
        QStatusBar {{
            background: {COLORS['bg']};
            {_SUNKEN}
        }}

        /* ── Tooltips ── */
        QToolTip {{
            background-color: #ffffe1;
            border: 1px solid {COLORS['border_dark']};
            padding: 2px;
            color: {COLORS['text']};
        }}

        /* ── Label ── */
        QLabel {{
            background: transparent;
            border: none;
        }}
    """
