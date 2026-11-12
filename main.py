"""
Entry point for the Hypothesis Consensus Analyzer application.
"""
import sys
from typing import cast

from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.theme import get_stylesheet


def main():
    app = QApplication(sys.argv)
    from gui.settings_persistence import migrate_legacy_settings
    migrate_legacy_settings()

    # Read saved font size (default 10pt)
    settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
    font_size = cast(int, settings.value("appearance/font_size", 10, type=int))

    # Apply Win98-style theme with user font size
    app.setStyleSheet(get_stylesheet(font_size))

    # Set application-wide font
    font = QFont("Monospace", font_size)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
