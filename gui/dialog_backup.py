"""Automatic project backup options dialog for the Sysys GUI."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from gui.dialog_common import _default_google_drive_folder, _qs_bool, _qs_int, _qs_str
from gui.project_manager import ProjectManager


class AutoBackupOptionsDialog(QDialog):
    """Configure lightweight automatic project backups."""

    def __init__(
        self,
        parent=None,
        default_enabled: bool | None = None,
        project_path: Path | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Automatic Project Backup")
        self.setMinimumWidth(520)
        self.project_path = Path(project_path).expanduser().resolve() if project_path else None

        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        intro = QLabel(
            "Choose a local Google Drive folder or another synced folder. "
            "SystS will copy only changed review files by default."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        self.enabled = QCheckBox("Enable automatic project backup")
        saved_enabled = _qs_bool(settings, "backup/enabled", False)
        self.enabled.setChecked(saved_enabled if default_enabled is None else default_enabled)
        outer.addWidget(self.enabled)

        destination_group = QGroupBox("Backup Destination")
        destination_layout = QHBoxLayout(destination_group)
        self.destination_edit = QLineEdit()
        self.destination_edit.setPlaceholderText("Google Drive / SystS Backups")
        self.destination_edit.setText(
            _qs_str(settings, "backup/destination", _default_google_drive_folder())
        )
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_destination)
        destination_layout.addWidget(self.destination_edit, 1)
        destination_layout.addWidget(browse_btn)
        outer.addWidget(destination_group)

        options_group = QGroupBox("What to Back Up")
        options_layout = QVBoxLayout(options_group)
        self.include_images = QCheckBox("Include generated plots/images")
        self.include_images.setChecked(_qs_bool(settings, "backup/include_images", True))
        self.include_fulltext = QCheckBox("Include full-text PDFs and raw downloads")
        self.include_fulltext.setChecked(_qs_bool(settings, "backup/include_fulltext", False))
        self.include_fulltext.setToolTip("Off by default to avoid large Drive sync jobs.")
        options_layout.addWidget(self.include_images)
        options_layout.addWidget(self.include_fulltext)
        outer.addWidget(options_group)

        timing_group = QGroupBox("When to Back Up")
        timing_layout = QFormLayout(timing_group)
        self.on_stage_complete = QCheckBox("After completed pipeline stages")
        self.on_stage_complete.setChecked(
            _qs_bool(settings, "backup/on_stage_complete", True)
        )
        self.on_exit = QCheckBox("When exiting the app")
        self.on_exit.setChecked(_qs_bool(settings, "backup/on_exit", True))
        self.min_interval = QSpinBox()
        self.min_interval.setRange(1, 1440)
        self.min_interval.setValue(_qs_int(settings, "backup/min_interval_minutes", 10))
        self.min_interval.setSuffix(" min")
        self.max_file_mb = QSpinBox()
        self.max_file_mb.setRange(1, 4096)
        self.max_file_mb.setValue(_qs_int(settings, "backup/max_file_mb", 100))
        self.max_file_mb.setSuffix(" MB")
        timing_layout.addRow(self.on_stage_complete)
        timing_layout.addRow(self.on_exit)
        timing_layout.addRow("Minimum interval:", self.min_interval)
        timing_layout.addRow("Skip files larger than:", self.max_file_mb)
        outer.addWidget(timing_group)

        note = QLabel(
            "Manual backups still create full timestamped zip files. Automatic backups "
            "update a lightweight latest/ folder for easy restore."
        )
        note.setWordWrap(True)
        outer.addWidget(note)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save Settings")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._accept_if_valid)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(save_btn)
        outer.addLayout(buttons)

    def _browse_destination(self) -> None:
        current = Path(self.destination_edit.text().strip() or _default_google_drive_folder())
        start_dir = current if current.exists() else current.parent
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose Backup Destination",
            str(start_dir),
        )
        if path:
            self.destination_edit.setText(path)

    def _accept_if_valid(self) -> None:
        destination = self.destination_edit.text().strip()
        if self.enabled.isChecked() and not destination:
            QMessageBox.warning(self, "Backup Destination", "Choose a backup destination folder.")
            return
        if self.enabled.isChecked():
            destination_path = Path(destination).expanduser()
            try:
                destination_path.mkdir(parents=True, exist_ok=True)
                if self.project_path is not None:
                    ProjectManager._validate_backup_destination(
                        self.project_path,
                        destination_path,
                    )
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Backup Destination",
                    f"Cannot use this backup destination:\n{exc}",
                )
                return
            self.destination_edit.setText(str(destination_path.resolve()))
        self.accept()

    def values(self) -> dict:
        return {
            "enabled": self.enabled.isChecked(),
            "destination": self.destination_edit.text().strip(),
            "include_images": self.include_images.isChecked(),
            "include_fulltext": self.include_fulltext.isChecked(),
            "on_stage_complete": self.on_stage_complete.isChecked(),
            "on_exit": self.on_exit.isChecked(),
            "min_interval_minutes": self.min_interval.value(),
            "max_file_mb": self.max_file_mb.value(),
        }
