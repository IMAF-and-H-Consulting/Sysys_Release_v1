"""Shared helpers for the Sysys settings and dialog modules."""
from __future__ import annotations

from typing import cast

from PySide6.QtCore import QSettings

from gui.project_manager import ProjectManager


def _default_google_drive_folder() -> str:
    """Return a likely local Google Drive folder, or the home directory."""
    return str(ProjectManager.default_auto_backup_destination())


def _qs_bool(settings: QSettings, key: str, default: bool) -> bool:
    """Read a bool from QSettings; `type=bool` makes Qt coerce the stored value."""
    return cast(bool, settings.value(key, default, type=bool))


def _qs_str(settings: QSettings, key: str, default: str = "") -> str:
    """Read a str from QSettings; `type=str` makes Qt coerce the stored value."""
    return cast(str, settings.value(key, default, type=str))


def _qs_int(settings: QSettings, key: str, default: int) -> int:
    """Read an int from QSettings; `type=int` makes Qt coerce the stored value."""
    return cast(int, settings.value(key, default, type=int))
