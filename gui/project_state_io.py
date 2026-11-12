"""Read and write a project's ``project_state.json``.

Saving reports failure instead of swallowing it, loading never returns
stored credentials, and a corrupt file is renamed aside rather than being
overwritten by the next save (audit 18-M5, 01-L6). The saved state records
the hash of ``research_config.yaml`` its domain snapshot was taken against
(18-M2).
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from core.domain_config import domain_config_snapshot
from core.io import atomic_write_json
from gui.project_backup_safety import is_credential_field, scrub_project_state

logger = logging.getLogger("gui.project_manager")  # keep the original logger name

STATE_FILE = "project_state.json"
RESEARCH_CONFIG_FILE = "research_config.yaml"


class ProjectStateLoad(NamedTuple):
    """Loaded project_state.json; ``corrupt_backup`` names a quarantined file."""

    state: dict
    corrupt_backup: Path | None


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _serializable_config(config: object) -> dict:
    """JSON-safe, credential-free config attributes."""
    cfg_dict: dict = {}
    for attr in dir(config):
        if attr.startswith("_") or is_credential_field(attr):
            continue
        try:
            val = getattr(config, attr)
        except Exception:  # noqa: BLE001, S112 - skip properties that cannot be read
            continue
        if callable(val):
            continue
        if isinstance(val, (str, int, float, bool, type(None))):
            cfg_dict[attr] = val
        elif isinstance(val, Path):
            cfg_dict[attr] = str(val)
    return cfg_dict


def _domain_config_source(project_path: Path) -> dict | None:
    """Which research config file the domain snapshot was taken against."""
    try:
        digest = file_sha256(project_path / RESEARCH_CONFIG_FILE)
    except OSError:
        return None
    return {"file": RESEARCH_CONFIG_FILE, "sha256": digest}


def save_project_state(project_path: Path, config: object, window_state: dict) -> bool:
    """Write project_state.json; return False (after logging) when it failed."""
    state_path = project_path / STATE_FILE
    payload: dict = {
        "config": _serializable_config(config) if config is not None else {},
        "window_state": window_state,
    }
    if config is not None:
        payload["domain_config"] = domain_config_snapshot(config)
        source = _domain_config_source(project_path)
        if source is not None:
            payload["domain_config_source"] = source
    try:
        atomic_write_json(state_path, payload)
    except (OSError, TypeError, ValueError) as exc:
        logger.error("Failed to save project state to %s: %s", state_path, exc)
        return False
    logger.info(f"Project saved to {state_path}")
    return True


def _quarantine_corrupt_state(state_path: Path, reason: object) -> Path | None:
    """Rename an unreadable state file aside so a later save cannot destroy it."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = state_path.with_name(f"{state_path.name}.corrupt-{stamp}")
    try:
        state_path.rename(target)
    except OSError as exc:
        logger.error("Corrupt project state %s could not be moved aside: %s", state_path, exc)
        return None
    logger.error("Corrupt project state (%s) moved to %s", reason, target)
    return target


def read_project_state(project_path: Path) -> ProjectStateLoad:
    """Read project_state.json without credentials; quarantine a corrupt file."""
    state_path = project_path / STATE_FILE
    if not state_path.exists():
        return ProjectStateLoad({}, None)
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.warning("Could not read project state %s: %s", state_path, exc)
        return ProjectStateLoad({}, None)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return ProjectStateLoad({}, _quarantine_corrupt_state(state_path, exc))
    if not isinstance(data, dict):
        reason = f"top level is {type(data).__name__}, not an object"
        return ProjectStateLoad({}, _quarantine_corrupt_state(state_path, reason))
    state, dropped = scrub_project_state(data)
    if dropped:
        logger.warning(
            "Ignored credentials stored in %s (removed at the next save): %s",
            state_path, ", ".join(dropped),
        )
    return ProjectStateLoad(state, None)
