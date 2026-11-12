"""Apply a loaded project_state.json to the main window.

Plain functions rather than mixin methods, so lightweight windows that
borrow ``ProjectMixin._open_project_path`` keep working. Values read from
the file are type-checked (audit 18-L4), and the saved domain snapshot is
only applied after asking when research_config.yaml changed since (18-M2).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

from PySide6.QtWidgets import QMessageBox

from core.domain_config import domain_config_snapshot, restore_domain_config
from gui.project_manager import ProjectManager

logger = logging.getLogger("gui.main_window_project")  # shown in the GUI console


def as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def as_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(cast(int, value))
    except (TypeError, ValueError):
        return default


def _yaml_changed_since_snapshot(state: dict, cfg_yaml: Path | None) -> bool:
    """True when research_config.yaml differs from the one the snapshot saw."""
    saved = as_dict(state.get("domain_config_source")).get("sha256")
    if not saved or cfg_yaml is None:
        return False
    try:
        return ProjectManager.file_sha256(cfg_yaml) != saved
    except OSError:
        return False


def _prefer_edited_yaml(window, cfg_yaml: Path) -> bool:
    answer = QMessageBox.question(
        window,
        "Research Config Changed",
        f"{cfg_yaml.name} was edited after this project was last saved.\n\n"
        f"Yes: use the edited {cfg_yaml.name}.\n"
        "No: keep the domain settings saved with the project (defaults, "
        "graph analysis, extraction vocabulary, method-review rules and "
        "co-occurring conditions).",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    return answer == QMessageBox.StandardButton.Yes


def apply_domain_snapshot(window, state: dict, cfg_yaml: Path | None) -> None:
    """Restore the saved domain sections over the loaded config (18-M2).

    Asks first when research_config.yaml was edited after the snapshot, and
    logs which sections the snapshot replaced.
    """
    snapshot = state.get("domain_config")
    if not isinstance(snapshot, dict) or not snapshot:
        return
    yaml_edited = cfg_yaml is not None and _yaml_changed_since_snapshot(state, cfg_yaml)
    if yaml_edited and _prefer_edited_yaml(window, cfg_yaml):
        logger.info("Using edited %s instead of the saved domain snapshot", cfg_yaml.name)
        return
    current = domain_config_snapshot(window.pipeline_config)
    replaced = sorted(k for k in snapshot if k in current and current[k] != snapshot[k])
    restore_domain_config(window.pipeline_config, snapshot)
    if replaced:
        logger.info(
            "Saved project domain snapshot replaces config sections: %s",
            ", ".join(replaced),
        )


def restore_window_widgets(window, window_state: dict) -> None:
    """Restore Search and Stage 7 widgets from saved window state."""
    saved_components = ProjectManager.deserialize_hypothesis_components(
        window_state.get("hypothesis_components", [])
    )
    if saved_components:
        window.hypothesis_components = saved_components
        window._refresh_component_list()
        window._update_search_workflow_buttons()
    saved_boolean_query = str(window_state.get("weighted_boolean_query") or "")
    if saved_boolean_query and not saved_components and hasattr(window, "boolean_query_preview"):
        window.boolean_query_preview.setPlainText(saved_boolean_query)
    saved_paper_limit = window_state.get("paper_limit")
    if saved_paper_limit is not None and hasattr(window, "paper_limit"):
        window.paper_limit.setValue(as_int(saved_paper_limit, window.paper_limit.value()))
    if hasattr(window, "sysys_ft_resume"):
        stage7 = as_dict(window_state.get("stage7"))
        window.sysys_ft_corpus.setText(str(stage7.get("claims_path") or ""))
        window.sysys_ft_resume.setChecked(bool(stage7.get("resume", True)))
        window.sysys_ft_workers.setValue(as_int(
            stage7.get("workers"), window.pipeline_config.fulltext_download_workers
        ))
