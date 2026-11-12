"""Project lifecycle: welcome, open/save, backups, auto-backup, research config.

Split from gui/main_window.py; mixed into MainWindow.
"""
from __future__ import annotations

import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import cast

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

from gui.main_window_common import PipelineState
from gui.original_question import OriginalQuestionState
from gui.project_manager import ProjectManager
from gui.project_state_restore import (
    apply_domain_snapshot,
    as_dict,
    restore_window_widgets,
)
from gui.workers import ProjectBackupWorker

# How long to wait before retrying an auto-backup deferred by a busy pipeline.
_BUSY_BACKUP_RETRY_MS = 60_000


class ProjectMixin:
    """Project lifecycle: welcome, open/save, backups, auto-backup, research config."""

    def _show_welcome_if_first_run(self):
        """Show the welcome dialog if the user hasn't dismissed it."""
        from PySide6.QtCore import QSettings

        from gui.dialogs import WelcomeDialog

        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        if not settings.value("welcome/shown", False, type=bool):
            WelcomeDialog(self).exec()

    def _maybe_prompt_auto_backup(self, reason: str) -> None:
        """Offer automatic backup on first launch or after creating a project."""
        from PySide6.QtCore import QSettings

        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        if settings.value("backup/enabled", False, type=bool):
            return
        if settings.value("backup/prompt_disabled", False, type=bool):
            return
        if reason == "first_launch" and settings.value(
            "backup/first_launch_prompted", False, type=bool
        ):
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Automatic Project Backup")
        box.setText("Turn on automatic project backup?")
        box.setInformativeText(
            "SystS can keep a lightweight backup in your Google Drive folder. "
            "It copies only changed review files by default and skips large full-text downloads."
        )
        enable_btn = box.addButton("Turn On...", QMessageBox.ButtonRole.AcceptRole)
        not_now_btn = box.addButton("Not Now", QMessageBox.ButtonRole.RejectRole)
        dont_ask_btn = box.addButton("No, Don't Ask Again", QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(enable_btn)
        box.exec()

        clicked = box.clickedButton()
        if reason == "first_launch":
            settings.setValue("backup/first_launch_prompted", True)
        if clicked == enable_btn:
            self._configure_auto_backup(default_enabled=True)
        elif clicked == dont_ask_btn:
            settings.setValue("backup/prompt_disabled", True)
        elif clicked == not_now_btn:
            return

    # ═══════════════════════════════════════════════════════════════
    # Feature C: Project System
    # ═══════════════════════════════════════════════════════════════

    def _populate_recent_menu(self) -> None:
        """Rebuild the Recent Projects submenu from QSettings."""
        self._recent_menu.clear()
        recents = ProjectManager.get_recent_projects()
        if not recents:
            no_action = QAction("(none)", self)
            no_action.setEnabled(False)
            self._recent_menu.addAction(no_action)
            return
        for path in recents:
            action = QAction(str(path), self)
            action.triggered.connect(lambda checked, p=path: self._open_recent_project(p))
            self._recent_menu.addAction(action)

    def _project_change_blocked(self, action: str) -> bool:
        """Refuse a project switch while a worker or Run All is active (17-M5).

        A running stage keeps writing to the current project, and its
        finished handlers would update the next project's GUI state.
        """
        busy = [label for label, _worker in self._running_workers()]
        if self._run_all_active and not busy:
            busy.append("Run All")
        if not busy:
            return False
        names = ", ".join(busy)
        self._log(f"{action} refused: still running: {names}", "WARNING")
        QMessageBox.warning(
            self,
            action,
            f"Cannot change the project while work is running ({names}).\n\n"
            "Wait for it to finish or cancel it, then try again.",
        )
        return True

    def _open_recent_project(self, path: Path) -> None:
        """File > Recent Projects > path."""
        if self._project_change_blocked("Open Recent Project"):
            return
        self._open_project_path(path)

    def _prepare_for_new_project(self) -> None:
        if self._project_path is not None:
            self._original_question_state = OriginalQuestionState()

    def _new_project(self) -> None:
        """File > New Project..."""
        if self._project_change_blocked("New Project"):
            return
        path = ProjectManager.new_project(self)
        if path:
            self._prepare_for_new_project()
            self._set_project_path(path)
            self._populate_recent_menu()
            # Try to auto-load config from new project
            cfg_yaml = path / "research_config.yaml"
            if cfg_yaml.exists():
                try:
                    self._load_research_config(cfg_yaml)
                except Exception as exc:
                    self._handle_project_config_failure(path, cfg_yaml, exc)
            self.statusBar().showMessage(f"New project created: {path.name}", 4000)
            self._maybe_prompt_auto_backup("new_project")

    def _open_project(self) -> None:
        """File > Open Project..."""
        if self._project_change_blocked("Open Project"):
            return
        path = ProjectManager.open_project(self)
        if path:
            self._open_project_path(path)

    def _open_project_path(self, path: Path) -> None:
        """Common handler for opening a project (from menu or recent list)."""
        self._original_question_state = OriginalQuestionState()
        self._set_project_path(path)
        self._populate_recent_menu()
        # Restore state
        loaded_state = ProjectManager.read_project_state(path)
        if loaded_state.corrupt_backup is not None:
            self._report_corrupt_project_state(loaded_state.corrupt_backup)
        state = loaded_state.state
        cfg_yaml = path / "research_config.yaml"
        loaded_config = False
        if cfg_yaml.exists():
            try:
                self._load_research_config(cfg_yaml)
                loaded_config = True
                self.statusBar().showMessage(f"Loaded config from {cfg_yaml.name}", 4000)
            except Exception as exc:
                self._handle_project_config_failure(path, cfg_yaml, exc)
        apply_domain_snapshot(self, state, cfg_yaml if loaded_config else None)
        screening_restored = self._restore_screening_project_config(
            state, defaults=self.pipeline_config if loaded_config else None
        )
        fulltext_restored = self._restore_fulltext_project_config(state)
        if screening_restored or fulltext_restored:
            self._sync_settings_to_tabs()
        # Restore hypothesis components if saved
        window_state = as_dict(state.get("window_state"))
        self._original_question_state = OriginalQuestionState.from_window_state(
            window_state
        )
        saved_query = str(window_state.get("search_query") or "").strip()
        if not saved_query:
            saved_query = self._original_question_state.value
        if saved_query and hasattr(self, "sysys_nl_query"):
            self.sysys_nl_query.setText(saved_query)
        restore_window_widgets(self, window_state)
        self._refresh_pipeline_progress_from_disk(self.pipeline_config)
        ProjectMixin._load_meta_analysis_reports(self)
        self._load_final_review_report()
        self.statusBar().showMessage(f"Opened project: {path.name}", 4000)

    def _load_meta_analysis_reports(self) -> None:
        """Restore saved meta-analysis reports, including model token actions."""
        cfg = getattr(self, "pipeline_config", None)
        output_dir = getattr(cfg, "output_dir", None)
        if output_dir is None:
            return
        output = Path(output_dir)
        report_specs = (
            ("meta_summary", output / "meta_analysis_report.md", ""),
            (
                "meta_enhanced_report",
                output / "meta_analysis_enhanced.md",
                "Model enhancement was not run.",
            ),
            (
                "meta_model_audit",
                output / "meta_analysis_evidence" / "model_grouping_review.md",
                "Model-assisted semantic grouping was not run.",
            ),
        )
        for widget_name, path, missing_text in report_specs:
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                widget.setMarkdown(
                    path.read_text(encoding="utf-8") if path.exists() else missing_text
                )
            except OSError as exc:
                self._log(f"Could not restore meta-analysis report {path}: {exc}", "ERROR")

    def _report_corrupt_project_state(self, quarantined: Path) -> None:
        """Tell the user a corrupt project_state.json was moved aside (18-M5)."""
        self._log(f"Corrupt project_state.json moved to {quarantined}", "ERROR")
        QMessageBox.warning(
            self,
            "Project State Unreadable",
            "project_state.json could not be read and was moved to\n"
            f"{quarantined}\n\nThe project opens with default window state. "
            "Saved components and the original research question may be "
            "recoverable from that file.",
        )

    def _project_window_state(self) -> dict:
        """Window state persisted in project_state.json."""
        weighted_boolean_query = ""
        if hasattr(self, "boolean_query_preview"):
            weighted_boolean_query = self.boolean_query_preview.toPlainText().strip()
        window_state = {
            **self._project_question_window_state(),
            "weighted_boolean_query": weighted_boolean_query,
            "hypothesis_components": ProjectManager.serialize_hypothesis_components(
                self.hypothesis_components
            ),
            "paper_limit": self.paper_limit.value() if hasattr(self, "paper_limit") else 500,
        }
        if hasattr(self, "sysys_ft_resume"):
            window_state["stage7"] = {
                "claims_path": self.sysys_ft_corpus.text().strip(),
                "workers": self.sysys_ft_workers.value(),
                "resume": self.sysys_ft_resume.isChecked(),
            }
        return window_state

    def _save_project(self) -> bool:
        """File > Save Project. Returns False when nothing was saved."""
        if self._project_path is None:
            if self._project_change_blocked("Save Project"):
                return False
            # Prompt to create/choose a folder
            path = ProjectManager.open_project(self)
            if not path:
                return False
            self._set_project_path(path)
        assert self._project_path is not None
        self._sync_fulltext_controls_to_config()
        if not ProjectManager.save_project(
            self._project_path, self.pipeline_config, self._project_window_state()
        ):
            self._report_project_save_failure(self._project_path)
            return False
        self.statusBar().showMessage(
            f"Project saved to {self._project_path.name}", 4000
        )
        self._schedule_auto_backup("save")
        return True

    def _report_project_save_failure(self, project_path: Path) -> None:
        state_path = project_path / "project_state.json"
        self._log(f"Project NOT saved: could not write {state_path}", "ERROR")
        QMessageBox.critical(
            self,
            "Save Project Failed",
            f"Could not write {state_path}.\n\nThe previous saved state is "
            "unchanged. Check free disk space and folder permissions; the "
            "log has details.",
        )

    def _backup_project(self) -> None:
        """File > Backup Project..."""
        if self._project_path is None:
            QMessageBox.warning(
                self,
                "Backup Project",
                "Open or save a project before creating a backup.",
            )
            return

        self._save_project()
        destination = QFileDialog.getExistingDirectory(
            self,
            "Choose Backup Destination",
            str(Path.home()),
        )
        if not destination:
            return

        try:
            backup_path = ProjectManager.backup_project(
                self._project_path,
                Path(destination),
            )
            self._log(f"Project backup created: {backup_path}")
            self.statusBar().showMessage(f"Backup created: {backup_path.name}", 5000)
            QMessageBox.information(
                self,
                "Backup Complete",
                f"Project backup created:\n{backup_path}",
            )
        except Exception as exc:
            self._log(f"Backup failed: {exc}", "ERROR")
            QMessageBox.critical(self, "Backup Failed", str(exc))

    def _restore_project_backup(self) -> None:
        """File > Restore Project Backup..."""
        if self._project_change_blocked("Restore Project Backup"):
            return
        backup = self._choose_backup_source()
        if not backup:
            return
        try:
            metadata = ProjectManager.backup_metadata(Path(backup))
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            self._log(f"Restore failed: {exc}", "ERROR")
            QMessageBox.critical(self, "Restore Failed", str(exc))
            return
        choice = self._choose_restore_destination(Path(backup), metadata)
        if choice is None:
            return
        destination_path, overwrite = choice
        if overwrite and not self._confirm_lightweight_overwrite(
            Path(backup), metadata, destination_path
        ):
            return
        self._run_project_restore(Path(backup), destination_path, overwrite)

    def _default_backup_dir(self) -> Path:
        from PySide6.QtCore import QSettings

        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        for key in ("backup/last_path", "backup/destination"):
            value = cast(str, settings.value(key, "", type=str)) or ""
            if value and Path(value).expanduser().exists():
                return Path(value).expanduser()
        return Path.home()

    def _choose_backup_source(self) -> str:
        """Ask for a backup zip or folder; '' when cancelled."""
        default_backup_dir = self._default_backup_dir()
        source_choice = QMessageBox.question(
            self,
            "Restore Project Backup",
            "Restore from a .zip backup?\n\nChoose No to restore from a backup folder instead.",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if source_choice == QMessageBox.StandardButton.Cancel:
            return ""
        if source_choice == QMessageBox.StandardButton.Yes:
            start = default_backup_dir if default_backup_dir.is_dir() else default_backup_dir.parent
            backup, _ = QFileDialog.getOpenFileName(
                self,
                "Choose SystS Backup Zip",
                str(start),
                "SystS backups (*.zip);;All files (*)",
            )
            return backup
        return QFileDialog.getExistingDirectory(
            self,
            "Choose SystS Backup Folder",
            str(default_backup_dir),
        )

    def _choose_restore_destination(
        self, backup: Path, metadata: dict
    ) -> tuple[Path, bool] | None:
        """Ask where to restore; (destination, overwrite) or None when cancelled."""
        project_name = str(metadata.get("project_name") or backup.stem or "SystS_Project")
        project_name = ProjectManager._safe_backup_name(project_name)
        source_project_value = str(metadata.get("source_project_path") or "")
        source_path = Path(source_project_value).expanduser() if source_project_value else None
        restore_parent = (
            source_path.parent
            if source_path is not None and source_path.parent.exists()
            else Path.home()
        )
        destination = QFileDialog.getExistingDirectory(
            self,
            "Choose Folder to Restore Into",
            str(restore_parent),
        )
        if not destination:
            return None
        destination_path = Path(destination) / project_name
        if not (destination_path.exists() and any(destination_path.iterdir())):
            return destination_path, False
        return self._ask_overwrite_or_copy(destination_path)

    def _ask_overwrite_or_copy(self, destination_path: Path) -> tuple[Path, bool] | None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Restore Destination Not Empty")
        box.setText(f"{destination_path} is not empty.")
        box.setInformativeText("Overwrite it or restore to a timestamped copy?")
        overwrite_btn = box.addButton("Overwrite", QMessageBox.ButtonRole.DestructiveRole)
        copy_btn = box.addButton("Timestamped Copy", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked == overwrite_btn:
            return destination_path, True
        if clicked == copy_btn:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return destination_path.parent / f"{destination_path.name}_restored_{stamp}", False
        return None

    @staticmethod
    def _files_missing_from_backup(backup: Path, destination: Path) -> int:
        """Files in destination that a folder backup does not contain."""
        source = ProjectManager.resolve_backup_source(backup)
        if not source.is_dir():
            return 0
        return sum(
            1 for path in destination.rglob("*")
            if path.is_file() and not (source / path.relative_to(destination)).exists()
        )

    def _confirm_lightweight_overwrite(
        self, backup: Path, metadata: dict, destination_path: Path
    ) -> bool:
        """Warn before a lightweight auto-backup replaces a whole project (18-M6)."""
        if metadata.get("format") != ProjectManager.AUTO_BACKUP_FORMAT:
            return True
        missing = self._files_missing_from_backup(backup, destination_path)
        options = as_dict(metadata.get("options"))
        skipped = "PDFs and other full-text files, " if not options.get("include_fulltext") else ""
        answer = QMessageBox.warning(
            self,
            "Restore Lightweight Backup?",
            "This is a lightweight automatic backup. It leaves out "
            f"{skipped}caches, logs and files over "
            f"{options.get('max_file_mb', '?')} MB.\n\n"
            f"Overwriting {destination_path} deletes {missing} files "
            "that the backup does not contain. Restore to a timestamped copy "
            "instead to keep them.\n\nOverwrite anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _run_project_restore(self, backup: Path, destination_path: Path, overwrite: bool) -> None:
        try:
            restored_path = ProjectManager.restore_project(
                backup,
                destination_path,
                overwrite=overwrite,
            )
            self._open_project_path(restored_path)
            self._log(f"Project restored from backup: {restored_path}")
            QMessageBox.information(
                self,
                "Restore Complete",
                f"Project restored and opened:\n{restored_path}",
            )
        except Exception as exc:
            self._log(f"Restore failed: {exc}", "ERROR")
            QMessageBox.critical(self, "Restore Failed", str(exc))

    def _configure_auto_backup(self, default_enabled: bool | None = None) -> bool:
        """Show automatic backup options and persist them to QSettings."""
        from PySide6.QtCore import QSettings

        from gui.dialogs import AutoBackupOptionsDialog

        dialog = AutoBackupOptionsDialog(
            self,
            default_enabled=default_enabled,
            project_path=self._project_path,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False

        values = dialog.values()
        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        settings.setValue("backup/enabled", values["enabled"])
        settings.setValue("backup/prompt_disabled", not values["enabled"])
        settings.setValue("backup/destination", values["destination"])
        settings.setValue("backup/include_images", values["include_images"])
        settings.setValue("backup/include_fulltext", values["include_fulltext"])
        settings.setValue("backup/on_stage_complete", values["on_stage_complete"])
        settings.setValue("backup/on_exit", values["on_exit"])
        settings.setValue("backup/min_interval_minutes", values["min_interval_minutes"])
        settings.setValue("backup/max_file_mb", values["max_file_mb"])
        if values["enabled"]:
            self._log(f"Automatic project backup enabled: {values['destination']}")
            self.statusBar().showMessage("Automatic project backup enabled", 4000)
        else:
            self._log("Automatic project backup disabled")
            self.statusBar().showMessage("Automatic project backup disabled", 4000)
        if values["enabled"] and self._project_path is not None:
            self._schedule_auto_backup("auto-backup enabled", force=True)
        return values["enabled"]

    def _auto_backup_options(self) -> dict:
        from PySide6.QtCore import QSettings

        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        return {
            "enabled": settings.value("backup/enabled", False, type=bool),
            "destination": settings.value("backup/destination", "", type=str),
            "include_images": settings.value("backup/include_images", True, type=bool),
            "include_fulltext": settings.value("backup/include_fulltext", False, type=bool),
            "on_stage_complete": settings.value("backup/on_stage_complete", True, type=bool),
            "on_exit": settings.value("backup/on_exit", True, type=bool),
            "min_interval_minutes": cast(int, settings.value("backup/min_interval_minutes", 10, type=int)),
            "max_file_mb": cast(int, settings.value("backup/max_file_mb", 100, type=int)),
        }

    def _schedule_auto_backup(self, reason: str, force: bool = False) -> None:
        options = self._auto_backup_options()
        if not options["enabled"] or self._project_path is None:
            return
        if not force and reason != "save" and not options["on_stage_complete"]:
            return
        if self._backup_worker is not None and self._backup_worker.isRunning():
            self._pending_auto_backup_reason = reason
            return

        now = time.monotonic()
        min_interval = max(1, options["min_interval_minutes"]) * 60
        debounce_seconds = 5 if reason == "save" else 45
        if force:
            delay_ms = 0
        else:
            next_allowed = (
                self._last_auto_backup_monotonic + min_interval
                if self._last_auto_backup_monotonic > 0
                else now
            )
            delay_seconds = max(debounce_seconds, next_allowed - now)
            delay_ms = int(max(0, delay_seconds) * 1000)
        self._pending_auto_backup_reason = reason
        self._auto_backup_timer.start(delay_ms)

    def _run_auto_backup(
        self,
        reason: str = "scheduled",
        *,
        force: bool = False,
        synchronous: bool = False,
    ) -> None:
        from PySide6.QtCore import QSettings

        options = self._auto_backup_options()
        if not options["enabled"] or self._project_path is None:
            return
        if not options["destination"]:
            self._log("Auto-backup skipped: no destination configured", "WARNING")
            return
        if not force and self.pipeline_state != PipelineState.IDLE:
            self._retry_auto_backup_later(reason)
            return
        if self._backup_worker is not None and self._backup_worker.isRunning():
            self._pending_auto_backup_reason = reason
            return

        current_reason = self._pending_auto_backup_reason or reason
        self._pending_auto_backup_reason = ""

        backup_options = {
            "include_fulltext": options["include_fulltext"],
            "include_images": options["include_images"],
            "max_file_mb": options["max_file_mb"],
        }
        if synchronous:
            try:
                result = ProjectManager.sync_project_backup(
                    self._project_path,
                    Path(options["destination"]),
                    **backup_options,
                )
                self._record_auto_backup_result(result)
            except Exception as exc:
                self._log(f"Auto-backup failed: {exc}", "ERROR")
            return

        self._backup_worker = ProjectBackupWorker(
            self._project_path,
            Path(options["destination"]),
            **backup_options,
        )
        self._backup_worker.progress.connect(lambda msg: self._log(msg))
        self._backup_worker.finished_backup.connect(self._on_auto_backup_finished)
        self._backup_worker.error.connect(self._on_auto_backup_error)
        QSettings("Sysys", "HypothesisConsensusAnalyzer").setValue(
            "backup/last_reason",
            current_reason,
        )
        self._log(f"Auto-backup scheduled/run reason: {current_reason}")
        self._backup_worker.start()

    def _record_auto_backup_result(self, result: dict) -> None:
        from PySide6.QtCore import QSettings

        self._last_auto_backup_monotonic = time.monotonic()
        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        settings.setValue("backup/last_success", result.get("created_at", ""))
        settings.setValue("backup/last_path", result.get("backup_root", ""))
        copied = int(result.get("copied", 0) or 0)
        unchanged = int(result.get("unchanged", 0) or 0)
        self._log(
            f"Auto-backup complete: {copied} copied, {unchanged} unchanged → "
            f"{result.get('backup_root', '')}"
        )
        self.statusBar().showMessage(
            f"Auto-backup complete: {copied} changed files copied",
            5000,
        )

    def _retry_auto_backup_later(self, reason: str) -> None:
        """Keep the pending reason and retry once the pipeline may be idle (18-L5)."""
        self._pending_auto_backup_reason = self._pending_auto_backup_reason or reason
        self._auto_backup_timer.start(_BUSY_BACKUP_RETRY_MS)

    def _reschedule_pending_auto_backup(self) -> None:
        pending = self._pending_auto_backup_reason
        self._pending_auto_backup_reason = ""
        if pending:
            self._schedule_auto_backup(pending)

    def _on_auto_backup_finished(self, result: dict) -> None:
        self._record_auto_backup_result(result)
        self._backup_worker = None
        self._reschedule_pending_auto_backup()

    def _on_auto_backup_error(self, error: str) -> None:
        self._backup_worker = None
        self._log(f"Auto-backup failed: {error}", "ERROR")
        self.statusBar().showMessage("Auto-backup failed; see log", 5000)
        self._reschedule_pending_auto_backup()

    def _set_project_path(self, path: Path) -> None:
        self._project_path = path
        self.setWindowTitle(
            f"SystS — {path.name}"
        )
        # No try/except: a root that cannot be applied must not leave the
        # stages writing into the previous project silently.
        self.pipeline_config.project_root = path
        self.pipeline_config._resolve_paths()
        if hasattr(self, "score_corpus_path"):
            # Paths from the previous project — auto-filled or chosen by the
            # user — must never be read or overwritten from this one.
            for dropped in self._reset_stage_path_fields():
                self._log(f"Cleared stage path from the previous project: {dropped}")
            self._sync_all_stage_paths(self.pipeline_config)

    def _load_research_config(self, cfg_path: Path) -> None:
        """Load a research_config.yaml file into pipeline_config.

        Raises RuntimeError, leaving pipeline_config untouched, when the file
        cannot be loaded; there is deliberately no fallback to another config.
        """
        from gui.settings_persistence import explicit_config_fields, read_config_mapping

        try:
            new_cfg = type(self.pipeline_config).load_config(
                cfg_path,
                project_root=cfg_path.parent,
            )
            file_fields = explicit_config_fields(read_config_mapping(cfg_path))
        except Exception as exc:
            raise RuntimeError(f"Could not load {cfg_path.name}: {exc}") from exc
        self.pipeline_config = new_cfg
        self._restore_saved_keys(file_fields, cfg_path.name)
        self._populate_components_from_config()
        self._sync_settings_to_tabs()
        if hasattr(self, "score_corpus_path"):
            self._sync_all_stage_paths(self.pipeline_config)

    def _handle_project_config_failure(
        self, path: Path, cfg_yaml: Path, exc: Exception
    ) -> None:
        """Report a project config load failure and drop the previous config.

        Keeping the old pipeline_config would run the previous project's
        research question under the new project root.
        """
        self._log(f"Could not load {cfg_yaml}: {exc}", "ERROR")
        self.pipeline_config = type(self.pipeline_config)(project_root=path)
        self._restore_saved_keys()
        self._sync_settings_to_tabs()
        QMessageBox.critical(
            self,
            "Project Config Error",
            f"Could not load {cfg_yaml}:\n\n{exc}\n\n"
            "The project is open with default settings. Fix the file and "
            "reopen the project before running any stage.",
        )

    # ═══════════════════════════════════════════════════════════════
    # Feature H: Config Auto-Load
    # ═══════════════════════════════════════════════════════════════

    def _auto_load_config(self) -> None:
        """Scan cwd (and parent) for research_config.yaml and auto-load it."""
        cwd = Path.cwd()
        for search_dir in [cwd, cwd.parent]:
            cfg_path = search_dir / "research_config.yaml"
            if cfg_path.exists():
                try:
                    self._load_research_config(cfg_path)
                    try:
                        rel = cfg_path.relative_to(cwd)
                    except ValueError:
                        rel = cfg_path
                    self.statusBar().showMessage(
                        f"Auto-loaded config from {rel}", 5000
                    )
                    return
                except Exception as exc:
                    self._log(f"Could not auto-load {cfg_path}: {exc}", "ERROR")
                    self.statusBar().showMessage(
                        f"Could not load {cfg_path.name}; see log", 5000
                    )
                    return
