"""Config apply, save, and reset handlers for SettingsDialog (mixin)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from core.config import Config

logger = logging.getLogger("gui.dialogs")  # kept so log records keep the original logger name


class SettingsIOMixin:
    """Config-file and widget-to-Config handlers; mixed into SettingsDialog."""

    def _apply_to_config(self, cfg: Config):
        """Write dialog widget values back into a Config object."""
        root = self.settings_project_root.text().strip()
        # An open project's root changes only through File > Open Project,
        # which also moves project_state.json and the window title (18-L3).
        has_project = getattr(self.main_window, "_project_path", None) is not None
        if root and not has_project and Path(root) != cfg.project_root:
            cfg.project_root = Path(root)
            cfg.__post_init__()  # recalculate derived paths

        # API keys
        cfg.openai_api_key = self.settings_openai_key.text().strip()
        cfg.custom_api_key = self.settings_custom_key.text().strip()
        cfg.anthropic_api_key = self.settings_anthropic_key.text().strip()
        cfg.s2_api_key = self.settings_s2_key.text().strip()
        cfg.pubmed_api_key = self.settings_pubmed_key.text().strip()
        cfg.pubmed_email = self.settings_pubmed_email.text().strip()
        cfg.elicit_api_key = self.settings_elicit_key.text().strip()
        cfg.scopus_api_key = self.settings_scopus_key.text().strip()
        cfg.scopus_insttoken = self.settings_scopus_insttoken.text().strip()
        cfg.wos_api_key = self.settings_wos_key.text().strip()
        if hasattr(cfg, "wos_researcher_api_key"):
            cfg.wos_researcher_api_key = self.settings_wos_researcher_key.text().strip()

        # LLM provider
        cfg.llm_provider = self.settings_provider.currentText()
        cfg.ollama_model = self.settings_ollama_model.text().strip()
        cfg.ollama_base_url = self.settings_ollama_url.text().strip()
        cfg.custom_api_format = self._custom_format_from_widget()
        cfg.custom_base_url = self.settings_custom_url.text().strip()
        cfg.custom_model = self.settings_custom_model.text().strip()
        cfg.openai_model = self.settings_openai_model.text().strip()
        cfg.openai_base_url = self.settings_openai_url.text().strip()
        cfg.anthropic_model = self.settings_anthropic_model.text().strip()
        cfg.llm_context_tokens = self.settings_context_tokens.value()
        cfg.llm_max_output_tokens = self.settings_max_output_tokens.value()

        # Processing
        cfg.relevance_threshold = self.settings_threshold.value()
        cfg.concurrent_workers = self.settings_workers.value()
        cfg.score_batch_size = self.settings_batch_size.value()
        cfg.score_max_tokens = self.settings_score_max_tokens.value()
        cfg.stagger_delay = self.settings_stagger_delay.value()
        cfg.worker_timeout_seconds = self.settings_timeout.value()
        cfg.fulltext_streaming = self.settings_fulltext_streaming.isChecked()
        cfg.fulltext_download_batch_size = self.settings_fulltext_batch_size.value()
        cfg.paperqa_index_enabled = self.settings_paperqa_index_enabled.isChecked()
        cfg.paperqa_index_batch_size = self.settings_paperqa_batch_size.value()

    # ── Button handlers ────────────────────────────────────────────────

    def _on_edit_original_question_toggled(self, checked: bool) -> None:
        state = getattr(self.main_window, "_original_question_state", None)
        captured = bool(state and state.captured)
        self.settings_original_question.setReadOnly(captured and not checked)

    def _apply_original_question(self) -> None:
        question = self.settings_original_question.text().strip()
        state = getattr(self.main_window, "_original_question_state", None)
        if state is None or not question:
            return
        if state.captured:
            if self.settings_edit_original_question.isChecked():
                self.main_window._edit_original_research_question(question)
                # Re-lock the field and uncheck the gate so a repeated Apply
                # (without deliberately re-checking "Edit") cannot silently
                # re-save an accidental extra edit.
                self.settings_edit_original_question.setChecked(False)
                self.settings_original_question.setReadOnly(True)
            return
        state.set_prospective(question)
        if hasattr(self.main_window, "sysys_nl_query"):
            self.main_window.sysys_nl_query.setText(question)
        if getattr(self.main_window, "_project_path", None) is not None:
            self.main_window._save_project()

    @staticmethod
    def _has_value(val) -> bool:
        """True if val is a non-empty, non-whitespace string (or non-default for other types)."""
        if isinstance(val, str):
            return bool(val.strip())
        if isinstance(val, list):
            return bool(val)
        return val is not None

    def _on_save_config(self):
        """Save current dialog values to a JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            str(self.main_window.pipeline_config.project_root / "pipeline_config.json"),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            cfg = self.main_window.pipeline_config

            def _checked(name: str, default: bool) -> bool:
                widget = getattr(self, name, None)
                return widget.isChecked() if widget is not None else default

            def _number(name: str, default: int) -> int:
                widget = getattr(self, name, None)
                return int(widget.value()) if widget is not None else default

            # Read current widget values directly
            data = {
                "project_root": self.settings_project_root.text().strip(),
                # LLM provider
                "llm_provider": self.settings_provider.currentText(),
                "ollama_model": self.settings_ollama_model.text().strip(),
                "ollama_base_url": self.settings_ollama_url.text().strip(),
                "custom_api_format": self._custom_format_from_widget(),
                "custom_base_url": self.settings_custom_url.text().strip(),
                "custom_model": self.settings_custom_model.text().strip(),
                "openai_model": self.settings_openai_model.text().strip(),
                "openai_base_url": self.settings_openai_url.text().strip(),
                "anthropic_model": self.settings_anthropic_model.text().strip(),
                "llm_context_tokens": self.settings_context_tokens.value(),
                "llm_max_output_tokens": self.settings_max_output_tokens.value(),
                # API keys are intentionally omitted to avoid leaking secrets
                # in plain JSON files. Use environment variables or the
                # Settings dialog with "Remember keys" enabled instead.
                "pubmed_email": self.settings_pubmed_email.text().strip(),
                # Processing
                "relevance_threshold": self.settings_threshold.value(),
                "concurrent_workers": self.settings_workers.value(),
                "score_batch_size": self.settings_batch_size.value(),
                "score_max_tokens": self.settings_score_max_tokens.value(),
                "stagger_delay": self.settings_stagger_delay.value(),
                "worker_timeout_seconds": self.settings_timeout.value(),
                "fulltext_download_workers": int(
                    getattr(cfg, "fulltext_download_workers", 3)
                ),
                "fulltext_streaming": _checked(
                    "settings_fulltext_streaming",
                    bool(getattr(cfg, "fulltext_streaming", True)),
                ),
                "fulltext_download_batch_size": _number(
                    "settings_fulltext_batch_size",
                    int(getattr(cfg, "fulltext_download_batch_size", 50)),
                ),
                "paperqa_index_enabled": _checked(
                    "settings_paperqa_index_enabled",
                    bool(getattr(cfg, "paperqa_index_enabled", True)),
                ),
                "paperqa_index_batch_size": _number(
                    "settings_paperqa_batch_size",
                    int(getattr(cfg, "paperqa_index_batch_size", 100)),
                ),
            }
            # Security: never persist API keys in plaintext JSON. Only the
            # sensitive field names are kept here so we can tell the user which
            # keys were intentionally dropped.
            sensitive_fields = {
                "openai_api_key": self.settings_openai_key.text().strip(),
                "custom_api_key": self.settings_custom_key.text().strip(),
                "anthropic_api_key": self.settings_anthropic_key.text().strip(),
                "s2_api_key": self.settings_s2_key.text().strip(),
                "pubmed_api_key": self.settings_pubmed_key.text().strip(),
                "elicit_api_key": self.settings_elicit_key.text().strip(),
                "scopus_api_key": self.settings_scopus_key.text().strip(),
                "scopus_insttoken": self.settings_scopus_insttoken.text().strip(),
                "wos_api_key": self.settings_wos_key.text().strip(),
                "wos_researcher_api_key": self.settings_wos_researcher_key.text().strip(),
            }
            stripped = [name for name, value in sensitive_fields.items() if value]
            if stripped:
                logger.warning(
                    "API keys excluded from saved config JSON for security: %s",
                    ", ".join(stripped),
                )
                self.main_window._log(
                    f"Saved config without API keys for security: {', '.join(stripped)}. "
                    "Use environment variables or QSettings to persist keys.",
                    level="WARNING",
                )
            from core.domain_config import domain_config_snapshot
            data.update(domain_config_snapshot(cfg))
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self.main_window._log(f"Config saved to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", str(exc))

    def _on_reset_config(self):
        """Reset all settings to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            default_cfg = Config()
            self.settings_font_size.setValue(10)
            self._populate_from_config(default_cfg)
            self.main_window._log("Settings reset to defaults.")

    def _browse_project_root(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Project Root",
            self.settings_project_root.text() or str(Path.home()),
        )
        if path:
            self.settings_project_root.setText(path)
