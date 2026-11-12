"""Dialog windows for the Hypothesis Consensus Analyzer GUI.

This module is a facade. The dialogs live in sibling modules:
- gui/dialog_help.py: CitationGuideDialog, HelpDialog, CITATION_GUIDE_TEXT
- gui/dialog_backup.py: AutoBackupOptionsDialog
- gui/dialog_component_editor.py: ComponentEditor
- gui/dialog_welcome.py: WelcomeDialog
- gui/dialog_settings_build.py / gui/dialog_settings_io.py: SettingsDialog mixins
SettingsDialog itself stays here so the QSettings binding in this module is preserved.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

from core.config import Config
from gui.dialog_backup import AutoBackupOptionsDialog
from gui.dialog_common import _qs_bool, _qs_int, _qs_str
from gui.dialog_component_editor import (
    ComponentEditor,
    _parse_keyword_analysis_metrics,
    _split_keyword_analysis_description,
)
from gui.dialog_help import CITATION_GUIDE_TEXT, CitationGuideDialog, HelpDialog
from gui.dialog_settings_build import SettingsBuildMixin
from gui.dialog_settings_io import SettingsIOMixin
from gui.dialog_welcome import WelcomeDialog
from gui.settings_persistence import (
    LEGACY_LLM_QSETTINGS_KEYS,
    explicit_config_fields,
    read_config_mapping,
)

if TYPE_CHECKING:
    from gui.main_window import MainWindow

__all__ = [
    "CITATION_GUIDE_TEXT",
    "AutoBackupOptionsDialog",
    "CitationGuideDialog",
    "ComponentEditor",
    "HelpDialog",
    "SettingsDialog",
    "WelcomeDialog",
    "_parse_keyword_analysis_metrics",
    "_split_keyword_analysis_description",
]

logger = logging.getLogger(__name__)

# Credentials whose QSettings copy follows the "Remember keys" checkbox.
# keys/pubmed_email is a contact address and is always stored (18-L7).
_PUBMED_EMAIL_QSETTINGS_KEY = "keys/pubmed_email"


def _merge_loaded_config(
    cfg: Config, loaded: Config, explicit: frozenset[str], *, replace_research: bool
) -> None:
    """Apply exactly the fields a loaded config file set (18-M8).

    A full YAML research config replaces the research sections; a JSON
    settings export (which holds only some sections) merges into them.
    Empty strings in the file never erase a current value.
    """
    research = dict(loaded.research_config)
    if replace_research and research:
        cfg._research_config = research
        cfg.co_occurring_conditions = list(loaded.co_occurring_conditions)
    else:
        cfg._research_config = {**cfg.research_config, **research}
    for name in sorted(explicit):
        value = getattr(loaded, name)
        if name == "co_occurring_conditions" or SettingsIOMixin._has_value(value):
            setattr(cfg, name, value)
    cfg._resolve_paths()


def _load_config_into(dialog, path: Path) -> None:
    """Load ``path`` into the main window's config (see ``_on_load_config``)."""
    cfg = dialog.main_window.pipeline_config
    data = read_config_mapping(path)
    loaded = Config.from_mapping(data, project_root=cfg.project_root)
    explicit = explicit_config_fields(data)
    if data.get("project_root") and Path(str(data["project_root"])) != cfg.project_root:
        dialog.main_window._log(
            f"Ignored project_root in {path.name}; use File > Open Project to switch projects.",
            level="WARNING",
        )
    _merge_loaded_config(cfg, loaded, explicit, replace_research=path.suffix.lower() != ".json")

    from gui.settings_persistence import restore_saved_settings

    restore_saved_settings(
        cfg, QSettings("Sysys", "HypothesisConsensusAnalyzer"), keep_fields=explicit
    )
    dialog._populate_from_config(cfg)
    if hasattr(dialog.main_window, '_populate_components_from_config'):
        dialog.main_window._populate_components_from_config()
    if hasattr(dialog.main_window, '_sync_settings_to_tabs'):
        dialog.main_window._sync_settings_to_tabs(force=True)


# ---------------------------------------------------------------------------
# SettingsDialog  (new — replaces the old Settings tab)
# ---------------------------------------------------------------------------

class SettingsDialog(SettingsBuildMixin, SettingsIOMixin, QDialog):
    """
    Modal dialog containing all pipeline settings.

    Previously these controls lived in tab #10 of MainWindow as
    ``_build_settings_tab``.  The dialog reads from and writes back to
    ``MainWindow.pipeline_config``, keeping the same widget attribute names
    so existing settings-sync code continues to work.
    """

    def __init__(self, main_window: MainWindow, parent=None):
        super().__init__(parent or main_window)
        self.main_window = main_window
        self.setWindowTitle("Sysys Settings")
        self.setMinimumSize(620, 580)
        self.resize(680, 720)

        self._build_ui()
        self._populate_from_config(main_window.pipeline_config)

    # ── UI construction ────────────────────────────────────────────────

    def _populate_from_config(self, cfg: Config):
        """Fill all dialog widgets from the given Config object.

        Priority for API keys / LLM settings:
        1. Value on the Config object (from loaded file or env var)
        2. Value persisted in QSettings (from a previous Apply)
        3. Empty string
        """
        if hasattr(self, "settings_domain_summary"):
            research = cfg.research_config
            pack = research.get("defaults", {}).get("pattern_pack", "general")
            self.settings_domain_summary.setText(
                f"Domain vocabulary: {pack}. Custom patterns and rules are loaded "
                "from the research configuration."
            )
        s = QSettings("Sysys", "HypothesisConsensusAnalyzer")

        # Appearance
        self.settings_font_size.setValue(
            _qs_int(s, "appearance/font_size", 10)
        )

        self.settings_project_root.setText(str(cfg.project_root))
        self._lock_project_root_when_project_open()

        # Helper: use config value if non-empty, else fall back to QSettings
        def _best(cfg_val: str, qsettings_key: str) -> str:
            if cfg_val and cfg_val.strip():
                return cfg_val.strip()
            saved = _qs_str(s, qsettings_key, "")
            return saved.strip() if saved else ""

        # API keys  (config → QSettings fallback)
        self.settings_openai_key.setText(_best(cfg.openai_api_key, "keys/openai_api_key"))
        self.settings_custom_key.setText(_best(cfg.custom_api_key, "keys/custom_api_key"))
        self.settings_anthropic_key.setText(_best(cfg.anthropic_api_key, "keys/anthropic_api_key"))
        self.settings_s2_key.setText(_best(cfg.s2_api_key, "keys/s2_api_key"))
        self.settings_pubmed_key.setText(_best(cfg.pubmed_api_key, "keys/pubmed_api_key"))
        self.settings_pubmed_email.setText(_best(cfg.pubmed_email, "keys/pubmed_email"))
        self.settings_elicit_key.setText(_best(cfg.elicit_api_key, "keys/elicit_api_key"))
        self.settings_scopus_key.setText(_best(cfg.scopus_api_key, "keys/scopus_api_key"))
        self.settings_scopus_insttoken.setText(_best(cfg.scopus_insttoken, "keys/scopus_insttoken"))
        self.settings_wos_key.setText(_best(cfg.wos_api_key, "keys/wos_api_key"))
        self.settings_wos_researcher_key.setText(_best(getattr(cfg, "wos_researcher_api_key", ""), "keys/wos_researcher_api_key"))

        # Restore remember-keys preference (default: off for safety)
        remember = _qs_bool(s, "keys/remember_keys", False)
        self.settings_remember_keys.setChecked(remember)

        # LLM provider  (config → QSettings fallback)
        provider = _best(cfg.llm_provider, "llm/provider") or "ollama"
        idx = self.settings_provider.findText(provider)
        if idx >= 0:
            self.settings_provider.setCurrentIndex(idx)

        self.settings_ollama_model.setText(_best(cfg.ollama_model, "llm/ollama_model"))
        self.settings_ollama_url.setText(_best(cfg.ollama_base_url, "llm/ollama_url"))
        self.settings_custom_url.setText(_best(cfg.custom_base_url, "llm/custom_url"))
        fmt_idx = self.settings_custom_format.findData(
            _best(cfg.custom_api_format, "llm/custom_format")
        )
        self.settings_custom_format.setCurrentIndex(max(fmt_idx, 0))
        self.settings_custom_model.setText(_best(cfg.custom_model, "llm/custom_model"))
        self.settings_openai_model.setText(_best(cfg.openai_model, "llm/openai_model"))
        self.settings_openai_url.setText(_best(cfg.openai_base_url, "llm/openai_url"))
        self.settings_anthropic_model.setText(_best(cfg.anthropic_model, "llm/anthropic_model"))
        saved_context = _qs_int(s, "llm/context_tokens", 0)
        self.settings_context_tokens.setValue(
            int(getattr(cfg, "llm_context_tokens", 0) or saved_context or 0)
        )
        saved_output_limit = _qs_int(s, "llm/max_output_tokens", 0)
        self.settings_max_output_tokens.setValue(
            int(getattr(cfg, "llm_max_output_tokens", 0) or saved_output_limit or 0)
        )

        # Processing
        self.settings_threshold.setValue(cfg.relevance_threshold)
        self.settings_workers.setValue(cfg.concurrent_workers)
        self.settings_batch_size.setValue(cfg.score_batch_size)
        self.settings_score_max_tokens.setValue(cfg.score_max_tokens)
        self.settings_stagger_delay.setValue(cfg.stagger_delay)
        self.settings_timeout.setValue(cfg.worker_timeout_seconds)
        self.settings_fulltext_streaming.setChecked(bool(cfg.fulltext_streaming))
        self.settings_fulltext_batch_size.setValue(
            int(cfg.fulltext_download_batch_size)
        )
        self.settings_paperqa_index_enabled.setChecked(
            bool(cfg.paperqa_index_enabled)
        )
        self.settings_paperqa_batch_size.setValue(
            int(cfg.paperqa_index_batch_size)
        )
        self.settings_fulltext_batch_size.setDisabled(
            self.settings_fulltext_streaming.isChecked()
        )
        self.settings_paperqa_batch_size.setEnabled(
            self.settings_paperqa_index_enabled.isChecked()
        )

        # Stage 2 advanced-LLM toggle (QSettings only — not part of Config)
        self.settings_show_score_advanced.setChecked(
            _qs_bool(s, "score/show_advanced_llm", False)
        )

        state = getattr(self.main_window, "_original_question_state", None)
        has_project = getattr(self.main_window, "_project_path", None) is not None
        prospective = state.value if state is not None else ""
        if not prospective and hasattr(self.main_window, "sysys_nl_query"):
            prospective = self.main_window.sysys_nl_query.text().strip()
        self.settings_original_question.setText(prospective)
        self.settings_original_question.setReadOnly(
            bool(state and state.captured)
        )
        self.settings_edit_original_question.setChecked(False)
        self.settings_edit_original_question.setEnabled(
            bool(state and state.captured)
        )
        self.original_question_group.setEnabled(has_project)
        self.original_question_group.setToolTip(
            "" if has_project
            else "Open or create a project to store an original question."
        )

    def _lock_project_root_when_project_open(self) -> None:
        """An open project's root changes only through File > Open Project (18-L3)."""
        has_project = getattr(self.main_window, "_project_path", None) is not None
        self.settings_project_root.setReadOnly(has_project)
        self.settings_project_root_browse.setEnabled(not has_project)
        self.settings_project_root.setToolTip(
            "Use File > Open Project to switch projects." if has_project else ""
        )

    # ── Write back to config ───────────────────────────────────────────

    def _on_apply(self):
        """Apply current dialog values to MainWindow's pipeline_config.

        The original question is applied last because it may save the
        project, and the save must contain the values just applied (18-L3).
        """
        self._apply_to_config(self.main_window.pipeline_config)

        # Persist API keys and LLM settings to QSettings so they survive restarts
        self._save_keys_to_qsettings()

        # Apply font size change live
        font_size = self.settings_font_size.value()
        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        settings.setValue("appearance/font_size", font_size)

        # Persist Stage 2 advanced-LLM toggle
        show_advanced = self.settings_show_score_advanced.isChecked()
        settings.setValue("score/show_advanced_llm", show_advanced)

        from gui.theme import get_stylesheet
        app = QApplication.instance()
        if app:
            # QApplication.instance() is typed to return QCoreApplication (the
            # base class); this process only ever constructs a QApplication.
            app = cast(QApplication, app)
            app.setStyleSheet(get_stylesheet(font_size))
            app.setFont(QFont("Tahoma", font_size))

        # Sync settings into all stage tab widgets. Preferences Apply is an
        # explicit global action, so it must override stale per-stage edits and
        # push provider/model names everywhere.
        if hasattr(self.main_window, '_sync_settings_to_tabs'):
            self.main_window._sync_settings_to_tabs(force=True)

        # Apply Stage 2 advanced-LLM visibility immediately
        if hasattr(self.main_window, '_apply_score_advanced_mode'):
            self.main_window._apply_score_advanced_mode(show_advanced)

        self._apply_original_question()

        self.main_window._log("Settings applied.", level="INFO")
        if hasattr(self.main_window, 'statusBar'):
            self.main_window.statusBar().showMessage("Settings applied", 3000)

    def _on_save(self):
        """Apply settings and close the Preferences dialog."""
        self._on_apply()
        self.accept()

    def _save_keys_to_qsettings(self):
        """Persist API keys and LLM settings so they survive app restarts.

        API keys are only written to QSettings when the user has explicitly
        opted in via the "Remember API keys" checkbox.  LLM provider/model
        settings (non-secret) are always persisted.
        """
        s = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        remember = self.settings_remember_keys.isChecked()
        s.setValue("keys/remember_keys", remember)

        # Map QSettings key → widget accessor for all API key fields
        key_widgets = {
            "keys/openai_api_key":   self.settings_openai_key,
            "keys/custom_api_key":   self.settings_custom_key,
            "keys/anthropic_api_key": self.settings_anthropic_key,
            "keys/s2_api_key":       self.settings_s2_key,
            "keys/pubmed_api_key":   self.settings_pubmed_key,
            "keys/elicit_api_key":   self.settings_elicit_key,
            "keys/scopus_api_key":   self.settings_scopus_key,
            "keys/scopus_insttoken": self.settings_scopus_insttoken,
            "keys/wos_api_key":      self.settings_wos_key,
            "keys/wos_researcher_api_key": self.settings_wos_researcher_key,
        }
        if remember:
            for qkey, widget in key_widgets.items():
                s.setValue(qkey, widget.text().strip())
        else:
            # Clear any previously stored keys when the user opts out
            for qkey in key_widgets:
                s.remove(qkey)
        # Pre-"custom" provider keys were migrated on load; don't leave
        # plaintext copies behind.
        for qkey in LEGACY_LLM_QSETTINGS_KEYS.values():
            s.remove(qkey)
        s.setValue(_PUBMED_EMAIL_QSETTINGS_KEY, self.settings_pubmed_email.text().strip())

        s.setValue("llm/provider", self.settings_provider.currentText())
        s.setValue("llm/ollama_model", self.settings_ollama_model.text().strip())
        s.setValue("llm/ollama_url", self.settings_ollama_url.text().strip())
        s.setValue("llm/custom_format", self._custom_format_from_widget())
        s.setValue("llm/custom_url", self.settings_custom_url.text().strip())
        s.setValue("llm/custom_model", self.settings_custom_model.text().strip())
        s.setValue("llm/openai_model", self.settings_openai_model.text().strip())
        s.setValue("llm/openai_url", self.settings_openai_url.text().strip())
        s.setValue("llm/anthropic_model", self.settings_anthropic_model.text().strip())
        s.setValue("llm/context_tokens", self.settings_context_tokens.value())
        s.setValue("llm/max_output_tokens", self.settings_max_output_tokens.value())

    def _on_load_config(self):
        """Load a YAML/JSON config file into the current config.

        The file goes through ``Config.from_mapping`` (legacy migration,
        validation, every research section). Exactly the fields the file
        sets are applied, including values equal to a default; empty
        strings never erase a current value, and the open project's root is
        kept. Saved LLM preferences fill only fields the file left unset.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            str(self.main_window.pipeline_config.project_root),
            "Config files (*.yaml *.yml *.json);;All files (*)",
        )
        if not path:
            return
        try:
            _load_config_into(self, Path(path))
            self.main_window._log(f"Config loaded from {path}")
        except Exception as exc:
            logger.exception("Failed to load config")
            QMessageBox.warning(self, "Load Error", str(exc))
