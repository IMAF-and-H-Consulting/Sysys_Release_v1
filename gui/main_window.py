"""
Main application window for the Hypothesis Consensus Analyzer.

Contains the MainWindow class with 13 tabs (Settings tab removed — now a
dialog accessible from File > Settings...) and a File menu bar.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QThread, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.config import Config
from core.models import (
    AnalysisSession,
)
from gui.main_window_common import (  # noqa: F401  (re-exported for existing importers)
    LLM_PROVIDER_CHOICES,
    PipelineState,
    _make_file_row,
    _make_llm_backend_row,
    _paths_resolve_under_root,
    original_question_for_model,
)
from gui.main_window_config_sync import ConfigSyncMixin
from gui.main_window_corpus import CorpusMixin
from gui.main_window_evidence import EvidenceMixin
from gui.main_window_human_relevance import HumanRelevanceMixin
from gui.main_window_legacy_workers import LegacyWorkersMixin
from gui.main_window_neuro import NeuroMixin
from gui.main_window_project import ProjectMixin
from gui.main_window_reports import ReportsMixin
from gui.main_window_run_all import RunAllMixin
from gui.main_window_shell import ShellMixin
from gui.main_window_tabs_analysis import AnalysisTabsMixin
from gui.main_window_tabs_screening import ScreeningTabsMixin
from gui.original_question import OriginalQuestionState
from gui.pipeline_board import PipelineBoardDialog, empty_stage_progress
from gui.robustness_verify_summary import (
    NO_LLM_LOG_LINE,
    StageSummary,
    summarize_robustness,
    summarize_verify,
)
from gui.synthesis_summary import summarize_synthesis
from gui.theme import COLORS
from gui.widgets import MetricCard
from gui.workers import (
    AnalysisWorker,
    FullTextRetrievalWorker,
    MetaAnalysisWorker,
    PipelineSearchWorker,
    PipelineStageWorker,
    ProjectBackupWorker,
    TextScalingWorker,
)

logger = logging.getLogger(__name__)

# Bounded wait for each running worker when the window closes.
CLOSE_WORKER_WAIT_MS = 5000
BACKUP_CLOSE_WAIT_MS = 3000

# Worker slots stopped (cancelled and waited for) before the window closes.
_CLOSE_WORKER_SLOTS = (
    ("pipeline stage", "_pipeline_worker"),
    ("Stage 1 search", "_search_worker"),
    ("keyword analysis", "_analysis_worker"),
    ("full-text retrieval", "_ft_worker"),
    ("text scaling", "_scaling_worker"),
    ("coordinate extraction", "_neuro_coord_worker"),
    ("NeuroQuery analysis", "_neuro_nq_worker"),
    ("meta-analysis", "_meta_worker"),
)


def _load_startup_config() -> tuple[Config, str]:
    """Load research_config.yaml; on failure log it and return defaults plus the error."""
    try:
        return Config.load_config(), ""
    except Exception as exc:
        logger.exception("Could not load research_config.yaml at startup; using defaults")
        return Config(), f"{type(exc).__name__}: {exc}"


# ── MainWindow ────────────────────────────────────────────────────────────

class MainWindow(
    CorpusMixin,
    NeuroMixin,
    ReportsMixin,
    EvidenceMixin,
    ScreeningTabsMixin,
    AnalysisTabsMixin,
    ShellMixin,
    ProjectMixin,
    ConfigSyncMixin,
    HumanRelevanceMixin,
    RunAllMixin,
    LegacyWorkersMixin,
    QMainWindow,
):
    """
    Main application window.

    14 tabs (Search, Score & Filter, Extract & Validate, Audit, Synthesis,
    Fulltext, Robustness & Verify, Text Scaling, Evidence Map,
    Results Dashboard, Neuroimaging, Compare Runs, Report & Export, Corpus).

    Settings are accessed via File > Settings... (Ctrl+,) which opens
    SettingsDialog.
    """

    def __init__(self):
        super().__init__()
        from gui.settings_persistence import migrate_legacy_settings
        migrate_legacy_settings()
        self.setWindowTitle("Hypothesis Consensus Analyzer \u2014 Sysys Merged Pipeline")
        self.setMinimumSize(QSize(1280, 850))
        self.resize(1440, 950)

        # ── Search analysis state ──
        self.session = AnalysisSession()
        self.hypothesis_components: list = []
        self.articles: dict = {}
        self._analysis_worker: AnalysisWorker | None = None
        self._ft_worker: FullTextRetrievalWorker | None = None
        self._scaling_worker: TextScalingWorker | None = None
        self._wordfish_model = None
        self._stylometry_profiles: dict = {}
        self._corpus_stats: dict = {}
        self._evidence_table_rows: list[dict] = []
        self._evidence_table_source: str = ""
        self._ft_article_row: dict = {}

        # ── Sysys pipeline state ──
        # Auto-load research_config.yaml if it exists in the project root;
        # a load failure is reported once the log handler is installed.
        self.pipeline_config: Config
        self.pipeline_config, self._config_load_error = _load_startup_config()
        self._restore_saved_keys()  # layer QSettings keys onto Config
        self.pipeline_state: str = PipelineState.IDLE
        self._pipeline_worker: PipelineStageWorker | None = None
        self._search_worker: PipelineSearchWorker | None = None
        self._meta_worker: MetaAnalysisWorker | None = None
        self._pipeline_corpus: list = []
        self._pipeline_board: PipelineBoardDialog | None = None
        self._pipeline_progress: dict[int, dict[str, str]] = empty_stage_progress()
        self._run_all_queue: list[str] = []
        self._run_all_active: bool = False
        self._project_path: Path | None = None
        self._original_question_state = OriginalQuestionState()
        self._backup_worker: ProjectBackupWorker | None = None
        self._pending_auto_backup_reason: str = ""
        self._last_auto_backup_monotonic: float = 0.0
        self._auto_backup_timer = QTimer(self)
        self._auto_backup_timer.setSingleShot(True)
        self._auto_backup_timer.timeout.connect(
            lambda: self._run_auto_backup(reason="scheduled")
        )
        self._sysys_search_runs: int = 0
        self._keyword_analysis_ready: bool = False

        # ── Build UI ──
        self._build_menu_bar()
        self._build_tabs()
        self._build_status_bar()
        self._build_output_console()
        self._build_log_window()
        self._install_log_handler()
        self._report_startup_config_error()

        # Populate hypothesis components and sync settings to all tabs.
        # Restore persisted LLM defaults (provider/model/base URLs) and any
        # remembered API keys from QSettings first, so the defaults the user
        # last set in Preferences show up in the stage rows on launch.
        self._populate_components_from_config()
        self._restore_saved_keys()
        self._sync_settings_to_tabs()
        self._update_search_workflow_buttons()

        self._show_welcome_if_first_run()
        self._maybe_prompt_auto_backup("first_launch")
        self._setup_shortcuts()
        self._auto_load_config()

    # ── Tab widget ────────────────────────────────────────────────────

    def _build_tabs(self):
        """Build tabs with At-a-Glance panel above (Settings tab removed — now a dialog)."""
        # ── Central widget: vertical container ──
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        # ── Feature D: At-a-Glance panel ──
        self._glance_frame = QFrame()
        self._glance_frame.setStyleSheet(
            f"QFrame {{ background-color: {COLORS['bg']}; "
            f"border-top: 2px solid #404040; border-left: 2px solid #404040; "
            f"border-bottom: 2px solid #ffffff; border-right: 2px solid #ffffff; }}"
        )
        glance_outer = QHBoxLayout(self._glance_frame)
        glance_outer.setContentsMargins(6, 4, 6, 4)
        glance_outer.setSpacing(8)

        self._card_corpus = MetricCard("Corpus", "0 articles")
        self._card_relevant = MetricCard("Relevant", "0 papers")
        self._card_grade = MetricCard("GRADE", "—")
        self._card_stage = MetricCard("Stage", "Ready")

        glance_outer.addWidget(self._card_corpus)
        glance_outer.addWidget(self._card_relevant)
        glance_outer.addWidget(self._card_grade)
        glance_outer.addWidget(self._card_stage)
        glance_outer.addStretch()

        self._glance_toggle_btn = QPushButton("▲ Hide")
        self._glance_toggle_btn.setFixedWidth(60)
        self._glance_toggle_btn.clicked.connect(self._toggle_glance_panel)
        glance_outer.addWidget(self._glance_toggle_btn)

        central_layout.addWidget(self._glance_frame)

        # ── Tab widget ──
        self.tabs = QTabWidget()
        central_layout.addWidget(self.tabs, 1)

        self.setCentralWidget(central)

        # Tab 0: Search
        self.tabs.addTab(self._build_search_tab(), "Search")
        # Tab 1: Score & Filter (also hosts Stage 2c Statistical Extraction in
        # its top-right column, and Human Relevance Calibration below)
        self.tabs.addTab(self._build_score_filter_tab(), "Score && Filter")
        # Tab 2: Extract & Validate
        self.tabs.addTab(self._build_extract_validate_tab(), "Extract && Validate")
        # Tab 3: Audit (Stage 5)
        self.tabs.addTab(self._build_audit_tab(), "Audit")
        # Tab 4: Synthesis (Stage 6 — abstract, hypothesis rationale,
        # source citations, and evidence network get their own roomy tab)
        self.tabs.addTab(self._build_synthesis_tab(), "Synthesis")
        # Tab 5: Fulltext
        self.tabs.addTab(self._build_fulltext_tab(), "Fulltext")
        # Tab 6: Robustness & Verify
        self.tabs.addTab(self._build_robustness_verify_tab(), "Robustness && Verify")
        # Tab 7: Text Scaling
        self.tabs.addTab(self._build_scaling_tab(), "Text Scaling")
        # Tab 8: Evidence Map
        self.tabs.addTab(self._build_evidence_tab(), "Evidence Map")
        # Tab 9: Results Dashboard
        self.tabs.addTab(self._build_alignment_tab(), "Results Dashboard")
        # Tab 10: Neuroimaging
        self.tabs.addTab(self._build_neuroimaging_tab(), "Neuroimaging")
        # Tab 11: Compare Runs
        self.tabs.addTab(self._build_compare_runs_tab(), "Compare Runs")
        # Tab 12: Report & Export
        self.tabs.addTab(self._build_report_export_tab(), "Report && Export")
        # Tab 13: Corpus Browser (Feature B)
        self.tabs.addTab(self._build_corpus_tab(), "Corpus")
        self._load_final_review_report()

    def closeEvent(self, event) -> None:
        """Stop running workers, persist project state, then close."""
        if not self._prepare_to_close():
            event.ignore()
            return
        super().closeEvent(event)

    def _prepare_to_close(self) -> bool:
        """Return True when the window may close; False keeps it open."""
        running = self._running_workers()
        if running and not self._stop_workers_for_close(running):
            return False
        self._auto_backup_timer.stop()
        backup_idle = self._wait_for_backup_worker()
        # A stage may have been writing; do not copy half-written outputs.
        self._persist_project_on_close(run_exit_backup=backup_idle and not running)
        if backup_idle:
            return True
        return self._confirm_close(
            "A project backup is still running. Quit anyway? "
            "The backup copy may be incomplete."
        )

    def _running_workers(self) -> list[tuple[str, QThread]]:
        workers = ((label, getattr(self, attr, None)) for label, attr in _CLOSE_WORKER_SLOTS)
        return [(label, w) for label, w in workers if w is not None and w.isRunning()]

    def _stop_workers_for_close(self, running: list[tuple[str, QThread]]) -> bool:
        """Ask, cancel and wait (bounded) for running workers; False vetoes closing."""
        names = ", ".join(label for label, _worker in running)
        if not self._confirm_close(
            f"Still running: {names}. Cancel and quit? "
            "Completed checkpoints are kept; the current step is lost."
        ):
            return False
        self._abort_run_all("RUN ALL cancelled: window closed.")
        for _label, worker in running:
            cancel = getattr(worker, "cancel", None)  # AnalysisWorker has none
            if cancel is not None:
                cancel()
        stuck = [label for label, w in running if not w.wait(CLOSE_WORKER_WAIT_MS)]
        if not stuck:
            return True
        self._log(f"Workers still running at close: {', '.join(stuck)}", "WARNING")
        return self._confirm_close(
            f"{', '.join(stuck)} did not stop within "
            f"{CLOSE_WORKER_WAIT_MS // 1000} s. Quit anyway? Outputs may be incomplete."
        )

    def _wait_for_backup_worker(self) -> bool:
        """Return True when no auto-backup is running (after a bounded wait)."""
        if self._backup_worker is None or not self._backup_worker.isRunning():
            return True
        self._log("Waiting briefly for active auto-backup before exit...")
        if not self._backup_worker.wait(BACKUP_CLOSE_WAIT_MS):
            self._log("Auto-backup still running at exit.", "WARNING")
            return False
        self._backup_worker = None
        return True

    def _persist_project_on_close(self, run_exit_backup: bool) -> None:
        """Save the project and, when safe, run the optional exit backup."""
        if self._project_path is None:
            return
        try:
            self._save_project()
            options = self._auto_backup_options()
            if not (options["enabled"] and options["on_exit"]):
                return
            if run_exit_backup:
                self._run_auto_backup(reason="exit", force=True, synchronous=True)
            else:
                self._log("Skipping exit backup: a worker was still running.", "WARNING")
        except Exception as exc:
            self._log(f"Close-time backup failed: {exc}", "ERROR")

    def _confirm_close(self, text: str) -> bool:
        reply = QMessageBox.question(
            self, "Quit", text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _report_startup_config_error(self) -> None:
        """Surface a research_config.yaml load failure from __init__ (17-M4)."""
        error = getattr(self, "_config_load_error", "")
        if not error:
            return
        self._log(f"Could not load research_config.yaml: {error}", "ERROR")
        self.statusBar().showMessage(
            "research_config.yaml failed to load; default settings are in use", 0
        )
        QTimer.singleShot(0, self._show_startup_config_error_dialog)

    def _show_startup_config_error_dialog(self) -> None:
        QMessageBox.warning(
            self,
            "Config Error",
            f"Could not load research_config.yaml:\n\n{self._config_load_error}\n\n"
            "The app started with default settings (no research question, query "
            "families or domains). Fix the file and reload it before running a stage.",
        )

    # ── Settings methods (delegate to SettingsDialog) ─────────────────
    #
    # These methods previously operated on widgets embedded in the
    # Settings tab.  They now operate on MainWindow.pipeline_config
    # directly, and the SettingsDialog handles the UI interaction.

    # ── Search and analysis worker launchers ──────────────────────────

    def _sync_analysis_articles_from_corpus(self, corpus: list) -> None:
        """Refresh keyword-analysis Article objects from the latest search corpus."""
        from core.models import Article

        self.articles = {}
        for idx, p in enumerate(corpus or []):
            if not isinstance(p, dict):
                continue
            title = p.get("title") or ""
            art_id = str(p.get("id") or p.get("doi") or p.get("pmid") or title[:60])
            if not art_id:
                art_id = f"article_{idx}"
            if art_id in self.articles:
                # Keys built from shared title prefixes must not drop papers.
                art_id = f"{art_id}#{idx}"
            authors = p.get("authors", [])
            if not isinstance(authors, list):
                authors = []
            self.articles[art_id] = Article(
                id=art_id,
                title=title,
                authors=authors,
                abstract=p.get("abstract", "") or "",
                full_text=p.get("full_text", "") or "",
                year=p.get("year"),
                journal=p.get("journal", "") or "",
                doi=p.get("doi", "") or "",
                pmid=p.get("pmid", "") or "",
                citations=p.get("citationCount", 0) or p.get("citations", 0) or 0,
                open_access=bool(
                    p.get("open_access")
                    or p.get("is_oa")
                    or p.get("openalex_is_oa")
                    or p.get("s2_open_access_pdf_url")
                    or p.get("scopus_openaccess") == "1"
                ),
                full_text_url=(
                    p.get("openalex_best_oa_pdf_url")
                    or p.get("openalex_oa_url")
                    or p.get("s2_open_access_pdf_url")
                    or p.get("oa_url")
                    or p.get("full_text_url")
                    or ""
                ),
                has_full_text=bool(p.get("has_full_text") or p.get("full_text")),
                url=p.get("url", "") or "",
                keywords=p.get("keywords", []) if isinstance(p.get("keywords", []), list) else [],
            )

    def _start_fetch(self):
        """Fetch Articles button — download full text for the existing corpus.

        This is a download-only action: it does NOT run a new literature search.
        Use "Run Sysys Search" to build/extend the corpus first. To search and
        download in one step, tick the "Download full text" checkbox before
        running the search.
        """
        self._log("═" * 60)
        self._log("FETCH ARTICLES (DOWNLOAD FULL TEXT)")
        self._log("═" * 60)

        # Ensure a corpus is available; load from disk if the in-memory copy is
        # empty (e.g. right after opening a project).
        if not self._pipeline_corpus:
            corpus_path = self.pipeline_config.corpus_dir / "corpus.json"
            if corpus_path.exists():
                try:
                    import json as _json
                    with corpus_path.open(encoding="utf-8") as f:
                        self._pipeline_corpus = _json.load(f)
                    self._log(
                        f"Loaded {len(self._pipeline_corpus)} papers from {corpus_path}"
                    )
                except Exception as exc:
                    self._log(f"Could not load corpus: {exc}", "ERROR")
                    return

        if not self._pipeline_corpus:
            self._log(
                "No corpus to download. Run Sysys Search first to build a corpus.",
                "ERROR",
            )
            return

        # Delegate to corpus full-text enrichment (download only, no search).
        self._start_sysys_corpus_fulltext(triggered_by_fetch=True)

    def _start_analysis(self):
        """Run Keyword Analysis — score fetched articles against hypothesis components."""
        self._log("═" * 60)
        self._log("KEYWORD ANALYSIS")
        self._log("═" * 60)

        if self._analysis_worker is not None and self._analysis_worker.isRunning():
            self._log("Keyword analysis is already running; wait for it to finish.",
                      "WARNING")
            return

        if not self.articles and not self._pipeline_corpus:
            self._log("No articles loaded. Run Sysys Search or Fetch Articles first.", "ERROR")
            return

        nl_query = self.sysys_nl_query.text().strip()
        if nl_query and not self.hypothesis_components:
            self._sync_query_components_from_question(nl_query)

        if not self.hypothesis_components:
            self._log(
                "No hypothesis components. Enter a question, edit components, "
                "or reset components first.",
                "ERROR",
            )
            return

        # Build Article objects from pipeline corpus if needed
        if not self.articles and self._pipeline_corpus:
            self._sync_analysis_articles_from_corpus(self._pipeline_corpus)

        self._log(f"Articles: {len(self.articles)}")
        self._log(f"Components: {len(self.hypothesis_components)}")
        self._log(f"Total pairs: {len(self.articles) * len(self.hypothesis_components)}")

        self.analysis_btn.setEnabled(False)
        self.search_progress.setVisible(True)
        self.search_progress.setRange(0, 0)

        # Run in a worker thread
        self._analysis_worker = AnalysisWorker(
            articles=self.articles,
            components=self.hypothesis_components,
        )
        self._analysis_worker.finished_analysis.connect(self._on_analysis_done)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.cancelled.connect(self._on_analysis_cancelled)
        self._analysis_worker.start()

    def _on_analysis_done(self, session):
        from analysis.scoring import update_component_weights_from_keyword_analysis

        self.session = session
        stats = update_component_weights_from_keyword_analysis(session)
        self.hypothesis_components = session.hypothesis_components
        self._refresh_component_list()
        self.analysis_btn.setEnabled(True)
        self.search_progress.setVisible(False)

        self._log("═" * 60)
        self._log("KEYWORD ANALYSIS COMPLETE")
        for ar in session.alignment_results:
            self._log(f"  {ar.component_label}: support={ar.support_score:.2f} "
                      f"({ar.supporting_count} sup, {ar.contradicting_count} contra, "
                      f"{ar.neutral_count} neutral)")
        if stats:
            ranked = sorted(
                self.hypothesis_components,
                key=lambda comp: getattr(comp, "weight", 1.0),
                reverse=True,
            )
            self._log("Updated query-term weights:")
            for comp in ranked[:12]:
                self._log(f"  {comp.label}: weight={comp.weight:.3f}")
        self._log("═" * 60)

        # Update dashboard if methods exist
        for name in ("_update_alignment_dashboard", "_refresh_evidence_table"):
            refresh = getattr(self, name, None)
            if refresh is None:
                continue
            try:
                refresh()
            except Exception as exc:
                logger.warning("%s failed after keyword analysis", name, exc_info=True)
                self._log(f"Could not refresh view ({name}): {exc}", "WARNING")

    def _on_analysis_cancelled(self):
        self._log("Keyword analysis cancelled; no results were kept.", "WARNING")
        self.analysis_btn.setEnabled(True)
        self.search_progress.setVisible(False)

    def _on_analysis_error(self, err: str):
        self._log(f"ANALYSIS ERROR: {err}", "ERROR")
        self.analysis_btn.setEnabled(True)
        self.search_progress.setVisible(False)

    # ── Sysys search launcher and handlers ───────────────────────────

    def _on_search_question_changed(self, text: str) -> None:
        self._update_search_workflow_buttons()
        self._original_question_state.set_prospective(text)

    def _capture_original_question_for_first_search(
        self, query: str, *, save: bool = True
    ) -> str:
        original = self._original_question_state.capture_first_search(query)
        if save and self._project_path is not None:
            self._save_project()
        return original

    def _original_question_for_model(self) -> str:
        return original_question_for_model(self)

    def _edit_original_research_question(self, question: str) -> str:
        original = self._original_question_state.edit_captured(question)
        if self._project_path is not None:
            self._save_project()
        return original

    def _project_question_window_state(self) -> dict[str, object]:
        return {
            "search_query": self.sysys_nl_query.text(),
            **self._original_question_state.to_window_state(),
        }

    def _start_sysys_search(self):
        if not self._pipeline_stage_guard("Stage 1 (Search)"):
            return
        self._log("═" * 60)
        self._log("STAGE 1: LITERATURE SEARCH")
        self._log("═" * 60)

        # Snapshot: a project switch must not redirect a running search.
        cfg = copy.deepcopy(self.pipeline_config)
        nl_query = self.sysys_nl_query.text().strip()

        if not nl_query:
            self._log("No search query provided.", "WARNING")
            return
        # Capture the original before syncing (model-backed sync may consult
        # it), but defer the project save until after sync so the persisted
        # state reflects the just-computed hypothesis components rather than
        # the stale pre-sync list.
        self._capture_original_question_for_first_search(nl_query, save=False)
        self._sync_query_components_from_question(nl_query)
        if self._project_path is not None:
            self._save_project()

        # Gather which providers are selected (from checkboxes)
        selected_providers = []
        if self.src_pubmed.isChecked():
            selected_providers.append("PubMed")
        if self.src_s2.isChecked():
            selected_providers.append("Semantic Scholar")
        if self.src_openalex.isChecked():
            selected_providers.append("OpenAlex")
        if self.src_elicit.isChecked():
            selected_providers.append("Elicit")
        if self.src_scopus.isChecked():
            selected_providers.append("Scopus")
        if self.src_wos.isChecked():
            selected_providers.append("Web of Science")

        paper_limit = self.paper_limit.value()
        self._log(f"Query: {nl_query}")
        self._log(f"Providers: {', '.join(selected_providers) or 'NONE'}")
        self._log(f"Paper limit: {paper_limit} total across selected providers")
        self._log(f"Config project root: {cfg.project_root}")
        self._log(f"Config corpus dir: {cfg.corpus_dir}")
        self._log(f"LLM provider: {cfg.llm_provider} / "
                   f"{getattr(cfg, 'llm_model', 'N/A')}")
        self._log(f"API keys configured: "
                   f"PubMed={'yes' if cfg.pubmed_api_key else 'NO'}, "
                   f"PubMed email={'yes' if cfg.pubmed_email else 'NO'}, "
                   f"S2={'yes' if cfg.s2_api_key else 'NO'}, "
                   f"OpenAI={'yes' if cfg.openai_api_key else 'NO'}, "
                   f"Custom={'yes' if cfg.custom_api_key else 'NO'}, "
                   f"Anthropic={'yes' if cfg.anthropic_api_key else 'NO'}, "
                   f"Elicit={'yes' if cfg.elicit_api_key else 'NO'}, "
                   f"Scopus={'yes' if cfg.scopus_api_key else 'NO'}, "
                   f"WoS={'yes' if cfg.wos_api_key else 'NO'}")

        if not selected_providers:
            self._log("No search providers selected!", "WARNING")
            return

        # Determine provider enum(s) for search_runner
        from gui.search_runner import ProviderChoice

        provider_map = {
            "PubMed": ProviderChoice.PUBMED,
            "Semantic Scholar": ProviderChoice.S2,
            "OpenAlex": ProviderChoice.OPENALEX,
            "Elicit": ProviderChoice.ELICIT,
            "Scopus": ProviderChoice.SCOPUS,
            "Web of Science": ProviderChoice.WOS,
        }
        providers = [provider_map[name] for name in selected_providers
                     if name in provider_map]
        if not providers:
            providers = [ProviderChoice.ALL]

        self._log(f"Provider enums: {[p.value for p in providers]}")
        self._log("Creating PipelineSearchWorker...")
        from analysis.scoring import (
            build_weighted_boolean_query,
            build_weighted_component_query_terms,
        )

        weighted_terms = build_weighted_component_query_terms(self.hypothesis_components)
        boolean_query = build_weighted_boolean_query(self.hypothesis_components)
        semantic_terms = weighted_terms[:10] or [nl_query]
        boolean_terms = [boolean_query] if boolean_query else []
        if boolean_query and hasattr(self, "boolean_query_preview"):
            self.boolean_query_preview.setPlainText(boolean_query)
        if boolean_query:
            self._log(f"Weighted Boolean query: {boolean_query}")
        search_filters = self._search_filter_kwargs()
        cfg.citation_snowball_enabled = self.citation_snowball.isChecked()
        self._log(
            f"Search filters: years {search_filters['min_year']}-"
            f"{search_filters['max_year']}, full text only="
            f"{'yes' if search_filters['fulltext_only'] else 'no'}, citation snowballing="
            f"{'on' if cfg.citation_snowball_enabled else 'off'}"
        )
        self._search_worker = PipelineSearchWorker(
            natural_language_question=nl_query,
            semantic_terms=semantic_terms,
            boolean_terms=boolean_terms,
            provider=providers,
            config=cfg,
            max_results=paper_limit,
            **search_filters,
        )
        self._search_worker.progress.connect(self._on_sysys_search_progress)
        self._search_worker.finished_search.connect(self._on_sysys_search_done)
        self._search_worker.error.connect(self._on_sysys_search_error)
        self._search_worker.cancelled.connect(self._on_sysys_search_cancelled)

        self.sysys_search_btn.setEnabled(False)
        self.sysys_cancel_btn.setEnabled(True)
        self._active_stage_run_btn = self.sysys_search_btn
        self._active_stage_cancel_btn = self.sysys_cancel_btn
        self._active_stage_progress = self.search_progress
        self.pipeline_state = PipelineState.SEARCHING
        self.search_progress.setVisible(True)
        self.search_progress.setRange(0, 0)  # indeterminate

        self._log("Starting search worker thread...")
        self._set_pipeline_stage(1, "running")
        self._search_worker.start()

    def _cancel_sysys_search(self):
        self._log("Cancel requested for Sysys search/full-text work")
        if self._search_worker:
            self._search_worker.cancel()
            self._log("Sent cancel signal to search worker")
        if self._pipeline_worker:
            self._pipeline_worker.cancel()
            self._log("Sent cancel signal to pipeline worker")
        self.sysys_search_btn.setEnabled(True)
        self.sysys_cancel_btn.setEnabled(False)
        self.search_progress.setVisible(False)
        self.pipeline_state = PipelineState.IDLE
        self._abort_run_all("RUN ALL cancelled by user (Search tab).")

    def _on_sysys_search_cancelled(self):
        """Search worker stopped on cancel: free Stage 1 and disarm Run All."""
        self.sysys_search_btn.setEnabled(True)
        self.sysys_cancel_btn.setEnabled(False)
        self.search_progress.setVisible(False)
        self.pipeline_state = PipelineState.IDLE
        self._set_pipeline_stage(1, "pending", summary="Cancelled")
        self._abort_run_all("RUN ALL cancelled at stage 1 (search).")
        self._log("Search cancelled.", "WARNING")
        self._update_search_workflow_buttons()

    def _on_sysys_search_progress(self, msg: str):
        self.search_status.setText(msg)
        self._log(msg)

    def _on_sysys_search_counts(self, current: int, total: int):
        total = max(1, total)
        current = min(max(0, current), total)
        self.search_progress.setRange(0, total)
        self.search_progress.setValue(current)

    def _on_sysys_search_done(self, corpus: list):
        self._pipeline_corpus = corpus
        self._sync_analysis_articles_from_corpus(corpus)
        self._sysys_search_runs += 1
        self.sysys_search_btn.setEnabled(True)
        self.sysys_cancel_btn.setEnabled(False)
        self.search_progress.setVisible(False)
        self.pipeline_state = PipelineState.IDLE

        # Feature D: update At-a-Glance corpus count
        self._update_glance_corpus(len(corpus))
        self._update_glance_stage("Search done")

        # Feature F: tab badge
        self._set_tab_badge(0, "Search", len(corpus))

        # Feature E: inline results preview (first 10 articles)
        lines = [f"Search complete — {len(corpus)} unique papers\n"]
        for i, p in enumerate(corpus[:10], 1):
            title = (p.get("title") or "(no title)")[:80]
            year = p.get("year", "")
            source = p.get("source", "")
            lines.append(f"{i:2}. [{year}] {title}  ({source})")
        if len(corpus) > 10:
            lines.append(f"… and {len(corpus) - 10} more")
        self.search_results_display.setPlainText("\n".join(lines))

        n_papers = len(corpus)
        label = "paper" if n_papers == 1 else "papers"
        self._set_pipeline_stage(1, "done", summary=f"{n_papers} {label}")

        # Feature B: populate corpus browser
        self._populate_corpus_browser(corpus)

        self._log("═" * 60)
        self._log(f"SEARCH COMPLETE: {len(corpus)} unique papers")
        self._log("═" * 60)

        # Source breakdown
        sources: dict[str, int] = {}
        for p in corpus:
            src = p.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        for src, count in sorted(sources.items()):
            self._log(f"  {src}: {count} papers")

        # Year distribution
        years = [p.get("year", 0) for p in corpus if p.get("year")]
        if years:
            self._log(f"  Year range: {min(years)} - {max(years)}")

        # Papers with/without abstracts
        with_abstract = sum(1 for p in corpus if p.get("abstract"))
        self._log(f"  With abstracts: {with_abstract}/{len(corpus)}")

        # Save path
        save_path = self.pipeline_config.corpus_dir / "corpus.json"
        if save_path.exists():
            size_kb = save_path.stat().st_size / 1024
            self._log(f"  Saved to: {save_path} ({size_kb:.1f} KB)")
        else:
            self._log(f"  Expected save path: {save_path}")

        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 1 search complete")
        self._keyword_analysis_ready = bool(corpus or self.articles)
        self._update_search_workflow_buttons()
        if (
            self.fetch_fulltext.isChecked()
            and corpus
            and not self._run_all_active
        ):
            self._start_sysys_corpus_fulltext(auto_started=True)

        if self._run_all_active:
            self._run_all_stages_advance()

    def _on_sysys_search_error(self, err: str):
        self._log(f"SEARCH ERROR: {err}", "ERROR")
        self._on_error(err)
        self._set_pipeline_stage(1, "failed", summary=self._pipeline_error_summary(err))
        self.sysys_search_btn.setEnabled(True)
        self.sysys_cancel_btn.setEnabled(False)
        self.search_progress.setVisible(False)
        self.pipeline_state = PipelineState.IDLE
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 1 (search error).", "ERROR")
    def _on_stat_extract_full_llm_toggled(self, checked: bool):
        if checked:
            self.stat_extract_llm_context.setChecked(True)
        self.stat_extract_max_contexts.setEnabled(not checked)

    def _on_stat_extract_llm_context_toggled(self, checked: bool):
        if not checked and self.stat_extract_full_llm.isChecked():
            self.stat_extract_full_llm.setChecked(False)

    def _start_statistical_extract(self):
        if not self._pipeline_stage_guard("Stage 2c (Statistical Extract)"):
            return
        self._log("═" * 60)
        self._log("STAGE 2c: STATISTICAL EVIDENCE EXTRACTION")
        self._log("═" * 60)

        cfg = self._apply_config_overrides_for_stage(
            self.stat_extract_provider, self.stat_extract_model
        )

        custom_input = self.stat_extract_input.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            input_path = _Path(custom_input)
        else:
            input_path = cfg.claims_dir / "relevant.json"

        if not input_path.exists():
            self._log(
                f"No relevant.json found at {input_path}. "
                "Run Stage 2 (Score) first.", "ERROR")
            return

        import json as _json
        with open(input_path, encoding="utf-8") as f:
            papers = _json.load(f)

        use_llm = self.stat_extract_llm_context.isChecked()
        full_llm = self.stat_extract_full_llm.isChecked()
        max_contexts = 0 if full_llm else self.stat_extract_max_contexts.value()
        bundle_tokens = self.stat_extract_bundle_tokens.value()
        self._log(f"Input: {len(papers)} relevant papers from {input_path}")
        self._log(f"Output: {self.stat_extract_output.text().strip() or 'default'}")
        self._log(f"LLM context check: {use_llm}")
        if use_llm:
            limit_label = (
                "all deterministic candidates (no synthetic scope cap)"
                if max_contexts <= 0 else
                f"explicit user scope limit of {max_contexts:,} candidates for runtime/cost"
            )
            bundle_label = (
                f"explicit synthetic limit {bundle_tokens:,} tokens"
                if bundle_tokens else "automatic from loaded model context"
            )
            self._log(f"LLM provider: {cfg.llm_provider} / {cfg.llm_model}")
            self._log(f"LLM statistical extraction contexts: {limit_label}")
            self._log(f"LLM bundles (statistics per call by paper): {bundle_label}")

        custom_output = self.stat_extract_output.text().strip()
        extra_kw = {
            "use_llm_context_check": use_llm,
            "max_llm_contexts": max_contexts,
            "llm_full_statistical_extraction": full_llm,
            "llm_bundle_tokens": bundle_tokens,
        }
        if custom_input:
            extra_kw["input_path"] = custom_input
        if custom_output:
            extra_kw["output_path"] = custom_output

        self.stat_extract_btn.setEnabled(False)
        self.stat_extract_cancel_btn.setEnabled(True)
        self.stat_extract_progress.setVisible(True)
        self.stat_extract_progress.setRange(0, 0)
        self.stat_extract_status.setText("Running statistical extraction...")
        self._active_stage_run_btn = self.stat_extract_btn
        self._active_stage_cancel_btn = self.stat_extract_cancel_btn
        self._active_stage_progress = self.stat_extract_progress
        self._active_stage_status = self.stat_extract_status

        self._pipeline_worker = PipelineStageWorker(
            stage_name="statistical_extract",
            config=cfg,
            **extra_kw,
        )
        self._pipeline_worker.progress.connect(self._on_statistical_extract_progress)
        self._pipeline_worker.finished.connect(
            lambda name, result: self._on_statistical_extract_done(result)
        )
        self._pipeline_worker.error.connect(
            lambda name, err: self._on_statistical_extract_error(err)
        )
        self._pipeline_worker.cancelled.connect(
            lambda: self._on_statistical_extract_error("Statistical extraction cancelled")
        )
        self._pipeline_worker.start()

    def _on_statistical_extract_progress(self, msg: str):
        self.stat_extract_status.setText(msg)
        self._log(msg)

    def _on_statistical_extract_done(self, result):
        self.stat_extract_btn.setEnabled(True)
        self.stat_extract_cancel_btn.setEnabled(False)
        self.stat_extract_progress.setVisible(False)
        self.stat_extract_progress.setRange(0, 100)
        if isinstance(result, dict):
            summary = result.get("summary") or {}
            self.stat_extract_status.setText(
                "Statistical extraction complete: "
                f"{summary.get('results', 0)} results, "
                f"{summary.get('llm_reviewed', 0)} LLM-reviewed, "
                f"{summary.get('human_review_items', 0)} review items"
            )
        else:
            self._log(f"Statistical extraction result: {result}")
            self.stat_extract_status.setText("Statistical extraction complete")
        self._log("═" * 60)
        self._sync_all_stage_paths(self.pipeline_config)

    def _on_statistical_extract_error(self, err):
        self._log(f"STATISTICAL EXTRACTION ERROR: {err}", "ERROR")
        self.stat_extract_btn.setEnabled(True)
        self.stat_extract_cancel_btn.setEnabled(False)
        self.stat_extract_progress.setVisible(False)
        self.stat_extract_progress.setRange(0, 100)
        self.stat_extract_status.setText(str(err))


    def _on_screening_strategy_changed(self, *_):
        mode = self.score_screening_mode.currentData()
        cfg = self.pipeline_config
        cfg.relevance_screening_mode = mode
        self.score_simple_label.setVisible(mode == "off" and self.score_advanced_box.isHidden())
        self.score_threshold.setEnabled(mode == "off")
        self.score_batch.setValue(cfg.score_batch_size if mode == "off" else cfg.laya_batch_size)
        self.score_cascade_hint.setText(
            f"pw4 cutoffs: Laya bin ≥ {cfg.laya_cut_bin}, BRT-BERT P ≥ {cfg.laya_brt_threshold:g}. "
            "Cascade/baselines use these cutoffs; Model Score is 0/5 for reject/keep. "
            "Detailed probabilities are saved. Batch Size controls local inference batches."
        )

    def _start_sysys_score(self):
        if not self._pipeline_stage_guard("Stage 2 (Score)"):
            return
        self._log("═" * 60)
        self._log("STAGE 2: RELEVANCE SCORING")
        self._log("═" * 60)

        cfg = self._apply_config_overrides_for_stage(self.score_provider, self.score_model)

        # Check corpus exists
        corpus_path = cfg.corpus_dir / "corpus.json"
        custom_path = self.score_corpus_path.text().strip()
        if custom_path:
            from pathlib import Path as _Path
            corpus_path = _Path(custom_path)
        if not corpus_path.exists():
            self._log("No corpus found. Run Stage 1 (Search) first.", "ERROR")
            return

        import json as _json
        with open(corpus_path, encoding="utf-8") as f:
            corpus = _json.load(f)

        cfg.relevance_threshold = self.score_threshold.value()
        cfg.score_batch_size = self.score_batch.value()
        if hasattr(self, "score_screening_mode"):
            cfg.relevance_screening_mode = self.score_screening_mode.currentData()
            if cfg.relevance_screening_mode != "off":
                cfg.laya_batch_size = self.score_batch.value()
                self.pipeline_config.laya_batch_size = cfg.laya_batch_size

        # If the user hasn't opted into advanced LLM options, lock Stage 2 to
        # the bespoke BRT-BERT model and ignore the (hidden) provider combo.
        simple_mode = not (
            hasattr(self, "score_advanced_box") and not self.score_advanced_box.isHidden()
        )
        if simple_mode:
            selected_provider = "brt-bert"
            cfg.relevance_model = "brt-bert"
        else:
            selected_provider = self.score_provider.currentText().strip()
            if selected_provider in ("brt-bert", "housecatbert-scibert-releevance"):
                score_model = self.score_model.text().strip()
                cfg.relevance_model = (
                    score_model
                    if score_model.lower().startswith(("brt", "housecatbert"))
                    else selected_provider
                )
            else:
                cfg.relevance_model = ""

        self._log(f"Screening strategy: {getattr(cfg, 'relevance_screening_mode', 'off')}")
        self._log(f"Corpus: {len(corpus)} papers from {corpus_path}")
        if cfg.relevance_screening_mode == "off":
            self._log(f"Provider: {selected_provider}")
            self._log(f"Model: {cfg.relevance_model if cfg.relevance_model else cfg.llm_model}")
            self._log(f"Threshold: {self.score_threshold.value()}")
        else:
            self._log(
                f"Screening gates: Laya bin ≥ {cfg.laya_cut_bin}; "
                f"BRT-BERT P ≥ {cfg.laya_brt_threshold:g}"
            )
        self._log(f"Batch size: {self.score_batch.value()}")
        self._log(f"Workers: {cfg.concurrent_workers}")


        self.score_run_btn.setEnabled(False)
        self.score_cancel_btn.setEnabled(True)
        self._active_stage_run_btn = self.score_run_btn
        self._active_stage_cancel_btn = self.score_cancel_btn

        self._pipeline_worker = PipelineStageWorker(
            stage_name="score",
            config=cfg,
            corpus_path=corpus_path,
            threshold=cfg.relevance_threshold,
            workload=self.score_workload.currentText().lower(),
        )
        self._connect_stage_worker(self._on_score_done, self._on_score_error)
        self._set_pipeline_stage(2, "running")
        self._pipeline_worker.start()

    def _on_score_done(self, result):
        self.score_run_btn.setEnabled(True)
        self.score_cancel_btn.setEnabled(False)
        if isinstance(result, list):
            self._log(f"Scoring complete: {len(result)} relevant papers")
            if hasattr(self, "_refresh_human_relevance_model_scores"):
                self._refresh_human_relevance_model_scores()
            if hasattr(self, "_set_corpus_stage_relevance"):
                self._set_corpus_stage_relevance("stage2", result)
            # Feature D + F
            self._update_glance_relevant(len(result))
            self._update_glance_stage("Score done")
            self._set_tab_badge(1, "Score && Filter", len(result))
            n_rel = len(result)
            rel_label = "paper" if n_rel == 1 else "papers"
            self._set_pipeline_stage(2, "done", summary=f"{n_rel} relevant {rel_label}")
        else:
            self._log(f"Scoring result: {result}")
            self._set_pipeline_stage(2, "failed", summary="Unexpected score result")
            if self._run_all_active:
                self._run_all_active = False
                self._run_all_queue.clear()
                self._log("RUN ALL aborted at stage 2 (unexpected score result).", "ERROR")
        self._log("═" * 60)
        # Sync before advancing so Run All's next stage sees relevant.json.
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 2 score complete")
        if isinstance(result, list) and self._run_all_active:
            self._run_all_stages_advance()

    def _on_score_error(self, err):
        self._log(f"SCORING ERROR: {err}", "ERROR")
        self._set_pipeline_stage(2, "failed", summary=self._pipeline_error_summary(err))
        self.score_run_btn.setEnabled(True)
        self.score_cancel_btn.setEnabled(False)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 2 (score error).", "ERROR")

    def _start_sysys_corpus_fulltext(self, auto_started: bool = False,
                                       triggered_by_fetch: bool = False):
        """Enrich the Stage 1 corpus with full text (download step).

        Used both by the "Download full text" auto-path after a search and by
        the "Fetch Articles" button (``triggered_by_fetch=True``), which is a
        download-only action over the existing corpus.
        """
        if not self._pipeline_stage_guard("Corpus full-text enrichment"):
            return
        self._sync_fulltext_controls_to_config()
        self._log("═" * 60)
        self._log("CORPUS FULL-TEXT ENRICHMENT")
        self._log("═" * 60)
        if not self._pipeline_corpus:
            self._log("No corpus loaded. Run Stage 1 (Search) first.", "ERROR")
            self._update_search_workflow_buttons()
            return
        self._log(f"Corpus: {len(self._pipeline_corpus)} papers")
        if auto_started:
            self._log("Download full text is enabled; enriching Stage 1 corpus now.")

        cfg = copy.deepcopy(self.pipeline_config)
        corpus_path = cfg.corpus_dir / "corpus.json"
        output_path = cfg.corpus_dir / "corpus_fulltext.json"
        if not corpus_path.exists():
            self._log(f"No Stage 1 corpus found at {corpus_path}", "ERROR")
            self._update_search_workflow_buttons()
            return

        self._log(f"Input: {corpus_path}")
        self._log(f"Output: {output_path}")
        self._log(f"Archive: {cfg.data_dir / 'fulltext_corpus' / 'source_archive'}")
        if cfg.paperqa_index_enabled:
            self._log(
                f"PaperQA store: {cfg.data_dir / 'fulltext_corpus' / 'paperqa'} "
                f"(checkpoint batch {cfg.paperqa_index_batch_size})"
            )
        else:
            self._log("PaperQA indexing: disabled")

        from gui.workers import PipelineStageWorker

        # Disable both run buttons while downloading and enable the cancel
        # button matching whichever button started the download.
        self.sysys_search_btn.setEnabled(False)
        if hasattr(self, "fetch_btn"):
            self.fetch_btn.setEnabled(False)
        if triggered_by_fetch and hasattr(self, "cancel_fetch_btn"):
            self.cancel_fetch_btn.setEnabled(True)
            active_cancel_btn = self.cancel_fetch_btn
        else:
            self.sysys_cancel_btn.setEnabled(True)
            active_cancel_btn = self.sysys_cancel_btn
        self.search_progress.setVisible(True)
        self.search_progress.setRange(0, 0)
        self.search_status.setText("Enriching corpus with full text...")
        self.pipeline_state = PipelineState.SEARCHING
        self._active_stage_run_btn = self.sysys_search_btn
        self._active_stage_cancel_btn = active_cancel_btn
        self._active_stage_progress = self.search_progress

        self._pipeline_worker = PipelineStageWorker(
            stage_name="corpus_fulltext",
            config=cfg,
            corpus_path=corpus_path,
            output_path=output_path,
        )
        self._pipeline_worker.progress.connect(self._on_sysys_search_progress)
        self._pipeline_worker.progress_counts.connect(self._on_sysys_search_counts)
        self._pipeline_worker.finished.connect(
            lambda name, result: self._on_sysys_corpus_fulltext_done(result)
        )
        self._pipeline_worker.error.connect(
            lambda name, err: self._on_sysys_corpus_fulltext_error(err)
        )
        self._pipeline_worker.cancelled.connect(
            lambda: self._on_sysys_corpus_fulltext_error("Corpus full-text enrichment cancelled")
        )
        self._pipeline_worker.start()

    def _on_sysys_corpus_fulltext_done(self, result):
        self.sysys_search_btn.setEnabled(True)
        self.sysys_cancel_btn.setEnabled(False)
        if hasattr(self, "cancel_fetch_btn"):
            self.cancel_fetch_btn.setEnabled(False)
        self.search_progress.setVisible(False)
        self.pipeline_state = PipelineState.IDLE

        if not isinstance(result, dict):
            self._log(f"Corpus full-text result: {result}")
            self._update_search_workflow_buttons()
            return

        corpus_path = result.get("corpus_path", "")
        fulltext_count = int(result.get("fulltext_count", 0) or 0)
        paper_count = int(result.get("paper_count", 0) or 0)
        self._log(
            f"Corpus full-text enrichment complete: {fulltext_count}/{paper_count} papers"
        )
        if corpus_path:
            self._log(f"  Enriched corpus: {corpus_path}")
            try:
                import json as _json
                from pathlib import Path as _Path

                enriched_path = _Path(corpus_path)
                with enriched_path.open(encoding="utf-8") as f:
                    self._pipeline_corpus = _json.load(f)
                self._populate_corpus_browser(self._pipeline_corpus)
                # Auto-filled, so a later search (newer corpus.json) replaces it.
                self._set_auto_stage_path(self.score_corpus_path, str(enriched_path))
            except Exception as exc:
                self._log(f"Could not load enriched corpus preview: {exc}", "WARNING")
        archive_dir = result.get("archive_dir", "")
        if archive_dir:
            self._log(f"  Compressed source archive: {archive_dir}")
        paperqa = result.get("paperqa") or {}
        if paperqa.get("available"):
            self._log(
                "  PaperQA indexed "
                f"{paperqa.get('added', 0)} new texts "
                f"({paperqa.get('indexed_total', 0)} total): "
                f"{paperqa.get('docs_path', '')}"
            )
        else:
            reason = (
                paperqa.get("reason")
                or paperqa.get("error")
                or "paper-qa not installed"
            )
            self._log(
                f"  PaperQA indexing skipped: {reason}",
                "INFO" if reason == "disabled" else "WARNING",
            )
        self._keyword_analysis_ready = bool(self._pipeline_corpus or self.articles)
        self._update_search_workflow_buttons()
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("corpus fulltext complete")

    def _on_sysys_corpus_fulltext_error(self, err: str):
        self._log(f"CORPUS FULLTEXT ERROR: {err}", "ERROR")
        self.sysys_search_btn.setEnabled(True)
        self.sysys_cancel_btn.setEnabled(False)
        if hasattr(self, "cancel_fetch_btn"):
            self.cancel_fetch_btn.setEnabled(False)
        self.search_progress.setVisible(False)
        self.pipeline_state = PipelineState.IDLE
        self._update_search_workflow_buttons()

    def _start_extract(self):
        if not self._pipeline_stage_guard("Stage 3 (Extract)"):
            return
        self._log("═" * 60)
        self._log("STAGE 3: CLAIM EXTRACTION")
        self._log("═" * 60)

        cfg = self._apply_config_overrides_for_stage(
            self.extract_provider, self.extract_model)

        # Resolve input path
        custom_input = self.extract_corpus.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            input_path = _Path(custom_input)
        else:
            input_path = cfg.claims_dir / "relevant.json"

        if not input_path.exists():
            self._log(
                f"No relevant.json found at {input_path}. "
                "Run Stage 2 (Score) first.", "ERROR")
            return

        import json as _json
        with open(input_path, encoding="utf-8") as f:
            papers = _json.load(f)

        self._log(f"Input: {len(papers)} relevant papers from {input_path}")
        self._log(f"Provider: {cfg.llm_provider}")
        self._log(f"Model: {cfg.llm_model}")
        output_override = self.extract_max_tokens.value()
        self._log(
            "Claim output budget: automatic from model context and schema"
            if not output_override else
            f"Claim output budget: explicit synthetic override {output_override:,} tokens"
        )
        self._log(f"Thinking: {self.extract_thinking.isChecked()}")

        # Resolve output path
        custom_output = self.extract_output.text().strip()
        extra_kw = {"use_thinking": self.extract_thinking.isChecked()}
        if output_override:
            extra_kw["max_tokens"] = output_override
        if custom_input:
            extra_kw["input_path"] = custom_input
        if custom_output:
            extra_kw["output_path"] = custom_output

        self.extract_btn.setEnabled(False)
        self.extract_cancel_btn.setEnabled(True)
        self.extract_validate_progress.setVisible(True)
        self._active_stage_run_btn = self.extract_btn
        self._active_stage_cancel_btn = self.extract_cancel_btn
        self._active_stage_progress = self.extract_validate_progress

        self._pipeline_worker = PipelineStageWorker(
            stage_name="extract",
            config=cfg,
            **extra_kw,
        )
        self._connect_stage_worker(self._on_extract_done, self._on_extract_error)
        self._set_pipeline_stage(3, "running")
        self._pipeline_worker.start()

    def _on_extract_done(self, result):
        self.extract_btn.setEnabled(True)
        self.extract_cancel_btn.setEnabled(False)
        self.extract_validate_progress.setVisible(False)
        if isinstance(result, dict):
            n = result.get("extracted", 0)
            n_f = result.get("with_findings", 0)
            n_m = result.get("with_mechanisms", 0)
            self._log(
                f"Extraction complete: {n} papers — "
                f"{n_f} with findings, {n_m} with mechanisms")
            self._set_pipeline_stage(
                3, "done",
                summary=f"{n} papers extracted ({n_f} with findings)",
            )
        else:
            self._log(f"Extraction result: {result}")
            self._set_pipeline_stage(3, "done", summary="Extraction complete")
        self._log("═" * 60)
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 3 extract complete")
        if self._run_all_active:
            self._run_all_stages_advance()

    def _on_extract_error(self, err):
        self._log(f"EXTRACTION ERROR: {err}", "ERROR")
        self._set_pipeline_stage(3, "failed", summary=self._pipeline_error_summary(err))
        self.extract_btn.setEnabled(True)
        self.extract_cancel_btn.setEnabled(False)
        self.extract_validate_progress.setVisible(False)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 3 (extract error).", "ERROR")

    def _start_validate(self):
        if not self._pipeline_stage_guard("Stage 4 (Validate)"):
            return
        self._log("═" * 60)
        self._log("STAGE 4: METHOD REVIEW / VALIDATION")
        self._log("═" * 60)

        cfg = self._apply_config_overrides_for_stage(
            self.validate_provider, self.validate_model)

        # Resolve input path
        custom_input = self.validate_claims.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            input_path = _Path(custom_input)
        else:
            input_path = cfg.claims_dir / "relevant.json"

        if not input_path.exists():
            self._log(
                f"No input found at {input_path}. "
                "Run Stage 2 (Score) or Stage 3 (Extract) first.", "ERROR")
            return

        import json as _json
        with open(input_path, encoding="utf-8") as f:
            papers = _json.load(f)

        self._log(f"Input: {len(papers)} papers from {input_path}")
        self._log(f"Provider: {cfg.llm_provider}")
        self._log(f"Model: {cfg.llm_model}")

        # Build kwargs. Stage 4 always writes to data/method_review, so the
        # read-only Validated Output field is not passed through.
        extra_kw = {}
        if custom_input:
            extra_kw["input_path"] = custom_input

        self.validate_btn.setEnabled(False)
        self.validate_cancel_btn.setEnabled(True)
        self.extract_validate_progress.setVisible(True)
        self._active_stage_run_btn = self.validate_btn
        self._active_stage_cancel_btn = self.validate_cancel_btn
        self._active_stage_progress = self.extract_validate_progress

        self._pipeline_worker = PipelineStageWorker(
            stage_name="validate",
            config=cfg,
            **extra_kw,
        )
        self._connect_stage_worker(self._on_validate_done, self._on_validate_error)
        self._set_pipeline_stage(4, "running")
        self._pipeline_worker.start()

    def _on_validate_done(self, result):
        self.validate_btn.setEnabled(True)
        self.validate_cancel_btn.setEnabled(False)
        self.extract_validate_progress.setVisible(False)
        if isinstance(result, dict):
            n = result.get("validated", 0)
            flagged = result.get("flagged", 0)
            self._log(
                f"Method review complete: {n} papers validated, "
                f"{flagged} flagged for manual review")
            summary = f"{n} profiles"
            if flagged:
                summary = f"{summary}, {flagged} flagged"
            self._set_pipeline_stage(4, "done", summary=summary)
        else:
            self._log(f"Validation result: {result}")
            self._set_pipeline_stage(4, "done", summary="Validation complete")
        self._log("═" * 60)
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 4 validate complete")
        if self._run_all_active:
            self._run_all_stages_advance()

    def _on_validate_error(self, err):
        self._log(f"VALIDATION ERROR: {err}", "ERROR")
        self._set_pipeline_stage(4, "failed", summary=self._pipeline_error_summary(err))
        self.validate_btn.setEnabled(True)
        self.validate_cancel_btn.setEnabled(False)
        self.extract_validate_progress.setVisible(False)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 4 (validate error).", "ERROR")

    def _start_audit(self):
        if not self._pipeline_stage_guard("Stage 5 (Audit)"):
            return
        self._log("═" * 60)
        self._log("STAGE 5: SPAN AUDIT & QUALITY FILTERING")
        self._log("═" * 60)

        import copy as _copy
        cfg = _copy.deepcopy(self.pipeline_config)

        # Resolve input — claims.json from Stage 3
        custom_input = self.audit_input.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            input_path = _Path(custom_input)
        else:
            input_path = cfg.claims_dir / "claims.json"

        if not input_path.exists():
            self._log(
                f"No claims found at {input_path}. "
                "Run Stage 3 (Extract) first.", "ERROR")
            return

        import json as _json
        with open(input_path, encoding="utf-8") as f:
            papers = _json.load(f)
        self._log(f"Input: {len(papers)} papers from {input_path}")

        # Stage 5 reads the slider value from the stage config.
        threshold = self.audit_threshold.value() / 100.0
        cfg.audit_similarity_threshold = threshold
        self._log(f"Similarity threshold: {threshold:.2f}")
        cfg.audit_exclude_ungrounded = self.audit_exclude_ungrounded.isChecked()
        self._log("Ungrounded papers: " + (
            "removed" if cfg.audit_exclude_ungrounded else "flagged only (kept)"))

        # Build kwargs. Stage 5 always writes claims_filtered.json in the claims
        # folder, so the read-only Audited Output field is not passed through.
        extra_kw = {}
        if custom_input:
            extra_kw["claims_path"] = custom_input

        self.audit_btn.setEnabled(False)
        self._active_stage_run_btn = self.audit_btn
        self._active_stage_cancel_btn = None

        self._pipeline_worker = PipelineStageWorker(
            stage_name="audit",
            config=cfg,
            **extra_kw,
        )
        self._connect_stage_worker(self._on_audit_done, self._on_audit_error)
        self._set_pipeline_stage(5, "running")
        self._pipeline_worker.start()

    def _on_audit_done(self, result):
        self.audit_btn.setEnabled(True)
        if isinstance(result, dict) and result.get("cancelled"):
            # Stage 5 removed its previous outputs; nothing current exists to show.
            self._log("Span audit cancelled; previous Stage 5 outputs were cleared.", "WARNING")
            self._set_pipeline_stage(5, "pending", summary="Cancelled")
            self._abort_run_all("RUN ALL cancelled at stage 5 (audit).")
            return
        if isinstance(result, dict):
            total = result.get("total", 0)
            kept = result.get("filtered", 0)
            removed = result.get("removed", 0)
            rescued = result.get("reclassified", 0)
            flagged = result.get("flagged", 0)
            self._log(
                f"Span audit complete: {total} papers in, "
                f"{kept} kept, {removed} removed, {rescued} rescued, {flagged} flagged")
            out = result.get("output_path", "")
            if out:
                self._log(f"  Output: {out}")
            self._set_pipeline_stage(5, "done", summary=f"{kept} kept, {removed} removed")
        else:
            self._log(f"Audit result: {result}")
            self._set_pipeline_stage(5, "done", summary="Audit complete")
        self._log("═" * 60)
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 5 audit complete")
        if self._run_all_active:
            self._run_all_stages_advance()

    def _on_audit_error(self, err):
        self._log(f"AUDIT ERROR: {err}", "ERROR")
        self._set_pipeline_stage(5, "failed", summary=self._pipeline_error_summary(err))
        self.audit_btn.setEnabled(True)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 5 (audit error).", "ERROR")

    def _start_synthesize(self, review_only=False):
        if not self._pipeline_stage_guard("Stage 6 (Synthesize)"):
            return
        self._log("═" * 60)
        self._log("STAGE 6: SYNTHESIS (CLUSTER · NARRATE · ABSTRACT)")
        self._log("═" * 60)

        cfg = self._apply_config_overrides_for_stage(
            self.synth_provider, self.synth_model)

        # Resolve input — prefer claims_filtered.json, then claims.json
        custom_input = self.synth_input.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            input_path = _Path(custom_input)
        else:
            # Prefer filtered output from Stage 5, fall back to Stage 3
            # (resolve_claims_path also falls back when the filtered file
            # exists but is empty)
            from process.paths import resolve_claims_path
            input_path = resolve_claims_path(cfg)

        if not input_path.exists():
            self._log(
                f"No claims found at {input_path}. "
                "Run Stage 3 (Extract) or Stage 5 (Audit) first.", "ERROR")
            return

        import json as _json
        with open(input_path, encoding="utf-8") as f:
            papers = _json.load(f)
        self._log(f"Input: {len(papers)} papers from {input_path}")

        auto_clusters = self.synth_auto_clusters.isChecked()
        n_clusters = 0 if (auto_clusters or review_only) else self.synth_min_cluster.value()
        max_tokens = self.synth_max_tokens.value()
        if auto_clusters:
            self._log(
                "Clusters: auto (silhouette sweep), output budget: "
                + (f"explicit synthetic override {max_tokens:,} tokens"
                   if max_tokens else "automatic from model context and cluster size")
            )
        else:
            self._log(
                f"Clusters: up to {n_clusters}, output budget: "
                + (f"explicit synthetic override {max_tokens:,} tokens"
                   if max_tokens else "automatic from model context and cluster size")
            )
        self._log(f"Provider: {cfg.llm_provider}, Model: {cfg.llm_model}")

        # Build kwargs — pass the resolved input so the stage uses exactly
        # the file validated above
        extra_kw: dict = {
            "n_clusters": n_clusters,
            "k_max": self.synth_sweep_max.value(),
            "review_only": review_only,
            "claims_path": str(input_path),
        }
        if max_tokens:
            extra_kw["max_tokens"] = max_tokens
        custom_output = self.synth_output.text().strip()
        if custom_output:
            extra_kw["output_path"] = custom_output

        self.synth_btn.setEnabled(False)
        self.synth_status.setText("Synthesis running…")
        self._active_stage_run_btn = self.synth_btn
        self._active_stage_cancel_btn = None

        self._pipeline_worker = PipelineStageWorker(
            stage_name="synthesize",
            config=cfg,
            **extra_kw,
        )
        self._connect_stage_worker(self._on_synthesize_done, self._on_synthesize_error)
        self._synth_review_running = review_only
        if not review_only:
            self._set_pipeline_stage(6, "running")
        self._pipeline_worker.start()

    def _show_synthesis_warnings(self, result: dict) -> None:
        """Stage 6 warnings (empty LLM replies, weak cluster structure) in log and status."""
        warnings = [w for w in result.get("warnings") or [] if w]
        for warning in warnings:
            self._log(f"  Synthesis warning: {warning}", "WARNING")
        if warnings and result.get("status") != "partial":
            self.synth_status.setText(
                f"Synthesis complete with {len(warnings)} warning(s); see the log.")
        for unresolved in result.get("unresolved_model_steps") or []:
            self._log(
                "  Synthesis model action: " + str(unresolved.get("message") or unresolved),
                "WARNING",
            )

    def _on_synthesize_done(self, result):
        self.synth_btn.setEnabled(True)
        if isinstance(result, dict) and result.get("review_only"):
            self._synth_review_running = False
            self.synth_status.setText("Cluster scores ready. Review the curves and choose K for synthesis.")
            self._review_silhouette_scores(path=result["k_selection_path"])
            return
        self.synth_status.setText("Synthesis complete.")
        # Sync first: it refreshes the board from disk, which would otherwise
        # overwrite the state set below (see _on_robustness_done).
        self._sync_all_stage_paths(self.pipeline_config)
        board_state = "done"
        if isinstance(result, dict):
            summary = summarize_synthesis(result)
            board_state = summary.board_state
            self._show_synthesis_warnings(result)
            backend = result.get("embedding_backend")
            if backend:
                self._log(f"  Embedding backend: {backend}")
            total = result.get("total_papers", 0)
            n_clust = result.get("n_clusters", 0)
            n_narr = result.get("narratives_count", 0)
            abs_len = result.get("abstract_length", 0)
            result_label = "partial" if result.get("status") == "partial" else "complete"
            self._log(
                f"Synthesis {result_label}: {total} papers, {n_clust} clusters, "
                f"{n_narr} narratives, abstract {abs_len} chars")
            k_sel = result.get("k_selection") or {}
            if k_sel.get("mode") == "auto":
                sweep = k_sel.get("silhouette_scores", [])
                if sweep:
                    self._log("  Silhouette sweep: " + ", ".join(
                        f"k={s['k']}:{s['score']:.3f}" for s in sweep))
                self._log(
                    f"  Auto-selected k={k_sel.get('chosen_k')} by silhouette score")
            abstract_path = result.get("abstract_path", "")
            if abstract_path:
                self._log(f"  Abstract: {abstract_path}")
                try:
                    with open(abstract_path, encoding="utf-8") as f:
                        abstract_text = f.read()
                    self.synth_narrative.setPlainText(abstract_text)
                except Exception as exc:
                    logger.warning("Could not load abstract for display: %s", exc)
                # Auto-open the abstract in a pop-out window
                self._open_abstract_window(Path(abstract_path))
            clusters_path = result.get("clusters_path", "")
            if clusters_path:
                try:
                    import json as _cj
                    with open(clusters_path, encoding="utf-8") as f:
                        clusters_data = _cj.load(f)
                    clusters_list = clusters_data if isinstance(clusters_data, list) else \
                        clusters_data.get("clusters", [])
                    self.cluster_table.setRowCount(len(clusters_list))
                    for row, cl in enumerate(clusters_list):
                        label = cl.get("label") or cl.get("name") or f"Cluster {row + 1}"
                        n_papers = cl.get("paper_count", 0) or len(cl.get("papers", []))
                        terms = ", ".join(
                            (cl.get("mechanisms", []) + cl.get("conditions", []))[:5]
                        )
                        self.cluster_table.setItem(row, 0, QTableWidgetItem(str(label)))
                        self.cluster_table.setItem(row, 1, QTableWidgetItem(str(n_papers)))
                        self.cluster_table.setItem(row, 2, QTableWidgetItem(terms))
                    self.cluster_table.resizeColumnsToContents()
                except Exception as exc:
                    logger.warning("Could not populate cluster table: %s", exc)
            if clusters_path:
                self._log(f"  Clusters: {clusters_path}")
            narratives_path = result.get("narratives_path", "")
            if narratives_path:
                self._log(f"  Narratives: {narratives_path}")
            explanation_md = result.get("hypothesis_explanation_md_path", "")
            if explanation_md:
                self._log(f"  Why This Hypothesis: {explanation_md}")
                self._display_hypothesis_explanation(Path(explanation_md))
            explanation_json = result.get("hypothesis_explanation_path", "")
            if explanation_json:
                self._log(f"  Explanation JSON: {explanation_json}")
            evidence_network = result.get("evidence_network_path", "")
            if evidence_network:
                self._log(f"  Evidence Network: {evidence_network}")
                self._load_evidence_network_path(Path(evidence_network))
            sources_md = result.get("synthesis_sources_md_path", "")
            if sources_md:
                self._log(f"  Source Citations: {sources_md}")
                self._display_synthesis_sources(Path(sources_md))
            sources_bib = result.get("synthesis_sources_bib_path", "")
            if sources_bib:
                self._log(f"  Source BibTeX: {sources_bib}")
            sources_json = result.get("synthesis_sources_path", "")
            if sources_json:
                self._log(f"  Source Map JSON: {sources_json}")
            for level, msg in summary.lines:
                self._log(msg, level)
            if board_state != "done":
                self.synth_status.setText(summary.status)
            self._set_pipeline_stage(6, board_state, summary=summary.board_summary)
            # Land on the Abstract sub-tab so the result is front and center
            self._show_synth_output_subtab(0)
        else:
            self._log(f"Synthesis result: {result}")
            self._set_pipeline_stage(6, "done", summary="Synthesis complete")
        self._log("═" * 60)
        self._schedule_auto_backup("stage 6 synthesize complete")
        if board_state != "done":
            self._abort_run_all("RUN ALL aborted at stage 6 (no abstract was produced).")
        elif self._run_all_active:
            self._run_all_stages_advance()

    def _on_synthesize_error(self, err):
        self._log(f"SYNTHESIS ERROR: {err}", "ERROR")
        self.synth_status.setText(f"Synthesis stopped: {err}")
        if not getattr(self, "_synth_review_running", False):
            self._set_pipeline_stage(6, "failed", summary=self._pipeline_error_summary(err))
        self._synth_review_running = False
        self.synth_btn.setEnabled(True)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 6 (synthesize error).", "ERROR")

    def _display_hypothesis_explanation(self, path: Path) -> bool:
        """Load hypothesis_explanation.md into the synthesis detail panel."""
        try:
            text = Path(path).read_text(encoding="utf-8")
            if hasattr(self.hypothesis_explanation, "setMarkdown"):
                self.hypothesis_explanation.setMarkdown(text)
            else:
                self.hypothesis_explanation.setPlainText(text)
            self._log(f"Hypothesis explanation loaded: {path}")
            return True
        except Exception as exc:
            self._log(f"Could not load hypothesis explanation: {exc}", "ERROR")
            return False

    def _show_synth_output_subtab(self, index: int) -> None:
        """Bring the requested Synthesis output sub-tab to the front."""
        tabs = getattr(self, "synth_output_tabs", None)
        if tabs is not None and 0 <= index < tabs.count():
            tabs.setCurrentIndex(index)

    def _load_hypothesis_explanation(self):
        """Load the latest deterministic hypothesis explanation Markdown."""
        default_path = self.pipeline_config.clusters_dir / "hypothesis_explanation.md"
        path = default_path
        if not path.exists():
            chosen, _ = QFileDialog.getOpenFileName(
                self,
                "Load Hypothesis Explanation",
                str(self.pipeline_config.clusters_dir),
                "Markdown files (*.md);;All files (*)",
            )
            if not chosen:
                return
            path = Path(chosen)
        self._display_hypothesis_explanation(path)
        self._show_synth_output_subtab(1)
        self._open_text_result_window("Why This Hypothesis", path)

    def _display_synthesis_sources(self, path: Path) -> bool:
        """Load synthesis_sources.md into the dedicated Source Citations view.

        Uses ``synth_sources_view`` so the abstract in ``synth_narrative``
        is never overwritten.
        """
        view = getattr(self, "synth_sources_view", None) or self.synth_narrative
        try:
            text = Path(path).read_text(encoding="utf-8")
            if hasattr(view, "setMarkdown"):
                view.setMarkdown(text)
            else:
                view.setPlainText(text)
            self._log(f"Synthesis source citations loaded: {path}")
            return True
        except Exception as exc:
            self._log(f"Could not load synthesis source citations: {exc}", "ERROR")
            return False

    def _load_synthesis_sources(self):
        """Load the latest synthesis source citation Markdown."""
        default_path = self.pipeline_config.clusters_dir / "synthesis_sources.md"
        path = default_path
        if not path.exists():
            chosen, _ = QFileDialog.getOpenFileName(
                self,
                "Load Synthesis Source Citations",
                str(self.pipeline_config.clusters_dir),
                "Markdown files (*.md);;All files (*)",
            )
            if not chosen:
                return
            path = Path(chosen)
        self._display_synthesis_sources(path)
        self._show_synth_output_subtab(2)
        self._open_text_result_window("Source Citations", path)

    def _load_evidence_network_path(self, path: Path) -> bool:
        """Load generalized evidence_network.json into the network widgets."""
        loaded = False
        try:
            import json as _json
            data = _json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            self._log(f"Could not load evidence network: {exc}", "ERROR")
            return False
        if hasattr(self, "neuro_network_widget"):
            self.neuro_network_widget.set_network_data(data)
            loaded = True
        if hasattr(self, "synth_network_widget"):
            self.synth_network_widget.set_network_data(data)
            loaded = True
        if loaded:
            if hasattr(self, "neuro_export_graph_btn"):
                self.neuro_export_graph_btn.setEnabled(True)
            self._log(f"Evidence network loaded: {path}")
        return loaded

    def _load_evidence_network(self):
        """Load the latest generalized evidence network JSON."""
        default_path = self.pipeline_config.clusters_dir / "evidence_network.json"
        path = default_path
        if not path.exists():
            chosen, _ = QFileDialog.getOpenFileName(
                self,
                "Load Evidence Network",
                str(self.pipeline_config.clusters_dir),
                "JSON files (*.json);;All files (*)",
            )
            if not chosen:
                return
            path = Path(chosen)
        self._load_evidence_network_path(path)
        self._show_synth_output_subtab(3)
        self._open_evidence_network_window(path)

    # ── Pop-out result viewer windows ──────────────────────────────────

    def _open_text_result_window(
        self,
        title: str,
        path: Path,
        render_markdown: bool = True,
    ) -> object | None:
        """Open a TextResultWindow showing the contents of *path*."""
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception as exc:
            self._log(f"Could not open result window for {path}: {exc}", "ERROR")
            return None
        from gui.result_viewer import TextResultWindow

        win = TextResultWindow(title, source_path=path, parent=self)
        win.set_text(text, render_markdown=render_markdown)
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        win.show()
        win.raise_()
        win.activateWindow()
        # Keep a reference so the window isn't garbage-collected
        if not hasattr(self, "_result_windows"):
            self._result_windows = []
        self._result_windows.append(win)
        win.destroyed.connect(lambda _w=None, win=win: self._result_windows.remove(win)
                              if win in self._result_windows else None)
        return win

    def _open_abstract_window(self, abstract_path: Path) -> None:
        """Open a TextResultWindow showing the synthesis abstract."""
        self._open_text_result_window(
            "Synthesis Abstract", abstract_path, render_markdown=True
        )

    def _open_evidence_network_window(self, path: Path) -> None:
        """Open a NetworkResultWindow showing the evidence network graph + JSON."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            self._log(f"Could not open evidence network window: {exc}", "ERROR")
            return
        from gui.result_viewer import NetworkResultWindow

        win = NetworkResultWindow(
            "Evidence Network", source_path=path, parent=self
        )
        win.set_network_data(data)
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        win.show()
        win.raise_()
        win.activateWindow()
        if not hasattr(self, "_result_windows"):
            self._result_windows = []
        self._result_windows.append(win)
        win.destroyed.connect(lambda _w=None, win=win: self._result_windows.remove(win)
                              if win in self._result_windows else None)

    def _start_sysys_fulltext(self):
        if not self._pipeline_stage_guard("Stage 7 (Fulltext)"):
            return
        self._sync_fulltext_controls_to_config()
        self._log("═" * 60)
        self._log("STAGE 7: FULL-TEXT RETRIEVAL & METRIC EXTRACTION")
        self._log("═" * 60)

        cfg = copy.deepcopy(self.pipeline_config)

        # Resolve input corpus path
        custom_input = self.sysys_ft_corpus.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            input_path = _Path(custom_input)
        else:
            # Default: claims from Stage 3/5 (resolve_claims_path falls
            # back to claims.json when the filtered file is empty)
            from process.paths import resolve_claims_path
            input_path = resolve_claims_path(cfg)

        if not input_path.exists():
            self._log(
                f"No claims found at {input_path}. "
                "Run Stage 3 (Extract) or Stage 5 (Audit) first.", "ERROR")
            return

        import json as _json
        with open(input_path, encoding="utf-8") as f:
            papers = _json.load(f)
        self._log(f"Input: {len(papers)} papers from {input_path}")

        workers = self.sysys_ft_workers.value()
        email = (getattr(cfg, "pubmed_email", "") or "").strip()
        self._log(f"Workers: {workers}, Email: {email or 'NOT SET'}")
        if not email:
            self._log(
                "No contact email set (Settings: PubMed email). Unpaywall and NCBI "
                "ask for one; set it so requests are not throttled or misattributed.",
                "WARNING",
            )
        if cfg.fulltext_streaming:
            self._log("Streaming: enabled (resumable JSONL checkpoints)")
        else:
            self._log(
                f"Streaming: disabled (buffered batch {cfg.fulltext_download_batch_size})"
            )

        extra_kw: dict = {
            "workers": workers,
            "email": email,
            "resume": self.sysys_ft_resume.isChecked(),
            # Pass the validated input so the stage uses the same file
            "claims_path": str(input_path),
        }

        self.sysys_ft_btn.setEnabled(False)
        self.sysys_ft_cancel_btn.setEnabled(True)
        self._active_stage_run_btn = self.sysys_ft_btn
        self._active_stage_cancel_btn = self.sysys_ft_cancel_btn
        if self._project_path is not None:
            self._save_project()

        from gui.workers import PipelineStageWorker
        self._pipeline_worker = PipelineStageWorker(
            stage_name="fulltext",
            config=cfg,
            **extra_kw,
        )
        self._connect_stage_worker(
            self._on_fulltext_done, self._on_fulltext_error,
            on_cancelled=self._on_fulltext_cancelled)
        self._set_pipeline_stage(7, "running")
        self._pipeline_worker.start()

    def _on_fulltext_cancelled(self):
        self.sysys_ft_btn.setEnabled(True)
        self.sysys_ft_cancel_btn.setEnabled(False)
        self._run_all_active = False
        self._run_all_queue.clear()
        self._set_pipeline_stage(7, "pending", summary="Cancelled; checkpoints preserved")
        self._log("Fulltext cancelled; completed paper checkpoints preserved.")

    def _on_fulltext_done(self, result):
        self.sysys_ft_btn.setEnabled(True)
        self.sysys_ft_cancel_btn.setEnabled(False)
        if isinstance(result, dict):
            targets = result.get("targets", 0)
            oa = result.get("oa_papers", 0)
            downloaded = result.get("downloaded", 0)
            metrics = result.get("metrics_extracted", 0)
            self._log(
                f"Fulltext complete: {targets} targets identified, "
                f"{oa} open access, {downloaded} downloaded, "
                f"{metrics} with metrics extracted")
            resumed = int(result.get("resumed", 0) or 0)
            remaining = int(result.get("remaining", 0) or 0)
            if resumed:
                self._log(f"  Resumed {resumed} completed papers from JSONL")
            if remaining:
                self._log(
                    f"  {remaining} papers remain eligible for retry on the next run",
                    "WARNING",
                )
            out = result.get("output_path", "")
            if out:
                self._log(f"  Output: {out}")
            self._set_pipeline_stage(
                7, "done",
                summary=f"{downloaded} downloaded, {metrics} with metrics",
            )
        else:
            self._log(f"Fulltext result: {result}")
            self._set_pipeline_stage(7, "done", summary="Fulltext complete")
        self._log("═" * 60)
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 7 fulltext complete")
        if self._run_all_active:
            self._run_all_stages_advance()

    def _on_fulltext_error(self, err):
        self._log(f"FULLTEXT ERROR: {err}", "ERROR")
        self._set_pipeline_stage(7, "failed", summary=self._pipeline_error_summary(err))
        self.sysys_ft_btn.setEnabled(True)
        self.sysys_ft_cancel_btn.setEnabled(False)
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL aborted at stage 7 (fulltext error).", "ERROR")

    def _start_robustness(self):
        if not self._pipeline_stage_guard("Stage 8 (Robustness)"):
            return
        self._log("═" * 60)
        self._log("STAGE 8: ROBUSTNESS CHECKS")
        self._log("═" * 60)
        self._log(NO_LLM_LOG_LINE)

        cfg = copy.deepcopy(self.pipeline_config)

        # Resolve corpus path — prefer custom path, then standard locations
        custom_input = self.robust_input.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            corpus_path = _Path(custom_input)
        else:
            filtered = cfg.claims_dir / "claims_filtered.json"
            base = cfg.claims_dir / "claims.json"
            relevant = cfg.claims_dir / "relevant.json"
            if filtered.exists():
                corpus_path = filtered
            elif base.exists():
                corpus_path = base
            elif relevant.exists():
                corpus_path = relevant
            else:
                corpus_path = None

        if corpus_path is None or not corpus_path.exists():
            self._log(
                "No scored corpus found. Run Stage 2 (Score) or "
                "Stage 3 (Extract) first.", "ERROR")
            return

        import json as _json
        with open(corpus_path, encoding="utf-8") as f:
            papers = _json.load(f)
        if isinstance(papers, list):
            self._log(f"Input: {len(papers)} papers from {corpus_path}")
        else:
            self._log(f"Input: {corpus_path}")

        extra_kw: dict = {}
        if custom_input:
            extra_kw["claims_path"] = custom_input
        output_path = self.robust_output.text().strip()
        if output_path:
            extra_kw["output_path"] = output_path

        self.robust_btn.setEnabled(False)
        self.robust_cancel_btn.setEnabled(True)
        self._active_stage_run_btn = self.robust_btn
        self._active_stage_cancel_btn = self.robust_cancel_btn
        self._active_stage_progress = self.robust_verify_progress

        self._pipeline_worker = PipelineStageWorker(
            stage_name="robustness",
            config=cfg,
            **extra_kw,
        )
        self._connect_stage_worker(
            self._on_robustness_done, self._on_robustness_error,
            lambda: self._on_robustness_done({"cancelled": True}))
        self._set_pipeline_stage(8, "running")
        self._begin_robust_verify_run("Stage 8 running (local statistics, no LLM)…")
        self._pipeline_worker.start()

    def _on_robustness_done(self, result):
        self.robust_btn.setEnabled(True)
        self.robust_cancel_btn.setEnabled(False)
        summary = summarize_robustness(result)
        # Sync first: it refreshes the board from disk, which would otherwise
        # overwrite this run's status (e.g. "failed") with stale disk state.
        self._sync_all_stage_paths(self.pipeline_config)
        self._show_robust_verify_summary(8, summary)
        if summary.board_state == "done" and isinstance(result, dict):
            self._auto_load_robustness_network(result)
        self._log("═" * 60)
        if isinstance(result, dict) and result.get("cancelled"):
            self._abort_run_all("RUN ALL cancelled at stage 8 (robustness).")
            return
        if summary.board_state != "done":
            self._abort_run_all("RUN ALL aborted at stage 8 (robustness failed).")
            return
        self._schedule_auto_backup("stage 8 robustness complete")
        if self._run_all_active:
            self._run_all_stages_advance()

    def _auto_load_robustness_network(self, result: dict) -> None:
        """Load Stage 8's condition network into the Neuroimaging tab."""
        net_path = result.get("results_path", "")
        if not net_path:
            return
        try:
            from process.graph_analysis import network_artifact_path
            net_json = network_artifact_path(Path(net_path).parent)
            if not net_json.exists():
                return
            with open(net_json, encoding="utf-8") as f:
                net_data = json.load(f)
            # Convert Stage 8 edge list to widget schema
            raw_edges = net_data.get("edges", [])
            cond_weights: dict[str, int] = {}
            reg_weights: dict[str, int] = {}
            reg_types: dict[str, str] = {}
            for e in raw_edges:
                src = e.get("source", "")
                tgt = e.get("target", "")
                w = e.get("weight", 1)
                if src:
                    cond_weights[src] = cond_weights.get(src, 0) + w
                if tgt:
                    reg_weights[tgt] = reg_weights.get(tgt, 0) + w
                    if tgt not in reg_types:
                        reg_types[tgt] = e.get("target_type", "entity")
            widget_data = {
                "condition_nodes": [
                    {"name": n, "paper_count": c}
                    for n, c in cond_weights.items()
                ],
                "region_nodes": [
                    {"name": n, "type": reg_types.get(n, "entity"),
                     "paper_count": c}
                    for n, c in reg_weights.items()
                ],
                "condition_region_edges": [
                    {"condition": e["source"], "region": e["target"],
                     "weight": e["weight"]}
                    for e in raw_edges
                    if e.get("source") and e.get("target")
                ],
            }
            self.neuro_network_widget.set_network_data(widget_data)
            self.neuro_export_graph_btn.setEnabled(True)
            self._log(
                f"Network graph auto-loaded: "
                f"{net_data.get('n_edges', 0)} edges, "
                f"{net_data.get('n_conditions', 0)} conditions")
        except Exception as exc:
            logger.warning("Auto-load network graph failed: %s", exc, exc_info=True)
            self._log(f"Could not load the Stage 8 network graph: {exc}", "WARNING")

    def _on_robustness_error(self, err):
        self._log(f"ROBUSTNESS ERROR: {err}", "ERROR")
        self._set_pipeline_stage(8, "failed", summary=self._pipeline_error_summary(err))
        self.robust_btn.setEnabled(True)
        self.robust_cancel_btn.setEnabled(False)
        self._end_robust_verify_run(f"Stage 8 FAILED: {err}")
        self._abort_run_all("RUN ALL aborted at stage 8 (robustness error).")

    def _start_verify(self):
        if not self._pipeline_stage_guard("Stage 9 (Verify)"):
            return
        self._log("═" * 60)
        self._log("STAGE 9: FINAL REVIEW")
        self._log("═" * 60)
        self._log(NO_LLM_LOG_LINE)

        cfg = copy.deepcopy(self.pipeline_config)

        # Resolve corpus path — prefer custom path, then standard locations
        custom_input = self.verify_claims.text().strip()
        if custom_input:
            from pathlib import Path as _Path
            corpus_path = _Path(custom_input)
        else:
            filtered = cfg.claims_dir / "claims_filtered.json"
            base = cfg.claims_dir / "claims.json"
            relevant = cfg.claims_dir / "relevant.json"
            if filtered.exists():
                corpus_path = filtered
            elif base.exists():
                corpus_path = base
            elif relevant.exists():
                corpus_path = relevant
            else:
                corpus_path = None

        if corpus_path is None or not corpus_path.exists():
            self._log(
                "No claims corpus found. Run Stage 3 (Extract) or "
                "Stage 5 (Audit) first.", "ERROR")
            return

        import json as _json
        with open(corpus_path, encoding="utf-8") as f:
            papers = _json.load(f)
        if isinstance(papers, list):
            self._log(f"Input: {len(papers)} papers from {corpus_path}")
        else:
            self._log(f"Input: {corpus_path}")

        extra_kw: dict = {}
        if custom_input:
            extra_kw["claims_path"] = custom_input

        selected_clusters = self.synth_output.text().strip()
        if selected_clusters:
            extra_kw["clusters_path"] = selected_clusters
        from gui.main_window_common import original_question_for_model
        question = original_question_for_model(self)
        if question:
            cfg._research_config["natural_language_question"] = question
        self.verify_report_view.clear()
        self.verify_btn.setEnabled(False)
        self._active_stage_run_btn = self.verify_btn
        self._active_stage_cancel_btn = None
        self._active_stage_progress = self.robust_verify_progress

        self._pipeline_worker = PipelineStageWorker(
            stage_name="verify",
            config=cfg,
            **extra_kw,
        )
        self._connect_stage_worker(self._on_verify_done, self._on_verify_error)
        self._set_pipeline_stage(9, "running")
        self._begin_robust_verify_run("Stage 9 assembling the final source-linked review…")
        self._pipeline_worker.start()

    def _on_verify_done(self, result):
        self.verify_btn.setEnabled(True)
        summary = summarize_verify(result)
        # Sync first: it refreshes the board from disk (see _on_robustness_done).
        self._sync_all_stage_paths(self.pipeline_config)
        self._show_robust_verify_summary(9, summary)
        report = result.get("report_path", "") if isinstance(result, dict) else ""
        if report:
            try:
                report_text = Path(report).read_text(encoding="utf-8")
                from PySide6.QtCore import QUrl
                self.verify_report_view.document().setBaseUrl(QUrl.fromLocalFile(str(Path(report).parent) + "/"))
                self.verify_report_view.setMarkdown(report_text)
            except OSError as exc:
                self._log(f"Could not display verification report {report}: {exc}",
                          "WARNING")
        self._log("═" * 60)
        if summary.board_state != "done":
            self._abort_run_all("RUN ALL aborted at stage 9 (verify failed).")
            return
        self._schedule_auto_backup("stage 9 verify complete")
        if self._run_all_active:
            self._run_all_active = False
            self._run_all_queue.clear()
            self._log("RUN ALL STAGES finished at stage 9 (verify).")

    def _load_final_review_report(self):
        """Restore only this project's report; clear the previous project's text."""
        self.verify_report_view.clear()
        self.robust_results.clear()
        self.robust_verify_status.clear()
        report = self.pipeline_config.output_dir / "final_review_report.md"
        if report.is_file():
            try:
                self.verify_report_view.setMarkdown(report.read_text(encoding="utf-8"))
                self.robust_verify_status.setText("Saved final review report")
            except OSError as exc:
                self._log(f"Could not restore final review: {exc}", "WARNING")

    def _on_verify_error(self, err):
        self._log(f"VERIFY ERROR: {err}", "ERROR")
        self._set_pipeline_stage(9, "failed", summary=self._pipeline_error_summary(err))
        self.verify_btn.setEnabled(True)
        self._end_robust_verify_run(f"Stage 9 FAILED: {err}")
        self._abort_run_all("RUN ALL aborted at stage 9 (verify error).")

    # ── Robustness & Verify tab status widgets ──

    def _begin_robust_verify_run(self, status: str) -> None:
        self.robust_verify_progress.setRange(0, 0)  # indeterminate: no counts
        self.robust_verify_progress.setVisible(True)
        self.robust_verify_status.setText(status)

    def _end_robust_verify_run(self, status: str) -> None:
        self.robust_verify_progress.setVisible(False)
        self.robust_verify_status.setText(status)

    def _show_robust_verify_summary(self, stage_num: int, summary: StageSummary) -> None:
        """Log a stage summary, mirror it on the tab, and update the board."""
        for level, msg in summary.lines:
            self._log(msg, level)
        self._end_robust_verify_run(summary.status)
        self.robust_results.append(f"[Stage {stage_num}] {summary.status}")
        for level, msg in summary.lines:
            if level != "INFO":
                self.robust_results.append(f"    {level}: {msg}")
        self._set_pipeline_stage(stage_num, summary.board_state,
                                 summary=summary.board_summary)

    def _review_effects(self) -> bool:
        from gui.effect_review_dialog import review_effects
        return review_effects(self)

    def _review_effects_and_rerun(self) -> None:
        if self._review_effects():
            self._start_meta_analysis()

    def _start_meta_analysis(self):
        """Assess all retained papers and pool compatible study results.

        Runs in ``MetaAnalysisWorker``: the assessment re-reads every saved full
        text and takes long enough to freeze the window if run here. It uses its
        own worker slot so a running pipeline stage is left alone.
        """
        running = getattr(self, "_meta_worker", None)
        if running is not None and running.isRunning():
            return
        enhance_control = getattr(self, "meta_enhance", None)
        enhance = bool(enhance_control and enhance_control.isChecked())
        model_control = getattr(self, "meta_use_model", None)
        use_model = bool(model_control and model_control.isChecked())
        if (enhance or use_model) and not self._pipeline_stage_guard("Meta-analysis with the model"):
            return
        self.meta_analysis_status.setText("Assessing statistics from all retained papers...")
        self.meta_analysis_btn.setEnabled(False)
        context_control = getattr(self, "meta_context_tokens", None)
        context_override = context_control.value() if context_control else 0
        self._log(
            "Meta-analysis token action: auto-detect loaded model context; each request "
            "will record its evidence, response, safety, and split/repair allocation."
            if not context_override else
            f"Meta-analysis token action: explicit {context_override:,}-token context override; "
            "each request will record its evidence, response, safety, and split/repair allocation."
        )
        cfg = copy.deepcopy(self.pipeline_config)
        if cfg is not None:
            from gui.main_window_config_sync import _custom_key_for_endpoint
            from gui.main_window_common import original_question_for_model
            cfg.custom_api_key = _custom_key_for_endpoint(cfg)
            if hasattr(self, "_original_question_state"):
                question = original_question_for_model(self)
                if question:
                    cfg._research_config["natural_language_question"] = question
        self._meta_worker = MetaAnalysisWorker(
            cfg, enhance=enhance, use_model=use_model,
            context_tokens=context_override)
        self._meta_worker.progress.connect(self.meta_analysis_status.setText)
        if hasattr(self, "meta_cancel_btn"):
            self.meta_cancel_btn.setEnabled(enhance or use_model)
        if hasattr(self, "meta_review_btn"):
            self.meta_review_btn.setEnabled(False)
        self._meta_worker.finished_analysis.connect(self._on_meta_analysis_done)
        self._meta_worker.error.connect(self._on_meta_analysis_error)
        self._meta_worker.start()

    def _on_meta_analysis_done(self, results):
        """Show a finished meta-analysis (on the GUI thread: it draws the plots)."""
        from gui.meta_analysis_summary import summary_text
        from process.meta_analysis import add_result_figures

        try:
            add_result_figures(results)
            self._meta_analysis_results = results
            report_path = results.get("report_path")
            self.meta_summary.setMarkdown(
                Path(report_path).read_text(encoding="utf-8") if report_path else summary_text(results))
            if hasattr(self, "meta_enhanced_report"):
                enhancement = results.get("enhancement", {})
                enhanced_path = enhancement.get("report_path")
                self.meta_enhanced_report.setMarkdown(
                    Path(enhanced_path).read_text(encoding="utf-8") if enhanced_path else
                    "Model enhancement was not run." if not enhancement else
                    "Model enhancement failed: " + enhancement.get("error", "Unknown error"))
                if enhancement:
                    self.meta_plot_tabs.setCurrentWidget(self.meta_enhanced_report)
                else:
                    self.meta_plot_tabs.setCurrentWidget(self.meta_summary)
            if hasattr(self, "meta_model_audit"):
                audit_path = results.get("model_grouping_review_path")
                self.meta_model_audit.setMarkdown(
                    Path(audit_path).read_text(encoding="utf-8")
                    if audit_path and Path(audit_path).exists()
                    else "Model-assisted semantic grouping was not run."
                )
            self.meta_group_combo.blockSignals(True)
            self.meta_group_combo.clear()
            for index, group in enumerate(results.get("groups", [])):
                if group["pooled"] is not None:
                    self.meta_group_combo.addItem(
                        f"{group['label']} ({group['n_studies']} studies; {group['analysis_scale']})", index)
            self.meta_group_combo.setEnabled(self.meta_group_combo.count() > 0)
            self.meta_group_combo.blockSignals(False)
            self._show_meta_analysis_group()
            n_corpus = results.get("coverage", {}).get("n_corpus_papers", 0)
            n_pooled = results.get("n_studies", 0)
            group_count = self.meta_group_combo.count()
            status = (f"{n_corpus} papers assessed; {n_pooled} papers pooled "
                      f"across {group_count} compatible groups.")
            coverage = results.get("coverage", {})
            if "n_rechecked" in coverage:
                status += (f" Model re-checked {coverage['n_rechecked']} statistics "
                           f"({coverage['n_recheck_failed']} still without a verdict).")
            grouping = results.get("model_grouping")
            if grouping:
                status += (
                    f" Model reviewed {grouping.get('candidates_returned', 0)} candidate groups; "
                    f"{len(grouping['groups'])} passed all semantic gates ({grouping['status']})."
                )
                if grouping["errors"]:
                    status += f" Grouping error: {grouping['errors'][0]}"
            if results.get("skipped"):
                status += " Evidence report written; no group currently meets pooling requirements."
            suggestions = len(results.get("evidence_synthesis", {}).get("suggested_groups", []))
            status += f" {suggestions} suggested groups available for review."
            enhancement = results.get("enhancement")
            if enhancement:
                status += f" Model enhancement: {enhancement['status']}."
            self.meta_analysis_status.setText(status)
            self._log("Meta-analysis: " + status)
        except Exception as exc:
            logger.exception("Meta-analysis failed")
            self._on_meta_analysis_error(str(exc))
        finally:
            self.meta_analysis_btn.setEnabled(True)
            if hasattr(self, "meta_cancel_btn"):
                self.meta_cancel_btn.setEnabled(False)
            if hasattr(self, "meta_review_btn"):
                self.meta_review_btn.setEnabled(True)

    def _on_meta_analysis_error(self, message: str) -> None:
        if hasattr(self, "meta_cancel_btn"):
            self.meta_cancel_btn.setEnabled(False)
        if hasattr(self, "meta_review_btn"):
            self.meta_review_btn.setEnabled(True)
        self.meta_analysis_status.setText(f"Error: {message}")
        self.meta_summary.setPlainText(f"Meta-analysis failed:\n\n{message}")
        self._log(f"Meta-analysis error: {message}")
        self.meta_analysis_btn.setEnabled(True)

    def _show_meta_analysis_group(self, *_):
        """Render the selected group, clearing stale plots on an unpoolable run."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from process.meta_analysis import create_forest_plot_figure, create_funnel_plot_figure
        import matplotlib.pyplot as plt

        results = getattr(self, "_meta_analysis_results", None)
        if not results:
            return
        index = self.meta_group_combo.currentData()
        if index is None:
            forest, funnel = results["forest_fig"], results["funnel_fig"]
        else:
            group = results["groups"][index]
            forest = create_forest_plot_figure(group["pooled"], title=group["label"])
            funnel = create_funnel_plot_figure(group["studies"], group["pooled"].pooled_effect,
                                               group["bias"], title=group["label"])
            plt.close(results["forest_fig"])
            plt.close(results["funnel_fig"])
        for attribute, title, figure in (("forest_canvas", "Forest Plot", forest),
                                         ("funnel_canvas", "Funnel Plot", funnel)):
            old = getattr(self, attribute)
            tab_index = self.meta_plot_tabs.indexOf(old)
            self.meta_plot_tabs.removeTab(tab_index)
            plt.close(old.figure)
            old.deleteLater()
            canvas = FigureCanvasQTAgg(figure)
            self.meta_plot_tabs.insertTab(tab_index, canvas, title)
            setattr(self, attribute, canvas)

    def _start_compare_runs(self):
        if not self._pipeline_stage_guard("Stage 10 (Compare Runs)"):
            return
        self._log("=" * 60)
        self._log("STAGE 10: COMPARE RUNS")
        self._log("=" * 60)

        from pathlib import Path as _Path

        path_a = self.compare_run_a.text().strip()
        path_b = self.compare_run_b.text().strip()

        if not path_a or not path_b:
            self._log("Please specify both Run A and Run B file paths.", "ERROR")
            return

        path_a = _Path(path_a)
        path_b = _Path(path_b)

        if not path_a.exists():
            self._log(f"Run A file not found: {path_a}", "ERROR")
            return
        if not path_b.exists():
            self._log(f"Run B file not found: {path_b}", "ERROR")
            return

        self._log(f"Run A: {path_a}")
        self._log(f"Run B: {path_b}")

        comparison_name = "run_compare"
        output_path = self.compare_output.text().strip()
        if output_path:
            comparison_name = _Path(output_path).stem
        # Only a path the user chose overrides Stage 10's default report location.
        user_report_path = output_path if output_path and output_path != getattr(
            self.compare_output, "_auto_path_text", None) else None

        self.compare_btn.setEnabled(False)
        self.compare_progress.setVisible(True)
        self.compare_progress.setRange(0, 0)  # indeterminate
        self.compare_status.setText("Comparing runs...")
        self._active_stage_run_btn = self.compare_btn
        self._active_stage_cancel_btn = None
        self._active_stage_progress = self.compare_progress

        self._pipeline_worker = PipelineStageWorker(
            stage_name="compare_runs",
            config=copy.deepcopy(self.pipeline_config),
            claims_paths=[str(path_a), str(path_b)],
            comparison_name=comparison_name,
            **({"report_path": user_report_path} if user_report_path else {}),
        )
        self._connect_stage_worker(self._on_compare_done, self._on_compare_error)
        self._set_pipeline_stage(10, "running")
        self._pipeline_worker.start()

    def _on_compare_done(self, result):
        self.compare_btn.setEnabled(True)
        self.compare_progress.setVisible(False)
        self.compare_status.setText("Comparison complete.")
        if isinstance(result, dict):
            overall = result.get("overall_summary", {})
            stability = overall.get("stability_score")
            interp = overall.get("interpretation", "N/A")
            n_runs = result.get("n_runs", 0)
            pairwise = result.get("pairwise_comparisons", [])

            self._log(f"Compared {n_runs} runs, {len(pairwise)} pair(s)")
            if stability is not None:
                self._log(f"Stability: {stability:.3f} ({interp})")
            else:
                self._log(f"Stability: {interp}")

            for i, p in enumerate(pairwise, 1):
                shared = p.get("shared_papers", 0)
                pj = p.get("paper_jaccard")
                wk = p.get("score_weighted_kappa")
                jaccard_text = f"{pj:.3f}" if pj is not None else "N/A"
                kappa_text = f"{wk:.3f}" if wk is not None else "N/A"
                self._log(
                    f"  Pair {i}: {shared} shared papers, "
                    f"Jaccard={jaccard_text}, wKappa={kappa_text}")

            report_path = result.get("report_path", "")
            if report_path:
                self._log(f"Report: {report_path}")
                try:
                    with open(report_path, encoding="utf-8") as f:
                        self.compare_results.setPlainText(f.read())
                except OSError as exc:
                    self.compare_results.clear()
                    self._log(f"Could not display compare report {report_path}: {exc}",
                              "WARNING")
            summary = f"{n_runs} runs compared"
            if interp and interp != "N/A":
                summary = f"{summary} ({interp})"
            self._set_pipeline_stage(10, "done", summary=summary)
        else:
            self._log(f"Compare result: {result}")
            self._set_pipeline_stage(10, "done", summary="Comparison complete")
        self._log("=" * 60)
        self._sync_all_stage_paths(self.pipeline_config)
        self._schedule_auto_backup("stage 10 compare complete")

    def _on_compare_error(self, err):
        self._log(f"COMPARE RUNS ERROR: {err}", "ERROR")
        self._set_pipeline_stage(10, "failed", summary=self._pipeline_error_summary(err))
        self.compare_btn.setEnabled(True)
        self.compare_progress.setVisible(False)
        self.compare_status.setText(f"Error: {err}")
