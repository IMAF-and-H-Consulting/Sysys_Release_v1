"""Tab builders for Search, Score & Filter, Extract & Validate, and Audit.

Split from gui/main_window.py; mixed into MainWindow.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.main_window_common import _make_file_row, _make_llm_backend_row
from gui.widgets import SectionHeader

_SEARCH_YEAR_FLOOR = 1900
_SEARCH_YEAR_CEILING = 2100
_STAT_BUNDLE_TOKENS_DEFAULT = 0  # automatic: derive prompt/reply packing from model context
_STAT_BUNDLE_TOKENS_MAX = 2_147_483_647


def _make_read_only_output(edit: QLineEdit, browse_btn: QPushButton, tip: str) -> None:
    """Turn a file row the stage does not read into a display of its fixed output."""
    edit.setReadOnly(True)
    browse_btn.setEnabled(False)
    edit.setToolTip(tip)
    browse_btn.setToolTip(tip)


def _citation_snowball_checkbox(config) -> QCheckBox:
    """Search-tab switch for ``config.citation_snowball_enabled``."""
    seeds = int(getattr(config, "citation_snowball_seed_limit", 0) or 0)
    per_seed = int(getattr(config, "citation_snowball_relation_limit", 0) or 0)
    box = QCheckBox("Citation snowballing")
    box.setChecked(bool(getattr(config, "citation_snowball_enabled", True)))
    box.setToolTip(
        "After the database search, fetch papers that cite or are cited by the "
        f"{seeds} most-cited Semantic Scholar results (up to {per_seed} of each "
        "per paper): up to "
        f"~{seeds * per_seed * 2} extra records, tagged citation_snowball."
    )
    return box


class ScreeningTabsMixin:
    """Tab builders for Search, Score & Filter, Extract & Validate, and Audit."""

        # The Log is no longer a tab; it lives in a floating window opened from
        # the File menu (built separately in __init__).

    def _section_help_button(self, help_key: str) -> QPushButton:
        """Small '?' button opening the help entry for one section of a tab."""
        from gui.dialogs import HelpDialog

        btn = QPushButton("?")
        btn.setFixedSize(24, 24)
        btn.setToolTip("Learn about this section")
        btn.setAccessibleName(f"Help for {help_key} section")
        btn.clicked.connect(lambda _checked=False: HelpDialog(help_key, self).exec())
        return btn

    # ── Tab 0: Search ─────────────────────────────────────────────────

    def _search_filter_kwargs(self) -> dict:
        """Search-tab year range and full-text-only filter for Stage 1."""
        return {
            "min_year": self.min_year.value(),
            "max_year": self.max_year.value(),
            "fulltext_only": self.fulltext_only.isChecked(),
        }

    def _build_search_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Hypothesis component list
        layout.addWidget(SectionHeader("Hypothesis Components",
                                       "Edit components of the research hypothesis"))
        self.component_list = QTableWidget(0, 5)
        self.component_list.setHorizontalHeaderLabels(["ID", "Label", "Keywords", "Weight", "Boolean Term"])
        self.component_list.setMinimumHeight(120)
        layout.addWidget(self.component_list)

        comp_btns = QHBoxLayout()
        edit_btn = QPushButton("Edit Component")
        edit_btn.clicked.connect(self._edit_component)
        reset_btn = QPushButton("Reset Components")
        reset_btn.clicked.connect(self._reset_components)
        comp_btns.addWidget(edit_btn)
        comp_btns.addWidget(reset_btn)
        comp_btns.addStretch()
        layout.addLayout(comp_btns)

        # Sysys search question. This is the first-class input; components
        # can be edited ahead of time or regenerated from this question.
        layout.addWidget(SectionHeader("Search Question"))
        self.sysys_nl_query = QLineEdit()
        self.sysys_nl_query.setPlaceholderText("Natural language research question...")
        self.sysys_nl_query.setAccessibleName("Search question")
        self.sysys_nl_query.textChanged.connect(self._on_search_question_changed)
        layout.addWidget(self.sysys_nl_query)

        self.boolean_query_preview = QTextEdit()
        self.boolean_query_preview.setReadOnly(True)
        self.boolean_query_preview.setAccessibleName("Weighted Boolean query preview")
        self.boolean_query_preview.setMaximumHeight(74)
        self.boolean_query_preview.setPlaceholderText(
            "Weighted Boolean query preview will appear after components are prepared."
        )
        layout.addWidget(self.boolean_query_preview)

        # API source checkboxes
        layout.addWidget(SectionHeader("Data Sources"))
        src_row = QHBoxLayout()
        self.src_pubmed = QCheckBox("PubMed")
        self.src_pubmed.setChecked(True)
        self.src_s2 = QCheckBox("Semantic Scholar")
        self.src_s2.setChecked(True)
        self.src_openalex = QCheckBox("OpenAlex")
        self.src_openalex.setChecked(True)
        self.src_elicit = QCheckBox("Elicit")
        self.src_scopus = QCheckBox("Scopus")
        self.src_wos = QCheckBox("Web of Science")
        for cb in [self.src_pubmed, self.src_s2, self.src_openalex,
                   self.src_elicit, self.src_scopus, self.src_wos]:
            src_row.addWidget(cb)
        src_row.addStretch()
        layout.addLayout(src_row)

        limit_row = QHBoxLayout()
        lbl_paper_limit = QLabel("Paper Limit:")
        limit_row.addWidget(lbl_paper_limit)
        self.paper_limit = QSpinBox()
        self.paper_limit.setRange(1, 10000)
        self.paper_limit.setSingleStep(50)
        self.paper_limit.setValue(500)
        self.paper_limit.setSuffix(" papers")
        self.paper_limit.setToolTip(
            "Maximum papers to pull across selected search engines. "
            "The total is divided across engines, then across each engine's query lines."
        )
        lbl_paper_limit.setBuddy(self.paper_limit)
        limit_row.addWidget(self.paper_limit)
        limit_row.addStretch()
        layout.addLayout(limit_row)

        # Year range + options
        yr_row = QHBoxLayout()
        lbl_min = QLabel("Min Year:")
        yr_row.addWidget(lbl_min)
        self.min_year = QSpinBox()
        self.min_year.setRange(_SEARCH_YEAR_FLOOR, _SEARCH_YEAR_CEILING)
        self.min_year.setValue(2000)
        self.min_year.setToolTip(
            "Papers published before this year are dropped from new search "
            "results. Papers without a year are kept."
        )
        lbl_min.setBuddy(self.min_year)
        yr_row.addWidget(self.min_year)
        lbl_max = QLabel("Max Year:")
        yr_row.addWidget(lbl_max)
        self.max_year = QSpinBox()
        self.max_year.setRange(_SEARCH_YEAR_FLOOR, _SEARCH_YEAR_CEILING)
        self.max_year.setValue(min(_SEARCH_YEAR_CEILING, date.today().year + 1))
        self.max_year.setToolTip(
            "Papers published after this year are dropped from new search results."
        )
        lbl_max.setBuddy(self.max_year)
        yr_row.addWidget(self.max_year)

        self.fulltext_only = QCheckBox("Full text only")
        self.fulltext_only.setToolTip(
            "Keep only new papers whose search provider reports freely available "
            "full text (PubMed free full text filter, Semantic Scholar open-access "
            "PDF, OpenAlex open-access status, or a PMC id)."
        )
        yr_row.addWidget(self.fulltext_only)

        self.fetch_fulltext = QCheckBox("Download full text")
        self.fetch_fulltext.setToolTip(
            "Search + download in one step: when checked, Run Sysys Search "
            "automatically downloads full text for the corpus after the search "
            "completes (same as clicking Fetch Articles afterwards)."
        )
        yr_row.addWidget(self.fetch_fulltext)

        self.citation_snowball = _citation_snowball_checkbox(self.pipeline_config)
        yr_row.addWidget(self.citation_snowball)

        yr_row.addStretch()
        layout.addLayout(yr_row)

        # Search / Fetch / Analysis buttons
        fetch_row = QHBoxLayout()
        self.sysys_search_btn = QPushButton("Run Sysys Search")
        self.sysys_search_btn.clicked.connect(self._start_sysys_search)
        self.fetch_btn = QPushButton("Fetch Articles")
        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setToolTip(
            "Download full text for the papers already in the corpus. "
            "This does not run a new search — use Run Sysys Search first to "
            "build the corpus."
        )
        self.fetch_btn.clicked.connect(self._start_fetch)
        self.analysis_btn = QPushButton("Run Keyword Analysis")
        self.analysis_btn.setEnabled(False)
        self.analysis_btn.setToolTip(
            "Available after fetched search results exist. It updates the query-term weights."
        )
        self.analysis_btn.clicked.connect(self._start_analysis)
        self.sysys_cancel_btn = QPushButton("Cancel Search")
        self.sysys_cancel_btn.setEnabled(False)
        self.sysys_cancel_btn.clicked.connect(self._cancel_sysys_search)
        self.cancel_fetch_btn = QPushButton("Cancel Fetch")
        self.cancel_fetch_btn.setEnabled(False)
        self.cancel_fetch_btn.clicked.connect(self._cancel_work)
        fetch_row.addWidget(self.sysys_search_btn)
        fetch_row.addWidget(self.fetch_btn)
        fetch_row.addWidget(self.analysis_btn)
        fetch_row.addWidget(self.sysys_cancel_btn)
        fetch_row.addWidget(self.cancel_fetch_btn)
        fetch_row.addStretch()
        layout.addLayout(fetch_row)

        self.search_progress = QProgressBar()
        self.search_progress.setVisible(False)
        layout.addWidget(self.search_progress)

        self.search_status = QLabel("Ready")
        layout.addWidget(self.search_status)

        # Feature E: inline search results preview
        self.search_results_display = QPlainTextEdit()
        self.search_results_display.setReadOnly(True)
        self.search_results_display.setAccessibleName("Search results preview")
        self.search_results_display.setPlaceholderText(
            "Search results will appear here after search completes…"
        )
        self.search_results_display.setMaximumHeight(200)
        layout.addWidget(self.search_results_display)

        layout.addStretch()
        self._add_help_button(tab, "search")
        return tab

    # ── Tab 1: Score & Filter ─────────────────────────────────────────

    def _build_score_filter_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        json_f = "JSON files (*.json);;All files (*)"

        # ── Top half: two columns ──────────────────────────────────────
        #   left  = BRT-BERT Score & Filter (Stage 2)
        #   right = Stage 2c Statistical Evidence Extraction
        top_row = QHBoxLayout()

        # ---- Left column: BRT-BERT Score & Filter ----
        left_col = QVBoxLayout()
        left_col.addWidget(
            SectionHeader("Score & Filter", "Stage 2: relevance screening")
        )

        # Workload selector
        wl_row = QHBoxLayout()
        lbl_workload = QLabel("Workload:")
        wl_row.addWidget(lbl_workload)
        self.score_workload = QComboBox()
        self.score_workload.addItems(["abstract", "fulltext"])
        lbl_workload.setBuddy(self.score_workload)
        wl_row.addWidget(self.score_workload)
        wl_row.addStretch()
        left_col.addLayout(wl_row)

        screening_row = QHBoxLayout()
        screening_row.addWidget(QLabel("Screening strategy:"))
        self.score_screening_mode = QComboBox()
        for label, mode in (
            ("Laya → BRT-BERT (default)", "laya-bert"),
            ("Existing BERT / LLM scorer", "off"),
            ("BRT-BERT → Laya (comparison)", "bert-laya"),
            ("Laya only (baseline)", "laya-only"),
            ("Packaged BRT-BERT only (baseline)", "packaged-bert"),
        ):
            self.score_screening_mode.addItem(label, mode)
        self.score_screening_mode.setCurrentIndex(max(0, self.score_screening_mode.findData(
            self.pipeline_config.relevance_screening_mode)))
        self.score_screening_mode.currentIndexChanged.connect(
            self._on_screening_strategy_changed)
        screening_row.addWidget(self.score_screening_mode, 1)
        left_col.addLayout(screening_row)
        self.score_cascade_hint = QLabel(
            "pw4: Laya bin ≥ 4 (P ≥ 0.6), then bundled BRT-BERT P ≥ 0.236. "
            "Cascade/baselines use these probability cutoffs, not the score threshold below. "
            "Model Score is 0/5 for reject/keep; detailed probabilities are saved."
        )
        self.score_cascade_hint.setWordWrap(True)
        left_col.addWidget(self.score_cascade_hint)

        # Legacy scorer information (shown only when the cascade is disabled)
        self.score_simple_label = QLabel(
            "Legacy relevance model: BRT-BERT (BERT Relevance Trained). "
            "(change via File → Settings → “Show advanced LLM options”)"
        )
        self.score_simple_label.setStyleSheet("color: #444; font-style: italic;")
        self.score_simple_label.setWordWrap(True)
        left_col.addWidget(self.score_simple_label)

        # Advanced container — provider/model row + relevance-model dropdown.
        # Hidden by default; revealed when the user opts into advanced LLM
        # options in Settings (QSettings key score/show_advanced_llm).
        self.score_advanced_box = QWidget()
        adv_layout = QVBoxLayout(self.score_advanced_box)
        adv_layout.setContentsMargins(0, 0, 0, 0)

        row, self.score_provider, self.score_model = _make_llm_backend_row(
            endpoint_getter=self._custom_endpoint
        )
        self.score_provider.addItem("brt-bert")
        adv_layout.addLayout(row)

        rm_row = QHBoxLayout()
        rm_lbl = QLabel("Relevance model:")
        rm_row.addWidget(rm_lbl)
        self.score_relevance_model = QComboBox()
        # brt-bert (resolves to brt-bert-relevance/final) is the DEFAULT
        # relevance scorer for Stage 2; other models remain selectable.
        self.score_relevance_model.setEditable(True)
        self.score_relevance_model.addItems(["brt-bert", "housecatbert-scibert"])
        self.score_relevance_model.setCurrentText(
            getattr(self.pipeline_config, "relevance_model", "") or "brt-bert"
        )
        # Persist user edits against programmatic syncs (see _sync_settings_to_tabs).
        self.score_relevance_model._user_customized = False
        self.score_relevance_model.activated.connect(
            lambda *_: setattr(self.score_relevance_model, "_user_customized", True)
        )
        relevance_model_line_edit = self.score_relevance_model.lineEdit()
        if relevance_model_line_edit is not None:
            relevance_model_line_edit.textEdited.connect(
                lambda *_: setattr(self.score_relevance_model, "_user_customized", True)
            )
        rm_lbl.setBuddy(self.score_relevance_model)
        rm_row.addWidget(self.score_relevance_model, 1)
        adv_layout.addLayout(rm_row)

        left_col.addWidget(self.score_advanced_box)

        # Threshold / batch
        param_row = QHBoxLayout()
        lbl_threshold = QLabel("Threshold:")

        param_row.addWidget(lbl_threshold)
        self.score_threshold = QSpinBox()
        self.score_threshold.setRange(1, 10)
        self.score_threshold.setValue(3)
        lbl_threshold.setBuddy(self.score_threshold)
        param_row.addWidget(self.score_threshold)
        lbl_batch = QLabel("Batch Size:")
        param_row.addWidget(lbl_batch)
        self.score_batch = QSpinBox()
        self.score_batch.setRange(1, 50)
        self.score_batch.setValue(1)
        lbl_batch.setBuddy(self.score_batch)
        param_row.addWidget(self.score_batch)
        param_row.addStretch()
        left_col.addLayout(param_row)

        # Corpus path
        cp_row, self.score_corpus_path, cp_btn = _make_file_row(
            "Corpus Path:", "path/to/corpus.json"
        )
        left_col.addLayout(cp_row)
        cp_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.score_corpus_path, json_f, self.pipeline_config.corpus_dir
            )
        )

        # Run / Cancel
        btn_row = QHBoxLayout()
        self.score_run_btn = QPushButton("Run Score & Filter")
        self.score_run_btn.clicked.connect(self._start_sysys_score)
        self.score_cancel_btn = QPushButton("Cancel")
        self.score_cancel_btn.setEnabled(False)
        self.score_cancel_btn.clicked.connect(self._cancel_pipeline_stage)
        btn_row.addWidget(self.score_run_btn)
        btn_row.addWidget(self.score_cancel_btn)
        btn_row.addStretch()
        left_col.addLayout(btn_row)

        self.score_progress = QProgressBar()
        self.score_progress.setVisible(False)
        left_col.addWidget(self.score_progress)
        self.score_status = QLabel("")
        left_col.addWidget(self.score_status)
        left_col.addStretch()

        # ---- Right column: Stage 2c Statistical Evidence Extraction ----
        right_col = QVBoxLayout()
        stat_header = QHBoxLayout()
        stat_header.addWidget(
            SectionHeader("Stage 2c: Statistical Evidence Extraction"), 1
        )
        stat_header.addWidget(
            self._section_help_button("statistical_extraction"),
            0, Qt.AlignmentFlag.AlignTop,
        )
        right_col.addLayout(stat_header)
        stat_in_row, self.stat_extract_input, stat_in_btn = _make_file_row(
            "Scored Corpus:", "path/to/relevant.json"
        )
        right_col.addLayout(stat_in_row)
        stat_out_row, self.stat_extract_output, stat_out_btn = _make_file_row(
            "Extraction Output:", "path/to/statistical_extractions.json", browse_save=True
        )
        right_col.addLayout(stat_out_row)
        stat_in_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.stat_extract_input, json_f, self.pipeline_config.claims_dir
            )
        )
        stat_out_btn.clicked.connect(
            lambda: self._browse_save_file(
                self.stat_extract_output, json_f, self.pipeline_config.claims_dir
            )
        )
        stat_llm_row, self.stat_extract_provider, self.stat_extract_model = (
            _make_llm_backend_row(
                endpoint_getter=self._custom_endpoint
            )
        )
        right_col.addLayout(stat_llm_row)

        stat_options = QHBoxLayout()
        self.stat_extract_llm_context = QCheckBox("LLM context check")
        self.stat_extract_llm_context.toggled.connect(
            self._on_stat_extract_llm_context_toggled
        )
        stat_options.addWidget(self.stat_extract_llm_context)
        lbl_stat_max = QLabel("Scope limit:")
        stat_options.addWidget(lbl_stat_max)
        self.stat_extract_max_contexts = QSpinBox()
        self.stat_extract_max_contexts.setRange(0, 2_147_483_647)
        self.stat_extract_max_contexts.setValue(0)
        self.stat_extract_max_contexts.setSpecialValueText("All candidates")
        self.stat_extract_max_contexts.setToolTip(
            "All candidates applies no synthetic candidate-count cap. A positive value "
            "is an explicit user scope limit used only to control model runtime/cost; "
            "the saved Stage 2c audit records how many candidates were not model-reviewed."
        )
        lbl_stat_max.setBuddy(self.stat_extract_max_contexts)
        stat_options.addWidget(self.stat_extract_max_contexts)
        self.stat_extract_full_llm = QCheckBox("LLM full statistical extraction")
        self.stat_extract_full_llm.setToolTip(
            "Review every deterministic Stage 2c statistical candidate with the selected LLM."
        )
        self.stat_extract_full_llm.toggled.connect(
            self._on_stat_extract_full_llm_toggled
        )
        stat_options.addWidget(self.stat_extract_full_llm)
        lbl_stat_bundle = QLabel("Bundle override:")
        stat_options.addWidget(lbl_stat_bundle)
        self.stat_extract_bundle_tokens = QSpinBox()
        self.stat_extract_bundle_tokens.setRange(0, _STAT_BUNDLE_TOKENS_MAX)
        self.stat_extract_bundle_tokens.setSingleStep(1024)
        self.stat_extract_bundle_tokens.setValue(_STAT_BUNDLE_TOKENS_DEFAULT)
        self.stat_extract_bundle_tokens.setSpecialValueText("Automatic")
        self.stat_extract_bundle_tokens.setToolTip(
            "Automatic packs a paper's statistics using the detected/configured model "
            "context while reserving a complete JSON verdict for every statistic and a "
            "safety margin. A positive value is an explicit synthetic total-request limit. "
            "Incomplete replies are removed from cache and retried in smaller bundles."
        )
        lbl_stat_bundle.setBuddy(self.stat_extract_bundle_tokens)
        stat_options.addWidget(self.stat_extract_bundle_tokens)
        stat_options.addStretch()
        right_col.addLayout(stat_options)

        stat_btn_row = QHBoxLayout()
        self.stat_extract_btn = QPushButton("Run Statistical Extraction")
        self.stat_extract_btn.clicked.connect(self._start_statistical_extract)
        self.stat_extract_cancel_btn = QPushButton("Cancel")
        self.stat_extract_cancel_btn.setEnabled(False)
        self.stat_extract_cancel_btn.clicked.connect(self._cancel_pipeline_stage)
        stat_btn_row.addWidget(self.stat_extract_btn)
        self.stat_review_btn = QPushButton("Review Effects && Groups…")
        self.stat_review_btn.setToolTip(
            "See every effect Stage 2c found or converted, choose which to pool, "
            "and group them.")
        self.stat_review_btn.clicked.connect(self._review_effects)
        stat_btn_row.addWidget(self.stat_review_btn)
        stat_btn_row.addWidget(self.stat_extract_cancel_btn)
        stat_btn_row.addStretch()
        right_col.addLayout(stat_btn_row)

        self.stat_extract_progress = QProgressBar()
        self.stat_extract_progress.setVisible(False)
        right_col.addWidget(self.stat_extract_progress)
        self.stat_extract_status = QLabel("")
        right_col.addWidget(self.stat_extract_status)
        right_col.addStretch()

        top_row.addLayout(left_col, 1)
        top_row.addLayout(right_col, 1)
        outer.addLayout(top_row, 1)

        # ── Bottom half: Human Relevance Calibration (full width) ───────
        human_group = QGroupBox("Human Relevance Calibration")
        human_layout = QVBoxLayout(human_group)

        human_controls = QHBoxLayout()
        lbl_human_sample = QLabel("Sample Size:")
        human_controls.addWidget(lbl_human_sample)
        self.human_review_sample = QSpinBox()
        self.human_review_sample.setRange(0, 500)
        self.human_review_sample.setSpecialValueText("Auto")
        self.human_review_sample.setValue(0)
        lbl_human_sample.setBuddy(self.human_review_sample)
        human_controls.addWidget(self.human_review_sample)
        self.human_review_build_btn = QPushButton("Build Queue")
        self.human_review_build_btn.clicked.connect(self._build_human_relevance_queue)
        self.human_review_edit_cells = QCheckBox("Edit Cells")
        self.human_review_edit_cells.setToolTip(
            "Allow corrections to the title, year, model score, ID, and abstract cells"
        )
        self.human_review_edit_cells.toggled.connect(
            self._set_human_relevance_editing
        )
        self.human_review_save_btn = QPushButton("Save Review")
        self.human_review_save_btn.clicked.connect(self._save_human_relevance_review)
        self.human_review_compare_btn = QPushButton("Compare to Model Scores")
        self.human_review_compare_btn.clicked.connect(self._compare_human_relevance_review)
        human_controls.addWidget(self.human_review_build_btn)
        human_controls.addWidget(self.human_review_edit_cells)
        human_controls.addWidget(self.human_review_save_btn)
        human_controls.addWidget(self.human_review_compare_btn)
        human_controls.addStretch()
        human_layout.addLayout(human_controls)

        self.human_review_table = QTableWidget(0, 7)
        self.human_review_table.setHorizontalHeaderLabels(
            [
                "Human Score",
                "Human Verified",
                "Title",
                "Year",
                "Model Score",
                "ID",
                "Abstract",
            ]
        )
        self.human_review_table.setAccessibleName("Human relevance review table")
        self.human_review_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.human_review_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.human_review_table.verticalHeader().setVisible(False)
        human_header = self.human_review_table.horizontalHeader()
        human_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        human_header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.human_review_table.setMinimumHeight(220)
        human_layout.addWidget(self.human_review_table)

        self.human_review_status = QLabel("No human queue loaded")
        human_layout.addWidget(self.human_review_status)
        outer.addWidget(human_group, 1)

        self._human_review_artifact: dict | None = None
        self._human_review_rows: list[dict] = []
        self._loaded_human_review_path: Path | None = None

        self._add_help_button(tab, "score_filter")
        return tab

    # ── Tab 2: Extract & Validate ─────────────────────────────────────

    def _build_extract_validate_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        json_f = "JSON files (*.json);;All files (*)"

        # Stage 3: Extract
        layout.addWidget(SectionHeader("Stage 3: Claim Extraction"))
        e_row, self.extract_corpus, e_btn = _make_file_row(
            "Scored Corpus:", "path/to/relevant.json"
        )
        layout.addLayout(e_row)
        eo_row, self.extract_output, eo_btn = _make_file_row(
            "Claims Output:", "path/to/claims.json", browse_save=True
        )
        layout.addLayout(eo_row)
        e_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.extract_corpus, json_f, self.pipeline_config.claims_dir
            )
        )
        eo_btn.clicked.connect(
            lambda: self._browse_save_file(
                self.extract_output, json_f, self.pipeline_config.claims_dir
            )
        )

        er, self.extract_provider, self.extract_model = _make_llm_backend_row(
            endpoint_getter=self._custom_endpoint
        )
        layout.addLayout(er)

        ex_params = QHBoxLayout()
        lbl_max_tokens = QLabel("Output Override:")
        ex_params.addWidget(lbl_max_tokens)
        self.extract_max_tokens = QSpinBox()
        self.extract_max_tokens.setRange(0, 2_147_483_647)
        self.extract_max_tokens.setValue(0)
        self.extract_max_tokens.setSpecialValueText("Automatic")
        self.extract_max_tokens.setToolTip(
            "Automatic sizes the response from the model context and extraction schema. "
            "A positive value is an explicit synthetic ceiling and is recorded with the run."
        )
        lbl_max_tokens.setBuddy(self.extract_max_tokens)
        ex_params.addWidget(self.extract_max_tokens)
        self.extract_thinking = QCheckBox("Use Thinking")
        ex_params.addWidget(self.extract_thinking)
        ex_params.addStretch()
        layout.addLayout(ex_params)

        ex_btn_row = QHBoxLayout()
        self.extract_btn = QPushButton("Run Claim Extraction")
        self.extract_btn.clicked.connect(self._start_extract)
        self.extract_cancel_btn = QPushButton("Cancel")
        self.extract_cancel_btn.setEnabled(False)
        self.extract_cancel_btn.clicked.connect(self._cancel_pipeline_stage)
        ex_btn_row.addWidget(self.extract_btn)
        ex_btn_row.addWidget(self.extract_cancel_btn)
        ex_btn_row.addStretch()
        layout.addLayout(ex_btn_row)

        # Stage 4: Validate
        layout.addWidget(SectionHeader("Stage 4: Method Review / Validation"))
        v_row, self.validate_claims, v_btn = _make_file_row(
            "Claims JSON:", "path/to/claims.json"
        )
        layout.addLayout(v_row)
        vo_row, self.validate_output, vo_btn = _make_file_row(
            "Validated Output:", "path/to/method_profiles_validated.json", browse_save=True
        )
        _make_read_only_output(
            self.validate_output, vo_btn,
            "Stage 4 always writes method_profiles_validated.json (plus "
            "method_review_manual.csv and method_type_summary.json) in the "
            "project's data/method_review folder; this field shows that location.",
        )
        layout.addLayout(vo_row)
        json_f = "JSON files (*.json);;All files (*)"
        v_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.validate_claims, json_f, self.pipeline_config.claims_dir
            )
        )
        mr_dir = self.pipeline_config.data_dir / "method_review"
        vo_btn.clicked.connect(
            lambda: self._browse_save_file(
                self.validate_output, json_f, mr_dir, self.pipeline_config.claims_dir
            )
        )
        vr, self.validate_provider, self.validate_model = _make_llm_backend_row(
            endpoint_getter=self._custom_endpoint
        )
        layout.addLayout(vr)

        v_btn_row = QHBoxLayout()
        self.validate_btn = QPushButton("Run Method Review")
        self.validate_btn.clicked.connect(self._start_validate)
        self.validate_cancel_btn = QPushButton("Cancel")
        self.validate_cancel_btn.setEnabled(False)
        self.validate_cancel_btn.clicked.connect(self._cancel_pipeline_stage)
        v_btn_row.addWidget(self.validate_btn)
        v_btn_row.addWidget(self.validate_cancel_btn)
        v_btn_row.addStretch()
        layout.addLayout(v_btn_row)

        self.extract_validate_progress = QProgressBar()
        self.extract_validate_progress.setVisible(False)
        layout.addWidget(self.extract_validate_progress)
        self.extract_validate_status = QLabel("")
        layout.addWidget(self.extract_validate_status)

        layout.addStretch()
        self._add_help_button(tab, "extract_validate")
        return tab

    # ── Tab 3: Audit ──────────────────────────────────────────────────

    def _build_audit_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Stage 5: Audit
        layout.addWidget(SectionHeader("Stage 5: Span Audit"))
        a_row, self.audit_input, a_btn = _make_file_row(
            "Claims JSON:", "path/to/claims.json"
        )
        self.audit_input.setToolTip("Stage 3 claims (claims.json) to audit.")
        layout.addLayout(a_row)
        ao_row, self.audit_output, ao_btn = _make_file_row(
            "Audited Output:", "path/to/claims_filtered.json", browse_save=True
        )
        _make_read_only_output(
            self.audit_output, ao_btn,
            "Stage 5 always writes claims_filtered.json (plus removal_log.json and "
            "audit_flags.json) in the project's claims folder; this field shows "
            "that location.",
        )
        layout.addLayout(ao_row)
        json_f = "JSON files (*.json);;All files (*)"
        a_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.audit_input, json_f, self.pipeline_config.claims_dir
            )
        )
        ao_btn.clicked.connect(
            lambda: self._browse_save_file(
                self.audit_output, json_f, self.pipeline_config.claims_dir
            )
        )

        sim_row = QHBoxLayout()
        lbl_sim_thresh = QLabel("Similarity Threshold:")
        sim_row.addWidget(lbl_sim_thresh)
        self.audit_threshold = QSlider(Qt.Orientation.Horizontal)
        self.audit_threshold.setRange(50, 100)
        lbl_sim_thresh.setBuddy(self.audit_threshold)
        self.audit_threshold.setValue(85)
        self.audit_threshold.setToolTip(
            "Semantic similarity at or above which a flagged quote is reclassified "
            "as grounded. Papers with no grounded quote are flagged in "
            "audit_flags.json; nothing is removed by default."
        )
        self.audit_threshold_label = QLabel("0.85")
        self.audit_threshold.valueChanged.connect(
            lambda v: self.audit_threshold_label.setText(f"{v / 100:.2f}")
        )
        sim_row.addWidget(self.audit_threshold)
        sim_row.addWidget(self.audit_threshold_label)
        layout.addLayout(sim_row)

        self.audit_exclude_ungrounded = QCheckBox("Exclude papers with no grounded quote")
        self.audit_exclude_ungrounded.setChecked(False)
        self.audit_exclude_ungrounded.setToolTip(
            "Off: papers with no grounded quote are kept and listed in audit_flags.json.\n"
            "On: those papers are removed and logged in removal_log.json."
        )
        layout.addWidget(self.audit_exclude_ungrounded)

        self.audit_btn = QPushButton("Run Span Audit")
        self.audit_btn.clicked.connect(self._start_audit)
        layout.addWidget(self.audit_btn)

        self.audit_progress = QProgressBar()
        self.audit_progress.setVisible(False)
        layout.addWidget(self.audit_progress)
        self.audit_status = QLabel("")
        layout.addWidget(self.audit_status)

        layout.addStretch()
        self._add_help_button(tab, "audit")
        return tab
