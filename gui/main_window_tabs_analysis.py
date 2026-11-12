"""Tab builders for Synthesis, Fulltext, Robustness & Verify, and Text Scaling.

Split from gui/main_window.py; mixed into MainWindow.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.main_window_common import _make_file_row, _make_llm_backend_row
from gui.widgets import MetricCard, SectionHeader


class AnalysisTabsMixin:
    """Tab builders for Synthesis, Fulltext, Robustness & Verify, and Text Scaling."""

    # ── Tab 4: Synthesis ──────────────────────────────────────────────

    def _build_synthesis_tab(self) -> QWidget:
        """Stage 6 gets its own tab so the cluster table, abstract,
        hypothesis rationale, source citations, and evidence network each
        have enough room to be readable."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Stage 6: Synthesize controls
        layout.addWidget(SectionHeader("Stage 6: Synthesis"))
        s_row, self.synth_input, s_btn = _make_file_row(
            "Claims JSON:", "path/to/claims_filtered.json"
        )
        self.synth_input.setToolTip(
            "Stage 5 output (claims_filtered.json), or Stage 3 claims.json "
            "when the audit has not run.")
        layout.addLayout(s_row)
        so_row, self.synth_output, so_btn = _make_file_row(
            "Clusters Output:", "path/to/clusters.json", browse_save=True
        )
        self.synth_output.setToolTip(
            "clusters.json; narratives.json, abstract_draft.md and the other "
            "Stage 6 files are written beside it.")
        layout.addLayout(so_row)
        json_f = "JSON files (*.json);;All files (*)"
        s_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.synth_input, json_f, self.pipeline_config.claims_dir
            )
        )
        so_btn.clicked.connect(
            lambda: self._browse_save_file(
                self.synth_output, json_f, self.pipeline_config.clusters_dir
            )
        )

        sr, self.synth_provider, self.synth_model = _make_llm_backend_row(
            endpoint_getter=self._custom_endpoint
        )
        layout.addLayout(sr)

        synth_params = QHBoxLayout()
        self.synth_auto_clusters = QCheckBox("Auto-select highest silhouette")
        self.synth_auto_clusters.setChecked(True)
        self.synth_auto_clusters.setToolTip(
            "Auto mode uses the highest silhouette score. The sweep evaluates every K "
            "through Sweep maximum K. Small clusters and declining scores are advisory. "
            "Use Calculate Cluster Scores to review both curves and choose K yourself."
        )
        synth_params.addWidget(self.synth_auto_clusters)
        lbl_min_cluster = QLabel("Clusters:")
        synth_params.addWidget(lbl_min_cluster)
        self.synth_min_cluster = QSpinBox()
        self.synth_min_cluster.setRange(2, 10000)
        self.synth_min_cluster.setValue(5)
        self.synth_min_cluster.setEnabled(False)
        lbl_min_cluster.setBuddy(self.synth_min_cluster)
        synth_params.addWidget(self.synth_min_cluster)
        self.synth_auto_clusters.toggled.connect(
            lambda checked: self.synth_min_cluster.setEnabled(not checked))
        lbl_max_cluster = QLabel("Cluster Output Override:")
        synth_params.addWidget(lbl_max_cluster)
        self.synth_max_tokens = QSpinBox()
        self.synth_max_tokens.setRange(0, 2_147_483_647)
        self.synth_max_tokens.setValue(0)
        self.synth_max_tokens.setSpecialValueText("Automatic")
        self.synth_max_tokens.setToolTip(
            "Automatic uses the loaded model context, cluster size, and requested narrative "
            "shape. A positive value is an explicit synthetic ceiling and is recorded. "
            "The 2-3 paragraph narrative and 350-word abstract remain intentional writing "
            "constraints; evidence input expands with available context."
        )
        lbl_max_cluster.setBuddy(self.synth_max_tokens)
        synth_params.addWidget(self.synth_max_tokens)
        synth_params.addStretch()
        layout.addLayout(synth_params)

        review_controls = QHBoxLayout()
        review_controls.addWidget(QLabel("Sweep maximum K:"))
        self.synth_sweep_max = QSpinBox()
        self.synth_sweep_max.setRange(2, 10000)
        self.synth_sweep_max.setValue(15)
        self.synth_sweep_max.setToolTip("Test every K from 2 through this value, up to article count minus one.")
        review_controls.addWidget(self.synth_sweep_max)
        self.synth_calculate_scores_btn = QPushButton("Calculate Cluster Scores")
        self.synth_calculate_scores_btn.clicked.connect(lambda: self._start_synthesize(review_only=True))
        review_controls.addWidget(self.synth_calculate_scores_btn)
        review_controls.addStretch()
        layout.addLayout(review_controls)

        self.synth_btn = QPushButton("Run Synthesis")
        self.synth_btn.clicked.connect(self._start_synthesize)
        layout.addWidget(self.synth_btn)

        self.synth_progress = QProgressBar()
        self.synth_progress.setVisible(False)
        layout.addWidget(self.synth_progress)
        self.synth_status = QLabel("")
        self.synth_status.setWordWrap(True)
        self.synth_status.setTextFormat(Qt.TextFormat.PlainText)
        self.synth_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.synth_status)

        # ── Output area: cluster table on top, one roomy sub-tab per output ──
        splitter = QSplitter(Qt.Orientation.Vertical)

        cluster_group = QGroupBox("Cluster Summary")
        cluster_layout = QVBoxLayout(cluster_group)
        self.cluster_table = QTableWidget(0, 3)
        self.cluster_table.setHorizontalHeaderLabels(["Cluster", "Papers", "Key Terms"])
        self.cluster_table.horizontalHeader().setStretchLastSection(True)
        self.cluster_table.setMinimumHeight(140)
        self.cluster_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        cluster_layout.addWidget(self.cluster_table)
        self.synth_review_scores_btn = QPushButton("Review Silhouette Scores")
        self.synth_review_scores_btn.setToolTip(
            "Open the saved score trajectory for this output in a separate window.")
        self.synth_review_scores_btn.clicked.connect(self._review_silhouette_scores)
        cluster_layout.addWidget(self.synth_review_scores_btn)
        splitter.addWidget(cluster_group)

        self.synth_output_tabs = QTabWidget()

        # Sub-tab: Abstract
        abstract_page = QWidget()
        abstract_layout = QVBoxLayout(abstract_page)
        self.synth_narrative = QTextEdit()
        self.synth_narrative.setAccessibleName("Synthesis narrative")
        self.synth_narrative.setPlaceholderText(
            "Run Stage 6 to generate the synthesized abstract."
        )
        abstract_layout.addWidget(self.synth_narrative, 1)
        abstract_btn_row = QHBoxLayout()
        synth_copy_btn = QPushButton("Copy Abstract")
        synth_copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(self.synth_narrative.toPlainText()))
        abstract_btn_row.addWidget(synth_copy_btn)
        abstract_btn_row.addStretch()
        abstract_layout.addLayout(abstract_btn_row)
        self.synth_output_tabs.addTab(abstract_page, "Abstract")

        # Sub-tab: Why This Hypothesis
        why_page = QWidget()
        why_layout = QVBoxLayout(why_page)
        self.hypothesis_explanation = QTextEdit()
        self.hypothesis_explanation.setReadOnly(True)
        self.hypothesis_explanation.setAccessibleName("Hypothesis explanation")
        self.hypothesis_explanation.setPlaceholderText(
            "Run Stage 6 or load hypothesis_explanation.md to inspect deterministic provenance."
        )
        why_layout.addWidget(self.hypothesis_explanation, 1)
        why_btn_row = QHBoxLayout()
        self.load_hypothesis_explanation_btn = QPushButton("Load Why This Hypothesis")
        self.load_hypothesis_explanation_btn.clicked.connect(
            self._load_hypothesis_explanation
        )
        why_btn_row.addWidget(self.load_hypothesis_explanation_btn)
        why_btn_row.addStretch()
        why_layout.addLayout(why_btn_row)
        self.synth_output_tabs.addTab(why_page, "Why This Hypothesis")

        # Sub-tab: Source Citations (its own view — no longer overwrites the abstract)
        sources_page = QWidget()
        sources_layout = QVBoxLayout(sources_page)
        self.synth_sources_view = QTextEdit()
        self.synth_sources_view.setReadOnly(True)
        self.synth_sources_view.setAccessibleName("Synthesis source citations")
        self.synth_sources_view.setPlaceholderText(
            "Run Stage 6 or load synthesis_sources.md to inspect per-claim citations."
        )
        sources_layout.addWidget(self.synth_sources_view, 1)
        sources_btn_row = QHBoxLayout()
        self.load_synthesis_sources_btn = QPushButton("Load Source Citations")
        self.load_synthesis_sources_btn.clicked.connect(self._load_synthesis_sources)
        sources_btn_row.addWidget(self.load_synthesis_sources_btn)
        sources_btn_row.addStretch()
        sources_layout.addLayout(sources_btn_row)
        self.synth_output_tabs.addTab(sources_page, "Source Citations")

        # Sub-tab: Evidence Network (embedded graph view)
        network_page = QWidget()
        network_layout = QVBoxLayout(network_page)
        from gui.network_view import NetworkGraphWidget
        self.synth_network_widget = NetworkGraphWidget()
        self.synth_network_widget.setMinimumHeight(320)
        network_layout.addWidget(self.synth_network_widget, 1)
        network_btn_row = QHBoxLayout()
        self.load_evidence_network_btn = QPushButton("Load Evidence Network")
        self.load_evidence_network_btn.clicked.connect(self._load_evidence_network)
        network_btn_row.addWidget(self.load_evidence_network_btn)
        network_btn_row.addStretch()
        network_layout.addLayout(network_btn_row)
        self.synth_output_tabs.addTab(network_page, "Evidence Network")

        splitter.addWidget(self.synth_output_tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)

        self._add_help_button(tab, "synthesis")
        return tab

    def _review_silhouette_scores(self, checked=False, path=None):
        from pathlib import Path
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        from stages.s6_synthesize import _output_location
        from gui.silhouette_review import SilhouetteReviewDialog, load_selection

        directory, _ = _output_location(
            self.pipeline_config, self.synth_output.text().strip() or None)
        if path is None:
            available = [p for p in (directory / "k_selection.json", directory / "k_review.json") if p.is_file()]
            # A later fixed-K synthesis has no curve. Keep its preceding review accessible.
            scored = []
            for candidate in available:
                try:
                    if load_selection(candidate).get("silhouette_scores"):
                        scored.append(candidate)
                except (OSError, ValueError, TypeError):
                    pass
            path = max(scored or available, key=lambda p: p.stat().st_mtime) if available else directory / "k_selection.json"
        path = Path(path)
        if not path.is_file():
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Open saved silhouette scores (k_selection.json)",
                str(directory), "JSON files (*.json)")
            if not chosen:
                return
            path = Path(chosen)
        try:
            selection = load_selection(path)
            dialog = SilhouetteReviewDialog(selection, path, self)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            QMessageBox.warning(self, "Cannot open silhouette scores",
                                f"Could not read {path}:\n{exc}")
            return
        dialog.k_chosen.connect(self._apply_reviewed_k)
        dialog.show()

    def _apply_reviewed_k(self, k):
        self.synth_auto_clusters.setChecked(False)
        self.synth_min_cluster.setValue(k)
        self.synth_status.setText(f"K={k} selected for the next synthesis. Click Run Synthesis to use it.")

    # ── Tab 5: Fulltext ──────────────────────────────────

    def _build_fulltext_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Corpus full-text retrieval
        layout.addWidget(SectionHeader("Full-Text Retrieval"))
        ft_row, self.ft_output_path, ft_btn = _make_file_row(
            "JSON Output:", "path/to/fulltext.json", browse_save=True
        )
        # Retrieve Full Texts does not read this field (it downloads to a
        # temporary folder named in the log), so the row is kept for the
        # project path sync but not shown (audit 19-M1).
        ft_row_widget = QWidget()
        ft_row_widget.setLayout(ft_row)
        ft_row_widget.setVisible(False)
        layout.addWidget(ft_row_widget)
        ft_btn.clicked.connect(
            lambda: self._browse_save_file(
                self.ft_output_path,
                "JSON files (*.json);;All files (*)",
                self.pipeline_config.data_dir / "fulltext",
            )
        )

        ft_kpi = QHBoxLayout()
        self.ft_total_card = MetricCard("Total")
        self.ft_retrieved_card = MetricCard("Retrieved")
        self.ft_abstract_card = MetricCard("Abstract Only")
        self.ft_rate_card = MetricCard("Rate")
        ft_kpi.addWidget(self.ft_total_card)
        ft_kpi.addWidget(self.ft_retrieved_card)
        ft_kpi.addWidget(self.ft_abstract_card)
        ft_kpi.addWidget(self.ft_rate_card)
        layout.addLayout(ft_kpi)

        self.ft_results_table = QTableWidget(0, 4)
        self.ft_results_table.setHorizontalHeaderLabels(
            ["Article ID", "Title", "Status", "Source"]
        )
        layout.addWidget(self.ft_results_table)

        ft_btns = QHBoxLayout()
        self.ft_run_btn = QPushButton("Retrieve Full Texts")
        self.ft_run_btn.clicked.connect(self._start_fulltext_retrieval)
        self.ft_cancel_btn = QPushButton("Cancel")
        self.ft_cancel_btn.setEnabled(False)
        self.ft_cancel_btn.clicked.connect(self._cancel_ft)
        ft_btns.addWidget(self.ft_run_btn)
        ft_btns.addWidget(self.ft_cancel_btn)
        ft_btns.addStretch()
        layout.addLayout(ft_btns)

        # Stage 7: pipeline fulltext
        layout.addWidget(SectionHeader("Stage 7: Full-Text Enrichment"))
        s7_row, self.sysys_ft_corpus, s7_btn = _make_file_row(
            "Scored Corpus:", "path/to/relevant.json"
        )
        layout.addLayout(s7_row)
        json_f = "JSON files (*.json);;All files (*)"
        s7_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.sysys_ft_corpus, json_f, self.pipeline_config.claims_dir
            )
        )

        s7_w = QHBoxLayout()
        lbl_workers = QLabel("Workers:")
        s7_w.addWidget(lbl_workers)
        self.sysys_ft_workers = QSpinBox()
        self.sysys_ft_workers.setRange(1, 32)
        self.sysys_ft_workers.setValue(3)
        lbl_workers.setBuddy(self.sysys_ft_workers)
        s7_w.addWidget(self.sysys_ft_workers)
        s7_w.addStretch()
        layout.addLayout(s7_w)

        ft_options = QGroupBox("Memory, Resume && Indexing")
        ft_options_form = QFormLayout(ft_options)

        self.sysys_ft_streaming = QCheckBox(
            "Stream one paper at a time and resume from JSONL"
        )
        self.sysys_ft_streaming.setChecked(True)
        self.sysys_ft_streaming.setToolTip(
            "Preserves completed Stage 7 records across cancellation or restart."
        )
        ft_options_form.addRow("Stage 7:", self.sysys_ft_streaming)

        self.sysys_ft_batch_size = QSpinBox()
        self.sysys_ft_batch_size.setRange(1, 10000)
        self.sysys_ft_batch_size.setValue(50)
        self.sysys_ft_batch_size.setToolTip(
            "Maximum papers held by the legacy buffered path. Streaming mode "
            "always processes one paper at a time."
        )
        ft_options_form.addRow("Buffered Batch:", self.sysys_ft_batch_size)

        self.sysys_ft_paperqa_enabled = QCheckBox(
            "Build PaperQA index during corpus full-text enrichment"
        )
        self.sysys_ft_paperqa_enabled.setChecked(True)
        self.sysys_ft_paperqa_enabled.setToolTip(
            "Applies to Download Full Text / Fetch Articles corpus enrichment. "
            "Disable to avoid loading the optional PaperQA document store."
        )
        ft_options_form.addRow("PaperQA:", self.sysys_ft_paperqa_enabled)

        self.sysys_ft_paperqa_batch_size = QSpinBox()
        self.sysys_ft_paperqa_batch_size.setRange(1, 10000)
        self.sysys_ft_paperqa_batch_size.setValue(100)
        self.sysys_ft_paperqa_batch_size.setToolTip(
            "Checkpoint the PaperQA document store after this many input rows."
        )
        ft_options_form.addRow(
            "Index Checkpoint Batch:", self.sysys_ft_paperqa_batch_size
        )

        for control in (
            self.sysys_ft_workers,
            self.sysys_ft_streaming,
            self.sysys_ft_batch_size,
            self.sysys_ft_paperqa_enabled,
            self.sysys_ft_paperqa_batch_size,
        ):
            signal = (
                control.toggled
                if isinstance(control, QCheckBox)
                else control.valueChanged
            )
            signal.connect(self._sync_fulltext_controls_to_config)
        self.sysys_ft_streaming.toggled.connect(
            lambda enabled: self.sysys_ft_batch_size.setDisabled(enabled)
        )
        self.sysys_ft_paperqa_enabled.toggled.connect(
            self.sysys_ft_paperqa_batch_size.setEnabled
        )
        self.sysys_ft_batch_size.setDisabled(True)
        layout.addWidget(ft_options)

        self.sysys_ft_resume = QCheckBox("Resume completed papers")
        self.sysys_ft_resume.setChecked(True)
        self.sysys_ft_resume.setToolTip(
            "Reuse successful download and metric checkpoints. "
            "Uncheck to retrieve all selected papers again."
        )
        layout.addWidget(self.sysys_ft_resume)
        self.sysys_ft_btn = QPushButton("Run Full-Text Enrichment")
        self.sysys_ft_btn.clicked.connect(self._start_sysys_fulltext)
        layout.addWidget(self.sysys_ft_btn)
        self.sysys_ft_cancel_btn = QPushButton("Cancel Full-Text Enrichment")
        self.sysys_ft_cancel_btn.setEnabled(False)
        self.sysys_ft_cancel_btn.clicked.connect(self._cancel_pipeline_stage)
        layout.addWidget(self.sysys_ft_cancel_btn)

        layout.addStretch()
        self._add_help_button(tab, "fulltext")
        return tab

    # ── Tab 6: Robustness & Verify ────────────────────────────────────

    def _build_robustness_verify_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        self.analysis_workflows = QTabWidget()
        outer.addWidget(self.analysis_workflows)
        checks_page = QWidget()
        self.analysis_workflows.addTab(checks_page, "Robustness & Verification")
        layout = QVBoxLayout(checks_page)

        # Stages 8 and 9 are deterministic statistics. They used to sit next
        # to a provider/model picker that nothing read, which implied an LLM
        # was involved; this notice says what actually happens.
        no_llm_notice = QLabel(
            "Stage 9 produces the final source-linked review with internal numerical checks. "
            "The Meta-Analysis page offers pooling and optional "
            "model enhancement in bounded sections."
        )
        no_llm_notice.setWordWrap(True)
        no_llm_notice.setAccessibleName("Analysis workflow description")
        layout.addWidget(no_llm_notice)

        # Stage 8: Robustness
        layout.addWidget(SectionHeader("Stage 8: Robustness Checks"))
        r_row, self.robust_input, r_btn = _make_file_row(
            "Audited Corpus:", "path/to/claims_filtered.json"
        )
        layout.addLayout(r_row)
        ro_row, self.robust_output, ro_btn = _make_file_row(
            "Results JSON:", "path/to/robustness_results.json", browse_save=True
        )
        layout.addLayout(ro_row)
        json_f = "JSON files (*.json);;All files (*)"
        val_dir = self.pipeline_config.data_dir / "validation"
        r_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.robust_input, json_f, self.pipeline_config.claims_dir
            )
        )
        ro_btn.clicked.connect(
            lambda: self._browse_save_file(
                self.robust_output, json_f, val_dir, self.pipeline_config.claims_dir
            )
        )

        robust_btn_row = QHBoxLayout()
        self.robust_btn = QPushButton("Run Robustness Checks")
        self.robust_btn.clicked.connect(self._start_robustness)
        self.robust_cancel_btn = QPushButton("Cancel")
        self.robust_cancel_btn.setEnabled(False)
        self.robust_cancel_btn.clicked.connect(self._cancel_pipeline_stage)
        robust_btn_row.addWidget(self.robust_btn)
        robust_btn_row.addWidget(self.robust_cancel_btn)
        robust_btn_row.addStretch()
        layout.addLayout(robust_btn_row)

        # Stage 9: Verify
        layout.addWidget(SectionHeader("Stage 9: Final Review"))
        v_row, self.verify_claims, v_btn = _make_file_row(
            "Saved Claims Corpus:", "path/to/claims.json"
        )
        layout.addLayout(v_row)
        json_f = "JSON files (*.json);;All files (*)"
        v_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.verify_claims, json_f, self.pipeline_config.claims_dir
            )
        )

        self.verify_btn = QPushButton("Generate Final Review")
        self.verify_btn.clicked.connect(self._start_verify)
        layout.addWidget(self.verify_btn)

        self.robust_verify_progress = QProgressBar()
        self.robust_verify_progress.setVisible(False)
        layout.addWidget(self.robust_verify_progress)
        self.robust_verify_status = QLabel("")
        layout.addWidget(self.robust_verify_status)

        self.robust_results = QTextEdit()
        self.robust_results.setReadOnly(True)
        self.robust_results.setAccessibleName("Robustness verification results")
        checks_results = QTabWidget()
        checks_results.addTab(self.robust_results, "Results")
        layout.addWidget(checks_results, 1)

        # Inline verification report viewer
        self.verify_report_view = QTextBrowser()
        self.verify_report_view.setMinimumHeight(260)
        self.verify_report_view.setAccessibleName("Verification report")
        self.verify_report_view.setPlaceholderText(
            "Generate the final review to see source-linked findings, statistical synthesis and limitations.")
        self.verify_report_view.setOpenExternalLinks(True)
        checks_results.addTab(self.verify_report_view, "Final Review Report")
        checks_results.setCurrentWidget(self.verify_report_view)

        meta_page = QWidget()
        self.analysis_workflows.addTab(meta_page, "Meta-Analysis")
        layout = QVBoxLayout(meta_page)
        # ── Meta-Analytic Synthesis section ──
        layout.addWidget(SectionHeader("Meta-Analytic Synthesis"))

        meta_ctrl = QHBoxLayout()
        self.meta_analysis_btn = QPushButton("Run Meta-Analysis")
        self.meta_analysis_btn.clicked.connect(self._start_meta_analysis)
        meta_ctrl.addWidget(self.meta_analysis_btn)
        self.meta_review_btn = QPushButton("Review Effects && Groups…")
        self.meta_review_btn.setToolTip(
            "Choose which effects to pool, their groups, and the direction of converted "
            "F / χ² / p-value effects. Saving re-runs the meta-analysis.")
        self.meta_review_btn.clicked.connect(self._review_effects_and_rerun)
        meta_ctrl.addWidget(self.meta_review_btn)
        self.meta_analysis_status = QLabel("Not yet run.")
        meta_ctrl.addWidget(self.meta_analysis_status, 1)
        layout.addLayout(meta_ctrl)

        enhancement_row = QHBoxLayout()
        self.meta_use_model = QCheckBox("Re-check statistics and group effects with model")
        self.meta_use_model.setChecked(True)
        self.meta_use_model.setToolTip(
            "Uses the model selected in Preferences. Asks about statistics Stage 2c left "
            "without a verdict, discovers cross-paper candidates, checks title and research-"
            "question relevance, then audits construct, measure, population, independence, "
            "comparison, design, direction, and scientific plausibility. Only groups that pass "
            "every gate are pooled.")
        enhancement_row.addWidget(self.meta_use_model)
        self.meta_enhance = QCheckBox("Enhance report with model (segmented)")
        self.meta_enhance.setToolTip(
            "Uses the model selected in Preferences. Writes context-aware evidence sections; "
            "intentional concision limits and every split/repair action are recorded. The "
            "complete local analysis is saved first.")
        enhancement_row.addWidget(self.meta_enhance)
        enhancement_row.addWidget(QLabel("Context override:"))
        self.meta_context_tokens = QSpinBox()
        self.meta_context_tokens.setRange(0, 2_147_483_647)
        self.meta_context_tokens.setValue(0)
        self.meta_context_tokens.setSingleStep(4096)
        self.meta_context_tokens.setSpecialValueText("Auto-detect")
        self.meta_context_tokens.setToolTip(
            "Auto-detect reads the currently loaded context from LM Studio native REST; for "
            "other providers use the Preferences context setting or enter an override here. "
            "SystS divides it between prompt, complete response accounting, and a safety reserve. "
            "A larger window lets candidate discovery compare more effects at once. "
            "Corpus-wide similarity seeds, overlapping boundaries, and alternate characteristic "
            "orderings cover split requests; every candidate still receives the same semantic "
            "audit. Incomplete replies are evicted and split/repaired, with each action and any "
            "intentional synthetic limit recorded in the audit artifacts.")
        enhancement_row.addWidget(self.meta_context_tokens)
        self.meta_cancel_btn = QPushButton("Stop model work")
        self.meta_cancel_btn.setEnabled(False)
        self.meta_cancel_btn.clicked.connect(
            lambda: self._meta_worker.cancel() if self._meta_worker else None)
        enhancement_row.addWidget(self.meta_cancel_btn)
        layout.addLayout(enhancement_row)
        self.meta_analysis_status.setWordWrap(True)

        self.meta_group_combo = QComboBox()
        self.meta_group_combo.setAccessibleName("Meta-analysis outcome group")
        self.meta_group_combo.setPlaceholderText("Compatible outcome groups appear after analysis")
        self.meta_group_combo.setEnabled(False)
        self.meta_group_combo.currentIndexChanged.connect(self._show_meta_analysis_group)
        layout.addWidget(self.meta_group_combo)

        # Plot tabs: Forest plot / Funnel plot
        self.meta_plot_tabs = QTabWidget()
        self.meta_plot_tabs.setMinimumHeight(350)

        # Forest plot placeholder
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        forest_fig = Figure(figsize=(10, 5), dpi=100, facecolor="white")
        forest_ax = forest_fig.add_subplot(111)
        forest_ax.text(0.5, 0.5, "Click 'Run Meta-Analysis' to generate forest plot",
                       ha="center", va="center", fontsize=11, color="#808080")
        forest_ax.axis("off")
        self.forest_canvas = FigureCanvasQTAgg(forest_fig)
        self.meta_plot_tabs.addTab(self.forest_canvas, "Forest Plot")

        # Funnel plot placeholder
        funnel_fig = Figure(figsize=(7, 6), dpi=100, facecolor="white")
        funnel_ax = funnel_fig.add_subplot(111)
        funnel_ax.text(0.5, 0.5, "Click 'Run Meta-Analysis' to generate funnel plot",
                       ha="center", va="center", fontsize=11, color="#808080")
        funnel_ax.axis("off")
        self.funnel_canvas = FigureCanvasQTAgg(funnel_fig)
        self.meta_plot_tabs.addTab(self.funnel_canvas, "Funnel Plot")

        layout.addWidget(self.meta_plot_tabs)

        # Summary statistics display
        self.meta_summary = QTextBrowser()
        self.meta_summary.setReadOnly(True)
        self.meta_summary.setOpenExternalLinks(True)
        self.meta_summary.setAccessibleName("Meta-analysis summary")
        self.meta_summary.setMinimumHeight(250)
        self.meta_summary.setPlaceholderText(
            "Meta-analysis summary statistics will appear here."
        )
        self.meta_plot_tabs.insertTab(0, self.meta_summary, "Evidence Report")
        self.meta_enhanced_report = QTextBrowser()
        self.meta_enhanced_report.setReadOnly(True)
        self.meta_enhanced_report.setOpenExternalLinks(True)
        self.meta_enhanced_report.setAccessibleName("Model-enhanced evidence report")
        self.meta_enhanced_report.setPlaceholderText(
            "Enable model enhancement before running to generate a segmented draft.")
        self.meta_plot_tabs.insertTab(1, self.meta_enhanced_report, "Model Enhancement")
        self.meta_model_audit = QTextBrowser()
        self.meta_model_audit.setReadOnly(True)
        self.meta_model_audit.setOpenExternalLinks(True)
        self.meta_model_audit.setAccessibleName("Meta-analysis model action audit")
        self.meta_model_audit.setPlaceholderText(
            "Model grouping budgets, split/repair actions, and semantic gate decisions "
            "appear here after a model-assisted meta-analysis."
        )
        self.meta_plot_tabs.insertTab(2, self.meta_model_audit, "Model Action Audit")
        self.meta_plot_tabs.setCurrentWidget(self.meta_summary)

        self._add_help_button(tab, "robustness_verify")
        return tab

    # ── Tab 7: Text Scaling ───────────────────────────────────────────

    def _build_scaling_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Upper: Wordfish
        upper = QWidget()
        u_layout = QVBoxLayout(upper)
        u_layout.addWidget(SectionHeader("Wordfish Scaling"))

        wf_params = QHBoxLayout()
        lbl_min_words = QLabel("Min Doc Words:")
        wf_params.addWidget(lbl_min_words)
        self.wf_min_words = QSpinBox()
        self.wf_min_words.setRange(10, 10000)
        self.wf_min_words.setValue(50)
        lbl_min_words.setBuddy(self.wf_min_words)
        wf_params.addWidget(self.wf_min_words)
        lbl_max_features = QLabel("Max Features:")
        wf_params.addWidget(lbl_max_features)
        self.wf_max_features = QSpinBox()
        self.wf_max_features.setRange(100, 100000)
        self.wf_max_features.setValue(5000)
        lbl_max_features.setBuddy(self.wf_max_features)
        wf_params.addWidget(self.wf_max_features)
        wf_params.addStretch()
        u_layout.addLayout(wf_params)

        wf_actions = QHBoxLayout()
        self.wf_run_btn = QPushButton("Run Text Scaling")
        self.wf_run_btn.clicked.connect(self._start_text_scaling)
        wf_actions.addWidget(self.wf_run_btn)
        self.wf_cancel_btn = QPushButton("Cancel")
        self.wf_cancel_btn.setEnabled(False)
        self.wf_cancel_btn.clicked.connect(self._cancel_scaling)
        wf_actions.addWidget(self.wf_cancel_btn)
        wf_actions.addStretch()
        u_layout.addLayout(wf_actions)

        wf_kpi = QHBoxLayout()
        self.wf_docs_card = MetricCard("Documents")
        self.wf_features_card = MetricCard("Features")
        self.wf_convergence_card = MetricCard("Convergence")
        wf_kpi.addWidget(self.wf_docs_card)
        wf_kpi.addWidget(self.wf_features_card)
        wf_kpi.addWidget(self.wf_convergence_card)
        u_layout.addLayout(wf_kpi)

        self.wf_table = QTableWidget(0, 3)
        self.wf_table.setHorizontalHeaderLabels(["Article", "Position", "Direction"])
        self.wf_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.wf_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.wf_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        u_layout.addWidget(self.wf_table)

        self.wf_terms_detail = QTextEdit()
        self.wf_terms_detail.setReadOnly(True)
        self.wf_terms_detail.setAccessibleName("Wordfish discriminating terms")
        self.wf_terms_detail.setMaximumHeight(72)
        self.wf_terms_detail.setPlaceholderText(
            "The vocabulary most associated with each pole will appear here."
        )
        u_layout.addWidget(self.wf_terms_detail)

        splitter.addWidget(upper)

        # Lower: Stylometry
        lower = QWidget()
        l_layout = QVBoxLayout(lower)
        l_layout.addWidget(SectionHeader("Stylometric Profiles"))

        sty_kpi = QHBoxLayout()
        self.sty_corpus_card = MetricCard("Corpus")
        self.sty_avg_hedge_card = MetricCard("Avg Hedge")
        self.sty_avg_certainty_card = MetricCard("Avg Certainty")
        sty_kpi.addWidget(self.sty_corpus_card)
        sty_kpi.addWidget(self.sty_avg_hedge_card)
        sty_kpi.addWidget(self.sty_avg_certainty_card)
        l_layout.addLayout(sty_kpi)

        self.sty_table = QTableWidget(0, 5)
        self.sty_table.setHorizontalHeaderLabels(
            ["Article", "Hedge", "Certainty", "FK Grade", "TTR"]
        )
        self.sty_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 5):
            self.sty_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.sty_table.cellClicked.connect(self._on_sty_row_clicked)
        l_layout.addWidget(self.sty_table)

        self.sty_detail = QTextEdit()
        self.sty_detail.setReadOnly(True)
        self.sty_detail.setAccessibleName("Stylometric details")
        self.sty_detail.setMaximumHeight(120)
        l_layout.addWidget(self.sty_detail)

        splitter.addWidget(lower)
        layout.addWidget(splitter)
        self._add_help_button(tab, "text_scaling")
        return tab
