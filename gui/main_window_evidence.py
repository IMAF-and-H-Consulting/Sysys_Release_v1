"""Evidence Map and Results Dashboard tabs: GRADE, alignment, evidence table.

Split from gui/main_window.py; mixed into MainWindow.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.theme import COLORS
from gui.widgets import AlignmentBar, GradeBadge, MetricCard, SectionHeader

logger = logging.getLogger("gui.main_window")

_NO_DATA = "—"
_NOT_ASSESSED = "Not assessed"
_SOF_CONCERN_COLOURS = {
    "no_concern": "#c8e6c9",    # light green
    "serious": "#fff9c4",       # light yellow
    "very_serious": "#ffcdd2",  # light red
}
_SOF_NOT_ASSESSED_COLOUR = "#e0e0e0"  # grey: no judgement, not "fine"


def _sof_domain_colour(cell) -> str:
    """Background for a GRADE domain cell; "Not assessed" is grey, not green."""
    if not isinstance(cell, dict):
        return "#ffffff"
    if cell.get("label") == _NOT_ASSESSED:
        return _SOF_NOT_ASSESSED_COLOUR
    return _SOF_CONCERN_COLOURS.get(cell.get("concern", ""), "#ffffff")


class EvidenceMixin:
    """Evidence Map and Results Dashboard tabs: GRADE, alignment, evidence table."""

    # ── Tab 8: Evidence Map ───────────────────────────────────────────

    def _build_evidence_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Evidence table
        upper = QWidget()
        u_layout = QVBoxLayout(upper)
        u_layout.addWidget(SectionHeader("Evidence Table"))

        filter_row = QHBoxLayout()
        lbl_dir = QLabel("Direction:")
        filter_row.addWidget(lbl_dir)
        self.ev_direction_filter = QComboBox()
        self.ev_direction_filter.addItems(
            ["All", "supports", "contradicts", "neutral", "tangential"]
        )
        lbl_dir.setBuddy(self.ev_direction_filter)
        filter_row.addWidget(self.ev_direction_filter)
        lbl_comp = QLabel("Component:")
        filter_row.addWidget(lbl_comp)
        self.ev_component_filter = QComboBox()
        self.ev_component_filter.addItem("All")
        lbl_comp.setBuddy(self.ev_component_filter)
        filter_row.addWidget(self.ev_component_filter)
        filter_row.addStretch()
        self.ev_refresh_btn = QPushButton("Load Evidence Table")
        self.ev_refresh_btn.setMinimumHeight(30)
        self.ev_refresh_btn.clicked.connect(self._refresh_evidence_table)
        filter_row.addWidget(self.ev_refresh_btn)
        u_layout.addLayout(filter_row)

        self.ev_status_label = QLabel("No evidence loaded.")
        self.ev_status_label.setWordWrap(True)
        u_layout.addWidget(self.ev_status_label)

        self.evidence_table = QTableWidget(0, 5)
        self.evidence_table.setHorizontalHeaderLabels(
            ["Article", "Component", "Score", "Direction", "Excerpt"]
        )
        evidence_header = self.evidence_table.horizontalHeader()
        evidence_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        evidence_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        evidence_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        evidence_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        evidence_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.evidence_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.evidence_table.cellClicked.connect(self._on_evidence_cell_clicked)
        u_layout.addWidget(self.evidence_table)

        self.ev_direction_filter.currentTextChanged.connect(
            self._apply_evidence_filters
        )
        self.ev_component_filter.currentTextChanged.connect(
            self._apply_evidence_filters
        )

        self.evidence_detail = QTextEdit()
        self.evidence_detail.setReadOnly(True)
        self.evidence_detail.setAccessibleName("Full evidence details")
        self.evidence_detail.setMaximumHeight(120)
        u_layout.addWidget(self.evidence_detail)

        layout.addWidget(upper)
        self._add_help_button(tab, "evidence_map")
        return tab

    # ── Tab 9: Results Dashboard ──────────────────────────────────────

    def _build_alignment_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(SectionHeader("Results Dashboard"))

        # KPI cards
        kpi_row = QHBoxLayout()
        self.dash_total_card = MetricCard("Total Articles")
        self.dash_analyzed_card = MetricCard("Analyzed")
        self.dash_supporting_card = MetricCard("Supporting")
        self.dash_contradicting_card = MetricCard("Contradicting")
        self.dash_overall_card = MetricCard("Overall Score")
        kpi_row.addWidget(self.dash_total_card)
        kpi_row.addWidget(self.dash_analyzed_card)
        kpi_row.addWidget(self.dash_supporting_card)
        kpi_row.addWidget(self.dash_contradicting_card)
        kpi_row.addWidget(self.dash_overall_card)
        layout.addLayout(kpi_row)

        # GRADE assessment button
        grade_row = QHBoxLayout()
        self.grade_run_btn = QPushButton("Run GRADE Assessment")
        self.grade_run_btn.clicked.connect(self._run_grade_assessment)
        grade_row.addWidget(self.grade_run_btn)
        self.grade_status_label = QLabel("")
        self.grade_status_label.setStyleSheet(f"font-size: 12pt; color: {COLORS['text_secondary']};")
        grade_row.addWidget(self.grade_status_label)
        grade_row.addStretch()
        layout.addLayout(grade_row)

        # Per-component alignment bars (with GRADE badge placeholders)
        layout.addWidget(SectionHeader("Per-Component Alignment"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.alignment_bars_widget = QWidget()
        self.alignment_bars_layout = QVBoxLayout(self.alignment_bars_widget)
        scroll.setWidget(self.alignment_bars_widget)
        layout.addWidget(scroll)

        # Collapsible GRADE Summary of Findings table
        self.grade_sof_group = QGroupBox("GRADE Summary of Findings")
        self.grade_sof_group.setCheckable(True)
        self.grade_sof_group.setChecked(False)
        sof_layout = QVBoxLayout(self.grade_sof_group)

        self.grade_sof_table = QTableWidget()
        self.grade_sof_table.setColumnCount(8)
        self.grade_sof_table.setHorizontalHeaderLabels([
            "Component", "N", "Risk of Bias", "Inconsistency",
            "Indirectness", "Imprecision", "Pub. Bias", "Certainty",
        ])
        self.grade_sof_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.grade_sof_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.grade_sof_table.currentCellChanged.connect(self._on_grade_row_selected)
        sof_layout.addWidget(self.grade_sof_table)
        self.grade_sof_group.setVisible(False)
        layout.addWidget(self.grade_sof_group)

        # Detail panel
        self.alignment_detail = QTextEdit()
        self.alignment_detail.setReadOnly(True)
        self.alignment_detail.setAccessibleName("Detailed alignment information")
        self.alignment_detail.setMaximumHeight(150)
        layout.addWidget(self.alignment_detail)

        # Internal state
        self._grade_badges: list[GradeBadge] = []
        self._grade_results: list = []  # list[ComponentGrade]
        self._grade_sof_data: list[dict] = []

        self._add_help_button(tab, "results_dashboard")
        return tab

    # ── GRADE assessment (Results Dashboard) ────────────────────────

    def _run_grade_assessment(self):
        """Run GRADE certainty assessment and update the dashboard."""
        from process.grade import generate_sof_table, grade_all_components

        self.grade_status_label.setText("Running GRADE assessment...")
        QApplication.processEvents()

        try:
            self._grade_results = grade_all_components(self.pipeline_config)

            if not self._grade_results:
                self.grade_status_label.setText(
                    "No components found. Run the pipeline first."
                )
                return

            # Build SoF table data
            self._grade_sof_data = generate_sof_table(self._grade_results)
            self._populate_grade_sof_table(self._grade_sof_data)
            self.grade_sof_group.setVisible(True)
            self.grade_sof_group.setChecked(True)

            self._update_alignment_dashboard()

            self.grade_status_label.setText(
                f"GRADE complete — {len(self._grade_results)} components assessed."
            )
            self._log("GRADE evidence assessment completed.")
        except Exception as exc:
            self.grade_status_label.setText(f"Error: {exc}")
            self._log(f"GRADE assessment failed: {exc}", level="ERROR")
            logger.exception("GRADE assessment failed")

    def _attach_grade_badges(self):
        """Add or update GradeBadge widgets next to each AlignmentBar row."""
        # Build a lookup from component_id -> certainty
        certainty_map: dict[str, str] = {}
        for g in self._grade_results:
            certainty_map[g.component_id] = g.overall_certainty.value

        # Walk the alignment_bars_layout and attach badges
        for i in range(self.alignment_bars_layout.count()):
            item = self.alignment_bars_layout.itemAt(i)
            if item is None:
                continue
            widget = item.widget()
            if widget is None:
                # Might be a layout
                sub_layout = item.layout()
                if sub_layout is None:
                    continue
                # Check if this row already has a badge (3rd widget)
                bar_widget = None
                for j in range(sub_layout.count()):
                    sub_item = sub_layout.itemAt(j)
                    w = sub_item.widget() if sub_item is not None else None
                    if isinstance(w, AlignmentBar):
                        bar_widget = w
                        break
                if bar_widget is not None:
                    # Try to find the component_id from the bar label
                    comp_id = getattr(bar_widget, "_component_id", None)
                    if comp_id and comp_id in certainty_map:
                        # Check for existing badge in this layout
                        existing_badge = None
                        for j in range(sub_layout.count()):
                            sub_item = sub_layout.itemAt(j)
                            w = sub_item.widget() if sub_item is not None else None
                            if isinstance(w, GradeBadge):
                                existing_badge = w
                                break
                        if existing_badge:
                            existing_badge.set_certainty(certainty_map[comp_id])
                        else:
                            badge = GradeBadge(certainty_map[comp_id])
                            sub_layout.addWidget(badge)
                            self._grade_badges.append(badge)
                continue

            # If the widget itself is an AlignmentBar in a direct layout, wrap it
            if isinstance(widget, AlignmentBar):
                comp_id = getattr(widget, "_component_id", None)
                if comp_id and comp_id in certainty_map:
                    # Replace the widget with an HBoxLayout containing bar + badge
                    self.alignment_bars_layout.removeWidget(widget)
                    row_layout = QHBoxLayout()
                    row_layout.addWidget(widget, stretch=1)
                    badge = GradeBadge(certainty_map[comp_id])
                    row_layout.addWidget(badge)
                    self._grade_badges.append(badge)
                    self.alignment_bars_layout.insertLayout(i, row_layout)

    def _populate_grade_sof_table(self, sof_data: list[dict]):
        """Fill the GRADE Summary of Findings QTableWidget."""
        domain_keys = [
            "risk_of_bias", "inconsistency", "indirectness",
            "imprecision", "publication_bias",
        ]

        table = self.grade_sof_table
        table.setRowCount(len(sof_data))

        for row, entry in enumerate(sof_data):
            # Component name
            comp_item = QTableWidgetItem(entry.get("component", ""))
            comp_item.setFlags(comp_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, comp_item)

            # N studies
            n_item = QTableWidgetItem(str(entry.get("n_studies", 0)))
            n_item.setFlags(n_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            n_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row, 1, n_item)

            # Domain columns
            for col_idx, dk in enumerate(domain_keys, start=2):
                d = entry.get(dk, {})
                label = d.get("label", "N/A") if isinstance(d, dict) else str(d)

                cell = QTableWidgetItem(label)
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setBackground(QColor(_sof_domain_colour(d)))
                table.setItem(row, col_idx, cell)

            # Overall certainty
            cert = entry.get("overall_certainty", "")
            cert_item = QTableWidgetItem(cert)
            cert_item.setFlags(cert_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            cert_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            cert_bg = GradeBadge.COLORS.get(cert, "#ffffff")
            cert_item.setBackground(QColor(cert_bg))
            cert_fg = GradeBadge.TEXT_COLORS.get(cert, "#000000")
            cert_item.setForeground(QColor(cert_fg))
            table.setItem(row, 7, cert_item)

    def _on_grade_row_selected(self, row: int, _col: int, _prev_row: int, _prev_col: int):
        """Show GRADE rationale in the detail panel when a SoF row is clicked."""
        if row < 0 or row >= len(self._grade_results):
            return
        grade = self._grade_results[row]
        lines = [f"<b>{grade.component_label}</b> — {grade.overall_certainty.value} certainty"]
        lines.append(f"<br>Starting certainty: {grade.starting_certainty} | Studies: {grade.n_studies}")
        lines.append("<br>")
        for d in grade.domains:
            icon = {
                "no_concern": "\u2705",
                "serious": "\u26a0\ufe0f",
                "very_serious": "\u274c",
            }.get(d.concern.value, "")
            lines.append(
                f"<b>{d.domain.replace('_', ' ').title()}</b>: "
                f"{icon} {d.concern.value.replace('_', ' ').title()} "
                f"(downgrade {d.downgrade_levels})<br>"
                f"<i>{d.rationale}</i><br>"
            )
        lines.append(f"<br><b>Summary:</b> {grade.summary}")
        self.alignment_detail.setHtml("".join(lines))

    # ── Search/full-text populate / refresh / update display methods ──

    def _populate_wf_table(self, model):
        positions = model.get("positions", {}) if isinstance(model, dict) else {}
        self.wf_table.setSortingEnabled(False)
        self.wf_table.setRowCount(len(positions))
        for row, (article_id, position) in enumerate(
            sorted(positions.items(), key=lambda item: item[1])
        ):
            article = self.articles.get(article_id)
            if isinstance(article, dict):
                title = article.get("title", article_id)
            else:
                title = getattr(article, "title", article_id)
            title_item = QTableWidgetItem(str(title or article_id))
            title_item.setData(Qt.ItemDataRole.UserRole, article_id)
            self.wf_table.setItem(row, 0, title_item)
            self.wf_table.setItem(row, 1, QTableWidgetItem(f"{float(position):+.4f}"))
            if position < -0.1:
                direction = "Negative pole"
            elif position > 0.1:
                direction = "Positive pole"
            else:
                direction = "Center"
            self.wf_table.setItem(row, 2, QTableWidgetItem(direction))
        self.wf_table.setSortingEnabled(True)
        self.wf_docs_card.set_value(str(model.get("n_documents", len(positions))))
        self.wf_features_card.set_value(str(model.get("n_features", 0)))
        convergence = "Yes" if model.get("converged") else "No"
        self.wf_convergence_card.set_value(
            f"{convergence} ({model.get('iterations', 0)})"
        )
        terms = model.get("discriminating_terms", [])
        negative_terms = sorted(
            (term for term in terms if term.get("beta", 0.0) < 0.0),
            key=lambda term: term.get("beta", 0.0),
        )[:10]
        positive_terms = sorted(
            (term for term in terms if term.get("beta", 0.0) > 0.0),
            key=lambda term: term.get("beta", 0.0),
            reverse=True,
        )[:10]
        negative_text = ", ".join(
            f"{term.get('term')} ({term.get('beta', 0.0):+.3f})"
            for term in negative_terms
        ) or "None"
        positive_text = ", ".join(
            f"{term.get('term')} ({term.get('beta', 0.0):+.3f})"
            for term in positive_terms
        ) or "None"
        self.wf_terms_detail.setPlainText(
            f"Negative pole: {negative_text}\nPositive pole: {positive_text}"
        )

    def _populate_sty_table(self, profiles: dict):
        ordered_profiles = sorted(
            profiles.items(),
            key=lambda item: str(item[1].get("title", item[0])).lower(),
        )
        self.sty_table.setSortingEnabled(False)
        self.sty_table.setRowCount(len(ordered_profiles))
        for row, (article_id, profile) in enumerate(ordered_profiles):
            title_item = QTableWidgetItem(str(profile.get("title") or article_id))
            title_item.setData(Qt.ItemDataRole.UserRole, article_id)
            self.sty_table.setItem(row, 0, title_item)
            self.sty_table.setItem(
                row, 1, QTableWidgetItem(f"{profile.get('hedge_density', 0.0):.2f}")
            )
            self.sty_table.setItem(
                row, 2, QTableWidgetItem(f"{profile.get('certainty_density', 0.0):.2f}")
            )
            self.sty_table.setItem(
                row, 3, QTableWidgetItem(f"{profile.get('flesch_kincaid_grade', 0.0):.1f}")
            )
            self.sty_table.setItem(
                row, 4, QTableWidgetItem(f"{profile.get('type_token_ratio', 0.0):.3f}")
            )
        self.sty_table.setSortingEnabled(True)

    def _update_sty_kpis(self, stats: dict):
        self.sty_corpus_card.set_value(str(stats.get("n_documents", 0)))
        self.sty_avg_hedge_card.set_value(
            f"{stats.get('average_hedge_density', 0.0):.2f}"
        )
        self.sty_avg_certainty_card.set_value(
            f"{stats.get('average_certainty_density', 0.0):.2f}"
        )

    def _on_sty_row_clicked(self, row: int, col: int):
        title_item = self.sty_table.item(row, 0)
        if title_item is None:
            return
        article_id = title_item.data(Qt.ItemDataRole.UserRole)
        profile = self._stylometry_profiles.get(str(article_id), {})
        if not profile:
            return
        self.sty_detail.setPlainText(
            f"{profile.get('title', article_id)}\n"
            f"Words: {profile.get('word_count', 0):,} | "
            f"Sentences: {profile.get('sentence_count', 0):,} | "
            f"Average sentence length: {profile.get('average_sentence_length', 0.0):.1f}\n"
            f"Hedges per 100 words: {profile.get('hedge_density', 0.0):.2f} | "
            f"Certainty terms per 100 words: {profile.get('certainty_density', 0.0):.2f} | "
            f"Passive constructions per 100 sentences: "
            f"{profile.get('passive_voice_density', 0.0):.2f}"
        )

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child = item.layout()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            elif child is not None:
                self._clear_layout(child)

    def _populate_alignment_bars(self, results, has_scores: bool = True) -> None:
        """Fill the per-component AlignmentBar rows from alignment results.

        ``has_scores`` False (GRADE-only components) shows "no alignment data"
        instead of a +0.00 score (audit 19-L4).
        """
        self._clear_layout(self.alignment_bars_layout)
        self._grade_badges = []
        if not results:
            empty = QLabel(
                "No per-component alignment yet. Run Keyword Analysis first."
            )
            empty.setWordWrap(True)
            self.alignment_bars_layout.addWidget(empty)
            return
        for result in results:
            row = QHBoxLayout()
            name = QLabel(result.component_label or result.component_id)
            name.setMinimumWidth(180)
            name.setWordWrap(True)
            bar = AlignmentBar()
            bar._component_id = result.component_id
            score = float(result.support_score or 0.0) if has_scores else 0.0
            bar.set_score(score, f"{score:+.2f}" if has_scores else "no alignment data")
            row.addWidget(name)
            row.addWidget(bar, stretch=1)
            self.alignment_bars_layout.addLayout(row)

    def _update_alignment_dashboard(self):
        session = getattr(self, "session", None)
        results = list(getattr(session, "alignment_results", None) or [])
        has_scores = bool(results)
        if not results and getattr(self, "_grade_results", None):
            from core.models import AlignmentResult
            results = [
                AlignmentResult(
                    component_id=grade.component_id,
                    component_label=grade.component_label,
                )
                for grade in self._grade_results
            ]
        self._populate_alignment_bars(results, has_scores)
        articles = getattr(session, "articles", None) or {}
        analyzed = int(getattr(session, "total_articles_analyzed", 0) or len(articles))
        supporting = sum(int(getattr(item, "supporting_count", 0) or 0) for item in results)
        contradicting = sum(int(getattr(item, "contradicting_count", 0) or 0) for item in results)
        overall = (
            sum(float(getattr(item, "support_score", 0.0) or 0.0) for item in results) / len(results)
            if results else 0.0
        )
        fetched = int(getattr(session, "total_articles_fetched", 0) or 0)
        self.dash_total_card.set_value(str(len(articles) or fetched))
        self.dash_analyzed_card.set_value(str(analyzed))
        self.dash_supporting_card.set_value(str(supporting) if has_scores else _NO_DATA)
        self.dash_contradicting_card.set_value(str(contradicting) if has_scores else _NO_DATA)
        self.dash_overall_card.set_value(f"{overall:+.2f}" if has_scores else _NO_DATA)
        if getattr(self, "_grade_results", None):
            self._attach_grade_badges()
        self._log("Updating alignment dashboard...")

    def _refresh_evidence_table(self):
        from process.evidence_table import load_evidence_rows

        self.ev_refresh_btn.setEnabled(False)
        self.ev_status_label.setText("Loading evidence...")
        QApplication.processEvents()
        try:
            rows, source = load_evidence_rows(
                self.pipeline_config,
                session=self.session,
                articles=self.articles,
                components=self.hypothesis_components,
            )
            self._evidence_table_rows = rows
            self._evidence_table_source = source

            current_component = self.ev_component_filter.currentText()
            components = sorted({
                str(row.get("component") or "Primary question")
                for row in rows
            })
            self.ev_component_filter.blockSignals(True)
            self.ev_component_filter.clear()
            self.ev_component_filter.addItem("All")
            self.ev_component_filter.addItems(components)
            if current_component in components:
                self.ev_component_filter.setCurrentText(current_component)
            self.ev_component_filter.blockSignals(False)

            self._apply_evidence_filters()
            if rows:
                self.ev_refresh_btn.setText("Refresh Evidence Table")
                self._log(f"Evidence table loaded: {len(rows)} rows from {source}")
            else:
                self.ev_status_label.setText(
                    "No evidence found. Run Keyword Analysis or Stage 3 Extract Claims first."
                )
                self._log("No evidence available for the Evidence Table.", "WARNING")
        except Exception as exc:
            self._evidence_table_rows = []
            self._evidence_table_source = ""
            self.evidence_table.setRowCount(0)
            self.ev_status_label.setText(f"Could not load evidence: {exc}")
            self._log(f"Evidence table load failed: {exc}", "ERROR")
        finally:
            self.ev_refresh_btn.setEnabled(True)

    def _apply_evidence_filters(self, *_):
        direction_filter = self.ev_direction_filter.currentText().lower()
        component_filter = self.ev_component_filter.currentText()
        filtered_rows = [
            row for row in self._evidence_table_rows
            if (direction_filter == "all" or row.get("direction") == direction_filter)
            and (component_filter == "All" or row.get("component") == component_filter)
        ]

        self.evidence_table.setSortingEnabled(False)
        self.evidence_table.setRowCount(len(filtered_rows))
        for table_row, evidence in enumerate(filtered_rows):
            title_item = QTableWidgetItem(str(evidence.get("title") or "Untitled"))
            title_item.setData(Qt.ItemDataRole.UserRole, evidence)
            self.evidence_table.setItem(table_row, 0, title_item)
            self.evidence_table.setItem(
                table_row,
                1,
                QTableWidgetItem(str(evidence.get("component") or "Primary question")),
            )
            score_item = QTableWidgetItem()
            score_item.setData(
                Qt.ItemDataRole.DisplayRole,
                round(float(evidence.get("score") or 0.0), 3),
            )
            self.evidence_table.setItem(table_row, 2, score_item)

            direction = str(evidence.get("direction") or "neutral")
            direction_item = QTableWidgetItem(direction)
            if direction == "supports":
                direction_item.setForeground(QColor(COLORS["success"]))
            elif direction == "contradicts":
                direction_item.setForeground(QColor(COLORS["error"]))
            self.evidence_table.setItem(table_row, 3, direction_item)

            excerpt = str(evidence.get("excerpt") or "")
            self.evidence_table.setItem(
                table_row,
                4,
                QTableWidgetItem(excerpt[:260] + ("..." if len(excerpt) > 260 else "")),
            )
        self.evidence_table.setSortingEnabled(True)

        source_text = self._evidence_table_source or "no source"
        self.ev_status_label.setText(
            f"Showing {len(filtered_rows):,} of {len(self._evidence_table_rows):,} "
            f"evidence rows from {source_text}"
        )

    def _on_evidence_cell_clicked(self, row, col):
        title_item = self.evidence_table.item(row, 0)
        evidence = title_item.data(Qt.ItemDataRole.UserRole) if title_item else None
        if not isinstance(evidence, dict):
            return

        details = [
            str(evidence.get("title") or "Untitled"),
            "",
            f"Component: {evidence.get('component') or 'Primary question'}",
            f"Direction: {evidence.get('direction') or 'neutral'}",
            f"Score: {float(evidence.get('score') or 0.0):.3f}",
        ]
        citation_bits = []
        if evidence.get("year"):
            citation_bits.append(str(evidence["year"]))
        if evidence.get("doi"):
            citation_bits.append(f"DOI {evidence['doi']}")
        if evidence.get("pmid"):
            citation_bits.append(f"PMID {evidence['pmid']}")
        if citation_bits:
            details.append("Citation: " + " | ".join(citation_bits))
        if evidence.get("authors"):
            details.append(f"Authors: {evidence['authors']}")

        for label, key in (
            ("Mechanisms", "mechanisms"),
            ("Conditions", "conditions"),
            ("Key entities", "key_entities"),
            ("Matched keywords", "matched_keywords"),
        ):
            values = evidence.get(key) or []
            if values:
                details.append(f"{label}: {', '.join(str(value) for value in values)}")
        if evidence.get("excerpt"):
            details.extend(["", "Evidence excerpt:", str(evidence["excerpt"])])
        self.evidence_detail.setPlainText("\n".join(details))
