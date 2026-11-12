"""Compare Runs and Report & Export tabs: snapshots, PRISMA, exports.

Split from gui/main_window.py; mixed into MainWindow.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.main_window_common import _make_file_row
from gui.widgets import SectionHeader

logger = logging.getLogger("gui.main_window")


def _checklist_key(item: dict) -> str:
    """Rows are matched by description: the table read-back turns item
    numbers such as "10a" into 0, so the number cannot be the key."""
    return str(item.get("description", ""))


def _with_manual_edits(checklist: list[dict], edits: dict) -> list[dict]:
    """New checklist rows with the reviewer's manual content put back."""
    return [
        {**item, "content": edits[_checklist_key(item)]}
        if item.get("status") != "auto" and _checklist_key(item) in edits else item
        for item in checklist
    ]


class ReportsMixin:
    """Compare Runs and Report & Export tabs: snapshots, PRISMA, exports."""

    # ── Tab 11: Compare Runs ──────────────────────────────────────────

    def _build_compare_runs_tab(self) -> QWidget:
        from PySide6.QtWidgets import QListWidget

        tab = QWidget()
        layout = QVBoxLayout(tab)

        layout.addWidget(SectionHeader("Stage 10: Compare Runs"))

        # ── Snapshot management section ──
        snap_group = QGroupBox("Living Review Snapshots")
        snap_layout = QVBoxLayout(snap_group)

        snap_btn_row = QHBoxLayout()
        self.snapshot_save_btn = QPushButton("Save Current Snapshot")
        self.snapshot_save_btn.clicked.connect(self._save_snapshot)
        snap_btn_row.addWidget(self.snapshot_save_btn)
        self.snapshot_refresh_btn = QPushButton("Refresh List")
        self.snapshot_refresh_btn.clicked.connect(self._refresh_snapshot_list)
        snap_btn_row.addWidget(self.snapshot_refresh_btn)
        self.snapshot_compare_btn = QPushButton("Compare Selected")
        self.snapshot_compare_btn.clicked.connect(self._compare_snapshots)
        self.snapshot_compare_btn.setEnabled(False)
        snap_btn_row.addWidget(self.snapshot_compare_btn)
        snap_btn_row.addStretch()
        snap_layout.addLayout(snap_btn_row)

        self.snapshot_status = QLabel("No snapshots loaded.")
        snap_layout.addWidget(self.snapshot_status)

        self.snapshot_list = QListWidget()
        self.snapshot_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.snapshot_list.setMaximumHeight(120)
        self.snapshot_list.itemSelectionChanged.connect(self._on_snapshot_selection_changed)
        snap_layout.addWidget(self.snapshot_list)

        layout.addWidget(snap_group)

        # ── Snapshot diff display ──
        self.snapshot_diff_display = QTextEdit()
        self.snapshot_diff_display.setReadOnly(True)
        self.snapshot_diff_display.setAccessibleName("Snapshot comparison results")
        self.snapshot_diff_display.setPlaceholderText(
            "Snapshot comparison results will appear here..."
        )
        layout.addWidget(self.snapshot_diff_display)

        # ── Manual run comparison section ──
        layout.addWidget(SectionHeader("Manual Run Comparison"))

        ra_row, self.compare_run_a, ra_btn = _make_file_row(
            "Run A:", "path/to/run_a.json"
        )
        layout.addLayout(ra_row)
        rb_row, self.compare_run_b, rb_btn = _make_file_row(
            "Run B:", "path/to/run_b.json"
        )
        layout.addLayout(rb_row)
        co_row, self.compare_output, co_btn = _make_file_row(
            "Output:", "path/to/compare.json", browse_save=True
        )
        layout.addLayout(co_row)
        json_f = "JSON files (*.json);;All files (*)"
        ra_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.compare_run_a, json_f, self.pipeline_config.claims_dir
            )
        )
        rb_btn.clicked.connect(
            lambda: self._browse_open_file(
                self.compare_run_b, json_f, self.pipeline_config.claims_dir
            )
        )
        co_btn.clicked.connect(
            lambda: self._browse_save_file(
                self.compare_output,
                "Markdown (*.md);;JSON files (*.json);;All files (*)",
                self.pipeline_config.output_dir,
                self.pipeline_config.data_dir / "validation",
            )
        )

        self.compare_btn = QPushButton("Run Compare Runs")
        self.compare_btn.clicked.connect(self._start_compare_runs)
        layout.addWidget(self.compare_btn)

        self.compare_progress = QProgressBar()
        self.compare_progress.setVisible(False)
        layout.addWidget(self.compare_progress)
        self.compare_status = QLabel("")
        layout.addWidget(self.compare_status)

        self.compare_results = QTextEdit()
        self.compare_results.setReadOnly(True)
        self.compare_results.setAccessibleName("Comparison results")
        layout.addWidget(self.compare_results)

        self._add_help_button(tab, "compare_runs")
        return tab

    # ── Snapshot handlers ─────────────────────────────────────────────

    def _save_snapshot(self):
        """Save current pipeline state as a snapshot."""
        from core.snapshots import save_snapshot
        try:
            path = save_snapshot(self.pipeline_config)
            self.snapshot_status.setText(f"Snapshot saved: {path.stem}")
            self._log(f"Snapshot saved: {path}")
            self._refresh_snapshot_list()
        except Exception as exc:
            self.snapshot_status.setText(f"Error: {exc}")
            self._log(f"Snapshot save failed: {exc}", level="ERROR")

    def _refresh_snapshot_list(self):
        """Reload the snapshot list from disk."""
        from core.snapshots import list_snapshots
        self.snapshot_list.clear()
        self._snapshots = list_snapshots(self.pipeline_config)
        for snap in self._snapshots:
            label = f"{snap.run_id}  ({snap.timestamp[:19]})  — {snap.summary.get('stages_completed', 0)} stages"
            self.snapshot_list.addItem(label)
        if self._snapshots:
            self.snapshot_status.setText(f"{len(self._snapshots)} snapshot(s) found.")
        else:
            self.snapshot_status.setText("No snapshots found.")
        self.snapshot_compare_btn.setEnabled(False)

    def _on_snapshot_selection_changed(self):
        """Enable compare button when exactly two snapshots are selected."""
        selected = self.snapshot_list.selectedItems()
        self.snapshot_compare_btn.setEnabled(len(selected) == 2)

    def _compare_snapshots(self):
        """Compare two selected snapshots and display the diff."""
        from core.snapshots import diff_snapshots, generate_diff_report
        selected = self.snapshot_list.selectedItems()
        if len(selected) != 2:
            return

        indices = sorted(self.snapshot_list.row(item) for item in selected)
        if not hasattr(self, "_snapshots") or len(self._snapshots) <= max(indices):
            return

        old_snap = self._snapshots[indices[1]]  # older (later in list, sorted newest first)
        new_snap = self._snapshots[indices[0]]  # newer

        try:
            diff = diff_snapshots(old_snap, new_snap, self.pipeline_config)

            # Display summary
            lines = [diff.summary_text, ""]
            if diff.new_articles:
                lines.append(f"### New Articles ({len(diff.new_articles)})")
                for art in diff.new_articles[:20]:
                    lines.append(f"  + {art.get('title', 'Untitled')} ({art.get('year', '')})")
                if len(diff.new_articles) > 20:
                    lines.append(f"  ... and {len(diff.new_articles) - 20} more")
                lines.append("")

            if diff.removed_articles:
                lines.append(f"### Removed Articles ({len(diff.removed_articles)})")
                for art in diff.removed_articles[:20]:
                    lines.append(f"  - {art.get('title', 'Untitled')} ({art.get('year', '')})")
                if len(diff.removed_articles) > 20:
                    lines.append(f"  ... and {len(diff.removed_articles) - 20} more")
                lines.append("")

            if diff.score_changes:
                lines.append(f"### Score Changes ({len(diff.score_changes)})")
                for sc in diff.score_changes[:20]:
                    lines.append(
                        f"  ~ {sc.get('title', '')[:60]}: "
                        f"{sc.get('old_score', '?')} -> {sc.get('new_score', '?')}"
                    )
                lines.append("")

            if diff.grade_changes:
                lines.append("### GRADE Assessment Changed")
                lines.append("")

            self.snapshot_diff_display.setPlainText("\n".join(lines))
            self.snapshot_status.setText(
                f"Compared {old_snap.run_id} vs {new_snap.run_id}"
            )
            self._log(
                f"Snapshot comparison: {old_snap.run_id} vs {new_snap.run_id}"
            )

            # Save report if output dir exists
            report_path = self.pipeline_config.output_dir / "snapshot_diff_report.md"
            try:
                generate_diff_report(diff, report_path)
                self._log(f"Diff report saved: {report_path}")
            except Exception as exc:
                logger.exception("Could not save snapshot diff report to %s", report_path)
                self._log(f"Diff report not saved to {report_path}: {exc}", level="ERROR")
                self.snapshot_status.setText(
                    f"Compared {old_snap.run_id} vs {new_snap.run_id}; "
                    f"report not saved: {exc}"
                )

        except Exception as exc:
            self.snapshot_diff_display.setPlainText(f"Error: {exc}")
            self._log(f"Snapshot comparison failed: {exc}", level="ERROR")

    # ── Tab 12: Report & Export ───────────────────────────────────────

    def _build_report_export_tab(self) -> QWidget:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        tab = QWidget()
        layout = QVBoxLayout(tab)

        # ── Top controls ──
        ctrl_row = QHBoxLayout()
        self.prisma_generate_btn = QPushButton("Generate PRISMA Report")
        self.prisma_generate_btn.setMinimumHeight(30)
        self.prisma_generate_btn.clicked.connect(self._generate_prisma_report)
        ctrl_row.addWidget(self.prisma_generate_btn)
        self.prisma_status_label = QLabel("No report generated yet.")
        ctrl_row.addWidget(self.prisma_status_label, 1)
        layout.addLayout(ctrl_row)

        # ── Splitter: diagram (top) / checklist (bottom) ──
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Upper: PRISMA flow diagram preview
        upper = QWidget()
        u_layout = QVBoxLayout(upper)
        u_layout.setContentsMargins(0, 0, 0, 0)
        u_layout.addWidget(SectionHeader("PRISMA 2020 Flow Diagram"))

        placeholder_fig = Figure(figsize=(8, 10), dpi=100, facecolor="white")
        placeholder_ax = placeholder_fig.add_subplot(111)
        placeholder_ax.text(
            0.5, 0.5,
            "Click 'Generate PRISMA Report' to create the flow diagram",
            ha="center", va="center", fontsize=11, color="#808080",
        )
        placeholder_ax.axis("off")

        self.prisma_canvas = FigureCanvasQTAgg(placeholder_fig)
        self.prisma_canvas.setMinimumHeight(300)
        u_layout.addWidget(self.prisma_canvas)
        splitter.addWidget(upper)

        # Lower: checklist table + export
        lower = QWidget()
        l_layout = QVBoxLayout(lower)
        l_layout.setContentsMargins(0, 0, 0, 0)
        l_layout.addWidget(SectionHeader("PRISMA Checklist"))

        self.prisma_checklist_table = QTableWidget(0, 5)
        self.prisma_checklist_table.setHorizontalHeaderLabels(
            ["#", "Section", "Description", "Status", "Content"]
        )
        header = self.prisma_checklist_table.horizontalHeader()
        header.setStretchLastSection(True)
        self.prisma_checklist_table.setColumnWidth(0, 32)
        self.prisma_checklist_table.setColumnWidth(1, 110)
        self.prisma_checklist_table.setColumnWidth(2, 280)
        self.prisma_checklist_table.setColumnWidth(3, 60)
        self.prisma_checklist_table.setMinimumHeight(150)
        l_layout.addWidget(self.prisma_checklist_table)

        export_scroll = QScrollArea()
        export_scroll.setWidgetResizable(True)
        export_scroll.setMinimumHeight(210)
        export_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        export_container = QWidget()
        exports_layout = QVBoxLayout(export_container)
        exports_layout.setContentsMargins(4, 4, 4, 4)
        exports_layout.setSpacing(8)

        # Export section — PRISMA
        export_group = QGroupBox("Export PRISMA Report")
        eg_layout = QHBoxLayout(export_group)
        self.prisma_export_btn = QPushButton("Export PRISMA Report...")
        self.prisma_export_btn.setEnabled(False)
        self.prisma_export_btn.setMinimumSize(210, 30)
        self.prisma_export_btn.clicked.connect(self._export_prisma_report)
        eg_layout.addWidget(self.prisma_export_btn)
        prisma_export_note = QLabel("Saves flow diagram + checklist to a directory")
        prisma_export_note.setWordWrap(True)
        eg_layout.addWidget(prisma_export_note, 1)
        eg_layout.addStretch()
        export_group.setMinimumHeight(58)
        exports_layout.addWidget(export_group)

        # Export section — Bibliography
        bib_group = QGroupBox("Export Bibliography")
        bg_layout = QHBoxLayout(bib_group)
        lbl_format = QLabel("Format:")
        bg_layout.addWidget(lbl_format)
        self.bib_format_combo = QComboBox()
        self.bib_format_combo.addItems(["BibTeX (.bib)", "RIS (.ris)", "EndNote XML (.xml)"])
        self.bib_format_combo.setMinimumWidth(180)
        lbl_format.setBuddy(self.bib_format_combo)
        bg_layout.addWidget(self.bib_format_combo)
        self.bib_export_btn = QPushButton("Export Bibliography...")
        self.bib_export_btn.setMinimumSize(210, 30)
        self.bib_export_btn.clicked.connect(self._export_bibliography)
        bg_layout.addWidget(self.bib_export_btn)
        bg_layout.addStretch()
        bib_group.setMinimumHeight(58)
        exports_layout.addWidget(bib_group)

        # Export section — Full Report Bundle
        bundle_group = QGroupBox("Full Report Bundle")
        bu_layout = QHBoxLayout(bundle_group)
        self.bundle_export_btn = QPushButton("Export Full Report Bundle...")
        self.bundle_export_btn.setMinimumSize(210, 30)
        self.bundle_export_btn.clicked.connect(self._export_report_bundle)
        bu_layout.addWidget(self.bundle_export_btn)
        self.bundle_status_label = QLabel(
            "One-click export of all pipeline outputs (PRISMA, GRADE, plots, bibliography)"
        )
        self.bundle_status_label.setWordWrap(True)
        bu_layout.addWidget(self.bundle_status_label, 1)
        bundle_group.setMinimumHeight(58)
        exports_layout.addWidget(bundle_group)

        # Export section — committee brief
        abct_group = QGroupBox("Committee Brief")
        abct_layout = QHBoxLayout(abct_group)
        self.abct_brief_export_btn = QPushButton("Export Committee Brief...")
        self.abct_brief_export_btn.setMinimumSize(210, 30)
        self.abct_brief_export_btn.clicked.connect(self._export_abct_committee_brief)
        abct_layout.addWidget(self.abct_brief_export_btn)
        self.abct_brief_status_label = QLabel(
            "Committee-facing evidence brief pack (audience and questions from "
            "the committee_brief config section)"
        )
        self.abct_brief_status_label.setWordWrap(True)
        abct_layout.addWidget(self.abct_brief_status_label, 1)
        abct_group.setMinimumHeight(58)
        exports_layout.addWidget(abct_group)

        # Export section — committee info-sheet
        abct_info_group = QGroupBox("Committee Info-Sheet")
        abct_info_layout = QHBoxLayout(abct_info_group)
        self.abct_infosheet_export_btn = QPushButton("Export Info-Sheet...")
        self.abct_infosheet_export_btn.setMinimumSize(210, 30)
        self.abct_infosheet_export_btn.clicked.connect(self._export_abct_tech_ai_infosheet)
        abct_info_layout.addWidget(self.abct_infosheet_export_btn)
        self.abct_infosheet_status_label = QLabel(
            "One-page handout (content from the abct_tech_ai_infosheet config section)"
        )
        self.abct_infosheet_status_label.setWordWrap(True)
        abct_info_layout.addWidget(self.abct_infosheet_status_label, 1)
        abct_info_group.setMinimumHeight(58)
        exports_layout.addWidget(abct_info_group)

        exports_layout.addStretch()
        export_scroll.setWidget(export_container)
        l_layout.addWidget(export_scroll)

        lower.setMinimumHeight(400)
        splitter.addWidget(lower)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([320, 480])

        layout.addWidget(splitter)
        self._add_help_button(tab, "report_export")
        return tab

    # ── PRISMA report handlers ────────────────────────────────────────

    def _generate_prisma_report(self):
        """Collect pipeline data and render the PRISMA diagram + checklist."""
        from process.prisma import (
            collect_prisma_data,
            generate_checklist,
            generate_flow_diagram_figure,
        )

        self.prisma_status_label.setText("Generating...")
        QApplication.processEvents()

        try:
            self._prisma_data = collect_prisma_data(
                self.pipeline_config,
            )
            fig = generate_flow_diagram_figure(self._prisma_data)

            # Replace the canvas figure
            self.prisma_canvas.figure = fig
            fig.set_canvas(self.prisma_canvas)
            self.prisma_canvas.draw()

            # Populate checklist table, keeping content typed into manual items
            edits = self._manual_checklist_edits()
            self._prisma_checklist = generate_checklist(self._prisma_data)
            self._populate_checklist_table(
                _with_manual_edits(self._prisma_checklist, edits))

            self.prisma_export_btn.setEnabled(True)
            self.prisma_status_label.setText(
                f"Report generated — {self._prisma_data.records_identified} records identified, "
                f"{self._prisma_data.studies_in_review} included."
            )
            self._log("PRISMA report generated successfully.")
        except Exception as exc:
            self.prisma_status_label.setText(f"Error: {exc}")
            self._log(f"PRISMA report generation failed: {exc}", level="ERROR")
            logger.exception("PRISMA report generation failed")

    def _populate_checklist_table(self, checklist: list):
        """Fill the checklist QTableWidget from checklist data."""
        table = self.prisma_checklist_table
        table.setRowCount(len(checklist))
        for row, item in enumerate(checklist):
            # Item number
            num_item = QTableWidgetItem(str(item["item"]))
            num_item.setFlags(num_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, num_item)

            # Section
            section_text = item["section"]
            if item.get("prisma_traice"):
                section_text += " (trAIce)"
            sec_item = QTableWidgetItem(section_text)
            sec_item.setFlags(sec_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 1, sec_item)

            # Description
            desc_item = QTableWidgetItem(item["description"])
            desc_item.setFlags(desc_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 2, desc_item)

            # Status
            status_item = QTableWidgetItem(item["status"])
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 3, status_item)

            # Content — editable for "manual" items
            content_item = QTableWidgetItem(item["content"])
            if item["status"] == "auto":
                content_item.setFlags(content_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 4, content_item)

    def _export_prisma_report(self):
        """Export the PRISMA flow diagram and checklist to a user-selected directory."""
        from process.prisma import (
            generate_flow_diagram,
            save_checklist_csv,
            save_checklist_markdown,
        )

        directory = QFileDialog.getExistingDirectory(
            self, "Select Export Directory"
        )
        if not directory:
            return

        out = Path(directory)

        try:
            # Read back any manual edits from the table
            checklist = self._read_checklist_from_table()

            # Save flow diagram in multiple formats
            generate_flow_diagram(self._prisma_data, out / "prisma_flow_diagram", fmt="png")
            generate_flow_diagram(self._prisma_data, out / "prisma_flow_diagram", fmt="svg")

            # Save checklist
            save_checklist_markdown(checklist, out / "prisma_checklist.md")
            save_checklist_csv(checklist, out / "prisma_checklist.csv")

            self.prisma_status_label.setText(f"Exported to {directory}")
            self._log(f"PRISMA report exported to {directory}")
            QMessageBox.information(
                self, "Export Complete",
                f"PRISMA report exported to:\n{directory}\n\n"
                "Files: prisma_flow_diagram.png, prisma_flow_diagram.svg, "
                "prisma_checklist.md, prisma_checklist.csv"
            )
        except Exception as exc:
            self.prisma_status_label.setText(f"Export error: {exc}")
            self._log(f"PRISMA export failed: {exc}", level="ERROR")
            logger.exception("PRISMA export failed")

    def _manual_checklist_edits(self) -> dict[str, str]:
        """Content typed into non-auto checklist rows, keyed by description.

        Only text that differs from what the last generation produced counts,
        so regenerating still refreshes untouched rows (audit 19-L2).
        """
        generated = {
            _checklist_key(item): item.get("content", "")
            for item in getattr(self, "_prisma_checklist", None) or []
        }
        return {
            _checklist_key(item): item["content"]
            for item in self._read_checklist_from_table()
            if item["status"] != "auto"
            and item["content"] != generated.get(_checklist_key(item), "")
        }

    def _read_checklist_from_table(self) -> list[dict]:
        """Read checklist data back from the table, capturing manual edits."""
        checklist = []
        table = self.prisma_checklist_table
        for row in range(table.rowCount()):
            item_widget = table.item(row, 0)
            if item_widget is None:
                continue
            section_item = table.item(row, 1)
            section_text = section_item.text() if section_item is not None else ""
            is_traice = "(trAIce)" in section_text
            section_text = section_text.replace(" (trAIce)", "")
            description_item = table.item(row, 2)
            status_item = table.item(row, 3)
            content_item = table.item(row, 4)
            checklist.append({
                "item": int(item_widget.text()) if item_widget.text().isdigit() else 0,
                "section": section_text,
                "description": description_item.text() if description_item is not None else "",
                "status": status_item.text() if status_item is not None else "",
                "content": content_item.text() if content_item is not None else "",
                "prisma_traice": is_traice,
            })
        return checklist

    # ── Bibliography & Bundle export handlers ─────────────────────────

    def _export_bibliography(self):
        """Export the corpus bibliography in the selected format."""
        from process.export import (
            export_bibtex,
            export_endnote_xml,
            export_ris,
            load_corpus_for_export,
        )

        fmt_text = self.bib_format_combo.currentText()
        if "BibTeX" in fmt_text:
            ext, export_fn = ".bib", export_bibtex
            filt = "BibTeX files (*.bib);;All files (*)"
        elif "RIS" in fmt_text:
            ext, export_fn = ".ris", export_ris
            filt = "RIS files (*.ris);;All files (*)"
        else:
            ext, export_fn = ".xml", export_endnote_xml
            filt = "XML files (*.xml);;All files (*)"

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Bibliography", f"corpus{ext}", filt
        )
        if not path:
            return

        try:
            corpus = load_corpus_for_export(self.pipeline_config)
            if not corpus:
                QMessageBox.warning(
                    self, "No Data",
                    "No corpus data found. Run the Search stage first."
                )
                return
            out = export_fn(corpus, Path(path))
            self._log(f"Bibliography exported to {out}")
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(corpus)} entries to:\n{out}"
            )
        except Exception as exc:
            self._log(f"Bibliography export failed: {exc}", level="ERROR")
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_report_bundle(self):
        """Export the full report bundle to a user-selected directory."""
        from process.export import export_report_bundle

        directory = QFileDialog.getExistingDirectory(
            self, "Select Bundle Export Directory"
        )
        if not directory:
            return

        try:
            self.bundle_status_label.setText("Exporting...")
            QApplication.processEvents()
            out = export_report_bundle(self.pipeline_config, Path(directory))
            self.bundle_status_label.setText(f"Bundle exported to {out}")
            self._log(f"Report bundle exported to {out}")
            QMessageBox.information(
                self, "Bundle Complete",
                f"Report bundle exported to:\n{out}\n\n"
                "See README.md in the directory for file descriptions."
            )
        except Exception as exc:
            self.bundle_status_label.setText(f"Error: {exc}")
            self._log(f"Report bundle export failed: {exc}", level="ERROR")
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_abct_committee_brief(self):
        """Export the committee evidence brief pack."""
        from process.committee_brief import export_abct_committee_brief

        directory = QFileDialog.getExistingDirectory(
            self, "Select Committee Brief Export Directory"
        )
        if not directory:
            return

        try:
            self.abct_brief_status_label.setText("Exporting...")
            QApplication.processEvents()
            out = export_abct_committee_brief(self.pipeline_config, Path(directory))
            self.abct_brief_status_label.setText(f"Committee brief exported to {out}")
            self._log(f"Committee brief exported to {out}")
            QMessageBox.information(
                self, "Committee Brief Complete",
                f"Committee brief exported to:\n{out}\n\n"
                "See README.md in the directory for file descriptions."
            )
        except Exception as exc:
            self.abct_brief_status_label.setText(f"Error: {exc}")
            self._log(f"Committee brief export failed: {exc}", level="ERROR")
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_abct_tech_ai_infosheet(self):
        """Export the one-page committee info-sheet pack."""
        from process.committee_brief import export_abct_tech_ai_infosheet

        directory = QFileDialog.getExistingDirectory(
            self, "Select Info-Sheet Export Directory"
        )
        if not directory:
            return

        try:
            self.abct_infosheet_status_label.setText("Exporting...")
            QApplication.processEvents()
            out = export_abct_tech_ai_infosheet(self.pipeline_config, Path(directory))
            self.abct_infosheet_status_label.setText(f"Info-sheet exported to {out}")
            self._log(f"Info-sheet exported to {out}")
            QMessageBox.information(
                self, "Info-Sheet Complete",
                f"Info-sheet exported to:\n{out}\n\n"
                "The Markdown file named in the pack README is the one-page handout."
            )
        except Exception as exc:
            self.abct_infosheet_status_label.setText(f"Error: {exc}")
            self._log(f"Info-sheet export failed: {exc}", level="ERROR")
            QMessageBox.critical(self, "Export Error", str(exc))
