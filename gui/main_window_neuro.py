"""Neuroimaging tab: coordinates, NeuroQuery convergence, network graph.

Split from gui/main_window.py; mixed into MainWindow.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.theme import COLORS
from gui.widgets import SectionHeader

logger = logging.getLogger(__name__)


# Convergence at or above this is shown as meaningful spatial agreement.
_CONVERGENCE_GOOD = 0.3


class _BackgroundCall(QThread):
    """Run one callable off the GUI thread; emit its result or error text."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[], object]):
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # reported to the user via ``failed``
            logger.exception("Neuroimaging background task failed")
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(result)


def _format_convergence(score) -> str:
    """Convergence score to 3 decimals; "n/a" when NeuroQuery gave no prediction."""
    return "n/a" if score is None else f"{score:.3f}"


def _convergence_score_item(score) -> QTableWidgetItem:
    """Score cell; a missing score is not coloured as good or weak agreement."""
    item = QTableWidgetItem(_format_convergence(score))
    if score is None:
        item.setToolTip("NeuroQuery returned no usable prediction for this component.")
    elif score >= _CONVERGENCE_GOOD:
        item.setForeground(QColor(COLORS["success"]))
    elif score > 0:
        item.setForeground(QColor(COLORS["warning"]))
    return item


class NeuroMixin:
    """Neuroimaging tab: coordinates, NeuroQuery convergence, network graph."""

    # ── Tab 10: Neuroimaging ───────────────────────────────────────────

    def _build_neuroimaging_tab(self) -> QWidget:
        from gui.network_view import NetworkGraphWidget

        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        inner = QWidget()
        layout = QVBoxLayout(inner)

        # ── Section 1: Coordinate Extraction ──
        layout.addWidget(SectionHeader("Coordinate Extraction"))

        coord_ctrl = QHBoxLayout()
        self.neuro_extract_btn = QPushButton("Extract Coordinates")
        self.neuro_extract_btn.clicked.connect(self._start_coordinate_extraction)
        coord_ctrl.addWidget(self.neuro_extract_btn)
        self.neuro_extract_status = QLabel("No coordinates extracted yet.")
        coord_ctrl.addWidget(self.neuro_extract_status, 1)
        layout.addLayout(coord_ctrl)

        # Coordinate results table
        self.neuro_coord_table = QTableWidget(0, 8)
        self.neuro_coord_table.setHorizontalHeaderLabels([
            "Article", "X", "Y", "Z", "Space", "Region", "Stat", "N",
        ])
        self.neuro_coord_table.horizontalHeader().setStretchLastSection(True)
        self.neuro_coord_table.setAlternatingRowColors(True)
        self.neuro_coord_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.neuro_coord_table)

        # Export buttons
        export_row = QHBoxLayout()
        self.neuro_export_nimare_btn = QPushButton("Export NiMARE")
        self.neuro_export_nimare_btn.clicked.connect(self._export_nimare)
        self.neuro_export_nimare_btn.setEnabled(False)
        export_row.addWidget(self.neuro_export_nimare_btn)
        self.neuro_export_ale_btn = QPushButton("Export GingerALE")
        self.neuro_export_ale_btn.clicked.connect(self._export_ginger_ale)
        self.neuro_export_ale_btn.setEnabled(False)
        export_row.addWidget(self.neuro_export_ale_btn)
        self.neuro_export_csv_btn = QPushButton("Export CSV")
        self.neuro_export_csv_btn.clicked.connect(self._export_coords_csv)
        self.neuro_export_csv_btn.setEnabled(False)
        export_row.addWidget(self.neuro_export_csv_btn)
        export_row.addStretch()
        layout.addLayout(export_row)

        # ── Section 2: NeuroQuery Cross-Reference ──
        layout.addWidget(SectionHeader("NeuroQuery Cross-Reference"))

        nq_ctrl = QHBoxLayout()
        self.neuro_nq_btn = QPushButton("Run NeuroQuery Analysis")
        self.neuro_nq_btn.clicked.connect(self._start_neuroquery_analysis)
        nq_ctrl.addWidget(self.neuro_nq_btn)
        self.neuro_nq_status = QLabel("No NeuroQuery analysis run yet.")
        nq_ctrl.addWidget(self.neuro_nq_status, 1)
        layout.addLayout(nq_ctrl)

        # Convergence results table
        self.neuro_convergence_table = QTableWidget(0, 5)
        self.neuro_convergence_table.setHorizontalHeaderLabels([
            "Component", "Convergence Score", "Matched", "Predicted-Only",
            "Observed-Only",
        ])
        self.neuro_convergence_table.horizontalHeader().setStretchLastSection(True)
        self.neuro_convergence_table.setAlternatingRowColors(True)
        self.neuro_convergence_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.neuro_convergence_table.currentCellChanged.connect(
            self._on_convergence_row_selected
        )
        layout.addWidget(self.neuro_convergence_table)

        # Detail panel for selected component
        self.neuro_convergence_detail = QTextEdit()
        self.neuro_convergence_detail.setReadOnly(True)
        self.neuro_convergence_detail.setAccessibleName("Convergence details")
        self.neuro_convergence_detail.setMaximumHeight(120)
        self.neuro_convergence_detail.setPlaceholderText(
            "Select a component row to see matched/unmatched region details."
        )
        layout.addWidget(self.neuro_convergence_detail)

        # ── Section 3: Network Visualization ──
        layout.addWidget(SectionHeader("Condition\u2013Region Network"))

        self.neuro_network_widget = NetworkGraphWidget()
        self.neuro_network_widget.setMinimumHeight(350)
        layout.addWidget(self.neuro_network_widget, 1)

        net_ctrl = QHBoxLayout()
        self.neuro_load_network_btn = QPushButton("Load Network Data")
        self.neuro_load_network_btn.clicked.connect(self._load_network_data)
        net_ctrl.addWidget(self.neuro_load_network_btn)
        self.neuro_export_graph_btn = QPushButton("Export Graph")
        self.neuro_export_graph_btn.clicked.connect(self._export_network_graph)
        self.neuro_export_graph_btn.setEnabled(False)
        net_ctrl.addWidget(self.neuro_export_graph_btn)
        net_ctrl.addStretch()
        layout.addLayout(net_ctrl)

        scroll.setWidget(inner)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)

        self._add_help_button(tab, "neuroimaging")

        # Internal state
        self._neuro_coordinates: list = []
        self._neuro_convergence_report: dict = {}

        return tab

    # ── Neuroimaging handlers ──────────────────────────────────────────

    def _start_coordinate_extraction(self):
        """Extract brain coordinates from the fulltext corpus on a worker thread."""
        from process.coordinate_extractor import (
            convert_tal_to_mni,
            extract_coordinates_from_corpus,
        )

        config = self.pipeline_config
        self.neuro_extract_status.setText("Extracting coordinates...")
        self.neuro_extract_btn.setEnabled(False)
        worker = _BackgroundCall(
            lambda: convert_tal_to_mni(extract_coordinates_from_corpus(config))
        )
        worker.succeeded.connect(self._on_coordinates_extracted)
        worker.failed.connect(self._on_coordinate_extraction_failed)
        self._neuro_coord_worker = worker
        worker.start()

    def _on_coordinates_extracted(self, coords) -> None:
        self._neuro_coordinates = coords
        self.neuro_coord_table.setRowCount(len(coords))
        for row, c in enumerate(coords):
            stat = f"{c.statistic_type}={c.statistic_value}" if c.statistic_type else ""
            values = (c.article_id, str(c.x), str(c.y), str(c.z), c.space,
                      c.region_label, stat, str(c.n_subjects))
            for col, value in enumerate(values):
                self.neuro_coord_table.setItem(row, col, QTableWidgetItem(value))
        n = len(coords)
        articles = len({c.article_id for c in coords})
        self.neuro_extract_status.setText(
            f"Extracted {n} coordinates from {articles} articles."
        )
        for btn in (self.neuro_export_nimare_btn, self.neuro_export_ale_btn,
                    self.neuro_export_csv_btn):
            btn.setEnabled(n > 0)
        self._log(f"Coordinate extraction: {n} peaks from {articles} articles")
        self.neuro_extract_btn.setEnabled(True)

    def _on_coordinate_extraction_failed(self, message: str) -> None:
        self.neuro_extract_status.setText(f"Error: {message}")
        self._log(f"Coordinate extraction failed: {message}", "ERROR")
        self.neuro_extract_btn.setEnabled(True)

    def _export_nimare(self):
        from process.coordinate_extractor import export_nimare_dataset

        path, _ = QFileDialog.getSaveFileName(
            self, "Export NiMARE Dataset", "nimare_dataset.json",
            "JSON files (*.json);;All files (*)",
        )
        if path:
            export_nimare_dataset(self._neuro_coordinates, Path(path))
            self._log(f"NiMARE dataset exported to {path}")

    def _export_ginger_ale(self):
        from process.coordinate_extractor import export_sleuth_format

        path, _ = QFileDialog.getSaveFileName(
            self, "Export GingerALE/Sleuth Format", "coordinates.txt",
            "Text files (*.txt);;All files (*)",
        )
        if path:
            export_sleuth_format(self._neuro_coordinates, Path(path))
            self._log(f"GingerALE/Sleuth file exported to {path}")

    def _export_coords_csv(self):
        from process.coordinate_extractor import export_csv

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Coordinates CSV", "coordinates.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            export_csv(self._neuro_coordinates, Path(path))
            self._log(f"Coordinate CSV exported to {path}")

    def _neuroquery_components(self) -> list[dict]:
        """Hypothesis components as plain dicts for the convergence report."""
        source_comps = (
            self.hypothesis_components
            or getattr(self.session, "hypothesis_components", [])
        )
        return [
            {
                "id": getattr(comp, "id", str(comp)),
                "label": getattr(comp, "label", str(comp)),
                "keywords": getattr(comp, "keywords", []),
            }
            for comp in source_comps
        ]

    def _neuroquery_observed_claims(self) -> list[dict]:
        """Claims from the one resolved claims file (Stage 5 output when present).

        Only that file is read, so papers Stage 5 removed and other JSON files
        in the claims folder do not count as observed (audits 19-M4, 16-M6).
        An unreadable file is reported, not skipped silently.
        """
        from process.neuroquery_bridge import load_convergence_claims
        from process.paths import resolve_claims_corpus

        config = self.pipeline_config
        path = resolve_claims_corpus(config)
        if path is None:
            self._log("NeuroQuery: no claims file found; no regions count as observed.",
                      "WARNING")
            return []
        try:
            claims = load_convergence_claims(config)
        except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
            self._log(f"NeuroQuery: could not read {path} ({exc}); no regions "
                      "count as observed.", "WARNING")
            return []
        return [claim for claim in claims if isinstance(claim, dict)]

    def _start_neuroquery_analysis(self):
        """Run NeuroQuery convergence for each component on a worker thread."""
        from process.neuroquery_bridge import generate_convergence_report

        components = self._neuroquery_components()
        claims = self._neuroquery_observed_claims()
        config = self.pipeline_config
        self.neuro_nq_status.setText("Querying NeuroQuery API...")
        self.neuro_nq_btn.setEnabled(False)
        worker = _BackgroundCall(
            lambda: generate_convergence_report(components, claims, config=config)
        )
        worker.succeeded.connect(self._on_neuroquery_report)
        worker.failed.connect(self._on_neuroquery_failed)
        self._neuro_nq_worker = worker
        worker.start()

    def _on_neuroquery_report(self, report: dict) -> None:
        self._neuro_convergence_report = report
        rows = list(report.values())
        self.neuro_convergence_table.setRowCount(len(rows))
        for i, conv in enumerate(rows):
            label = conv.get("component_label", conv.get("component_id", ""))
            counts = (len(conv.get("matched_regions", [])),
                      len(conv.get("predicted_not_observed", [])),
                      len(conv.get("observed_not_predicted", [])))
            self.neuro_convergence_table.setItem(i, 0, QTableWidgetItem(str(label)))
            self.neuro_convergence_table.setItem(
                i, 1, _convergence_score_item(conv.get("convergence_score")))
            for col, count in enumerate(counts, start=2):
                self.neuro_convergence_table.setItem(i, col, QTableWidgetItem(str(count)))
        n = len(rows)
        api_ok = sum(1 for r in rows if r.get("neuroquery_available"))
        self.neuro_nq_status.setText(
            f"Analysis complete: {n} components ({api_ok} with API data)."
        )
        self._log(f"NeuroQuery analysis: {n} components, {api_ok} with API data")
        self.neuro_nq_btn.setEnabled(True)

    def _on_neuroquery_failed(self, message: str) -> None:
        self.neuro_nq_status.setText(f"Error: {message}")
        self._log(f"NeuroQuery analysis failed: {message}", "ERROR")
        self.neuro_nq_btn.setEnabled(True)

    def _on_convergence_row_selected(self, row: int, col: int, prev_row: int, prev_col: int):
        """Show detail for selected convergence table row."""
        if row < 0:
            self.neuro_convergence_detail.clear()
            return
        rows = list(self._neuro_convergence_report.values())
        if row >= len(rows):
            return
        conv = rows[row]
        lines = [
            f"<b>{conv.get('component_label', '')}</b> "
            f"(score: {_format_convergence(conv.get('convergence_score'))})<br><br>",
        ]
        matched = conv.get("matched_regions", [])
        if matched:
            lines.append(f"<b>Matched:</b> {', '.join(matched)}<br>")
        pred = conv.get("predicted_not_observed", [])
        if pred:
            lines.append(f"<b>Predicted only:</b> {', '.join(pred)}<br>")
        obs = conv.get("observed_not_predicted", [])
        if obs:
            lines.append(f"<b>Observed only:</b> {', '.join(obs)}<br>")
        if not conv.get("neuroquery_available"):
            lines.append("<i>NeuroQuery API was unavailable for this query.</i>")
        self.neuro_convergence_detail.setHtml("".join(lines))

    def _load_network_data(self):
        """Load network analysis JSON file into the network widget."""
        import json

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Network Data", "",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.neuro_network_widget.set_network_data(data)
            self.neuro_export_graph_btn.setEnabled(True)
            self._log(f"Network data loaded from {path}")
        except Exception as exc:
            self._log(f"Failed to load network data: {exc}", "ERROR")
            QMessageBox.warning(self, "Load Error", str(exc))

    def _export_network_graph(self):
        """Export the network graph as SVG or PNG."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Network Graph", "network_graph.svg",
            "SVG files (*.svg);;PNG files (*.png);;All files (*)",
        )
        if not path:
            return
        fmt = "png" if path.lower().endswith(".png") else "svg"
        self.neuro_network_widget.export_image(Path(path), fmt=fmt)
        self._log(f"Network graph exported to {path}")
