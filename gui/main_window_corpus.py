"""Corpus Browser tab: build, populate, filter, open, and export the corpus table.

Split from gui/main_window.py; mixed into MainWindow.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from core.doi import doi_url
from gui.corpus_table_model import HEADERS

# A spreadsheet treats a cell starting with one of these as a formula.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value) -> str:
    """Prefix formula-like text with ' so Excel shows it as text (audit 19-L3).

    Plain numbers (a score of -1, say) are left alone.
    """
    text = str(value)
    if not text.startswith(_FORMULA_PREFIXES):
        return text
    try:
        float(text)
        return text
    except ValueError:
        return "'" + text


def _write_corpus_csv(proxy, path: Path) -> None:
    """The proxy's visible rows as CSV, with formula-like cells neutralised."""
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        for row in range(proxy.rowCount()):
            writer.writerow([
                _csv_safe(proxy.index(row, col).data() or "")
                for col in range(len(HEADERS))
            ])


class CorpusMixin:
    """Corpus Browser tab: build, populate, filter, open, and export the corpus table."""

    # ═══════════════════════════════════════════════════════════════
    # Feature B: Corpus Browser Tab
    # ═══════════════════════════════════════════════════════════════

    def _build_corpus_tab(self) -> QWidget:
        """Build the Corpus Browser tab (Tab 13)."""

        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Filter bar
        filter_row = QHBoxLayout()
        lbl_corpus_filter = QLabel("Filter:")
        filter_row.addWidget(lbl_corpus_filter)
        self._corpus_filter = QLineEdit()
        self._corpus_filter.setPlaceholderText("Filter by title, author, year…")
        self._corpus_filter.setAccessibleName("Corpus filter")
        self._corpus_filter.textChanged.connect(self._filter_corpus_table)
        lbl_corpus_filter.setBuddy(self._corpus_filter)
        filter_row.addWidget(self._corpus_filter, 1)

        lbl_relevance_filter = QLabel("Rated:")
        filter_row.addWidget(lbl_relevance_filter)
        self._corpus_relevance_filter = QComboBox()
        self._corpus_relevance_filter.addItem("All papers", "all")
        self._corpus_relevance_filter.addItem("Stage 2 relevant", "stage2")
        self._corpus_relevance_filter.currentIndexChanged.connect(
            lambda _: self._filter_corpus_table(self._corpus_filter.text())
        )
        lbl_relevance_filter.setBuddy(self._corpus_relevance_filter)
        filter_row.addWidget(self._corpus_relevance_filter)

        open_doi_btn = QPushButton("Open DOI")
        open_doi_btn.setToolTip("Open the selected paper's DOI in the browser")
        open_doi_btn.clicked.connect(self._open_corpus_doi)
        filter_row.addWidget(open_doi_btn)

        export_btn = QPushButton("Export Visible")
        export_btn.setToolTip("Export visible rows to CSV")
        export_btn.clicked.connect(self._export_corpus_visible)
        filter_row.addWidget(export_btn)

        layout.addLayout(filter_row)

        # Table — virtualized model/view so large corpora never materialize a
        # widget per cell (the old QTableWidget created 7 items per paper).
        from gui.corpus_table_model import CorpusFilterProxyModel, CorpusTableModel

        self._corpus_model = CorpusTableModel(self)
        self._corpus_proxy = CorpusFilterProxyModel(self)
        self._corpus_proxy.setSourceModel(self._corpus_model)
        self._corpus_table = QTableView()
        self._corpus_table.setModel(self._corpus_proxy)
        self._corpus_table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self._corpus_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._corpus_table.setSortingEnabled(True)
        hh = self._corpus_table.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self._corpus_table.verticalHeader().setVisible(False)
        self._corpus_table.doubleClicked.connect(self._show_corpus_abstract)
        layout.addWidget(self._corpus_table, 1)

        # Abstract preview pane
        layout.addWidget(QLabel("Abstract:"))
        self._corpus_abstract = QPlainTextEdit()
        self._corpus_abstract.setReadOnly(True)
        self._corpus_abstract.setAccessibleName("Corpus abstract preview")
        self._corpus_abstract.setMaximumHeight(160)
        self._corpus_abstract.setPlaceholderText("Double-click a row to view the abstract…")
        layout.addWidget(self._corpus_abstract)

        # Internal corpus cache for double-click lookup
        self._corpus_data: list[dict] = []
        self._corpus_stage_relevance: dict[str, set[str]] = {}
        self._corpus_stage_records: dict[str, dict[str, dict]] = {}

        return tab

    def _corpus_paper_key(self, paper: dict) -> str:
        for field in ("doi", "pmid", "s2_paper_id", "paperId", "openalex_id"):
            value = str(paper.get(field, "") or "").strip().lower()
            if value:
                return f"{field}:{value}"
        title = str(paper.get("title", "") or "").strip().lower()
        year = str(paper.get("year", "") or "").strip()
        return f"title:{title}|year:{year}"

    def _set_corpus_stage_relevance(self, stage_key: str, papers: list[dict]) -> None:
        records = {
            self._corpus_paper_key(paper): paper
            for paper in papers
            if isinstance(paper, dict)
        }
        self._corpus_stage_relevance[stage_key] = set(records)
        self._corpus_stage_records[stage_key] = records
        if self._corpus_data:
            self._populate_corpus_browser(self._corpus_data)
            self._filter_corpus_table(self._corpus_filter.text())

    def _populate_corpus_browser(self, corpus: list[dict]) -> None:
        """Fill the corpus browser table from a list of article dicts."""
        self._corpus_data = corpus
        self._corpus_model.set_corpus(
            corpus,
            self._corpus_stage_records.get("stage2", {}),
            self._corpus_paper_key,
        )
        self._corpus_table.resizeColumnToContents(0)

    def _filter_corpus_table(self, text: str) -> None:
        """Show/hide corpus table rows based on filter text."""
        stage_filter = "all"
        if hasattr(self, "_corpus_relevance_filter"):
            stage_filter = self._corpus_relevance_filter.currentData() or "all"
        if stage_filter == "all":
            self._corpus_proxy.set_stage_keys(None)
        else:
            self._corpus_proxy.set_stage_keys(
                self._corpus_stage_relevance.get(stage_filter, set())
            )
        self._corpus_proxy.set_text_filter(text)

    def _corpus_paper_for_proxy_row(self, proxy_row: int) -> dict | None:
        """Map a visible (proxy) row back to its paper dict."""
        proxy_index = self._corpus_proxy.index(proxy_row, 0)
        if not proxy_index.isValid():
            return None
        source_row = self._corpus_proxy.mapToSource(proxy_index).row()
        idx = self._corpus_model.corpus_index(source_row)
        if 0 <= idx < len(self._corpus_data):
            return self._corpus_data[idx]
        return None

    def _show_corpus_abstract(self, index) -> None:
        """Show the abstract for the double-clicked row."""
        paper = self._corpus_paper_for_proxy_row(index.row())
        if paper is not None:
            abstract = paper.get("abstract", "(no abstract available)")
            title = paper.get("title", "")
            self._corpus_abstract.setPlainText(f"{title}\n\n{abstract}")

    def _open_corpus_doi(self) -> None:
        """Open the DOI of the selected corpus row in the browser."""
        import webbrowser
        current = self._corpus_table.currentIndex()
        if not current.isValid():
            return
        paper = self._corpus_paper_for_proxy_row(current.row())
        url = doi_url((paper or {}).get("doi"))
        if url:
            webbrowser.open(url)

    def _export_corpus_visible(self) -> None:
        """Export currently visible corpus rows to CSV."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Corpus to CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            _write_corpus_csv(self._corpus_proxy, Path(path))
        except OSError as exc:
            self._log(f"Corpus export to {path} failed: {exc}", "ERROR")
            QMessageBox.warning(self, "Export Failed",
                                f"Could not write the corpus CSV to\n{path}\n\n{exc}")
            return
        self._log(f"Corpus exported to {path}")
