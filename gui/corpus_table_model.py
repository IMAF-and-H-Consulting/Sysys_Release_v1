"""Virtualized model/view backing for the Corpus Browser table.

The browser previously materialized a ``QTableWidgetItem`` for every cell of
the corpus (7 C++ objects per paper), which made startup and stage-relevance
refreshes slow and memory-heavy for large corpora. ``CorpusTableModel`` holds
one small tuple of display strings per paper instead; the ``QTableView`` only
instantiates delegates for the rows actually on screen.

``CorpusFilterProxyModel`` reproduces the browser's row filtering (free text
over Title, every author and Year, plus the stage-relevance combo) and adds numerically
correct sorting for the #, Year, and Score columns, which the old
string-sorting table got wrong ("10" < "2").
"""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    QSortFilterProxyModel,
    Qt,
)

_Index = QModelIndex | QPersistentModelIndex

HEADERS = ["#", "Title", "Authors", "Year", "Source", "Score", "DOI"]
_SORT_ROLE = Qt.ItemDataRole.UserRole + 1


def _author_str(paper: dict) -> str:
    authors = paper.get("authors", [])
    if isinstance(authors, list):
        text = ", ".join(str(a) for a in authors[:3])
        if len(authors) > 3:
            text += " et al."
        return text
    return str(authors)


def _search_text(paper: dict, year: str) -> str:
    """Lower-case title, every author and year for the free-text filter."""
    authors = paper.get("authors", [])
    author_text = (" ".join(str(a) for a in authors) if isinstance(authors, list)
                   else str(authors or ""))
    return " ".join((str(paper.get("title") or ""), author_text, year)).lower()


def _to_number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class CorpusTableModel(QAbstractTableModel):
    """Read-only table model over a list of article dicts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[tuple[str, ...]] = []
        self._keys: list[str] = []
        self._sort_keys: list[tuple] = []
        self._search_texts: list[str] = []

    # ── population ────────────────────────────────────────────────────

    def set_corpus(
        self,
        corpus: list[dict],
        stage2_records: dict[str, dict],
        key_fn: Callable[[dict], str],
    ) -> None:
        """Rebuild the rows from the corpus.

        ``stage2_records`` supplies the relevance-score fallback for papers
        whose own record has no score yet; ``key_fn`` produces the stable
        paper key used by the stage-relevance filter.
        """
        rows: list[tuple[str, ...]] = []
        keys: list[str] = []
        sort_keys: list[tuple] = []
        search_texts: list[str] = []
        for i, paper in enumerate(corpus, 1):
            paper_key = key_fn(paper)
            score = paper.get("relevance_score", paper.get("score", ""))
            if score == "":
                record = stage2_records.get(paper_key, {})
                score = record.get("relevance_score", record.get("score", ""))
            score_str = (
                f"{score:.2f}" if isinstance(score, float)
                else str(score) if score is not None and score != "" else ""
            )
            year = str(paper.get("year") or "")
            rows.append((
                str(i),
                str(paper.get("title") or ""),
                _author_str(paper),
                year,
                str(paper.get("source") or ""),
                score_str,
                str(paper.get("doi") or ""),
            ))
            keys.append(paper_key)
            sort_keys.append((i, _to_number(year), _to_number(score)))
            search_texts.append(_search_text(paper, year))

        self.beginResetModel()
        self._rows = rows
        self._keys = keys
        self._sort_keys = sort_keys
        self._search_texts = search_texts
        self.endResetModel()

    # ── row lookups used by the browser ───────────────────────────────

    def paper_key(self, row: int) -> str:
        return self._keys[row] if 0 <= row < len(self._keys) else ""

    def search_text(self, row: int) -> str:
        """Title, all authors (not just the three shown) and year, lower-cased."""
        return self._search_texts[row] if 0 <= row < len(self._search_texts) else ""

    def corpus_index(self, row: int) -> int:
        """Index into the original corpus list (rows are in corpus order)."""
        return row

    # ── QAbstractTableModel interface ─────────────────────────────────

    def rowCount(self, parent: _Index = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: _Index = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(HEADERS)

    def data(self, index: _Index, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        row, col = index.row(), index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            return self._rows[row][col]
        if role == Qt.ItemDataRole.TextAlignmentRole and col == 0:
            return Qt.AlignmentFlag.AlignCenter
        if role == _SORT_ROLE:
            if col == 0:
                return self._sort_keys[row][0]
            if col == 3:
                return self._sort_keys[row][1]
            if col == 5:
                return self._sort_keys[row][2]
            return self._rows[row][col].lower()
        return None

    def headerData(self, section, orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if (role == Qt.ItemDataRole.DisplayRole
                and orientation == Qt.Orientation.Horizontal
                and 0 <= section < len(HEADERS)):
            return HEADERS[section]
        return None


class CorpusFilterProxyModel(QSortFilterProxyModel):
    """Free-text + stage-relevance filtering with numeric-aware sorting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ""
        self._stage_keys: set[str] | None = None

    def set_text_filter(self, text: str) -> None:
        self._set_filter_value("_text", text.lower())

    def set_stage_keys(self, keys: set[str] | None) -> None:
        """``None`` disables the stage filter; a set restricts to its keys."""
        self._set_filter_value("_stage_keys", keys)

    def _set_filter_value(self, attribute: str, value) -> None:
        # The project supports Qt 6.5; the complete filter-change API only
        # arrived in Qt 6.10. Qt 6.9 has beginFilterChange but no end method.
        begin = getattr(self, "beginFilterChange", None)
        end = getattr(self, "endFilterChange", None)
        if callable(begin) and callable(end):
            begin()
            setattr(self, attribute, value)
            end()
        else:
            setattr(self, attribute, value)
            self.invalidate()

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:
        model = self.sourceModel()
        if not isinstance(model, CorpusTableModel):
            return True
        if self._stage_keys is not None:
            if model.paper_key(source_row) not in self._stage_keys:
                return False
        if not self._text:
            return True
        return self._text in model.search_text(source_row)

    def lessThan(self, left: _Index, right: _Index) -> bool:
        lv = left.data(_SORT_ROLE)
        rv = right.data(_SORT_ROLE)
        if lv is None:
            return rv is not None
        if rv is None:
            return False
        try:
            return lv < rv
        except TypeError:
            return str(lv) < str(rv)
