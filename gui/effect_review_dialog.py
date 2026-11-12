"""Review extracted effects: include them, group them, and give conversions a direction.

Rows come from ``effects_for_review.json`` (written by Stage 2c and the
meta-analysis). Directly reported effects with their own uncertainty and
signed conversions (t, z) are included by default; conversions whose statistic
carries no sign (F, χ², p-value only) are excluded until the researcher ticks
them and gives a direction. A model group appears only after its semantic audit
passes; outcome-label suggestions only fill the Suggested group column.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

COLUMNS = ("Include", "Paper", "Statistic", "Effect", "How it was obtained",
           "Direction", "Suggested group", "Group")
COL_INCLUDE, COL_PAPER, COL_STAT, COL_EFFECT, COL_HOW, COL_DIRECTION, COL_SUGGESTED, COL_GROUP = range(8)
DIRECTIONS = ("", "+", "-")
FILTERS = ("All effects", "Reported directly", "Converted from a test statistic",
           "Needs a direction", "No group yet")
_INTRO = (
    "Tick the effects to pool and give each a group; effects in the same group are pooled "
    "together (mixed measures are put on one scale). An empty Group uses a model group only "
    "when its recorded semantic audit passed every gate. F, χ² and p-value-only conversions "
    "are off until you include them and "
    "give a direction (+ or −).")


def _item(text: str, editable: bool = False) -> QTableWidgetItem:
    item = QTableWidgetItem(str(text))
    flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
    if editable:
        flags |= Qt.ItemFlag.ItemIsEditable
    item.setFlags(flags)
    return item


def _how(row: dict) -> str:
    parts = [row.get("uncertainty_source") or (f"conversion:{row['route']}" if row.get("route") else "")]
    if row.get("assumptions"):
        parts.append(row["assumptions"])
    if row.get("status") == "converted" and row.get("reason"):
        parts.append(row["reason"])
    return " | ".join(p for p in parts if p)


class EffectReviewDialog(QDialog):
    """Table of effects with include / direction / group controls."""

    def __init__(self, rows: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review Effects & Groups")
        self.resize(1200, 700)
        self._rows = list(rows)
        layout = QVBoxLayout(self)
        intro = QLabel(_INTRO)
        intro.setWordWrap(True)
        layout.addWidget(intro)
        layout.addLayout(self._controls())
        self.table = QTableWidget(len(self._rows), len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setAccessibleName("Effects to review")
        for index, row in enumerate(self._rows):
            self._fill_row(index, row)
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_count()

    def _controls(self) -> QHBoxLayout:
        controls = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(FILTERS)
        self.filter_combo.setAccessibleName("Show effects")
        self.filter_combo.currentIndexChanged.connect(self.apply_filter)
        controls.addWidget(QLabel("Show:"))
        controls.addWidget(self.filter_combo)
        suggest_btn = QPushButton("Apply suggested groups")
        suggest_btn.setToolTip("Copy each suggested group into Group where Group is empty.")
        suggest_btn.clicked.connect(self.apply_suggestions)
        controls.addWidget(suggest_btn)
        self.count_label = QLabel("")
        controls.addWidget(self.count_label, 1)
        return controls

    def _fill_row(self, index: int, row: dict) -> None:
        include = _item("")
        include.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        include.setCheckState(Qt.CheckState.Checked if row.get("include") else Qt.CheckState.Unchecked)
        self.table.setItem(index, COL_INCLUDE, include)
        self.table.setItem(index, COL_PAPER, _item(row.get("title") or row.get("article_id", "")))
        self.table.setItem(index, COL_STAT, _item(row.get("test_text", "")))
        self.table.setItem(index, COL_EFFECT, _item(row.get("effect_used", "")))
        self.table.setItem(index, COL_HOW, _item(_how(row)))
        direction = QComboBox()
        direction.addItems(DIRECTIONS)
        direction.setCurrentText(row.get("direction") or "")
        direction.setEnabled(not row.get("sign_known", True))
        direction.setToolTip("Only needed when the statistic carries no sign (F, χ², p-value only).")
        self.table.setCellWidget(index, COL_DIRECTION, direction)
        suggested = _item(row.get("suggested_group", ""))
        if row.get("suggested_group_audit_id"):
            suggested.setToolTip(
                f"Semantic audit {row['suggested_group_audit_id']}: "
                f"{row.get('suggested_group_validation') or 'validation status not recorded'}"
            )
        self.table.setItem(index, COL_SUGGESTED, suggested)
        self.table.setItem(index, COL_GROUP, _item(row.get("group", ""), editable=True))

    def _included(self, index: int) -> bool:
        return self.table.item(index, COL_INCLUDE).checkState() == Qt.CheckState.Checked

    def _group(self, index: int) -> str:
        return self.table.item(index, COL_GROUP).text().strip()

    def _direction(self, index: int) -> str:
        return self.table.cellWidget(index, COL_DIRECTION).currentText()

    def _matches(self, index: int, choice: str) -> bool:
        row = self._rows[index]
        if choice == FILTERS[1]:
            return not row.get("route")
        if choice == FILTERS[2]:
            return bool(row.get("route"))
        if choice == FILTERS[3]:
            return not row.get("sign_known", True) and not self._direction(index)
        if choice == FILTERS[4]:
            return not self._group(index)
        return True

    def apply_filter(self, *_):
        choice = self.filter_combo.currentText()
        for index in range(len(self._rows)):
            self.table.setRowHidden(index, not self._matches(index, choice))
        self._update_count()

    def apply_suggestions(self):
        for index in range(len(self._rows)):
            suggested = self.table.item(index, COL_SUGGESTED).text().strip()
            if suggested and not self._group(index):
                self.table.item(index, COL_GROUP).setText(suggested)
        self._update_count()

    def _update_count(self):
        included = sum(self._included(i) for i in range(len(self._rows)))
        grouped = sum(bool(self._included(i) and self._group(i)) for i in range(len(self._rows)))
        self.count_label.setText(
            f"{len(self._rows)} effects; {included} included; {grouped} included with a group")

    def _reversed(self, index: int) -> bool:
        """A group taken from a validated model proposal keeps its direction."""
        row, group = self._rows[index], self._group(index)
        if group and group == row.get("suggested_group"):
            return bool(row.get("suggested_reversed"))
        return bool(row.get("reversed")) and group == row.get("group")

    def decisions(self) -> dict[str, dict]:
        """``{effect_id: {"include", "group", "direction", "reversed"}}`` for every row."""
        return {row["effect_id"]: {"include": self._included(i), "group": self._group(i),
                                   "direction": self._direction(i), "reversed": self._reversed(i)}
                for i, row in enumerate(self._rows)}


def latest_review_path(config):
    """The newer of the meta-analysis and Stage 2c review files, if any."""
    from pathlib import Path

    from process.meta_analysis_groups import REVIEW_FILE

    candidates = [Path(config.output_dir) / "meta_analysis_evidence" / REVIEW_FILE,
                  Path(config.claims_dir) / REVIEW_FILE]
    existing = [p for p in candidates if p.exists()]
    return max(existing, key=lambda p: p.stat().st_mtime) if existing else None


def review_effects(window) -> bool:
    """Open the review table for *window*'s project; save decisions on Save."""
    import json

    from process.meta_analysis_groups import load_decisions, save_decisions

    config = window.pipeline_config
    path = latest_review_path(config)
    if path is None:
        window._log("No effects to review yet: run Stage 2c (Statistical Extraction) "
                    "or the meta-analysis first.", "WARNING")
        return False
    try:
        rows = json.loads(path.read_text(encoding="utf-8")).get("effects") or []
    except (OSError, ValueError) as exc:
        window._log(f"Could not read {path}: {exc}", "ERROR")
        return False
    if not rows:
        window._log(f"No calculable or convertible effects in {path}.", "WARNING")
        return False
    dialog = EffectReviewDialog(rows, window if isinstance(window, QWidget) else None)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return False
    # Only rows changed in the dialog are decisions; untouched rows keep their
    # defaults, so a later default (e.g. a semantically validated model group) still applies.
    shown = {row["effect_id"]: {"include": bool(row.get("include")), "group": row.get("group") or "",
                                "direction": row.get("direction") or "",
                                "reversed": bool(row.get("reversed"))} for row in rows}
    chosen = {eid: d for eid, d in dialog.decisions().items() if d != shown[eid]}
    saved = save_decisions(config, {**load_decisions(config), **chosen})
    included = sum(1 for d in chosen.values() if d["include"])
    grouped = sum(1 for d in chosen.values() if d["include"] and d["group"])
    window._log(f"Effect decisions saved to {saved}: {len(chosen)} changed; "
                f"{included} included, {grouped} with a group.")
    return True
