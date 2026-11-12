"""Human relevance review: queue, table editing, save, and model comparison.

Split from gui/main_window.py; mixed into MainWindow.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QTableWidget, QTableWidgetItem

from core.io import atomic_write_json
from process.human_relevance_review import _score_from_row

_MODEL_SCORE_FIELDS = ("model_score", "model_quote")
_CORRECTED = "model_score_human_corrected"
_MODEL_VALUE = "model_score_model_value"
_CORRECTION_KEYS = (_CORRECTED, _MODEL_VALUE)
_TITLE_KEY_CHARS = 120


def format_review_metric(value, suffix: str = "") -> str:
    """Agreement metric for display; ``None`` (not applicable, e.g. ordinal
    metrics on binary cascade scores) reads as "N/A" (as `format_adjacent_agreement`) rather than "None"."""
    return "N/A" if value is None else f"{value}{suffix}"


def _review_row_keys(row: dict) -> list[str]:
    """Lookup keys for matching review rows: article id, then title prefix."""
    keys = [str(row.get("article_id") or "")]
    title = str(row.get("title") or "").strip()
    if title:
        keys.append(title[:_TITLE_KEY_CHARS])
    return [key for key in keys if key]


def _with_model_scores(record: dict, source: dict | None) -> dict:
    """Copy refreshed model-score fields onto a saved record, nothing else."""
    if source is None or source.get("model_score_human_corrected"):
        return record
    if record.get("model_score_human_corrected"):
        return record
    return {**record, **{field: source.get(field) for field in _MODEL_SCORE_FIELDS}}


def merge_model_scores_into_saved_review(saved: dict, rows: list[dict]) -> dict:
    """Return the saved review with model scores refreshed from ``rows``.

    Human decisions (``human_score``, ``human_verified``, corrections) and the
    saved record set are never changed: rows absent from the saved review are
    not added, and saved rows are only given new model-score fields.
    """
    by_key: dict[str, dict] = {}
    for row in rows:
        for key in _review_row_keys(row):
            by_key.setdefault(key, row)
    records = []
    for record in saved.get("records", []):
        match = next(
            (by_key[key] for key in _review_row_keys(record) if key in by_key), None
        )
        records.append(_with_model_scores(record, match))
    return {
        **saved,
        "records": records,
        "reviewed_count": sum(1 for r in records if r.get("human_score") is not None),
        "verified_count": sum(1 for r in records if r.get("human_verified")),
    }


def _read_saved_human_review(path: Path) -> dict:
    import json as _json

    with open(path, encoding="utf-8") as f:
        saved = _json.load(f)
    records = saved.get("records", []) if isinstance(saved, dict) else None
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise ValueError(f"Unexpected human review format in {path}")
    return saved


def _with_model_score(row: dict, model_score) -> dict:
    """``row`` with the table's model score; a hand correction is flagged.

    The model's own value is kept in ``model_score_model_value`` so that
    reverting the cell to it clears the flag (audit 19-M2). A legacy row
    flagged without that value stays flagged.
    """
    base = {key: value for key, value in row.items() if key not in _CORRECTION_KEYS}
    corrected = bool(row.get(_CORRECTED))
    if corrected and _MODEL_VALUE not in row:
        return {**base, "model_score": model_score, _CORRECTED: True}
    model_value = row[_MODEL_VALUE] if corrected else row.get("model_score")
    if model_score == model_value:
        return {**base, "model_score": model_score}
    return {**base, "model_score": model_score, _CORRECTED: True, _MODEL_VALUE: model_value}


def _agreement_with_corrections_reported(rows: list[dict], scored: list[dict],
                                         threshold: int) -> dict:
    """Human/model agreement plus how many model scores were corrected by hand.

    Hand-corrected model scores are not model output, so the summary also
    gives the agreement over the uncorrected rows only (audit 19-M2).
    """
    from process.human_relevance_review import compare_human_to_model

    summary = compare_human_to_model(rows, scored, threshold=threshold)
    corrected = sum(
        1 for row in rows
        if row.get(_CORRECTED) and _score_from_row(row, "human_score") is not None
    )
    uncorrected = [row for row in rows if not row.get(_CORRECTED)]
    without = compare_human_to_model(uncorrected, scored, threshold=threshold)
    without.pop("comparisons", None)
    return {**summary, "model_score_corrected": corrected, "excluding_corrected": without}


def _corrected_note(summary: dict) -> str:
    """Status-line disclosure of hand-corrected model scores ("" when none)."""
    corrected = summary.get("model_score_corrected", 0)
    if not corrected:
        return ""
    without = summary.get("excluding_corrected") or {}
    noun = "score" if corrected == 1 else "scores"
    return (f"; {corrected} model {noun} corrected by hand (without them: "
            f"{without.get('matched', 0)} matched, "
            f"wκ={format_review_metric(without.get('score_weighted_kappa'))})")


def _collected_human_score(value: str, row: dict):
    """Score from the combo; keep a saved value the combo cannot display."""
    if value:
        return int(value)
    original = row.get("human_score")
    if original is not None and _score_from_row(row, "human_score") is None:
        return original
    return None


class HumanRelevanceMixin:
    """Human relevance review: queue, table editing, save, and model comparison."""

    # ── Sysys individual stage launchers ─────────────────────────────

    def _score_corpus_path(self) -> Path:
        raw = self.score_corpus_path.text().strip()
        return Path(raw) if raw else self.pipeline_config.corpus_dir / "corpus.json"

    def _human_review_source_path(self) -> Path:
        """Papers to sample the human queue from.

        Stage 2's ``scored_all.json`` carries the model scores the stratified
        sample needs; ``relevant.json`` is only the kept subset, so it is never
        used. Before scoring, fall back to the Stage 2 input corpus.
        """
        scored = self.pipeline_config.claims_dir / "scored_all.json"
        return scored if scored.exists() else self._score_corpus_path()

    def _human_relevance_artifact_path(self) -> Path:
        return self.pipeline_config.data_dir / "validation" / "human_relevance_review.json"

    def _human_relevance_summary_path(self) -> Path:
        return self.pipeline_config.data_dir / "validation" / "human_relevance_agreement.json"

    def _load_json_list(self, path: Path) -> list[dict]:
        import json as _json

        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {path}")
        return data

    def _load_existing_human_relevance_review(self, cfg) -> None:
        """Restore the current project's saved review without clobbering edits."""
        import json as _json

        review_path = cfg.data_dir / "validation" / "human_relevance_review.json"
        try:
            resolved_path = review_path.resolve()
        except OSError:
            resolved_path = review_path
        if self._loaded_human_review_path == resolved_path:
            return

        self._loaded_human_review_path = resolved_path
        if not review_path.exists():
            self._human_review_artifact = None
            self._populate_human_relevance_table([])
            self.human_review_status.setText("No human queue loaded")
            return

        try:
            with open(review_path, encoding="utf-8") as f:
                artifact = _json.load(f)
            if not isinstance(artifact, dict):
                raise ValueError("Expected a review artifact object")
            rows = artifact.get("records", [])
            if not isinstance(rows, list) or not all(
                isinstance(row, dict) for row in rows
            ):
                raise ValueError("Expected review records to be a list of objects")
            self._human_review_artifact = artifact
            self._populate_human_relevance_table(rows)
            reviewed = artifact.get(
                "reviewed_count",
                sum(1 for row in rows if row.get("human_score") is not None),
            )
            verified = artifact.get(
                "verified_count",
                sum(1 for row in rows if row.get("human_verified")),
            )
            self.human_review_status.setText(
                f"Loaded {reviewed}/{len(rows)} human scores; "
                f"{verified} human verified"
            )
            self._log(f"Loaded human relevance review: {review_path}")
        except Exception as exc:
            self._human_review_artifact = None
            self._populate_human_relevance_table([])
            self.human_review_status.setText("Could not load saved human review")
            self._log(f"Could not load human relevance review: {exc}", "ERROR")

    def _set_human_relevance_editing(self, enabled: bool) -> None:
        """Toggle direct editing for the correctable review-table cells."""
        triggers = (
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.SelectedClicked
            | QTableWidget.EditTrigger.EditKeyPressed
        )
        self.human_review_table.setEditTriggers(
            triggers if enabled else QTableWidget.EditTrigger.NoEditTriggers
        )
        for row_index in range(self.human_review_table.rowCount()):
            for column in range(2, 7):
                item = self.human_review_table.item(row_index, column)
                if item is None:
                    continue
                flags = item.flags()
                if enabled:
                    flags |= Qt.ItemFlag.ItemIsEditable
                else:
                    flags &= ~Qt.ItemFlag.ItemIsEditable
                item.setFlags(flags)

    def _populate_human_relevance_table(self, rows: list[dict]) -> None:
        self._human_review_rows = rows
        self.human_review_table.setSortingEnabled(False)
        self.human_review_table.setRowCount(0)
        for row_index, row in enumerate(rows):
            self.human_review_table.insertRow(row_index)

            score_combo = QComboBox()
            score_combo.addItems(["", "0", "1", "2", "3", "4", "5"])
            human_score = _score_from_row(row, "human_score")
            if human_score is not None:
                score_combo.setCurrentText(str(human_score))
            self.human_review_table.setCellWidget(row_index, 0, score_combo)

            verified_item = QTableWidgetItem()
            verified_item.setFlags(
                (verified_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                & ~Qt.ItemFlag.ItemIsEditable
            )
            verified_item.setCheckState(
                Qt.CheckState.Checked
                if bool(row.get("human_verified"))
                else Qt.CheckState.Unchecked
            )
            verified_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            verified_item.setToolTip("Mark this row as checked by a human reviewer")
            verified_item.setData(Qt.ItemDataRole.UserRole, row_index)
            self.human_review_table.setItem(row_index, 1, verified_item)

            values = [
                row.get("title", ""),
                row.get("year", ""),
                row.get("model_score", ""),
                row.get("article_id", ""),
                row.get("abstract", ""),
            ]
            for offset, value in enumerate(values, start=2):
                item = QTableWidgetItem(str(value if value is not None else ""))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setData(Qt.ItemDataRole.UserRole, row_index)
                self.human_review_table.setItem(row_index, offset, item)
        self._set_human_relevance_editing(self.human_review_edit_cells.isChecked())
        self.human_review_table.setSortingEnabled(False)
        self.human_review_table.resizeColumnToContents(0)
        self.human_review_table.resizeColumnToContents(1)
        self.human_review_table.resizeColumnToContents(3)
        self.human_review_table.resizeColumnToContents(4)
        self.human_review_table.resizeColumnToContents(5)

    def _collect_human_relevance_rows(self) -> list[dict]:
        def _text(row_index: int, column: int) -> str:
            item = self.human_review_table.item(row_index, column)
            return item.text().strip() if item is not None else ""

        def _number_or_text(value: str, *, integer_only: bool = False):
            if not value:
                return None
            try:
                return int(value) if integer_only else float(value)
            except ValueError:
                return value

        rows: list[dict] = []
        for row_index, row in enumerate(self._human_review_rows):
            model_score = _number_or_text(_text(row_index, 4))
            if isinstance(model_score, float) and model_score.is_integer():
                model_score = int(model_score)
            updated = _with_model_score(row, model_score)
            widget = self.human_review_table.cellWidget(row_index, 0)
            if isinstance(widget, QComboBox):
                updated["human_score"] = _collected_human_score(
                    widget.currentText().strip(), row
                )
            verified_item = self.human_review_table.item(row_index, 1)
            updated["human_verified"] = bool(
                verified_item is not None
                and verified_item.checkState() == Qt.CheckState.Checked
            )
            updated["title"] = _text(row_index, 2)
            updated["year"] = _number_or_text(
                _text(row_index, 3), integer_only=True
            )
            updated["article_id"] = _text(row_index, 5)
            updated["abstract"] = _text(row_index, 6)
            rows.append(updated)
        self._human_review_rows = rows
        return rows

    def _build_human_relevance_queue(self) -> None:
        from process.human_relevance_review import (
            build_review_queue,
            recommended_sample_size,
        )

        corpus_path = self._human_review_source_path()
        if not corpus_path.exists():
            self._log("No corpus found for human calibration queue. Run search first.", "ERROR")
            return
        if corpus_path.name != "scored_all.json":
            self._log(
                "Corpus not scored yet: the human queue is a plain random sample. "
                "Run Score & Filter first to stratify it by model decision.",
                "WARNING",
            )
        try:
            papers = self._load_json_list(corpus_path)
            requested = self.human_review_sample.value()
            sample_size = requested or recommended_sample_size(len(papers))
            artifact = build_review_queue(
                papers,
                sample_size=sample_size,
                seed=42,
                source_path=str(corpus_path),
            )
            self._human_review_artifact = artifact
            try:
                self._loaded_human_review_path = (
                    self._human_relevance_artifact_path().resolve()
                )
            except OSError:
                self._loaded_human_review_path = self._human_relevance_artifact_path()
            self._populate_human_relevance_table(artifact.get("records", []))
            self.human_review_status.setText(
                f"Queue ready: {artifact.get('sample_size', 0)} of "
                f"{artifact.get('corpus_size', 0)} papers"
            )
            self._log(
                "Human relevance queue ready: "
                f"{artifact.get('sample_size', 0)} papers from {corpus_path}"
            )
        except Exception as exc:
            self._log(f"Could not build human relevance queue: {exc}", "ERROR")

    def _save_human_relevance_review(self) -> None:
        from core.io import atomic_write_json

        if not self._human_review_artifact:
            self._log("No human relevance queue to save. Build a queue first.", "ERROR")
            return
        rows = self._collect_human_relevance_rows()
        artifact = dict(self._human_review_artifact)
        artifact["records"] = rows
        artifact["reviewed_count"] = sum(1 for row in rows if row.get("human_score") is not None)
        artifact["verified_count"] = sum(
            1 for row in rows if row.get("human_verified")
        )
        out_path = self._human_relevance_artifact_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_path, artifact)
        self._human_review_artifact = artifact
        try:
            self._loaded_human_review_path = out_path.resolve()
        except OSError:
            self._loaded_human_review_path = out_path
        self.human_review_status.setText(
            f"Saved {artifact['reviewed_count']}/{len(rows)} human scores; "
            f"{artifact['verified_count']} human verified"
        )
        self._log(f"Human relevance review saved: {out_path}")

    def _scored_relevance_path(self) -> Path | None:
        candidates = [
            self.pipeline_config.claims_dir / "scored_all.json",
            self.pipeline_config.claims_dir / "relevant.json",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _refresh_human_relevance_model_scores(self) -> None:
        if not self._human_review_rows:
            return
        scored_path = self._scored_relevance_path()
        if scored_path is None:
            return
        try:
            from process.human_relevance_review import article_id

            scored = self._load_json_list(scored_path)
            by_id: dict[str, dict] = {}
            for index, paper in enumerate(scored):
                key = article_id(paper, index)
                by_id[key] = paper
                title = str(paper.get("title") or "").strip()
                if title:
                    by_id.setdefault(title[:120], paper)
            updated = []
            for row in self._collect_human_relevance_rows():
                match = by_id.get(str(row.get("article_id") or ""))
                if match is None:
                    match = by_id.get(str(row.get("title") or "")[:120])
                if match is not None:
                    row = dict(row)
                    if not row.get("model_score_human_corrected"):
                        row["model_score"] = match.get("relevance_score")
                        row["model_quote"] = match.get("relevance_quote", "")
                updated.append(row)
            self._populate_human_relevance_table(updated)
            self._persist_human_relevance_model_scores(updated)
        except Exception as exc:
            self._log(f"Could not refresh human queue model scores: {exc}", "ERROR")

    def _persist_human_relevance_model_scores(
        self, rows: list[dict]
    ) -> None:
        """Write refreshed model scores into the saved review artifact.

        Keeps the on-disk queue in step with the latest scoring run so a
        reopened project shows the same Model Score column as the table. Only
        model-score fields of records already saved are updated: the saved
        human decisions are never replaced by the in-memory queue, which may
        be a rebuilt, unscored queue or hold unsaved edits.
        """
        if not self._human_review_artifact:
            return
        out_path = self._human_relevance_artifact_path()
        if not out_path.exists():
            return
        try:
            saved = _read_saved_human_review(out_path)
            atomic_write_json(out_path, merge_model_scores_into_saved_review(saved, rows))
        except (OSError, ValueError) as exc:
            self._log(f"Could not save refreshed model scores: {exc}", "ERROR")

    def _load_saved_review_for_compare(self) -> bool:
        """Load the saved review when none is in memory; False when unusable."""
        saved = self._human_relevance_artifact_path()
        if not saved.exists():
            self._log("No human relevance queue to compare. Build or save one first.", "ERROR")
            return False
        try:
            artifact = _read_saved_human_review(saved)
        except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
            self._log(f"Could not read saved human review {saved}: {exc}", "ERROR")
            return False
        self._human_review_artifact = artifact
        try:
            self._loaded_human_review_path = saved.resolve()
        except OSError:
            self._loaded_human_review_path = saved
        self._populate_human_relevance_table(artifact.get("records", []))
        return True

    def _compare_human_relevance_review(self) -> None:
        from core.agreement import format_adjacent_agreement
        from core.io import atomic_write_json

        if not self._human_review_artifact and not self._load_saved_review_for_compare():
            return

        scored_path = self._scored_relevance_path()
        if scored_path is None:
            self._log("No model scored output found yet. Run Score & Filter first.", "ERROR")
            return
        try:
            rows = self._collect_human_relevance_rows()
            scored = self._load_json_list(scored_path)
            summary = _agreement_with_corrections_reported(
                rows, scored, self.score_threshold.value()
            )
            out_path = self._human_relevance_summary_path()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(out_path, summary)
            self._refresh_human_relevance_model_scores()
            self.human_review_status.setText(
                "Human vs model: "
                f"{summary.get('human_verified', 0)} verified, "
                f"{summary.get('matched', 0)} matched, "
                f"Percent Adjacent Agreement (within 1): {format_adjacent_agreement(summary)}, "
                f"threshold agreement "
                f"{format_review_metric(summary.get('threshold_agreement_pct'), '%')}, "
                f"wκ={format_review_metric(summary.get('score_weighted_kappa'))}"
                f"{_corrected_note(summary)}"
            )
            if summary["model_score_corrected"]:
                self._log(
                    f"{summary['model_score_corrected']} reviewed row(s) use a model "
                    "score corrected by hand; agreement without them is in "
                    "'excluding_corrected' of the summary file.",
                    "WARNING",
                )
            self._log(
                "Human relevance agreement: "
                f"verified={summary.get('human_verified', 0)}, "
                f"{summary.get('matched', 0)} matched, "
                f"exact={format_review_metric(summary.get('exact_agreement_pct'), '%')}, "
                f"Percent Adjacent Agreement (within 1)={format_adjacent_agreement(summary)}, "
                f"threshold="
                f"{format_review_metric(summary.get('threshold_agreement_pct'), '%')}, "
                f"wκ={format_review_metric(summary.get('score_weighted_kappa'))}"
            )
            self._log(f"  Summary: {out_path}")
        except Exception as exc:
            self._log(f"Could not compare human relevance review: {exc}", "ERROR")
