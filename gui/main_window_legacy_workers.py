"""Legacy fulltext-retrieval and text-scaling workers and their callbacks.

Split from gui/main_window.py; mixed into MainWindow.
"""
from __future__ import annotations

import copy

from PySide6.QtWidgets import QMessageBox

from gui.workers import FullTextRetrievalWorker, TextScalingWorker


class LegacyWorkersMixin:
    """Legacy fulltext-retrieval and text-scaling workers and their callbacks."""

    def _start_fulltext_retrieval(self):
        """Run corpus full-text retrieval."""
        self._log("═" * 60)
        self._log("FULL-TEXT RETRIEVAL")
        self._log("═" * 60)
        if not self.articles:
            self._log("No articles loaded.", "ERROR")
            return
        self._log(f"Checking OA status for {len(self.articles)} articles...")
        self.ft_run_btn.setEnabled(False)
        self.ft_cancel_btn.setEnabled(True)
        self._ft_worker = FullTextRetrievalWorker(
            articles=self.articles,
            config=copy.deepcopy(self.pipeline_config),
        )
        self._ft_worker.status.connect(lambda msg: self._log(msg))
        self._ft_worker.finished_retrieval.connect(self._on_ft_finished)
        self._ft_worker.error.connect(lambda e: self._log(f"FT retrieval error: {e}", "ERROR"))
        self._ft_worker.start()

    def _on_ft_finished(self, success_count: int, total: int, json_path: str):
        self.ft_run_btn.setEnabled(True)
        self.ft_cancel_btn.setEnabled(False)
        self._log(f"Full-text retrieval: {success_count}/{total} downloaded")
        if json_path:
            self._log(f"Output: {json_path}")

    def _start_text_scaling(self):
        """Run Wordfish text scaling + stylometric profiling."""
        self._log("═" * 60)
        self._log("TEXT SCALING & STYLOMETRY")
        self._log("═" * 60)
        if not self.articles:
            self._log("No articles loaded.", "ERROR")
            return
        min_words = self.wf_min_words.value() if hasattr(self, 'wf_min_words') else 50
        max_feat = self.wf_max_features.value() if hasattr(self, 'wf_max_features') else 5000
        self._log(f"Articles: {len(self.articles)}, min_words={min_words}, max_features={max_feat}")
        self._scaling_worker = TextScalingWorker(
            articles=self.articles,
            components=self.hypothesis_components,
            min_doc_words=min_words,
            max_features=max_feat,
        )
        self.wf_run_btn.setEnabled(False)
        self.wf_cancel_btn.setEnabled(True)
        self._scaling_worker.status.connect(lambda msg: self._log(msg))
        self._scaling_worker.finished_scaling.connect(self._on_scaling_finished)
        self._scaling_worker.cancelled.connect(self._on_scaling_cancelled)
        self._scaling_worker.error.connect(self._on_scaling_error)
        self._scaling_worker.start()

    # ── Search/full-text cancel methods ───────────────────────────────

    def _cancel_work(self):
        # "Fetch Articles" runs the corpus full-text download via the
        # pipeline worker, so cancelling that worker cancels the fetch.
        if self._pipeline_worker:
            self._pipeline_worker.cancel()
            self._log("Fetch cancelled.")
        self.fetch_btn.setEnabled(True)
        self.cancel_fetch_btn.setEnabled(False)
        self.search_progress.setVisible(False)

    def _cancel_ft(self):
        if self._ft_worker:
            self._ft_worker.cancel()
        self.ft_run_btn.setEnabled(True)
        self.ft_cancel_btn.setEnabled(False)

    def _cancel_scaling(self):
        if self._scaling_worker:
            self._scaling_worker.cancel()
            self.wf_cancel_btn.setEnabled(False)
            self._log("Cancelling text scaling...")

    # ── Search/full-text signal handler slots ─────────────────────────

    def _on_error(self, msg):
        self._log(msg, level="ERROR")
        QMessageBox.warning(self, "Error", msg)

    def _on_scaling_finished(self, model, profiles: dict, stats: dict):
        self._wordfish_model = model
        self._stylometry_profiles = profiles
        self._corpus_stats = stats
        self.session.wordfish_model = model
        self.session.stylometry_profiles = profiles
        self.session.corpus_statistics = stats
        self.wf_run_btn.setEnabled(True)
        self.wf_cancel_btn.setEnabled(False)
        self._populate_wf_table(model)
        self._populate_sty_table(profiles)
        self._update_sty_kpis(stats)
        self._log(
            f"Text scaling complete: {len(profiles)} profiles, "
            f"{stats.get('n_features', 0)} features"
        )

    def _on_scaling_cancelled(self):
        self.wf_run_btn.setEnabled(True)
        self.wf_cancel_btn.setEnabled(False)
        self._log("Text scaling cancelled.", "WARNING")

    def _on_scaling_error(self, message: str):
        self.wf_run_btn.setEnabled(True)
        self.wf_cancel_btn.setEnabled(False)
        self._log(f"Scaling error: {message}", "ERROR")
