"""Separate, nonmodal review of saved Stage 6 silhouette selection."""
from __future__ import annotations

import json
import math
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QHeaderView, QHBoxLayout, QSpinBox, QPushButton,
)


STOP_LABELS = {
    "search_cap": "Reached the search limit",
    "monotonic_decay": "Three consecutive score decreases from K=2",
    "noise_floor": "A candidate cluster fell below the minimum article count",
    "too_few_articles": "Too few articles for the automatic search range",
    "sklearn_unavailable": "Silhouette scoring was unavailable",
    "no_valid_candidates": "No eligible candidate could be scored",
}
STATUS_LABELS = {
    "eligible": "Eligible", "below_noise_floor": "Below minimum size",
    "collapsed_clusters": "Fewer clusters formed than requested",
    "unscorable": "Could not score",
}


def load_selection(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("silhouette_scores"), list):
        raise ValueError("Expected a Stage 6 k_selection.json with silhouette_scores.")
    seen = set()
    for row in data["silhouette_scores"]:
        if not isinstance(row, dict):
            raise ValueError("Invalid silhouette score entry.")
        k, score = row.get("k"), row.get("score")
        if (type(k) is not int or k < 2 or k in seen
                or type(score) not in (int, float) or not math.isfinite(score)
                or not -1 <= score <= 1):
            raise ValueError("Silhouette scores need distinct integer K values and finite scores from -1 to 1.")
        seen.add(k)
    return data


class SilhouetteReviewDialog(QDialog):
    k_chosen = Signal(int)

    def __init__(self, selection: dict, source: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stage 6 — Silhouette Score Trajectory")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(1100, 840)
        self.setMinimumSize(850, 700)
        layout = QVBoxLayout(self)
        source_label = QLabel(f"Saved result: {source}")
        source_label.setTextFormat(Qt.TextFormat.PlainText)
        source_label.setWordWrap(True)
        source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(source_label)
        scores = sorted(selection.get("silhouette_scores", []), key=lambda row: row["k"])
        chosen = selection.get("chosen_k")
        parts = [f"Silhouette recommendation: K={chosen}" if chosen is not None else "Selected K: unavailable"]
        for key, label in (("article_count", "Articles"), ("search_max_k", "Maximum K"),
                           ("min_cluster_size", "Small-cluster advisory threshold")):
            if key in selection:
                parts.append(f"{label}: {selection[key]}")
        if selection.get("declining_from_k2"):
            parts.append("Scores decline from K=2; the full requested range was still evaluated.")
        stop = selection.get("stop_reason")
        parts.append("Stopped: " + STOP_LABELS.get(stop, str(stop or "not recorded in this saved result")))
        if selection.get("structure_warning"):
            parts.append(str(selection["structure_warning"]))
        self.summary = QLabel("\n".join(parts))
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)

        self.figure = Figure(figsize=(8, 4), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(121)
        ax = self.axes
        ax.set_title("Silhouette score by number of clusters")
        ax.set_xlabel("Number of clusters (K)")
        ax.set_ylabel("Mean silhouette score (higher is better)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=.2)
        if scores:
            # Missing K values are gaps, not invented scores or interpolated points.
            lookup = {row["k"]: row["score"] for row in scores}
            ks = list(range(min(lookup), max(lookup) + 1))
            ax.plot(ks, [lookup.get(k, float("nan")) for k in ks],
                    "o-", color="#24658f", label="Scored candidates")
            if chosen in lookup:
                ax.scatter([chosen], [lookup[chosen]], s=150, marker="*", color="#b55400",
                           zorder=4, label=f"Silhouette maximum K={chosen}")
                ax.axvline(chosen, color="#b55400", linestyle=":", alpha=.6)
            ax.set_xticks(ks)
            ax.set_xlim(min(ks)-.5, max(ks)+.5)
            ax.legend(loc="best")
        else:
            message = ("Fixed cluster count — no silhouette sweep was run."
                       if selection.get("mode") == "fixed" else
                       "No eligible silhouette scores are available for this run.")
            ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
        self.elbow_axes = self.figure.add_subplot(122)
        elbow = self.elbow_axes
        elbow.set_title("Elbow plot")
        elbow.set_xlabel("Number of clusters (K)")
        elbow.set_ylabel("Inertia (within-cluster sum of squares)")
        elbow.xaxis.set_major_locator(MaxNLocator(integer=True))
        elbow.grid(True, alpha=.2)
        inertia = selection.get("inertia_scores") or []
        if inertia:
            elbow.plot([r["k"] for r in inertia], [r["inertia"] for r in inertia], "o-", color="#24658f")
        else:
            elbow.text(.5, .5, "Inertia was not recorded.\nCalculate Cluster Scores to obtain an elbow plot.",
                       ha="center", va="center", transform=elbow.transAxes, wrap=True)
        self.canvas.setMinimumHeight(260)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 1)
        note = QLabel("Review both curves and choose K. Small clusters and score declines are advisory. "
                      "Older saved runs may end early; calculate scores again to see the full requested range.")
        note.setWordWrap(True)
        layout.addWidget(note)
        rows = selection.get("candidates") or [dict(row, status="eligible") for row in scores]
        self.table = QTableWidget(len(rows), 4)
        self.table.setHorizontalHeaderLabels(["K", "Silhouette", "Smallest cluster", "Result"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(180)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        for index, row in enumerate(rows):
            score = row.get("score")
            sizes = row.get("cluster_sizes") or []
            status = STATUS_LABELS.get(row.get("status"), str(row.get("status", "Unknown")))
            if row.get("k") == chosen and row.get("status") == "eligible":
                status += " — selected"
            if row.get("small_cluster_warning"):
                status += " — small cluster (advisory)"
            values = [str(row.get("k", "")), f"{score:.4f}" if score is not None else "—",
                      str(min(sizes)) if sizes else "Not recorded", status]
            for col, value in enumerate(values):
                self.table.setItem(index, col, QTableWidgetItem(value))
        layout.addWidget(self.table)
        choice = QHBoxLayout()
        choice.addWidget(QLabel("Your K:"))
        self.k_choice = QSpinBox()
        self.k_choice.setRange(2, max(2, int(selection.get("article_count") or selection.get("search_max_k") or 50)))
        self.k_choice.setValue(max(2, chosen or 2))
        choice.addWidget(self.k_choice)
        self.use_k_button = QPushButton("Use K for Next Synthesis")
        self.use_k_button.clicked.connect(lambda: self.k_chosen.emit(self.k_choice.value()))
        choice.addWidget(self.use_k_button)
        choice.addStretch()
        layout.addLayout(choice)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.close)
        layout.addWidget(close)
        self.canvas.draw()
