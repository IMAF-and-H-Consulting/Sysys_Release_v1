"""
Network graph widget for condition–entity co-occurrence visualization.

Uses matplotlib for rendering and a simple spring-layout algorithm
implemented with numpy (no networkx dependency).
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)

from gui.theme import CHART_SEQUENCE, COLORS

logger = logging.getLogger(__name__)

# ── Spring-layout implementation (no networkx) ────────────────────────


def spring_layout(
    n_nodes: int,
    edges: list[tuple[int, int, float]],
    iterations: int = 100,
    k: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute a 2-D spring/force-directed layout.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    edges : list[tuple[int, int, float]]
        Edge list as (source_idx, target_idx, weight).
    iterations : int
        Number of simulation steps.
    k : float
        Ideal spring length (scales repulsion/attraction balance).
    seed : int
        Random seed for reproducible initial positions.

    Returns
    -------
    np.ndarray of shape (n_nodes, 2) — final positions.
    """
    if n_nodes == 0:
        return np.empty((0, 2))
    if n_nodes == 1:
        return np.array([[0.0, 0.0]])

    rng = np.random.RandomState(seed)
    pos = rng.randn(n_nodes, 2)

    # Optimal distance
    area = float(n_nodes)
    k_opt = k * np.sqrt(area / n_nodes)

    temperature = area * 0.1
    cooling = temperature / (iterations + 1)

    for _it in range(iterations):
        disp = np.zeros_like(pos)

        # Repulsion between all pairs
        for i in range(n_nodes):
            diff = pos[i] - pos  # (n_nodes, 2)
            dist = np.sqrt((diff ** 2).sum(axis=1))
            dist = np.where(dist < 0.01, 0.01, dist)  # avoid division by zero
            # Coulomb-like repulsion: F = k^2 / d
            force_mag = (k_opt ** 2) / dist
            force_mag[i] = 0.0  # no self-force
            fx = (diff[:, 0] / dist) * force_mag
            fy = (diff[:, 1] / dist) * force_mag
            disp[i, 0] += fx.sum()
            disp[i, 1] += fy.sum()

        # Attraction along edges
        for src, tgt, w in edges:
            diff = pos[src] - pos[tgt]
            dist = max(np.sqrt((diff ** 2).sum()), 0.01)
            # Hooke-like attraction: F = d^2 / k, scaled by weight
            force_mag = (dist ** 2) / k_opt * (0.5 + 0.5 * min(w, 5.0))
            direction = diff / dist
            disp[src] -= direction * force_mag
            disp[tgt] += direction * force_mag

        # Apply displacement with temperature limit
        disp_mag = np.sqrt((disp ** 2).sum(axis=1, keepdims=True))
        disp_mag = np.where(disp_mag < 0.01, 1.0, disp_mag)
        pos += (disp / disp_mag) * np.minimum(disp_mag, temperature)

        temperature -= cooling

    # Normalise to [0, 1] range
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    span = maxs - mins
    span = np.where(span < 1e-6, 1.0, span)
    pos = (pos - mins) / span

    return pos


# ── Network graph widget ──────────────────────────────────────────────


class NetworkGraphWidget(QWidget):
    """
    Interactive condition–entity co-occurrence network graph.

    Displays condition nodes (circles) and entity nodes (squares; diamonds
    for functional networks from the neuroscience pack),
    connected by edges weighted by co-occurrence frequency.
    Uses matplotlib for rendering with zoom/pan navigation.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._network_data: dict = {}
        self._node_names: list[str] = []
        self._node_ids: list[str] = []
        self._node_types: list[str] = []   # "condition" or "region"
        self._node_subtypes: list[str] = []  # "structural", "functional", ""
        self._node_counts: list[int] = []
        self._edges: list[tuple[int, int, float]] = []
        self._edge_types: list[str] = []
        self._positions: np.ndarray | None = None
        self._selected_node: int | None = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Matplotlib figure
        self._figure = Figure(figsize=(8, 6), facecolor=COLORS["surface"])
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)
        self._ax.set_facecolor(COLORS["surface"])
        self._ax.set_axis_off()
        layout.addWidget(self._canvas, 1)

        # Connect click event
        self._canvas.mpl_connect("button_press_event", self._on_click)

        # Status label
        self._status = QLabel("No network data loaded.")
        self._status.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 2px;")
        layout.addWidget(self._status)

    def set_network_data(self, network_analysis: dict):
        """
        Load network data from condition_networks.json (or legacy network output).

        Expected structure:
        {
            "condition_region_edges": [{"condition": str, "region": str, "weight": int}, ...],
            "condition_nodes": [{"name": str, "paper_count": int}, ...],
            "region_nodes": [{"name": str, "type": "structural"|"functional", "paper_count": int}, ...],
            ...
        }
        """
        self._network_data = network_analysis
        self._parse_network()
        self._compute_layout()
        self._draw_graph()

    def _parse_network(self):
        """Parse the network_analysis dict into internal node/edge lists."""
        self._node_names = []
        self._node_ids = []
        self._node_types = []
        self._node_subtypes = []
        self._node_counts = []
        self._edges = []
        self._edge_types = []

        data = self._network_data
        if isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list):
            self._parse_general_network(data)
            return

        name_to_idx: dict[str, int] = {}

        # Condition nodes
        for node in data.get("condition_nodes", []):
            name = node.get("name", "")
            if name and name not in name_to_idx:
                name_to_idx[name] = len(self._node_names)
                self._node_ids.append(name)
                self._node_names.append(name)
                self._node_types.append("condition")
                self._node_subtypes.append("")
                self._node_counts.append(node.get("paper_count", 1))

        # Region nodes
        for node in data.get("region_nodes", []):
            name = node.get("name", "")
            if name and name not in name_to_idx:
                name_to_idx[name] = len(self._node_names)
                self._node_ids.append(name)
                self._node_names.append(name)
                self._node_types.append("region")
                self._node_subtypes.append(node.get("type", ""))
                self._node_counts.append(node.get("paper_count", 1))

        # Edges
        for edge in data.get("condition_region_edges", []):
            cond = edge.get("condition", "")
            region = edge.get("region", "")
            weight = edge.get("weight", 1)
            src = name_to_idx.get(cond)
            tgt = name_to_idx.get(region)
            if src is not None and tgt is not None:
                self._edges.append((src, tgt, float(weight)))
                self._edge_types.append("co_occurs")

        n = len(self._node_names)
        ne = len(self._edges)
        self._status.setText(f"Network: {n} nodes, {ne} edges")

    def _parse_general_network(self, data: dict):
        """Parse generalized evidence-network JSON into internal lists."""
        id_to_idx: dict[str, int] = {}

        for node in data.get("nodes", []):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or node.get("label") or "")
            label = str(node.get("label") or node_id)
            if not node_id or node_id in id_to_idx:
                continue
            id_to_idx[node_id] = len(self._node_names)
            self._node_ids.append(node_id)
            self._node_names.append(label)
            self._node_types.append(str(node.get("type") or "concept"))
            self._node_subtypes.append(str(node.get("subtype") or ""))
            try:
                count = int(node.get("count") or 1)
            except (TypeError, ValueError):
                count = 1
            self._node_counts.append(max(count, 1))

        for edge in data.get("edges", []):
            if not isinstance(edge, dict):
                continue
            src = id_to_idx.get(str(edge.get("source") or ""))
            tgt = id_to_idx.get(str(edge.get("target") or ""))
            if src is None or tgt is None:
                continue
            try:
                weight = float(edge.get("weight") or 1)
            except (TypeError, ValueError):
                weight = 1.0
            self._edges.append((src, tgt, weight))
            self._edge_types.append(str(edge.get("type") or "related"))

        n = len(self._node_names)
        ne = len(self._edges)
        kind = data.get("kind", "general network")
        self._status.setText(f"{kind}: {n} nodes, {ne} edges")

    def _compute_layout(self):
        """Run spring layout on the parsed graph."""
        n = len(self._node_names)
        if n == 0:
            self._positions = np.empty((0, 2))
            return
        self._positions = spring_layout(n, self._edges, iterations=120, k=1.2)

    def _draw_graph(self):
        """Draw the full network graph on the matplotlib canvas."""
        ax = self._ax
        ax.clear()
        ax.set_axis_off()
        ax.set_facecolor(COLORS["surface"])

        if self._positions is None or len(self._positions) == 0:
            ax.text(
                0.5, 0.5, "No network data to display",
                ha="center", va="center", fontsize=11,
                color=COLORS["text_secondary"],
                transform=ax.transAxes,
            )
            self._canvas.draw_idle()
            return

        pos = self._positions
        max_weight = max((w for _, _, w in self._edges), default=1.0)

        # Draw edges
        for src, tgt, w in self._edges:
            alpha = 0.2 + 0.6 * (w / max_weight)
            lw = 0.5 + 2.5 * (w / max_weight)
            color = COLORS["text_secondary"]
            if self._selected_node is not None:
                if self._selected_node not in (src, tgt):
                    alpha *= 0.15
            ax.plot(
                [pos[src, 0], pos[tgt, 0]],
                [pos[src, 1], pos[tgt, 1]],
                color=color, alpha=alpha, linewidth=lw, zorder=1,
            )

        # Build community color map (if available)
        communities = (self._network_data or {}).get("communities", {})
        community_map = communities.get("communities", {}) if isinstance(communities, dict) else {}
        # Use tab10 palette for communities (up to 10 distinct colors)
        _tab10_cmap = getattr(plt.cm, "tab10", None)
        _TAB10 = _tab10_cmap.colors if _tab10_cmap is not None else CHART_SEQUENCE
        use_community_colors = bool(community_map)

        # Draw nodes
        for i, (name, ntype, subtype, count) in enumerate(zip(
            self._node_names, self._node_types, self._node_subtypes, self._node_counts
        )):
            x, y = pos[i]
            size = 30 + min(count, 20) * 8

            marker = self._marker_for_node(ntype, subtype)

            # Color: community-based if available, else type-based
            if use_community_colors and name in community_map:
                cid = community_map[name] % len(_TAB10)
                color = _TAB10[cid]
            else:
                color = self._color_for_node(ntype, subtype)

            alpha = 1.0
            if self._selected_node is not None and self._selected_node != i:
                # Dim nodes not connected to selected
                connected = any(
                    (s == self._selected_node and t == i) or
                    (t == self._selected_node and s == i)
                    for s, t, _ in self._edges
                )
                if not connected:
                    alpha = 0.2

            ax.scatter(
                x, y, s=size, c=color, marker=marker,
                alpha=alpha, edgecolors="black", linewidths=0.5, zorder=2,
            )
            # Label
            fontsize = 6 if len(name) > 15 else 7
            ax.annotate(
                name, (x, y),
                textcoords="offset points", xytext=(0, 6),
                ha="center", va="bottom", fontsize=fontsize,
                color=COLORS["text"], alpha=alpha,
            )

        # Legend
        if use_community_colors:
            n_comm = communities.get("n_communities", 0)
            mod = communities.get("modularity", 0)
            legend_items = [
                ax.scatter([], [], c=_TAB10[cid % len(_TAB10)], marker="o",
                           s=40, label=f"Community {cid}")
                for cid in range(min(n_comm, 8))
            ]
            # Add shape legend
            for ntype, subtype in dict.fromkeys(zip(self._node_types, self._node_subtypes)):
                legend_items.append(ax.scatter(
                    [], [], c="gray", marker=self._marker_for_node(ntype, subtype),
                    s=30, label=self._legend_label(ntype, subtype),
                ))
            ax.set_title(
                f"{n_comm} communities (Q={mod:.2f})",
                fontsize=8, color=COLORS["text_secondary"], pad=2,
            )
        else:
            legend_items = []
            seen_types: list[tuple[str, str]] = []
            for ntype, subtype in zip(self._node_types, self._node_subtypes):
                key = (ntype, subtype)
                if key not in seen_types:
                    seen_types.append(key)
            for ntype, subtype in seen_types[:8]:
                legend_items.append(
                    ax.scatter(
                        [],
                        [],
                        c=self._color_for_node(ntype, subtype),
                        marker=self._marker_for_node(ntype, subtype),
                        s=40,
                        label=self._legend_label(ntype, subtype),
                    )
                )
        ax.legend(
            handles=legend_items, loc="lower right", fontsize=7,
            framealpha=0.8, facecolor=COLORS["surface"],
        )

        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, 1.08)
        self._figure.tight_layout()
        self._canvas.draw_idle()

    def _on_click(self, event):
        """Handle click to select/deselect a node."""
        if event.inaxes != self._ax or self._positions is None:
            return
        if len(self._positions) == 0:
            return

        # Find nearest node
        click = np.array([event.xdata, event.ydata])
        dists = np.sqrt(((self._positions - click) ** 2).sum(axis=1))
        nearest = int(dists.argmin())

        if dists[nearest] < 0.05:
            if self._selected_node == nearest:
                self._selected_node = None
                self._status.setText(
                    f"Network: {len(self._node_names)} nodes, {len(self._edges)} edges"
                )
            else:
                self._selected_node = nearest
                name = self._node_names[nearest]
                ntype = self._node_types[nearest]
                count = self._node_counts[nearest]
                conn = sum(
                    1 for s, t, _ in self._edges
                    if s == nearest or t == nearest
                )
                edge_types = sorted({
                    self._edge_types[i]
                    for i, (s, t, _w) in enumerate(self._edges)
                    if s == nearest or t == nearest
                })
                relation_text = f"; {', '.join(edge_types[:3])}" if edge_types else ""
                self._status.setText(
                    f"Selected: {name} ({ntype}, n={count}, {conn} connections{relation_text})"
                )
        else:
            self._selected_node = None
            self._status.setText(
                f"Network: {len(self._node_names)} nodes, {len(self._edges)} edges"
            )

        self._draw_graph()

    def _marker_for_node(self, ntype: str, subtype: str) -> str:
        if ntype == "condition":
            return "o"
        if ntype == "region" and subtype == "functional":
            return "D"
        if ntype == "region":
            return "s"
        if ntype == "paper":
            return "."
        if ntype == "cluster":
            return "h"
        if ntype == "method":
            return "^"
        if ntype == "finding":
            return "v"
        return "o"

    def _color_for_node(self, ntype: str, subtype: str) -> str:
        type_order = [
            "condition",
            "region",
            "entity",
            "mechanism",
            "paper",
            "cluster",
            "method",
            "finding",
        ]
        if subtype == "functional":
            return CHART_SEQUENCE[2]
        try:
            idx = type_order.index(ntype) % len(CHART_SEQUENCE)
        except ValueError:
            idx = 0
        return CHART_SEQUENCE[idx]

    def _legend_label(self, ntype: str, subtype: str) -> str:
        # "brain_region" is the node kind in evidence networks saved before
        # the key_entities rename.
        if subtype == "entity" or ntype in ("entity", "brain_region"):
            return "Entity"
        if ntype == "region" and subtype:
            return f"{subtype.title()} region"
        return ntype.replace("_", " ").title()

    def export_image(self, path: Path, fmt: str = "svg"):
        """Save the current graph as an image file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(
            str(path), format=fmt, dpi=150,
            bbox_inches="tight", facecolor=COLORS["surface"],
        )
        logger.info("Exported network graph to %s", path)
