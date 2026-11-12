"""Compatibility imports for the renamed :mod:`process.graph_analysis`.

The old module path remains importable; select defaults.pattern_pack: neuroscience
explicitly to use the historical vocabulary.
"""
from process.graph_analysis import (
    build_cooccurrence_edges,
    compute_edge_threshold,
    detect_communities,
    extract_entities,
    load_patterns,
    temporal_network_analysis,
)

__all__ = [
    "build_cooccurrence_edges",
    "compute_edge_threshold",
    "detect_communities",
    "extract_entities",
    "load_patterns",
    "temporal_network_analysis",
]
