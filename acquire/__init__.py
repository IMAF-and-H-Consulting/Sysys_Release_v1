"""Acquire package — multi-provider literature retrieval clients.

Primary entry point: :func:`gui.search_runner.run_gui_search`, which uses the
provider clients, normalizers and dedup logic in this package directly.
"""

from acquire.deduplicator import deduplicate_papers

__all__ = [
    "deduplicate_papers",
]
