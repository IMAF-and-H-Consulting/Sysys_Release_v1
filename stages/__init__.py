"""Stages package — the SystS analysis pipeline (GUI Stages 1–10).

Each stage module exposes a ``run(config, ...)`` function and is imported
lazily on first access. Whether a stage's output exists on disk is decided in
one place, ``core.stage_outputs.check_output(stage_num, config)``, which the
Pipeline Board uses.

Stage map (later module filenames retain historical numbers for compatibility)
---------
(Stage 1, literature search, runs live through ``gui.search_runner.run_gui_search``
and intentionally has no stage module here.)
s2_score        — Combined Laya + BRT-BERT screening
s2c_statistical_extract — Programmatic statistical evidence extraction
s2b_fulltext    — Full-text corpus enrichment (pre-scoring)
s3_extract      — Claim extraction
s4_validate     — Method review & span validation
s5_audit        — Span quality audit & filtering
s6_synthesize   — Clustering, narration, abstract draft
s7_fulltext     — Full-text retrieval & metric extraction
s9_robustness   — Stage 8: Robustness analyses
s10_verify      — Stage 9: Abstract number verification
s11_compare_runs — Stage 10: Multi-run comparison
"""

import importlib
from types import ModuleType

__all__ = [
    "s2_score",
    "s2b_fulltext",
    "s2c_statistical_extract",
    "s3_extract",
    "s4_validate",
    "s5_audit",
    "s6_synthesize",
    "s7_fulltext",
    "s9_robustness",
    "s10_verify",
    "s11_compare_runs",
]


def __getattr__(name: str) -> ModuleType:
    """Import a stage module on first access (``stages.s2_score``).

    Importing one light helper (``stages._common``) used to import every
    stage and with it torch/transformers/sklearn.
    """
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
