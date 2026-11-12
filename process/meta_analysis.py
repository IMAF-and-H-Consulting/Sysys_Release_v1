"""Meta-analytic effect size pooling for Sysys (facade).

The implementation lives in process/meta_analysis_*.py; this module re-exports
every name so existing importers keep working.
"""
from __future__ import annotations

import logging

from process.meta_analysis_bias import (
    eggers_test,
    fail_safe_n,
    run_bias_tests,
    trim_and_fill,
)
from process.meta_analysis_models import (
    STAGE2C_META_ANALYSIS_INPUT,
    STAGE7_FULLTEXT_METRICS,
    BiasTestResult,
    PooledEstimate,
    StudyEffect,
)
from process.meta_analysis_plots import (
    create_forest_plot_figure,
    create_funnel_plot_figure,
)
from process.meta_analysis_pooling import (
    _first_numeric,
    benjamini_hochberg,
    compute_se_from_n,
    load_effect_sizes,
    pool_random_effects,
    resolve_meta_analysis_input,
)
from process.meta_analysis_run import add_result_figures, run_full_analysis

logger = logging.getLogger("process.meta_analysis")

__all__ = [
    "STAGE2C_META_ANALYSIS_INPUT",
    "STAGE7_FULLTEXT_METRICS",
    "BiasTestResult",
    "PooledEstimate",
    "StudyEffect",
    "_first_numeric",
    "add_result_figures",
    "benjamini_hochberg",
    "compute_se_from_n",
    "create_forest_plot_figure",
    "create_funnel_plot_figure",
    "eggers_test",
    "fail_safe_n",
    "load_effect_sizes",
    "pool_random_effects",
    "resolve_meta_analysis_input",
    "run_bias_tests",
    "run_full_analysis",
    "trim_and_fill",
]
