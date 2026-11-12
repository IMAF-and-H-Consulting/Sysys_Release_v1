"""Data models and constants for the Sysys meta-analysis pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field

STAGE2C_META_ANALYSIS_INPUT = "statistical_meta_analysis_input.json"
STAGE7_FULLTEXT_METRICS = "auto_extracted_metrics.json"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StudyEffect:
    """A single study's effect size data."""
    article_id: str
    title: str = ""
    effect_size: float = 0.0       # Hedges' g or Cohen's d
    standard_error: float = 0.0
    variance: float = 0.0
    sample_size: int = 0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    weight: float = 0.0            # inverse-variance weight (computed)
    component_id: str = ""
    condition: str = ""
    effect_type: str = "g"
    analysis_scale: str = "Hedges' g"
    analysis_group: str = ""
    outcome: str = ""
    comparison: str = ""
    population: str = ""
    source_location: dict = field(default_factory=dict)
    # Which reported value gave the standard error (confidence_interval,
    # standard_error, p_value, sample_size, paper_sample_size) and any
    # assumption the calculation needed, so a reader can audit each estimate.
    uncertainty_source: str = ""
    assumptions: tuple[str, ...] = ()
    # "researcher" when assigned in the effects table; "model_validated" only
    # after the model's title/RQ and full scientific-compatibility audits pass.
    # Such an explicit group may mix convertible effect types.
    group_source: str = ""


@dataclass
class PooledEstimate:
    """Result of a random-effects meta-analysis."""
    pooled_effect: float = 0.0
    pooled_se: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    z_value: float = 0.0
    p_value: float = 0.0
    tau_squared: float = 0.0       # between-study variance
    i_squared: float = 0.0         # heterogeneity %
    q_statistic: float = 0.0       # Cochran's Q
    q_p_value: float = 0.0
    n_studies: int = 0
    prediction_interval: tuple = (0.0, 0.0)
    studies: list = field(default_factory=list)  # list of StudyEffect


@dataclass
class BiasTestResult:
    """Results from publication bias testing."""
    # Egger's regression test
    eggers_intercept: float = 0.0
    eggers_se: float = 0.0
    eggers_t: float = 0.0
    eggers_p: float = 0.0
    eggers_interpretation: str = ""

    # Trim-and-fill
    trimfill_n_imputed: int = 0
    trimfill_adjusted_effect: float = 0.0
    trimfill_adjusted_ci_lower: float = 0.0
    trimfill_adjusted_ci_upper: float = 0.0
    trimfill_original_effect: float = 0.0

    # Fail-safe N (Rosenthal)
    failsafe_n: int = 0
    failsafe_interpretation: str = ""

    n_studies: int = 0
    overall_interpretation: str = ""
