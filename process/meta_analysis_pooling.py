"""Random-effects pooling, effect-size loading and input resolution for Sysys."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

from process.meta_analysis_models import (
    STAGE2C_META_ANALYSIS_INPUT,
    STAGE7_FULLTEXT_METRICS,
    PooledEstimate,
    StudyEffect,
)

# Logger name is kept as "process.meta_analysis" so log records are unchanged.
logger = logging.getLogger("process.meta_analysis")

Z_95 = 1.96
# Sampling variances at or below zero are replaced by this floor.
MIN_VARIANCE = 1e-10
# REML Fisher scoring (DL start); metafor's defaults are 100 iterations, 1e-5.
REML_MAX_ITER = 100
REML_TOL = 1e-10
# p-values keep this many significant digits (fixed decimals rounded tiny p to 0).
P_VALUE_SIG_DIGITS = 6


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_se_from_n(effect_size: float, n: int) -> float:
    """Approximate SE from effect size and sample size.

    Assumes equal groups: SE ~ sqrt(4/n + d^2 / (2*n))
    """
    if n <= 0:
        return 0.0
    return math.sqrt(4.0 / n + effect_size ** 2 / (2.0 * n))


def _round_p(p_value: float) -> float:
    """Round a p-value to significant digits so small values stay non-zero."""
    return float(f"{p_value:.{P_VALUE_SIG_DIGITS}g}")


def _two_sided_normal_p(z_value: float) -> float:
    """Two-sided normal p-value via the survival function (no 1 - cdf underflow)."""
    return float(2.0 * sp_stats.norm.sf(abs(z_value)))


# ---------------------------------------------------------------------------
# Core meta-analysis
# ---------------------------------------------------------------------------

def _tau2_dl(variances: np.ndarray, q_stat: float) -> float:
    """DerSimonian-Laird between-study variance, truncated at 0."""
    w_fe = 1.0 / variances
    sum_w = float(np.sum(w_fe))
    c = sum_w - float(np.sum(w_fe ** 2)) / sum_w
    return max(0.0, (q_stat - (len(variances) - 1)) / c) if c > 0 else 0.0


def _tau2_reml(effects: np.ndarray, variances: np.ndarray, start: float) -> float:
    """REML between-study variance by Fisher scoring from ``start``.

    Score: 0.5 * (y'PPy - tr(P)); expected information: 0.5 * tr(PP), with
    P = W - W11'W / sum(W) (Viechtbauer 2005).  Non-convergence is logged and
    the last iterate returned.
    """
    tau2 = start
    for _ in range(REML_MAX_ITER):
        w = 1.0 / (variances + tau2)
        sw, sw2, sw3 = float(np.sum(w)), float(np.sum(w ** 2)), float(np.sum(w ** 3))
        resid = effects - float(np.sum(w * effects)) / sw
        score = 0.5 * (float(np.sum(w ** 2 * resid ** 2)) - (sw - sw2 / sw))
        info = 0.5 * (sw2 - 2.0 * sw3 / sw + (sw2 / sw) ** 2)
        if info <= 0:
            break
        new_tau2 = max(0.0, tau2 + score / info)
        if abs(new_tau2 - tau2) < REML_TOL:
            return new_tau2
        tau2 = new_tau2
    logger.warning("REML did not converge in %d iterations; tau^2 = %.6g", REML_MAX_ITER, tau2)
    return tau2


def _i_squared(variances: np.ndarray, tau2: float) -> float:
    """I^2 = tau^2 / (tau^2 + s^2), Higgins & Thompson's typical variance s^2.

    Equals (Q - df) / Q for DerSimonian-Laird and matches metafor for REML.
    """
    w = 1.0 / variances
    k = len(variances)
    denom = float(np.sum(w)) ** 2 - float(np.sum(w ** 2))
    if denom <= 0:
        return 0.0
    typical = (k - 1) * float(np.sum(w)) / denom
    total = tau2 + typical
    return 100.0 * tau2 / total if total > 0 else 0.0


def _weighted_copies(studies: list[StudyEffect], weights: np.ndarray) -> list[StudyEffect]:
    """Copies of ``studies`` carrying weights relative to the largest (audit 15-M1)."""
    max_w = float(np.max(weights)) if weights.size else 0.0
    return [replace(s, weight=float(w) / max_w if max_w > 0 else 0.0)
            for s, w in zip(studies, weights)]


def _pool_single(study: StudyEffect) -> PooledEstimate:
    se = study.standard_error
    z_val = study.effect_size / se if se > 0 else 0.0
    return PooledEstimate(
        pooled_effect=study.effect_size,
        pooled_se=se,
        ci_lower=study.effect_size - Z_95 * se,
        ci_upper=study.effect_size + Z_95 * se,
        z_value=z_val,
        p_value=_round_p(_two_sided_normal_p(z_val)) if se > 0 else 1.0,
        n_studies=1,
        studies=_weighted_copies([study], np.array([1.0])),
    )


def _prediction_interval(y_re: float, se_re: float, tau2: float, k: int) -> tuple:
    """Higgins prediction interval y +/- t(k-2) * sqrt(SE^2 + tau^2); (0, 0) for k <= 2."""
    if k <= 2:
        return (0.0, 0.0)
    half = float(sp_stats.t.ppf(0.975, k - 2)) * math.sqrt(se_re ** 2 + tau2)
    return (round(y_re - half, 4), round(y_re + half, 4))


def pool_random_effects(studies: list[StudyEffect], method: str = "DL") -> PooledEstimate:
    """Random-effects meta-analysis with DerSimonian-Laird or REML tau^2.

    ``method`` is "DL" (default) or "REML" (Fisher scoring from the DL
    estimate).  Q is Cochran's fixed-effect statistic; I^2 is derived from
    tau^2 as in metafor.  The caller's studies are never modified: the result
    holds copies whose ``weight`` is the random-effects weight relative to
    the largest one.
    """
    k = len(studies)
    if k == 0:
        return PooledEstimate()
    if k == 1:
        return _pool_single(studies[0])

    effects = np.array([s.effect_size for s in studies], dtype=np.float64)
    variances = np.array([s.variance for s in studies], dtype=np.float64)
    variances = np.where(variances > 0, variances, MIN_VARIANCE)

    w_fe = 1.0 / variances
    y_fe = float(np.sum(w_fe * effects) / np.sum(w_fe))
    q_stat = float(np.sum(w_fe * (effects - y_fe) ** 2))
    q_p = float(sp_stats.chi2.sf(q_stat, k - 1))

    tau2 = _tau2_dl(variances, q_stat)
    if method.upper() == "REML":
        tau2 = _tau2_reml(effects, variances, tau2)

    w_re = 1.0 / (variances + tau2)
    y_re = float(np.sum(w_re * effects) / np.sum(w_re))
    se_re = float(np.sqrt(1.0 / np.sum(w_re)))
    z_val = y_re / se_re if se_re > 0 else 0.0

    return PooledEstimate(
        pooled_effect=round(y_re, 6),
        pooled_se=round(se_re, 6),
        ci_lower=round(y_re - Z_95 * se_re, 6),
        ci_upper=round(y_re + Z_95 * se_re, 6),
        z_value=round(z_val, 4),
        p_value=_round_p(_two_sided_normal_p(z_val)),
        tau_squared=round(tau2, 6),
        i_squared=round(_i_squared(variances, tau2), 2),
        q_statistic=round(q_stat, 4),
        q_p_value=_round_p(q_p),
        n_studies=k,
        prediction_interval=_prediction_interval(y_re, se_re, tau2, k),
        studies=_weighted_copies(studies, w_re),
    )


def resolve_meta_analysis_input(
    config,
    *,
    input_source: str = "stage2c",
    metrics_path: str | Path | None = None,
) -> tuple[Path, str]:
    """Resolve the explicitly selected meta-analysis input artifact."""
    source = str(input_source or "stage2c").strip().lower()
    if metrics_path is not None:
        path = Path(metrics_path)
        expected_paths = {
            "stage2c": config.claims_dir / STAGE2C_META_ANALYSIS_INPUT,
            "stage7": config.data_dir / "fulltext" / STAGE7_FULLTEXT_METRICS,
            "legacy": config.output_dir / STAGE7_FULLTEXT_METRICS,
        }
        if source in expected_paths and path == expected_paths[source]:
            return path, source
        return path, "custom"

    if source == "stage2c":
        return config.claims_dir / STAGE2C_META_ANALYSIS_INPUT, source
    if source == "stage7":
        return config.data_dir / "fulltext" / STAGE7_FULLTEXT_METRICS, source
    if source == "legacy":
        return config.output_dir / STAGE7_FULLTEXT_METRICS, source
    raise ValueError(
        "input_source must be one of: 'stage2c', 'stage7', 'legacy', "
        "or pass metrics_path for an explicit file"
    )


def _first_numeric(values: list) -> float | None:
    for value in values:
        if isinstance(value, dict):
            value = value.get("value")
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def load_effect_sizes(
    config,
    *,
    metrics_path: str | Path | None = None,
    input_source: str = "stage2c",
) -> list[StudyEffect]:
    """Load all linked estimates, preserving their effect measures and uncertainty.

    This low-level API does not decide whether outcomes are compatible for
    pooling. ``run_full_analysis`` applies grouping and independent-study checks.
    """
    from process.meta_analysis_inputs import parse_metrics

    metrics_path, resolved_source = resolve_meta_analysis_input(
        config, input_source=input_source, metrics_path=metrics_path)
    if not metrics_path.exists():
        logger.warning("No %s meta-analysis input found at %s", resolved_source, metrics_path)
        return []
    if resolved_source == "stage2c":
        from process.statistical_extractor_pipeline import refresh_stage2c_outputs_for_legacy_failures
        refresh_stage2c_outputs_for_legacy_failures(metrics_path)
    try:
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        studies, audit = parse_metrics(raw)
    except (ValueError, OSError) as exc:
        logger.error("Failed to load metrics: %s", exc)
        return []
    for row in audit:
        for reason, count in row["exclusion_reasons"].items():
            logger.info("%s: %s (%d results)", row["article_id"], reason, count)
    logger.info("Loaded %d linked estimates from %s", len(studies), metrics_path)
    return studies


# ---------------------------------------------------------------------------
# Multiple comparison correction
# ---------------------------------------------------------------------------

def benjamini_hochberg(
    p_values: dict[str, float],
    alpha: float = 0.05,
) -> dict[str, dict]:
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Parameters
    ----------
    p_values : dict mapping test name → raw p-value
    alpha : significance threshold (default 0.05)

    Returns
    -------
    dict mapping test name → {"raw_p", "adjusted_p", "significant"}
    """
    if not p_values:
        return {}

    # Sort by p-value ascending
    sorted_items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(sorted_items)

    # Compute adjusted p-values: p_adj[i] = p[i] * m / rank
    adjusted = []
    for rank_1based, (name, raw_p) in enumerate(sorted_items, start=1):
        adj = raw_p * m / rank_1based
        adjusted.append((name, raw_p, adj))

    # Enforce monotonicity (step-up): walk backwards, each must be ≤ successor
    for i in range(len(adjusted) - 2, -1, -1):
        name, raw_p, adj = adjusted[i]
        _, _, adj_next = adjusted[i + 1]
        adjusted[i] = (name, raw_p, min(adj, adj_next))

    # Cap at 1.0 and build result
    result = {}
    for name, raw_p, adj in adjusted:
        adj_capped = min(adj, 1.0)
        result[name] = {
            "raw_p": raw_p,
            "adjusted_p": round(adj_capped, 10),
            "significant": adj_capped < alpha,
        }

    return result
