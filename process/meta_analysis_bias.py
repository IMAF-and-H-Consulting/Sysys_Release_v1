"""Publication bias tests (Egger, trim-and-fill, fail-safe N) for Sysys."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sp_stats

from process.meta_analysis_models import BiasTestResult, StudyEffect
from process.meta_analysis_pooling import pool_random_effects

# Rosenthal fail-safe N: one-tailed alpha, as Rosenthal (1979) and metafor::fsn.
FAILSAFE_ALPHA = 0.05
# Values within this of an integer are not bumped up by ceil (float noise).
FAILSAFE_ROUND_TOL = 1e-8
# Below this many studies the bias tests are not run (Egger needs df >= 1).
MIN_STUDIES_FOR_BIAS_TESTS = 3

# ---------------------------------------------------------------------------
# Publication bias tests
# ---------------------------------------------------------------------------

def eggers_test(studies: list[StudyEffect]) -> dict:
    """Egger's regression test for funnel plot asymmetry.

    Regress (effect / SE) on (1/SE).
    Test if the intercept is significantly different from 0.
    Significant intercept (p < 0.1 conventionally) suggests asymmetry.

    Needs >= 10 studies to be meaningful.
    Uses simple OLS via numpy.
    """
    k = len(studies)
    if k < MIN_STUDIES_FOR_BIAS_TESTS:
        return {
            "intercept": 0.0, "se": 0.0, "t_value": 0.0, "p_value": 1.0,
            "interpretation": "Too few studies for Egger's test.",
            "n_studies": k,
        }

    precision = np.array([1.0 / s.standard_error for s in studies if s.standard_error > 0])
    std_effect = np.array([s.effect_size / s.standard_error for s in studies if s.standard_error > 0])

    if len(precision) < MIN_STUDIES_FOR_BIAS_TESTS:
        return {
            "intercept": 0.0, "se": 0.0, "t_value": 0.0, "p_value": 1.0,
            "interpretation": "Too few studies with valid SE for Egger's test.",
            "n_studies": len(precision),
        }

    # OLS: std_effect = intercept + slope * precision
    n = len(precision)
    X = np.column_stack([np.ones(n), precision])
    # Identical SEs make precision collinear with the intercept column. Not every
    # LAPACK build raises on the singular X.T @ X (some return ~1e15 garbage that
    # yields a fabricated p = 0), so test identifiability directly.
    if np.linalg.matrix_rank(X) < 2:
        return {
            "intercept": 0.0, "se": 0.0, "t_value": 0.0, "p_value": 1.0,
            "interpretation": "Egger's test could not be computed: standard "
                              "errors do not vary enough across studies.",
            "n_studies": n,
        }
    beta = np.linalg.lstsq(X, std_effect, rcond=None)[0]
    intercept = float(beta[0])
    residuals = std_effect - X @ beta
    mse = float(np.sum(residuals ** 2) / (n - 2))
    try:
        var_beta = mse * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        # Standard errors identical (or degenerate) across studies: precision
        # is collinear with the intercept column, so the slope is
        # unidentifiable. Report this plainly rather than a fabricated value.
        return {
            "intercept": 0.0, "se": 0.0, "t_value": 0.0, "p_value": 1.0,
            "interpretation": "Egger's test could not be computed: standard "
                              "errors do not vary enough across studies.",
            "n_studies": n,
        }
    se_intercept = float(np.sqrt(var_beta[0, 0]))

    t_val = intercept / se_intercept if se_intercept > 0 else 0.0
    p_val = float(2.0 * sp_stats.t.sf(abs(t_val), n - 2))

    if k < 10:
        interpretation = (
            f"Egger's intercept = {intercept:.3f} (p = {p_val:.3f}). "
            f"Note: only {k} studies; test has low power with < 10 studies."
        )
    elif p_val < 0.05:
        interpretation = (
            f"Egger's intercept = {intercept:.3f} (p = {p_val:.3f}). "
            f"Significant asymmetry detected — publication bias likely."
        )
    elif p_val < 0.10:
        interpretation = (
            f"Egger's intercept = {intercept:.3f} (p = {p_val:.3f}). "
            f"Borderline asymmetry — possible publication bias."
        )
    else:
        interpretation = (
            f"Egger's intercept = {intercept:.3f} (p = {p_val:.3f}). "
            f"No significant asymmetry detected."
        )

    return {
        "intercept": round(intercept, 4),
        "se": round(se_intercept, 4),
        "t_value": round(t_val, 4),
        "p_value": round(p_val, 4),
        "interpretation": interpretation,
        "n_studies": k,
    }


# Duval & Tweedie (2000) trim-and-fill settings.  metafor::trimfill uses the
# same defaults (L0 estimator, at most 100 trim iterations).
TRIMFILL_ESTIMATORS = ("L0", "R0")
TRIMFILL_DEFAULT_ESTIMATOR = "L0"
TRIMFILL_MAX_ITER = 100
SIDE_LEFT = "left"    # studies missing below the pooled effect
SIDE_RIGHT = "right"  # studies missing above the pooled effect


def _pool_values(effects: np.ndarray, variances: np.ndarray):
    """Random-effects pool of plain values (trimmed or filled sets)."""
    temp = [
        StudyEffect(article_id=f"_tf_{i}", effect_size=float(y),
                    variance=float(v), standard_error=float(np.sqrt(v)))
        for i, (y, v) in enumerate(zip(effects, variances, strict=True))
    ]
    return pool_random_effects(temp)


def _study_variances(studies: list[StudyEffect]) -> np.ndarray:
    """Sampling variances, falling back to SE^2 when ``variance`` is unset."""
    return np.array([
        s.variance if s.variance > 0 else s.standard_error ** 2
        for s in studies
    ], dtype=np.float64)


def _estimate_missing_side(effects: np.ndarray, variances: np.ndarray) -> str:
    """Side on which studies are missing, as in metafor::trimfill.

    Fits a DerSimonian-Laird meta-regression of the effects on their standard
    errors.  A positive slope (small studies show larger effects) means
    studies are missing on the left; a negative slope means the right.
    """
    sei = np.sqrt(variances)
    if np.ptp(sei) == 0:
        return SIDE_LEFT  # slope undefined; assume the conventional side
    x = np.column_stack([np.ones_like(sei), sei])
    w = 1.0 / np.maximum(variances, 1e-10)
    xtwx_inv = np.linalg.inv(x.T @ (w[:, None] * x))
    beta_fe = xtwx_inv @ (x.T @ (w * effects))
    q_resid = float(np.sum(w * (effects - x @ beta_fe) ** 2))
    trace_p = float(np.sum(w) - np.trace(xtwx_inv @ (x.T @ ((w ** 2)[:, None] * x))))
    df = len(effects) - x.shape[1]
    tau2 = max(0.0, (q_resid - df) / trace_p) if trace_p > 0 else 0.0
    w_re = 1.0 / (variances + tau2)
    beta_re = np.linalg.lstsq(x * np.sqrt(w_re)[:, None],
                              effects * np.sqrt(w_re), rcond=None)[0]
    return SIDE_RIGHT if beta_re[1] < 0 else SIDE_LEFT


def _estimate_k0(centred: np.ndarray, estimator: str) -> int:
    """Duval-Tweedie L0 or R0 estimate of the number of missing studies.

    ``centred`` holds the effects minus the current pooled estimate, oriented
    so the missing studies lie on the left (the excess on the right).  Ranks
    are of |centred| with ties broken by position, as R's ``ties="first"``.
    """
    k = len(centred)
    ranks = np.empty(k, dtype=np.float64)
    ranks[np.argsort(np.abs(centred), kind="stable")] = np.arange(1, k + 1)
    if estimator == "R0":
        # gamma* = length of the rightmost run of positive ranks; R0 = gamma* - 1
        negative = ranks[centred < 0]
        largest_negative = float(negative.max()) if negative.size else 0.0
        k0 = k - largest_negative - 1
    else:
        # L0 = (4 * T_n - n(n + 1)) / (2n - 1), T_n = Wilcoxon rank sum of positives
        rank_sum = float(ranks[centred > 0].sum())
        k0 = (4.0 * rank_sum - k * (k + 1)) / (2.0 * k - 1)
    return max(0, round(k0))


def _iterate_k0(effects: np.ndarray, variances: np.ndarray,
                estimator: str) -> tuple[int, float] | None:
    """Trim, re-pool and re-estimate k0 until it stops changing.

    ``effects`` must be sorted ascending and oriented so the excess studies
    are the largest.  Returns (k0, pooled estimate of the trimmed set), or
    None when the iteration does not converge.
    """
    k = len(effects)
    k0, previous = 0, -1
    centre = 0.0
    for _ in range(TRIMFILL_MAX_ITER):
        if k0 == previous:
            return k0, centre
        previous = k0
        centre = _pool_values(effects[:k - k0], variances[:k - k0]).pooled_effect
        k0 = _estimate_k0(effects - centre, estimator)
    return (k0, centre) if k0 == previous else None


def _no_imputation(initial, interpretation: str, side: str, estimator: str) -> dict:
    return {
        "n_imputed": 0,
        "adjusted_effect": round(initial.pooled_effect, 6),
        "adjusted_ci_lower": round(initial.ci_lower, 6),
        "adjusted_ci_upper": round(initial.ci_upper, 6),
        "original_effect": round(initial.pooled_effect, 6),
        "imputed_studies": [],
        "side": side,
        "estimator": estimator,
        "interpretation": interpretation,
    }


def _mirror_studies(sources: list[StudyEffect], centre: float) -> list[StudyEffect]:
    """Mirror images of the trimmed studies about the trimmed pooled estimate."""
    return [
        StudyEffect(
            article_id=f"imputed_{s.article_id}",
            title=f"[Imputed] {s.title}",
            effect_size=round(2.0 * centre - s.effect_size, 6),
            standard_error=s.standard_error,
            variance=s.variance,
            sample_size=s.sample_size,
        )
        for s in sources
    ]


def trim_and_fill(studies: list[StudyEffect],
                  estimator: str = TRIMFILL_DEFAULT_ESTIMATOR) -> dict:
    """Duval & Tweedie (2000) iterative trim-and-fill, as metafor::trimfill.

    1. Choose the side with missing studies from the sign of the slope of a
       meta-regression of the effects on their standard errors.
    2. Orient the data so the excess studies are the largest effects, then
       iterate: pool the k - k0 smallest, centre all k effects on that
       estimate, rank |centred effects| and re-estimate k0 (L0 or R0) until
       k0 stops changing.
    3. Fill k0 mirror images of the trimmed studies about the final trimmed
       estimate and re-pool the original plus filled studies.

    The caller's StudyEffect objects are never modified.
    """
    if estimator not in TRIMFILL_ESTIMATORS:
        raise ValueError(
            f"Unknown trim-and-fill estimator {estimator!r}; "
            f"expected one of {TRIMFILL_ESTIMATORS}"
        )
    effects = np.array([s.effect_size for s in studies], dtype=np.float64)
    variances = _study_variances(studies)
    initial = _pool_values(effects, variances)
    if len(studies) < MIN_STUDIES_FOR_BIAS_TESTS:
        return _no_imputation(initial, "Too few studies for trim-and-fill.",
                              "", estimator)

    side = _estimate_missing_side(effects, variances)
    sign = -1.0 if side == SIDE_RIGHT else 1.0
    order = np.argsort(sign * effects, kind="stable")
    converged = _iterate_k0(sign * effects[order], variances[order], estimator)
    if converged is None:
        return _no_imputation(
            initial, "Trim-and-fill did not converge; no studies imputed.",
            side, estimator)
    k0, centre = converged
    if k0 == 0:
        return _no_imputation(
            initial, "No asymmetry detected; no studies imputed.", side, estimator)

    trimmed = [studies[i] for i in order[len(studies) - k0:]]
    imputed = _mirror_studies(trimmed, sign * centre)
    filled = list(studies) + imputed
    adjusted = _pool_values(
        np.array([s.effect_size for s in filled], dtype=np.float64),
        _study_variances(filled))
    theta0 = initial.pooled_effect
    return {
        "n_imputed": k0,
        "adjusted_effect": round(adjusted.pooled_effect, 6),
        "adjusted_ci_lower": round(adjusted.ci_lower, 6),
        "adjusted_ci_upper": round(adjusted.ci_upper, 6),
        "original_effect": round(theta0, 6),
        "imputed_studies": imputed,
        "side": side,
        "estimator": estimator,
        "interpretation": (
            f"{k0} studies imputed on the {side} side ({estimator}). "
            f"Adjusted effect: {adjusted.pooled_effect:.3f} "
            f"[{adjusted.ci_lower:.3f}, {adjusted.ci_upper:.3f}] "
            f"(original: {theta0:.3f})."
        ),
    }


def fail_safe_n(studies: list[StudyEffect], alpha: float = FAILSAFE_ALPHA) -> dict:
    """Rosenthal's (1979) fail-safe N, as metafor::fsn(type = "Rosenthal").

    Number of unpublished null-result studies needed to raise the combined
    (Stouffer) one-tailed p-value above ``alpha``:

        N_fs = ceil((sum(z_i) / z_alpha)^2 - k),  z_alpha one-tailed (1.645)

    N_fs is 0 when the combined result is already non-significant.
    """
    k = len(studies)
    if k == 0:
        return {
            "failsafe_n": 0,
            "interpretation": "No studies available.",
        }
    if not 0.0 < alpha < 0.5:  # z_alpha must be positive
        return {"failsafe_n": 0, "interpretation": "Invalid alpha."}

    z_alpha = float(sp_stats.norm.isf(alpha))
    sum_z = sum(
        s.effect_size / s.standard_error if s.standard_error > 0 else 0.0
        for s in studies
    )
    raw_nfs = (sum_z / z_alpha) ** 2 - k
    nfs = max(0, math.ceil(raw_nfs - FAILSAFE_ROUND_TOL))

    # Interpretation thresholds (Rosenthal's 5k + 10 rule)
    threshold = 5 * k + 10
    if nfs > threshold:
        interpretation = (
            f"Fail-safe N = {nfs} (threshold: {threshold}). "
            f"Result is robust to unpublished null studies."
        )
    else:
        interpretation = (
            f"Fail-safe N = {nfs} (threshold: {threshold}). "
            f"Result may be vulnerable to publication bias."
        )

    return {
        "failsafe_n": nfs,
        "threshold": threshold,
        "interpretation": interpretation,
    }


def run_bias_tests(studies: list[StudyEffect]) -> BiasTestResult:
    """Run all three bias tests and return combined result.

    Below ``MIN_STUDIES_FOR_BIAS_TESTS`` the fields still hold each test's own
    "not run" values (Egger p = 1.0, no imputation, pooled effect as the
    trim-and-fill estimate) rather than the dataclass zeros, which would read
    as a highly significant Egger test (audit 15-M2).  Fail-safe N is defined
    for any k and is always computed.
    """
    k = len(studies)
    result = BiasTestResult(n_studies=k)

    # Egger's test
    egger = eggers_test(studies)
    result.eggers_intercept = egger["intercept"]
    result.eggers_se = egger["se"]
    result.eggers_t = egger["t_value"]
    result.eggers_p = egger["p_value"]
    result.eggers_interpretation = egger["interpretation"]

    # Trim-and-fill
    tf = trim_and_fill(studies)
    result.trimfill_n_imputed = tf["n_imputed"]
    result.trimfill_adjusted_effect = tf["adjusted_effect"]
    result.trimfill_adjusted_ci_lower = tf["adjusted_ci_lower"]
    result.trimfill_adjusted_ci_upper = tf["adjusted_ci_upper"]
    result.trimfill_original_effect = tf["original_effect"]

    # Fail-safe N
    fsn = fail_safe_n(studies)
    result.failsafe_n = fsn["failsafe_n"]
    result.failsafe_interpretation = fsn["interpretation"]

    if k < MIN_STUDIES_FOR_BIAS_TESTS:
        result.overall_interpretation = (
            "Too few studies to assess publication bias."
        )
        return result

    # Overall interpretation
    concerns = []
    if result.eggers_p < 0.05:
        concerns.append("Egger's test significant (publication bias likely)")
    elif result.eggers_p < 0.10:
        concerns.append("Egger's test borderline")

    if result.trimfill_n_imputed > 0:
        concerns.append(f"{result.trimfill_n_imputed} studies imputed by trim-and-fill")

    threshold = 5 * k + 10
    if result.failsafe_n < threshold:
        concerns.append("Fail-safe N below Rosenthal threshold")

    if concerns:
        result.overall_interpretation = (
            "Publication bias concerns: " + "; ".join(concerns) + "."
        )
    else:
        result.overall_interpretation = (
            "No strong evidence of publication bias detected."
        )

    return result
