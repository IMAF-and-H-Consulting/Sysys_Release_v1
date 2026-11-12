"""Stage 8: Robustness analyses.

Six statistical tests to confirm that findings are not artifacts:
1. Bootstrap confidence intervals for mean relevance score
2. Split-half reliability (random splits + correlation)
3. Sensitivity analysis at different relevance thresholds
4. Permutation test on the condition–support gradient
5. Condition–entity co-occurrence network analysis
6. Co-occurring condition sensitivity (each configured condition's papers
   included vs excluded)

No LLM calls — pure statistics with numpy.

scipy is optional. Spearman rank correlation is always computed manually.
When scipy is present it supplies the exact Kendall tau-b p-value and the
ICC confidence interval; without it, Kendall tau-b falls back to a normal
approximation and the ICC is reported without a confidence interval.
"""

# ruff: noqa: F401  (facade: names below are re-exported for importers)
import logging
import math
from typing import cast

import numpy as np

from stages.s9_robustness_analyses import (
    MATERIAL_RATE_SHIFT,
    _condition_gradient,
    _single_condition_sensitivity,
    _support_rate,
    condition_sensitivity_analysis,
    condition_sensitivity_summary,
    permutation_test,
    sensitivity_analysis,
)
from stages.s9_robustness_config import (
    DEFAULT_OUTCOME,
    RobustnessOutcome,
    _extract_conditions,
    _get_relevance_score,
    _load_condition_config,
    _load_outcome,
    _normalize_condition,
)
from stages.s9_robustness_report import (
    _condition_sensitivity_report_lines,
    _write_network_report,
    _write_report,
)
from stages.s9_robustness_stats import (
    CancelCheck,
    RobustnessCancelled,
    _rank_array,
    _spearman_rank_correlation,
    bootstrap_ci,
    check_cancel_every,
    cluster_effect_size,
)

logger = logging.getLogger(__name__)

# ── Optional scipy ────────────────────────────────────────────────────────
# Probed once at import so the stage degrades gracefully instead of raising
# ImportError deep inside an analysis. Catches Exception rather than
# ImportError because a broken install can fail with other error types.
_HAS_SCIPY = False
try:
    from scipy.stats import f as _sp_f_dist
    from scipy.stats import kendalltau as _sp_kendalltau
    _HAS_SCIPY = True
except Exception as _exc:  # pragma: no cover - depends on local environment
    _sp_kendalltau = None
    _sp_f_dist = None
    logger.info(
        "scipy unavailable (%s: %s) — Kendall tau p-values use a normal "
        "approximation and ICC confidence intervals are omitted",
        type(_exc).__name__, _exc,
    )


# ── Kendall's τ and ICC ─────────────────────────────────────────────────

def _kendall_tau_b_numpy(x: list[float], y: list[float]) -> tuple[float, float]:
    """Kendall's tau-b and an approximate p-value, without scipy.

    Counts concordant/discordant pairs with a vectorised sign-product over the
    upper triangle, applies the standard tie corrections for tau-b, and derives
    a two-sided p-value from the normal approximation to the null distribution
    of the pair-count statistic. The p-value ignores the tie correction to the
    variance, so it is approximate; scipy is used instead whenever available.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    n = a.size

    sign_a = np.sign(a[:, None] - a[None, :])
    sign_b = np.sign(b[:, None] - b[None, :])
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    products = (sign_a * sign_b)[upper]
    concordant_minus_discordant = float(products.sum())

    n0 = n * (n - 1) / 2.0
    # Tie corrections: n1 over ties in x, n2 over ties in y.
    def _tie_term(values: np.ndarray) -> float:
        _, counts = np.unique(values, return_counts=True)
        return float((counts * (counts - 1) / 2.0).sum())

    n1 = _tie_term(a)
    n2 = _tie_term(b)
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    # A constant series has no ordering to correlate: tau-b is undefined.
    tau = concordant_minus_discordant / denom if denom > 0 else math.nan

    var = n * (n - 1) * (2 * n + 5) / 18.0
    if var <= 0:
        return tau, 1.0
    z = concordant_minus_discordant / math.sqrt(var)
    # Two-sided p from the standard normal survival function.
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return tau, min(1.0, max(0.0, p))


def _tau_not_estimable(reason: str) -> dict:
    return {"tau": None, "p_value": None, "estimable": False, "reason": reason}


def kendall_tau(x: list[float], y: list[float]) -> dict:
    """Kendall's tau-b with p-value (scipy when available, else approximate).

    Input with no defined tau (unequal lengths, fewer than two pairs, or a
    constant series) returns None values, ``estimable: False`` and a
    ``reason`` rather than a fake tau=0 / p=1 that reads as a measured null
    result (audit 22-L5).
    """
    if len(x) != len(y):
        return _tau_not_estimable(
            f"x and y differ in length ({len(x)} vs {len(y)})")
    if len(x) < 2:
        return _tau_not_estimable(
            f"only {len(x)} pair(s); Kendall's tau needs at least 2")

    if _HAS_SCIPY:
        assert _sp_kendalltau is not None
        # scipy's kendalltau result is built via _make_tuple_bunch, which
        # has no static return type, so pyright sees an opaque element type.
        result = _sp_kendalltau(x, y)
        tau = cast(float, result[0])
        p = cast(float, result[1])
        approximate = False
    else:
        tau, p = _kendall_tau_b_numpy(x, y)
        approximate = True

    if tau is None or math.isnan(tau):
        return _tau_not_estimable("a series is constant; Kendall's tau is undefined")
    return {
        "tau": float(tau),
        "p_value": None if p is None or math.isnan(p) else float(p),
        "p_value_approximate": approximate,
        "estimable": True,
    }


# Fewest subjects (rows) an ICC is computed from.
MIN_ICC_SUBJECTS = 3


def _icc_not_estimable(reason: str) -> dict:
    return {"icc": None, "ci_lower": None, "ci_upper": None, "reason": reason}


def icc_oneway(ratings: np.ndarray) -> dict:
    """ICC(1,1) — one-way random, single measures.

    Parameters
    ----------
    ratings : (n_subjects, k_raters) array

    Returns
    -------
    dict with icc, ci_lower, ci_upper.  Degenerate input (fewer than
    ``MIN_ICC_SUBJECTS`` subjects, or no variance at all) has no ICC: every
    value is None and ``reason`` says why, rather than a fake [0, 0] CI that
    reads as perfect precision (audit 12-L7).
    """
    ratings = np.asarray(ratings, dtype=np.float64)
    n, k = ratings.shape
    if n < MIN_ICC_SUBJECTS:
        return _icc_not_estimable(
            f"only {n} subject(s); ICC needs at least {MIN_ICC_SUBJECTS}")

    # One-way ANOVA decomposition
    grand_mean = np.mean(ratings)
    row_means = np.mean(ratings, axis=1)

    ss_between = k * np.sum((row_means - grand_mean) ** 2)
    ss_within = np.sum((ratings - row_means[:, np.newaxis]) ** 2)

    df_between = n - 1
    df_within = n * (k - 1)

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    # ICC(1,1) = (MS_between - MS_within) / (MS_between + (k-1)*MS_within)
    denom = ms_between + (k - 1) * ms_within
    if denom < 1e-12:
        return _icc_not_estimable("ratings have no variance; ICC is undefined")

    icc = (ms_between - ms_within) / denom

    # Confidence interval via F-distribution — requires scipy. The ICC point
    # estimate is pure ANOVA and always available, so report it either way
    # rather than failing the whole analysis.
    if not _HAS_SCIPY:
        return {
            "icc": float(icc),
            "ci_lower": None,
            "ci_upper": None,
            "ci_unavailable_reason": "scipy not installed",
        }

    assert _sp_f_dist is not None  # guaranteed by the _HAS_SCIPY check above
    f_dist = _sp_f_dist
    f_val = ms_between / ms_within if ms_within > 1e-12 else 1e6
    alpha = 0.05

    fl = f_val / f_dist.ppf(1 - alpha / 2, df_between, df_within)
    fu = f_val / f_dist.ppf(alpha / 2, df_between, df_within)

    ci_lower = (fl - 1) / (fl + k - 1)
    ci_upper = (fu - 1) / (fu + k - 1)

    return {
        "icc": round(float(icc), 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
    }


# ── Analysis 2: Split-half reliability ────────────────────────────────────

def _spearman_brown(r_half: float) -> float:
    """Project a half-length reliability to full length.

    The Spearman-Brown prophecy formula for doubling test length is
    ``r_full = 2 * r_half / (1 + r_half)``. Returns -1.0 at the singularity
    ``r_half == -1`` and clamps the result to [-1, 1].
    """
    denominator = 1.0 + r_half
    if abs(denominator) < 1e-12:
        return -1.0
    return max(-1.0, min(1.0, 2.0 * r_half / denominator))


# Percentiles of the split-to-split distribution reported as the uncertainty
# interval for the per-split Kendall tau and ICC (audit 12-H2).
AGREEMENT_INTERVAL_PERCENTILES = (2.5, 97.5)
AGREEMENT_INTERVAL_LABEL = "split-to-split 2.5-97.5 percentile"


def _split_agreement(vec_a: np.ndarray, vec_b: np.ndarray) -> tuple[float | None, float | None]:
    """Kendall tau-b and ICC(1,1) between one split's two half-profiles.

    Each split contributes one value per statistic computed over its own k
    conditions, so no condition is counted more than once per estimate.
    Either value is None when it is not estimable for this split.
    """
    tau = kendall_tau(vec_a.tolist(), vec_b.tolist())["tau"]
    icc = icc_oneway(np.column_stack([vec_a, vec_b]))["icc"]
    return (None if tau is None else float(tau)), (None if icc is None else float(icc))


def _summarise_across_splits(values: list[float]) -> tuple[float, float, float]:
    """Median and split-to-split percentile interval of a per-split statistic.

    The median (not the mean) is the point estimate so it always lies inside
    the reported percentile interval; with k ~ 5 conditions per split the
    per-split values are coarse and a few outlying splits can drag the mean
    outside it.
    """
    arr = np.asarray(values, dtype=np.float64)
    lower, upper = np.percentile(arr, AGREEMENT_INTERVAL_PERCENTILES)
    return float(np.median(arr)), float(lower), float(upper)


def _not_estimable(note: str, categories: int = 0) -> dict:
    """Uniform 'reliability could not be estimated' result.

    Returning an explicit note beats emitting a plausible-looking number that
    does not measure anything, which is exactly how the previous
    implementation misled readers.
    """
    return {
        "mean_correlation": 0.0, "std_correlation": 0.0,
        "min_correlation": 0.0, "max_correlation": 0.0,
        "mean_half_correlation": 0.0,
        "kendall_tau": 0.0, "kendall_tau_p": None,
        "kendall_tau_ci_lower": None, "kendall_tau_ci_upper": None,
        "icc": 0.0, "icc_ci_lower": None, "icc_ci_upper": None,
        "agreement_interval": AGREEMENT_INTERVAL_LABEL,
        "n_splits": 0, "profile_categories": categories,
        "spearman_brown_applied": True,
        "estimable": False,
        "interpretation": "not estimable",
        "note": note,
    }


def split_half(papers: list[dict], n_splits: int = 1000,
               seed: int = 42,
               condition_patterns: dict[str, str] | None = None,
               condition_ranking: list[str] | None = None,
               outcome: RobustnessOutcome = DEFAULT_OUTCOME,
               min_category_n: int = 3,
               min_categories: int = 3,
               max_agreement_pairs: int | None = None,
               check_cancel: CancelCheck | None = None) -> dict:
    """Split-half reliability of the per-condition support profile.

    Randomly partitions the papers into two disjoint halves ``n_splits`` times.
    For each split it computes, independently in each half, the support
    rate for every configured condition. Those two rate vectors are correlated
    across conditions, and the resulting half-length correlation is projected
    to full corpus length with the Spearman-Brown formula
    ``r_full = 2*r_half / (1 + r_half)``.

    Why the profile is per-condition rates
    --------------------------------------
    Two earlier formulations were degenerate because they compared quantities
    that are *constrained to sum* across the two halves:

    * Correlating the halves' **mean scores** fails because the halves
      partition the corpus, so ``mean_a + mean_b == 2 * grand_mean`` is an
      identity. That pins the correlation — and the ICC built on it — to
      exactly -1.0 for even ``n``, whatever the data says.
    * Correlating the halves' **score distributions** fails for the same
      reason in compositional form: the per-level counts of the two halves sum
      to the corpus counts, so the two profiles are anti-correlated by
      construction.

    Per-condition rates avoid this. ``rate_a[c]`` and ``rate_b[c]`` are
    computed on disjoint paper sets and each has its own denominator, so they
    are two *independent estimates of the same underlying quantity* — the
    textbook precondition for a reliability coefficient. When conditions truly
    differ in support, both halves recover that pattern and the
    correlation is high; when the apparent gradient is sampling noise, it is
    low. That is precisely the question Stage 8 exists to answer.

    Agreement statistics (Kendall tau-b, ICC(1,1)) are computed *within* each
    split over its k shared conditions and summarised by their median across
    splits, with the split-to-split 2.5-97.5 percentile range as interval.  Pooling
    every split's k values into one sample would treat the same conditions as
    ~1000 independent subjects, so no pooled p-value or parametric CI is
    reported (``kendall_tau_p`` is None).  ``max_agreement_pairs`` is accepted
    for backward compatibility and ignored: nothing is pooled or thinned.

    Conditions must be configured (``co_occurring_conditions`` or
    ``graph_analysis.conditions``); without them
    there is no profile to replicate and the result is reported as not
    estimable rather than fabricated.
    """
    del max_agreement_pairs  # retained only for caller compatibility
    logger.info("split_half: START — %d papers, %d splits", len(papers), n_splits)

    n = len(papers)
    if n < 4:
        logger.warning("split_half: < 4 papers, reliability not estimable")
        return _not_estimable("fewer than 4 papers — reliability not estimable")

    ranking = list(condition_ranking or [])
    if not ranking:
        logger.warning(
            "split_half: no conditions configured — split-half reliability "
            "needs a per-condition profile to replicate"
        )
        return _not_estimable(
            "no conditions configured (`co_occurring_conditions` or "
            "`graph_analysis.conditions`) — "
            "split-half reliability is not defined without a profile to replicate"
        )

    patterns = condition_patterns or {}

    # Extract conditions and the outcome flag once per paper. These are
    # regex scans over title+abstract, so doing them inside the split loop
    # would repeat the same work n_splits times.
    paper_conditions: list[set[str]] = []
    paper_supports: list[bool] = []
    for paper in papers:
        conditions = set(_extract_conditions(paper, patterns)) & set(ranking)
        paper_conditions.append(conditions)
        paper_supports.append(outcome.supports(paper))

    if not any(paper_conditions):
        return _not_estimable(
            "no paper matched any configured condition — "
            "reliability not estimable", categories=len(ranking),
        )

    def _rates(indices: np.ndarray) -> dict[str, float]:
        totals: dict[str, int] = {c: 0 for c in ranking}
        supporting: dict[str, int] = {c: 0 for c in ranking}
        for i in indices:
            supports = paper_supports[i]
            for condition in paper_conditions[i]:
                totals[condition] += 1
                if supports:
                    supporting[condition] += 1
        return {
            c: supporting[c] / totals[c]
            for c in ranking
            if totals[c] >= min_category_n
        }

    half = n // 2
    rng = np.random.default_rng(seed)

    corrected: list[float] = []
    half_corrs: list[float] = []
    split_taus: list[float] = []
    split_iccs: list[float] = []
    categories_used: list[int] = []

    for iteration in range(n_splits):
        check_cancel_every(check_cancel, iteration)
        perm = rng.permutation(n)
        rates_a = _rates(perm[:half])
        rates_b = _rates(perm[half:half + half])

        # Only conditions estimable in *both* halves can be compared.
        shared = [c for c in ranking if c in rates_a and c in rates_b]
        if len(shared) < min_categories:
            continue

        vec_a = np.array([rates_a[c] for c in shared], dtype=np.float64)
        vec_b = np.array([rates_b[c] for c in shared], dtype=np.float64)
        categories_used.append(len(shared))

        # A half with a flat profile carries no variance to correlate.
        if np.std(vec_a) < 1e-12 or np.std(vec_b) < 1e-12:
            continue
        r = float(np.corrcoef(vec_a, vec_b)[0, 1])
        if np.isnan(r):
            continue
        half_corrs.append(r)
        corrected.append(_spearman_brown(r))
        tau, icc = _split_agreement(vec_a, vec_b)
        if tau is not None:
            split_taus.append(tau)
        if icc is not None:
            split_iccs.append(icc)

    if not corrected:
        return _not_estimable(
            f"fewer than {min_categories} conditions had at least "
            f"{min_category_n} papers in both halves — reliability not estimable",
            categories=len(ranking),
        )

    mean_r = float(np.mean(corrected))
    std_r = float(np.std(corrected))
    min_r = float(np.min(corrected))
    max_r = float(np.max(corrected))
    mean_half = float(np.mean(half_corrs))

    # Per-split agreement, summarised across splits (no pseudo-replication).
    tau_median, tau_lo, tau_hi = (
        _summarise_across_splits(split_taus) if split_taus else (None, None, None))
    icc_median, icc_lo, icc_hi = (
        _summarise_across_splits(split_iccs) if split_iccs else (None, None, None))

    if mean_r > 0.8:
        interpretation = "excellent"
    elif mean_r > 0.6:
        interpretation = "good"
    elif mean_r > 0.4:
        interpretation = "fair"
    else:
        interpretation = "poor"

    result = {
        "mean_correlation": mean_r,
        "std_correlation": std_r,
        "min_correlation": min_r,
        "max_correlation": max_r,
        "mean_half_correlation": mean_half,
        "kendall_tau": tau_median,
        "kendall_tau_p": None,
        "kendall_tau_ci_lower": tau_lo,
        "kendall_tau_ci_upper": tau_hi,
        "icc": icc_median,
        "icc_ci_lower": icc_lo,
        "icc_ci_upper": icc_hi,
        "agreement_interval": AGREEMENT_INTERVAL_LABEL,
        "n_splits": len(corrected),
        "profile_categories": round(float(np.mean(categories_used))),
        "spearman_brown_applied": True,
        "estimable": True,
        "interpretation": interpretation,
    }
    logger.info(
        "split_half: DONE — Spearman-Brown r=%.3f (half r=%.3f), tau=%.3f, "
        "icc=%s (%s), std=%.3f",
        mean_r, mean_half, tau_median, icc_median,
        interpretation, std_r,
    )
    return result


def __getattr__(name: str):
    # ``run`` lives in s9_robustness_run, which imports split_half from this
    # module. Resolving it lazily keeps the module load free of a cycle.
    if name == "run":
        from stages.s9_robustness_run import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
