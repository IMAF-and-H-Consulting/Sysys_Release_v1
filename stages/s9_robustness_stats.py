"""Stage 8 robustness: rank correlation, effect size and bootstrap helpers.

Split out of stages/s9_robustness.py, which re-exports these names.
"""

import logging
import math
from collections.abc import Callable

import numpy as np

from stages.s9_robustness_config import _is_missing_score, _parse_relevance_score

# Same logger name as the original single module (stages.s9_robustness).
logger = logging.getLogger("stages.s9_robustness")


class RobustnessCancelled(RuntimeError):
    """Raised from inside a long analysis loop when the user cancels Stage 8."""


# Called periodically inside long loops; raises RobustnessCancelled to stop.
CancelCheck = Callable[[], None]
# Loop iterations between cancellation checks (audit 12-L5).
CANCEL_CHECK_INTERVAL = 100


# ── Spearman rank correlation (no scipy) ──────────────────────────────────

def _rank_array(values: list[float]) -> list[float]:
    """Assign fractional ranks to values (handles ties)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        # Find all tied values
        while j < n - 1 and values[indexed[j]] == values[indexed[j + 1]]:
            j += 1
        # Assign average rank to ties (ranks are 1-based)
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1

    return ranks


def _spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation without scipy.

    Ranks both arrays, then computes Pearson correlation on the ranks.
    Returns 0.0 if the data is degenerate (constant values, too few points).
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    rx = _rank_array(x)
    ry = _rank_array(y)

    n = len(rx)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_x = sum((rx[i] - mean_rx) ** 2 for i in range(n))
    var_y = sum((ry[i] - mean_ry) ** 2 for i in range(n))

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-12:
        return 0.0

    return cov / denom


# ── Standardised effect size ─────────────────────────────────────────────

def cluster_effect_size(
    group_a: np.ndarray,
    group_b: np.ndarray,
    hedges_correct: bool = True,
) -> dict:
    """Cohen's d (and optionally Hedges' g) between two groups.

    Returns dict with d, hedges_g, ci_lower, ci_upper, magnitude, n_a, n_b.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    n_a, n_b = len(a), len(b)
    df = n_a + n_b - 2

    if df < 1 or n_a < 1 or n_b < 1:
        return {"d": 0.0, "hedges_g": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "magnitude": "negligible", "n_a": n_a, "n_b": n_b}

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a = float(np.var(a, ddof=1)) if n_a > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if n_b > 1 else 0.0

    pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / df)
    if pooled_sd < 1e-12:
        return {"d": 0.0, "hedges_g": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "magnitude": "negligible", "n_a": n_a, "n_b": n_b}

    d = (mean_a - mean_b) / pooled_sd

    # Hedges' correction factor J
    j = 1.0 - 3.0 / (4.0 * df - 1.0) if hedges_correct else 1.0
    g = d * j

    # SE of d (Borenstein et al., 2009)
    se = math.sqrt((n_a + n_b) / (n_a * n_b) + d * d / (2.0 * (n_a + n_b)))
    ci_lower = d - 1.96 * se
    ci_upper = d + 1.96 * se

    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        "d": round(d, 6),
        "hedges_g": round(g, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "magnitude": magnitude,
        "n_a": n_a,
        "n_b": n_b,
    }


# ── Analysis 1: Bootstrap CI ─────────────────────────────────────────────

# Fewest scored papers a bootstrap CI is computed from.
MIN_BOOTSTRAP_PAPERS = 3
# Resamples drawn per block, so memory stays O(chunk * n) rather than
# O(n_iter * n) (audit 12-L1). Blocks consume the generator in order, so the
# result equals one (n_iter, n) draw.
BOOTSTRAP_CHUNK = 1000


def _usable_scores(papers: list[dict]) -> tuple[np.ndarray, int, int]:
    """Parsed relevance scores plus the counts of missing and unparseable ones."""
    scores: list[float] = []
    n_unscored = n_unparseable = 0
    for paper in papers:
        value = _parse_relevance_score(paper)
        if value is not None:
            scores.append(value)
        elif _is_missing_score(paper):
            n_unscored += 1
        else:
            n_unparseable += 1
    return np.asarray(scores, dtype=np.float64), n_unscored, n_unparseable


def _bootstrap_means(scores: np.ndarray, n_iter: int, seed: int,
                     check_cancel: CancelCheck | None) -> np.ndarray:
    """Means of ``n_iter`` resamples of ``scores``, drawn in bounded blocks."""
    rng = np.random.default_rng(seed)
    n = scores.size
    blocks: list[np.ndarray] = []
    for start in range(0, n_iter, BOOTSTRAP_CHUNK):
        if check_cancel is not None:
            check_cancel()
        size = min(BOOTSTRAP_CHUNK, n_iter - start)
        blocks.append(scores[rng.integers(0, n, size=(size, n))].mean(axis=1))
    return np.concatenate(blocks)


def bootstrap_ci(papers: list[dict], n_iter: int = 10000,
                 seed: int = 42,
                 check_cancel: CancelCheck | None = None) -> dict:
    """Bootstrap confidence intervals for the mean relevance score.

    Resamples the papers that have a usable relevance score with replacement
    ``n_iter`` times and computes the mean score of each resample. Papers
    without a score (``n_unscored``) or with an unparseable one
    (``n_unparseable``) are excluded rather than imputed as 0, which used to
    pull the mean and CI down (audit 12-L8). Returns the overall mean,
    2.5th/97.5th percentile CI, standard deviation, and metadata.
    """
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}")
    logger.info("bootstrap_ci: START — %d papers, %d iterations", len(papers), n_iter)

    n = len(papers)
    scores, n_unscored, n_unparseable = _usable_scores(papers)
    n_scored = int(scores.size)
    counts = {"n_papers": n, "n_scored": n_scored,
              "n_unscored": n_unscored, "n_unparseable": n_unparseable}
    # A skipped result carries None, not 0.0: GRADE and Stage 10 read these
    # bounds, and a fake [0, 0] CI reads as perfect precision.
    skip_reason = ""
    if n_scored == 0:
        skip_reason = (
            "no paper has a usable relevance_score (run Stage 2 Score first); "
            "a CI over default zeros would be meaningless"
        )
    elif n_scored < MIN_BOOTSTRAP_PAPERS:
        skip_reason = (
            f"only {n_scored} paper(s) with a usable relevance_score; "
            f"bootstrap needs at least {MIN_BOOTSTRAP_PAPERS}"
        )
    if skip_reason:
        logger.warning("bootstrap_ci: SKIPPED — %s", skip_reason)
        return {
            "mean": None, "ci_lower": None, "ci_upper": None, "std": None,
            **counts, "n_iterations": 0, "skipped": True, "reason": skip_reason,
        }

    means = _bootstrap_means(scores, n_iter, seed, check_cancel)

    result = {
        "mean": float(np.mean(means)),
        "ci_lower": float(np.percentile(means, 2.5)),
        "ci_upper": float(np.percentile(means, 97.5)),
        "std": float(np.std(means)),
        **counts,
        "n_iterations": n_iter,
        "skipped": False,
    }
    logger.info(
        "bootstrap_ci: DONE — mean=%.3f, 95%% CI=[%.3f, %.3f]",
        result["mean"], result["ci_lower"], result["ci_upper"],
    )
    return result


def check_cancel_every(check_cancel: CancelCheck | None, iteration: int) -> None:
    """Call ``check_cancel`` every ``CANCEL_CHECK_INTERVAL`` loop iterations."""
    if check_cancel is not None and iteration % CANCEL_CHECK_INTERVAL == 0:
        check_cancel()
