"""Stage 8 robustness: sensitivity, permutation and co-occurring condition analyses.

Split out of stages/s9_robustness.py, which re-exports these names.
"""

import logging
import math
import random

import numpy as np

from stages.s9_robustness_config import (
    DEFAULT_OUTCOME,
    RobustnessOutcome,
    _extract_conditions,
    _get_relevance_score,
)
from stages.s9_robustness_stats import (
    CancelCheck,
    _spearman_rank_correlation,
    check_cancel_every,
    cluster_effect_size,
)

# Same logger name as the original single module (stages.s9_robustness).
logger = logging.getLogger("stages.s9_robustness")


# Significance level of the condition-gradient permutation test.
PERMUTATION_ALPHA = 0.05

# The gradient test correlates support with an explicit, ordered condition
# list; a pattern pack's dict order is not one (audit 12-M5).
NO_RANKING_REASON = (
    "no ordered co_occurring_conditions list configured; the gradient test "
    "needs an explicit condition order"
)


def _min_attainable_p(n_conditions: int) -> float:
    """Smallest two-sided permutation p for a Spearman r over k conditions.

    Only the identity and reversed orders reach |r| = 1, so p >= 2 / k!.
    With k <= 4 this exceeds 0.05 and the test cannot reject (audit 12-M2).
    """
    return min(1.0, 2.0 / math.factorial(n_conditions))


# ── Analysis 3: Sensitivity analysis ──────────────────────────────────────

def sensitivity_analysis(papers: list[dict],
                         outcome: RobustnessOutcome = DEFAULT_OUTCOME) -> dict:
    """Test how results change at different relevance thresholds (1–5).

    For each threshold, counts papers passing, computes the percentage
    supporting the configured outcome, and the mean relevance score of passing papers.

    A threshold no paper passes has no support rate: its row is marked
    ``evaluable: False`` (``pct_supporting``/``mean_score`` None) and it takes
    no part in the stability verdict.  ``stable`` is None when no threshold
    is evaluable.
    """
    logger.info("sensitivity_analysis: START — %d papers", len(papers))

    thresholds: dict[int, dict] = {}
    prev_direction: bool | None = None
    direction_flipped = False

    for thresh in range(1, 6):
        passing = [p for p in papers if _get_relevance_score(p) >= thresh]
        n_pass = len(passing)

        if n_pass == 0:
            thresholds[thresh] = {
                "n_pass": 0, "pct_supporting": None, "mean_score": None,
                "evaluable": False,
            }
            continue

        n_support = sum(1 for p in passing if outcome.supports(p))
        pct_support = (n_support / n_pass) * 100.0
        mean_score = sum(_get_relevance_score(p) for p in passing) / n_pass
        row = {
            "n_pass": n_pass,
            "pct_supporting": round(pct_support, 2),
            "mean_score": round(mean_score, 3),
            "evaluable": True,
        }

        # Effect size: passing vs failing papers on relevance scores
        failing = [p for p in papers if _get_relevance_score(p) < thresh]
        if n_pass >= 2 and len(failing) >= 2:
            pass_scores = np.array([_get_relevance_score(p) for p in passing],
                                   dtype=np.float64)
            fail_scores = np.array([_get_relevance_score(p) for p in failing],
                                   dtype=np.float64)
            row["effect_size"] = cluster_effect_size(pass_scores, fail_scores)

        thresholds[thresh] = row

        # Track whether the majority direction flips between evaluable thresholds
        current_above_50 = pct_support > 50.0
        if prev_direction is not None and current_above_50 != prev_direction:
            direction_flipped = True
        prev_direction = current_above_50

    stable = None if prev_direction is None else not direction_flipped

    result = {
        "thresholds": thresholds,
        "stable": stable,
        "skipped": False,
    }
    logger.info(
        "sensitivity_analysis: DONE — stable=%s, thresholds tested=5",
        stable,
    )
    return result


# ── Analysis 4: Permutation test ──────────────────────────────────────────

def _condition_gradient(papers: list[dict],
                        patterns: dict[str, str],
                        ranking: list[str],
                        min_n: int = 3,
                        outcome: RobustnessOutcome = DEFAULT_OUTCOME) -> dict:
    """Compute per-condition support rates.

    Returns a dict mapping condition name to
    ``{"n_papers": int, "support_rate": float}``.
    Conditions with fewer than ``min_n`` papers are excluded.
    """
    counts: dict[str, dict] = {}
    for cond in ranking:
        counts[cond] = {"total": 0, "supporting": 0}

    for paper in papers:
        conds = _extract_conditions(paper, patterns)
        supports = outcome.supports(paper)
        for c in conds:
            if c in counts:
                counts[c]["total"] += 1
                if supports:
                    counts[c]["supporting"] += 1

    gradient: dict[str, dict] = {}
    for cond in ranking:
        total = counts[cond]["total"]
        if total >= min_n:
            rate = counts[cond]["supporting"] / total
            gradient[cond] = {
                "n_papers": total,
                "support_rate": round(rate, 4),
            }
        elif total > 0:
            logger.info(
                "permutation_test: skipping %s (n=%d < min_n=%d)",
                cond, total, min_n,
            )

    return gradient


def permutation_test(papers: list[dict],
                     patterns: dict[str, str],
                     ranking: list[str],
                     n_perm: int = 10000,
                     seed: int = 42,
                     outcome: RobustnessOutcome = DEFAULT_OUTCOME,
                     check_cancel: CancelCheck | None = None) -> dict:
    """Permutation test on the condition–support gradient.

    Tests whether the observed Spearman correlation between the configured
    condition ranking and each condition's support rate is
    significant by permuting condition labels.

    The p-value is ``(n_extreme + 1) / (n_perm + 1)`` so it is never 0, and
    the result is flagged ``underpowered`` when even a perfect gradient could
    not reach ``PERMUTATION_ALPHA`` with this many conditions.
    """
    if n_perm < 1:
        raise ValueError(f"n_perm must be at least 1, got {n_perm}")
    logger.info("permutation_test: START — %d papers, %d permutations", len(papers), n_perm)

    gradient = _condition_gradient(papers, patterns, ranking,
                                   outcome=outcome)

    if len(gradient) < 3:
        reason = (
            NO_RANKING_REASON if not ranking else
            f"only {len(gradient)} condition(s) had enough papers; "
            f"the gradient test needs at least 3"
        )
        logger.warning("permutation_test: SKIPPED — %s", reason)
        # None, not r=0/p=1: Stage 10 cross-checks abstract statistics
        # against these values, and zeros would look like a real null result.
        return {
            "observed_correlation": None,
            "p_value": None,
            "n_permutations": 0,
            "significant": False,
            "condition_gradient": gradient,
            "skipped": True,
            "reason": reason,
        }

    # Build ordered vectors: condition rank (1-based) vs support rate.
    # Only include conditions present in the gradient.
    ranks = []
    rates = []
    for i, cond in enumerate(ranking):
        if cond in gradient:
            ranks.append(float(i + 1))
            rates.append(gradient[cond]["support_rate"])

    observed_r = _spearman_rank_correlation(ranks, rates)

    # Permutation: shuffle rates, recompute correlation
    rng = random.Random(seed)
    n_extreme = 0
    for iteration in range(n_perm):
        check_cancel_every(check_cancel, iteration)
        shuffled = list(rates)
        rng.shuffle(shuffled)
        perm_r = _spearman_rank_correlation(ranks, shuffled)
        if abs(perm_r) >= abs(observed_r):
            n_extreme += 1

    # +1 counts the observed labelling itself (Phipson & Smyth 2010).
    p_value = (n_extreme + 1) / (n_perm + 1)
    min_p = _min_attainable_p(len(rates))

    result = {
        "observed_correlation": round(observed_r, 4),
        "p_value": round(p_value, 4),
        "n_permutations": n_perm,
        "significant": p_value < PERMUTATION_ALPHA,
        "min_attainable_p": round(min_p, 4),
        "underpowered": min_p >= PERMUTATION_ALPHA,
        "condition_gradient": gradient,
        "skipped": False,
    }
    logger.info(
        "permutation_test: DONE — r=%.4f, p=%.4f, significant=%s",
        observed_r, p_value, result["significant"],
    )
    return result


# ── Analysis 6: Co-occurring condition sensitivity ────────────────────────

# Support-rate change (absolute) at which a condition is reported as
# materially shifting the pooled result.
MATERIAL_RATE_SHIFT = 0.05


def _support_rate(subset: list[dict],
                  outcome: RobustnessOutcome = DEFAULT_OUTCOME) -> float:
    if not subset:
        return 0.0
    return sum(1 for p in subset if outcome.supports(p)) / len(subset)


def _single_condition_sensitivity(condition: str, with_cond: list[dict],
                                  without_cond: list[dict],
                                  outcome: RobustnessOutcome = DEFAULT_OUTCOME) -> dict:
    """Compare papers that mention ``condition`` against the rest of the corpus.

    Both groups must be non-empty: an empty group has no support rate, and
    treating it as 0% fabricated a shift equal to the other group's rate
    (audit 12-M1).
    """
    reason = ""
    if not with_cond:
        reason = f"no papers mention {condition!r}"
    elif not without_cond:
        reason = (f"every paper mentions {condition!r}, so there is no "
                  "comparison group; not comparable")
    if reason:
        return {"skipped": True, "n_with": len(with_cond),
                "n_without": len(without_cond), "reason": reason}

    rate_with = _support_rate(with_cond, outcome)
    rate_without = _support_rate(without_cond, outcome)
    diff = rate_with - rate_without
    es = cluster_effect_size(
        np.array([_get_relevance_score(p) for p in with_cond], dtype=np.float64),
        np.array([_get_relevance_score(p) for p in without_cond], dtype=np.float64),
    )
    shifts = abs(diff) >= MATERIAL_RATE_SHIFT
    return {
        "skipped": False,
        "n_with": len(with_cond),
        "n_without": len(without_cond),
        "support_rate_with": round(rate_with, 4),
        "support_rate_without": round(rate_without, 4),
        "rate_difference": round(diff, 4),
        "effect_size": es,
        "hedges_g": round(es.get("hedges_g", 0.0), 4),
        "interpretation": (
            f"Papers mentioning {condition} {'substantially' if shifts else 'do not substantially'}"
            f"{' shift' if shifts else ' alter'} the observed support rate"
        ),
    }


def condition_sensitivity_summary(sens: dict) -> str:
    """One-line summary for progress and GUI logs."""
    if not sens:
        return "Condition sensitivity: not run"
    if sens.get("skipped"):
        return f"Condition sensitivity skipped ({sens.get('reason', '')})"
    largest = sens.get("largest_shift")
    if not largest:
        return "Condition sensitivity: no configured condition appears in the corpus"
    res = sens["conditions"][largest]
    return (
        f"Condition sensitivity: {sens['n_conditions_tested']} tested; largest shift "
        f"{largest} ({res['rate_difference']:+.1%} support, g={res['hedges_g']:.3f})"
    )


def condition_sensitivity_analysis(papers: list[dict],
                                   patterns: dict[str, str],
                                   ranking: list[str],
                                   outcome: RobustnessOutcome = DEFAULT_OUTCOME) -> dict:
    """Test whether any configured co-occurring condition drives the result.

    For each condition in ``ranking``, splits the corpus into papers that
    mention it (via the same pattern matching as the other condition
    analyses) and papers that do not, then reports both support rates and
    Hedges' g on relevance scores. A large shift means the pooled finding
    depends on that condition's literature. Conditions whose "with" or
    "without" group is empty are reported as skipped and never become the
    ``largest_shift``.
    """
    if not ranking:
        return {"skipped": True,
                "reason": "no conditions configured (co_occurring_conditions "
                          "or graph_analysis.conditions)"}

    paper_conditions = [set(_extract_conditions(p, patterns)) for p in papers]
    per_condition: dict[str, dict] = {}
    for condition in ranking:
        with_cond = [p for p, conds in zip(papers, paper_conditions) if condition in conds]
        without_cond = [p for p, conds in zip(papers, paper_conditions) if condition not in conds]
        per_condition[condition] = _single_condition_sensitivity(
            condition, with_cond, without_cond, outcome
        )

    tested = {name: res for name, res in per_condition.items() if not res["skipped"]}
    largest = max(tested, key=lambda name: abs(tested[name]["rate_difference"]), default=None)
    logger.info("condition_sensitivity: tested %d/%d conditions, largest shift=%s",
                len(tested), len(ranking), largest)
    return {
        "skipped": False,
        "conditions": per_condition,
        "n_conditions_tested": len(tested),
        "largest_shift": largest,
    }
