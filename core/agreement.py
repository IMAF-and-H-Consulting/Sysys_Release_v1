"""Descriptive agreement on the original numeric rating scale."""
from __future__ import annotations

import logging
import numbers

logger = logging.getLogger(__name__)


def adjacent_agreement(scores_a: list[float], scores_b: list[float]) -> dict:
    """Percent of paired ratings within one point, including exact matches.

    Callers supply matched, valid ratings. No binning or rounding is applied
    before comparison. With no pairs the percentage is undefined (None).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Adjacent agreement requires equally sized paired ratings")
    total = len(scores_a)
    count = sum(abs(a - b) <= 1 for a, b in zip(scores_a, scores_b))
    return {
        "adjacent_agreement_pct": round(100 * count / total, 2) if total else None,
        "adjacent_agreement_count": count,
        "adjacent_agreement_total": total,
    }


def format_adjacent_agreement(summary: dict) -> str:
    """Render a percentage with its denominator, or N/A for absent ratings."""
    pct = summary.get("adjacent_agreement_pct")
    if pct is None:
        return "N/A"
    return (f"{pct:.2f}% ({summary['adjacent_agreement_count']}/"
            f"{summary['adjacent_agreement_total']})")


def _check_paired(labels_a: list, labels_b: list) -> None:
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Kappa needs paired ratings: got {len(labels_a)} and {len(labels_b)} values")


def cohens_kappa(labels_a: list, labels_b: list) -> float | None:
    """Compute Cohen's kappa for nominal agreement.

    κ = (p_o - p_e) / (1 - p_e)
    where p_o = observed agreement, p_e = expected agreement by chance.

    Raises ValueError for unpaired input. κ is undefined (None) with no
    pairs, or when chance agreement is certain (p_e = 1, one shared category).
    """
    _check_paired(labels_a, labels_b)
    if not labels_a:
        return None

    n = len(labels_a)
    categories = sorted(set(labels_a) | set(labels_b))

    if len(categories) <= 1:
        return None

    # Observed agreement
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n

    # Expected agreement by chance
    p_e = 0.0
    for cat in categories:
        count_a = sum(1 for x in labels_a if x == cat)
        count_b = sum(1 for x in labels_b if x == cat)
        p_e += (count_a / n) * (count_b / n)

    if abs(1.0 - p_e) < 1e-10:
        return None

    kappa = (p_o - p_e) / (1.0 - p_e)
    return kappa


def _ordinal_index(score, max_score: int) -> int:
    """A score as a confusion-matrix index; integral floats (3.0) are accepted.

    Scores outside 0..max_score are clamped with a WARNING.
    """
    is_number = isinstance(score, numbers.Real) and not isinstance(score, bool)
    if not is_number or not float(score).is_integer():
        raise ValueError(f"Weighted kappa needs integer scores, got {score!r}")
    value = int(score)
    if not 0 <= value <= max_score:
        logger.warning("Weighted kappa: score %s outside 0-%d clamped", value, max_score)
    return max(0, min(max_score, value))


def weighted_kappa(scores_a: list[int], scores_b: list[int],
                   max_score: int = 5) -> float | None:
    """Compute linearly weighted kappa for ordinal scores (0..max_score).

    κ_w = 1 - Σ w_ij·O_ij / Σ w_ij·E_ij with linear disagreement weights
    w_ij = |i - j| / max_score (equivalent to agreement weights
    1 - |i - j| / max_score). Raises ValueError for unpaired input or
    non-integer scores. κ_w is undefined (None) with no pairs, or with no
    expected disagreement (both raters use one shared category).
    """
    _check_paired(scores_a, scores_b)
    if not scores_a:
        return None

    n = len(scores_a)
    categories = list(range(max_score + 1))
    n_cat = len(categories)

    # Build observed confusion matrix
    observed = [[0] * n_cat for _ in range(n_cat)]
    for a, b in zip(scores_a, scores_b):
        observed[_ordinal_index(a, max_score)][_ordinal_index(b, max_score)] += 1

    # Marginals
    row_sums = [sum(observed[i]) for i in range(n_cat)]
    col_sums = [sum(observed[i][j] for i in range(n_cat)) for j in range(n_cat)]

    # Expected confusion matrix under independence
    expected = [[0.0] * n_cat for _ in range(n_cat)]
    for i in range(n_cat):
        for j in range(n_cat):
            expected[i][j] = (row_sums[i] * col_sums[j]) / n

    # Linear disagreement weights
    weights = [[abs(i - j) / max_score for j in range(n_cat)] for i in range(n_cat)]

    # Compute weighted observed and expected
    w_observed = sum(
        weights[i][j] * observed[i][j]
        for i in range(n_cat) for j in range(n_cat)
    ) / n

    w_expected = sum(
        weights[i][j] * expected[i][j]
        for i in range(n_cat) for j in range(n_cat)
    ) / n

    if abs(w_expected) < 1e-10:
        return None

    kappa = 1.0 - (w_observed / w_expected)
    return kappa


# Landis & Koch (1977): lower bound (exclusive) of each band, highest first.
_KAPPA_BANDS = (
    (0.80, "almost perfect"),
    (0.60, "substantial"),
    (0.40, "moderate"),
    (0.20, "fair"),
    (0.0, "slight"),
)


def interpret_kappa(kappa: float | None) -> str:
    """Landis & Koch band for a κ value; ``undefined`` when κ is None."""
    if kappa is None:
        return "undefined"
    for floor, label in _KAPPA_BANDS:
        if kappa > floor:
            return label
    return "poor"


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard index for set overlap."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
