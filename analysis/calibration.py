"""Confidence calibration with held-out Expected Calibration Error.

Split out of ``analysis.scoring`` (audit 03-L10); ``calibrate_confidence`` is
still importable from there.  The ECE is cross-validated: each score is
calibrated by a model fitted without it, so a calibrator that memorises its
training data cannot report perfect calibration (03-M5).
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_METHODS = ("isotonic", "platt")
MIN_CALIBRATION_SAMPLES = 3
CROSS_VALIDATION_FOLDS = 5


def _validated_arrays(
    predicted: list[float], observed: list[int], method: str, n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    if method not in CALIBRATION_METHODS:
        raise ValueError(
            f"Unknown calibration method {method!r}; use one of {CALIBRATION_METHODS}"
        )
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    if len(predicted) != len(observed):
        raise ValueError(
            f"predicted and observed differ in length ({len(predicted)} vs {len(observed)})"
        )
    pred = np.asarray(predicted, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64)
    if not np.all(np.isfinite(pred)) or np.any((pred < 0) | (pred > 1)):
        raise ValueError("predicted scores must be finite values between 0 and 1")
    if not np.all(np.isin(obs, (0.0, 1.0))):
        raise ValueError("observed labels must be 0 or 1")
    return pred, obs


def _fit_platt(pred: np.ndarray, obs: np.ndarray) -> tuple[float, float, bool]:
    """Fit P(y=1|f) = 1/(1+exp(a*f+b)); return a, b and whether it converged."""
    from scipy.optimize import minimize

    def _neg_log_likelihood(params: np.ndarray) -> float:
        a, b = params
        p = np.clip(1.0 / (1.0 + np.exp(a * pred + b)), 1e-10, 1 - 1e-10)
        return float(-np.sum(obs * np.log(p) + (1 - obs) * np.log(1 - p)))

    result = minimize(_neg_log_likelihood, [0.0, 0.0], method="Nelder-Mead")
    a_opt, b_opt = result.x
    return float(a_opt), float(b_opt), bool(result.success)


def _fit_and_apply(
    method: str, train_pred: np.ndarray, train_obs: np.ndarray, target: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Fit *method* on the training scores and calibrate *target*."""
    if method == "platt":
        a, b, converged = _fit_platt(train_pred, train_obs)
        return 1.0 / (1.0 + np.exp(a * target + b)), converged

    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(out_of_bounds="clip").fit(train_pred, train_obs)
    return model.predict(target), True


def _held_out_predictions(method: str, pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Cross-validated calibrated scores: each fold is calibrated by the others."""
    folds = min(CROSS_VALIDATION_FOLDS, len(pred))
    fold_of = np.arange(len(pred)) % folds
    held_out = np.empty_like(pred)
    for fold in range(folds):
        test = fold_of == fold
        held_out[test], _ = _fit_and_apply(method, pred[~test], obs[~test], pred[test])
    return np.clip(held_out, 0.0, 1.0)


def _expected_calibration_error(
    scores: np.ndarray, obs: np.ndarray, n_bins: int,
) -> tuple[float, list[dict]]:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_stats: list[dict] = []
    ece = 0.0
    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge in last bin
            mask = mask | (scores == bin_edges[i + 1])
        bin_count = int(np.sum(mask))
        if bin_count == 0:
            continue
        mean_pred = float(np.mean(scores[mask]))
        mean_obs = float(np.mean(obs[mask]))
        ece += (bin_count / len(scores)) * abs(mean_pred - mean_obs)
        bin_stats.append({
            "bin_center": round((bin_edges[i] + bin_edges[i + 1]) / 2, 2),
            "bin_count": bin_count,
            "mean_predicted": round(mean_pred, 4),
            "mean_observed": round(mean_obs, 4),
        })
    return ece, bin_stats


def calibrate_confidence(
    predicted: list[float],
    observed: list[int],
    method: str = "isotonic",
    n_bins: int = 10,
) -> dict:
    """Calibrate confidence scores against ground truth labels.

    Parameters
    ----------
    predicted : list of raw confidence scores (0-1)
    observed : list of binary ground truth labels (0 or 1)
    method : "isotonic" (default) or "platt" (logistic sigmoid)
    n_bins : number of bins for Expected Calibration Error

    Returns
    -------
    dict with:
        calibrated : list of calibrated probabilities (model fitted on all data)
        ece : cross-validated Expected Calibration Error (lower is better)
        method : method used
        converged : False when the Platt optimiser did not converge
        bin_stats : list of {bin_center, bin_count, mean_predicted, mean_observed}

    Raises ``ValueError`` for an unknown method, mismatched lengths, scores
    outside [0, 1] or labels other than 0/1.
    """
    pred, obs = _validated_arrays(predicted, observed, method, n_bins)
    if len(pred) < MIN_CALIBRATION_SAMPLES:
        return {
            "calibrated": list(predicted),
            "ece": 1.0,
            "method": method,
            "converged": True,
            "bin_stats": [],
        }

    calibrated, converged = _fit_and_apply(method, pred, obs, pred)
    if not converged:
        logger.warning("Platt calibration did not converge; calibrated scores are unreliable")
    calibrated = np.clip(calibrated, 0.0, 1.0)
    ece, bin_stats = _expected_calibration_error(
        _held_out_predictions(method, pred, obs), obs, n_bins,
    )
    return {
        "calibrated": [round(float(c), 6) for c in calibrated],
        "ece": round(float(ece), 6),
        "method": method,
        "converged": converged,
        "bin_stats": bin_stats,
    }
