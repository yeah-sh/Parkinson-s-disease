"""How well out-of-fold probabilities are calibrated, and the rule that says whether to rescale.

A probability is calibrated when, among recordings given about p, about p are positive. Three
measures are taken, each with a bootstrap interval over participants (predictions fixed):

- the Brier score, the mean squared distance of the probability from the 0/1 label, next to
  that of the reference that predicts the prevalence for everyone;
- the calibration slope, from a logistic regression of the label on logit(p): 1 when
  probabilities spread as they should, below 1 when they are too extreme (overfitting);
- calibration in the large, the intercept of that regression with the slope held at 1: 0 when
  probabilities are right on average.

The rule was fixed on 2026-10-26, before any classifier was run on the data: a stage is poorly
calibrated if its Brier score is not below the reference's, or the interval of its slope
leaves out 1, or that of its intercept leaves out 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import expit, log_expit

_EPS = 1e-6


def brier(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Mean squared difference between probability and label, rows weighted by ``weights``."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.average((p - y) ** 2, weights=weights))


def reference_brier(y: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Brier score of predicting the prevalence for every row."""
    y = np.asarray(y, dtype=float)
    return brier(y, np.full(y.shape, np.average(y, weights=weights)), weights)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p) - np.log1p(-p)


def _log_likelihood(y: np.ndarray, eta: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * (y * log_expit(eta) + (1 - y) * log_expit(-eta))))


def recalibration(
    y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, float]:
    """Calibration in the large and calibration slope of probabilities ``p`` for labels ``y``.

    The slope is that of the weighted logistic regression of ``y`` on logit(p); calibration in
    the large is the intercept a of logit P(y) = a + logit(p). Both are maximum likelihood fits;
    with labels that the probabilities separate perfectly they run large, as they should.
    """
    y = np.asarray(y, dtype=float)
    x = _logit(p)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    if len(set(y)) < 2:
        return np.nan, np.nan

    def loss(ab: np.ndarray) -> float:
        return -_log_likelihood(y, ab[0] + ab[1] * x, w)

    def gradient(ab: np.ndarray) -> np.ndarray:
        residual = w * (y - expit(ab[0] + ab[1] * x))
        return -np.array([residual.sum(), (residual * x).sum()])

    fit = optimize.minimize(loss, x0=np.array([0.0, 1.0]), jac=gradient, method="BFGS")
    large = optimize.minimize_scalar(
        lambda a: -_log_likelihood(y, a + x, w), bounds=(-20.0, 20.0), method="bounded"
    )
    return float(large.x), float(fit.x[1])


def reliability_table(
    y: np.ndarray, p: np.ndarray, n_bins: int, weights: np.ndarray | None = None
) -> pd.DataFrame:
    """Recordings in ``n_bins`` bins of equal count by probability: mean probability, observed
    share of positives, and a 95 % Wilson interval of that share.

    With weights the share is weighted and the interval uses the summed weight as its count
    (participants, for a task weighted per participant).
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(p, kind="stable")
    rows = []
    for index, chosen in enumerate(np.array_split(order, n_bins)):
        if not chosen.size:
            continue
        n = float(w[chosen].sum())
        share = float(np.average(y[chosen], weights=w[chosen]))
        z = 1.959963984540054
        centre = (share + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * np.sqrt(share * (1 - share) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        rows.append(
            {
                "bin": index,
                "n_recordings": int(chosen.size),
                "weight": n,
                "mean_probability": float(np.average(p[chosen], weights=w[chosen])),
                "observed": share,
                # The interval holds the share; clipping undoes rounding at 0 and 1.
                "observed_low": min(share, max(0.0, centre - half)),
                "observed_high": max(share, min(1.0, centre + half)),
            }
        )
    return pd.DataFrame(rows)


CALIBRATION_METRICS = ("brier", "reference_brier", "brier_skill", "intercept", "slope")


def calibration_values(
    y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, float]:
    """The Brier score, its reference, the skill 1 - Brier / reference, intercept and slope."""
    score, reference = brier(y, p, weights), reference_brier(y, weights)
    intercept, slope = recalibration(y, p, weights)
    return {
        "brier": score,
        "reference_brier": reference,
        "brier_skill": 1.0 - score / reference if reference > 0 else np.nan,
        "intercept": intercept,
        "slope": slope,
    }


def calibration_summary(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray | None,
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """:func:`calibration_values` and their percentile intervals, resampling participants."""
    y, p, groups = np.asarray(y, dtype=float), np.asarray(p, dtype=float), np.asarray(groups)
    names = np.unique(groups)
    rows = {name: np.flatnonzero(groups == name) for name in names}
    draws = []
    for _ in range(n_resamples):
        chosen = np.concatenate([rows[name] for name in rng.choice(names, size=names.size)])
        w = None if weights is None else weights[chosen]
        draws.append(calibration_values(y[chosen], p[chosen], w))
    table = pd.DataFrame(draws)
    tail = 100.0 * (1.0 - confidence) / 2.0
    observed = calibration_values(y, p, weights)
    return {
        name: {
            "value": observed[name],
            "low": float(np.nanpercentile(table[name], tail)),
            "high": float(np.nanpercentile(table[name], 100.0 - tail)),
        }
        for name in CALIBRATION_METRICS
    }


def miscalibration(summary: dict[str, dict[str, float]]) -> list[str]:
    """The reasons the rule fixed in advance finds in ``summary``; none means well calibrated."""
    reasons = []
    if not summary["brier"]["value"] < summary["reference_brier"]["value"]:
        reasons.append("Brier score not below that of predicting the prevalence")
    slope = summary["slope"]
    if not slope["low"] <= 1.0 <= slope["high"]:
        reasons.append("interval of the calibration slope leaves out 1")
    intercept = summary["intercept"]
    if not intercept["low"] <= 0.0 <= intercept["high"]:
        reasons.append("interval of calibration in the large leaves out 0")
    return reasons


def multiclass_brier(
    probabilities: np.ndarray, label: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Brier score of class probabilities (rows sum to 1) for integer labels: 0 is perfect, 2
    the worst."""
    probabilities = np.asarray(probabilities, dtype=float)
    onehot = np.eye(probabilities.shape[1])[np.asarray(label)]
    return float(np.average(((probabilities - onehot) ** 2).sum(axis=1), weights=weights))
