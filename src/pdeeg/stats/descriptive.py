"""Bootstrap intervals of means, and the smallest effect a test of a given size can see."""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import brentq


def bootstrap_mean_ci(
    values: np.ndarray, n_resamples: int, confidence: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap interval of the mean along the first axis.

    The rows of ``values`` (participants, or pairs) are drawn with replacement; whatever
    follows the first axis is kept together, so a participant's curve is resampled whole.
    """
    values = np.asarray(values, dtype=float)
    draws = rng.integers(0, values.shape[0], size=(n_resamples, values.shape[0]))
    means = values[draws].mean(axis=1)
    tail = 100.0 * (1.0 - confidence) / 2.0
    low, high = np.percentile(means, [tail, 100.0 - tail], axis=0)
    return low, high


def detectable_effect(n_a: int, n_b: int | None = None, power: float = 0.8, alpha: float = 0.05):
    """Smallest standardised effect a two-sided t-test detects with probability ``power``.

    Two groups of ``n_a`` and ``n_b`` (Cohen's d), or ``n_a`` paired differences when ``n_b``
    is None (d_z). It is a single test at one channel: a guide to what this sample can show,
    not the power of the cluster test.
    """
    if n_b is None:
        df, scale = n_a - 1, np.sqrt(n_a)
    else:
        df, scale = n_a + n_b - 2, np.sqrt(n_a * n_b / (n_a + n_b))
    critical = scipy_stats.t.ppf(1.0 - alpha / 2.0, df)

    def shortfall(effect: float) -> float:
        noncentrality = effect * scale
        # The far tail on the other side is all but 0 for a positive effect; SciPy returns NaN
        # for it when the noncentrality is large.
        other_side = np.nan_to_num(scipy_stats.nct.cdf(-critical, df, noncentrality), nan=0.0)
        return scipy_stats.nct.sf(critical, df, noncentrality) + other_side - power

    return float(brentq(shortfall, 1e-6, 20.0))
