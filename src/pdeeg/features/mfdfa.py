"""Multifractal detrended fluctuation analysis (MFDFA) of a one-dimensional series.

Follows Kantelhardt et al., "Multifractal detrended fluctuation analysis of nonstationary time
series", Physica A 316 (2002) 87-114:

1. Profile ``Y = cumsum(x - mean(x))``.
2. For each scale ``s``, cut ``Y`` into ``N // s`` non-overlapping windows counted from the start
   and as many counted from the end, so that no sample is left out when ``s`` does not divide N.
3. Fit a polynomial of order ``m`` in every window; ``F²(v, s)`` is the variance of the residual.
4. ``Fq(s) = mean(F²(v, s) ** (q / 2)) ** (1 / q)``, with the limit
   ``exp(0.5 * mean(log F²(v, s)))`` at ``q = 0``.
5. ``h(q)`` is the slope of ``log Fq(s)`` against ``log s``.
6. ``tau(q) = q h(q) - 1``, ``alpha = d tau / dq`` and ``f(alpha) = q alpha - tau``.

Nothing in this module knows about EEG or about the configuration: every parameter is an
argument, and ``configs/features/mfdfa.yaml`` holds the values the project uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

# Keys of the dictionary returned by spectrum_features.
FEATURE_NAMES = ("h2", "delta_h", "delta_alpha", "alpha0", "asymmetry", "min_r2")

# A straight line through fewer points than this has no meaningful R².
MIN_FIT_SCALES = 3


@dataclass(frozen=True)
class MfdfaResult:
    """What :func:`mfdfa` returns, for ``n_q`` moment orders and ``n_scales`` scales.

    ``Fq`` has shape (n_q, n_scales): row ``i`` is the fluctuation function of ``qs[i]``. It is
    NaN at a scale where every window was discarded. ``n_windows``, shape (n_scales,), is the
    number of windows kept at each scale. Everything else has shape (n_q,): the generalised
    Hurst exponent ``h``, the R² of the fit that gave it (``h_r2``), the ``intercept`` of that
    fit (``log Fq = intercept + h log s``, natural logarithms), the scaling exponent ``tau``,
    and the singularity spectrum as the pair ``alpha``, ``f_alpha``.
    """

    Fq: np.ndarray
    h: np.ndarray
    h_r2: np.ndarray
    tau: np.ndarray
    alpha: np.ndarray
    f_alpha: np.ndarray
    intercept: np.ndarray
    n_windows: np.ndarray


def make_qs(q_min: float, q_max: float, q_step: float) -> np.ndarray:
    """Moment orders from ``q_min`` to ``q_max`` inclusive, ``q_step`` apart."""
    if not q_step > 0:
        raise ValueError(f"q_step must be positive, got {q_step}")
    if not q_min < q_max:
        raise ValueError(f"q_min ({q_min}) must be below q_max ({q_max})")
    n_steps = round((q_max - q_min) / q_step)
    if not np.isclose(q_min + n_steps * q_step, q_max):
        raise ValueError(f"q_max - q_min is not a whole number of steps of {q_step}")
    # Rounded so that a grid such as -5, -4.9, ... holds exactly 0 and 2, not 2.0000000000000004.
    return np.round(q_min + q_step * np.arange(n_steps + 1), 12)


def make_scales(n_samples: int, smin: int, smax_frac: float, n_scales: int) -> np.ndarray:
    """Window sizes in samples, log-spaced from ``smin`` to ``smax_frac * n_samples``.

    The sizes are rounded to integers and duplicates are dropped, so fewer than ``n_scales``
    come back when the range is too narrow to hold that many different integers. The largest
    is rounded down, which leaves at least ``1 / smax_frac`` windows at every scale.
    """
    if n_scales < MIN_FIT_SCALES:
        raise ValueError(f"n_scales must be at least {MIN_FIT_SCALES}, got {n_scales}")
    if smin < 1:
        raise ValueError(f"smin must be at least 1, got {smin}")
    smax = int(n_samples * smax_frac)
    if smax <= smin:
        raise ValueError(
            f"largest scale ({smax} = {smax_frac} x {n_samples} samples) "
            f"does not exceed the smallest ({smin})"
        )
    return np.unique(np.round(np.geomspace(smin, smax, n_scales)).astype(int))


def _as_qs(qs: ArrayLike) -> np.ndarray:
    qs = np.asarray(qs, dtype=float)
    if qs.ndim != 1 or qs.size == 0:
        raise ValueError("qs must be a non-empty one-dimensional array")
    if not np.isfinite(qs).all():
        raise ValueError("qs contains NaN or infinity")
    if np.any(np.diff(qs) <= 0):
        raise ValueError("qs must be strictly increasing")
    return qs


def _as_scales(scales: ArrayLike, order: int, n_samples: int) -> np.ndarray:
    raw = np.asarray(scales)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("scales must be a non-empty one-dimensional array")
    scales = raw.astype(int)
    if not np.array_equal(scales, raw):
        raise ValueError("scales must be whole numbers of samples")
    if np.any(np.diff(scales) <= 0):
        raise ValueError("scales must be strictly increasing")
    # An order-m polynomial passes through m + 1 points exactly, leaving no residual.
    if scales[0] < order + 2:
        raise ValueError(f"order {order} needs scales of at least {order + 2}, got {scales[0]}")
    if scales[-1] > n_samples:
        raise ValueError(f"scale {scales[-1]} is longer than the series ({n_samples} samples)")
    return scales


def _window_variances(profile: np.ndarray, scale: int, order: int) -> np.ndarray:
    """Residual variance ``F²(v, s)`` of a polynomial fit in each window of ``scale`` samples.

    The ``N // scale`` windows counted from the start of the profile come first, followed by
    the same number counted from its end.
    """
    n_windows = profile.size // scale
    used = n_windows * scale
    windows = np.concatenate(
        [
            profile[:used].reshape(n_windows, scale),
            profile[profile.size - used :].reshape(n_windows, scale),
        ]
    )
    # Least squares for all windows at once: with an orthonormal polynomial basis the fit is a
    # projection. Legendre polynomials on a time axis scaled to [-1, 1] keep it well
    # conditioned whatever the window length and order.
    time = np.linspace(-1.0, 1.0, scale)
    basis, _ = np.linalg.qr(np.polynomial.legendre.legvander(time, order))
    residual = windows - (windows @ basis) @ basis.T
    return np.einsum("ij,ij->i", residual, residual) / scale


def _fluctuation(variances: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """``Fq(s)`` for every q, from the window variances ``F²(v, s)`` of one scale."""
    log_f2 = np.log(variances)
    zero = np.isclose(qs, 0.0)
    q = np.where(zero, 1.0, qs)
    # mean(F² ** (q/2)) ** (1/q), evaluated in logarithms so that no power can overflow.
    log_fq = (logsumexp(0.5 * q[:, None] * log_f2, axis=1) - np.log(log_f2.size)) / q
    log_fq[zero] = 0.5 * log_f2.mean()
    return np.exp(log_fq)


def _loglog_fit(scales: np.ndarray, fq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slope, intercept and R² of ``log fq`` against ``log scales``, one fit per row of ``fq``."""
    log_s = np.log(scales)
    log_f = np.log(fq)
    ds = log_s - log_s.mean()
    df = log_f - log_f.mean(axis=1, keepdims=True)
    slope = df @ ds / (ds @ ds)
    intercept = log_f.mean(axis=1) - slope * log_s.mean()
    residual = df - slope[:, None] * ds
    r2 = 1.0 - (residual**2).sum(axis=1) / (df**2).sum(axis=1)
    return slope, intercept, r2


def mfdfa(
    x: ArrayLike,
    scales: ArrayLike,
    qs: ArrayLike,
    order: int = 1,
    fit_range: tuple[float, float] | None = None,
    *,
    min_variance_ratio: float = 1e-12,
) -> MfdfaResult:
    """MFDFA of the series ``x`` at the window sizes ``scales`` and moment orders ``qs``.

    ``scales`` are whole numbers of samples and ``qs`` any real numbers; both must be strictly
    increasing. ``order`` is the order of the polynomial removed from each window. ``h(q)`` is
    fitted over the scales inside ``fit_range``, a (low, high) pair in samples with both ends
    included, or over all of them when it is None.

    A window whose residual variance is zero, or no more than ``min_variance_ratio`` times the
    median over the windows of its scale, is discarded: such a window is flat, and for negative
    q it would dominate the moment. Scales left without any window are left out of the fit.
    The spectrum (``alpha`` and ``f_alpha``) is NaN when there are fewer than three ``qs``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be one-dimensional, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or infinity")
    if order < 0:
        raise ValueError(f"order must not be negative, got {order}")
    if min_variance_ratio < 0:
        raise ValueError(f"min_variance_ratio must not be negative, got {min_variance_ratio}")
    qs = _as_qs(qs)
    scales = _as_scales(scales, order, x.size)

    profile = np.cumsum(x - x.mean())
    fq = np.full((qs.size, scales.size), np.nan)
    n_windows = np.zeros(scales.size, dtype=int)
    for i, scale in enumerate(scales):
        variances = _window_variances(profile, scale, order)
        variances = variances[variances > min_variance_ratio * np.median(variances)]
        n_windows[i] = variances.size
        if variances.size:
            fq[:, i] = _fluctuation(variances, qs)

    fit = n_windows > 0
    if fit_range is not None:
        low, high = fit_range
        fit &= (scales >= low) & (scales <= high)
    if fit.sum() < MIN_FIT_SCALES:
        raise ValueError(
            f"{fit.sum()} scale(s) left to fit h(q), need at least {MIN_FIT_SCALES}: check "
            "fit_range, and that the series is not constant"
        )
    h, intercept, r2 = _loglog_fit(scales[fit], fq[:, fit])

    tau = qs * h - 1.0
    alpha = np.full(qs.size, np.nan)
    if qs.size >= 3:
        # Second-order differences at the ends too: the spectrum width is read off there.
        alpha = np.gradient(tau, qs, edge_order=2)
    return MfdfaResult(
        Fq=fq,
        h=h,
        h_r2=r2,
        tau=tau,
        alpha=alpha,
        f_alpha=qs * alpha - tau,
        intercept=intercept,
        n_windows=n_windows,
    )


def spectrum_features(result: MfdfaResult, qs: ArrayLike) -> dict[str, float]:
    """Summarise an MFDFA result as scalar features; ``qs`` are the orders it was computed for.

    - ``h2``: ``h(2)``, the Hurst exponent of ordinary DFA (interpolated if 2 is not in ``qs``).
    - ``delta_h``: ``h(q_min) - h(q_max)``.
    - ``delta_alpha``: width of the singularity spectrum, ``max(alpha) - min(alpha)``.
    - ``alpha0``: the ``alpha`` at which ``f(alpha)`` peaks.
    - ``asymmetry``: ``(left - right) / (left + right)`` with ``left = alpha0 - min(alpha)`` and
      ``right = max(alpha) - alpha0``. It lies in [-1, 1], is 0 for a symmetric spectrum, and is
      positive when the left branch (q > 0, large fluctuations) is the longer one.
    - ``min_r2``: the worst R² among the ``h(q)`` fits; a quality measure, not a feature.
    """
    qs = _as_qs(qs)
    if qs.size != result.h.size:
        raise ValueError(f"{qs.size} qs for a result with {result.h.size} moment orders")
    if qs.size < 3:
        raise ValueError("the singularity spectrum needs at least three moment orders")
    if not qs[0] <= 2.0 <= qs[-1]:
        raise ValueError(f"h2 needs q = 2 inside the range of qs ({qs[0]} to {qs[-1]})")
    alpha0 = result.alpha[np.argmax(result.f_alpha)]
    left = alpha0 - result.alpha.min()
    right = result.alpha.max() - alpha0
    width = left + right
    features = {
        "h2": np.interp(2.0, qs, result.h),
        "delta_h": result.h[0] - result.h[-1],
        "delta_alpha": width,
        "alpha0": alpha0,
        "asymmetry": (left - right) / width if width > 0 else 0.0,
        "min_r2": result.h_r2.min(),
    }
    return {name: float(features[name]) for name in FEATURE_NAMES}
