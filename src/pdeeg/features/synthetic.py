"""Synthetic series whose scaling exponents are known exactly, for validating the MFDFA code."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def fgn(n_samples: int, hurst: float, rng: np.random.Generator) -> np.ndarray:
    """Fractional Gaussian noise with Hurst exponent ``hurst``, zero mean and unit variance.

    Uses the Davies-Harte method: the autocovariance, laid out as the first row of a circulant
    matrix of size ``2 * n_samples``, is diagonalised by the FFT, so the sample has exactly the
    covariance of fractional Gaussian noise rather than an approximation to it. White noise is
    the case ``hurst = 0.5``.
    """
    if not 0.0 < hurst < 1.0:
        raise ValueError(f"hurst must lie strictly between 0 and 1, got {hurst}")
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2, got {n_samples}")
    lags = np.arange(n_samples + 1, dtype=float)
    autocov = 0.5 * (
        (lags + 1) ** (2 * hurst) - 2 * lags ** (2 * hurst) + np.abs(lags - 1) ** (2 * hurst)
    )
    eigenvalues = np.fft.fft(np.concatenate([autocov, autocov[-2:0:-1]])).real
    # Non-negative in exact arithmetic for every Hurst exponent; rounding can leave -1e-16.
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    size = eigenvalues.size
    noise = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    return np.fft.fft(np.sqrt(eigenvalues / size) * noise).real[:n_samples]


def power_law_noise(n_samples: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise whose power spectrum falls as ``1 / f ** beta``, with unit variance.

    White noise shaped in the frequency domain: ``beta = 0`` leaves it white, 1 makes it pink
    and 2 brown. Unlike :func:`fgn` it is not confined to stationary series, and its exponent
    is ``h(2) = (beta + 1) / 2`` as long as the detrending order is above ``(beta - 1) / 2``.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be at least 2, got {n_samples}")
    freqs = np.fft.rfftfreq(n_samples)
    shape = np.zeros(freqs.size)
    shape[1:] = freqs[1:] ** (-beta / 2.0)
    series = np.fft.irfft(np.fft.rfft(rng.standard_normal(n_samples)) * shape, n_samples)
    return series / series.std()


def binomial_cascade(n_levels: int, a: float) -> np.ndarray:
    """Binomial multiplicative cascade of length ``2 ** n_levels`` (Kantelhardt et al., eq. 18).

    Sample ``k`` is ``a ** ones(k) * (1 - a) ** (n_levels - ones(k))``, where ``ones(k)`` counts
    the 1 digits in the binary form of ``k``. The series is deterministic and sums to 1.
    """
    if not 0.5 < a < 1.0:
        raise ValueError(f"a must lie strictly between 0.5 and 1, got {a}")
    ones = np.bitwise_count(np.arange(2**n_levels))
    return a**ones * (1.0 - a) ** (n_levels - ones)


def binomial_cascade_spectrum(qs: ArrayLike, a: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact ``h(q)``, ``alpha(q)`` and ``f(alpha(q))`` of the binomial cascade.

    From ``tau(q) = -log2(a ** q + (1 - a) ** q)`` (Kantelhardt et al., eq. 19) with
    ``h = (tau + 1) / q``, taken to its limit ``-log2(a (1 - a)) / 2`` at ``q = 0``.
    """
    qs = np.asarray(qs, dtype=float)
    weight_a, weight_b = a**qs, (1.0 - a) ** qs
    tau = -np.log2(weight_a + weight_b)
    alpha = -(weight_a * np.log2(a) + weight_b * np.log2(1.0 - a)) / (weight_a + weight_b)
    zero = qs == 0
    h = np.where(zero, -0.5 * np.log2(a * (1.0 - a)), (tau + 1.0) / np.where(zero, 1.0, qs))
    return h, alpha, qs * alpha - tau
