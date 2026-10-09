"""Surrogate series, for testing where the multifractality of a signal comes from.

A shuffled surrogate keeps only the amplitude distribution: every correlation is destroyed. An
IAAFT surrogate also keeps the power spectrum, and with it the linear correlations, while the
Fourier phases are randomised, which removes the nonlinear structure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _as_series(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError(f"x must be one-dimensional with at least 2 samples, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or infinity")
    return x


def shuffle_surrogate(x: ArrayLike, rng: np.random.Generator) -> np.ndarray:
    """A random permutation of ``x``: the same values, in an order that carries no memory."""
    return rng.permutation(_as_series(x))


def iaaft_surrogate(x: ArrayLike, rng: np.random.Generator, max_iter: int = 100) -> np.ndarray:
    """Iterative amplitude-adjusted Fourier transform surrogate (Schreiber and Schmitz, 1996).

    Starting from a shuffle of ``x``, two steps alternate: the Fourier amplitudes are replaced
    by those of ``x`` with the phases kept, then the values are replaced by those of ``x`` in
    the same rank order. The iteration stops when the rank order no longer changes, or after
    ``max_iter`` rounds. It ends on the second step, so the surrogate is exactly a permutation
    of ``x`` and its power spectrum matches that of ``x`` closely but not exactly.

    How closely depends on the amplitude distribution. Gaussian data converges within about
    100 rounds to a spectrum that is off by under 0.1 %; skewed data converges more slowly and
    is off by a few percent after 100 rounds. For a series in which a few extreme values carry
    most of the power, such as a multiplicative cascade, no ordering of the values has the
    right spectrum and the mismatch stays large.
    """
    x = _as_series(x)
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1, got {max_iter}")
    sorted_x = np.sort(x)
    amplitudes = np.abs(np.fft.rfft(x))
    surrogate = rng.permutation(x)
    order = None
    for _ in range(max_iter):
        phases = np.angle(np.fft.rfft(surrogate))
        spectral = np.fft.irfft(amplitudes * np.exp(1j * phases), n=x.size)
        previous, order = order, np.argsort(spectral)
        surrogate[order] = sorted_x
        if previous is not None and np.array_equal(order, previous):
            break
    return surrogate
