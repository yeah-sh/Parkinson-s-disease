"""Amplitude envelope of a frequency band: band-pass, then the modulus of the analytic signal."""

from __future__ import annotations

import mne
import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import hilbert


def trim_samples(edge_trim: float, sfreq: float) -> int:
    """Samples that :func:`band_envelope` drops from each end for ``edge_trim`` seconds."""
    return int(round(edge_trim * sfreq))


def band_envelope(
    data: ArrayLike,
    sfreq: float,
    band: tuple[float, float],
    *,
    trans_bandwidth: float,
    edge_trim: float = 0.0,
) -> np.ndarray:
    """Hilbert amplitude envelope of ``data`` in the frequency ``band``, along the last axis.

    The band-pass is a zero-phase FIR filter (firwin design, Hamming window) whose transition
    bands are ``trans_bandwidth`` Hz wide and lie outside ``band``. Its length is
    ``3.3 / trans_bandwidth`` seconds whatever the band, and samples of the envelope further
    apart than that share no filter tap.

    ``edge_trim`` seconds are dropped from each end of the result, where the filter and the
    Hilbert transform have data on one side only. With a 2 Hz transition, the 4-8 Hz envelope
    of a 180 s stretch of 1/f noise differed from the envelope computed with 30 s of context
    on either side by 5 % of its mean (median absolute error) a quarter to half a second from
    the edge, by 1 % at 0.75 to 1 s and by 0.7 % or less beyond; higher bands settle sooner.
    """
    data = np.asarray(data, dtype=float)
    low, high = band
    if not 0 < low < high < sfreq / 2:
        raise ValueError(f"band must satisfy 0 < low < high < {sfreq / 2:g} Hz, got {band}")
    if edge_trim < 0:
        raise ValueError(f"edge_trim must not be negative, got {edge_trim}")
    trim = trim_samples(edge_trim, sfreq)
    if 2 * trim >= data.shape[-1]:
        raise ValueError(
            f"trimming {edge_trim:g} s from each end leaves nothing of {data.shape[-1]} samples"
        )
    filtered = mne.filter.filter_data(
        data,
        sfreq,
        low,
        high,
        l_trans_bandwidth=trans_bandwidth,
        h_trans_bandwidth=trans_bandwidth,
        method="fir",
        fir_design="firwin",
        fir_window="hamming",
        phase="zero",
        verbose="error",
    )
    envelope = np.abs(hilbert(filtered, axis=-1))
    return envelope[..., trim : envelope.shape[-1] - trim]
