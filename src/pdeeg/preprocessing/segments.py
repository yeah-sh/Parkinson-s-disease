"""Mark bad stretches of a continuous recording without removing them."""

from __future__ import annotations

from typing import Any

import mne
import numpy as np

from pdeeg.config import BadSegmentsConfig

PEAK = "BAD_peak"
FLAT = "BAD_flat"


def window_peak_to_peak(data: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Peak-to-peak amplitude of ``data`` (channels x samples) in consecutive windows.

    Returns the amplitudes, shape (channels, windows), and the window edges in samples, shape
    (windows + 1,). Samples left over after the last full window are added to that window, so
    every sample belongs to exactly one window.
    """
    n_windows = max(data.shape[1] // window, 1)
    edges = np.arange(n_windows + 1) * window
    edges[-1] = data.shape[1]
    ptp = np.stack(
        [
            np.ptp(data[:, start:stop], axis=1)
            for start, stop in zip(edges[:-1], edges[1:], strict=True)
        ],
        axis=1,
    )
    return ptp, edges


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start and stop (exclusive) of each run of True in a 1-D boolean array."""
    padded = np.concatenate([[False], mask, [False]])
    change = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(change[::2].tolist(), change[1::2].tolist(), strict=True))


def annotate_bad_segments(
    raw: mne.io.BaseRaw, config: BadSegmentsConfig
) -> tuple[mne.Annotations, dict[str, Any]]:
    """Find windows whose peak-to-peak amplitude is too large or too small in any channel.

    Returns ``BAD_peak`` / ``BAD_flat`` annotations, timed from the first sample of ``raw``
    and with neighbouring bad windows merged, and a summary for the log. ``raw`` itself is
    not changed; pass the annotations to ``raw.set_annotations`` to attach them.
    """
    sfreq = raw.info["sfreq"]
    ptp, edges = window_peak_to_peak(raw.get_data() * 1e6, int(round(config.window * sfreq)))
    too_large = ptp > config.peak_to_peak_uv
    too_small = ptp < config.flat_uv

    onsets, durations, descriptions = [], [], []
    for description, mask in ((PEAK, too_large.any(axis=0)), (FLAT, too_small.any(axis=0))):
        for start, stop in _runs(mask):
            onsets.append(edges[start] / sfreq)
            durations.append((edges[stop] - edges[start]) / sfreq)
            descriptions.append(description)
    order = np.argsort(onsets, kind="stable")
    annotations = mne.Annotations(
        onset=np.array(onsets)[order],
        duration=np.array(durations)[order],
        description=np.array(descriptions, dtype=str)[order],
    )

    lengths = np.diff(edges)

    def percent(mask: np.ndarray) -> float:
        return float(100.0 * lengths[mask].sum() / lengths.sum())

    summary = {
        "window_s": config.window,
        "percent": percent(too_large.any(axis=0) | too_small.any(axis=0)),
        "peak_percent": percent(too_large.any(axis=0)),
        "flat_percent": percent(too_small.any(axis=0)),
        "segments": [
            {"onset_s": float(onset), "duration_s": float(duration), "description": str(desc)}
            for onset, duration, desc in zip(
                annotations.onset, annotations.duration, annotations.description, strict=True
            )
        ],
        "channel_peak_percent": [percent(row) for row in too_large],
        "channel_flat_percent": [percent(row) for row in too_small],
    }
    return annotations, summary
