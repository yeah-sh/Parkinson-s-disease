"""Band-power features from the power spectrum: the baseline the MFDFA features are held against.

The spectrum is Welch's: the recording is cut into overlapping segments, each is tapered with a
Hann window, and the periodograms are averaged. As with MFDFA, a recording is analysed twice,
with every segment (``full``) and without the segments that touch a bad stretch (``clean``);
segments are dropped where they stand and nothing is joined.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import spectrogram

from pdeeg.config import Config, PsdConfig
from pdeeg.features.extract import CLEAN, FULL, bad_sample_mask
from pdeeg.features.tables import RECORDING_KEYS, add_regions
from pdeeg.preprocessing.pipeline import output_paths, read_clean

# Columns of the band-power long table.
PSD_COLUMNS = (*RECORDING_KEYS, "level", "name", "band", "segments", "feature", "value")

_VOLTS_TO_UV = 1e6


def feature_name(config: PsdConfig) -> str:
    """What a band-power value is called, given how the configuration scales it."""
    name = "relative_power" if config.relative else "power"
    return f"log_{name}" if config.log else name


def segment_spectra(
    data: np.ndarray, sfreq: float, config: PsdConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periodogram of every Welch segment of ``data`` (channels x samples).

    Returns the frequencies, the first sample of each segment, and the power density with
    shape (channels, frequencies, segments). The mean over segments is the Welch spectrum.
    """
    if config.method != "welch":
        raise ValueError(f"only the welch method is implemented, got {config.method!r}")
    n_segment = int(round(config.window_sec * sfreq))
    n_overlap = int(round(config.overlap * n_segment))
    if n_segment > data.shape[-1]:
        raise ValueError(
            f"a segment of {config.window_sec:g} s is longer than the data "
            f"({data.shape[-1] / sfreq:g} s)"
        )
    freqs, _, power = spectrogram(
        data,
        fs=sfreq,
        window="hann",
        nperseg=n_segment,
        noverlap=n_overlap,
        detrend="constant",
        scaling="density",
        mode="psd",
    )
    starts = np.arange(power.shape[-1]) * (n_segment - n_overlap)
    return freqs, starts, power


def band_powers(
    freqs: np.ndarray, spectrum: np.ndarray, config: PsdConfig
) -> dict[str, np.ndarray]:
    """Power of ``spectrum`` (channels x frequencies) in each configured band, per channel.

    A band takes the frequencies from its lower edge up to, not including, its upper edge.
    With ``relative`` the power is divided by the power between ``fmin`` and ``fmax``, counted
    the same way, and with ``log`` the base-10 logarithm is taken.
    """
    step = freqs[1] - freqs[0]

    def power(low: float, high: float) -> np.ndarray:
        inside = (freqs >= low) & (freqs < high)
        return spectrum[:, inside].sum(axis=1) * step

    total = power(config.fmin, config.fmax)
    values = {}
    for band, (low, high) in config.bands.items():
        value = power(low, high)
        if config.relative:
            value = value / total
        values[band] = np.log10(value) if config.log else value
    return values


def recording_band_powers(
    data: np.ndarray, bad: np.ndarray, sfreq: float, channels: list[str], config: PsdConfig
) -> pd.DataFrame:
    """Band powers of every channel of one recording, with and without the bad stretches.

    ``data`` is channels x samples and ``bad`` one boolean per sample. Returns the columns
    ``channel``, ``band``, ``segments``, ``feature`` and ``value``. The ``clean`` values are
    NaN when every segment touches a bad sample.
    """
    freqs, starts, power = segment_spectra(data, sfreq, config)
    n_segment = int(round(config.window_sec * sfreq))
    bad_before = np.concatenate([[0], np.cumsum(bad)])
    untouched = bad_before[starts + n_segment] == bad_before[starts]
    name = feature_name(config)
    rows = []
    for segments, keep in ((FULL, np.ones(starts.size, dtype=bool)), (CLEAN, untouched)):
        if keep.any():
            values = band_powers(freqs, power[:, :, keep].mean(axis=2), config)
        else:
            values = {band: np.full(len(channels), np.nan) for band in config.bands}
        for band, per_channel in values.items():
            for channel, value in zip(channels, per_channel, strict=True):
                rows.append(
                    {
                        "channel": channel,
                        "band": band,
                        "segments": segments,
                        "feature": name,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _recording_table(recording: Mapping[str, Any], config: Config) -> pd.DataFrame:
    raw = read_clean(
        output_paths(config, recording["subject"], recording["session"])["raw"], preload=True
    )
    channels = recording_band_powers(
        raw.get_data() * _VOLTS_TO_UV,
        bad_sample_mask(raw),
        float(raw.info["sfreq"]),
        raw.ch_names,
        config.psd,
    )
    table = add_regions(channels, config.data.regions, ["band", "segments", "feature"])
    for position, key in enumerate(RECORDING_KEYS):
        table.insert(position, key, recording[key])
    return table


def build_psd_table(config: Config, recordings: pd.DataFrame, *, n_jobs: int = 1) -> pd.DataFrame:
    """The band-power long table of every recording, channels and region averages."""
    tables = Parallel(n_jobs=n_jobs)(
        delayed(_recording_table)(row, config) for row in recordings.to_dict("records")
    )
    return pd.concat(tables, ignore_index=True)[list(PSD_COLUMNS)]
