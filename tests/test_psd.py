import dataclasses
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import welch

from pdeeg.config import load_config
from pdeeg.features.extract import CLEAN, FULL
from pdeeg.features.psd import (
    band_powers,
    feature_name,
    recording_band_powers,
    segment_spectra,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SFREQ = 256.0
CONFIG = load_config(CONFIG_DIR).psd
TIMES = np.arange(int(60 * SFREQ)) / SFREQ


def test_mean_of_the_segment_spectra_is_the_welch_spectrum():
    data = np.random.default_rng(0).standard_normal((3, TIMES.size))

    freqs, starts, power = segment_spectra(data, SFREQ, CONFIG)

    n_segment = int(CONFIG.window_sec * SFREQ)
    expected_freqs, expected = welch(data, SFREQ, nperseg=n_segment, noverlap=n_segment // 2)
    np.testing.assert_allclose(freqs, expected_freqs)
    np.testing.assert_allclose(power.mean(axis=2), expected, rtol=1e-12)
    # Segments of 2 s every second: 59 of them in 60 s.
    assert power.shape == (3, n_segment // 2 + 1, 59)
    np.testing.assert_array_equal(starts, np.arange(59) * n_segment // 2)
    with pytest.raises(ValueError, match="longer than the data"):
        segment_spectra(data[:, :100], SFREQ, CONFIG)
    with pytest.raises(ValueError, match="welch"):
        segment_spectra(data, SFREQ, dataclasses.replace(CONFIG, method="multitaper"))


def test_band_powers_put_a_rhythm_in_its_band_and_add_up():
    rng = np.random.default_rng(1)
    noise = 0.05 * rng.standard_normal((2, TIMES.size))
    data = noise + np.array([np.sin(2 * np.pi * 10.0 * TIMES), np.sin(2 * np.pi * 20.0 * TIMES)])
    freqs, _, power = segment_spectra(data, SFREQ, CONFIG)

    values = band_powers(freqs, power.mean(axis=2), CONFIG)

    assert list(values) == ["delta", "theta", "alpha", "beta", "gamma"]
    relative = np.array([10.0 ** values[band] for band in values])
    # The bands are contiguous from fmin to fmax, so their shares add up to one.
    np.testing.assert_allclose(relative.sum(axis=0), 1.0)
    # The white noise holds 0.0025 / 128 per Hz against 0.5 for the sine: 0.9 % of the power
    # from 1 to 48 Hz lies outside the sine's band.
    assert relative[2, 0] > 0.98 and relative[3, 1] > 0.98
    assert values["delta"][0] < -2.5

    absolute = band_powers(
        freqs, power.mean(axis=2), dataclasses.replace(CONFIG, relative=False, log=False)
    )
    # A sine of amplitude 1 has power 0.5.
    assert absolute["alpha"][0] == pytest.approx(0.5, rel=0.02)


def test_band_edges_take_the_lower_frequency_and_leave_the_upper():
    freqs = np.arange(0.0, 60.0, 0.5)
    spectrum = np.zeros((1, freqs.size))
    spectrum[0, freqs == 8.0] = 1.0  # the edge between theta and alpha
    plain = dataclasses.replace(CONFIG, relative=False, log=False)

    values = band_powers(freqs, spectrum, plain)

    assert values["theta"][0] == 0.0
    assert values["alpha"][0] == pytest.approx(0.5)  # density 1 over a bin of 0.5 Hz


def test_feature_name_says_how_the_value_is_scaled():
    assert feature_name(CONFIG) == "log_relative_power"
    assert feature_name(dataclasses.replace(CONFIG, log=False)) == "relative_power"
    assert feature_name(dataclasses.replace(CONFIG, relative=False)) == "log_power"


def test_clean_band_powers_drop_the_segments_on_bad_stretches():
    rng = np.random.default_rng(2)
    data = rng.standard_normal((2, TIMES.size))
    bad = np.zeros(TIMES.size, dtype=bool)
    bad[int(20 * SFREQ) : int(22 * SFREQ)] = True
    spoiled = data.copy()
    # A slow artefact a hundred times the signal, inside the bad stretch only.
    spoiled[:, bad] += 100.0 * np.sin(2 * np.pi * 2.0 * TIMES[bad])

    table = recording_band_powers(spoiled, bad, SFREQ, ["a", "b"], CONFIG)
    reference = recording_band_powers(data, bad, SFREQ, ["a", "b"], CONFIG)

    assert list(table.columns) == ["channel", "band", "segments", "feature", "value"]
    assert len(table) == 2 * 5 * 2
    assert set(table["feature"]) == {"log_relative_power"}
    clean = table[table["segments"] == CLEAN].reset_index(drop=True)
    expected = reference[reference["segments"] == CLEAN].reset_index(drop=True)
    # What lies in the bad stretch does not reach the clean values at all.
    np.testing.assert_allclose(clean["value"], expected["value"], rtol=1e-12)
    full = table[table["segments"] == FULL].set_index(["channel", "band"])["value"]
    assert full["a", "delta"] > -0.05  # the artefact takes nearly all the power
    assert clean.set_index(["channel", "band"])["value"]["a", "delta"] < -1.0

    # Segments of 2 s every second: those starting at 19, 20 and 21 s touch 20 to 22 s.
    freqs, starts, power = segment_spectra(data, SFREQ, CONFIG)
    keep = (starts + int(2 * SFREQ) <= int(20 * SFREQ)) | (starts >= int(22 * SFREQ))
    assert keep.sum() == starts.size - 3
    by_hand = band_powers(freqs, power[:, :, keep].mean(axis=2), CONFIG)
    np.testing.assert_allclose(
        expected[expected["channel"] == "a"]["value"], [by_hand[band][0] for band in by_hand]
    )


def test_clean_band_powers_are_missing_when_nothing_is_clean_and_equal_when_all_is():
    data = np.random.default_rng(3).standard_normal((1, TIMES.size))

    nothing = recording_band_powers(data, np.ones(TIMES.size, dtype=bool), SFREQ, ["a"], CONFIG)
    everything = recording_band_powers(data, np.zeros(TIMES.size, dtype=bool), SFREQ, ["a"], CONFIG)

    assert nothing[nothing["segments"] == CLEAN]["value"].isna().all()
    assert nothing[nothing["segments"] == FULL]["value"].notna().all()
    np.testing.assert_array_equal(
        everything[everything["segments"] == CLEAN]["value"].to_numpy(),
        everything[everything["segments"] == FULL]["value"].to_numpy(),
    )
