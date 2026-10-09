import numpy as np
import pytest

from pdeeg.features.envelope import band_envelope, trim_samples
from pdeeg.features.mfdfa import mfdfa
from pdeeg.features.scaling import scales_in_range

SFREQ = 256.0
ALPHA = (8.0, 13.0)
TIMES = np.arange(int(60 * SFREQ)) / SFREQ


def test_envelope_recovers_a_slow_amplitude_modulation():
    modulation = 1.0 + 0.5 * np.sin(2 * np.pi * 0.3 * TIMES)
    x = modulation * np.sin(2 * np.pi * 10.0 * TIMES)

    envelope = band_envelope(x, SFREQ, ALPHA, trans_bandwidth=2.0, edge_trim=1.0)
    untrimmed = band_envelope(x, SFREQ, ALPHA, trans_bandwidth=2.0)

    trim = trim_samples(1.0, SFREQ)
    assert trim == 256
    assert envelope.shape == (x.size - 2 * trim,)
    assert np.array_equal(envelope, untrimmed[trim:-trim])
    # The error was 0.0023 at most; the modulation itself swings by 0.5.
    assert np.abs(envelope - modulation[trim:-trim]).max() < 0.01
    # The ends are what the trim is for: there the error reached 0.21 in the first quarter second.
    assert np.abs(untrimmed - modulation)[:64].max() > 0.1


@pytest.mark.parametrize(
    ("frequency", "low", "high"),
    # 15 Hz is where the stop band starts for a 13 Hz edge with a 2 Hz transition: 0.005 came
    # through. 14 Hz is the middle of the transition, where a firwin filter passes half: 0.501.
    [(20.0, 0.0, 0.01), (15.0, 0.0, 0.02), (14.0, 0.45, 0.55), (10.0, 0.99, 1.01)],
)
def test_envelope_keeps_the_band_and_rejects_what_lies_beyond_the_transition(frequency, low, high):
    x = np.sin(2 * np.pi * frequency * TIMES)

    envelope = band_envelope(x, SFREQ, ALPHA, trans_bandwidth=2.0, edge_trim=1.0)

    assert low <= envelope.max() <= high


def test_envelope_treats_every_row_separately():
    data = np.random.default_rng(0).standard_normal((3, 4096))

    together = band_envelope(data, SFREQ, ALPHA, trans_bandwidth=2.0, edge_trim=0.5)

    assert together.shape == (3, 4096 - 2 * 128)
    for row, expected in zip(data, together, strict=True):
        alone = band_envelope(row, SFREQ, ALPHA, trans_bandwidth=2.0, edge_trim=0.5)
        np.testing.assert_allclose(alone, expected, rtol=1e-10)
    assert np.all(together >= 0)


@pytest.mark.parametrize(
    ("band", "short", "main"),
    # h(2) of the envelope of white noise at 512 Hz, mean of 40 realisations, fitted from 0.25
    # to 1 s and from 2 to 17.8 s: theta 0.92 and 0.54, alpha 0.86 and 0.53, beta 0.64 and 0.52.
    # The standard deviation of one realisation was 0.02 and 0.05, so that of the mean of ten
    # is 0.007 and 0.015; the bounds leave four of those.
    [((4.0, 8.0), 0.85, 0.60), ((8.0, 13.0), 0.80, 0.59), ((13.0, 30.0), 0.60, 0.58)],
)
def test_filter_memory_of_the_envelope_is_gone_above_two_seconds(band, short, main):
    """Why the envelope is fitted from 2 s: below, the band-pass alone looks like memory."""
    sfreq, n_samples = 512.0, 92160
    below, above = [], []
    for seed in range(10):
        noise = np.random.default_rng(seed).standard_normal(n_samples)
        envelope = band_envelope(noise, sfreq, band, trans_bandwidth=2.0, edge_trim=1.0)
        below.append(mfdfa(envelope, scales_in_range((0.25, 1.0), sfreq, 12), [2.0]).h[0])
        above.append(mfdfa(envelope, scales_in_range((2.0, 17.8), sfreq, 12), [2.0]).h[0])

    assert np.mean(below) > short
    assert 0.46 < np.mean(above) < main


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"band": (13.0, 8.0)}, "band"),
        ({"band": (0.0, 8.0)}, "band"),
        ({"band": (100.0, 200.0)}, "band"),
        ({"edge_trim": -1.0}, "edge_trim"),
        ({"edge_trim": 40.0}, "leaves nothing"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs, message):
    arguments = {"band": ALPHA, "trans_bandwidth": 2.0, "edge_trim": 1.0}
    with pytest.raises(ValueError, match=message):
        band_envelope(np.zeros(TIMES.size), SFREQ, **{**arguments, **kwargs})
