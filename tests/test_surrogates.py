import numpy as np
import pytest

from pdeeg.features.mfdfa import make_scales, mfdfa
from pdeeg.features.surrogates import iaaft_surrogate, shuffle_surrogate
from pdeeg.features.synthetic import fgn

N_SAMPLES = 2**13
SIGNALS = {
    # Correlated in both cases; the second also has a skewed, long-tailed distribution.
    "gaussian": lambda rng: fgn(N_SAMPLES, 0.8, rng),
    "lognormal": lambda rng: np.exp(fgn(N_SAMPLES, 0.8, rng)),
}


def spectrum_mismatch(surrogate: np.ndarray, x: np.ndarray) -> float:
    """Summed absolute difference between two periodograms, as a fraction of the power of ``x``.

    The zero-frequency bin is left out: it holds the mean, which any permutation preserves.
    """
    power = np.abs(np.fft.rfft(x)[1:]) ** 2
    power_surrogate = np.abs(np.fft.rfft(surrogate)[1:]) ** 2
    return float(np.abs(power_surrogate - power).sum() / power.sum())


def lag_one_correlation(x: np.ndarray) -> float:
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def h2(x: np.ndarray) -> float:
    return float(mfdfa(x, make_scales(x.size, 16, 0.1, 20), [2.0]).h[0])


@pytest.mark.parametrize("name", SIGNALS)
def test_shuffle_keeps_the_amplitude_distribution_and_nothing_else(name):
    x = SIGNALS[name](np.random.default_rng(0))
    surrogate = shuffle_surrogate(x, np.random.default_rng(1))

    assert np.array_equal(np.sort(surrogate), np.sort(x))
    assert not np.array_equal(surrogate, x)
    # The lag-one correlation of x is 0.53 (Gaussian) or 0.35 (lognormal); that of an
    # uncorrelated series of this length has a standard deviation of 1 / sqrt(8192) = 0.011,
    # and the bound is four times that.
    assert lag_one_correlation(x) > 0.3
    assert abs(lag_one_correlation(surrogate)) < 0.045
    # The spectrum goes from red to flat: the mismatch was 1.07 to 1.20 in 30 realisations.
    assert spectrum_mismatch(surrogate, x) > 0.5


@pytest.mark.parametrize(
    ("name", "tolerance"),
    # Measured over 30 realisations with the default 100 rounds. The Gaussian series converges
    # and its mismatch was 0.0006 every time. The lognormal one is still improving slowly at
    # that point: 0.044 +- 0.014, at most 0.078, and the bound is the mean plus five standard
    # deviations. Shuffling gives 1.07 or more.
    [("gaussian", 0.005), ("lognormal", 0.12)],
)
def test_iaaft_keeps_the_amplitude_distribution_and_the_power_spectrum(name, tolerance):
    x = SIGNALS[name](np.random.default_rng(0))
    surrogate = iaaft_surrogate(x, np.random.default_rng(1))

    assert np.array_equal(np.sort(surrogate), np.sort(x))
    assert not np.array_equal(surrogate, x)
    assert spectrum_mismatch(surrogate, x) < tolerance
    # The lag-one correlation is fixed by the spectrum; it moved by at most 0.011.
    assert lag_one_correlation(surrogate) == pytest.approx(lag_one_correlation(x), abs=0.03)


def test_iaaft_keeps_the_hurst_exponent_and_shuffling_does_not():
    x = fgn(2**15, 0.8, np.random.default_rng(0))

    # h(2) is set by the power spectrum, which IAAFT preserves: over ten surrogates it differed
    # from the original by at most 0.01. A shuffle is white noise, whose h(2) at this length
    # has a standard deviation of 0.014, so the bound there is four of those.
    assert h2(iaaft_surrogate(x, np.random.default_rng(1))) == pytest.approx(h2(x), abs=0.03)
    assert h2(shuffle_surrogate(x, np.random.default_rng(1))) == pytest.approx(0.5, abs=0.06)


def test_iaaft_improves_with_more_rounds_and_handles_odd_lengths():
    x = np.exp(fgn(4097, 0.8, np.random.default_rng(0)))

    rough = iaaft_surrogate(x, np.random.default_rng(1), max_iter=1)
    better = iaaft_surrogate(x, np.random.default_rng(1), max_iter=50)

    assert better.shape == x.shape
    assert np.array_equal(np.sort(better), np.sort(x))
    assert spectrum_mismatch(better, x) < 0.5 * spectrum_mismatch(rough, x)


@pytest.mark.parametrize("make", [shuffle_surrogate, iaaft_surrogate])
def test_surrogates_are_reproducible_and_leave_the_input_alone(make):
    x = fgn(1024, 0.7, np.random.default_rng(0))
    original = x.copy()

    first = make(x, np.random.default_rng(5))
    again = make(x, np.random.default_rng(5))
    other = make(x, np.random.default_rng(6))

    assert np.array_equal(x, original)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    with pytest.raises(ValueError, match="one-dimensional"):
        make(np.zeros((4, 4)), np.random.default_rng(5))
