"""The synthetic signals are the ground truth of tests/test_mfdfa.py, so they are checked too."""

import numpy as np
import pytest

from pdeeg.features.mfdfa import make_scales, mfdfa
from pdeeg.features.synthetic import (
    binomial_cascade,
    binomial_cascade_spectrum,
    fgn,
    power_law_noise,
)


def fgn_autocovariance(lag: int, hurst: float) -> float:
    return 0.5 * ((lag + 1) ** (2 * hurst) - 2 * lag ** (2 * hurst) + (lag - 1) ** (2 * hurst))


@pytest.mark.parametrize("hurst", [0.3, 0.5, 0.7])
def test_fgn_has_the_autocovariance_of_fractional_gaussian_noise(hurst):
    x = fgn(2**16, hurst, np.random.default_rng(11))

    # Over 20 realisations of this length the sample values scatter around the exact ones with
    # a standard deviation of at most 0.008; the tolerance is four times that. H = 0.9 is left
    # out because its sample autocovariance scatters twenty times more.
    assert np.mean(x * x) == pytest.approx(1.0, abs=0.032)
    for lag in (1, 2, 8):
        expected = fgn_autocovariance(lag, hurst)
        assert np.mean(x[:-lag] * x[lag:]) == pytest.approx(expected, abs=0.032)


def test_fgn_is_reproducible_and_validates_its_arguments():
    first = fgn(1000, 0.7, np.random.default_rng(1))
    again = fgn(1000, 0.7, np.random.default_rng(1))
    other = fgn(1000, 0.7, np.random.default_rng(2))

    assert first.shape == (1000,)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    for hurst in (0.0, 1.0, 1.2):
        with pytest.raises(ValueError, match="hurst"):
            fgn(1000, hurst, np.random.default_rng(1))


@pytest.mark.parametrize(
    ("beta", "tolerance"),
    # h(2) over 100 realisations of 65536 samples, 20 scales from 16 samples to a tenth of the
    # length: 0.502 +- 0.010, 1.000 +- 0.015 and 1.495 +- 0.023. Four standard deviations, plus
    # the bias of 0.005 for the last.
    [(0.0, 0.04), (1.0, 0.06), (2.0, 0.1)],
)
def test_power_law_noise_has_the_exponent_of_its_spectrum(beta, tolerance):
    x = power_law_noise(2**16, beta, np.random.default_rng(11))

    h2 = mfdfa(x, make_scales(x.size, 16, 0.1, 20), [2.0]).h[0]

    assert h2 == pytest.approx((beta + 1.0) / 2.0, abs=tolerance)
    assert x.std() == pytest.approx(1.0)
    assert x.mean() == pytest.approx(0.0, abs=1e-12)


def test_power_law_noise_is_reproducible_and_validates_its_arguments():
    first = power_law_noise(1001, 1.0, np.random.default_rng(1))

    assert first.shape == (1001,)
    assert np.array_equal(first, power_law_noise(1001, 1.0, np.random.default_rng(1)))
    assert not np.array_equal(first, power_law_noise(1001, 1.0, np.random.default_rng(2)))
    with pytest.raises(ValueError, match="n_samples"):
        power_law_noise(1, 1.0, np.random.default_rng(1))


def test_binomial_cascade_is_built_by_repeated_splitting():
    a = 0.75
    expected = np.ones(1)
    for _ in range(10):
        # Each sample splits in two: the left half keeps (1 - a) of its mass, the right gets a.
        expected = np.column_stack([expected * (1 - a), expected * a]).ravel()

    cascade = binomial_cascade(10, a)

    np.testing.assert_allclose(cascade, expected, rtol=1e-12)
    assert cascade.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="between 0.5 and 1"):
        binomial_cascade(10, 0.5)


def test_binomial_cascade_spectrum_is_consistent():
    a = 0.75
    qs = np.linspace(-5.0, 5.0, 2001)
    h, alpha, f_alpha = binomial_cascade_spectrum(qs, a)
    tau = qs * h - 1.0

    # alpha is the derivative of tau: compare with a fine finite difference.
    np.testing.assert_allclose(alpha, np.gradient(tau, qs, edge_order=2), atol=1e-4)
    # A measure that sums to 1 has tau(1) = 0, and its support has dimension f(alpha(0)) = 1.
    assert tau[qs == 1.0][0] == pytest.approx(0.0, abs=1e-12)
    assert f_alpha.max() == pytest.approx(1.0)
    assert qs[np.argmax(f_alpha)] == pytest.approx(0.0)
    # h is continuous through q = 0, where the general formula is 0 / 0.
    middle = np.flatnonzero(qs == 0.0)[0]
    assert h[middle] == pytest.approx(0.5 * (h[middle - 1] + h[middle + 1]), abs=1e-5)
    # The singularity strengths run from -log2(a) to -log2(1 - a) as q goes to +-infinity.
    limits = binomial_cascade_spectrum(np.array([-200.0, 200.0]), a)[1]
    np.testing.assert_allclose(limits, [-np.log2(1 - a), -np.log2(a)], rtol=1e-9)
