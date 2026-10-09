"""Validation of pdeeg.features.mfdfa on signals whose scaling exponents are known.

Where the tolerances come from. The tests on random signals use one fixed realisation of 65536
samples (about one 180 s recording at 512 Hz) with q from -5 to 5 in steps of 0.5, order 1, and
20 scales from 16 samples to a tenth of the length. Over 100 realisations with those settings
(the table in notebooks/01_mfdfa_validation.ipynb) the estimates were, as mean +- standard
deviation:

    signal        h(-5)            h(2)             h(5)             delta_alpha (largest)
    white noise   0.510 +- 0.010   0.502 +- 0.010   0.499 +- 0.012   0.033 +- 0.017 (0.089)
    random walk   1.534 +- 0.027   1.498 +- 0.022   1.489 +- 0.025   0.124 +- 0.040 (0.255)
    fGn, H = 0.3  0.310 +- 0.007   0.300 +- 0.008   0.297 +- 0.010   0.030 +- 0.012 (0.053)
    fGn, H = 0.7  0.708 +- 0.014   0.698 +- 0.014   0.692 +- 0.016   0.046 +- 0.021 (0.106)
    fGn, H = 0.9  0.910 +- 0.018   0.897 +- 0.016   0.890 +- 0.019   0.060 +- 0.027 (0.133)

A tolerance on an exponent is the bias in that table plus four standard deviations, so a test
fails for a wrong estimator and not for an unlucky seed. The estimator is slightly biased
towards a wider spectrum at finite length (h(-5) above and h(5) below the true value), which is
why "narrow" is a bound on delta_alpha rather than a comparison with zero.

The binomial cascade is deterministic, so its tolerances are margins on measured errors, given
with the tests. Comparisons that should hold exactly use tolerances near rounding error.
"""

from pathlib import Path

import numpy as np
import pytest
from MFDFA import MFDFA as reference_mfdfa

from pdeeg.config import load_config
from pdeeg.features.mfdfa import (
    FEATURE_NAMES,
    MfdfaResult,
    make_qs,
    make_scales,
    mfdfa,
    spectrum_features,
)
from pdeeg.features.synthetic import binomial_cascade, binomial_cascade_spectrum, fgn

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# Pinned here rather than read from the configuration, so that tuning the YAML for the EEG
# cannot shift what these tests validate.
QS = make_qs(-5.0, 5.0, 0.5)
N_SAMPLES = 2**16
SCALE_MIN, SCALE_MAX_FRAC, N_SCALES = 16, 0.1, 20
CASCADE_A = 0.75


def default_scales(n_samples: int) -> np.ndarray:
    return make_scales(n_samples, SCALE_MIN, SCALE_MAX_FRAC, N_SCALES)


def analyse(x: np.ndarray, **kwargs) -> tuple[MfdfaResult, dict[str, float]]:
    result = mfdfa(x, default_scales(len(x)), QS, **kwargs)
    return result, spectrum_features(result, QS)


def h_at(result: MfdfaResult, q: float) -> float:
    return float(result.h[np.flatnonzero(q == QS)[0]])


def brute_force_fq(x: np.ndarray, scales, qs, order: int) -> np.ndarray:
    """Kantelhardt's steps 1-4 written out one window at a time, with np.polyfit."""
    profile = np.cumsum(x - np.mean(x))
    n = len(profile)
    fq = np.empty((len(qs), len(scales)))
    for j, s in enumerate(scales):
        starts = [v * s for v in range(n // s)] + [n - (v + 1) * s for v in range(n // s)]
        time = np.arange(s)
        f2 = []
        for start in starts:
            window = profile[start : start + s]
            trend = np.polyval(np.polyfit(time, window, order), time)
            f2.append(np.mean((window - trend) ** 2))
        f2 = np.array(f2)
        for i, q in enumerate(qs):
            if q == 0:
                fq[i, j] = np.exp(0.5 * np.mean(np.log(f2)))
            else:
                fq[i, j] = np.mean(f2 ** (q / 2)) ** (1 / q)
    return fq


# --- signals with known exponents --------------------------------------------------------------


def test_white_noise_is_monofractal_with_h_one_half():
    result, features = analyse(np.random.default_rng(2002).standard_normal(N_SAMPLES))

    # Largest bias 0.010 (at q = -5) plus four times the largest deviation 0.012 (at q = 5).
    assert np.abs(result.h - 0.5).max() < 0.06
    # Mean 0.033 plus five standard deviations of 0.017; the cascade below is thirteen times wider.
    assert features["delta_alpha"] < 0.12
    # The lowest value in 100 realisations was 0.994.
    assert features["min_r2"] > 0.99


def test_cumulative_white_noise_has_h2_three_halves():
    walk = np.cumsum(np.random.default_rng(2002).standard_normal(N_SAMPLES))
    result, _ = analyse(walk)

    # Bias -0.002, standard deviation 0.022: four deviations.
    assert h_at(result, 2.0) == pytest.approx(1.5, abs=0.09)


@pytest.mark.parametrize(
    ("hurst", "tolerance"),
    # Four standard deviations of h(2) (0.008, 0.014, 0.016) plus its bias (0.000 to -0.003).
    [(0.3, 0.035), (0.7, 0.06), (0.9, 0.07)],
)
def test_fractional_gaussian_noise_has_h2_equal_to_hurst(hurst, tolerance):
    result, features = analyse(fgn(N_SAMPLES, hurst, np.random.default_rng(2002)))

    assert h_at(result, 2.0) == pytest.approx(hurst, abs=tolerance)
    # Above the widest of 100 realisations at any of the three exponents (0.133, H = 0.9), and
    # mean plus five standard deviations for that one; an eighth of the cascade's width.
    assert features["delta_alpha"] < 0.2


def test_binomial_cascade_on_its_own_scales_matches_theory():
    """With windows that coincide with the cascade's dyadic boxes the comparison is sharp.

    There ``F²(v, s)`` is exactly the squared mass of box ``v`` times a factor that depends on
    ``s`` alone. The factor approaches a constant as ``s`` grows, so it shifts every ``h(q)``
    by the same amount, which shrinks as the fit moves to larger scales, and it cancels from
    every difference between moment orders: the shape of the spectrum should be exact.
    """
    x = binomial_cascade(16, CASCADE_A)
    scales = 2 ** np.arange(4, 14)
    result = mfdfa(x, scales, QS, fit_range=(256, scales[-1]))
    features = spectrum_features(result, QS)
    h_theory, alpha_theory, _ = binomial_cascade_spectrum(QS, CASCADE_A)
    error = result.h - h_theory

    # The shift is -0.011 when fitting from 256 samples (it is -0.053 when fitting from 16).
    assert np.abs(error).max() < 0.02
    # The same shift at every q, to rounding error (the spread was 8e-15).
    assert np.ptp(error) < 1e-8
    assert features["delta_h"] == pytest.approx(h_theory[0] - h_theory[-1], abs=1e-8)
    # alpha is a finite difference of tau on a grid of 0.5 in q; on the exact tau that alone
    # moves the width from 1.5720 to 1.5739.
    assert features["delta_alpha"] == pytest.approx(np.ptp(alpha_theory), abs=0.005)
    # The peak is at q = 0, where alpha carries the common shift of h.
    assert features["alpha0"] == pytest.approx(alpha_theory[QS == 0][0], abs=0.02)
    # The exact spectrum is symmetric about its peak, and central differences keep it so.
    assert abs(features["asymmetry"]) < 1e-6
    assert features["min_r2"] > 0.999


def test_binomial_cascade_with_default_scales_is_close_to_theory_and_wide():
    """The same series with log-spaced scales from 16 samples, as used for everything else.

    Those windows straddle the dyadic boxes and include the small scales where the cascade has
    not reached its asymptotic scaling, so the agreement is looser: the largest error in h(q)
    is 0.065 (at q = 3) and the width comes out as 1.63 against 1.57.
    """
    result, features = analyse(binomial_cascade(16, CASCADE_A))
    h_theory, alpha_theory, _ = binomial_cascade_spectrum(QS, CASCADE_A)

    assert np.abs(result.h - h_theory).max() < 0.1
    assert features["delta_alpha"] == pytest.approx(np.ptp(alpha_theory), abs=0.1)
    assert features["delta_h"] == pytest.approx(h_theory[0] - h_theory[-1], abs=0.1)
    # Every window of the cascade has structure: none may be mistaken for a flat one.
    np.testing.assert_array_equal(result.n_windows, 2 * (2**16 // default_scales(2**16)))


# --- detrending --------------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3])
def test_order_m_removes_a_trend_of_degree_m_in_the_profile_exactly(order):
    """A trend of degree m - 1 in the series is one of degree m in its profile (step 1).

    For m = 1 that is a constant offset. The trend is 200 standard deviations of the noise, and
    adding it changed no Fq by more than 3e-12 in relative terms. One order lower, the same
    trend multiplies Fq at the largest scale by 6 to 8 (m = 3) or by 160 to 220 (m = 2).
    """
    x = np.random.default_rng(order).standard_normal(8192)
    trend = 200.0 * np.linspace(-1.0, 1.0, x.size) ** (order - 1)
    scales = default_scales(x.size)

    plain = mfdfa(x, scales, QS, order=order)
    trended = mfdfa(x + trend, scales, QS, order=order)

    np.testing.assert_allclose(trended.Fq, plain.Fq, rtol=1e-8)
    if order > 1:
        lower_plain = mfdfa(x, scales, QS, order=order - 1)
        lower_trended = mfdfa(x + trend, scales, QS, order=order - 1)
        assert np.all(lower_trended.Fq[:, -1] > 3.0 * lower_plain.Fq[:, -1])


def test_order_2_removes_a_quadratic_trend_that_order_1_does_not():
    """White noise plus a quadratic trend that reaches ten standard deviations.

    The trend is a cubic in the profile, so order 2 does not remove it exactly; it leaves a
    remainder that is still far below the noise at these scales (h(2) moves by 0.003, against
    +0.34 for order 1). Order 3 removes it exactly, as the test above shows.
    """
    x = np.random.default_rng(2002).standard_normal(N_SAMPLES)
    trended = x + 10.0 * np.linspace(0.0, 1.0, N_SAMPLES) ** 2

    first, _ = analyse(trended, order=1)
    second, _ = analyse(trended, order=2)
    second_plain, _ = analyse(x, order=2)

    assert h_at(first, 2.0) > 0.75
    assert h_at(second, 2.0) == pytest.approx(h_at(second_plain, 2.0), abs=0.01)
    assert h_at(second, 2.0) == pytest.approx(0.5, abs=0.06)


# --- agreement with independent implementations ------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize(
    "make_signal",
    [
        lambda rng: rng.standard_normal(8192),
        lambda rng: np.cumsum(rng.standard_normal(8192)),
        lambda rng: fgn(8192, 0.7, rng),
        lambda rng: rng.uniform(-1.0, 1.0, 5000) ** 3,
    ],
    ids=["white", "random_walk", "fgn", "skewed"],
)
def test_agrees_with_the_mfdfa_package(make_signal, order):
    """Both follow Kantelhardt et al.; they differ only in how the polynomial fit is solved.

    The package fits a Vandermonde matrix on 1..s where this code projects on an orthonormal
    basis, so the two agree to rounding error: at most 1e-12 in Fq and 1e-13 in h(q) when
    measured. 1e-9 keeps three orders of margin and still catches any difference in method.
    The package drops q = 0, which the brute-force test below covers.
    """
    x = make_signal(np.random.default_rng(7))
    scales = default_scales(x.size)
    nonzero = QS != 0

    result = mfdfa(x, scales, QS, order=order)
    lags, fq_reference = reference_mfdfa(x, scales, q=QS[nonzero], order=order)
    h_reference = np.polyfit(np.log(lags), np.log(fq_reference), 1)[0]

    np.testing.assert_array_equal(lags, scales)
    np.testing.assert_allclose(result.Fq[nonzero].T, fq_reference, rtol=1e-9)
    np.testing.assert_allclose(result.h[nonzero], h_reference, rtol=0, atol=1e-9)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_agrees_with_a_window_by_window_implementation(order):
    x = np.random.default_rng(order).standard_normal(1000) ** 3
    scales = np.array([8, 13, 21, 50, 100, 333])
    qs = np.array([-4.0, -1.0, 0.0, 0.5, 2.0, 3.0])

    result = mfdfa(x, scales, qs, order=order)

    np.testing.assert_allclose(result.Fq, brute_force_fq(x, scales, qs, order), rtol=1e-9)
    np.testing.assert_array_equal(result.n_windows, 2 * (x.size // scales))


# --- structure of the result -------------------------------------------------------------------


def test_exponents_and_spectrum_follow_from_the_fluctuation_functions():
    x = np.random.default_rng(1).standard_normal(8192)
    scales = default_scales(x.size)
    result = mfdfa(x, scales, QS, fit_range=(30, 400))
    fitted = (scales >= 30) & (scales <= 400)
    log_s, log_f = np.log(scales[fitted]), np.log(result.Fq[:, fitted])

    assert result.Fq.shape == (QS.size, scales.size)
    for i in range(QS.size):
        slope, intercept = np.polyfit(log_s, log_f[i], 1)
        residual = log_f[i] - (intercept + slope * log_s)
        r2 = 1.0 - residual.var() / log_f[i].var()
        assert result.h[i] == pytest.approx(slope, abs=1e-10)
        assert result.intercept[i] == pytest.approx(intercept, abs=1e-10)
        assert result.h_r2[i] == pytest.approx(r2, abs=1e-10)
    np.testing.assert_allclose(result.tau, QS * result.h - 1.0)
    np.testing.assert_allclose(result.alpha, np.gradient(result.tau, QS, edge_order=2))
    np.testing.assert_allclose(result.f_alpha, QS * result.alpha - result.tau)
    # f(alpha) = -tau(0) = 1 at q = 0, whatever the signal.
    assert result.f_alpha[QS == 0][0] == pytest.approx(1.0)
    # A power mean does not decrease with its order.
    assert np.all(np.diff(result.Fq, axis=0) >= 0)


def test_rescaling_and_shifting_the_series_change_only_the_amplitude():
    x = np.random.default_rng(3).standard_normal(4096)
    scales = default_scales(x.size)

    plain = mfdfa(x, scales, QS)
    moved = mfdfa(3.0e-6 * x + 40.0, scales, QS)

    np.testing.assert_allclose(moved.Fq, 3.0e-6 * plain.Fq, rtol=1e-6)
    np.testing.assert_allclose(moved.h, plain.h, atol=1e-6)


def test_flat_windows_are_discarded():
    """A flat stretch must leave the result as if it had been cut out of the series.

    Every window here is either entirely inside the stretch or entirely outside it. Inside,
    the profile is a straight line, so the residual variance is rounding error (about 1e-30
    of the others), which would otherwise take over every moment with negative q.
    """
    x = np.random.default_rng(4).standard_normal(4096)
    flat = x.copy()
    flat[1024:1536] = 0.3
    cut_out = np.delete(x, np.s_[1024:1536])
    scales = np.array([16, 32, 64, 128])

    result = mfdfa(flat, scales, QS)
    expected = mfdfa(cut_out, scales, QS)

    np.testing.assert_array_equal(result.n_windows, 2 * (cut_out.size // scales))
    np.testing.assert_allclose(result.Fq, expected.Fq, rtol=1e-9)
    np.testing.assert_allclose(result.h, expected.h, atol=1e-9)


def test_constant_series_is_rejected():
    with pytest.raises(ValueError, match="scale"):
        mfdfa(np.full(2048, 3.0), default_scales(2048), QS)


def test_fewer_than_three_moment_orders_give_exponents_without_a_spectrum():
    x = np.random.default_rng(5).standard_normal(4096)
    result = mfdfa(x, default_scales(x.size), [2.0])

    assert result.h.shape == (1,)
    assert np.isnan(result.alpha).all() and np.isnan(result.f_alpha).all()
    with pytest.raises(ValueError, match="three moment orders"):
        spectrum_features(result, [2.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"x": np.zeros((2, 500))}, "one-dimensional"),
        ({"x": np.array([1.0, np.nan] * 250)}, "NaN"),
        ({"scales": [16, 32.5, 64]}, "whole numbers"),
        ({"scales": [32, 16, 64]}, "increasing"),
        ({"scales": [2, 16, 64]}, "at least 3"),
        ({"scales": [16, 64, 501]}, "longer than the series"),
        ({"qs": [2.0, 1.0, 0.0]}, "increasing"),
        ({"order": -1}, "order"),
        ({"fit_range": (16, 40)}, "left to fit"),
        ({"min_variance_ratio": -1.0}, "min_variance_ratio"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs, message):
    arguments = {
        "x": np.random.default_rng(6).standard_normal(500),
        "scales": [16, 32, 64, 128],
        "qs": [-2.0, 0.0, 2.0],
    }
    with pytest.raises(ValueError, match=message):
        mfdfa(**{**arguments, **kwargs})


# --- scales, moment orders and features --------------------------------------------------------


def test_make_scales_are_log_spaced_unique_integers():
    scales = make_scales(N_SAMPLES, SCALE_MIN, SCALE_MAX_FRAC, N_SCALES)

    assert scales.dtype.kind == "i"
    assert len(scales) == N_SCALES
    assert scales[0] == 16 and scales[-1] == 6553
    assert np.all(np.diff(scales) > 0)
    # Evenly spaced in the logarithm, up to the rounding of each end of a step to a whole
    # sample: at most half a sample in 16 plus half a sample in 22, the two smallest scales.
    steps = np.diff(np.log(scales))
    assert np.abs(steps - np.log(6553 / 16) / (N_SCALES - 1)).max() < 0.5 / 16 + 0.5 / 22
    assert np.all(N_SAMPLES // scales >= 10)


def test_make_scales_drops_duplicates_and_rejects_an_empty_range():
    # Twenty log-spaced values between 16 and 25 round to only ten different integers.
    narrow = make_scales(250, 16, 0.1, 20)
    assert np.array_equal(narrow, np.arange(16, 26))

    with pytest.raises(ValueError, match="does not exceed"):
        make_scales(100, 16, 0.1, 20)
    with pytest.raises(ValueError, match="n_scales"):
        make_scales(N_SAMPLES, 16, 0.1, 2)


def test_make_qs_holds_zero_and_two_exactly():
    assert np.array_equal(QS, np.arange(-10, 11) / 2)
    fine = make_qs(-5.0, 5.0, 0.1)
    assert len(fine) == 101
    assert 0.0 in fine and 2.0 in fine

    with pytest.raises(ValueError, match="whole number of steps"):
        make_qs(-5.0, 5.0, 0.3)
    with pytest.raises(ValueError, match="positive"):
        make_qs(-5.0, 5.0, 0.0)


def test_spectrum_features_of_a_known_spectrum():
    qs = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
    nan = np.full(5, np.nan)
    result = MfdfaResult(
        Fq=np.empty((5, 0)),
        h=np.array([1.4, 1.2, 1.0, 0.9, 0.7]),
        h_r2=np.array([0.99, 0.97, 0.999, 0.98, 0.995]),
        tau=nan,
        alpha=np.array([1.5, 1.3, 1.0, 0.8, 0.6]),
        f_alpha=np.array([0.2, 0.7, 1.0, 0.8, 0.1]),
        intercept=nan,
        n_windows=np.empty(0, dtype=int),
    )

    features = spectrum_features(result, qs)

    assert tuple(features) == FEATURE_NAMES
    assert features["h2"] == pytest.approx(0.8)  # halfway between h(1) and h(3)
    assert features["delta_h"] == pytest.approx(0.7)
    assert features["delta_alpha"] == pytest.approx(0.9)
    assert features["alpha0"] == pytest.approx(1.0)
    # Left branch 1.0 - 0.6, right branch 1.5 - 1.0: the right one is longer.
    assert features["asymmetry"] == pytest.approx((0.4 - 0.5) / 0.9)
    assert features["min_r2"] == pytest.approx(0.97)
    assert all(isinstance(value, float) for value in features.values())

    with pytest.raises(ValueError, match="q = 2"):
        spectrum_features(result, qs - 2.0)


def test_configured_parameters_run_end_to_end():
    config = load_config(CONFIG_DIR).mfdfa
    x = np.random.default_rng(8).standard_normal(8192)
    qs = make_qs(config.q_min, config.q_max, config.q_step)
    scales = make_scales(x.size, config.scale_min, config.scale_max_frac, config.n_scales)

    result = mfdfa(
        x,
        scales,
        qs,
        order=config.detrend_order,
        fit_range=config.fit_range,
        min_variance_ratio=config.min_variance_ratio,
    )
    features = spectrum_features(result, qs)

    assert set(config.features) <= set(features)
    assert all(np.isfinite(value) for value in features.values())
