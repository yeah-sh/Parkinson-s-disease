import mne
import numpy as np
import pytest
from joblib import Parallel, delayed
from scipy import stats as scipy_stats

from pdeeg.stats.cluster import (
    benjamini_hochberg,
    channel_adjacency,
    hedges_g,
    paired_hedges_g,
    paired_test,
    residualise,
    t_threshold,
    two_sample_test,
)
from pdeeg.stats.descriptive import bootstrap_mean_ci, detectable_effect

CHANNELS = mne.channels.make_standard_montage("biosemi32").ch_names
ADJACENCY = channel_adjacency("biosemi32", CHANNELS)
# Neighbours of one another in the biosemi32 template, at the front of the head.
PLANTED = ["Fz", "FC1", "FC2", "Cz"]
FAR = ["O1", "Oz", "O2", "PO3", "PO4"]


def index(names):
    return [CHANNELS.index(name) for name in names]


def smooth_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    """Noise that, like EEG features, is correlated between neighbouring channels."""
    z = rng.standard_normal((n, len(CHANNELS)))
    neighbours = ADJACENCY.toarray().astype(float)
    np.fill_diagonal(neighbours, 0.0)
    return z + 0.5 * z @ (neighbours / neighbours.sum(axis=0))


def channels_of(mask) -> set[str]:
    return {CHANNELS[i] for i in np.flatnonzero(mask)}


# --- building blocks -----------------------------------------------------------------------------


def test_adjacency_follows_the_order_it_is_asked_for():
    order = ["Oz", "Fz", "Cz", "O1"]
    adjacency = channel_adjacency("biosemi32", order).toarray()

    assert adjacency.shape == (4, 4)
    assert (adjacency == adjacency.T).all()
    assert adjacency[order.index("Oz"), order.index("O1")]
    assert adjacency[order.index("Fz"), order.index("Cz")]
    assert not adjacency[order.index("Fz"), order.index("Oz")]
    with pytest.raises(ValueError, match="no channel"):
        channel_adjacency("biosemi32", ["Fz", "E99"])


def test_threshold_is_the_two_sided_t_quantile():
    assert t_threshold(0.05, 29) == pytest.approx(2.0452, abs=1e-4)
    assert scipy_stats.t.sf(t_threshold(0.01, 12), 12) == pytest.approx(0.005)


def test_hedges_g_is_cohens_d_with_the_small_sample_correction():
    a = np.array([[2.0], [4.0], [6.0], [8.0]])
    b = np.array([[1.0], [2.0], [3.0]])
    pooled = np.sqrt((3 * np.var(a, ddof=1) + 2 * np.var(b, ddof=1)) / 5)
    d = (5.0 - 2.0) / pooled

    assert hedges_g(a, b)[0] == pytest.approx(d * (1 - 3 / (4 * 7 - 9)))
    assert hedges_g(b, a)[0] == pytest.approx(-hedges_g(a, b)[0])


def test_paired_g_is_the_standardised_mean_difference():
    differences = np.array([[1.0, -2.0], [2.0, -1.0], [3.0, -3.0], [2.0, -2.0]])
    correction = 1 - 3 / (4 * 3 - 1)
    expected = differences.mean(axis=0) / differences.std(axis=0, ddof=1) * correction

    np.testing.assert_allclose(paired_hedges_g(differences), expected)


def test_residuals_carry_nothing_the_covariates_explain():
    rng = np.random.default_rng(0)
    age = rng.uniform(50, 80, 40)
    sex = rng.integers(0, 2, 40).astype(float)
    values = 0.05 * age[:, None] + 0.4 * sex[:, None] + rng.standard_normal((40, 3))

    residuals = residualise(values, np.column_stack([age, sex]))

    np.testing.assert_allclose(residuals.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(residuals.T @ age, 0.0, atol=1e-8)
    np.testing.assert_allclose(residuals.T @ sex, 0.0, atol=1e-8)


def test_benjamini_hochberg_by_hand():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    # Sorted 0.01, 0.03, 0.04, 0.20: p * m / rank = 0.04, 0.06, 0.0533, 0.20, then the running
    # minimum from the top.
    np.testing.assert_allclose(benjamini_hochberg(p), [0.04, 0.16 / 3, 0.16 / 3, 0.20])
    assert benjamini_hochberg(np.array([])).size == 0


def test_bootstrap_interval_contains_the_mean_and_narrows_with_more_people():
    rng = np.random.default_rng(1)
    few, many = rng.normal(1.0, 1.0, (10, 2)), rng.normal(1.0, 1.0, (200, 2))

    low, high = bootstrap_mean_ci(few, 2000, 0.95, np.random.default_rng(2))
    wide = high - low
    low_many, high_many = bootstrap_mean_ci(many, 2000, 0.95, np.random.default_rng(2))

    assert ((low < few.mean(axis=0)) & (few.mean(axis=0) < high)).all()
    assert ((high_many - low_many) < wide).all()


def test_detectable_effect_matches_the_textbook_values():
    # 64 per group for d = 0.5 at 80 % power, and about 34 pairs for d_z = 0.5.
    assert detectable_effect(64, 64) == pytest.approx(0.5, abs=0.01)
    assert detectable_effect(34) == pytest.approx(0.5, abs=0.02)


# --- the cluster tests on synthetic data ---------------------------------------------------------


def test_a_planted_group_difference_in_neighbouring_channels_is_found():
    rng = np.random.default_rng(10)
    patients, controls = smooth_noise(rng, 15), smooth_noise(rng, 16)
    patients[:, index(PLANTED)] += 1.5

    result = two_sample_test(
        patients,
        controls,
        ADJACENCY,
        n_permutations=2000,
        forming_p=0.05,
        rng=np.random.default_rng(0),
    )

    best = int(np.argmin(result.p))
    assert result.p[best] < 0.01
    assert set(PLANTED) <= channels_of(result.clusters[best])
    assert not channels_of(result.clusters[best]) & set(FAR)
    assert result.mass[best] > 0  # patients higher
    assert result.df == 29 and result.n_permutations == 2000 and not result.exact
    # The planted channels carry the effect; elsewhere g only scatters about 0 (by about 0.36
    # with groups this size, so single channels can still reach 0.8).
    g = hedges_g(patients, controls)
    assert g[index(PLANTED)].mean() > 1.0
    assert abs(g[index(FAR)].mean()) < 0.5


def test_a_planted_paired_difference_is_found_with_an_exact_test():
    rng = np.random.default_rng(11)
    differences = smooth_noise(rng, 13)
    differences[:, index(PLANTED)] -= 1.3

    result = paired_test(
        differences, ADJACENCY, n_permutations=10000, forming_p=0.05, rng=np.random.default_rng(0)
    )

    best = int(np.argmin(result.p))
    assert result.p[best] < 0.01
    assert set(PLANTED) <= channels_of(result.clusters[best])
    assert result.mass[best] < 0
    # 13 pairs have 2 ** 12 sign flips up to the symmetry of a two-sided test.
    assert result.exact and result.n_permutations == 2**12


def test_the_same_seed_gives_the_same_result():
    rng = np.random.default_rng(12)
    a, b = smooth_noise(rng, 10), smooth_noise(rng, 10)
    a[:, index(PLANTED)] += 1.0

    def run():
        return two_sample_test(
            a, b, ADJACENCY, n_permutations=300, forming_p=0.05, rng=np.random.default_rng(5)
        )

    first, second = run(), run()

    np.testing.assert_array_equal(first.p, second.p)
    assert first.p.size


def _null_two_sample(seed: int, n_permutations: int) -> bool:
    rng = np.random.default_rng(seed)
    result = two_sample_test(
        smooth_noise(rng, 15),
        smooth_noise(rng, 16),
        ADJACENCY,
        n_permutations=n_permutations,
        forming_p=0.05,
        rng=rng,
    )
    return bool(result.p.size and result.p.min() < 0.05)


def _null_paired(seed: int, n_permutations: int) -> bool:
    rng = np.random.default_rng(seed)
    result = paired_test(
        smooth_noise(rng, 13), ADJACENCY, n_permutations=n_permutations, forming_p=0.05, rng=rng
    )
    return bool(result.p.size and result.p.min() < 0.05)


@pytest.mark.parametrize("simulate", [_null_two_sample, _null_paired])
def test_without_an_effect_a_cluster_is_found_about_as_often_as_alpha(simulate):
    """With no difference anywhere, some cluster has p < 0.05 in about 5 % of datasets.

    The cluster test controls the chance of any false cluster across all channels, so this is
    the family-wise rate. The bounds are the central 99 % of what a test at exactly 5 % would
    give over this many datasets.
    """
    n_datasets = 300
    found = Parallel(n_jobs=4)(delayed(simulate)(seed, 200) for seed in range(n_datasets))
    low, high = scipy_stats.binom.interval(0.99, n_datasets, 0.05)

    assert low <= sum(found) <= high, f"{sum(found)} of {n_datasets} datasets"
