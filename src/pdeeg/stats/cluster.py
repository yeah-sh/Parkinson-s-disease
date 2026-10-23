"""Cluster-based permutation tests across channels, effect sizes, and the false discovery rate.

The tests are MNE's (Maris and Oostenveld, J Neurosci Methods 164 (2007) 177-190). A t-value is
computed at every channel; channels whose |t| exceeds a threshold and that are neighbours form a
cluster, whose mass is the sum of its t-values. The same is done after every relabelling of the
data (a permutation of the group labels, or a flip of the signs of some differences), and a
cluster's p-value is the share of relabellings whose largest |mass| reaches its own, the
observed labelling counted in. That controls the chance of any false cluster across channels.

Nothing here knows about the configuration: every setting is an argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
from scipy import sparse
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class ClusterTest:
    """What :func:`two_sample_test` and :func:`paired_test` return.

    ``t`` has one value per channel. ``clusters`` holds one boolean mask over the channels per
    cluster, ``p`` and ``mass`` (the sum of its t-values) one value per cluster.
    ``n_permutations`` counts the relabellings the p-values rest on, the observed one included,
    and ``exact`` says whether that was every one there is.
    """

    t: np.ndarray
    clusters: list[np.ndarray]
    p: np.ndarray
    mass: np.ndarray
    threshold: float
    df: int
    n_permutations: int
    exact: bool


def channel_adjacency(name: str, channels: list[str]) -> sparse.csr_matrix:
    """Which of ``channels`` are neighbours, from MNE's built-in template ``name``.

    The matrix follows the order of ``channels``.
    """
    adjacency, names = mne.channels.read_ch_adjacency(name)
    absent = [channel for channel in channels if channel not in names]
    if absent:
        raise ValueError(f"adjacency {name!r} has no channel(s) {', '.join(absent)}")
    index = [names.index(channel) for channel in channels]
    return sparse.csr_matrix(adjacency)[index][:, index]


def t_threshold(p: float, df: float) -> float:
    """|t| above which a two-sided test at ``df`` degrees of freedom has a p-value below ``p``."""
    return float(scipy_stats.t.ppf(1.0 - p / 2.0, df))


def hedges_g(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hedges' g of ``a`` - ``b`` along the first axis: Cohen's d with the small-sample correction.

    d is the difference of the means over the pooled standard deviation, and g is d times
    ``1 - 3 / (4 (n_a + n_b) - 9)``.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n_a, n_b = a.shape[0], b.shape[0]
    pooled = ((n_a - 1) * a.var(axis=0, ddof=1) + (n_b - 1) * b.var(axis=0, ddof=1)) / (
        n_a + n_b - 2
    )
    d = (a.mean(axis=0) - b.mean(axis=0)) / np.sqrt(pooled)
    return d * (1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0))


def paired_hedges_g(differences: np.ndarray) -> np.ndarray:
    """Hedges' g_z of paired ``differences`` along the first axis.

    The mean difference over the standard deviation of the differences, times
    ``1 - 3 / (4 (n - 1) - 1)``. It measures how consistent the change is within people, and is
    not comparable with a between-group g.
    """
    differences = np.asarray(differences, dtype=float)
    n = differences.shape[0]
    d = differences.mean(axis=0) / differences.std(axis=0, ddof=1)
    return d * (1.0 - 3.0 / (4.0 * (n - 1) - 1.0))


def residualise(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """``values`` less what a least-squares fit on ``covariates`` and an intercept explains.

    ``values`` is observations x channels and ``covariates`` observations x covariates; every
    channel is fitted separately.
    """
    values = np.asarray(values, dtype=float)
    covariates = np.asarray(covariates, dtype=float).reshape(len(values), -1)
    design = np.column_stack([np.ones(len(values)), covariates])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def _result(t, clusters, p, h0, threshold, df, exact) -> ClusterTest:
    t = np.asarray(t, dtype=float)
    clusters = [np.asarray(cluster, dtype=bool) for cluster in clusters]
    return ClusterTest(
        t=t,
        clusters=clusters,
        p=np.asarray(p, dtype=float),
        mass=np.array([t[cluster].sum() for cluster in clusters]),
        threshold=threshold,
        df=int(df),
        n_permutations=int(np.size(h0)),
        exact=exact,
    )


def two_sample_test(
    a: np.ndarray,
    b: np.ndarray,
    adjacency: sparse.spmatrix,
    *,
    n_permutations: int,
    forming_p: float,
    rng: np.random.Generator,
    df: int | None = None,
) -> ClusterTest:
    """Clusters of channels where group ``a`` differs from group ``b`` (observations x channels).

    The statistic is Student's t of a - b with pooled variance, two-sided, and the threshold is
    :func:`t_threshold` of ``forming_p`` at ``df`` degrees of freedom, by default
    ``n_a + n_b - 2``. Pass a smaller ``df`` when the values are residuals of a fit.
    """
    df = a.shape[0] + b.shape[0] - 2 if df is None else df
    threshold = t_threshold(forming_p, df)
    t, clusters, p, h0 = mne.stats.permutation_cluster_test(
        [np.asarray(a, dtype=float), np.asarray(b, dtype=float)],
        threshold=threshold,
        n_permutations=n_permutations,
        tail=0,
        stat_fun=mne.stats.ttest_ind_no_p,
        adjacency=adjacency,
        out_type="mask",
        n_jobs=1,
        rng=rng,
        verbose="error",
    )
    return _result(t, clusters, p, h0, threshold, df, exact=False)


def paired_test(
    differences: np.ndarray,
    adjacency: sparse.spmatrix,
    *,
    n_permutations: int,
    forming_p: float,
    rng: np.random.Generator,
) -> ClusterTest:
    """Clusters of channels where paired ``differences`` (pairs x channels) are not centred on 0.

    One-sample t, two-sided, with ``n - 1`` degrees of freedom; the relabellings flip the sign
    of whole pairs. When there are fewer distinct flips than ``n_permutations`` (``2 ** (n - 1)``
    up to the symmetry of a two-sided test), all are used and the test is exact.
    """
    differences = np.asarray(differences, dtype=float)
    n = differences.shape[0]
    df = n - 1
    threshold = t_threshold(forming_p, df)
    t, clusters, p, h0 = mne.stats.permutation_cluster_1samp_test(
        differences,
        threshold=threshold,
        n_permutations=n_permutations,
        tail=0,
        adjacency=adjacency,
        out_type="mask",
        n_jobs=1,
        rng=rng,
        verbose="error",
    )
    exact = 2 ** (n - 1) - 1 < n_permutations
    return _result(t, clusters, p, h0, threshold, df, exact=exact)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values of ``p``: a p-value is significant at FDR q when q-value < q."""
    p = np.asarray(p, dtype=float)
    if not p.size:
        return p.copy()
    return scipy_stats.false_discovery_control(p, method="bh")
