"""Every test of the statistical analysis, with the false discovery rate, as tables.

Each feature of :func:`pdeeg.stats.features.catalogue` goes through four analyses:

- ``pd_vs_hc``: the patients' ``between.pd_session`` against the controls, two-sample.
- ``pd_vs_hc_adjusted``: the same on the residuals after the covariates are regressed out.
- ``off_vs_on``: within patients, ``paired.first`` - ``paired.second``, without the subjects
  under ``paired.exclude_subjects``.
- ``off_vs_on_all_pairs``: the same with every patient who has both sessions.

The first and the third are the main analyses; the other two are sensitivity analyses of them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pdeeg.config import Config
from pdeeg.stats.cluster import (
    ClusterTest,
    benjamini_hochberg,
    channel_adjacency,
    hedges_g,
    paired_hedges_g,
    paired_test,
    residualise,
    two_sample_test,
)
from pdeeg.stats.confounds import compare_groups, covariate_matrix
from pdeeg.stats.features import (
    EXPLORATORY,
    PRIMARY,
    Feature,
    StatsError,
    between_recordings,
    catalogue,
    channel_values,
    paired_subjects,
)

PD_VS_HC = "pd_vs_hc"
PD_VS_HC_ADJUSTED = "pd_vs_hc_adjusted"
OFF_VS_ON = "off_vs_on"
OFF_VS_ON_ALL = "off_vs_on_all_pairs"
ANALYSES = (PD_VS_HC, PD_VS_HC_ADJUSTED, OFF_VS_ON, OFF_VS_ON_ALL)
PAIRED = (OFF_VS_ON, OFF_VS_ON_ALL)
MAIN = (PD_VS_HC, OFF_VS_ON)
# Benjamini-Hochberg runs over the tests of one family in both contrasts. A sensitivity analysis
# is corrected in the family as it would be had it taken the place of the main analysis of its
# own contrast.
FDR_SETS = {
    PD_VS_HC: (PD_VS_HC, OFF_VS_ON),
    OFF_VS_ON: (PD_VS_HC, OFF_VS_ON),
    PD_VS_HC_ADJUSTED: (PD_VS_HC_ADJUSTED, OFF_VS_ON),
    OFF_VS_ON_ALL: (PD_VS_HC, OFF_VS_ON_ALL),
}


@dataclass(frozen=True)
class Design:
    """Who is in each analysis.

    ``between`` is :func:`pdeeg.stats.features.between_recordings`; ``pairs`` maps each paired
    analysis to its subjects; ``channels`` is the channel order of every array.
    """

    between: pd.DataFrame
    pairs: dict[str, list[str]]
    channels: tuple[str, ...]


@dataclass(frozen=True)
class Results:
    """What :func:`run_analysis` returns.

    ``tests`` has a row per analysis and feature, ``clusters`` a row per cluster found and
    ``channels`` a row per analysis, feature and channel. ``values`` maps a feature to its
    channel values in every recording (:func:`pdeeg.stats.features.channel_values`).
    """

    design: Design
    features: list[Feature]
    demographics: pd.DataFrame
    tests: pd.DataFrame
    clusters: pd.DataFrame
    channels: pd.DataFrame
    values: dict[str, pd.DataFrame]


def random_stream(seed: int, analysis: str, feature: str) -> np.random.Generator:
    """The random numbers of one test: they depend on ``seed`` and the names, and nothing else.

    A test then gives the same result whatever else is run, in whatever order and by however
    many processes.
    """
    digest = hashlib.sha256(f"{analysis}\x1f{feature}".encode()).digest()
    return np.random.default_rng(np.random.SeedSequence([seed, int.from_bytes(digest[:16], "big")]))


def design(config: Config, recordings: pd.DataFrame, channels: tuple[str, ...]) -> Design:
    """The recordings and pairs of every analysis."""
    return Design(
        between=between_recordings(recordings, config),
        pairs={
            OFF_VS_ON: paired_subjects(recordings, config, exclude=True),
            OFF_VS_ON_ALL: paired_subjects(recordings, config, exclude=False),
        },
        channels=channels,
    )


@dataclass(frozen=True)
class _Inputs:
    """The arrays of one test (observations x channels) and who they came from."""

    a: np.ndarray
    b: np.ndarray
    # The values the test is run on: residuals in the adjusted analysis, else a and b.
    tested_a: np.ndarray
    tested_b: np.ndarray
    df: int | None
    left_out: list[str]


def _rows(values: pd.DataFrame, keys: list[tuple[str, str]]) -> np.ndarray:
    return values.loc[pd.MultiIndex.from_tuples(keys, names=values.index.names)].to_numpy()


def analysis_inputs(analysis: str, values: pd.DataFrame, plan: Design, config: Config) -> _Inputs:
    """The arrays ``analysis`` tests for a feature with ``values`` (recordings x channels).

    A participant (or pair) with a missing value in any channel is left out of this test.
    """
    if analysis in PAIRED:
        paired = config.stats.paired
        subjects = plan.pairs[analysis]
        first = _rows(values, [(subject, paired.first) for subject in subjects])
        second = _rows(values, [(subject, paired.second) for subject in subjects])
        keep = np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
        left_out = [subject for subject, kept in zip(subjects, keep, strict=True) if not kept]
        return _Inputs(first[keep], second[keep], first[keep], second[keep], None, left_out)

    rows = plan.between
    data = _rows(values, list(zip(rows["subject"], rows["session"], strict=True)))
    keep = np.isfinite(data).all(axis=1)
    left_out = rows.loc[~keep, "subject"].tolist()
    rows, data = rows[keep], data[keep]
    is_a = (rows["side"] == "a").to_numpy()
    tested, df = data, None
    if analysis == PD_VS_HC_ADJUSTED:
        covariates = covariate_matrix(rows, config.stats.covariates)
        tested = residualise(data, covariates)
        df = len(data) - 2 - covariates.shape[1]
    return _Inputs(data[is_a], data[~is_a], tested[is_a], tested[~is_a], df, left_out)


def _run(
    analysis: str, inputs: _Inputs, adjacency, config: Config, feature: str
) -> tuple[ClusterTest, np.ndarray]:
    cluster = config.stats.cluster
    rng = random_stream(config.stats.seed, analysis, feature)
    if analysis in PAIRED:
        differences = inputs.tested_a - inputs.tested_b
        result = paired_test(
            differences,
            adjacency,
            n_permutations=cluster.n_permutations,
            forming_p=cluster.cluster_forming_p,
            rng=rng,
        )
        return result, paired_hedges_g(differences)
    result = two_sample_test(
        inputs.tested_a,
        inputs.tested_b,
        adjacency,
        n_permutations=cluster.n_permutations,
        forming_p=cluster.cluster_forming_p,
        rng=rng,
        df=inputs.df,
    )
    return result, hedges_g(inputs.tested_a, inputs.tested_b)


def _tables(
    analysis: str,
    feature: Feature,
    inputs: _Inputs,
    result: ClusterTest,
    g: np.ndarray,
    channels: tuple[str, ...],
) -> tuple[dict, list[dict], pd.DataFrame]:
    """The test row, cluster rows and channel rows of one test."""
    order = sorted(range(len(result.clusters)), key=lambda i: (result.p[i], -abs(result.mass[i])))
    cluster_of = np.zeros(len(channels), dtype=int)
    cluster_rows = []
    for number, i in enumerate(order, start=1):
        mask = result.clusters[i]
        cluster_of[mask] = number
        inside = np.flatnonzero(mask)
        peak = inside[np.argmax(np.abs(result.t[inside]))]
        cluster_rows.append(
            {
                "analysis": analysis,
                "feature": feature.key,
                "tier": feature.tier,
                "cluster": number,
                "sign": int(np.sign(result.mass[i])),
                "channels": ", ".join(channels[j] for j in inside),
                "n_channels": int(mask.sum()),
                "mass": float(result.mass[i]),
                "p": float(result.p[i]),
                "mean_g": float(g[mask].mean()),
                "peak_channel": channels[peak],
                "peak_t": float(result.t[peak]),
                "peak_g": float(g[peak]),
            }
        )
    largest = int(np.argmax(np.abs(g)))
    test_row = {
        "analysis": analysis,
        "feature": feature.key,
        "tier": feature.tier,
        "n_a": int(inputs.a.shape[0]),
        "n_b": int(inputs.b.shape[0]),
        "left_out": ", ".join(inputs.left_out),
        "df": result.df,
        "threshold": result.threshold,
        "n_permutations": result.n_permutations,
        "exact": result.exact,
        "n_clusters": len(result.clusters),
        "min_p": float(result.p.min()) if result.p.size else np.nan,
        "max_abs_g": float(abs(g[largest])),
        "max_abs_g_channel": channels[largest],
    }
    channel_rows = pd.DataFrame(
        {
            "analysis": analysis,
            "feature": feature.key,
            "tier": feature.tier,
            "channel": list(channels),
            "mean_a": inputs.a.mean(axis=0),
            "mean_b": inputs.b.mean(axis=0),
            "t": result.t,
            "g": g,
            "cluster": cluster_of,
        }
    )
    return test_row, cluster_rows, channel_rows


def add_fdr(
    tests: pd.DataFrame, clusters: pd.DataFrame, level: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benjamini-Hochberg q-values for every cluster, and the smallest of them for every test.

    Within a tier (primary or exploratory), each analysis is corrected over the analyses of its
    ``FDR_SETS`` entry. A test that found no cluster takes part as a single p-value of 1; its
    ``q`` is the q-value of that stand-in. ``significant`` is ``q < level``; ``m`` is how many
    p-values the correction ran over.
    """
    tests = tests.assign(q=np.nan, m=0)
    clusters = clusters.assign(q=np.nan)
    for tier in (PRIMARY, EXPLORATORY):
        for analysis, members in FDR_SETS.items():
            entries = []  # (analysis, feature, cluster number or 0, p)
            for row in tests[
                (tests["tier"] == tier) & tests["analysis"].isin(members)
            ].itertuples():
                found = clusters[
                    (clusters["analysis"] == row.analysis) & (clusters["feature"] == row.feature)
                ]
                if found.empty:
                    entries.append((row.analysis, row.feature, 0, 1.0))
                else:
                    entries += [
                        (row.analysis, row.feature, number, p)
                        for number, p in zip(found["cluster"], found["p"], strict=True)
                    ]
            if not entries:
                continue
            q_values = benjamini_hochberg(np.array([entry[3] for entry in entries]))
            for (owner, feature, number, _), q in zip(entries, q_values, strict=True):
                if owner != analysis:
                    continue
                is_test = (tests["analysis"] == owner) & (tests["feature"] == feature)
                tests.loc[is_test, "m"] = len(entries)
                if number == 0:
                    tests.loc[is_test, "q"] = q
                else:
                    is_cluster = (
                        (clusters["analysis"] == owner)
                        & (clusters["feature"] == feature)
                        & (clusters["cluster"] == number)
                    )
                    clusters.loc[is_cluster, "q"] = q
    smallest = clusters.groupby(["analysis", "feature"])["q"].min()
    keys = pd.MultiIndex.from_frame(tests[["analysis", "feature"]])
    found = smallest.reindex(keys).to_numpy()
    tests["q"] = np.where(np.isfinite(found), found, tests["q"])
    tests["significant"] = tests["q"] < level
    clusters["significant"] = clusters["q"] < level
    return tests, clusters


def run_analysis(
    config: Config,
    recordings: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    *,
    n_jobs: int | None = None,
) -> Results:
    """Run every test on the long feature tables ``tables`` (``mfdfa``, ``psd``)."""
    settings = config.stats
    features = catalogue(config)
    absent = sorted({feature.table for feature in features} - set(tables))
    if absent:
        raise StatsError(f"no {', '.join(absent)} feature table")
    channels = tuple(
        name for name in tables[features[0].table].query("level == 'channel'")["name"].unique()
    )
    plan = design(config, recordings, channels)
    adjacency = channel_adjacency(settings.cluster.adjacency, list(channels))
    values = {
        feature.key: channel_values(tables[feature.table], feature, settings.segments, channels)
        for feature in features
    }
    jobs = [
        (analysis, feature, analysis_inputs(analysis, values[feature.key], plan, config))
        for feature in features
        for analysis in ANALYSES
    ]
    finished = Parallel(n_jobs=settings.n_jobs if n_jobs is None else n_jobs)(
        delayed(_run)(analysis, inputs, adjacency, config, feature.key)
        for analysis, feature, inputs in jobs
    )
    test_rows, cluster_rows, channel_tables = [], [], []
    for (analysis, feature, inputs), (result, g) in zip(jobs, finished, strict=True):
        test_row, clusters, channel_rows = _tables(analysis, feature, inputs, result, g, channels)
        test_rows.append(test_row)
        cluster_rows += clusters
        channel_tables.append(channel_rows)
    cluster_columns = [
        "analysis", "feature", "tier", "cluster", "sign", "channels", "n_channels", "mass", "p",
        "mean_g", "peak_channel", "peak_t", "peak_g",
    ]  # fmt: skip
    tests, clusters = add_fdr(
        pd.DataFrame(test_rows),
        pd.DataFrame(cluster_rows, columns=cluster_columns),
        settings.fdr_q,
    )
    channel_table = pd.concat(channel_tables, ignore_index=True)
    significant = clusters.loc[clusters["significant"], ["analysis", "feature", "cluster"]]
    channel_table = channel_table.merge(
        significant.assign(significant=True), on=["analysis", "feature", "cluster"], how="left"
    )
    # Channels outside every significant cluster found no match.
    channel_table["significant"] = channel_table["significant"].eq(True)
    return Results(
        design=plan,
        features=features,
        demographics=compare_groups(plan.between),
        tests=tests,
        clusters=clusters,
        channels=channel_table,
        values=values,
    )


def result_paths(config: Config) -> dict[str, Path]:
    """Where the result tables are written."""
    dataset = config.data.dataset
    processed = config.data.paths.processed
    return {
        name: processed / f"stats_{name}_{dataset.id}-{dataset.version}.parquet"
        for name in ("tests", "clusters", "channels", "demographics")
    }


def write_results(config: Config, results: Results) -> dict[str, Path]:
    """Write the result tables as Parquet; returns their paths."""
    paths = result_paths(config)
    paths["tests"].parent.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        getattr(results, name).to_parquet(path, index=False)
    return paths
