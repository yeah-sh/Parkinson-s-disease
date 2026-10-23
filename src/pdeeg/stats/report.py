"""The results report: what was tested, what came out, and what it does and does not show."""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pdeeg.config import Config
from pdeeg.features.extract import CLEAN
from pdeeg.stats.analysis import (
    ANALYSES,
    MAIN,
    OFF_VS_ON,
    OFF_VS_ON_ALL,
    PAIRED,
    PD_VS_HC,
    PD_VS_HC_ADJUSTED,
    Results,
    random_stream,
)
from pdeeg.stats.descriptive import bootstrap_mean_ci, detectable_effect
from pdeeg.stats.features import EXPLORATORY, PRIMARY, mfdfa_series, series_label
from pdeeg.viz.stats import (
    GroupCurves,
    PairedPanel,
    Topography,
    montage_info,
    plot_curves,
    plot_paired,
    plot_topographies,
)
from pdeeg.viz.style import save

# The power a single t-test is given in the report's statement of what the sample can detect.
_POWER = 0.8
# Exploratory clusters listed in the report; the clusters table has all of them.
_LISTED_P = 0.05


def session_label(config: Config, session: str) -> str:
    """How figures and text call the people of ``session``: HC, PD OFF, PD ON."""
    group = config.data.sessions[session].group
    return group if group == "HC" else f"{group} {session.upper()}"


def analysis_sides(config: Config, analysis: str) -> tuple[str, str]:
    """The two sides of ``analysis``, a then b: the difference is a - b."""
    stats = config.stats
    if analysis in PAIRED:
        return session_label(config, stats.paired.first), session_label(config, stats.paired.second)
    return (
        session_label(config, stats.between.pd_session),
        session_label(config, stats.between.hc_session),
    )


def analysis_title(config: Config, analysis: str, results: Results) -> str:
    a, b = analysis_sides(config, analysis)
    titles = {
        PD_VS_HC: f"{a} vs {b}",
        PD_VS_HC_ADJUSTED: f"{a} vs {b}, adjusted for {' and '.join(config.stats.covariates)}",
        OFF_VS_ON: f"{a} vs {b}, {len(results.design.pairs[OFF_VS_ON])} pairs",
        OFF_VS_ON_ALL: f"{a} vs {b}, all {len(results.design.pairs[OFF_VS_ON_ALL])} pairs",
    }
    return titles[analysis]


def _image(path: Path, alt: str, base: Path) -> str:
    return f"![{alt}]({Path(os.path.relpath(path, base)).as_posix()})"


def _markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _p(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def _direction(config: Config, analysis: str, sign: int) -> str:
    a, b = analysis_sides(config, analysis)
    return f"{a} > {b}" if sign > 0 else f"{a} < {b}"


# --- figures -------------------------------------------------------------------------------------


def _limits(arrays: list[np.ndarray], *, symmetric: bool, floor: float = 0.0) -> tuple:
    values = np.concatenate([np.ravel(array) for array in arrays])
    values = values[np.isfinite(values)]
    if symmetric:
        top = max(float(np.abs(values).max()), floor)
        return (-top, top)
    return (float(values.min()), float(values.max()))


def _channel_rows(results: Results, analysis: str, feature: str) -> pd.DataFrame:
    channels = results.channels
    rows = channels[(channels["analysis"] == analysis) & (channels["feature"] == feature)]
    return rows.set_index("channel").loc[list(results.design.channels)]


def _t_map(results: Results, analysis: str, feature: str, title: str) -> Topography:
    rows = _channel_rows(results, analysis, feature)
    return Topography(
        values=rows["t"].to_numpy(),
        title=title,
        kind="t",
        limits=(0.0, 0.0),
        mask=rows["significant"].to_numpy(),
    )


def _with_limits(panels: list[Topography], floor: float) -> list[Topography]:
    """The panels with shared colour limits: one range for the means, one for the t-maps.

    The t range is symmetric about 0 and reaches at least ``floor``.
    """
    limits = {
        kind: _limits(
            [panel.values for panel in panels if panel.kind == kind],
            symmetric=kind == "t",
            floor=floor,
        )
        for kind in {panel.kind for panel in panels}
    }
    return [dataclasses.replace(panel, limits=limits[panel.kind]) for panel in panels]


def primary_topographies(
    config: Config, results: Results, main: str, sensitivity: str
) -> list[tuple[str, list[Topography]]]:
    """Rows of the primary-feature figure of one contrast: both means and both t-maps."""
    a, b = analysis_sides(config, main)
    tests = results.tests.set_index(["analysis", "feature"])
    rows = []
    for feature in (f for f in results.features if f.tier == PRIMARY):
        channel_rows = _channel_rows(results, main, feature.key)
        main_threshold = tests.loc[(main, feature.key), "threshold"]
        panels = [
            Topography(channel_rows["mean_a"].to_numpy(), f"{a} mean", "mean", (0.0, 0.0)),
            Topography(channel_rows["mean_b"].to_numpy(), f"{b} mean", "mean", (0.0, 0.0)),
            _t_map(results, main, feature.key, f"t, {a} - {b}"),
            _t_map(
                results,
                sensitivity,
                feature.key,
                "t, " + analysis_title(config, sensitivity, results).split(", ", 1)[1],
            ),
        ]
        rows.append((feature.label, _with_limits(panels, floor=1.25 * main_threshold)))
    return rows


def exploratory_topographies(
    config: Config, results: Results, analysis: str, per_row: int = 5
) -> list[tuple[str, list[Topography]]]:
    """Rows of t-maps of every exploratory feature in ``analysis``, ``per_row`` to a row."""
    features = [f for f in results.features if f.tier == EXPLORATORY]
    panels = [_t_map(results, analysis, f.key, f.label) for f in features]
    threshold = results.tests.loc[results.tests["analysis"] == analysis, "threshold"].max()
    panels = _with_limits(panels, floor=1.25 * threshold)
    return [("", panels[i : i + per_row]) for i in range(0, len(panels), per_row)]


def _recording_means(
    curves: pd.DataFrame, series: str, fit_range: str, segments: str
) -> dict[str, pd.DataFrame]:
    """Per recording and q, the mean over channels of h, alpha and f_alpha."""
    rows = curves[
        (curves["series"] == series)
        & (curves["fit_range"] == fit_range)
        & (curves["segments"] == segments)
    ]
    means = rows.groupby(["subject", "session", "q"])[["h", "alpha", "f_alpha"]].mean()
    return {name: means[name].unstack("q") for name in ("h", "alpha", "f_alpha")}


def group_curves(
    means: dict[str, pd.DataFrame], keys: list[tuple[str, str]], config: Config, label: str
) -> GroupCurves:
    """Mean curves of the recordings ``keys`` with bootstrap intervals over recordings."""
    index = pd.MultiIndex.from_tuples(keys, names=["subject", "session"])
    stacked = np.stack([means[name].reindex(index).to_numpy() for name in means], axis=1)
    stacked = stacked[np.isfinite(stacked).all(axis=(1, 2))]
    bootstrap = config.stats.bootstrap
    low, high = bootstrap_mean_ci(
        stacked,
        bootstrap.n_resamples,
        bootstrap.confidence,
        random_stream(config.stats.seed, "curves", label),
    )
    mean = stacked.mean(axis=0)
    triples = [(mean[i], low[i], high[i]) for i in range(3)]
    return GroupCurves(
        q=means["h"].columns.to_numpy(dtype=float),
        h=triples[0],
        alpha=triples[1],
        f=triples[2],
        n=int(stacked.shape[0]),
    )


def curve_groups(config: Config, results: Results, curves: pd.DataFrame) -> dict:
    """What :func:`pdeeg.viz.stats.plot_curves` draws: each configured series, both contrasts."""
    plan = results.design
    between = plan.between
    paired = config.stats.paired
    pairs = plan.pairs[OFF_VS_ON]
    figure = {}
    series = mfdfa_series(config.mfdfa)
    for key in config.stats.curves:
        name, fit_range = series[key]
        means = _recording_means(curves, name, fit_range, config.stats.segments)
        a, b = analysis_sides(config, PD_VS_HC)
        first, second = analysis_sides(config, OFF_VS_ON)
        sides = {
            side: list(zip(rows["subject"], rows["session"], strict=True))
            for side, rows in between.groupby("side")
        }
        title = series_label(name, fit_range, config.mfdfa)
        figure[title[0].upper() + title[1:]] = {
            f"{a} vs {b}": {
                b: group_curves(means, sides["b"], config, f"{key} {b}"),
                a: group_curves(means, sides["a"], config, f"{key} {a}"),
            },
            f"{first} vs {second}": {
                first: group_curves(
                    means, [(s, paired.first) for s in pairs], config, f"{key} {first} paired"
                ),
                second: group_curves(
                    means, [(s, paired.second) for s in pairs], config, f"{key} {second} paired"
                ),
            },
        }
    return figure


def paired_panels(config: Config, results: Results) -> dict[str, PairedPanel]:
    """Mean over channels of each primary feature in both sessions of every patient."""
    paired = config.stats.paired
    included = set(results.design.pairs[OFF_VS_ON])
    subjects = results.design.pairs[OFF_VS_ON_ALL]
    panels = {}
    for feature in (f for f in results.features if f.tier == PRIMARY):
        values = results.values[feature.key].mean(axis=1, skipna=False)
        table = pd.DataFrame(
            {
                "subject": subjects,
                "first": [values.get((s, paired.first), np.nan) for s in subjects],
                "second": [values.get((s, paired.second), np.nan) for s in subjects],
            }
        )
        table["included"] = table["subject"].isin(included)
        table = table.dropna(subset=["first", "second"])
        kept = table.loc[table["included"], ["first", "second"]].to_numpy()
        bootstrap = config.stats.bootstrap
        low, high = bootstrap_mean_ci(
            kept,
            bootstrap.n_resamples,
            bootstrap.confidence,
            random_stream(config.stats.seed, "paired figure", feature.key),
        )
        panels[feature.label] = PairedPanel(table, kept.mean(axis=0), low, high)
    return panels


# --- text ----------------------------------------------------------------------------------------


def _best_cluster(results: Results, analysis: str, feature: str) -> pd.Series | None:
    clusters = results.clusters
    found = clusters[(clusters["analysis"] == analysis) & (clusters["feature"] == feature)]
    return None if found.empty else found.sort_values(["p", "cluster"]).iloc[0]


def result_rows(config: Config, results: Results, tier: str, analyses) -> list[list[str]]:
    """Rows of a results table: one per feature of ``tier`` and analysis in ``analyses``."""
    labels = {f.key: f.label for f in results.features}
    tests = results.tests
    rows = []
    for feature in (f.key for f in results.features if f.tier == tier):
        for analysis in analyses:
            test = tests[(tests["analysis"] == analysis) & (tests["feature"] == feature)].iloc[0]
            best = _best_cluster(results, analysis, feature)
            n = f"{test['n_a']} / {test['n_b']}" if analysis not in PAIRED else f"{test['n_a']}"
            if best is None:
                cluster = ["none", "", "", "", _p(test["q"])]
            else:
                cluster = [
                    _direction(config, analysis, best["sign"]),
                    best["channels"],
                    _p(best["p"]),
                    f"{best['mean_g']:+.2f}",
                    _p(best["q"]) + (" **(significant)**" if best["significant"] else ""),
                ]
            rows.append(
                [
                    labels[feature],
                    analysis_title(config, analysis, results),
                    n,
                    str(test["n_clusters"]),
                    *cluster,
                    f"{test['max_abs_g']:.2f} ({test['max_abs_g_channel']})",
                ]
            )
    return rows


_RESULT_HEADER = [
    "Feature",
    "Analysis",
    "n",
    "Clusters",
    "Best cluster",
    "Its channels",
    "Its p",
    "Its mean g",
    "Its q",
    "Largest |g| (channel)",
]


def cluster_rows(config: Config, results: Results, chosen: pd.DataFrame) -> list[list[str]]:
    labels = {f.key: f.label for f in results.features}
    return [
        [
            labels[row["feature"]],
            analysis_title(config, row["analysis"], results),
            str(row["cluster"]),
            _direction(config, row["analysis"], row["sign"]),
            row["channels"],
            f"{row['mass']:+.1f}",
            _p(row["p"]),
            _p(row["q"]),
            f"{row['mean_g']:+.2f}",
            f"{row['peak_channel']} (t = {row['peak_t']:+.2f}, g = {row['peak_g']:+.2f})",
        ]
        for row in chosen.to_dict("records")
    ]


def _found(results: Results, tier: str, analysis: str) -> pd.DataFrame:
    tests = results.tests
    return tests[(tests["tier"] == tier) & (tests["analysis"] == analysis) & tests["significant"]]


def _describe_significant(config: Config, results: Results, analysis: str, tier: str) -> list[str]:
    labels = {f.key: f.label for f in results.features}
    clusters = results.clusters
    lines = []
    for feature in _found(results, tier, analysis)["feature"]:
        chosen = clusters[
            (clusters["analysis"] == analysis)
            & (clusters["feature"] == feature)
            & clusters["significant"]
        ]
        for row in chosen.to_dict("records"):
            lines.append(
                f"{labels[feature]}: {_direction(config, analysis, row['sign'])} over "
                f"{row['n_channels']} channel(s) ({row['channels']}); cluster p = {_p(row['p'])}, "
                f"q = {_p(row['q'])}, mean Hedges' g = {row['mean_g']:+.2f}"
            )
    return lines


def _smallest(results: Results, tier: str, analysis: str) -> str:
    tests = results.tests
    chosen = tests[(tests["tier"] == tier) & (tests["analysis"] == analysis)]
    best = chosen.sort_values("q").iloc[0]
    labels = {f.key: f.label for f in results.features}
    return f"smallest q = {_p(best['q'])}, for {labels[best['feature']]}"


def plain_summary(config: Config, results: Results) -> list[str]:
    """The plain-language summary, built from the numbers so that it says no more than they do."""
    plan = results.design
    a, b = analysis_sides(config, PD_VS_HC)
    first, second = analysis_sides(config, OFF_VS_ON)
    n_a = int((plan.between["side"] == "a").sum())
    n_b = int((plan.between["side"] == "b").sum())
    n_pairs = len(plan.pairs[OFF_VS_ON])
    n_primary = sum(f.tier == PRIMARY for f in results.features)
    lines = []

    for main, sensitivity, what in (
        (PD_VS_HC, PD_VS_HC_ADJUSTED, f"**{a} ({n_a}) against {b} ({n_b}).**"),
        (OFF_VS_ON, OFF_VS_ON_ALL, f"**{first} against {second} ({n_pairs} patients).**"),
    ):
        found = _describe_significant(config, results, main, PRIMARY)
        if not found:
            text = (
                f"{what} No cluster of channels in any of the {n_primary} pre-registered "
                f"features survived the false discovery rate correction "
                f"({_smallest(results, PRIMARY, main)})."
            )
        else:
            text = f"{what} After correction: " + "; ".join(found) + "."
        again = set(_found(results, PRIMARY, sensitivity)["feature"])
        before = set(_found(results, PRIMARY, main)["feature"])
        label = analysis_title(config, sensitivity, results).split(", ", 1)[1]
        if again == before:
            text += f" The sensitivity analysis ({label}) gives the same answer."
        else:
            labels = {f.key: f.label for f in results.features}
            gained = ", ".join(labels[k] for k in sorted(again - before)) or "none"
            lost = ", ".join(labels[k] for k in sorted(before - again)) or "none"
            text += (
                f" The sensitivity analysis ({label}) differs: significant there but not in the "
                f"main analysis: {gained}; the reverse: {lost}. A result that depends on this "
                "choice is fragile."
            )
        lines += [f"- {text}"]

    for analysis in MAIN:
        found = _found(results, EXPLORATORY, analysis)
        n_exploratory = sum(f.tier == EXPLORATORY for f in results.features)
        labels = {f.key: f.label for f in results.features}
        names = ", ".join(labels[k] for k in found["feature"])
        lines.append(
            f"- **Exploratory, {analysis_title(config, analysis, results)}.** {len(found)} of the "
            f"{n_exploratory} exploratory features had a cluster that survived the correction "
            f"within the exploratory family" + (f": {names}." if names else ".")
        )

    demographics = results.demographics.set_index("variable")
    unbalanced = [name for name, row in demographics.iterrows() if row["p"] < 0.05]
    lines.append(
        f"- **Age and sex.** {a}: age {demographics.loc['age', 'a']}, "
        f"{demographics.loc['sex', 'a']}; {b}: age {demographics.loc['age', 'b']}, "
        f"{demographics.loc['sex', 'b']}. "
        + (
            f"The groups differ in {' and '.join(unbalanced)} (p < 0.05), so the adjusted "
            "analysis matters."
            if unbalanced
            else "Neither differs between the groups at p < 0.05; with groups this small that "
            "does not prove they are matched, which is why the adjusted analysis is reported too."
        )
    )
    between_effect = detectable_effect(n_a, n_b, _POWER)
    paired_effect = detectable_effect(n_pairs, None, _POWER)
    lines.append(
        f"- **What this sample can see.** With {n_a} and {n_b} people, a single two-sided t-test "
        f"has {_POWER:.0%} power only for differences of about g = {between_effect:.2f} or more; "
        f"with {n_pairs} pairs, for d_z = {paired_effect:.2f} or more. Effects smaller than that "
        "are easily missed, so a test without a significant cluster does not show that the "
        "groups are alike. Effects that do reach significance in samples this small tend to "
        "overstate the true size."
    )
    lines.append(
        "- **What this is not.** These are group averages from one dataset of "
        f"{n_a + n_b} people at one site. No result here is a diagnostic test or a biomarker, and "
        "the dataset's curators warn specifically against presenting classification of these "
        "patients as diagnostic. Exploratory results are leads for an independent dataset."
    )
    return lines


def _registration(config: Config) -> str:
    path = config.config_dir / "stats.yaml"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"`{path.relative_to(config.root).as_posix()}`, SHA-256 `{digest}`"


def write_report(config: Config, results: Results, curves: pd.DataFrame | None) -> Path:
    """Write the Markdown results report and its figures; returns the report's path.

    ``curves`` is the table of :mod:`pdeeg.features.curves`; without it the h(q) figure is left
    out and the report says how to make it.
    """
    settings = config.stats
    report, base = settings.report, settings.report.parent
    figures = settings.figures_dir
    plan = results.design
    info = montage_info(plan.channels, config.data.dataset.montage)
    a, b = analysis_sides(config, PD_VS_HC)
    first, second = analysis_sides(config, OFF_VS_ON)
    cluster = settings.cluster
    marked = f"White dots: channels in a cluster with q < {settings.fdr_q:g}."

    # --- figures
    paths = {}
    for main, sensitivity, name in (
        (PD_VS_HC, PD_VS_HC_ADJUSTED, "primary_pd_vs_hc"),
        (OFF_VS_ON, OFF_VS_ON_ALL, "primary_off_vs_on"),
    ):
        paths[name] = figures / f"{name}.png"
        save(
            plot_topographies(
                info,
                primary_topographies(config, results, main, sensitivity),
                f"Primary features: {analysis_title(config, main, results)}",
                f"Group means and t per channel ({settings.segments} segments). {marked}",
            ),
            paths[name],
        )
    for analysis in MAIN:
        name = f"exploratory_{analysis}"
        paths[name] = figures / f"{name}.png"
        save(
            plot_topographies(
                info,
                exploratory_topographies(config, results, analysis),
                f"Exploratory features: t per channel, {analysis_title(config, analysis, results)}",
                f"Positive: {_direction(config, analysis, 1)}. {marked} Exploratory family.",
            ),
            paths[name],
        )
    paths["paired"] = figures / "paired_off_on.png"
    save(
        plot_paired(
            paired_panels(config, results),
            (first, second),
            f"Primary features in both sessions, {len(plan.pairs[OFF_VS_ON_ALL])} patients",
            f"Mean over channels. Dots: mean of the {len(plan.pairs[OFF_VS_ON])} patients in the "
            f"main analysis, {settings.bootstrap.confidence:.0%} bootstrap interval. Dashed: "
            f"left out of it ({', '.join(settings.paired.exclude_subjects) or 'none'}).",
        ),
        paths["paired"],
    )
    if curves is not None:
        paths["curves"] = figures / "mfdfa_curves.png"
        save(
            plot_curves(
                curve_groups(config, results, curves),
                "Generalised Hurst exponent and singularity spectrum",
                f"Mean over channels, then over people ({settings.segments} segments); "
                f"{settings.bootstrap.confidence:.0%} bootstrap interval over people. "
                "Descriptive: nothing is tested on these curves.",
            ),
            paths["curves"],
        )

    # --- text
    tests = results.tests
    primary = [f for f in results.features if f.tier == PRIMARY]
    exploratory = [f for f in results.features if f.tier == EXPLORATORY]
    excluded = [s for s in plan.pairs[OFF_VS_ON_ALL] if s not in plan.pairs[OFF_VS_ON]]
    n_permutations = tests.groupby("analysis")["n_permutations"].max()
    exact = tests.groupby("analysis")["exact"].all()
    left_out = tests[tests["left_out"] != ""]
    lines = [
        "# Statistical analysis",
        "",
        f"Written by `pdeeg stats`. Pre-registered in "
        f"{_registration(config)}; the file was written before any group was compared, and a "
        "change to it since would make the analysis it affects exploratory.",
        "",
        "## Summary in plain language",
        "",
        *plain_summary(config, results),
        "",
        "## What was tested",
        "",
        f"- **PD vs HC:** {a} (session `{settings.between.pd_session}`, "
        f"{int((plan.between['side'] == 'a').sum())} patients) against {b} "
        f"({int((plan.between['side'] == 'b').sum())} controls), one recording per person.",
        f"- **OFF vs ON:** the difference {first} - {second} within "
        f"{len(plan.pairs[OFF_VS_ON])} patients. Left out: {', '.join(excluded) or 'nobody'}, "
        "whose on-medication recordings were exported already preprocessed. The sensitivity "
        f"analysis keeps all {len(plan.pairs[OFF_VS_ON_ALL])}.",
        "- **Primary features** (pre-registered): "
        + "; ".join(f"{f.label} (`{f.key}`)" for f in primary)
        + f". **Exploratory:** the other {len(exploratory)} features of the "
        f"{' and '.join(settings.features.exploratory_tables)} tables.",
        f"- **Values:** one per channel ({len(plan.channels)} channels, "
        f"{config.data.dataset.montage}), from the `{settings.segments}` segments"
        + (
            " (windows touching a stretch marked bad left out)."
            if settings.segments == CLEAN
            else "."
        ),
        f"- **Test:** cluster-based permutation test across channels (MNE). Student's t per "
        f"channel, two-sided: two-sample with pooled variance for PD vs HC, one-sample on the "
        f"differences for OFF vs ON. Channels with |t| above the two-sided p = "
        f"{cluster.cluster_forming_p:g} quantile that are neighbours ({cluster.adjacency} "
        "template) form a cluster; its mass is the sum of its t-values, and its p-value is the "
        "share of relabellings whose largest |mass| reaches it. Relabellings: "
        + "; ".join(
            f"{analysis_title(config, analysis, results)}: {n_permutations[analysis]}"
            + (" (every one, exact)" if exact[analysis] else "")
            for analysis in ANALYSES
        )
        + f". Seed {settings.seed}, a separate stream per test.",
        "- **Effect size:** Hedges' g per channel for PD vs HC (difference of means over the "
        "pooled standard deviation, corrected for small samples), Hedges' g_z for OFF vs ON (mean "
        "difference over the standard deviation of the differences). The two are not on the same "
        "scale.",
        f"- **Multiple comparisons:** within a cluster test the p-values already account for the "
        f"channels. Across tests, Benjamini-Hochberg over the cluster p-values of the primary "
        f"family in both contrasts, and separately of the exploratory family; a test without a "
        f"cluster counts as one p-value of 1. Each sensitivity analysis is corrected in the "
        f"family as it would be had it replaced the main analysis of its contrast. Significant: "
        f"q < {settings.fdr_q:g}.",
        f"- **Confounds:** age (Welch's t-test) and sex (Fisher's exact test) compared between "
        f"the groups; the PD vs HC tests repeated on the residuals of a least-squares fit of every "
        f"channel on {' and '.join(settings.covariates)} with an intercept, over both groups "
        "together, with the degrees of freedom of the threshold reduced accordingly. "
        "Residualising without the group in the model removes any group difference that age "
        "or sex can account for, so it leans conservative when the groups differ in them.",
    ]
    if not left_out.empty:
        lines.append(
            "- **Missing values:** left out of single tests: "
            + "; ".join(
                f"{row['feature']} ({row['analysis']}): {row['left_out']}"
                for row in left_out.to_dict("records")
            )
            + "."
        )
    demographics = results.demographics
    lines += [
        "",
        "## Age and sex",
        "",
        *_markdown_table(
            [
                "Variable",
                f"{a} (n = {int((plan.between['side'] == 'a').sum())})",
                f"{b} (n = {int((plan.between['side'] == 'b').sum())})",
                "Test",
                "Statistic",
                "p",
            ],
            [
                [row["variable"], row["a"], row["b"], row["test"], row["statistic"], _p(row["p"])]
                for row in demographics.to_dict("records")
            ],
        ),  # fmt: skip
        "",
        "Age is mean ± standard deviation in years.",
        "",
        "## Primary features",
        "",
        "One row per feature and analysis. The best cluster is the one with the smallest p. "
        "g is Hedges' g (PD vs HC) or g_z (OFF vs ON), signed as the direction column says "
        "(a > b is positive). n is patients / controls, or pairs. When no cluster formed, the q "
        "column is that of the test's stand-in p-value of 1.",
        "",
        *_markdown_table(_RESULT_HEADER, result_rows(config, results, PRIMARY, ANALYSES)),
        "",
        "Every cluster of the primary features:",
        "",
    ]
    clusters = results.clusters
    primary_clusters = clusters[clusters["tier"] == PRIMARY]
    cluster_header = [
        "Feature", "Analysis", "Cluster", "Direction", "Channels", "Mass (sum of t)", "p", "q",
        "Mean g", "Peak channel",
    ]  # fmt: skip
    if primary_clusters.empty:
        lines += ["No channel passed the cluster-forming threshold in any primary test.", ""]
    else:
        lines += [
            *_markdown_table(cluster_header, cluster_rows(config, results, primary_clusters)),
            "",
        ]
    lines += [
        _image(paths["primary_pd_vs_hc"], "Primary features, PD vs HC", base),
        "",
        _image(paths["primary_off_vs_on"], "Primary features, OFF vs ON", base),
        "",
        _image(paths["paired"], "Primary features in both sessions", base),
        "",
    ]
    if "curves" in paths:
        lines += [_image(paths["curves"], "h(q) and f(alpha) by group", base), ""]
    else:
        lines += ["The h(q) and f(alpha) figure needs `pdeeg mfdfa-curves` to be run first.", ""]
    listed = clusters[(clusters["tier"] == EXPLORATORY) & (clusters["p"] < _LISTED_P)]
    lines += [
        "## Exploratory features",
        "",
        "Not pre-registered. The q-values are corrected within the exploratory family only, and "
        "a significant result here is a lead, not a finding.",
        "",
        *_markdown_table(_RESULT_HEADER, result_rows(config, results, EXPLORATORY, ANALYSES)),
        "",
        f"Exploratory clusters with p < {_LISTED_P:g} before the correction (the clusters table "
        "has all of them):",
        "",
    ]
    if listed.empty:
        lines += ["None.", ""]
    else:
        lines += [*_markdown_table(cluster_header, cluster_rows(config, results, listed)), ""]
    for analysis in MAIN:
        lines += [
            _image(paths[f"exploratory_{analysis}"], f"Exploratory features, {analysis}", base),
            "",
        ]
    dataset = config.data.dataset
    lines += [
        "## Files",
        "",
        f"`data/processed/stats_*_{dataset.id}-{dataset.version}.parquet`: `tests` (one row per "
        "analysis and feature), `clusters` (every cluster found), `channels` (means, t and g per "
        "channel), `demographics`.",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report
