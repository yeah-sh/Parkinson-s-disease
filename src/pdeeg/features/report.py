"""QC report of the feature tables: fit quality, surrogates, and what the bad stretches change.

Nothing here compares groups. The report describes the features over all recordings alike, so
that it can be read before any hypothesis is tested without giving a result away.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pdeeg.config import Config, MfdfaConfig
from pdeeg.features.extract import CLEAN, FULL, IAAFT, N_SCALES, SHUFFLE, cache_paths
from pdeeg.features.scaling import BROADBAND, fit_ranges, series_names
from pdeeg.features.tables import CHANNEL, ENVELOPE, RECORDING_KEYS, surrogate_feature
from pdeeg.preprocessing.pipeline import output_paths, read_log
from pdeeg.preprocessing.report import robust_z
from pdeeg.viz.features_qc import plot_full_against_clean, plot_surrogate_z
from pdeeg.viz.style import save

# A z-score above this is called wider than the surrogates. With 20 surrogates the rank test
# next to it has the level 1 / 21: the original is wider than every one of them.
Z_THRESHOLD = 2.0
_SERIES = ["variant", "band", "fit_range"]
_FIGURE_FEATURES = ("h2", "delta_alpha")


def _image(path: Path, alt: str, base: Path) -> str:
    return f"![{alt}]({Path(os.path.relpath(path, base)).as_posix()})"


def _markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _spread(values: pd.Series | np.ndarray, digits: int = 2) -> str:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return "n/a"
    low, median, high = np.percentile(values, [10, 50, 90])
    return f"{median:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> str:
    """Spearman's r of two arrays, or n/a when there are too few values or one does not vary."""
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return "n/a"
    return f"{spearmanr(a, b).statistic:.2f}"


def _percent(share: float) -> str:
    return "n/a" if not np.isfinite(share) else f"{100 * share:.1f} %"


def series_ranges(config: MfdfaConfig) -> list[tuple[str, str, str, str]]:
    """Every analysed series and fit range as (variant, band, fit range, title), in order."""
    rows = []
    for name in series_names(config):
        ranges = fit_ranges(config, name)
        for fit_range, (low, high) in ranges.items():
            what = "Broadband" if name == BROADBAND else f"{name.capitalize()} envelope"
            title = f"{what}, {fit_range} ({low:g}-{high:g} s)"
            rows.append((BROADBAND if name == BROADBAND else ENVELOPE, name, fit_range, title))
    return rows


def channel_features(long: pd.DataFrame) -> pd.DataFrame:
    """The channel rows of the MFDFA long table with one column per feature."""
    channels = long[long["level"] == CHANNEL]
    index = [*RECORDING_KEYS, "name", *_SERIES, "segments"]
    table = channels.pivot(index=index, columns="feature", values="value").reset_index()
    table.columns.name = None
    return table


def _select(table: pd.DataFrame, variant: str, band: str, fit_range: str, segments: str):
    return table[
        (table["variant"] == variant)
        & (table["band"] == band)
        & (table["fit_range"] == fit_range)
        & (table["segments"] == segments)
    ]


def _recording_label(row) -> str:
    return f"sub-{row['subject']} {row['session']}"


def bad_time(config: Config, recordings: pd.DataFrame) -> pd.Series:
    """Percentage of each recording marked bad, indexed like ``recordings``."""
    values = []
    for row in recordings.to_dict("records"):
        log = read_log(cache_paths(config, row["subject"], row["session"])["log"])
        values.append(np.nan if log is None else log["bad_percent"])
    return pd.Series(values, index=recordings.index, dtype=float)


def fit_quality(table: pd.DataFrame, config: MfdfaConfig) -> list[list[str]]:
    """Rows of the fit-quality table: one per series, fit range and ``segments``."""
    rows = []
    for variant, band, fit_range, title in series_ranges(config):
        for segments in (FULL, CLEAN):
            fits = _select(table, variant, band, fit_range, segments)
            made = fits["min_r2"].notna()
            rows.append(
                [
                    title,
                    segments,
                    str(len(fits)),
                    _percent((fits.loc[made, "min_r2"] < config.poor_fit_r2).mean()),
                    _spread(fits["min_r2"], 3),
                    str(int((made & (fits[N_SCALES] < config.scales_per_range)).sum())),
                    str(int((~made).sum())),
                ]
            )
    return rows


def poor_fits_by_recording(table: pd.DataFrame, config: MfdfaConfig) -> pd.DataFrame:
    """Share of each recording's channels with a poor fit, one column per series and range."""
    full = table[table["segments"] == FULL].assign(
        poor=lambda t: t["min_r2"] < config.poor_fit_r2,
        recording=lambda t: "sub-" + t["subject"] + " " + t["session"],
    )
    shares = {}
    for variant, band, fit_range, title in series_ranges(config):
        fits = _select(full, variant, band, fit_range, FULL)
        shares[title] = fits.groupby("recording", sort=False)["poor"].mean()
    return pd.DataFrame(shares)


def flat_recordings(table: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Per recording, the median over channels of h2 on each broadband range, with a robust z.

    A recording far below the others on the long range has lost its slow power before it
    reached this pipeline: the fluctuation function is flat there and h2 is near zero.
    """
    columns = {}
    for fit_range in config.mfdfa.broadband.fit_ranges:
        fits = _select(table, BROADBAND, BROADBAND, fit_range, FULL)
        columns[fit_range] = fits.groupby(["subject", "session"], sort=False)["h2"].median()
    medians = pd.DataFrame(columns).reset_index()
    for fit_range in config.mfdfa.broadband.fit_ranges:
        medians[f"{fit_range}_z"] = robust_z(medians[fit_range])
    slow = []
    for row in medians.to_dict("records"):
        log = read_log(output_paths(config, row["subject"], row["session"])["log"])
        slow.append(np.nan if log is None else log.get("summary", {}).get("raw_slow_db", np.nan))
    return medians.assign(raw_slow_db=slow)


def surrogate_summary(table: pd.DataFrame, config: MfdfaConfig) -> list[list[str]]:
    """Rows of the surrogate table: one per series, fit range and kind of surrogate."""
    rows = []
    for variant, band, fit_range, title in series_ranges(config):
        fits = _select(table, variant, band, fit_range, FULL)
        for kind in (IAAFT, SHUFFLE):
            z = fits[surrogate_feature(kind, "z")]
            p = fits[surrogate_feature(kind, "p")]
            n_surrogates = getattr(config.surrogates, f"n_{kind}")
            per_recording = fits.assign(z=z).groupby(["subject", "session"], sort=False)["z"]
            rows.append(
                [
                    title,
                    kind,
                    _spread(fits["delta_alpha"]),
                    _spread(fits[surrogate_feature(kind, "mean")]),
                    _spread(z, 1),
                    _percent((z > Z_THRESHOLD).mean()),
                    _percent((p <= 1.0 / (n_surrogates + 1) + 1e-12).mean()),
                    _percent((z < -Z_THRESHOLD).mean()),
                    f"{int((per_recording.median() > Z_THRESHOLD).sum())} of "
                    f"{per_recording.ngroups}",
                    _spread(fits["h2"] - fits[surrogate_feature(kind, "mean", "h2")], 3),
                ]
            )
    return rows


def full_against_clean(
    table: pd.DataFrame, config: MfdfaConfig, with_bad: pd.DataFrame
) -> tuple[list[list[str]], dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    """Whole recording against clean windows, for the recordings in ``with_bad``.

    Returns the rows of the comparison table, one per series, fit range and feature, and the
    paired values per series and range for the figure.
    """
    keys = ["subject", "session"]
    table = table.merge(with_bad[keys], on=keys)
    rows, pairs = [], {}
    for variant, band, fit_range, title in series_ranges(config):
        index = [*keys, "name"]
        full = _select(table, variant, band, fit_range, FULL).set_index(index)
        clean = _select(table, variant, band, fit_range, CLEAN).set_index(index).reindex(full.index)
        pairs[title] = {}
        for feature in config.features:
            a, b = full[feature].to_numpy(), clean[feature].to_numpy()
            pairs[title][feature] = (a, b)
            both = np.isfinite(a) & np.isfinite(b)
            difference = np.abs(b - a)[both]
            spread = float(np.nanstd(a))
            rows.append(
                [
                    title,
                    feature,
                    str(int(both.sum())),
                    f"{np.median(difference):.3f}" if difference.size else "n/a",
                    f"{np.percentile(difference, 90):.3f}" if difference.size else "n/a",
                    f"{difference.max():.3f}" if difference.size else "n/a",
                    f"{spread:.3f}",
                    _rank_correlation(a[both], b[both]),
                ]
            )
    return rows, pairs


def _psd_rows(psd: pd.DataFrame, with_bad: pd.DataFrame) -> list[list[str]]:
    channels = psd[psd["level"] == CHANNEL]
    keys = ["subject", "session", "name", "band"]
    full = channels[channels["segments"] == FULL].set_index(keys)["value"]
    clean = channels[channels["segments"] == CLEAN].set_index(keys)["value"].reindex(full.index)
    difference = (clean - full).abs().reset_index()
    difference = difference.merge(with_bad[["subject", "session"]], on=["subject", "session"])
    rows = []
    for band in dict.fromkeys(channels["band"]):
        values = full[full.index.get_level_values("band") == band]
        moved = difference.loc[difference["band"] == band, "value"]
        rows.append(
            [
                band,
                _spread(values),
                f"{moved.median():.3f}" if len(moved) else "n/a",
                f"{moved.quantile(0.9):.3f}" if len(moved) else "n/a",
                f"{moved.max():.3f}" if len(moved) else "n/a",
            ]
        )
    return rows


def write_report(
    config: Config, recordings: pd.DataFrame, mfdfa: pd.DataFrame, psd: pd.DataFrame | None = None
) -> Path:
    """Write the Markdown QC report of the feature tables and its figures; returns its path.

    ``mfdfa`` is the MFDFA long table and ``psd``, if given, the band-power long table.
    """
    settings = config.mfdfa
    extraction = settings.extraction
    report, base = extraction.report, extraction.report.parent
    table = channel_features(mfdfa)
    n_recordings = len(recordings)
    n_channels = table["name"].nunique()
    bad = bad_time(config, recordings)
    with_bad = recordings[bad > 0]
    panels = {
        title: (variant, band, fit_range)
        for variant, band, fit_range, title in series_ranges(settings)
    }
    short_titles = {title: title.split(" (")[0] for title in panels}

    # --- figures
    z_values = {
        short_titles[title]: {
            kind: _select(table, *key, FULL)[surrogate_feature(kind, "z")].to_numpy()
            for kind in (IAAFT, SHUFFLE)
        }
        for title, key in panels.items()
    }
    z_figure = extraction.figures_dir / "surrogate_z.png"
    save(
        plot_surrogate_z(
            z_values,
            {
                IAAFT: "z against IAAFT surrogates",
                SHUFFLE: "z against shuffled surrogates",
            },
            Z_THRESHOLD,
            f"Is delta_alpha wider than in surrogates? {n_recordings} recordings, "
            f"{n_channels} channels each",
        ),
        z_figure,
    )
    comparison_rows, pairs = full_against_clean(table, settings, with_bad)
    clean_figure = extraction.figures_dir / "full_against_clean.png"
    save(
        plot_full_against_clean(
            {short_titles[title]: values for title, values in pairs.items()},
            _FIGURE_FEATURES,
            "Whole recording against clean windows only",
            f"One point per channel of the {len(with_bad)} recordings with a bad stretch. "
            "On the line, the bad stretches changed nothing.",
        ),
        clean_figure,
    )

    # --- text
    surrogates = settings.surrogates
    limits = settings.bad_windows
    level = 1.0 / (surrogates.n_iaaft + 1)
    lines = [
        "# MFDFA feature QC",
        "",
        f"Written by `pdeeg mfdfa-features`. {n_recordings} recordings, {n_channels} channels "
        f"each. Nothing below compares groups.",
        "",
        "## What was computed",
        "",
        f"- **Series:** the cleaned EEG (broadband) and the Hilbert envelope in "
        f"{', '.join(settings.envelope.bands)}.",
        "- **Fit ranges:** "
        + "; ".join(title[0].lower() + title[1:] for title in panels)
        + f". {settings.scales_per_range} scales per range, q from {settings.q_min:g} to "
        f"{settings.q_max:g} in steps of {settings.q_step:g}, detrending order "
        f"{settings.detrend_order}.",
        "- **Segments:** `full` uses every window. `clean` drops every window that touches a "
        "stretch annotated BAD (widened by "
        f"{settings.envelope.edge_trim:g} s on both sides for an envelope); nothing is cut out "
        f"and joined. A scale with fewer than {limits.min_windows} clean windows is left out of "
        f"the fit, and no fit is made on fewer than {limits.min_scales} scales. "
        f"{len(with_bad)} of the {n_recordings} recordings have a bad stretch"
        + (
            f" ({bad[bad > 0].min():.1f} to {bad.max():.1f} % of the time); for the others "
            "`clean` equals `full`."
            if len(with_bad)
            else "."
        ),
        f"- **Surrogates:** {surrogates.n_iaaft} IAAFT and {surrogates.n_shuffle} shuffled per "
        "channel and series, made from the whole series and analysed like `full`.",
        "",
        "## Fit quality",
        "",
        f"A fit is poor when its worst R² over q is below {settings.poor_fit_r2:g}. R² is low "
        "when the points scatter about the line, and also when the line is nearly level: a "
        "fluctuation function that does not rise has no slope to explain.",
        "",
        *_markdown_table(
            [
                "Series and fit range",
                "Segments",
                "Channel fits",
                "Poor fits",
                "min R²: median [10 %, 90 %]",
                "Fits on fewer scales",
                "Fits not made",
            ],
            fit_quality(table, settings),
        ),
        "",
    ]

    poor = poor_fits_by_recording(table, settings)
    worst = poor[(poor >= 0.25).any(axis=1)]
    if not worst.empty:
        worst = worst.loc[worst.max(axis=1).sort_values(ascending=False).index]
        lines += [
            "Recordings in which a quarter or more of the channels fit poorly on some range "
            "(whole recording):",
            "",
            *_markdown_table(
                ["Recording", *(short_titles[title] for title in poor.columns)],
                [
                    [name, *(_percent(share) for share in row)]
                    for name, row in zip(worst.index, worst.to_numpy(), strict=True)
                ],
            ),
            "",
        ]
    lines += [
        f"{int((poor.max(axis=1) < 0.25).sum())} of the {len(poor)} recordings have fewer than a "
        "quarter of their channels fitting poorly on every range.",
        "",
    ]

    flat = flat_recordings(table, config)
    outlier_z = config.preprocessing.qc.outlier_z
    lines += [
        "## Recordings without slow power",
        "",
        "Median over channels of h2 on each broadband range, per recording. A recording that "
        "was high-pass filtered before it reached this pipeline has a flat fluctuation function "
        "on the long range, so its h2 there is near zero whatever the brain did.",
        "",
    ]
    low = flat[flat["long_z"] < -outlier_z] if "long_z" in flat else flat.iloc[:0]
    if low.empty:
        lines += ["No recording stands out.", ""]
    else:
        rest = flat.drop(index=low.index)
        lines += [
            *_markdown_table(
                [
                    "Recording",
                    "Median h2, long",
                    "Robust z",
                    "Median h2, short",
                    "Power below 0.5 Hz before filtering (dB re 1-4 Hz)",
                ],
                [
                    [
                        _recording_label(row),
                        f"{row['long']:.2f}",
                        f"{row['long_z']:+.1f}",
                        f"{row['short']:.2f}" if "short" in row else "n/a",
                        f"{row['raw_slow_db']:.0f}",
                    ]
                    for row in low.sort_values("long").to_dict("records")
                ],
            ),
            "",
            f"Listed: more than {outlier_z:g} robust standard deviations below the median. The "
            f"other {len(rest)} recordings have a median h2 on the long range between "
            f"{rest['long'].min():.2f} and {rest['long'].max():.2f}, and power below 0.5 Hz "
            f"before filtering between {rest['raw_slow_db'].min():.0f} and "
            f"{rest['raw_slow_db'].max():.0f} dB.",
            "",
        ]

    lines += [
        "## Surrogates: is the multifractality more than the spectrum and the distribution?",
        "",
        "An IAAFT surrogate has the amplitude distribution of the series and, closely, its "
        "power spectrum, with the Fourier phases randomised. If delta_alpha of the series is no "
        "wider than that of its IAAFT surrogates, its width is explained by linear correlations "
        "and the distribution. A shuffled surrogate keeps the distribution alone.",
        "",
        f"z is (delta_alpha - mean of the surrogates) / their standard deviation. The rank test "
        f"asks whether the series is wider than every one of its surrogates; for a series that "
        f"is itself such a surrogate that happens with probability {level:.3f}, so "
        f"{100 * level:.1f} % of channels would pass by chance. Channels of one recording are "
        "not independent, so the last-but-one column counts recordings instead.",
        "",
        *_markdown_table(
            [
                "Series and fit range",
                "Surrogate",
                "delta_alpha",
                "Surrogate mean",
                "z",
                f"z > {Z_THRESHOLD:g}",
                "Wider than every surrogate",
                f"z < -{Z_THRESHOLD:g}",
                f"Recordings with median z > {Z_THRESHOLD:g}",
                "h2 - surrogate h2",
            ],
            surrogate_summary(table, settings),
        ),
        "",
        "Cells with brackets are the median with the 10th and 90th percentile over all channels "
        "of all recordings. The last column checks the surrogates themselves: IAAFT should "
        "leave h2 where it was, and shuffling should bring it to 0.5.",
        "",
        _image(z_figure, "z-scores of delta_alpha against surrogates", base),
        "",
        "## Whole recording against clean windows",
        "",
        f"For the channels of the {len(with_bad)} recordings with a bad stretch: the absolute "
        "difference between a feature on clean windows and on the whole recording. The spread "
        "is the standard deviation of the feature over those same channels, as a yardstick.",
        "",
        *_markdown_table(
            [
                "Series and fit range",
                "Feature",
                "Channel pairs",
                "Median difference",
                "90th percentile",
                "Largest",
                "Spread",
                "Spearman r",
            ],
            comparison_rows,
        ),
        "",
        _image(clean_figure, "Features on the whole recording against clean windows", base),
        "",
    ]

    if psd is not None:
        feature = psd["feature"].iloc[0]
        lines += [
            "## Band power",
            "",
            f"`{feature}` per channel, from the Welch spectrum ({config.psd.window_sec:g} s "
            f"segments, {100 * config.psd.overlap:g} % overlap) between {config.psd.fmin:g} and "
            f"{config.psd.fmax:g} Hz. The difference columns are for the channels of the "
            f"{len(with_bad)} recordings with a bad stretch: clean segments against all segments.",
            "",
            *_markdown_table(
                [
                    "Band",
                    "Median [10 %, 90 %]",
                    "Median difference",
                    "90th percentile",
                    "Largest",
                ],
                _psd_rows(psd, with_bad),
            ),
            "",
        ]

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report
