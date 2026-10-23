"""Which features are tested, and on whose recordings.

A feature is one value per channel of a recording. It is named as in the wide tables, without
the channel: ``<series>_<fit range>_<feature>`` for MFDFA, where the series is ``broadband`` or
``envelope_<band>``, and ``<feature>_<band>`` for band power. The primary features are the ones
``configs/stats.yaml`` names; every other feature of the tables it lists is exploratory.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from pdeeg.config import Config, MfdfaConfig
from pdeeg.features.psd import feature_name
from pdeeg.features.scaling import BROADBAND, fit_ranges, series_names
from pdeeg.features.tables import CHANNEL, ENVELOPE

PRIMARY, EXPLORATORY = "primary", "exploratory"
MFDFA, PSD = "mfdfa", "psd"
RECORDING_INDEX = ["subject", "session"]


class StatsError(Exception):
    """The configuration or the tables do not allow the analysis to be run."""


@dataclass(frozen=True)
class Feature:
    """One feature: its name, the table it is in, and how to find its rows there.

    ``selection`` pairs columns of the long table with the value that picks this feature's
    rows; ``label`` is how figures and the report call it.
    """

    key: str
    table: str
    selection: tuple[tuple[str, str], ...]
    label: str
    tier: str = EXPLORATORY


def series_key(name: str, fit_range: str) -> str:
    """Name of an MFDFA series and fit range, as feature names begin with it."""
    return f"{BROADBAND}_{fit_range}" if name == BROADBAND else f"{ENVELOPE}_{name}_{fit_range}"


def mfdfa_series(config: MfdfaConfig) -> dict[str, tuple[str, str]]:
    """Every analysed series and fit range: ``series_key`` -> (series name, fit range)."""
    return {
        series_key(name, fit_range): (name, fit_range)
        for name in series_names(config)
        for fit_range in fit_ranges(config, name)
    }


def series_label(name: str, fit_range: str, config: MfdfaConfig) -> str:
    """How the report calls a series and fit range, e.g. ``beta envelope``."""
    if name == BROADBAND:
        return f"broadband EEG, {fit_range} range"
    single = len(config.envelope.fit_ranges) == 1
    return f"{name} envelope" + ("" if single else f", {fit_range} range")


def _all_features(config: Config) -> list[Feature]:
    """Every feature of the wide tables, MFDFA first, in the order of the configuration."""
    features = []
    settings = config.mfdfa
    for key, (name, fit_range) in mfdfa_series(settings).items():
        section = settings.broadband if name == BROADBAND else settings.envelope
        variant = BROADBAND if name == BROADBAND else ENVELOPE
        for feature in section.wide_features[fit_range]:
            features.append(
                Feature(
                    key=f"{key}_{feature}",
                    table=MFDFA,
                    selection=(
                        ("variant", variant),
                        ("band", name),
                        ("fit_range", fit_range),
                        ("feature", feature),
                    ),
                    label=f"{feature}, {series_label(name, fit_range, settings)}",
                )
            )
    power = feature_name(config.psd)
    for band in config.psd.bands:
        features.append(
            Feature(
                key=f"{power}_{band}",
                table=PSD,
                selection=(("band", band), ("feature", power)),
                label=f"{band} power ({power.replace('_', ' ')})",
            )
        )
    return features


def catalogue(config: Config) -> list[Feature]:
    """The features that are tested: the primary ones in the configured order, then the rest.

    Raises :class:`StatsError` when a primary feature or a curve series is not in the tables.
    """
    settings = config.stats.features
    known = {feature.key: feature for feature in _all_features(config)}
    unknown = [key for key in settings.primary if key not in known]
    if unknown:
        raise StatsError(
            f"stats.features.primary: no feature called {', '.join(unknown)}; the tables have "
            f"{', '.join(known)}"
        )
    series = mfdfa_series(config.mfdfa)
    unknown = [key for key in config.stats.curves if key not in series]
    if unknown:
        raise StatsError(
            f"stats.curves: no series called {', '.join(unknown)}; there are {', '.join(series)}"
        )
    primary = [dataclasses.replace(known[key], tier=PRIMARY) for key in settings.primary]
    exploratory = [
        feature
        for key, feature in known.items()
        if key not in settings.primary and feature.table in settings.exploratory_tables
    ]
    return primary + exploratory


def channel_values(
    long: pd.DataFrame, feature: Feature, segments: str, channels: Sequence[str]
) -> pd.DataFrame:
    """The feature's value in every channel of every recording in the long table ``long``.

    One row per recording (index ``subject``, ``session``), one column per channel, in the
    order of ``channels``.
    """
    rows = long[(long["level"] == CHANNEL) & (long["segments"] == segments)]
    for column, value in feature.selection:
        rows = rows[rows[column] == value]
    if rows.empty:
        raise StatsError(f"{feature.key}: no {segments} channel values in the table")
    table = rows.pivot(index=RECORDING_INDEX, columns="name", values="value")
    absent = [channel for channel in channels if channel not in table.columns]
    if absent:
        raise StatsError(f"{feature.key}: no values for channel(s) {', '.join(absent)}")
    table = table[list(channels)]
    table.columns.name = None
    return table


def between_recordings(recordings: pd.DataFrame, config: Config) -> pd.DataFrame:
    """The recordings of the PD vs HC comparison: one per participant.

    The rows of ``recordings`` whose session is ``between.pd_session`` (patients) or
    ``between.hc_session`` (controls), with a column ``side`` that is ``a`` for the patients
    and ``b`` for the controls. Differences are taken as a - b.
    """
    between = config.stats.between
    chosen = recordings[recordings["session"].isin([between.pd_session, between.hc_session])]
    repeated = chosen["subject"][chosen["subject"].duplicated()].tolist()
    if repeated:
        raise StatsError(f"more than one recording for {', '.join(repeated)} in PD vs HC")
    side = (chosen["session"] == between.pd_session).map({True: "a", False: "b"})
    return chosen.assign(side=side).reset_index(drop=True)


def paired_subjects(recordings: pd.DataFrame, config: Config, *, exclude: bool = True) -> list[str]:
    """Subjects with both sessions of the paired comparison, in the order of ``recordings``.

    With ``exclude``, the subjects listed under ``paired.exclude_subjects`` are left out.
    """
    paired = config.stats.paired
    sessions = recordings.groupby("subject", sort=False)["session"].agg(set)
    both = [subject for subject, held in sessions.items() if {paired.first, paired.second} <= held]
    if exclude:
        both = [subject for subject in both if subject not in paired.exclude_subjects]
    return both
