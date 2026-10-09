"""Feature tables: long format, region averages and a wide format for modelling.

The MFDFA tables are built from the runs stored by :mod:`pdeeg.features.extract`. Building
them is quick, so a change to a summary statistic does not repeat any analysis.

A long table has one value per row. Its columns say whose value it is (``subject``,
``session``, ``group``), where on the scalp (``level`` is ``channel`` or ``region`` and
``name`` the channel or region), what was analysed and which value it is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from pdeeg.config import Config, MfdfaConfig
from pdeeg.features.extract import (
    FULL,
    IAAFT,
    N_SCALES,
    ORIGINAL,
    RUN_VALUES,
    SHUFFLE,
    cache_paths,
    is_current,
)
from pdeeg.features.scaling import BROADBAND

RECORDING_KEYS = ("subject", "session", "group")
CHANNEL, REGION = "channel", "region"
# Columns of the MFDFA long table.
MFDFA_COLUMNS = (
    *RECORDING_KEYS,
    "level",
    "name",
    "variant",
    "band",
    "fit_range",
    "segments",
    "feature",
    "value",
)
ENVELOPE = "envelope"
# Statistics of delta_alpha over a channel's surrogates of one kind, as feature-name suffixes:
# their mean and standard deviation, the original's distance from the mean in those standard
# deviations, and the share of surrogates, the original counted in, at least as wide as it.
SURROGATE_STATISTICS = ("mean", "sd", "z", "p")
# Not averaged over a region: a mean of these says nothing about the region.
_NOT_REGIONAL = ("_p", "_sd", N_SCALES)

_UNIT = ["channel", "series", "fit_range"]


class TablesError(Exception):
    """The feature tables cannot be built from what is stored."""


def surrogate_feature(kind: str, statistic: str, feature: str = "delta_alpha") -> str:
    """Name of the feature that holds ``statistic`` of ``feature`` over surrogates of ``kind``."""
    return f"{feature}_{kind}_{statistic}"


def summarise_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Channel features of one recording from its runs.

    Returns the columns ``channel``, ``series``, ``fit_range``, ``segments``, ``feature`` and
    ``value``: every value of the original series on all windows (``full``) and without the
    windows on bad stretches (``clean``), and, as ``full``, what its surrogates say about
    ``delta_alpha`` (see ``SURROGATE_STATISTICS``) and the mean ``h2`` of the surrogates, which
    should equal the original's for IAAFT and be 0.5 for shuffles.
    """
    original = runs[runs["kind"] == ORIGINAL]
    parts = [
        original.melt(
            id_vars=[*_UNIT, "segments"],
            value_vars=list(RUN_VALUES),
            var_name="feature",
            value_name="value",
        )
    ]
    width = original[original["segments"] == FULL].set_index(_UNIT)["delta_alpha"]
    for kind in (IAAFT, SHUFFLE):
        surrogates = runs[runs["kind"] == kind]
        if surrogates.empty:
            continue
        widths = surrogates.pivot(index=_UNIT, columns="replicate", values="delta_alpha")
        observed = width.reindex(widths.index).to_numpy()
        values = widths.to_numpy()
        mean = np.nanmean(values, axis=1)
        sd = (
            np.nanstd(values, axis=1, ddof=1)
            if values.shape[1] > 1
            else np.full(mean.shape, np.nan)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (observed - mean) / sd
        at_least = (values >= observed[:, None]).sum(axis=1)
        p = (1.0 + at_least) / (1.0 + np.isfinite(values).sum(axis=1))
        p = np.where(np.isfinite(observed), p, np.nan)
        statistics = dict(zip(SURROGATE_STATISTICS, (mean, sd, z, p), strict=True))
        summary = pd.DataFrame(
            {surrogate_feature(kind, name): column for name, column in statistics.items()},
            index=widths.index,
        )
        summary[surrogate_feature(kind, "mean", "h2")] = (
            surrogates.groupby(_UNIT, sort=False)["h2"].mean().reindex(widths.index)
        )
        parts.append(
            summary.reset_index()
            .melt(id_vars=_UNIT, var_name="feature", value_name="value")
            .assign(segments=FULL)
        )
    columns = [*_UNIT, "segments", "feature", "value"]
    return pd.concat(parts, ignore_index=True)[columns]


def add_regions(
    channels: pd.DataFrame, regions: Mapping[str, Sequence[str]], keys: Sequence[str]
) -> pd.DataFrame:
    """``channels`` with ``level`` and ``name`` columns, followed by its region averages.

    ``channels`` has a ``channel`` column, the columns ``keys`` that say which value a row
    holds, and ``value``. A region's value is the mean of its channels' values, leaving out
    those that are NaN. Probabilities, standard deviations over surrogates and scale counts are
    not averaged.
    """
    known = set(channels["channel"])
    absent = sorted({name for names in regions.values() for name in names} - known)
    if absent:
        raise TablesError(f"region channel(s) not in the data: {', '.join(absent)}")
    keys = list(keys)
    by_channel = channels.rename(columns={"channel": "name"}).assign(level=CHANNEL)
    regional = channels[~channels["feature"].str.endswith(_NOT_REGIONAL)]
    parts = [by_channel]
    for region, names in regions.items():
        inside = regional[regional["channel"].isin(names)]
        mean = inside.groupby(keys, sort=False, dropna=False)["value"].mean().reset_index()
        parts.append(mean.assign(level=REGION, name=region))
    return pd.concat(parts, ignore_index=True)[["level", "name", *keys, "value"]]


def recording_features(runs: pd.DataFrame, regions: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Channel and region features of one recording; the MFDFA columns without the recording."""
    channels = summarise_runs(runs)
    broadband = channels["series"] == BROADBAND
    channels = channels.assign(
        variant=np.where(broadband, BROADBAND, ENVELOPE), band=channels["series"]
    ).drop(columns="series")
    keys = ["variant", "band", "fit_range", "segments", "feature"]
    return add_regions(channels, regions, keys)


def build_mfdfa_table(config: Config, recordings: pd.DataFrame) -> pd.DataFrame:
    """The MFDFA long table of every recording, from the stored runs.

    Raises :class:`TablesError` if a recording has no stored runs for the current
    configuration.
    """
    rows = recordings.to_dict("records")
    stale = [
        f"sub-{row['subject']} ses-{row['session']}" for row in rows if not is_current(row, config)
    ]
    if stale:
        raise TablesError(
            f"no current MFDFA runs for {len(stale)} recording(s): {', '.join(stale)}"
        )
    tables = []
    for row in rows:
        runs = pd.read_parquet(cache_paths(config, row["subject"], row["session"])["runs"])
        features = recording_features(runs, config.data.regions)
        for position, key in enumerate(RECORDING_KEYS):
            features.insert(position, key, row[key])
        tables.append(features)
    return pd.concat(tables, ignore_index=True)[list(MFDFA_COLUMNS)]


def _series_prefix(variant: pd.Series, band: pd.Series) -> pd.Series:
    return variant.where(variant == BROADBAND, variant + "_" + band)


def mfdfa_wide(long: pd.DataFrame, config: MfdfaConfig) -> pd.DataFrame:
    """One row per recording and ``segments``; one column per kept feature, range and site.

    The features kept for a fit range are its ``wide_features``. A column is called
    ``<series>_<fit range>_<feature>_<channel or region>``, where the series is ``broadband``
    or ``envelope_<band>``.
    """
    kept = []
    for variant, section in ((BROADBAND, config.broadband), (ENVELOPE, config.envelope)):
        for fit_range, features in section.wide_features.items():
            chosen = long[
                (long["variant"] == variant)
                & (long["fit_range"] == fit_range)
                & long["feature"].isin(features)
            ]
            missing = sorted(set(features) - set(chosen["feature"]))
            if missing:
                raise TablesError(
                    f"{variant} {fit_range}: no feature called {', '.join(missing)} in the table"
                )
            kept.append(chosen)
    kept = pd.concat(kept, ignore_index=True)
    column = (
        _series_prefix(kept["variant"], kept["band"])
        + "_"
        + kept["fit_range"]
        + "_"
        + kept["feature"]
        + "_"
        + kept["name"]
    )
    return _pivot(kept.assign(column=column))


def _pivot(table: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``table`` spread over its ``column`` names, in order of first appearance."""
    index = [*RECORDING_KEYS, "segments"]
    wide = table.pivot(index=index, columns="column", values="value")
    wide = wide[list(dict.fromkeys(table["column"]))]
    order = table[index].drop_duplicates()
    wide = wide.reindex(pd.MultiIndex.from_frame(order)).reset_index()
    wide.columns.name = None
    return wide


def psd_wide(long: pd.DataFrame) -> pd.DataFrame:
    """The band-power table with one row per recording and ``segments``.

    A column is called ``<feature>_<band>_<channel or region>``.
    """
    column = long["feature"] + "_" + long["band"] + "_" + long["name"]
    return _pivot(long.assign(column=column))


def table_paths(config: Config, kind: str) -> dict[str, Path]:
    """Where the long and the wide table of ``kind`` (``mfdfa`` or ``psd``) are written."""
    dataset = config.data.dataset
    stem = f"{kind}_features_{dataset.id}-{dataset.version}"
    processed = config.data.paths.processed
    return {"long": processed / f"{stem}.parquet", "wide": processed / f"{stem}_wide.parquet"}


def write_tables(config: Config, kind: str, long: pd.DataFrame, wide: pd.DataFrame) -> dict:
    """Write both tables of ``kind`` as Parquet; returns their paths."""
    paths = table_paths(config, kind)
    paths["long"].parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(paths["long"], index=False)
    wide.to_parquet(paths["wide"], index=False)
    return paths
