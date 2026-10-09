from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdeeg.config import load_config
from pdeeg.features.extract import CLEAN, FULL, IAAFT, ORIGINAL, RUN_KEYS, RUN_VALUES, SHUFFLE
from pdeeg.features.tables import (
    CHANNEL,
    REGION,
    TablesError,
    add_regions,
    mfdfa_wide,
    psd_wide,
    recording_features,
    summarise_runs,
    surrogate_feature,
    table_paths,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
REGIONS = {"front": ["Fz", "Cz"], "back": ["Pz"]}


def run_row(channel, series, fit_range, kind, segments, replicate, delta_alpha, h2=0.8):
    values = dict.fromkeys(RUN_VALUES, 0.0) | {"delta_alpha": delta_alpha, "h2": h2, "n_scales": 12}
    keys = dict(zip(RUN_KEYS, (channel, series, fit_range, kind, segments, replicate), strict=True))
    return keys | values


def make_runs(surrogate_widths=(0.1, 0.2, 0.3, 0.4), observed=0.5) -> pd.DataFrame:
    """One channel and series: an original (full and clean) and four surrogates of each kind."""
    rows = [
        run_row("Fz", "alpha", "main", ORIGINAL, FULL, 0, observed, h2=0.9),
        run_row("Fz", "alpha", "main", ORIGINAL, CLEAN, 0, observed - 0.1, h2=0.85),
    ]
    for i, width in enumerate(surrogate_widths):
        rows.append(run_row("Fz", "alpha", "main", IAAFT, FULL, i, width, h2=0.9 + 0.01 * i))
        rows.append(run_row("Fz", "alpha", "main", SHUFFLE, FULL, i, width / 10, h2=0.5))
    return pd.DataFrame(rows)


def value(table: pd.DataFrame, feature: str, segments: str = FULL) -> float:
    rows = table[(table["feature"] == feature) & (table["segments"] == segments)]
    assert len(rows) == 1
    return float(rows["value"].iloc[0])


def test_summary_keeps_the_original_and_adds_what_the_surrogates_say():
    summary = summarise_runs(make_runs())

    assert list(summary.columns) == [
        "channel",
        "series",
        "fit_range",
        "segments",
        "feature",
        "value",
    ]
    assert value(summary, "delta_alpha") == 0.5
    assert value(summary, "delta_alpha", CLEAN) == pytest.approx(0.4)
    assert value(summary, "h2", CLEAN) == 0.85
    assert value(summary, "n_scales") == 12
    widths = np.array([0.1, 0.2, 0.3, 0.4])
    assert value(summary, surrogate_feature(IAAFT, "mean")) == pytest.approx(0.25)
    assert value(summary, surrogate_feature(IAAFT, "sd")) == pytest.approx(widths.std(ddof=1))
    assert value(summary, surrogate_feature(IAAFT, "z")) == pytest.approx(
        (0.5 - 0.25) / widths.std(ddof=1)
    )
    # Wider than all four surrogates: one in five, the original counted in.
    assert value(summary, surrogate_feature(IAAFT, "p")) == pytest.approx(1 / 5)
    assert value(summary, surrogate_feature(IAAFT, "mean", "h2")) == pytest.approx(0.915)
    assert value(summary, surrogate_feature(SHUFFLE, "mean")) == pytest.approx(0.025)
    assert value(summary, surrogate_feature(SHUFFLE, "mean", "h2")) == 0.5
    # What the surrogates say belongs to the whole recording, which they were made from.
    assert summary.loc[summary["feature"].str.contains("iaaft|shuffle"), "segments"].eq(FULL).all()


@pytest.mark.parametrize(
    ("observed", "p"), [(0.05, 1.0), (0.25, 3 / 5), (0.3, 3 / 5), (0.45, 1 / 5)]
)
def test_rank_probability_counts_surrogates_at_least_as_wide(observed, p):
    summary = summarise_runs(make_runs(observed=observed))

    assert value(summary, surrogate_feature(IAAFT, "p")) == pytest.approx(p)


def test_summary_without_surrogates_and_with_a_fit_that_was_not_made():
    runs = make_runs()
    only_original = summarise_runs(runs[runs["kind"] == ORIGINAL])
    assert not only_original["feature"].str.contains("iaaft|shuffle").any()

    runs.loc[(runs["kind"] == ORIGINAL) & (runs["segments"] == FULL), "delta_alpha"] = np.nan
    summary = summarise_runs(runs)
    assert np.isnan(value(summary, surrogate_feature(IAAFT, "z")))
    assert np.isnan(value(summary, surrogate_feature(IAAFT, "p")))
    assert value(summary, surrogate_feature(IAAFT, "mean")) == pytest.approx(0.25)


def channel_table() -> pd.DataFrame:
    rows = []
    for channel, h2 in (("Fz", 1.0), ("Cz", 3.0), ("Pz", 5.0)):
        for feature, shift in (("h2", 0.0), ("delta_alpha_iaaft_z", 10.0)):
            rows.append({"channel": channel, "feature": feature, "value": h2 + shift})
        for feature in ("delta_alpha_iaaft_p", "delta_alpha_iaaft_sd", "n_scales"):
            rows.append({"channel": channel, "feature": feature, "value": 0.5})
    return pd.DataFrame(rows).assign(segments=FULL)


def test_regions_are_means_of_their_channels():
    table = add_regions(channel_table(), REGIONS, ["segments", "feature"])

    assert list(table.columns) == ["level", "name", "segments", "feature", "value"]
    channels = table[table["level"] == CHANNEL]
    assert len(channels) == 15 and set(channels["name"]) == {"Fz", "Cz", "Pz"}
    regions = table[table["level"] == REGION].set_index(["name", "feature"])["value"]
    assert regions["front", "h2"] == 2.0
    assert regions["back", "h2"] == 5.0
    assert regions["front", "delta_alpha_iaaft_z"] == 12.0
    # A mean of probabilities, of spreads or of scale counts would mean nothing.
    assert set(regions.index.get_level_values("feature")) == {"h2", "delta_alpha_iaaft_z"}


def test_region_mean_skips_missing_values_and_rejects_unknown_channels():
    channels = channel_table()
    channels.loc[(channels["channel"] == "Fz") & (channels["feature"] == "h2"), "value"] = np.nan
    channels.loc[(channels["channel"] == "Pz") & (channels["feature"] == "h2"), "value"] = np.nan

    table = add_regions(channels, REGIONS, ["segments", "feature"])

    regions = table[table["level"] == REGION].set_index(["name", "feature"])["value"]
    assert regions["front", "h2"] == 3.0
    assert np.isnan(regions["back", "h2"])
    with pytest.raises(TablesError, match="Oz"):
        add_regions(channels, {"back": ["Pz", "Oz"]}, ["segments", "feature"])


def recording_runs_table() -> pd.DataFrame:
    rows = []
    for channel, offset in (("Fz", 0.0), ("Cz", 0.2), ("Pz", 0.4)):
        for series, fit_range in (("broadband", "short"), ("broadband", "long"), ("alpha", "main")):
            for segments in (FULL, CLEAN):
                rows.append(
                    run_row(channel, series, fit_range, ORIGINAL, segments, 0, 0.3 + offset, h2=1.0)
                )
            for i in range(3):
                rows.append(run_row(channel, series, fit_range, IAAFT, FULL, i, 0.1 * i, h2=1.0))
    return pd.DataFrame(rows)


def long_table() -> pd.DataFrame:
    features = recording_features(recording_runs_table(), REGIONS)
    tables = []
    for subject, group in (("a1", "HC"), ("b2", "PD")):
        tables.append(features.assign(subject=subject, session="s", group=group))
    return pd.concat(tables, ignore_index=True)


def test_recording_features_name_the_variant_and_the_band():
    features = recording_features(recording_runs_table(), REGIONS)

    assert list(features.columns) == [
        "level",
        "name",
        "variant",
        "band",
        "fit_range",
        "segments",
        "feature",
        "value",
    ]
    pairs = set(zip(features["variant"], features["band"], features["fit_range"], strict=True))
    assert pairs == {
        ("broadband", "broadband", "short"),
        ("broadband", "broadband", "long"),
        ("envelope", "alpha", "main"),
    }
    front = features[
        (features["name"] == "front")
        & (features["feature"] == "delta_alpha")
        & (features["band"] == "alpha")
        & (features["segments"] == FULL)
    ]
    assert front["value"].tolist() == pytest.approx([0.4])


def test_wide_table_has_one_row_per_recording_and_segments_and_only_the_chosen_features():
    config = load_config(CONFIG_DIR).mfdfa
    long = long_table()

    wide = mfdfa_wide(long, config)

    assert list(wide.columns[:4]) == ["subject", "session", "group", "segments"]
    assert wide[["subject", "segments"]].values.tolist() == [
        ["a1", FULL],
        ["a1", CLEAN],
        ["b2", FULL],
        ["b2", CLEAN],
    ]
    columns = set(wide.columns[4:])
    # Only h2 from the short range; channels and regions alike.
    assert {c for c in columns if c.startswith("broadband_short_")} == {
        f"broadband_short_h2_{name}" for name in ("Fz", "Cz", "Pz", "front", "back")
    }
    assert "broadband_long_delta_alpha_Cz" in columns
    assert "envelope_alpha_main_asymmetry_back" in columns
    assert not any("iaaft" in c or "min_r2" in c or "n_scales" in c for c in columns)
    assert len(columns) == 5 * (1 + 5 + 5)
    row = wide[(wide["subject"] == "b2") & (wide["segments"] == FULL)].iloc[0]
    assert row["envelope_alpha_main_delta_alpha_Pz"] == pytest.approx(0.7)
    assert row["envelope_alpha_main_delta_alpha_front"] == pytest.approx(0.4)
    assert wide.notna().all().all()


def test_wide_table_rejects_a_feature_that_is_not_in_the_long_table():
    config = load_config(CONFIG_DIR).mfdfa
    long = long_table()

    with pytest.raises(TablesError, match="delta_h"):
        mfdfa_wide(long[long["feature"] != "delta_h"], config)


def test_psd_wide_and_table_paths():
    rows = []
    for subject in ("a1", "b2"):
        for segments in (FULL, CLEAN):
            for band in ("delta", "alpha"):
                for name in ("Fz", "front"):
                    rows.append(
                        {
                            "subject": subject,
                            "session": "s",
                            "group": "HC",
                            "level": CHANNEL if name == "Fz" else REGION,
                            "name": name,
                            "band": band,
                            "segments": segments,
                            "feature": "log_relative_power",
                            "value": -1.0,
                        }
                    )
    wide = psd_wide(pd.DataFrame(rows))

    assert wide.shape == (4, 4 + 4)
    assert "log_relative_power_alpha_front" in wide.columns

    config = load_config(CONFIG_DIR)
    paths = table_paths(config, "mfdfa")
    assert paths["long"].name == "mfdfa_features_ds002778-1.0.5.parquet"
    assert paths["wide"].name == "mfdfa_features_ds002778-1.0.5_wide.parquet"
    assert paths["long"].parent == config.data.paths.processed
    assert table_paths(config, "psd")["long"].name == "psd_features_ds002778-1.0.5.parquet"
