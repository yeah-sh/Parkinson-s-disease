import dataclasses
import datetime
import json
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from pdeeg.config import Config, load_config
from pdeeg.features.extract import (
    CLEAN,
    FULL,
    IAAFT,
    N_SCALES,
    ORIGINAL,
    RUN_KEYS,
    RUN_VALUES,
    SHUFFLE,
    bad_sample_mask,
    cache_paths,
    channel_runs,
    config_hash,
    fit_features,
    is_current,
    range_scales,
    recording_runs,
    run,
    series_masks,
    surrogate_seed,
    widen,
)
from pdeeg.features.mfdfa import FEATURE_NAMES, make_qs
from pdeeg.features.psd import build_psd_table
from pdeeg.features.report import write_report
from pdeeg.features.scaling import BROADBAND, analysis_series
from pdeeg.features.synthetic import power_law_noise
from pdeeg.features.tables import (
    CHANNEL,
    MFDFA_COLUMNS,
    REGION,
    TablesError,
    build_mfdfa_table,
    mfdfa_wide,
    psd_wide,
    surrogate_feature,
    table_paths,
    write_tables,
)
from pdeeg.preprocessing.pipeline import output_paths, read_clean

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

SFREQ = 256.0
DURATION = 60.0
N_TIMES = int(DURATION * SFREQ)
CHANNELS = ["Fz", "Cz", "Pz", "Oz"]
REGIONS = {"front": ("Fz", "Cz"), "back": ("Pz", "Oz")}
# Subject, group, and the bad stretch of each test recording in seconds (None: no bad stretch).
RECORDINGS = [("a1", "HC", None), ("b2", "PD", (20.0, 22.0)), ("c3", "PD", (40.0, 41.0))]


def small_config(root: Path | None = None) -> Config:
    """The repository configuration cut down to 60 s recordings and a few quick surrogates."""
    full = load_config(CONFIG_DIR)
    settings = full.mfdfa
    settings = dataclasses.replace(
        settings,
        iaaft_max_iter=20,
        # A tenth of the trimmed envelope is 5.8 s.
        envelope=dataclasses.replace(settings.envelope, fit_ranges={"main": (2.0, 5.5)}),
        surrogates=dataclasses.replace(settings.surrogates, n_iaaft=3, n_shuffle=2),
    )
    config = dataclasses.replace(
        full, mfdfa=settings, data=dataclasses.replace(full.data, regions=REGIONS)
    )
    if root is None:
        return config
    prep = config.preprocessing
    return dataclasses.replace(
        config,
        preprocessing=dataclasses.replace(
            prep, output=dataclasses.replace(prep.output, dir=root / "clean")
        ),
        mfdfa=dataclasses.replace(
            settings,
            extraction=dataclasses.replace(
                settings.extraction,
                cache_dir=root / "mfdfa",
                report=root / "reports" / "qc_mfdfa.md",
                figures_dir=root / "reports" / "figures",
                n_jobs=1,
            ),
        ),
        data=dataclasses.replace(
            config.data, paths=dataclasses.replace(config.data.paths, processed=root / "processed")
        ),
    )


def noise(seed: int, n_channels: int = len(CHANNELS)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([power_law_noise(N_TIMES, 1.0, rng) for _ in range(n_channels)])


# --- bad stretches -------------------------------------------------------------------------------


def annotated_raw(first_samp: int, meas_date) -> mne.io.RawArray:
    info = mne.create_info(["Fz", "Cz"], SFREQ, "eeg")
    raw = mne.io.RawArray(np.zeros((2, 2560)), info, first_samp=first_samp, verbose="error")
    raw.set_meas_date(meas_date)
    # Timed from the first sample, as pdeeg.preprocessing.segments writes them.
    raw.set_annotations(
        mne.Annotations(
            [1.0, 3.0, 5.0, 9.5], [0.5, 1.0, 1.0, 2.0], ["BAD_peak", "1", "bad_flat", "BAD_peak"]
        ),
        emit_warning=False,
    )
    return raw


@pytest.mark.parametrize("first_samp", [0, 3828])
@pytest.mark.parametrize("meas_date", [None, datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)])
def test_bad_sample_mask_marks_the_bad_annotations_wherever_the_recording_starts(
    tmp_path, first_samp, meas_date
):
    raw = annotated_raw(first_samp, meas_date)
    expected = np.zeros(2560, dtype=bool)
    expected[256:384] = True  # 1.0 to 1.5 s
    expected[1280:1536] = True  # 5.0 to 6.0 s; the annotation "1" at 3 s is not a bad one
    expected[2432:] = True  # 9.5 s to the end: the annotation runs past it

    np.testing.assert_array_equal(bad_sample_mask(raw), expected)

    # And the same after a trip through a file, which is how the cleaned recordings arrive.
    path = tmp_path / "sub-x_ses-y_clean.fif"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "_clean.fif" is not one of MNE's suffixes
        raw.save(path, verbose="error")
    np.testing.assert_array_equal(bad_sample_mask(read_clean(path)), expected)


def test_recording_without_annotations_has_no_bad_sample():
    raw = mne.io.RawArray(
        np.zeros((1, 100)), mne.create_info(["Fz"], SFREQ, "eeg"), verbose="error"
    )

    assert not bad_sample_mask(raw).any()


def test_widen_extends_every_bad_stretch_on_both_sides():
    mask = np.zeros(20, dtype=bool)
    mask[[5, 12, 13]] = True

    assert np.flatnonzero(widen(mask, 2)).tolist() == [3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15]
    assert np.flatnonzero(widen(mask, 30)).tolist() == list(range(20))
    np.testing.assert_array_equal(widen(mask, 0), mask)
    assert not widen(np.zeros(5, dtype=bool), 3).any()
    assert widen(mask, 2) is not mask


def test_series_masks_follow_the_series_they_belong_to():
    config = small_config().mfdfa
    bad = np.zeros(N_TIMES, dtype=bool)
    bad[5120:5632] = True  # 20 to 22 s
    series = analysis_series(noise(0, 1), SFREQ, config)

    masks = series_masks(bad, SFREQ, config)

    assert list(masks) == list(series)
    for name, mask in masks.items():
        assert mask.shape == series[name].shape[-1:]
    np.testing.assert_array_equal(masks[BROADBAND], bad)
    # An envelope starts 1 s later, and a bad stretch reaches 1 s further on both sides in it:
    # 19 to 23 s of the recording is 18 to 22 s of the envelope.
    np.testing.assert_array_equal(np.flatnonzero(masks["alpha"])[[0, -1]], [4608, 5631])
    assert masks["theta"] is masks["alpha"]


# --- one fit --------------------------------------------------------------------------------------


def test_fit_features_are_the_features_and_the_number_of_scales():
    config = small_config().mfdfa
    qs = make_qs(config.q_min, config.q_max, config.q_step)
    x = noise(1, 1)[0]
    scales = range_scales(BROADBAND, x.size, SFREQ, config)["long"]

    values = fit_features(x, scales, qs, config)

    assert tuple(values) == RUN_VALUES == (*FEATURE_NAMES, N_SCALES)
    assert values[N_SCALES] == len(scales) == 12
    assert values["h2"] == pytest.approx(1.0, abs=0.15)
    # No bad sample: the same numbers.
    assert fit_features(x, scales, qs, config, exclude=np.zeros(x.size, dtype=bool)) == values


def test_fit_on_too_few_scales_is_not_made():
    config = small_config().mfdfa
    qs = make_qs(config.q_min, config.q_max, config.q_step)
    x = noise(2, 1)[0]
    scales = range_scales(BROADBAND, x.size, SFREQ, config)["long"]  # 77 to 256 samples
    assert (scales[0], scales[-1]) == (77, 256)

    def bad_every(step: int) -> np.ndarray:
        exclude = np.zeros(x.size, dtype=bool)
        exclude[step // 2 :: step] = True
        return exclude

    # Bad samples 200 apart leave no clean window of 200 samples or more: 9 of 12 scales fit.
    shortened = fit_features(x, scales, qs, config, exclude=bad_every(200))
    assert shortened[N_SCALES] == int((scales < 200).sum()) == 9
    assert np.isfinite(shortened["h2"])
    # 120 apart: fewer than the 8 scales a fit needs.
    refused = fit_features(x, scales, qs, config, exclude=bad_every(120))
    assert 0 < refused[N_SCALES] < config.bad_windows.min_scales
    assert all(np.isnan(refused[name]) for name in FEATURE_NAMES)
    # 60 apart: no window of any scale is clean.
    nothing = fit_features(x, scales, qs, config, exclude=bad_every(60))
    assert nothing[N_SCALES] == 0 and np.isnan(nothing["h2"])


# --- surrogates and runs -------------------------------------------------------------------------


def test_surrogate_seed_depends_on_the_names_and_nothing_else():
    def draw(*key: str) -> float:
        return float(np.random.default_rng(surrogate_seed(7, *key)).random())

    assert draw("a1", "s", "Fz", "alpha") == draw("a1", "s", "Fz", "alpha")
    others = [("a1", "s", "Fz", "beta"), ("a1", "s", "Cz", "alpha"), ("a1s", "", "Fz", "alpha")]
    assert len({draw("a1", "s", "Fz", "alpha"), *(draw(*key) for key in others)}) == 4
    first = np.random.default_rng(surrogate_seed(8, "a1", "s", "Fz", "alpha")).random()
    assert first != draw("a1", "s", "Fz", "alpha")


def test_channel_runs_hold_the_original_twice_and_every_surrogate():
    config = small_config().mfdfa
    data = noise(3, 1)
    bad = np.zeros(N_TIMES, dtype=bool)
    bad[5120:5632] = True
    data[0, bad] += 300.0  # an artefact where the recording is marked bad
    series = {name: values[0] for name, values in analysis_series(data, SFREQ, config).items()}
    masks = series_masks(bad, SFREQ, config)

    rows = pd.DataFrame(channel_runs(series, masks, SFREQ, config, ("a1", "s", "Fz")))

    # Two broadband ranges and one for each of three bands; per range the original on all
    # windows and on clean windows, 3 IAAFT surrogates and 2 shuffles.
    assert len(rows) == 5 * (2 + 3 + 2)
    assert set(rows.columns) == set(RUN_KEYS[1:]) | set(RUN_VALUES)
    counts = rows.groupby(["kind", "segments"]).size().to_dict()
    assert counts == {
        (IAAFT, FULL): 15,
        (ORIGINAL, CLEAN): 5,
        (ORIGINAL, FULL): 5,
        (SHUFFLE, FULL): 10,
    }
    assert rows[rows["kind"] == IAAFT].groupby(["series", "fit_range"])["replicate"].apply(
        list
    ).iloc[0] == [0, 1, 2]
    assert rows[list(RUN_VALUES)].notna().all().all()

    def long_range(table: pd.DataFrame) -> pd.DataFrame:
        return table[(table["series"] == BROADBAND) & (table["fit_range"] == "long")]

    original = long_range(rows)[lambda t: t["kind"] == ORIGINAL].set_index("segments")
    # The artefact flattens the fluctuation function of the whole recording (h2 was 0.05); the
    # clean windows keep the exponent of the noise, which is 1.
    assert original.loc[CLEAN, "h2"] == pytest.approx(1.0, abs=0.15)
    assert original.loc[FULL, "h2"] < 0.5

    # Without the artefact: IAAFT keeps the spectrum and with it h2; a shuffle has no memory.
    plain = {
        name: values[0] for name, values in analysis_series(noise(3, 1), SFREQ, config).items()
    }
    rows = long_range(pd.DataFrame(channel_runs(plain, masks, SFREQ, config, ("a1", "s", "Fz"))))
    whole = rows[(rows["kind"] == ORIGINAL) & (rows["segments"] == FULL)]["h2"].iloc[0]
    assert whole == pytest.approx(1.0, abs=0.15)
    assert rows.loc[rows["kind"] == IAAFT, "h2"].mean() == pytest.approx(whole, abs=0.05)
    assert rows.loc[rows["kind"] == SHUFFLE, "h2"].mean() == pytest.approx(0.5, abs=0.1)


def test_recording_runs_do_not_depend_on_order_or_on_the_number_of_processes():
    config = small_config().mfdfa
    data = noise(4, 3)
    bad = np.zeros(N_TIMES, dtype=bool)
    bad[2560:3072] = True
    key = ("a1", "s")

    runs = recording_runs(data, bad, SFREQ, ["Fz", "Cz", "Pz"], config, key)
    again = recording_runs(data[::-1], bad, SFREQ, ["Pz", "Cz", "Fz"], config, key, n_jobs=2)
    alone = recording_runs(data[1:2], bad, SFREQ, ["Cz"], config, key)
    other = recording_runs(data[1:2], bad, SFREQ, ["Cz"], config, ("b2", "s"))

    assert list(runs.columns) == [*RUN_KEYS, *RUN_VALUES]
    assert len(runs) == 3 * 5 * 7

    def ordered(table: pd.DataFrame) -> pd.DataFrame:
        return table.sort_values(list(RUN_KEYS)).reset_index(drop=True)

    pd.testing.assert_frame_equal(ordered(runs), ordered(again))
    pd.testing.assert_frame_equal(ordered(runs[runs["channel"] == "Cz"]), ordered(alone))
    # Another recording draws other surrogates; the original does not change.
    surrogates = alone["kind"] != ORIGINAL
    assert not np.allclose(
        alone.loc[surrogates, "delta_alpha"], other.loc[surrogates, "delta_alpha"]
    )
    pd.testing.assert_frame_equal(alone[~surrogates], other[~surrogates])


# --- files, caching and the tables, end to end ---------------------------------------------------


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """Three cleaned recordings, two with a bad stretch that holds an artefact, analysed once.

    Returns the configuration, the recordings table and the statuses of that first run.
    """
    root = tmp_path_factory.mktemp("features")
    config = small_config(root)
    info = mne.create_info(CHANNELS, SFREQ, "eeg")
    for seed, (subject, _, stretch) in enumerate(RECORDINGS):
        data = noise(10 + seed) * 5e-6
        raw = mne.io.RawArray(data, info, verbose="error")
        if stretch is not None:
            start, stop = (int(second * SFREQ) for second in stretch)
            raw._data[:, start:stop] += 4e-4
            raw.set_annotations(
                mne.Annotations([stretch[0]], [stretch[1] - stretch[0]], ["BAD_peak"])
            )
        paths = output_paths(config, subject, "s")
        paths["raw"].parent.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "_clean.fif" is not one of MNE's suffixes
            raw.save(paths["raw"], verbose="error")
        log = {"config_hash": "cleaned-once", "summary": {"raw_slow_db": 10.0}}
        paths["log"].write_text(json.dumps(log), encoding="utf-8")
    recordings = pd.DataFrame(
        {
            "subject": [subject for subject, _, _ in RECORDINGS],
            "session": "s",
            "group": [group for _, group, _ in RECORDINGS],
        }
    )
    return config, recordings, run(config, recordings)


def test_run_stores_the_runs_of_every_recording(project):
    config, recordings, first = project

    assert [result["status"] for result in first] == ["processed"] * 3, first[0]["detail"]
    for subject, _, stretch in RECORDINGS:
        paths = cache_paths(config, subject, "s")
        assert paths["runs"].name == f"sub-{subject}_ses-s_mfdfa.parquet"
        runs = pd.read_parquet(paths["runs"])
        assert list(runs.columns) == [*RUN_KEYS, *RUN_VALUES]
        assert len(runs) == len(CHANNELS) * 5 * 7
        log = json.loads(paths["log"].read_text(encoding="utf-8"))
        assert log["config_hash"] == config_hash(config, "cleaned-once")
        assert log["channels"] == CHANNELS and log["n_runs"] == len(runs)
        expected = 0.0 if stretch is None else 100.0 * (stretch[1] - stretch[0]) / DURATION
        assert log["bad_percent"] == pytest.approx(expected)
    assert not list(config.mfdfa.extraction.cache_dir.glob("*.pkl"))


def test_stored_runs_are_reused_until_something_that_matters_changes(project):
    config, recordings, _ = project
    row = recordings.iloc[0].to_dict()

    assert [result["status"] for result in run(config, recordings)] == ["cached"] * 3
    assert is_current(row, config)

    settings = config.mfdfa
    # How the runs are tabulated, reported or scheduled does not make them out of date.
    same = dataclasses.replace(
        settings,
        poor_fit_r2=0.5,
        broadband=dataclasses.replace(settings.broadband, wide_features={"short": (), "long": ()}),
        extraction=dataclasses.replace(settings.extraction, n_jobs=7),
    )
    assert is_current(row, dataclasses.replace(config, mfdfa=same))
    for changed in (
        dataclasses.replace(settings, q_step=1.0),
        dataclasses.replace(settings, surrogates=dataclasses.replace(settings.surrogates, seed=1)),
        dataclasses.replace(
            settings, bad_windows=dataclasses.replace(settings.bad_windows, min_windows=4)
        ),
        dataclasses.replace(
            settings, envelope=dataclasses.replace(settings.envelope, edge_trim=1.5)
        ),
    ):
        assert not is_current(row, dataclasses.replace(config, mfdfa=changed))

    # Cleaning the recording again with other settings does.
    log_path = output_paths(config, "a1", "s")["log"]
    original = log_path.read_text(encoding="utf-8")
    log_path.write_text(json.dumps({"config_hash": "cleaned-again"}), encoding="utf-8")
    assert not is_current(row, config)
    log_path.write_text(original, encoding="utf-8")
    assert is_current(row, config)


def test_missing_cleaned_recording_is_reported_not_raised(project):
    config, recordings, _ = project
    absent = pd.DataFrame({"subject": ["zz"], "session": ["s"], "group": ["HC"]})

    result = run(config, absent)[0]

    assert result["status"] == "missing" and "not found" in result["detail"]
    with pytest.raises(TablesError, match="sub-zz ses-s"):
        build_mfdfa_table(config, pd.concat([recordings, absent], ignore_index=True))


@pytest.fixture(scope="module")
def tables(project):
    config, recordings, _ = project
    long = build_mfdfa_table(config, recordings)
    return long, mfdfa_wide(long, config.mfdfa)


def test_long_table_has_channels_and_regions_for_every_recording(project, tables):
    config, recordings, _ = project
    long, _ = tables

    assert list(long.columns) == list(MFDFA_COLUMNS)
    assert long[["subject", "group"]].drop_duplicates().values.tolist() == [
        ["a1", "HC"],
        ["b2", "PD"],
        ["c3", "PD"],
    ]
    assert set(long.loc[long["level"] == CHANNEL, "name"]) == set(CHANNELS)
    assert set(long.loc[long["level"] == REGION, "name"]) == set(REGIONS)
    assert set(zip(long["variant"], long["band"], long["fit_range"], strict=True)) == {
        ("broadband", "broadband", "short"),
        ("broadband", "broadband", "long"),
        ("envelope", "theta", "main"),
        ("envelope", "alpha", "main"),
        ("envelope", "beta", "main"),
    }
    channels = long[long["level"] == CHANNEL]
    # Per channel, series and range: 7 values on all windows, 7 on clean windows, and 5 from
    # each kind of surrogate.
    assert len(channels) == 3 * len(CHANNELS) * 5 * (7 + 7 + 5 + 5)
    assert set(channels.loc[channels["segments"] == CLEAN, "feature"]) == set(RUN_VALUES)

    def h2(subject: str, segments: str) -> np.ndarray:
        rows = long[
            (long["subject"] == subject)
            & (long["level"] == CHANNEL)
            & (long["variant"] == "broadband")
            & (long["fit_range"] == "long")
            & (long["feature"] == "h2")
            & (long["segments"] == segments)
        ]
        return rows["value"].to_numpy()

    # No bad stretch: clean is the whole recording. A bad stretch with an artefact: it is not.
    np.testing.assert_array_equal(h2("a1", CLEAN), h2("a1", FULL))
    assert np.all(np.abs(h2("b2", CLEAN) - h2("b2", FULL)) > 0.01)
    np.testing.assert_allclose(h2("b2", CLEAN), 1.0, atol=0.2)

    # A region is the mean of its channels.
    front = long[
        (long["subject"] == "c3")
        & (long["band"] == "alpha")
        & (long["segments"] == FULL)
        & (long["feature"] == surrogate_feature(IAAFT, "z"))
    ].set_index("name")["value"]
    assert front["front"] == pytest.approx((front["Fz"] + front["Cz"]) / 2)


def test_tables_are_written_as_parquet_and_come_back_unchanged(project, tables):
    config, _, _ = project
    long, wide = tables

    paths = write_tables(config, "mfdfa", long, wide)

    assert paths == table_paths(config, "mfdfa")
    assert paths["long"].name == "mfdfa_features_ds002778-1.0.5.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(paths["long"]), long)
    pd.testing.assert_frame_equal(pd.read_parquet(paths["wide"]), wide)
    assert wide.shape[0] == 6 and list(wide.columns[:4]) == [
        "subject",
        "session",
        "group",
        "segments",
    ]
    # h2 from the short range and five features from each of the other four, at six sites.
    assert wide.shape[1] == 4 + (1 + 4 * 5) * 6


def test_report_describes_the_features_without_comparing_groups(project, tables):
    config, recordings, _ = project
    long, _ = tables
    psd = build_psd_table(config, recordings)
    write_tables(config, "psd", psd, psd_wide(psd))

    report = write_report(config, recordings, long, psd)

    extraction = config.mfdfa.extraction
    assert report == extraction.report
    text = report.read_text(encoding="utf-8")
    for figure in ("surrogate_z.png", "full_against_clean.png"):
        assert (extraction.figures_dir / figure).is_file()
        assert f"figures/{figure}" in text
    for heading in (
        "## Fit quality",
        "## Recordings without slow power",
        "## Surrogates",
        "## Whole recording against clean windows",
        "## Band power",
    ):
        assert heading in text
    assert "3 recordings, 4 channels each" in text
    assert "2 of the 3 recordings have a bad stretch" in text
    assert "Nothing below compares groups" in text
    # No group label, and no statistic by group, anywhere.
    assert "HC" not in text and "PD" not in text
    assert len(psd) == 3 * (len(CHANNELS) + len(REGIONS)) * 5 * 2
