import dataclasses
import json
import os
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
from mne_bids import BIDSPath, write_raw_bids

from pdeeg.config import BadSegmentsConfig, Config, SessionConfig, load_config
from pdeeg.data.bids import list_recordings
from pdeeg.preprocessing.pipeline import (
    RecordingTooShort,
    crop_window,
    detect_line_noise,
    is_current,
    median_psd,
    output_paths,
    preprocess_raw,
    read_clean,
    run,
)
from pdeeg.preprocessing.report import write_report
from pdeeg.preprocessing.segments import FLAT, PEAK, annotate_bad_segments

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

MONTAGE = "biosemi32"
SFREQ = 256.0
LINE_FREQ = 60.0
TRIGGER_AT = 2.0  # seconds; where the synthetic recording carries trigger "1"
DURATION = 40.0  # seconds kept by the test configuration
BLINK_UV = 150.0


def synthetic_raw(seconds: float = TRIGGER_AT + DURATION + 2.0, seed: int = 0) -> mne.io.RawArray:
    """A 32-channel recording with pink-noise sources, alpha, DC offsets and frontal blinks."""
    rng = np.random.default_rng(seed)
    montage = mne.channels.make_standard_montage(MONTAGE)
    positions = np.array([montage.get_positions()["ch_pos"][name] for name in montage.ch_names])
    directions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    n_times = int(seconds * SFREQ)
    times = np.arange(n_times) / SFREQ

    def pink() -> np.ndarray:
        spectrum = np.fft.rfft(rng.normal(size=n_times))
        freqs = np.fft.rfftfreq(n_times, 1 / SFREQ)
        freqs[0] = freqs[1]
        series = np.fft.irfft(spectrum / np.sqrt(freqs), n_times)
        return series / series.std()

    # "Brain" sources: smooth patches on the upper half of the head, four of them with alpha.
    n_sources = 14
    centres = rng.normal(size=(n_sources, 3))
    centres[:, 2] = np.abs(centres[:, 2])
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    patches = np.exp((directions @ centres.T - 1.0) / 0.15)
    sources = np.array([pink() for _ in range(n_sources)])
    for i in range(4):
        sources[i] += 1.5 * np.sin(2 * np.pi * (9.5 + 0.4 * i) * times + rng.uniform(0, 6.28))
    data = patches @ sources * 4e-6

    # Blinks: a bump every few seconds, largest at the frontal pole and fading backwards.
    front = np.array([0.0, 1.0, 0.0])
    blink_map = np.exp((directions @ front - 1.0) / 0.12)
    blink_map /= blink_map.max()
    blinks = np.zeros(n_times)
    onsets = np.arange(1.5, seconds - 1.0, 3.1)
    for onset in onsets + rng.uniform(-0.4, 0.4, size=onsets.size):
        blinks += np.exp(-0.5 * ((times - onset) / 0.07) ** 2)
    data += np.outer(blink_map, blinks) * BLINK_UV * 1e-6

    data += rng.normal(scale=0.3e-6, size=data.shape)
    data += rng.normal(scale=3e-3, size=(len(directions), 1))  # electrode offsets, as in BioSemi
    raw = mne.io.RawArray(data, mne.create_info(montage.ch_names, SFREQ, "eeg"), verbose="error")
    raw.set_montage(montage)
    raw.info["line_freq"] = LINE_FREQ
    raw.set_annotations(mne.Annotations([TRIGGER_AT], [0.0], ["1"]))
    return raw


@pytest.fixture(scope="module")
def config() -> Config:
    """The repository configuration, shortened so that a test recording is 40 s long."""
    full = load_config(CONFIG_DIR)
    prep = full.preprocessing
    prep = dataclasses.replace(prep, crop=dataclasses.replace(prep.crop, duration=DURATION))
    return dataclasses.replace(full, preprocessing=prep)


@pytest.fixture(scope="module")
def raw() -> mne.io.RawArray:
    return synthetic_raw()


@pytest.fixture(scope="module")
def cleaned(raw: mne.io.RawArray, config: Config):
    with mne.use_log_level("ERROR"):
        return preprocess_raw(raw, config.preprocessing, LINE_FREQ)


def test_preprocess_raw_returns_one_continuous_recording_of_the_configured_length(
    raw: mne.io.RawArray, cleaned, config: Config
):
    clean, ica, log = cleaned

    assert clean.ch_names == raw.ch_names
    assert clean.n_times == int(DURATION * SFREQ)
    assert clean.get_data().shape == (32, int(DURATION * SFREQ))
    assert clean.info["sfreq"] == SFREQ
    assert (clean.info["highpass"], clean.info["lowpass"]) == (0.5, 50.0)
    assert log["crop"] == {
        "start_s": TRIGGER_AT,
        "duration_s": DURATION,
        "anchor": "event 1",
        "available_s": raw.n_times / SFREQ - TRIGGER_AT,
    }
    # An average reference leaves 31 independent channels; ICA is given exactly that many.
    assert log["ica"]["rank"] == 31
    assert log["ica"]["n_components"] == ica.n_components_ == 31
    assert np.abs(clean.get_data().mean(axis=0)).max() < 1e-12
    # The input is left as it was.
    assert raw.n_times == int((TRIGGER_AT + DURATION + 2.0) * SFREQ)
    assert list(raw.annotations.description) == ["1"]


def test_preprocess_raw_removes_the_blink_component(raw: mne.io.RawArray, cleaned, config: Config):
    clean, _, log = cleaned

    removed = log["ica"]["removed"]
    assert "eye blink" in [component["label"] for component in removed]
    assert all(
        component["probability"] >= config.preprocessing.ica.threshold for component in removed
    )
    assert all(
        component["label"] in config.preprocessing.ica.exclude_labels for component in removed
    )
    # Same filtering and reference without ICA: the blinks are still there.
    plain = raw.copy().crop(TRIGGER_AT, TRIGGER_AT + DURATION, include_tmax=False)
    plain.filter(0.5, 50.0, h_trans_bandwidth=5.0, verbose="error").set_eeg_reference(
        "average", verbose="error"
    )
    frontal = plain.ch_names.index("Fp1")
    assert np.ptp(plain.get_data()[frontal]) > 0.5 * BLINK_UV * 1e-6
    assert np.ptp(clean.get_data()[frontal]) < 0.4 * np.ptp(plain.get_data()[frontal])
    assert log["ica"]["variance_removed_percent"] > 10


def test_log_is_json_and_describes_every_channel(cleaned):
    clean, _, log = cleaned

    restored = json.loads(json.dumps(log))

    assert set(restored["channels"]) == set(clean.ch_names)
    assert set(restored["channels"]["Cz"]) == {"sd_uv", "peak_percent", "flat_percent"}
    assert 0.0 <= restored["bad_segments"]["percent"] <= 100.0
    assert restored["line_noise"]["configured_hz"] == LINE_FREQ
    assert len(restored["psd"]["freqs_hz"]) == len(restored["psd"]["after_db"])


def test_same_seed_gives_the_same_output(raw: mne.io.RawArray, cleaned, config: Config):
    clean, _, log = cleaned

    with mne.use_log_level("ERROR"):
        again, _, log_again = preprocess_raw(raw, config.preprocessing, LINE_FREQ)

    np.testing.assert_allclose(again.get_data(), clean.get_data(), rtol=1e-9, atol=1e-15)
    assert log_again == log


def test_a_recording_shorter_than_the_crop_is_rejected(config: Config):
    short = synthetic_raw(seconds=TRIGGER_AT + DURATION - 1.0)

    with pytest.raises(RecordingTooShort, match="39.0 s available from event 1, 40.0 s needed"):
        preprocess_raw(short, config.preprocessing, LINE_FREQ)


def test_crop_starts_at_the_trigger_or_at_the_file_start(raw: mne.io.RawArray, config: Config):
    crop = config.preprocessing.crop

    assert crop_window(raw, crop) == (int(TRIGGER_AT * SFREQ), "event 1")
    assert crop_window(raw.copy().set_annotations(None), crop) == (0, "file start")
    assert crop_window(raw, dataclasses.replace(crop, start_event=None)) == (0, "file start")


def test_mains_frequency_is_read_from_the_spectrum(raw: mne.io.RawArray, config: Config):
    line_noise = config.preprocessing.line_noise
    noisy = raw.copy()
    hum = 20e-6 * np.sin(2 * np.pi * LINE_FREQ * noisy.times)
    noisy.apply_function(lambda channel: channel + hum, verbose="error")

    detected, ratios = detect_line_noise(*median_psd(noisy), line_noise)
    absent, _ = detect_line_noise(*median_psd(raw), line_noise)

    assert detected == LINE_FREQ
    assert ratios["60"] > line_noise.min_peak_ratio > ratios["50"]
    assert absent is None


def test_bad_segments_are_annotated_where_they_are_and_nothing_is_removed():
    rng = np.random.default_rng(1)
    sfreq = 100.0
    data = rng.normal(scale=5e-6, size=(3, 1200))
    data[0, 520:560] += 400e-6  # a burst in second 5 of the uncropped recording
    data[2, 800:1000] = 0.0  # a dead stretch in seconds 8 and 9
    raw = mne.io.RawArray(data, mne.create_info(3, sfreq, "eeg"), verbose="error")
    raw.set_meas_date(1_600_000_000)
    raw.crop(2.0, None)  # a cropped recording does not start at sample 0
    config = BadSegmentsConfig(window=1.0, peak_to_peak_uv=150.0, flat_uv=1.0)

    annotations, summary = annotate_bad_segments(raw, config)
    raw.set_annotations(annotations)

    assert raw.n_times == 1000
    assert list(annotations.description) == [PEAK, FLAT]
    assert list(annotations.onset) == [3.0, 6.0]
    assert list(annotations.duration) == [1.0, 2.0]
    assert summary["percent"] == 30.0
    assert (summary["peak_percent"], summary["flat_percent"]) == (10.0, 20.0)
    assert summary["channel_peak_percent"] == [10.0, 0.0, 0.0]
    assert summary["channel_flat_percent"] == [0.0, 0.0, 20.0]
    # The annotations cover exactly those samples of the cropped recording.
    marked = np.isnan(raw.get_data(reject_by_annotation="NaN")[0])
    expected = np.zeros(1000, dtype=bool)
    expected[300:400] = expected[600:800] = True
    np.testing.assert_array_equal(marked, expected)


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory, config: Config, raw: mne.io.RawArray):
    """A one-recording BIDS dataset, preprocessed once into a temporary directory.

    Returns the configuration, the recordings table and the statuses of that first run.
    """
    root = tmp_path_factory.mktemp("project")
    bids_root = root / "raw"
    write_raw_bids(
        raw.copy(),
        BIDSPath(subject="x1", session="hc", task="rest", datatype="eeg", root=bids_root),
        format="BDF",
        allow_preload=True,
        verbose="error",
    )
    pd.DataFrame({"participant_id": ["sub-x1"], "age": [60]}).to_csv(
        bids_root / "participants.tsv", sep="\t", index=False
    )
    prep = config.preprocessing
    prep = dataclasses.replace(
        prep,
        # These tests are about files and caching, not about a good decomposition.
        ica=dataclasses.replace(prep.ica, max_iter=20),
        output=dataclasses.replace(prep.output, dir=root / "clean"),
        qc=dataclasses.replace(
            prep.qc, report=root / "reports" / "qc.md", figures_dir=root / "reports" / "figures"
        ),
    )
    data = dataclasses.replace(
        config.data, paths=dataclasses.replace(config.data.paths, raw=bids_root)
    )
    sessions = {"hc": SessionConfig(group="HC", condition="HC")}
    recordings = list_recordings(bids_root, sessions)
    config = dataclasses.replace(config, data=data, preprocessing=prep)
    return config, recordings, run(config, recordings, n_jobs=1)


def test_run_writes_the_recording_its_log_and_its_figures(project):
    config, recordings, first = project
    paths = output_paths(config, "x1", "hc")

    assert [result["status"] for result in first] == ["processed"], first[0]["detail"]
    assert paths["raw"].name == "sub-x1_ses-hc_clean.fif"
    assert paths["log"].name == "sub-x1_ses-hc_clean.json"
    clean = read_clean(paths["raw"])
    assert len(clean.ch_names) == 32
    assert clean.n_times == int(DURATION * SFREQ)
    log = json.loads(paths["log"].read_text(encoding="utf-8"))
    assert log["source"] == recordings.bids_path[0]
    assert log["crop"]["anchor"] == "event 1"
    assert {"label", "probability", "probabilities", "index"} <= set(log["ica"]["removed"][0])
    assert "percent" in log["bad_segments"]
    assert set(log["channels"]) == set(clean.ch_names)
    assert paths["psd"].is_file()
    assert paths["ica"].is_file()


def test_outputs_are_reused_until_the_settings_or_the_input_change(project):
    config, recordings, _ = project
    recording = recordings.iloc[0].to_dict()
    data_file = config.data.paths.raw / recording["bids_path"]
    prep = config.preprocessing

    assert is_current(recording, config)
    assert [result["status"] for result in run(config, recordings, n_jobs=1)] == ["cached"]

    # A setting that changes the result.
    stricter = dataclasses.replace(prep, ica=dataclasses.replace(prep.ica, threshold=0.9))
    assert not is_current(recording, dataclasses.replace(config, preprocessing=stricter))
    # A setting that does not.
    faster = dataclasses.replace(prep, n_jobs=prep.n_jobs + 1)
    assert is_current(recording, dataclasses.replace(config, preprocessing=faster))

    # An input newer than the outputs.
    before = data_file.stat()
    later = time.time() + 60
    os.utime(data_file, (later, later))
    try:
        assert not is_current(recording, config)
    finally:
        os.utime(data_file, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert is_current(recording, config)


def test_report_lists_the_recording_and_its_figures(project):
    config, recordings, _ = project

    report = write_report(config, recordings)

    text = report.read_text(encoding="utf-8")
    assert report == config.preprocessing.qc.report
    assert "sub-x1 hc" in text
    assert "figures/overview.png" in text
    assert "figures/sub-x1_ses-hc_psd.png" in text
    assert (config.preprocessing.qc.figures_dir / "overview.png").is_file()
    assert (config.preprocessing.qc.figures_dir / "psd_overview.png").is_file()
