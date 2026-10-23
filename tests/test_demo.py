"""The research demo: reads one file, describes it, and never predicts anything."""

import dataclasses
import shutil
from pathlib import Path

import mne
import numpy as np
import pytest

from pdeeg import demo
from pdeeg.config import CONFIG_DIR_ENV, load_config
from test_preprocessing import DURATION, LINE_FREQ, MONTAGE, SFREQ, TRIGGER_AT, synthetic_raw

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
CHANNELS = mne.channels.make_standard_montage(MONTAGE).ch_names


def short_config():
    """The repository configuration for a 40 s recording: a shorter crop and envelope range."""
    config = load_config(ROOT / "configs")
    prep = config.preprocessing
    mfdfa = config.mfdfa
    return dataclasses.replace(
        config,
        preprocessing=dataclasses.replace(
            prep, crop=dataclasses.replace(prep.crop, duration=DURATION)
        ),
        mfdfa=dataclasses.replace(
            mfdfa,
            envelope=dataclasses.replace(
                mfdfa.envelope, bands={"alpha": (8.0, 13.0)}, fit_ranges={"main": (1.0, 3.5)}
            ),
        ),
    )


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> Path:
    """The synthetic recording as a BDF file, with an auxiliary electrode as in the dataset."""
    raw = synthetic_raw()
    extra = mne.io.RawArray(
        np.zeros((1, raw.n_times)), mne.create_info(["EXG1"], SFREQ, "eeg"), verbose="error"
    )
    raw.add_channels([extra], force_update_info=True)
    path = tmp_path_factory.mktemp("demo") / "recording.bdf"
    mne.export.export_raw(path, raw, fmt="bdf", verbose="error")
    return path


@pytest.fixture(scope="module")
def analysed(recording):
    config = short_config()
    raw = demo.read_eeg(recording, MONTAGE)
    with mne.use_log_level("ERROR"):
        cleaned, log = demo.clean(raw, config, LINE_FREQ)
    return config, demo.series(cleaned, config), log


def test_the_file_is_read_as_the_montage_channels_with_its_trigger(recording, tmp_path):
    copy = tmp_path / "copy.bdf"
    copy.write_bytes(recording.read_bytes())
    raw = demo.read_eeg(copy, MONTAGE)
    copy.unlink()  # as an upload's temporary file is removed

    raw.copy().crop(1.0, 2.0)  # needs no file

    assert raw.ch_names == CHANNELS  # EXG1 has no position and is dropped
    assert raw.get_montage() is not None
    descriptions = list(raw.annotations.description)
    assert "1" in descriptions
    assert raw.annotations.onset[descriptions.index("1")] == pytest.approx(TRIGGER_AT, abs=0.01)


def test_files_the_demo_cannot_use_are_refused_with_a_reason(recording, tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("x", encoding="utf-8")
    with pytest.raises(demo.DemoError, match=r"expected a \.bdf or \.edf"):
        demo.read_eeg(text, MONTAGE)

    raw = mne.io.read_raw_bdf(recording, preload=True, verbose="error")
    raw.rename_channels({"Oz": "E99"})
    renamed = tmp_path / "renamed.bdf"
    mne.export.export_raw(renamed, raw, fmt="bdf", verbose="error")
    with pytest.raises(demo.DemoError, match="no channel named Oz"):
        demo.read_eeg(renamed, MONTAGE)

    config = load_config(ROOT / "configs")  # 180 s crop: the 44 s recording is too short
    with pytest.raises(demo.DemoError, match="too short"):
        demo.clean(demo.read_eeg(recording, MONTAGE), config, LINE_FREQ)


def test_the_cleaned_recording_is_described_as_the_pipeline_would(analysed):
    config, data, log = analysed

    assert log["crop"]["anchor"] == "event 1"
    assert data.channels == CHANNELS
    assert set(data.values) == {"broadband", "alpha"}
    scales, qs, fq = demo.fluctuation_functions(data, "broadband", "Cz", config)
    assert fq.shape == (len(qs), len(scales)) and np.isfinite(fq).all()
    assert (np.diff(np.log(fq[list(qs).index(2.0)])) > 0).mean() > 0.9  # Fq grows with s

    curves = demo.spectrum(data, "broadband", "long", "Cz", config)
    table = demo.channel_features(data, "broadband", "long", config)
    assert list(table["channel"]) == CHANNELS
    assert table["h2"].notna().all()
    h2 = float(curves.loc[curves["q"] == 2.0, "h"].iloc[0])
    assert h2 == pytest.approx(table.set_index("channel").loc["Cz", "h2"])


def test_the_app_starts_with_the_disclaimer():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=120).run()

    assert not app.exception
    assert "Research demo — not a diagnostic tool." in app.warning[0].value
    assert "Parkinson's disease" in app.warning[0].value


def short_config_dir(root: Path) -> Path:
    """The repository's configs, edited as :func:`short_config` edits them in memory."""
    configs = root / "configs"
    shutil.copytree(ROOT / "configs", configs)
    for name, old, new in (
        ("preprocessing.yaml", "  duration: 180.0", f"  duration: {DURATION}"),
        ("features/mfdfa.yaml", "    main: [2.0, 17.8]", "    main: [1.0, 3.5]"),
    ):
        path = configs / name
        text = path.read_text(encoding="utf-8")
        assert text.count(old) == 1
        path.write_text(text.replace(old, new), encoding="utf-8")
    return configs


def test_the_app_describes_an_uploaded_recording(recording, tmp_path, monkeypatch):
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv(CONFIG_DIR_ENV, str(short_config_dir(tmp_path)))
    streamlit.cache_data.clear()
    streamlit.cache_resource.clear()  # the configuration of an earlier test
    app = AppTest.from_file(str(APP), default_timeout=900).run()
    upload = ("recording.bdf", recording.read_bytes(), "application/octet-stream")
    app.file_uploader[0].set_value(upload).run()

    assert not app.exception and not app.error
    assert [header.value for header in app.subheader] == [
        "Cleaning",
        "Fluctuation functions",
        "Singularity spectrum",
        "Over the scalp",
    ]
    assert len(app.dataframe) == 1
    assert "not a diagnostic tool" in app.caption[-1].value
    streamlit.cache_data.clear()
    streamlit.cache_resource.clear()


def test_the_demo_has_no_classifier_and_predicts_nothing():
    for path in (
        APP,
        ROOT / "src" / "pdeeg" / "demo.py",
        ROOT / "src" / "pdeeg" / "viz" / "demo.py",
    ):
        text = path.read_text(encoding="utf-8")
        assert "pdeeg.models" not in text and "sklearn" not in text, path
        assert "predict" not in text.lower(), path
