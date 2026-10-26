"""The research demo's description of a recording, and the app without trained models.

The app's predictions are tested in test_predict.py, with the final models trained there.
"""

import dataclasses
import shutil
from pathlib import Path

import mne
import numpy as np
import pytest

from pdeeg import demo
from pdeeg.config import CONFIG_DIR_ENV, load_config
from pdeeg.predict.features import clean_input, work_files
from pdeeg.predict.inputs import read_recording
from test_preprocessing import DURATION, MONTAGE, SFREQ, synthetic_raw

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
def analysed(recording, tmp_path_factory):
    """The recording read, checked and cleaned as `pdeeg predict` does, then described."""
    config = short_config()
    folder = tmp_path_factory.mktemp("clean")
    raw = read_recording(recording, config).raw
    with mne.use_log_level("ERROR"):
        log = clean_input(raw, config, folder)
    cleaned = mne.io.read_raw_fif(work_files(config, folder)["raw"], preload=True, verbose="error")
    return config, demo.series(cleaned, config), log


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
    banner = app.warning[0].value
    assert banner.startswith("**Research demo – not a diagnostic tool.**")
    assert "(OpenNeuro ds002778)" in banner and "not validated on other recordings" in banner
    assert app.sidebar.warning[0].value == banner


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

    assert not app.exception
    assert "No models have been trained here yet" in app.warning[0].value
    assert len(app.error) == 1 and "run `pdeeg train-final` first" in app.error[0].value
    assert [header.value for header in app.header] == ["The recording"]
    assert [header.value for header in app.subheader] == [
        "Input checks",
        "Cleaning",
        "Fluctuation functions",
        "Singularity spectrum",
        "Over the scalp",
    ]
    assert len(app.dataframe) == 1
    assert "not a diagnostic tool" in app.caption[-1].value
    streamlit.cache_data.clear()
    streamlit.cache_resource.clear()


def test_the_app_predicts_only_through_pdeeg_predict():
    """No model code of the app's own: its predictions are those of `pdeeg predict`."""
    for path in (
        APP,
        ROOT / "src" / "pdeeg" / "demo.py",
        ROOT / "src" / "pdeeg" / "viz" / "demo.py",
        ROOT / "src" / "pdeeg" / "viz" / "predict.py",
    ):
        text = path.read_text(encoding="utf-8")
        for code in ("pdeeg.models", "sklearn", "joblib", "predict_proba", "load_stage"):
            assert code not in text, (path, code)
    assert "predict_files(" in APP.read_text(encoding="utf-8")
