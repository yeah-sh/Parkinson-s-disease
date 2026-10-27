"""The research demo's description of a recording, and the research app without trained models.

The app's predictions are tested in test_predict.py, with the final models trained there.
"""

import dataclasses
import shutil
import sys
import time
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
APP = ROOT / "app" / "research_app.py"
PAGES = ("overview", "recording", "quality", "multifractal", "topography", "evaluation",
         "report", "about")  # fmt: skip
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
    return config, demo.series(cleaned, config), log, raw, cleaned


def test_the_cleaned_recording_is_described_as_the_pipeline_would(analysed):
    config, data, log, _, _ = analysed

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


def test_the_window_before_cleaning_is_the_one_that_was_cleaned(analysed):
    """Sample for sample the window the cleaning kept, re-referenced and nothing else."""
    config, _, log, raw, cleaned = analysed

    before = demo.before_cleaning(raw, config)

    assert before.shape == cleaned.get_data().shape
    assert np.abs(before.mean(axis=0)).max() < 1e-6  # average reference: channels sum to 0
    start = int(round(log["crop"]["start_s"] * raw.info["sfreq"]))
    reference = raw.get_data(start=start, stop=start + before.shape[1]) * 1e6
    np.testing.assert_allclose(before, reference - reference.mean(axis=0), atol=1e-6)
    # Cleaning band-passed it: the slow drift and the DC offsets are only in the "before".
    assert (
        np.abs(before.mean(axis=1)).max() > 10 * np.abs(cleaned.get_data().mean(axis=1) * 1e6).max()
    )


def test_band_power_is_computed_per_channel_as_for_training(analysed):
    config, _, _, _, cleaned = analysed

    power = demo.band_power(cleaned, config)

    assert set(power["band"]) == set(config.psd.bands)
    assert set(power["segments"]) == {"full", "clean"}
    clean = power[power["segments"] == "clean"]
    assert len(clean) == len(CHANNELS) * len(config.psd.bands)
    # Relative power on a log scale: every band is a share of the total, so below 0.
    assert np.isfinite(clean["value"]).all() and (clean["value"] < 0).all()


# --- the research app (app/research_app.py) ----------------------------------------------------


def html(app) -> str:
    """Everything the page drew with st.html."""
    return " ".join(element.proto.body for element in app.get("html"))


def finish_analysis(app, timeout: float = 900.0):
    """Start the analysis and rerun the page until it has finished in the background."""
    run = next(button for button in app.button if button.label == "Run analysis")
    run.click().run()
    deadline = time.monotonic() + timeout
    while any(status.label.startswith("Analysing") for status in app.status):
        assert time.monotonic() < deadline, "the analysis did not finish"
        time.sleep(0.5)
        app.run()
    return app


def disclaimer(app) -> str:
    return next(m.value for m in app.sidebar.markdown if "not a diagnostic tool" in m.value)


def test_the_app_opens_on_the_overview_with_the_disclaimer():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=120).run()

    assert not app.exception
    banner = disclaimer(app)
    assert banner.startswith("**Research demo – not a diagnostic tool.**")
    assert "(OpenNeuro ds002778)" in banner and "not validated on other recordings" in banner
    assert len(app.file_uploader) == 1
    page = html(app)
    assert "Research mode" in page and "No recording loaded" in page


@pytest.mark.parametrize("page", PAGES)
def test_every_page_opens_without_a_recording(page):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.switch_page(f"views/{page}.py").run()

    assert not app.exception
    assert not app.error


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
    assert "No models have been trained here yet" in disclaimer(app)
    assert "recording.bdf" in html(app) and "Passed" in html(app)
    finish_analysis(app)

    assert not app.exception
    (warning,) = [w.value for w in app.warning if "no prediction" in w.value]
    assert "pdeeg train-final" in warning  # the models are missing, said as `predict` says it
    app.switch_page("views/recording.py").run()
    assert not app.exception
    assert "ICA" in html(app) and "Bad windows" in html(app)
    app.switch_page("views/quality.py").run()
    assert not app.exception and not app.error
    app.switch_page("views/multifractal.py").run()
    assert not app.exception and not app.error
    assert [tab.label for tab in app.tabs] == [
        "Fluctuation functions",
        "Singularity spectrum",
        "Scalp maps",
        "Feature table",
    ]
    assert len(app.dataframe) == 1  # the features of every channel
    app.switch_page("views/topography.py").run()
    assert not app.exception and not app.error
    app.switch_page("views/evaluation.py").run()
    assert not app.exception
    assert any("No prediction" in w.value for w in app.warning)
    app.switch_page("views/report.py").run()
    next(b for b in app.button if b.label == "Build report").click().run()
    assert not app.exception
    labels = [button.label for button in app.get("download_button")]
    assert "Report (HTML)" in labels and "Result (JSON)" not in labels  # no result to export
    assert "not a diagnostic tool" in disclaimer(app)
    streamlit.cache_data.clear()
    streamlit.cache_resource.clear()


@pytest.fixture
def neurolab():
    pytest.importorskip("streamlit")
    if str(APP.parent) not in sys.path:
        sys.path.insert(0, str(APP.parent))
    import neurolab.charts
    import neurolab.engine

    return neurolab


def test_traces_are_thinned_for_the_screen_keeping_every_peak(neurolab):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 10_000))
    x[1, 4321] = 50.0  # a spike
    x[2, 777] = -50.0

    index, values = neurolab.charts.thin(x, points=400)

    assert index.shape[1] <= 400 and values.shape == index.shape
    assert (np.diff(index, axis=1) >= 0).all()  # still in time order
    np.testing.assert_array_equal(values, np.take_along_axis(x, index, axis=1))
    assert values[1].max() == 50.0 and values[2].min() == -50.0
    small = rng.normal(size=(2, 100))
    assert neurolab.charts.thin(small, points=400)[1] is small  # nothing to thin


def test_an_exported_result_names_the_uploads_and_not_the_deleted_work_files(neurolab):
    result = {
        "inputs": [
            {"path": "/tmp/x/upload1/a.bdf", "cleaned": "/tmp/x/c.fif", "cleaning_log": "l"},
        ],
        "stages": [],
    }

    exported = neurolab.engine.export_result(result, ["a.bdf"])

    assert exported["inputs"] == [{"path": "a.bdf"}]
    assert result["inputs"][0]["cleaned"] == "/tmp/x/c.fif"  # the original is left alone


def test_the_app_predicts_only_through_pdeeg_predict():
    """No model code of the app's own: its predictions are those of `pdeeg predict`."""
    app_files = sorted((ROOT / "app").rglob("*.py"))
    for path in (
        *app_files,
        ROOT / "src" / "pdeeg" / "demo.py",
        ROOT / "src" / "pdeeg" / "viz" / "demo.py",
        ROOT / "src" / "pdeeg" / "viz" / "predict.py",
    ):
        text = path.read_text(encoding="utf-8")
        for code in ("pdeeg.models", "sklearn", "joblib", "predict_proba", "load_stage"):
            assert code not in text, (path, code)
    assert any("predict_files(" in path.read_text(encoding="utf-8") for path in app_files)
