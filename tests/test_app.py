"""The single-page app (app/streamlit_app.py): what it says for each result, and when it stops.

predict_files and the stored evaluation are stood in for, so that every state can be shown on
demand; the cleaned recording the graphs are drawn from is real, cleaned once with the short
test configuration of test_demo.py. The models' own predictions are tested in test_predict.py.
"""

import json
import sys
from pathlib import Path

import mne
import numpy as np
import pytest

from pdeeg.config import CONFIG_DIR_ENV, load_config
from pdeeg.predict.features import BAND_POWER, CLEANING, MFDFA
from pdeeg.predict.predictor import RUNNING_MODELS
from test_demo import ROOT, short_config_dir
from test_preprocessing import DURATION, SFREQ, TRIGGER_AT, synthetic_raw

APP = ROOT / "app" / "streamlit_app.py"
if str(APP.parent) not in sys.path:
    sys.path.insert(0, str(APP.parent))

CONFIG = load_config(ROOT / "configs")
TRAINED = {"participants": 31, "recordings": 46, "dataset": {"id": "ds002778"}}


def _cv(accuracy: float, n: int, better: bool) -> dict:
    return {
        "balanced_accuracy": accuracy,
        "ci95": [accuracy - 0.15, accuracy + 0.15],
        "chance": 0.5,
        "permutation_p": 0.02 if better else 0.3,
        "q": 0.01 if better else 0.4,
        "better_than_chance": better,
        "test": "q below 0.05 and the whole 95 % interval above chance (0.5)",
        "not_better_because": [] if better else ["q = 0.4 is not below 0.05"],
        "n_participants": n,
        "n_recordings": n * 2,
        "note": "optimistic",
    }


# The stored evaluation, with numbers unlike the real ones: the page must show these.
CV = {
    "stage1": _cv(0.58, 31, better=False),
    "stage2_single": _cv(0.77, 13, better=True),
    "stage2_paired": _cv(0.92, 13, better=False),
}
BANDS = {"stage1": [0.305, 0.68], "stage2_single": [0.0, 1.0], "stage2_paired": [0.5, 0.5]}
EVIDENCE = {
    stage: {
        "stage": stage,
        "title": stage,
        "model": "linear_svm on psd_mfdfa",
        "threshold": 0.5,
        "uncertain_band": BANDS[stage],
        "cross_validated": cv,
        "model_card": {"limitations": ["One dataset."]},
        "audit_above": 0.85,
    }
    for stage, cv in CV.items()
}
TOP = {
    "kind": "coefficient x standardised value",
    "features": [
        {"feature": "envelope_beta_main_h2_frontal", "contribution": 0.9, "towards": "PD"},
        {"feature": "log_relative_power_alpha_occipital", "contribution": -0.5, "towards": "HC"},
        {"feature": "broadband_long_delta_alpha_central", "contribution": 0.3, "towards": "PD"},
    ],
}
LABEL = {"detected": "PD", "clear": "HC", "inconclusive": "uncertain"}
P = {"detected": 0.83, "clear": 0.12, "inconclusive": 0.5}


def _entry(stage: str, label: str, p: float, **extra) -> dict:
    return {
        "stage": stage,
        "title": stage,
        "label": label,
        "probability": p,
        "uncertain": label == "uncertain",
        "threshold": 0.5,
        "uncertain_band": BANDS[stage],
        "cross_validated": CV[stage],
        "top_features": TOP,
        **extra,
    }


def canned(kind: str, names: list[str], paired_label: str = "first recording is OFF") -> dict:
    """A result of predict_files whose stage 1 says ``kind`` for every recording."""
    stages = []
    for name in names:
        stages.append(_entry("stage1", LABEL[kind], P[kind], recording=name))
        if kind == "detected":
            stages.append(_entry("stage2_single", "uncertain", 0.64, recording=name))
        else:
            stages.append(
                {"stage": "stage2_single", "title": "", "skipped": "no", "recording": name}
            )
    if len(names) == 2:
        stages.append(_entry("stage2_paired", paired_label, 0.71, recordings=names))
    return {
        "disclaimer": "Research demo – not a diagnostic tool",
        "inputs": [{"path": n, "sha256": "0" * 64, "checks": []} for n in names],
        "training_recording": False,
        "stages": stages,
    }


@pytest.fixture(scope="module")
def recordings(tmp_path_factory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("uploads")
    paths = {}
    for name, raw in (
        ("first.bdf", synthetic_raw(seed=0)),
        ("second.bdf", synthetic_raw(seed=1)),
        ("short.bdf", synthetic_raw(seconds=TRIGGER_AT + DURATION - 8.0)),
    ):
        paths[name] = folder / name
        mne.export.export_raw(paths[name], raw, fmt="bdf", verbose="error")
    return paths


@pytest.fixture(scope="module")
def described(recordings, tmp_path_factory):
    """The first recording cleaned and described as the app does it, once."""
    pytest.importorskip("streamlit")
    from neurolab import engine

    root = tmp_path_factory.mktemp("described")
    config = load_config(short_config_dir(root))
    with mne.use_log_level("ERROR"):
        return engine.describe(recordings["first.bdf"], {}, 0, root / "work", config)


@pytest.fixture
def page(described, tmp_path, monkeypatch):
    """Opens the page with predict_files standing in: ``page.kind`` picks what stage 1 says.

    ``page.calls`` lists the files of every call of predict_files.
    """
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv(CONFIG_DIR_ENV, str(short_config_dir(tmp_path)))

    class Page:
        kind = "detected"
        paired_label = "first recording is OFF"
        calls: list[list[str]] = []

        def open(self):
            return AppTest.from_file(str(APP), default_timeout=300).run()

    opened = Page()

    def predict_files(config, paths, *, out_dir=None, n_jobs=1, say=print):
        names = [Path(p).name for p in paths]
        opened.calls.append(names)
        for name in names:
            for step in (CLEANING, MFDFA, BAND_POWER):
                say(f"{name}: {step}")
        say(RUNNING_MODELS)
        return canned(opened.kind, names, opened.paired_label)

    monkeypatch.setattr("pdeeg.predict.predictor.predict_files", predict_files)
    monkeypatch.setattr("pdeeg.predict.predictor.stage_evidence", lambda _config: EVIDENCE)
    monkeypatch.setattr("pdeeg.predict.predictor.trained_on", lambda _config: TRAINED)
    monkeypatch.setattr("neurolab.engine.describe", lambda *_args, **_kwargs: described)
    streamlit.cache_data.clear()
    streamlit.cache_resource.clear()
    yield opened
    streamlit.cache_data.clear()
    streamlit.cache_resource.clear()


def _upload(*paths: Path) -> list[tuple[str, bytes, str]]:
    return [(p.name, p.read_bytes(), "application/octet-stream") for p in paths]


def _html(app) -> str:
    return " ".join(element.proto.body for element in app.get("html"))


def _charts(app) -> int:
    return len(app.get("vega_lite_chart"))


def _analysed(page, *paths: Path):
    app = page.open()
    app.file_uploader[0].set_value(_upload(*paths)).run()
    assert not app.exception, app.exception
    return app


# --- the three verdicts -----------------------------------------------------------------------


def test_the_empty_page_offers_an_upload_and_a_sample():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=120).run()

    assert not app.exception
    page = _html(app)
    assert "Upload a resting-state EEG recording. We analyse its brain-wave patterns" in page
    assert len(app.file_uploader) == 1 and app.file_uploader[0].accept_multiple_files
    assert "Try a sample recording" in [b.label for b in app.button]
    assert "Research demo, not a medical diagnosis." in page
    assert not app.get("expander") and not app.get("tab") and not app.get("popover")


def test_a_parkinsons_pattern_gets_the_verdict_and_the_medication_card(page, recordings):
    app = _analysed(page, recordings["first.bdf"])

    assert page.calls == [["first.bdf"]]
    html = _html(app)
    verdict = html.index("Parkinson&#x27;s pattern detected")
    assert 'class="rw-card rw-verdict detected"' in html
    # How much to trust it: every number from the stored evaluation, which did not beat chance.
    assert "How much to trust this: low." in html
    assert "In testing on 31 people, this step was right 58% of the time (chance is 50%)." in html
    assert "not clearly better than chance" in html
    # The medication card: stage 2 never labels one recording, and says why.
    medication = html.index("Medication")
    assert verdict < medication
    assert "rw-verdict inconclusive small" in html
    assert "In testing on 13 patients, this step was right 77% of the time" in html
    assert "overlap too much to label any single recording" in html
    # The verdict comes before the graphs, and the four cards are all there.
    assert verdict < html.index("What we saw in the signal")
    for title in ("Your brain signal", "Brain-wave balance", "Signal complexity",
                  "What drove the decision"):  # fmt: skip
        assert title in html
    assert _charts(app) == 3
    assert "Persistence of beta rhythm, front of head" in html
    assert "Share of alpha waves (8–13 Hz), back of head" in html
    assert "towards Parkinson&#x27;s" in html and "towards healthy" in html
    assert "Download report (HTML)" in [b.label for b in app.get("download_button")]
    assert "Trained on 31 people from one public dataset (OpenNeuro ds002778)." in html
    assert not app.get("expander") and not app.get("tab")


def test_no_parkinsons_pattern_shows_no_medication_card(page, recordings):
    page.kind = "clear"
    app = _analysed(page, recordings["first.bdf"])

    html = _html(app)
    assert "No Parkinson&#x27;s pattern detected" in html
    assert 'class="rw-card rw-verdict clear"' in html
    assert "Medication" not in html and "rw-verdict inconclusive small" not in html
    assert _charts(app) == 3


def test_an_inconclusive_result_says_the_signal_sits_between_the_groups(page, recordings):
    page.kind = "inconclusive"
    app = _analysed(page, recordings["first.bdf"])

    html = _html(app)
    assert "Inconclusive" in html and "The signal sits between the two groups" in html
    assert 'class="rw-card rw-verdict inconclusive"' in html
    assert "Medication" not in html
    assert "scores from 0.30 to 0.68 are inconclusive" in html  # the gauge marks the band


def test_the_group_ranges_are_drawn_behind_the_recording(page, recordings, monkeypatch):
    import pandas as pd

    from restwave import analysis

    bands = list(CONFIG.psd.bands)
    groups = analysis.Groups(
        band=pd.DataFrame(
            [
                {"group": g, "band": b, "mean": -1.0, "low": -1.2, "high": -0.8}
                for g in ("Healthy group", "Parkinson's group")
                for b in bands
            ]  # fmt: skip
        ),
        spectrum=pd.DataFrame(
            [
                {
                    "group": g,
                    "q": q,
                    "alpha": 1.0 - 0.05 * q,
                    "low": 0.9 - 0.05 * q,
                    "high": 1.1 - 0.05 * q,
                    "f_alpha": 1.0 - 0.01 * q * q,
                }
                for g in ("Healthy group", "Parkinson's group")
                for q in (-2.0, 0.0, 2.0)
            ]
        ),  # fmt: skip
        people={"Healthy group": 16, "Parkinson's group": 15},
    )
    monkeypatch.setattr("restwave.analysis.group_ranges", lambda *_args: groups)
    app = _analysed(page, recordings["first.bdf"])

    html = _html(app)
    assert "Healthy group (16 people)" in html and "Parkinson&#x27;s group (15 people)" in html
    assert "training feature tables" not in html  # no "missing" note
    assert _charts(app) == 3


def test_a_rerun_is_answered_from_the_cache(page, recordings):
    app = _analysed(page, recordings["first.bdf"])
    app.run()

    assert not app.exception
    assert page.calls == [["first.bdf"]]
    assert "Parkinson&#x27;s pattern detected" in _html(app)


# --- when the page stops ---------------------------------------------------------------------


def test_a_file_that_fails_a_check_is_explained_and_nothing_runs(page, recordings):
    app = _analysed(page, recordings["short.bdf"])

    html = _html(app)
    assert 'class="rw-chip failed"' in html and "Too short" in html
    assert "This recording is too short" in html and "How to fix it:" in html
    assert "40 s" in html  # what the analysis needs, from the configuration
    assert page.calls == []
    assert 'class="rw-card rw-verdict' not in html and _charts(app) == 0


def test_two_files_wait_for_the_same_patient_box_then_show_the_pair(page, recordings):
    page.kind = "detected"
    app = _analysed(page, recordings["first.bdf"], recordings["second.bdf"])

    assert page.calls == []
    assert "are they the same patient?" in _html(app)
    app.checkbox[0].check().run()

    assert not app.exception
    assert page.calls == [["first.bdf", "second.bdf"]]
    html = _html(app)
    assert "Recording A · first.bdf" in html and "Recording B · second.bdf" in html
    assert "Recording A looks like the OFF-medication recording" in html
    # Above the audit threshold the score is not reported (reports/classification_results.md).
    assert "In testing on 13 patients, this step scored above 85%" in html and "92%" not in html
    assert html.count("rw-verdict inconclusive small") == 0  # no single-recording stage 2 card


def test_the_paired_result_can_name_the_second_recording(page, recordings):
    page.paired_label = "second recording is OFF"
    app = _analysed(page, recordings["first.bdf"], recordings["second.bdf"])
    app.checkbox[0].check().run()

    assert "Recording B looks like the OFF-medication recording" in _html(app)


# --- the words, without the page -------------------------------------------------------------


def test_every_column_the_models_can_use_has_plain_words():
    pytest.importorskip("streamlit")
    from restwave import plain

    regions = list(CONFIG.data.regions)
    names = [f"log_relative_power_{b}_{r}" for b in CONFIG.psd.bands for r in regions]
    for fit_range, features in CONFIG.mfdfa.broadband.wide_features.items():
        names += [f"broadband_{fit_range}_{f}_{r}" for f in features for r in regions]
    for band in CONFIG.mfdfa.envelope.bands:
        for fit_range, features in CONFIG.mfdfa.envelope.wide_features.items():
            names += [f"envelope_{band}_{fit_range}_{f}_{r}" for f in features for r in regions]
    saved = sorted((ROOT / "models").glob("*/feature_columns.json"))
    for path in saved:  # the final models' own columns, where they have been trained
        names += json.loads(path.read_text(encoding="utf-8"))

    missing = [name for name in names if plain.feature_words(name, CONFIG) is None]

    assert not missing
    assert plain.feature("envelope_beta_main_h2_frontal", CONFIG) == (
        "Persistence of beta rhythm, front of head"
    )
    assert plain.feature("broadband_long_delta_alpha_temporal", CONFIG) == (
        "Irregularity of the brain signal's slow swings (0.3–1 s), sides of head"
    )
    assert plain.feature("log_relative_power_beta_central", CONFIG) == (
        "Share of beta waves (13–30 Hz), top of head"
    )


def test_the_trust_line_reads_every_number_from_the_evaluation():
    pytest.importorskip("streamlit")
    from restwave import verdict

    entry = _entry("stage1", "PD", 0.8)
    low = verdict.trust("stage1", EVIDENCE, entry)
    better = verdict.trust("stage2_single", EVIDENCE, _entry("stage2_single", "uncertain", 0.6))
    fallback = verdict.trust("stage1", {}, entry | {"cross_validated": _cv(0.66, 20, True)})

    assert low.level == "low" and "31 people" in low.line and "58%" in low.line
    assert better.level == "moderate" and "13 patients" in better.line
    assert "overlap too much" in better.line and "overlap" not in low.line
    assert "20 people" in fallback.line and "66%" in fallback.line


def test_a_training_recording_gets_a_note_with_its_true_group():
    pytest.importorskip("streamlit")
    from restwave import verdict

    result = {
        "training_recording": True,
        "stages": [
            {"stage": "stage1", "recording": "sub-pd3 ses-off", "true_label": "PD"},
            {"stage": "stage2_single", "recording": "sub-pd3 ses-off", "true_label": "OFF"},
        ],
    }

    note = verdict.held_out_note(result)

    assert "models that never saw this person" in note
    assert note.endswith("True group: Parkinson's group, recorded off medication.")
    assert verdict.held_out_note({"training_recording": False, "stages": []}) is None


def test_the_signal_window_is_the_first_one_without_a_bad_sample():
    pytest.importorskip("streamlit")
    from restwave.analysis import first_clean_start

    bad = np.zeros(100, dtype=bool)
    bad[[3, 40]] = True

    assert first_clean_start(bad, 36) == 4  # samples 4 to 39
    assert first_clean_start(bad, 37) == 41  # 4 to 40 would hold the bad sample at 40
    assert first_clean_start(np.ones(100, dtype=bool), 10) == 0
    assert SFREQ > 0  # the recordings above are at the test rate
