"""The final models and `pdeeg predict`: inputs, training recordings, features, end to end."""

import argparse
import dataclasses
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import mne
import numpy as np
import pandas as pd
import pytest

from pdeeg import cli, pipeline
from pdeeg.config import FINAL_STAGES, FinalStageConfig, load_config
from pdeeg.features.tables import table_paths
from pdeeg.models.calibration import uncertain_band
from pdeeg.predict.features import feature_digest, recording_tables
from pdeeg.predict.final import (
    FinalError,
    best_choices,
    check_bands,
    check_final,
    plan_stage,
    stage_tasks,
    train_final,
)
from pdeeg.predict.inputs import InputError, identify_training, read_recording
from pdeeg.predict.predictor import predict_files, summary_text
from test_pipeline import pipeline_config, write_inputs

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG = load_config(CONFIG_DIR)
SCALP = mne.channels.make_standard_montage("biosemi32").ch_names


def synthetic_raw(
    seconds: float = 200.0, sfreq: float = 512.0, seed: int = 0, names=None, scale=10e-6
) -> mne.io.RawArray:
    """Pink noise mixed over the scalp, posterior alpha, and trigger "1" at 5 s."""
    rng = np.random.default_rng(seed)
    names = SCALP if names is None else names
    n = int(seconds * sfreq)
    freqs = np.fft.rfftfreq(n, 1 / sfreq)
    freqs[0] = freqs[1]

    def pink(k):
        spectrum = np.fft.rfft(rng.standard_normal((k, n)), axis=1) / np.sqrt(freqs)
        return np.fft.irfft(spectrum, n=n, axis=1)

    data = rng.standard_normal((len(names), 8)) @ pink(8) + 0.5 * pink(len(names))
    t = np.arange(n) / sfreq
    alpha = np.sin(2 * np.pi * 10.0 * t + np.cumsum(rng.normal(0.0, 0.05, n)))
    posterior = np.array([2.0 if name[0] in "OP" else 0.3 for name in names])
    data = data / data.std(axis=1, keepdims=True) + posterior[:, None] * alpha
    data = data / data.std(axis=1, keepdims=True) * scale
    raw = mne.io.RawArray(data, mne.create_info(list(names), sfreq, "eeg"), verbose="error")
    raw.set_annotations(mne.Annotations([5.0], [0.0], ["1"]))
    return raw


def write_edf(raw: mne.io.BaseRaw, path: Path) -> Path:
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose="error")
    return path


# --- input checks ----------------------------------------------------------------------------


def test_a_good_recording_is_read_in_montage_order(tmp_path):
    names = [{"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}.get(n, n) for n in SCALP]
    names = [f"EEG {n}-REF" if n == "Fp1" else n for n in names][::-1]  # any order
    path = write_edf(synthetic_raw(names=names), tmp_path / "old_names.edf")

    recording = read_recording(path, CONFIG)

    assert recording.raw.ch_names == SCALP
    assert recording.raw.info["sfreq"] == 512.0
    assert recording.raw.get_montage() is not None
    assert any("T3 -> T7" in note and "EEG Fp1-REF -> Fp1" in note for note in recording.notes)
    assert recording.available_s == pytest.approx(195.0, abs=0.01)  # from trigger "1" at 5 s


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"names": [n for n in SCALP if n != "Cz"] + ["Ch33"]}, "missing: Cz"),
        ({"names": [f"Ch{i}" for i in range(32)]}, "32 of the 32 montage channels"),
        ({"seconds": 100.0}, "at least 180 s"),
        ({"scale": 1e-12}, "are the units"),
        ({"scale": 1e-1}, "are the units"),
    ],
)
def test_unusable_recordings_are_refused(tmp_path, change, message):
    path = write_edf(synthetic_raw(**change), tmp_path / "bad.edf")

    with pytest.raises(InputError, match=message):
        read_recording(path, CONFIG)


def test_two_channels_claiming_one_name_are_refused(tmp_path):
    names = [*SCALP[:-1], "EEG Fp1"]  # Cz missing, Fp1 twice
    path = write_edf(synthetic_raw(names=names), tmp_path / "twice.edf")

    with pytest.raises(InputError, match="would both be Fp1"):
        read_recording(path, CONFIG)


def test_other_files_are_refused(tmp_path):
    with pytest.raises(InputError, match="no such file"):
        read_recording(tmp_path / "absent.bdf", CONFIG)
    text = tmp_path / "notes.txt"
    text.write_text("not EEG", encoding="utf-8")
    with pytest.raises(InputError, match="expected a .bdf or .edf"):
        read_recording(text, CONFIG)
    broken = tmp_path / "broken.edf"
    broken.write_bytes(b"0" * 300)
    with pytest.raises(InputError, match="cannot be read as EEG"):
        read_recording(broken, CONFIG)


def test_another_sampling_rate_is_resampled_with_a_note(tmp_path):
    path = write_edf(synthetic_raw(sfreq=256.0), tmp_path / "slow.edf")

    recording = read_recording(path, CONFIG)

    assert recording.raw.info["sfreq"] == 512.0
    assert any(note.startswith("resampled from 256 Hz to 512 Hz") for note in recording.notes)


def test_every_check_is_kept_to_show_the_person_who_gave_the_file(tmp_path):
    names = ["file", "channels", "sampling rate", "duration", "amplitude"]
    slow = read_recording(write_edf(synthetic_raw(sfreq=256.0), tmp_path / "slow.edf"), CONFIG)

    assert [check.name for check in slow.checks] == names
    assert {c.name: c.status for c in slow.checks}["sampling rate"] == "note"
    assert {c.status for c in slow.checks} == {"passed", "note"}

    with pytest.raises(InputError) as refused:
        read_recording(write_edf(synthetic_raw(seconds=100.0), tmp_path / "short.edf"), CONFIG)
    checks = [(check.name, check.status) for check in refused.value.checks]
    assert checks == [*((name, "passed") for name in names[:3]), ("duration", "failed")]
    assert refused.value.checks[-1].detail == str(refused.value)


# --- recordings the models were trained on --------------------------------------------------


def _registry(recording, subject="pd3", session="off"):
    return pd.DataFrame(
        [
            {
                "subject": subject,
                "session": session,
                "bids_path": f"sub-{subject}/ses-{session}/eeg/x.bdf",
                "file_sha256": recording.file_sha256,
                "signal_sha256": recording.signal_sha256,
            }
        ]
    )


def test_a_training_recording_is_recognised_by_file_signal_or_bids_name(tmp_path):
    raw = synthetic_raw(seed=3)
    original = read_recording(write_edf(raw, tmp_path / "original.edf"), CONFIG)
    registry = _registry(original)

    copy = tmp_path / "renamed.edf"
    shutil.copy(original.path, copy)
    match = identify_training(read_recording(copy, CONFIG), registry, CONFIG)
    assert match["subject"] == "pd3" and match["matched_by"] == "SHA-256 of the file"

    # The same signal in a file with another header.
    again = mne.io.read_raw_edf(original.path, preload=True, verbose="error")
    again.info["subject_info"] = {"his_id": "someone-else"}
    other = read_recording(write_edf(again, tmp_path / "header.edf"), CONFIG)
    assert other.file_sha256 != original.file_sha256
    assert identify_training(other, registry, CONFIG)["matched_by"] == "hash of the signal"

    unrelated = read_recording(write_edf(synthetic_raw(seed=4), tmp_path / "new.edf"), CONFIG)
    assert identify_training(unrelated, registry, CONFIG) is None

    # By name, in a BIDS dataset of the training data, whatever the signal.
    from mne_bids import BIDSPath, write_raw_bids

    root = tmp_path / "bids"
    raw4 = synthetic_raw(seed=4)
    raw4.info["line_freq"] = 60.0
    bids_path = BIDSPath(subject="pd3", session="off", task="rest", datatype="eeg", root=root)
    write_raw_bids(raw4, bids_path, format="EDF", allow_preload=True, verbose="error")
    description = root / "dataset_description.json"
    info = json.loads(description.read_text(encoding="utf-8"))
    info["DatasetDOI"] = "doi:10.18112/openneuro.ds002778.v1.0.5"
    description.write_text(json.dumps(info), encoding="utf-8")
    in_bids = read_recording(bids_path.fpath, CONFIG)
    assert in_bids.bids["subject"] == "pd3" and in_bids.bids["session"] == "off"
    match = identify_training(in_bids, registry.assign(signal_sha256="x", file_sha256="x"), CONFIG)
    assert match["matched_by"] == "BIDS subject and session"


# --- the uncertain band ------------------------------------------------------------------------


def test_the_band_is_narrow_for_strong_evidence_and_everything_for_none():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 600)
    p = 1 / (1 + np.exp(-3 * x))
    y = (rng.random(x.size) < p).astype(int)
    groups = np.arange(x.size)

    def band(probability, labels):
        rng = np.random.default_rng(1)
        return uncertain_band(labels, probability, groups, None, 300, 0.95, rng)

    low, high = band(p, y)
    assert low <= 0.5 <= high and high - low < 0.3  # it always holds the threshold
    unrelated = rng.permutation(np.repeat([0, 1], x.size // 2))
    assert band(p, unrelated) == (0.0, 1.0)  # probabilities that say nothing about the labels
    assert band(p, 1 - y) == (0.0, 1.0)  # confidently wrong backs no label either
    assert band(np.full(x.size, 0.5), y) == (0.0, 1.0)  # one probability for everyone


# --- better than chance --------------------------------------------------------------------------


def _cross_validated(p, q, low):
    interval = {"value": 0.7, "low": low, "high": 0.9, "p": p}
    entry = {
        "feature_set": "psd",
        "model": "logistic_regression",
        "n_participants": 31,
        "n_recordings": 44,
        "balanced_accuracy": interval,
        "balanced_accuracy_q": q,
    }
    return {
        "selected": entry | {"selected_among": 9, "note": "optimistic"},
        "prespecified": entry | {"note": "fixed"},
    }


@pytest.mark.parametrize(
    ("p", "q", "low", "rule", "better", "reason"),
    [
        (0.001, 0.009, 0.65, "q", True, None),
        (0.020, 0.176, 0.55, "q", False, "q = 0.176 is not below 0.05"),
        (0.020, 0.176, 0.55, "p", True, None),
        (0.001, 0.009, 0.45, "q", False, "reaches chance (0.5)"),
    ],
)
def test_a_stage_is_better_than_chance_only_by_the_configured_test(p, q, low, rule, better, reason):
    from pdeeg.config import ChanceConfig
    from pdeeg.predict.predictor import _validation

    entry = _validation(_cross_validated(p, q, low), ChanceConfig(alpha=0.05, p_value=rule))

    assert entry["better_than_chance"] is better and entry["chance"] == 0.5
    assert (entry["permutation_p"], entry["q"]) == (p, q)
    if reason:
        assert any(reason in why for why in entry["not_better_because"])
    assert ("best of 9" in entry["test"]) == (rule == "q")


# --- the final models, end to end ---------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Classification, cascade and final models on the synthetic tables of the pipeline tests."""
    root = tmp_path_factory.mktemp("final")
    config = pipeline_config(root)
    model = config.model
    model = dataclasses.replace(
        model,
        tasks={t: model.tasks[t] for t in ("off_vs_on", "pd_any_vs_hc", "off_vs_on_single")},
        final=None,  # set below from this ablation
        feature_sets={"mfdfa": ("mfdfa",)},
        cascade=dataclasses.replace(model.cascade, feature_set="mfdfa"),
    )
    predict = dataclasses.replace(
        config.predict,
        models_dir=root / "models",
        output_dir=root / "predictions",
        band=dataclasses.replace(config.predict.band, n_bootstrap=200),
    )
    config = dataclasses.replace(config, model=model, predict=predict)
    write_inputs(config)
    quiet = []
    pipeline.classify_step(config, write=False, say=quiet.append)
    pipeline.cascade_step(config, write=False, say=quiet.append)
    recordings = pd.read_parquet(config.data.paths.recordings)
    tables = {kind: pd.read_parquet(table_paths(config, kind)["wide"]) for kind in ("psd", "mfdfa")}
    summary = pd.read_parquet(pipeline.summary_path(config))
    unset = config

    best = best_choices(summary, config)
    tasks = stage_tasks(config)
    final = {stage: FinalStageConfig(tasks[stage], *best[stage]) for stage in FINAL_STAGES}
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, final=final))
    plans = [plan_stage(config, recordings, tables, stage) for stage in FINAL_STAGES]
    stages = {
        plan.stage: dataclasses.replace(config.predict.stages[plan.stage], uncertain=plan.band)
        for plan in plans
    }
    unbanded = config
    config = dataclasses.replace(config, predict=dataclasses.replace(config.predict, stages=stages))

    training = read_recording(write_edf(synthetic_raw(seed=2), root / "training.edf"), config)
    written = train_final(
        config, recordings, tables, summary, registry=_registry(training), say=quiet.append
    )
    return SimpleNamespace(
        config=config,
        unset=unset,
        unbanded=unbanded,
        summary=summary,
        plans=plans,
        written=written,
        training=training.path,
        root=root,
    )


def test_every_stage_is_saved_with_what_predictions_need(trained):
    config = trained.config
    for stage in FINAL_STAGES:
        folder = config.predict.models_dir / stage
        info = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        columns = json.loads((folder / "feature_columns.json").read_text(encoding="utf-8"))
        pipeline_ = joblib.load(folder / "model.joblib")

        assert info["config"]["feature_sha256"] == feature_digest(config)
        assert info["dataset"] == {"id": "ds002778", "version": "1.0.5"}
        assert {"git_commit", "git_dirty"} <= set(info["git"])
        assert info["thresholds"]["uncertain"] == list(config.predict.stages[stage].uncertain)
        assert (
            info["cross_validated"]["selected"]["balanced_accuracy"]["low"]
            <= (info["cross_validated"]["selected"]["balanced_accuracy"]["value"])
        )
        assert "optimistic" in info["cross_validated"]["selected"]["note"]
        assert info["cross_validated"]["prespecified"]["model"] == "logistic_regression"
        assert ("in_cascade" in info["cross_validated"]) == (stage != "stage2_paired")
        assert info["model_card"]["disclaimer"] == "Research demo – not a diagnostic tool"
        probability = pipeline_.predict_proba(np.zeros((1, len(columns))))[0]
        assert probability.sum() == pytest.approx(1.0)
        oof = pd.read_parquet(folder / "oof_predictions.parquet")
        assert oof["probability"].between(0, 1).all() and (oof["fold"] >= 0).all()
    registry = pd.read_parquet(config.predict.models_dir / "training_recordings.parquet")
    assert list(registry["subject"]) == ["pd3"]


def test_train_final_refuses_a_configuration_it_was_not_given(trained):
    with pytest.raises(FinalError, match="model.final is null"):
        check_final(trained.unset, trained.summary)
    with pytest.raises(FinalError, match="uncertain bands the rule gives"):
        check_bands(trained.unbanded, trained.plans)
    wrong = dict(trained.config.model.final)
    other = (
        "logistic_regression" if wrong["stage1"].model != "logistic_regression" else "linear_svm"
    )
    wrong["stage1"] = dataclasses.replace(wrong["stage1"], model=other)
    config = dataclasses.replace(
        trained.config, model=dataclasses.replace(trained.config.model, final=wrong)
    )
    with pytest.raises(FinalError, match="not the best of the ablation for stage1"):
        check_final(config, trained.summary)


def test_a_training_recording_gets_its_out_of_fold_results(trained, monkeypatch, tmp_path):
    def no_model(*_args, **_kwargs):
        raise AssertionError("a final model was loaded for a training recording")

    monkeypatch.setattr("pdeeg.predict.predictor.load_stage", no_model)
    copy = tmp_path / "my_recording.edf"
    shutil.copy(trained.training, copy)
    args = argparse.Namespace(file=copy, paired=None, out_dir=tmp_path / "out", n_jobs=1)

    cli._predict(trained.config, args)

    result = json.loads((tmp_path / "out" / "my_recording_prediction.json").read_text("utf-8"))
    assert result["training_recording"] is True and "never saw" in result["explanation"]
    assert result["inputs"][0]["training_recording"]["matched_by"] == "SHA-256 of the file"
    stage1 = next(s for s in result["stages"] if s["stage"] == "stage1")
    oof = pd.read_parquet(trained.config.predict.models_dir / "stage1" / "oof_predictions.parquet")
    row = oof[(oof["subject"] == "pd3") & (oof["session"] == "off")].iloc[0]
    assert stage1["out_of_fold"] and stage1["probability"] == pytest.approx(row["probability"])
    assert stage1["true_label"] == "PD"
    assert isinstance(stage1["cross_validated"]["better_than_chance"], bool)
    text = (tmp_path / "out" / "my_recording_prediction.txt").read_text("utf-8")
    assert "Research demo – not a diagnostic tool" in text and "out of fold" in text


def test_two_new_recordings_go_through_every_stage(trained, tmp_path):
    first = write_edf(synthetic_raw(seed=11), tmp_path / "first.edf")
    second = write_edf(synthetic_raw(seed=12, sfreq=256.0), tmp_path / "second.edf")

    result = predict_files(
        trained.config, [first, second], out_dir=tmp_path, n_jobs=2, say=lambda _: None
    )

    json.dumps(result)  # all of it is JSON
    assert result["disclaimer"] == "Research demo – not a diagnostic tool"
    assert result["training_recording"] is False
    assert any(note.startswith("resampled") for note in result["inputs"][1]["notes"])
    for entry in result["inputs"]:
        assert Path(entry["cleaned"]).is_file() and Path(entry["cleaning_log"]).is_file()
        assert [check["name"] for check in entry["checks"]][:3] == [
            "file",
            "channels",
            "sampling rate",
        ]
    by_stage = {}
    for entry in result["stages"]:
        by_stage.setdefault(entry["stage"], []).append(entry)
    assert len(by_stage["stage1"]) == len(by_stage["stage2_single"]) == 2
    for one, two in zip(by_stage["stage1"], by_stage["stage2_single"], strict=True):
        assert one["label"] in ("PD", "HC", "uncertain") and 0 <= one["probability"] <= 1
        assert one["uncertain"] == (one["label"] == "uncertain")
        assert len(one["top_features"]["features"]) == trained.config.predict.n_top_features
        cv = one["cross_validated"]
        assert cv["ci95"][0] <= cv["balanced_accuracy"] <= cv["ci95"][1]
        assert cv["better_than_chance"] == (cv["q"] < 0.05 and cv["ci95"][0] > cv["chance"])
        assert ("skipped" in two) == (one["label"] != "PD")  # stage 2 only after a sure PD
    paired = by_stage["stage2_paired"][0]
    assert 0 <= paired["probability"] <= 1 and paired["recordings"] == ["first.edf", "second.edf"]
    assert (tmp_path / "first_work" / "qc").is_dir()  # the input's own QC figures
    text = summary_text(result)
    assert "Stage 1: patient or control" in text and text.endswith(result["disclaimer"])


def test_a_model_trained_with_other_settings_is_refused(trained, tmp_path):
    from pdeeg.predict.predictor import PredictError, load_stage

    config = trained.config
    other = dataclasses.replace(
        config, psd=dataclasses.replace(config.psd, window_sec=config.psd.window_sec * 2)
    )
    with pytest.raises(PredictError, match="other preprocessing or feature settings"):
        load_stage(other, "stage1")


# --- the saved models' evidence, without a recording ------------------------------------------


def test_each_stage_says_how_it_did_without_a_recording(trained):
    from pdeeg.predict.predictor import stage_evidence

    evidence = stage_evidence(trained.config)

    assert list(evidence) == list(FINAL_STAGES)
    for stage, entry in evidence.items():
        folder = trained.config.predict.models_dir / stage
        info = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        selected = info["cross_validated"]["selected"]["balanced_accuracy"]
        cv = entry["cross_validated"]
        assert cv["balanced_accuracy"] == selected["value"]
        assert cv["ci95"] == [selected["low"], selected["high"]]
        assert set(entry["metrics"]) == {
            "balanced_accuracy",
            "roc_auc",
            "sensitivity",
            "specificity",
        }
        assert entry["uncertain_band"] == list(trained.config.predict.stages[stage].uncertain)
        oof = pd.read_parquet(folder / "oof_predictions.parquet")
        pd.testing.assert_frame_equal(entry["oof"], oof)
    elsewhere = dataclasses.replace(trained.config.predict, models_dir=trained.root / "none")
    assert stage_evidence(dataclasses.replace(trained.config, predict=elsewhere)) == {}


# --- the app (app/streamlit_app.py) -------------------------------------------------------------


APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture
def open_app(trained, monkeypatch):
    """Opens the app, with the configuration and final models of ``trained``."""
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr("pdeeg.config.load_config", lambda *_args, **_kwargs: trained.config)
    streamlit.cache_data.clear()
    streamlit.cache_resource.clear()
    yield lambda: AppTest.from_file(str(APP), default_timeout=900).run()
    streamlit.cache_data.clear()
    streamlit.cache_resource.clear()


def _upload(path: Path) -> tuple[str, bytes, str]:
    return (path.name, path.read_bytes(), "application/octet-stream")


def _spy(monkeypatch) -> list[dict]:
    """The results of every call of predict_files, which the app is to call for predictions."""
    from pdeeg.predict import predictor

    results, real = [], predictor.predict_files

    def spy(*args, **kwargs):
        results.append(real(*args, **kwargs))
        return results[-1]

    monkeypatch.setattr("pdeeg.predict.predictor.predict_files", spy)
    return results


def _html(app) -> str:
    return " ".join(element.proto.body for element in app.get("html"))


def _analyse(app, timeout: float = 900.0):
    """Start the analysis and rerun the page until it has finished in the background."""
    next(button for button in app.button if button.label == "Run analysis").click().run()
    deadline = time.monotonic() + timeout
    while any(status.label.startswith("Analysing") for status in app.status):
        assert time.monotonic() < deadline, "the analysis did not finish"
        time.sleep(0.5)
        app.run()
    assert not app.exception
    return app


def test_the_app_checks_an_upload_then_shows_what_predict_gives(
    open_app, trained, tmp_path, monkeypatch
):
    from pdeeg.predict.predictor import trained_on

    results = _spy(monkeypatch)
    path = write_edf(synthetic_raw(seed=21), tmp_path / "new_recording.edf")
    app = open_app()
    people = trained_on(trained.config)["participants"]
    banner = next(m.value for m in app.sidebar.markdown if "not a diagnostic tool" in m.value)
    assert "Research demo – not a diagnostic tool." in banner
    assert f"Trained on {people} people from one dataset (OpenNeuro ds002778)" in banner
    assert "not validated on other recordings" in banner

    app.file_uploader[0].set_value(_upload(path)).run()

    assert not app.exception and not app.error
    assert "new_recording.edf" in _html(app)
    assert not results  # checked, but nothing is predicted before it is asked for
    app.switch_page("views/recording.py").run()
    checks = _html(app)
    for check in ("file", "channels", "sampling rate", "duration", "amplitude"):
        assert f'<div class="n">{check}</div>' in checks
    assert "nl-check failed" not in checks
    _analyse(app)
    (result,) = results  # the app predicted through predict_files, once

    app.switch_page("views/evaluation.py").run()
    assert not app.exception and not app.error
    assert "External, unseen recording" in _html(app)
    this = app.tabs[0]
    ran = [stage for stage in result["stages"] if "skipped" not in stage]
    assert [e.label for e in this.expander].count("How reliable is this?") == len(ran)
    stage1 = result["stages"][0]
    shown = f"Model score for Parkinson's: **{stage1['probability']:.2f}**"
    assert any(m.value == shown for m in this.markdown)
    cv = stage1["cross_validated"]
    metrics = {m.label: m.value for m in this.metric}
    assert metrics["Balanced accuracy"] == f"{cv['balanced_accuracy']:.2f}"
    assert metrics["95% CI"] == f"{cv['ci95'][0]:.2f} – {cv['ci95'][1]:.2f}"
    assert metrics["Chance"] == "0.50"
    assert metrics["Permutation p"] == f"{cv['permutation_p']:.3f}"
    weak = [w for w in this.warning if "Not better than chance" in w.value]
    assert len(weak) == sum(not stage["cross_validated"]["better_than_chance"] for stage in ran)
    assert "#### Stage 2 · OFF vs ON medication" in [m.value for m in this.markdown]

    app.switch_page("views/multifractal.py").run()
    assert not app.exception and len(app.dataframe) == 1
    app.switch_page("views/report.py").run()
    next(b for b in app.button if b.label == "Build report").click().run()
    assert not app.exception
    labels = [button.label for button in app.get("download_button")]
    assert {"Report (HTML)", "Result (JSON)", "Summary (text)"} <= set(labels)
    assert any("not a diagnostic tool" in m.value for m in app.sidebar.markdown)


def test_the_app_stops_before_predicting_a_file_that_fails_a_check(open_app, tmp_path, monkeypatch):
    results = _spy(monkeypatch)
    short = write_edf(synthetic_raw(seconds=100.0), tmp_path / "short.edf")
    app = open_app()

    app.file_uploader[0].set_value(_upload(short)).run()

    assert not app.exception
    assert "short.edf cannot be used" in app.error[0].value and "at least" in app.error[0].value
    assert "Run analysis" not in [button.label for button in app.button] and not results
    app.switch_page("views/recording.py").run()
    assert "nl-check failed" in _html(app) and "How to fix" in _html(app)
    app.switch_page("views/multifractal.py").run()
    assert not app.exception and "did not pass its checks" in _html(app)

    app.switch_page("views/overview.py").run()
    next(b for b in app.button if b.label == "Clear recording").click().run()
    assert len(app.file_uploader) == 1
    app.segmented_control[0].set_value("paired").run()  # two good files, not said to be one patient
    first = write_edf(synthetic_raw(seed=31), tmp_path / "first.edf")
    second = write_edf(synthetic_raw(seed=32), tmp_path / "second.edf")
    app.file_uploader[0].set_value(_upload(first))
    app.file_uploader[1].set_value(_upload(second)).run()

    assert not app.exception and not app.error
    assert "first.edf" in _html(app) and "second.edf" in _html(app)  # both checked
    assert any("same patient" in warning.value for warning in app.warning)
    assert next(b for b in app.button if b.label == "Run analysis").disabled
    assert not results


def test_the_app_shows_a_training_recording_out_of_fold(open_app, trained, monkeypatch):
    results = _spy(monkeypatch)
    app = open_app()

    app.file_uploader[0].set_value(_upload(trained.training)).run()
    _analyse(app)

    (result,) = results
    assert result["training_recording"] is True
    app.switch_page("views/evaluation.py").run()
    assert not app.exception and not app.error
    explanation = next(i.value for i in app.info if "training recording" in i.value)
    assert "sub-pd3 ses-off" in explanation and "never saw this participant" in explanation
    assert any("Out of fold (fold" in m.value for m in app.markdown)
    assert any("Out-of-fold results come from models" in c.value for c in app.caption)
    page = _html(app)
    assert page.count('<div class="nl-prov active">') == 2  # held-out participant, out of fold
    assert "Never shown" in page  # a prediction on training data


# --- the same features as training, on a ds002778 recording -------------------------------------


@pytest.mark.dataset
def test_predict_makes_the_training_features_of_a_ds002778_recording(tmp_path):
    """Cleaning, MFDFA (without surrogates) and band power of one recording through `predict`'s
    path give the values of the training tables."""
    from pdeeg.data.bids import load_raw

    config = CONFIG
    paths = {kind: table_paths(config, kind)["wide"] for kind in ("psd", "mfdfa")}
    if not config.data.paths.recordings.is_file() or not all(p.is_file() for p in paths.values()):
        pytest.skip("the ds002778 feature tables have not been made; run `pdeeg features`")
    recordings = pd.read_parquet(config.data.paths.recordings)
    row = recordings[(recordings["subject"] == "pd3") & (recordings["session"] == "off")].iloc[0]
    raw = load_raw(SimpleNamespace(**row), config.data.paths.raw, config.data.dataset.montage)

    tables = recording_tables(raw, config, tmp_path, n_jobs=8)

    keys = ["subject", "session", "group", "segments"]
    for kind, path in paths.items():
        stored = pd.read_parquet(path)
        stored = stored[(stored["subject"] == "pd3") & (stored["session"] == "off")]
        stored = stored.drop(columns=["subject", "session", "group"]).set_index("segments")
        mine = tables[kind].drop(columns=keys[:3]).set_index("segments").loc[stored.index]
        assert list(mine.columns) == list(stored.columns), kind
        np.testing.assert_allclose(
            mine.to_numpy(), stored.to_numpy(), rtol=1e-6, atol=1e-9, err_msg=kind
        )
