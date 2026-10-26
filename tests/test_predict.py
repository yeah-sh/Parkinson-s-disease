"""The final models and `pdeeg predict`: inputs, training recordings, features, end to end."""

import argparse
import dataclasses
import json
import shutil
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
