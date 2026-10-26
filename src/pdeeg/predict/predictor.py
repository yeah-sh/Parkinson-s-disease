"""``pdeeg predict``: one recording (or a patient's two) through the final models.

1. Every input is read and checked (:func:`pdeeg.predict.inputs.read_recording`) before
   anything else.
2. An input the final models were trained on gets no prediction from them: its out-of-fold
   predictions from cross-validation, made by models that never saw that participant, are
   shown instead.
3. Otherwise the features are made by the training pipeline's code
   (:mod:`pdeeg.predict.features`), after checking that the configuration is the one the models
   were trained with, and the stages run in turn: stage 1, P(patient); stage 2 on one
   recording, P(OFF), only when stage 1 calls a patient outside its uncertain band; with two
   recordings of one patient, the paired model, which of them is OFF.

Every stage's result carries how the stage did in cross-validation and whether that was better
than chance (``predict.better_than_chance``). The result is a JSON-ready dict
(:func:`predict_files`) and a readable summary (:func:`summary_text`). Research demo, not a
diagnostic tool.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from pdeeg.config import ChanceConfig, Config
from pdeeg.predict.features import feature_digest, feature_vector, recording_tables, work_files
from pdeeg.predict.final import REGISTRY, STAGE_TITLES
from pdeeg.predict.inputs import InputError, Recording, identify_training, read_recording

Say = Callable[[str], None]
LABELS = {
    "stage1": {1: "PD", 0: "HC"},
    "stage2_single": {1: "OFF", 0: "ON"},
    "stage2_paired": {1: "first recording is OFF", 0: "second recording is OFF"},
}
UNCERTAIN = "uncertain"
# Balanced accuracy of guessing, for every stage: each has two classes.
CHANCE = 0.5
_MATCH_KEYS = ("subject", "session", "matched_by")


class PredictError(Exception):
    """The models cannot be used for this prediction; the message says why."""


@dataclass(frozen=True)
class StageModel:
    """A saved stage: the fitted pipeline, its feature columns and its metadata."""

    stage: str
    pipeline: Pipeline
    columns: list[str]
    metadata: dict[str, Any]

    @property
    def threshold(self) -> float:
        return float(self.metadata["thresholds"]["threshold"])

    @property
    def band(self) -> tuple[float, float]:
        low, high = self.metadata["thresholds"]["uncertain"]
        return float(low), float(high)


def load_stage(config: Config, stage: str) -> StageModel:
    """A stage's saved model, refused unless it fits this configuration and installation."""
    folder = config.predict.models_dir / stage
    if not (folder / "metadata.json").is_file():
        raise PredictError(f"no model in {folder}; run `pdeeg train-final` first")
    info = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    digest = feature_digest(config)
    if info["config"]["feature_sha256"] != digest:
        raise PredictError(
            f"{stage}: the model was trained with other preprocessing or feature settings "
            f"(digest {info['config']['feature_sha256'][:12]}, now {digest[:12]}); run "
            "`pdeeg train-final` again"
        )
    if info["versions"]["scikit-learn"] != sklearn.__version__:
        raise PredictError(
            f"{stage}: the model was saved with scikit-learn {info['versions']['scikit-learn']}, "
            f"this is {sklearn.__version__}; run `pdeeg train-final` again"
        )
    configured = config.predict.stages[stage]
    saved = info["thresholds"]
    if configured.uncertain is None or not (
        np.isclose(configured.threshold, saved["threshold"])
        and np.allclose(configured.uncertain, saved["uncertain"])
    ):
        raise PredictError(
            f"{stage}: configs/predict.yaml has other thresholds than the model was trained "
            "with; run `pdeeg train-final` again"
        )
    columns = json.loads((folder / "feature_columns.json").read_text(encoding="utf-8"))
    return StageModel(stage, joblib.load(folder / "model.joblib"), columns, info)


def call(probability: float, threshold: float, band: tuple[float, float]) -> tuple[int, bool]:
    """(1 or 0 by the threshold, whether the probability falls in the uncertain band)."""
    low, high = band
    return int(probability >= threshold), bool(low <= probability <= high)


def contributions(model: StageModel, x: np.ndarray, n: int) -> dict[str, Any]:
    """The features that moved this prediction most.

    For a linear model, coefficient times standardised value, before the Platt scaling (which
    only rescales); positive values push towards the positive class. A boosting model has no
    such split, and its global impurity importances are given instead.
    """
    calibrated = model.pipeline.named_steps["calibrated"]
    inner = calibrated.calibrated_classifiers_[0].estimator
    names = np.asarray(model.columns)
    transformed = inner[:-1].transform(x.reshape(1, -1))[0]
    if "select" in inner.named_steps:
        names = names[inner.named_steps["select"].get_support()]
    estimator = inner.named_steps["model"]
    positive = model.metadata["classes"]["positive"]
    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_).ravel() * transformed
        order = np.argsort(-np.abs(values))[:n]
        return {
            "kind": "coefficient x standardised value",
            "features": [
                {
                    "feature": str(names[i]),
                    "standardised_value": float(transformed[i]),
                    "contribution": float(values[i]),
                    "towards": positive if values[i] > 0 else model.metadata["classes"]["negative"],
                }
                for i in order
            ],
        }
    importances = np.asarray(estimator.feature_importances_)
    order = np.argsort(-importances)[:n]
    return {
        "kind": "global impurity importance (not specific to this recording)",
        "features": [
            {"feature": str(names[i]), "importance": float(importances[i])} for i in order
        ],
    }


def _validation(cv: Mapping[str, Any], rule: ChanceConfig) -> dict[str, Any]:
    """What the stage's cross-validation said (``cross_validated`` of its metadata), and
    whether that was better than chance by ``rule``, for the output."""
    selected = cv["selected"]["balanced_accuracy"]
    q = cv["selected"]["balanced_accuracy_q"]
    tested = {"p": selected["p"], "q": q}[rule.p_value]
    reasons = []
    if not tested < rule.alpha:
        reasons.append(f"{rule.p_value} = {tested:.3g} is not below {rule.alpha:g}")
    if not selected["low"] > CHANCE:
        reasons.append(
            f"the 95 % interval [{selected['low']:.2f}, {selected['high']:.2f}] reaches "
            f"chance ({CHANCE:g})"
        )
    entry = {
        "balanced_accuracy": selected["value"],
        "ci95": [selected["low"], selected["high"]],
        "chance": CHANCE,
        "permutation_p": selected["p"],
        "q": q,
        "better_than_chance": not reasons,
        "test": (
            f"{rule.p_value} below {rule.alpha:g} and the whole 95 % interval above chance "
            f"({CHANCE:g})"
            + (
                f"; q is p adjusted (Benjamini-Hochberg) for picking the best of "
                f"{cv['selected']['selected_among']} combinations"
                if rule.p_value == "q"
                else ""
            )
        ),
        "not_better_because": reasons,
        "n_participants": cv["selected"]["n_participants"],
        "n_recordings": cv["selected"]["n_recordings"],
        "note": cv["selected"]["note"],
        "prespecified_configuration": {
            "feature_set": cv["prespecified"]["feature_set"],
            "model": cv["prespecified"]["model"],
            "balanced_accuracy": cv["prespecified"]["balanced_accuracy"]["value"],
            "ci95": [
                cv["prespecified"]["balanced_accuracy"]["low"],
                cv["prespecified"]["balanced_accuracy"]["high"],
            ],
        },
    }
    if "in_cascade" in cv:
        cascade = cv["in_cascade"]["balanced_accuracy"]
        entry["in_cascade"] = {
            "balanced_accuracy": cascade["value"],
            "ci95": [cascade["low"], cascade["high"]],
            "permutation_p": cascade["p"],
        }
    return entry


def stage_result(
    model: StageModel, probability: float, x: np.ndarray | None, n_top: int, rule: ChanceConfig
) -> dict[str, Any]:
    label, uncertain = call(probability, model.threshold, model.band)
    result = {
        "stage": model.stage,
        "title": STAGE_TITLES[model.stage],
        "model": f"{model.metadata['model']} on {model.metadata['feature_set']}",
        "label": UNCERTAIN if uncertain else LABELS[model.stage][label],
        "probability": float(probability),
        "probability_of": model.metadata["classes"]["positive"]
        if model.stage != "stage2_paired"
        else "the first recording being OFF",
        "uncertain": uncertain,
        "threshold": model.threshold,
        "uncertain_band": list(model.band),
        "cross_validated": _validation(model.metadata["cross_validated"], rule),
    }
    if x is not None:
        result["top_features"] = contributions(model, x, n_top)
    return result


def _skipped(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "title": STAGE_TITLES[stage], "skipped": reason}


# --- recordings the models were trained on ------------------------------------------------------


EXPLANATION = (
    "This recording is one of the ds002778 recordings the final models were trained on, so "
    "their prediction for it would be a prediction on their own training data and would look "
    "better than it is. Shown instead are its out-of-fold predictions from cross-validation, "
    "made by models that never saw this participant."
)


def _oof(config: Config, stage: str) -> pd.DataFrame:
    path = config.predict.models_dir / stage / "oof_predictions.parquet"
    if not path.is_file():
        raise PredictError(f"{path} not found; run `pdeeg train-final` first")
    return pd.read_parquet(path)


def _metadata(config: Config, stage: str) -> dict[str, Any]:
    path = config.predict.models_dir / stage / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def trained_on(config: Config) -> dict[str, Any] | None:
    """The people and dataset stage 1 was trained on (every later stage uses some of them),
    or None while there is no model."""
    if not (config.predict.models_dir / "stage1" / "metadata.json").is_file():
        return None
    info = _metadata(config, "stage1")
    training = info["model_card"]["training_data"]
    return {
        "participants": training["participants"],
        "recordings": training["recordings"],
        "dataset": info["dataset"],
    }


def training_result(
    config: Config, matches: Sequence[dict[str, Any]], paired: bool
) -> dict[str, Any]:
    """The stored out-of-fold predictions of training recordings, stage by stage."""
    rule = config.predict.better_than_chance
    stages = []
    for match in matches:
        subject, session = match["subject"], match["session"]
        for stage in ("stage1", "stage2_single"):
            info = _metadata(config, stage)
            rows = _oof(config, stage)
            row = rows[(rows["subject"] == subject) & (rows["session"] == session)]
            if row.empty:
                stages.append(
                    _skipped(
                        stage,
                        f"sub-{subject} ses-{session} was not part of this stage's "
                        "cross-validation (a control's recording, or one left out)",
                    )
                    | {"recording": f"sub-{subject} ses-{session}"}
                )
                continue
            what = f"sub-{subject} ses-{session}"
            stages.append(_oof_entry(stage, info, row.iloc[0], what, rule))
    if paired:
        same = {m["subject"] for m in matches}
        info = _metadata(config, "stage2_paired")
        rows = _oof(config, "stage2_paired")
        first, second = matches
        row = rows[(rows["subject"] == first["subject"]) & (rows["session"] == first["session"])]
        if len(same) != 1 or first["session"] == second["session"] or row.empty:
            stages.append(
                _skipped(
                    "stage2_paired",
                    "the two recordings are not the OFF and ON recordings of one patient in "
                    "the paired cross-validation",
                )
            )
        else:
            stages.append(_oof_entry("stage2_paired", info, row.iloc[0], "the pair", rule))
    return {"training_recording": True, "explanation": EXPLANATION, "stages": stages}


def _oof_entry(
    stage: str, info: dict, row: pd.Series, what: str, rule: ChanceConfig
) -> dict[str, Any]:
    thresholds = info["thresholds"]
    probability = float(row["probability"])
    label, uncertain = call(probability, thresholds["threshold"], tuple(thresholds["uncertain"]))
    truth = LABELS[stage][int(row["label"])]
    return {
        "stage": stage,
        "title": STAGE_TITLES[stage],
        "recording": what,
        "out_of_fold": True,
        "fold": int(row["fold"]),
        "label": UNCERTAIN if uncertain else LABELS[stage][label],
        "probability": probability,
        "uncertain": uncertain,
        "true_label": truth,
        "threshold": thresholds["threshold"],
        "uncertain_band": thresholds["uncertain"],
        "cross_validated": _validation(info["cross_validated"], rule),
    }


# --- new recordings ------------------------------------------------------------------------------


def check_pair(recordings: Sequence[Recording]) -> None:
    """Refuse two inputs with one signal: a pair is a patient's two sessions."""
    if len(recordings) == 2 and recordings[0].signal_sha256 == recordings[1].signal_sha256:
        raise InputError("the two recordings have the same signal; --paired needs two sessions")


def check_inputs(config: Config, paths: Sequence[str | Path]) -> list[Recording]:
    """Every input read and checked (:func:`read_recording`), and two inputs as a pair."""
    if len(paths) not in (1, 2):
        raise InputError("give one recording, or two of the same patient")
    recordings = [read_recording(path, config) for path in paths]
    check_pair(recordings)
    return recordings


def predict_files(
    config: Config,
    paths: Sequence[str | Path],
    *,
    out_dir: Path | None = None,
    n_jobs: int = 1,
    say: Say = print,
) -> dict[str, Any]:
    """Check the inputs, then predict them (or show their out-of-fold results); see the module.

    ``paths`` is one recording, or two recordings of one patient (``--paired``). Work files
    (the cleaned data, its QC figures, the MFDFA runs) go under ``out_dir``.
    """
    recordings = check_inputs(config, paths)
    paired = len(recordings) == 2
    result: dict[str, Any] = {
        "disclaimer": config.predict.disclaimer,
        "inputs": [_input_entry(r) for r in recordings],
    }
    registry_path = config.predict.models_dir / REGISTRY
    if not registry_path.is_file():
        raise PredictError(f"{registry_path} not found; run `pdeeg train-final` first")
    registry = pd.read_parquet(registry_path)
    matches = [identify_training(r, registry, config) for r in recordings]
    for entry, match in zip(result["inputs"], matches, strict=True):
        entry["training_recording"] = (
            None if match is None else {key: match[key] for key in _MATCH_KEYS}
        )
    if any(matches):
        if not all(matches):
            raise InputError(
                "one recording is a training recording and the other is not; give them "
                "one at a time"
            )
        return result | training_result(config, matches, paired)

    stages = ["stage1", "stage2_single"] + (["stage2_paired"] if paired else [])
    models = {stage: load_stage(config, stage) for stage in stages}
    out_dir = Path(out_dir or config.predict.output_dir)
    n_top, rule = config.predict.n_top_features, config.predict.better_than_chance
    vectors = []
    for index, recording in enumerate(recordings):
        say(f"features of {recording.path.name} (cleaning, MFDFA, band power)")
        folder, name = out_dir / f"{recording.path.stem}_work", f"input{index + 1}"
        tables = recording_tables(
            recording.raw, config, folder, name=name, source=str(recording.path), n_jobs=n_jobs
        )
        files = work_files(config, folder, name)
        result["inputs"][index] |= {"cleaned": str(files["raw"]), "cleaning_log": str(files["log"])}
        vectors.append(
            {
                stage: feature_vector(
                    tables, config, models[stage].metadata["feature_set"], models[stage].columns
                )
                for stage in stages
            }
        )
    outputs = []
    for index, vector in enumerate(vectors):
        name = recordings[index].path.name
        first = models["stage1"]
        p1 = float(first.pipeline.predict_proba(vector["stage1"].reshape(1, -1))[0, 1])
        entry = stage_result(first, p1, vector["stage1"], n_top, rule) | {"recording": name}
        outputs.append(entry)
        if entry["label"] == LABELS["stage1"][1]:
            second = models["stage2_single"]
            x = vector["stage2_single"]
            p2 = float(second.pipeline.predict_proba(x.reshape(1, -1))[0, 1])
            outputs.append(stage_result(second, p2, x, n_top, rule) | {"recording": name})
        else:
            outputs.append(
                _skipped(
                    "stage2_single",
                    "stage 1 did not call this a patient's recording outside its uncertain band",
                )
                | {"recording": name}
            )
    if paired:
        model = models["stage2_paired"]
        pair = np.vstack([vector["stage2_paired"] for vector in vectors])
        centred = pair - pair.mean(axis=0)
        p = model.pipeline.predict_proba(centred)[:, 1]
        p_first = float(p[0] / p.sum())
        entry = stage_result(model, p_first, centred[0], n_top, rule) | {
            "recordings": [r.path.name for r in recordings]
        }
        called = [o for o in outputs if o["stage"] == "stage1"]
        if any(o["label"] != LABELS["stage1"][1] for o in called):
            entry["note"] = (
                "stage 1 did not call both recordings a patient's; the paired model assumes "
                "they are a patient's OFF and ON recordings"
            )
        outputs.append(entry)
    return result | {"training_recording": False, "stages": outputs}


def _input_entry(recording: Recording) -> dict[str, Any]:
    return {
        "path": str(recording.path),
        "sha256": recording.file_sha256,
        "signal_sha256": recording.signal_sha256,
        "seconds_from_crop_start": round(recording.available_s, 1),
        "median_amplitude_uv": round(recording.median_amplitude_uv, 3),
        "notes": recording.notes,
        "checks": [asdict(check) for check in recording.checks],
    }


# --- the readable summary ------------------------------------------------------------------------


def _ci(entry: dict[str, Any]) -> str:
    low, high = entry["ci95"]
    return f"{entry['balanced_accuracy']:.2f} [{low:.2f}, {high:.2f}]"


def summary_text(result: dict[str, Any]) -> str:
    """The result of :func:`predict_files` for people."""
    lines = [f"pdeeg predict - {result['disclaimer']}", ""]
    for entry in result["inputs"]:
        lines.append(f"Input: {entry['path']}")
        for check in entry["checks"]:
            lines.append(f"  {check['name']}: {check['status']} ({check['detail']})")
        if entry.get("training_recording"):
            match = entry["training_recording"]
            lines.append(
                f"  training recording sub-{match['subject']} ses-{match['session']} "
                f"(recognised by {match['matched_by']})"
            )
    lines.append("")
    if result.get("training_recording"):
        lines += [result["explanation"], ""]
    for stage in result["stages"]:
        where = stage.get("recording") or ", ".join(stage.get("recordings", []))
        lines.append(f"{stage['title']}" + (f" ({where})" if where else ""))
        if "skipped" in stage:
            lines += [f"  not run: {stage['skipped']}", ""]
            continue
        low, high = stage["uncertain_band"]
        lines.append(
            f"  {stage['label']}: probability {stage['probability']:.2f} (threshold "
            f"{stage['threshold']:g}; uncertain from {low:g} to {high:g})"
        )
        if stage.get("out_of_fold"):
            lines.append(f"  out of fold (fold {stage['fold']}); true label {stage['true_label']}")
        cv = stage["cross_validated"]
        lines.append(
            f"  cross-validated balanced accuracy {_ci(cv)}, chance {cv['chance']:g}, "
            f"permutation p = {cv['permutation_p']:.3g} (q = {cv['q']:.3g}): {cv['note']}"
        )
        if cv["better_than_chance"]:
            lines.append(f"  better than chance: {cv['test']}")
        else:
            lines.append(
                "  NOT BETTER THAN CHANCE: " + "; ".join(cv["not_better_because"]) + "; "
                f"the test is {cv['test']}"
            )
        pre = cv["prespecified_configuration"]
        lines.append(
            f"  configuration fixed in advance ({pre['model']}, {pre['feature_set']}): {_ci(pre)}"
        )
        if "in_cascade" in cv:
            lines.append(f"  inside the cascade: {_ci(cv['in_cascade'])}")
        if "top_features" not in stage:
            lines.append("")
            continue
        top = stage["top_features"]
        lines.append(f"  top features ({top['kind']}):")
        for feature in top["features"]:
            if "contribution" in feature:
                lines.append(
                    f"    {feature['feature']}: {feature['contribution']:+.2f} "
                    f"(towards {feature['towards']})"
                )
            else:
                lines.append(f"    {feature['feature']}: {feature['importance']:.3f}")
        if "note" in stage:
            lines.append(f"  note: {stage['note']}")
        lines.append("")
    lines.append(result["disclaimer"])
    return "\n".join(lines)
