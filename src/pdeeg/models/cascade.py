"""The two-stage cascade: patient or control, then (for a patient) OFF or ON, on one recording.

In every outer fold (grouped by participant, stratified by patient or control) stage 1 and
stage 2 are each tuned by their own grouped inner loop and fitted on the training participants
alone: stage 1 on all their recordings (task ``cascade.stage1``, weighted as that task says),
stage 2 on their patients' recordings (task ``cascade.stage2``, with its exclusions). Every
held-out recording is then called by stage 1 and, unless that says control, by stage 2. Stage 2
is applied to every held-out recording, so that its errors on true patients can be told apart
from stage 1's.

Labels are integers: 0 for stage 1's negative side (HC), 1 for stage 2's positive side (PD_OFF)
and 2 for its negative side (PD_ON).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pdeeg.config import Config
from pdeeg.models.calibration import calibration_summary, miscalibration, multiclass_brier
from pdeeg.models.cv import check_grouped, outer_splitter, platt, search, seed_for
from pdeeg.models.data import TaskError, participant_weights, task_data, task_rows
from pdeeg.models.evaluate import (
    fit_counting_warnings,
    metrics,
    permutation_p,
)

# Metrics with a permutation p-value, and their chance levels.
CHANCE = {
    "macro_balanced_accuracy": 1.0 / 3.0,
    "stage1_balanced_accuracy": 0.5,
    "stage1_roc_auc": 0.5,
    "stage2_balanced_accuracy": 0.5,
    "stage2_roc_auc": 0.5,
}
# Reported for the other outer splits.
REPEATED = ("macro_balanced_accuracy", "stage1_balanced_accuracy", "stage2_balanced_accuracy")
FILES = (
    "predictions.parquet",
    "permutation_null.parquet",
    "split_repeats.parquet",
    "evaluation.json",
)


@dataclass(frozen=True)
class CascadeData:
    """The recordings of the cascade.

    ``X`` holds the features of ``cascade.feature_set``, ``label`` is 0, 1 or 2 (``classes``
    names them) and ``groups`` the participant of every row. ``weights`` are stage 1's (None:
    every recording counts once). ``stage2_allowed`` marks the recordings stage 2 may train on
    when they are a patient's; it follows from stage 2's exclusions alone, never from a label.
    """

    X: np.ndarray
    label: np.ndarray
    groups: np.ndarray
    weights: np.ndarray | None
    stage2_allowed: np.ndarray
    rows: pd.DataFrame
    features: list[str]
    classes: tuple[str, str, str]
    left_out: list[str]


def cascade_data(
    config: Config, recordings: pd.DataFrame, tables: Mapping[str, pd.DataFrame]
) -> CascadeData:
    """The recordings of stage 1's task, labelled with the three classes."""
    model = config.model
    cascade = model.cascade
    first = task_data(config, recordings, tables, cascade.stage1, cascade.feature_set)
    second = model.tasks[cascade.stage2]
    rows = first.rows
    label = np.select(
        [rows["condition"].isin(second.positive), rows["condition"].isin(second.negative)],
        [1, 2],
        default=0,
    )
    dropped = set(second.exclude_recordings)
    allowed = ~rows["subject"].isin(second.exclude_subjects).to_numpy() & ~np.array(
        [(s, t) in dropped for s, t in zip(rows["subject"], rows["session"], strict=True)]
    )
    # Stage 2 trains on exactly the recordings its own task has, among these participants.
    own = task_rows(config, recordings, cascade.stage2)
    own = own[own[model.cv.group_key].isin(first.groups)]
    used = rows.loc[(label != 0) & allowed, ["subject", "session"]].to_numpy()
    if sorted(map(tuple, used)) != sorted(map(tuple, own[["subject", "session"]].to_numpy())):
        raise TaskError(f"cascade: stage 2 rows differ from those of task {cascade.stage2}")
    negative = model.tasks[cascade.stage1].negative
    return CascadeData(
        X=first.X,
        label=label,
        groups=first.groups,
        weights=first.weights,
        stage2_allowed=allowed,
        rows=rows,
        features=first.features,
        classes=("/".join(negative), "/".join(second.positive), "/".join(second.negative)),
        left_out=first.left_out,
    )


@dataclass(frozen=True)
class CascadePredictions:
    """Out-of-fold results of one run of the cascade.

    ``p_stage1`` is P(patient) and ``p_stage2`` P(stage 2 positive, OFF) for every recording,
    ``stage2`` stage 2's call (1: OFF) and ``predicted`` the cascade's (0, 1 or 2). The
    ``_platt`` probabilities are NaN unless the run calibrated. ``folds`` lists every outer
    fold's test participants and the participants each stage was trained on.
    """

    p_stage1: np.ndarray
    p_stage2: np.ndarray
    stage2: np.ndarray
    predicted: np.ndarray
    fold: np.ndarray
    p_stage1_platt: np.ndarray
    p_stage2_platt: np.ndarray
    folds: list[dict[str, Any]]
    best_params: list[dict[str, Any]]
    convergence_warnings: int


def run_cascade(
    data: CascadeData,
    config: Config,
    seed: int,
    label: np.ndarray | None = None,
    *,
    calibrate: bool = False,
) -> CascadePredictions:
    """Out-of-fold calls of the cascade for every recording of ``data``.

    ``label`` replaces the labels (for a permutation). The outer split is drawn with ``seed``,
    the inner loops of both stages with seeds derived from it.
    """
    settings = config.model
    name = settings.cascade.model
    second = settings.tasks[settings.cascade.stage2]
    label = data.label if label is None else np.asarray(label)
    X, groups = data.X, data.groups
    patient = (label != 0).astype(int)
    n = len(label)
    p1, p2, p1_platt, p2_platt = (np.full(n, np.nan) for _ in range(4))
    first_call, second_call, fold = (np.full(n, -1) for _ in range(3))
    folds, best, n_warnings = [], [], 0
    for k, (train, test) in enumerate(outer_splitter(settings.cv, seed).split(X, patient, groups)):
        check_grouped(train, test, groups)
        train2 = train[(label[train] != 0) & data.stage2_allowed[train]]
        check_grouped(train2, test, groups)
        y2 = (label[train2] == 1).astype(int)
        w1 = None if data.weights is None else data.weights[train]
        w2 = participant_weights(groups[train2]) if second.weights == "per_participant" else None
        stages = []
        for stage, rows, y, w in (
            ("stage 1", train, patient[train], w1),
            ("stage 2", train2, y2, w2),
        ):
            tuned = search(
                name,
                settings,
                X.shape[1],
                seed_for(seed, stage, "inner", k),
                weighted=w is not None,
            )
            n_warnings += fit_counting_warnings(tuned, X[rows], y, groups[rows], w)
            stages.append((stage, rows, y, w, tuned))
        (_, _, _, _, stage1), (_, _, _, _, stage2) = stages
        p1[test] = stage1.predict_proba(X[test])[:, 1]
        first_call[test] = stage1.predict(X[test])
        p2[test] = stage2.predict_proba(X[test])[:, 1]
        second_call[test] = stage2.predict(X[test])
        if calibrate:
            for stage, rows, y, w, tuned in stages:
                platt_seed = seed_for(seed, stage, "platt", k)
                scaled, count = platt(
                    tuned,
                    settings,
                    X[rows],
                    y,
                    groups[rows],
                    w,
                    platt_seed,
                    settings.cascade.calibration.method,
                )
                n_warnings += count
                target = p1_platt if stage == "stage 1" else p2_platt
                target[test] = scaled.predict_proba(X[test])[:, 1]
        fold[test] = k
        folds.append(
            {
                "fold": k,
                "test": sorted(map(str, set(groups[test]))),
                "stage1_train": sorted(map(str, set(groups[train]))),
                "stage2_train": sorted(map(str, set(groups[train2]))),
            }
        )
        best.append({"stage1": stage1.best_params_, "stage2": stage2.best_params_})
    if (fold < 0).any():
        raise RuntimeError("the outer loop left recordings without a call")
    predicted = np.where(first_call == 0, 0, np.where(second_call == 1, 1, 2))
    return CascadePredictions(
        p_stage1=p1,
        p_stage2=p2,
        stage2=second_call,
        predicted=predicted,
        fold=fold,
        p_stage1_platt=p1_platt,
        p_stage2_platt=p2_platt,
        folds=folds,
        best_params=best,
        convergence_warnings=n_warnings,
    )


# --- metrics ---------------------------------------------------------------------------------


def confusion(label: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """3 x 3 counts of recordings: rows the true class, columns the call."""
    counts = np.zeros((3, 3), dtype=int)
    np.add.at(counts, (np.asarray(label), np.asarray(predicted)), 1)
    return counts


def cascade_metrics(
    label: np.ndarray,
    predicted: np.ndarray,
    stage2: np.ndarray,
    p_stage1: np.ndarray,
    p_stage2: np.ndarray,
    weights: np.ndarray | None,
    classes: tuple[str, str, str],
) -> dict[str, float]:
    """The cascade's metrics, and those of each stage on its own part of the job.

    - ``sensitivity_<class>``: share of a class's recordings called that class; their mean is
      ``macro_balanced_accuracy`` (chance 1/3).
    - ``stage1_*``: patient or control, weighted as stage 1 (each participant once).
    - ``stage2_*``: OFF or ON on the true patients' recordings, whatever stage 1 said.
    - ``pd_missed``, ``pd_wrong_state``, ``pd_correct``: where the patients' recordings went;
      ``hc_called_pd``: share of controls' recordings called a patient's.
    """
    label, predicted, stage2 = np.asarray(label), np.asarray(predicted), np.asarray(stage2)
    values = {}
    for index, name in enumerate(classes):
        rows = label == index
        values[f"sensitivity_{name}"] = (
            float(np.mean(predicted[rows] == index)) if rows.any() else np.nan
        )
    recalls = [values[f"sensitivity_{name}"] for name in classes]
    values["macro_balanced_accuracy"] = (
        float(np.nanmean(recalls)) if np.isfinite(recalls).any() else np.nan
    )
    patient = (label != 0).astype(int)
    first = metrics(patient, p_stage1, (predicted != 0).astype(int), weights)
    values["stage1_balanced_accuracy"] = first["balanced_accuracy"]
    values["stage1_roc_auc"] = first["roc_auc"]
    ill = label != 0
    if ill.any():
        second = metrics((label[ill] == 1).astype(int), p_stage2[ill], stage2[ill])
        values["stage2_balanced_accuracy"] = second["balanced_accuracy"]
        values["stage2_roc_auc"] = second["roc_auc"]
        values["pd_missed"] = float(np.mean(predicted[ill] == 0))
        values["pd_wrong_state"] = float(
            np.mean((predicted[ill] != 0) & (predicted[ill] != label[ill]))
        )
        values["pd_correct"] = float(np.mean(predicted[ill] == label[ill]))
    else:
        missing = ("stage2_balanced_accuracy", "stage2_roc_auc", "pd_missed", "pd_wrong_state")
        values |= dict.fromkeys((*missing, "pd_correct"), np.nan)
    healthy = label == 0
    values["hc_called_pd"] = float(np.mean(predicted[healthy] != 0)) if healthy.any() else np.nan
    return values


def bootstrap_cascade(
    label: np.ndarray,
    predicted: np.ndarray,
    stage2: np.ndarray,
    p_stage1: np.ndarray,
    p_stage2: np.ndarray,
    weights: np.ndarray | None,
    groups: np.ndarray,
    classes: tuple[str, str, str],
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals of :func:`cascade_metrics`, resampling participants with all their
    recordings; the calls stay as they are."""
    names = np.unique(groups)
    rows = {name: np.flatnonzero(groups == name) for name in names}
    draws = []
    for _ in range(n_resamples):
        chosen = np.concatenate([rows[name] for name in rng.choice(names, size=names.size)])
        w = None if weights is None else weights[chosen]
        draws.append(
            cascade_metrics(
                label[chosen],
                predicted[chosen],
                stage2[chosen],
                p_stage1[chosen],
                p_stage2[chosen],
                w,
                classes,
            )
        )
    table = pd.DataFrame(draws)
    tail = 100.0 * (1.0 - confidence) / 2.0
    return {
        name: tuple(float(v) for v in np.nanpercentile(table[name], [tail, 100.0 - tail]))
        for name in table.columns
    }


def permuted_cascade_labels(
    label: np.ndarray, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Three-class labels relabelled at the level of participants, as the null allows.

    Patient or control is shuffled among the participants, whole participants at a time. Every
    participant who is then a patient gets OFF and ON by a coin flip: one of each if they have
    two recordings, OFF or ON for each recording otherwise.
    """
    label, groups = np.asarray(label), np.asarray(groups)
    names = np.unique(groups)
    patient = {}
    for name in names:
        held = label[groups == name] != 0
        if len(set(held)) > 1:
            raise ValueError(f"participant {name} is both a patient and a control")
        patient[name] = bool(held[0])
    shuffled = dict(zip(names, rng.permutation([patient[name] for name in names]), strict=True))
    permuted = np.zeros_like(label)
    for name in names:
        rows = np.flatnonzero(groups == name)
        if not shuffled[name]:
            continue
        if rows.size == 2:
            permuted[rows] = rng.permutation([1, 2])
        else:
            permuted[rows] = rng.choice([1, 2], size=rows.size)
    return permuted


def _scored(data: CascadeData, predictions: CascadePredictions, label=None) -> dict[str, float]:
    return cascade_metrics(
        data.label if label is None else label,
        predictions.predicted,
        predictions.stage2,
        predictions.p_stage1,
        predictions.p_stage2,
        data.weights,
        data.classes,
    )


def _permutation(data: CascadeData, config: Config, seed: int, i: int) -> dict[str, float]:
    rng = np.random.default_rng(seed_for(seed, "permutation", i))
    label = permuted_cascade_labels(data.label, data.groups, rng)
    return _scored(data, run_cascade(data, config, seed, label=label), label)


def _repeat(data: CascadeData, config: Config, seed: int, r: int) -> dict[str, float]:
    scored = _scored(data, run_cascade(data, config, seed_for(seed, "split repeat", r)))
    return {name: scored[name] for name in REPEATED}


# --- calibration -----------------------------------------------------------------------------


def three_class_probabilities(p_stage1: np.ndarray, p_stage2: np.ndarray) -> np.ndarray:
    """P(HC), P(OFF), P(ON) of the cascade: 1 - p1, p1 p2 and p1 (1 - p2)."""
    p1, p2 = np.asarray(p_stage1), np.asarray(p_stage2)
    return np.column_stack([1.0 - p1, p1 * p2, p1 * (1.0 - p2)])


def stage_probabilities(
    data: CascadeData, predictions: CascadePredictions, version: str
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]:
    """Per stage: labels, probabilities (``raw`` or ``platt``), participants and weights.

    Stage 1 on every recording, weighted as stage 1; stage 2 on the true patients' recordings.
    """
    suffix = "" if version == "raw" else "_platt"
    p1 = getattr(predictions, f"p_stage1{suffix}")
    p2 = getattr(predictions, f"p_stage2{suffix}")
    ill = data.label != 0
    return {
        "stage1": ((data.label != 0).astype(int), p1, data.groups, data.weights),
        "stage2": ((data.label[ill] == 1).astype(int), p2[ill], data.groups[ill], None),
    }


def assess_calibration(
    data: CascadeData, predictions: CascadePredictions, config: Config, seed: int
) -> dict[str, Any]:
    """Brier scores, intercept and slope of both stages, raw and Platt-scaled, with the rule.

    ``<stage>.use`` is ``platt`` when the rule finds the raw probabilities poorly calibrated.
    ``platt_cascade`` holds the metrics of the cascade called from the Platt-scaled
    probabilities (at 0.5), reported only as secondary.
    """
    cv = config.model.cv
    result: dict[str, Any] = {}
    for version in ("raw", "platt"):
        for stage, (y, p, groups, w) in stage_probabilities(data, predictions, version).items():
            rng = np.random.default_rng(seed_for(seed, "calibration", stage, version))
            summary = calibration_summary(y, p, groups, w, cv.n_bootstrap, cv.confidence, rng)
            result.setdefault(stage, {})[version] = {
                "summary": summary,
                "reasons": miscalibration(summary),
            }
    for stage in ("stage1", "stage2"):
        result[stage]["use"] = "platt" if result[stage]["raw"]["reasons"] else "raw"
    prevalence = np.bincount(data.label, minlength=3) / len(data.label)
    result["three_class_brier"] = {
        "raw": multiclass_brier(
            three_class_probabilities(predictions.p_stage1, predictions.p_stage2), data.label
        ),
        "platt": multiclass_brier(
            three_class_probabilities(predictions.p_stage1_platt, predictions.p_stage2_platt),
            data.label,
        ),
        "reference": multiclass_brier(np.tile(prevalence, (len(data.label), 1)), data.label),
    }
    stage2 = (predictions.p_stage2_platt > 0.5).astype(int)
    predicted = np.where(predictions.p_stage1_platt > 0.5, np.where(stage2 == 1, 1, 2), 0)
    args = (
        data.label,
        predicted,
        stage2,
        predictions.p_stage1_platt,
        predictions.p_stage2_platt,
        data.weights,
    )
    result["platt_cascade"] = {
        "observed": cascade_metrics(*args, data.classes),
        "intervals": bootstrap_cascade(
            *args,
            data.groups,
            data.classes,
            cv.n_bootstrap,
            cv.confidence,
            np.random.default_rng(seed_for(seed, "platt cascade bootstrap")),
        ),
        "confusion": confusion(data.label, predicted).tolist(),
    }
    return result


# --- the evaluation --------------------------------------------------------------------------


@dataclass(frozen=True)
class CascadeEvaluation:
    """Everything known about the cascade."""

    feature_set: str
    model: str
    data: CascadeData
    predictions: CascadePredictions
    observed: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    null: pd.DataFrame
    p_values: dict[str, float]
    repeats: pd.DataFrame
    calibration: dict[str, Any]
    seed: int

    @property
    def confusion(self) -> np.ndarray:
        return confusion(self.data.label, self.predictions.predicted)


def evaluate_cascade(data: CascadeData, config: Config, *, n_jobs: int) -> CascadeEvaluation:
    """The cascade with intervals, a permutation null, other outer splits and calibration."""
    cascade, cv = config.model.cascade, config.model.cv
    seed = seed_for(config.model.random_state, "cascade", cascade.feature_set, cascade.model)
    predictions = run_cascade(data, config, seed, calibrate=True)
    observed = _scored(data, predictions)
    intervals = bootstrap_cascade(
        data.label,
        predictions.predicted,
        predictions.stage2,
        predictions.p_stage1,
        predictions.p_stage2,
        data.weights,
        data.groups,
        data.classes,
        cv.n_bootstrap,
        cv.confidence,
        np.random.default_rng(seed_for(seed, "bootstrap")),
    )
    null = pd.DataFrame(
        Parallel(n_jobs=n_jobs)(
            delayed(_permutation)(data, config, seed, i) for i in range(cv.n_permutations)
        )
    )
    p_values = {name: permutation_p(observed[name], null[name]) for name in CHANCE}
    repeats = pd.DataFrame(
        Parallel(n_jobs=n_jobs)(
            delayed(_repeat)(data, config, seed, r) for r in range(cv.split_repeats)
        ),
        columns=list(REPEATED),
    )
    return CascadeEvaluation(
        feature_set=cascade.feature_set,
        model=cascade.model,
        data=data,
        predictions=predictions,
        observed=observed,
        intervals=intervals,
        null=null,
        p_values=p_values,
        repeats=repeats,
        calibration=assess_calibration(data, predictions, config, seed),
        seed=seed,
    )


# --- on disk ---------------------------------------------------------------------------------


class CascadeStoreError(Exception):
    """The stored cascade is missing or does not match the data it is said to come from."""


def cascade_dir(config: Config) -> Path:
    cascade = config.model.cascade
    return config.data.paths.processed / "cascade" / cascade.feature_set / cascade.model


def stored_cascade_files(config: Config) -> list[Path]:
    return [cascade_dir(config) / name for name in FILES]


def predictions_table(result: CascadeEvaluation) -> pd.DataFrame:
    """A row per recording: who, the true class, both stages' probabilities and calls, fold."""
    data, predictions = result.data, result.predictions
    return data.rows.assign(
        group=data.groups,
        weight=1.0 if data.weights is None else data.weights,
        stage2_allowed=data.stage2_allowed,
        label=data.label,
        p_stage1=predictions.p_stage1,
        p_stage2=predictions.p_stage2,
        p_stage1_platt=predictions.p_stage1_platt,
        p_stage2_platt=predictions.p_stage2_platt,
        stage2=predictions.stage2,
        predicted=predictions.predicted,
        fold=predictions.fold,
    )


def _summary(result: CascadeEvaluation) -> dict[str, Any]:
    return {
        "feature_set": result.feature_set,
        "model": result.model,
        "seed": result.seed,
        "classes": list(result.data.classes),
        "observed": result.observed,
        "intervals": {name: list(pair) for name, pair in result.intervals.items()},
        "p_values": result.p_values,
        "confusion": result.confusion.tolist(),
        "calibration": result.calibration,
        "folds": result.predictions.folds,
        "best_params": result.predictions.best_params,
        "convergence_warnings": result.predictions.convergence_warnings,
        "features": result.data.features,
        "left_out": result.data.left_out,
    }


def save_cascade(config: Config, result: CascadeEvaluation) -> Path:
    folder = cascade_dir(config)
    folder.mkdir(parents=True, exist_ok=True)
    predictions_table(result).to_parquet(folder / "predictions.parquet", index=False)
    result.null.to_parquet(folder / "permutation_null.parquet", index=False)
    result.repeats.to_parquet(folder / "split_repeats.parquet", index=False)
    (folder / "evaluation.json").write_text(json.dumps(_summary(result), indent=1), "utf-8")
    return folder


def load_cascade(config: Config, data: CascadeData) -> CascadeEvaluation:
    """The stored cascade, for the recordings of ``data``."""
    folder = cascade_dir(config)
    missing = [name for name in FILES if not (folder / name).is_file()]
    if missing:
        raise CascadeStoreError(f"{folder}: no {', '.join(missing)}; run `pdeeg cascade` first")
    summary = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    rows = pd.read_parquet(folder / "predictions.parquet")
    expected = data.rows[["subject", "session"]].reset_index(drop=True)
    if (
        summary["features"] != data.features
        or not rows[["subject", "session"]].equals(expected)
        or not np.array_equal(rows["label"].to_numpy(), data.label)
    ):
        raise CascadeStoreError(
            f"{folder}: stored for other data than the feature tables now give; run "
            "`pdeeg cascade --force`"
        )
    predictions = CascadePredictions(
        p_stage1=rows["p_stage1"].to_numpy(),
        p_stage2=rows["p_stage2"].to_numpy(),
        stage2=rows["stage2"].to_numpy(),
        predicted=rows["predicted"].to_numpy(),
        fold=rows["fold"].to_numpy(),
        p_stage1_platt=rows["p_stage1_platt"].to_numpy(),
        p_stage2_platt=rows["p_stage2_platt"].to_numpy(),
        folds=summary["folds"],
        best_params=summary["best_params"],
        convergence_warnings=summary["convergence_warnings"],
    )
    return CascadeEvaluation(
        feature_set=summary["feature_set"],
        model=summary["model"],
        data=data,
        predictions=predictions,
        observed=summary["observed"],
        intervals={name: tuple(pair) for name, pair in summary["intervals"].items()},
        null=pd.read_parquet(folder / "permutation_null.parquet"),
        p_values=summary["p_values"],
        repeats=pd.read_parquet(folder / "split_repeats.parquet"),
        calibration=summary["calibration"],
        seed=summary["seed"],
    )
