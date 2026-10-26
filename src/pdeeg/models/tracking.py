"""Every evaluation as an MLflow run: parameters, what produced it, metrics and artifacts.

Runs go to the SQLite database and artifact directory of ``model.tracking``. Each run is tagged
with the digest of the configuration, the dataset and its version, the digests of the feature
tables, and the git commit, marked dirty when the working tree had uncommitted changes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pdeeg.config import Config
from pdeeg.models.cascade import CascadeEvaluation
from pdeeg.models.evaluate import METRICS, Evaluation


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def config_digest(config: Config) -> str:
    """SHA-256 of the model settings and the dataset they were run on."""
    payload = {
        "model": dataclasses.asdict(config.model),
        "dataset": dataclasses.asdict(config.data.dataset),
        "regions": config.data.regions,
    }
    # Where things are written does not change a result.
    for key in ("tracking", "report", "figures_dir", "n_jobs"):
        payload["model"].pop(key)
    return hashlib.sha256(json.dumps(_plain(payload), sort_keys=True).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(root: Path) -> dict[str, str]:
    """The commit checked out in ``root`` and whether the working tree differs from it."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": "unknown"}
    return {"git_commit": commit, "git_dirty": str(bool(status.strip())).lower()}


class Tracker:
    """Logs evaluations to the experiment of ``model.tracking``, creating it when needed."""

    def __init__(self, config: Config, inputs: Mapping[str, Path]):
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow

        self._mlflow = mlflow
        tracking = config.model.tracking
        tracking.db.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///" + tracking.db.as_posix())
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(tracking.experiment)
        self.experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else client.create_experiment(
                tracking.experiment, artifact_location=tracking.artifacts.resolve().as_uri()
            )
        )
        dataset = config.data.dataset
        self.tags = {
            "config_sha256": config_digest(config),
            "dataset": dataset.id,
            "dataset_version": dataset.version,
            **{f"{name}_table_sha256": file_digest(path) for name, path in inputs.items()},
            **git_state(config.root),
        }
        self.config = config

    def log(self, evaluation: Evaluation) -> str:
        """One run for one task, feature set and model; returns its id."""
        mlflow = self._mlflow
        model = self.config.model
        data = evaluation.data
        name = f"{evaluation.task}/{evaluation.feature_set}/{evaluation.model}"
        grid = getattr(model.classifiers, evaluation.model)
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=name) as run:
            mlflow.set_tags(
                self.tags
                | {
                    "task": evaluation.task,
                    "feature_set": evaluation.feature_set,
                    "model": evaluation.model,
                }
            )
            mlflow.log_params(
                {
                    "level": model.level,
                    "segments": model.segments,
                    "n_recordings": len(data.y),
                    "n_participants": len(np.unique(data.groups)),
                    "n_features": data.X.shape[1],
                    "paired": data.paired,
                    "within": data.within,
                    "weights": model.tasks[evaluation.task].weights,
                    "exclude_subjects": ",".join(model.tasks[evaluation.task].exclude_subjects),
                    "exclude_recordings": ",".join(
                        f"{subject}/{session}"
                        for subject, session in model.tasks[evaluation.task].exclude_recordings
                    ),
                    "left_out": ",".join(data.left_out),
                    "outer": model.cv.outer,
                    "outer_folds": model.cv.outer_folds,
                    "inner_folds": model.cv.inner_folds,
                    "inner_scoring": model.cv.inner_scoring,
                    "n_permutations": model.cv.n_permutations,
                    "n_bootstrap": model.cv.n_bootstrap,
                    "split_repeats": model.cv.split_repeats,
                    "grid": json.dumps(_plain(dataclasses.asdict(grid))),
                    "seed": evaluation.seed,
                }
            )
            values = {}
            for metric in METRICS:
                low, high = evaluation.intervals[metric]
                values |= {
                    metric: evaluation.observed[metric],
                    f"{metric}_ci_low": low,
                    f"{metric}_ci_high": high,
                    f"{metric}_p": evaluation.p_values[metric],
                }
            repeats = evaluation.repeats
            if repeats.size:
                values |= {
                    "split_repeat_ba_min": float(repeats.min()),
                    "split_repeat_ba_median": float(np.median(repeats)),
                    "split_repeat_ba_max": float(repeats.max()),
                }
            values["convergence_warnings"] = evaluation.predictions.convergence_warnings
            mlflow.log_metrics({key: float(value) for key, value in values.items()})
            with tempfile.TemporaryDirectory() as scratch:
                folder = Path(scratch)
                for file_name, table in artifacts(evaluation).items():
                    path = folder / file_name
                    if isinstance(table, pd.DataFrame):
                        table.to_parquet(path, index=False)
                    else:
                        path.write_text(json.dumps(_plain(table), indent=1), encoding="utf-8")
                    mlflow.log_artifact(str(path))
            return run.info.run_id

    def log_cascade(self, evaluation: CascadeEvaluation, files: list[Path]) -> str:
        """One run for the cascade, with its stored ``files`` as artifacts; returns its id."""
        mlflow = self._mlflow
        model = self.config.model
        cascade = model.cascade
        data = evaluation.data
        name = f"cascade/{evaluation.feature_set}/{evaluation.model}"
        grid = getattr(model.classifiers, evaluation.model)
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=name) as run:
            mlflow.set_tags(
                self.tags
                | {
                    "task": "cascade",
                    "feature_set": evaluation.feature_set,
                    "model": evaluation.model,
                }
            )
            mlflow.log_params(
                {
                    "stage1": cascade.stage1,
                    "stage2": cascade.stage2,
                    "classes": ",".join(data.classes),
                    "level": model.level,
                    "segments": model.segments,
                    "n_recordings": len(data.label),
                    "n_participants": len(np.unique(data.groups)),
                    "n_features": data.X.shape[1],
                    "left_out": ",".join(data.left_out),
                    "outer": model.cv.outer,
                    "outer_folds": model.cv.outer_folds,
                    "inner_folds": model.cv.inner_folds,
                    "inner_scoring": model.cv.inner_scoring,
                    "n_permutations": model.cv.n_permutations,
                    "n_bootstrap": model.cv.n_bootstrap,
                    "split_repeats": model.cv.split_repeats,
                    "calibration_method": cascade.calibration.method,
                    "grid": json.dumps(_plain(dataclasses.asdict(grid))),
                    "seed": evaluation.seed,
                }
            )
            values = {}
            for metric, value in evaluation.observed.items():
                low, high = evaluation.intervals[metric]
                values |= {metric: value, f"{metric}_ci_low": low, f"{metric}_ci_high": high}
            values |= {f"{metric}_p": p for metric, p in evaluation.p_values.items()}
            for column in evaluation.repeats.columns:
                values[f"split_repeat_{column}_median"] = float(evaluation.repeats[column].median())
            calibration = evaluation.calibration
            for stage in ("stage1", "stage2"):
                for version in ("raw", "platt"):
                    for metric, entry in calibration[stage][version]["summary"].items():
                        values[f"{stage}_{version}_{metric}"] = entry["value"]
                values[f"{stage}_poorly_calibrated"] = float(
                    bool(calibration[stage]["raw"]["reasons"])
                )
            for version, value in calibration["three_class_brier"].items():
                values[f"three_class_brier_{version}"] = value
            values["convergence_warnings"] = evaluation.predictions.convergence_warnings
            mlflow.log_metrics(
                {key: float(value) for key, value in values.items() if value is not None}
            )
            for path in files:
                mlflow.log_artifact(str(path))
            return run.info.run_id

    def log_final(self, stage: str, info: dict[str, Any], files: list[Path]) -> str:
        """One run for a final model (``pdeeg train-final``), its files as artifacts."""
        mlflow = self._mlflow
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"final/{stage}") as run:
            mlflow.set_tags(
                self.tags
                | {
                    "task": f"final/{stage}",
                    "feature_set": info["feature_set"],
                    "model": info["model"],
                }
            )
            low, high = info["thresholds"]["uncertain"]
            mlflow.log_params(
                {
                    "stage_task": info["task"],
                    "hyperparameters": json.dumps(_plain(info["hyperparameters"])),
                    "threshold": info["thresholds"]["threshold"],
                    "uncertain_low": low,
                    "uncertain_high": high,
                    "feature_sha256": info["config"]["feature_sha256"],
                }
            )
            selected = info["cross_validated"]["selected"]["balanced_accuracy"]
            values = {
                "cv_balanced_accuracy": selected["value"],
                "cv_balanced_accuracy_ci_low": selected["low"],
                "cv_balanced_accuracy_ci_high": selected["high"],
                "cv_balanced_accuracy_p": selected["p"],
                "convergence_warnings": info["convergence_warnings"],
                **{
                    f"oof_{name}": value
                    for name, value in info["thresholds"]["out_of_fold"].items()
                },
            }
            mlflow.log_metrics(
                {
                    key: float(value)
                    for key, value in values.items()
                    if value is not None and np.isfinite(value)
                }
            )
            for path in files:
                mlflow.log_artifact(str(path))
            return run.info.run_id

    def log_summary(self, files: list[Path], task: str = "summary") -> str:
        """A run holding a report and its figures."""
        mlflow = self._mlflow
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=task) as run:
            mlflow.set_tags(self.tags | {"task": task})
            for path in files:
                mlflow.log_artifact(str(path))
            return run.info.run_id


def artifacts(evaluation: Evaluation) -> dict[str, Any]:
    """The files of a run: predictions, the permutation null, weights per fold, choices."""
    data, predictions = evaluation.data, evaluation.predictions
    return {
        "predictions.parquet": data.rows.assign(
            weight=1.0 if data.weights is None else data.weights,
            label=data.y,
            score=predictions.score,
            predicted=predictions.predicted,
            fold=predictions.fold,
        ),
        "permutation_null.parquet": evaluation.null,
        "weights.parquet": pd.DataFrame(predictions.weights, columns=data.features).assign(
            fold=np.arange(len(predictions.weights))
        ),
        "split_repeats.json": evaluation.repeats.tolist(),
        "best_params.json": predictions.best_params,
    }
