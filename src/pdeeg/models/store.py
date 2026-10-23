"""Evaluations on disk, so that the report can be written again without running anything.

Each evaluation is a directory ``<processed>/classification/<task>/<feature set>/<model>/``:

- ``predictions.parquet``: every recording with its label, score, predicted label and fold;
- ``permutation_null.parquet``: the metrics of every relabelling;
- ``weights.parquet``: the weights of each outer fold's model, a column per feature;
- ``evaluation.json``: the metrics, their intervals and p-values, the balanced accuracy on the
  other outer splits, what the inner loop chose, and the seed.

The data a classifier saw is not stored: it is taken from the feature tables again, and checked
against the stored predictions row by row.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from pdeeg.config import Config
from pdeeg.models.data import task_data
from pdeeg.models.evaluate import METRICS, Evaluation, Predictions

FILES = ("predictions.parquet", "permutation_null.parquet", "weights.parquet", "evaluation.json")


class StoreError(Exception):
    """A stored evaluation is missing or does not match the data it is said to come from."""


def evaluation_dir(config: Config, task: str, feature_set: str, model: str) -> Path:
    return config.data.paths.processed / "classification" / task / feature_set / model


def stored_files(config: Config, plan: Sequence[tuple[str, str, str]]) -> list[Path]:
    """Every file the evaluations of ``plan`` are stored in."""
    return [evaluation_dir(config, *key) / name for key in plan for name in FILES]


def save_evaluation(config: Config, result: Evaluation) -> Path:
    folder = evaluation_dir(config, result.task, result.feature_set, result.model)
    folder.mkdir(parents=True, exist_ok=True)
    data, predictions = result.data, result.predictions
    data.rows.assign(
        group=data.groups,
        label=data.y,
        score=predictions.score,
        predicted=predictions.predicted,
        fold=predictions.fold,
    ).to_parquet(folder / "predictions.parquet", index=False)
    result.null.to_parquet(folder / "permutation_null.parquet", index=False)
    pd.DataFrame(predictions.weights, columns=data.features).to_parquet(
        folder / "weights.parquet", index=False
    )
    summary = {
        "task": result.task,
        "feature_set": result.feature_set,
        "model": result.model,
        "seed": result.seed,
        "observed": result.observed,
        "intervals": {name: list(pair) for name, pair in result.intervals.items()},
        "p_values": result.p_values,
        "repeats": [float(value) for value in result.repeats],
        "best_params": predictions.best_params,
        "convergence_warnings": predictions.convergence_warnings,
        "features": data.features,
        "left_out": data.left_out,
    }
    (folder / "evaluation.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return folder


def save_evaluations(config: Config, results: Sequence[Evaluation]) -> list[Path]:
    return [save_evaluation(config, result) for result in results]


def load_evaluation(config: Config, key: tuple[str, str, str], data) -> Evaluation:
    """The stored evaluation ``key`` (task, feature set, model) of the task data ``data``."""
    task, feature_set, model = key
    folder = evaluation_dir(config, *key)
    missing = [name for name in FILES if not (folder / name).is_file()]
    if missing:
        raise StoreError(f"{folder}: no {', '.join(missing)}; run `pdeeg classify` first")
    summary = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    rows = pd.read_parquet(folder / "predictions.parquet")
    expected = data.rows[["subject", "session"]].reset_index(drop=True)
    if (
        summary["features"] != data.features
        or not rows[["subject", "session"]].equals(expected)
        or not np.array_equal(rows["label"].to_numpy(), data.y)
    ):
        raise StoreError(
            f"{folder}: stored for other data than the feature tables now give; run "
            "`pdeeg classify --force`"
        )
    weights = pd.read_parquet(folder / "weights.parquet")[data.features].to_numpy()
    predictions = Predictions(
        score=rows["score"].to_numpy(),
        predicted=rows["predicted"].to_numpy(),
        fold=rows["fold"].to_numpy(),
        weights=weights,
        best_params=summary["best_params"],
        convergence_warnings=summary["convergence_warnings"],
    )
    return Evaluation(
        task=task,
        feature_set=feature_set,
        model=model,
        data=data,
        predictions=predictions,
        observed={name: summary["observed"][name] for name in METRICS},
        intervals={name: tuple(summary["intervals"][name]) for name in METRICS},
        null=pd.read_parquet(folder / "permutation_null.parquet"),
        p_values={name: summary["p_values"][name] for name in METRICS},
        repeats=np.asarray(summary["repeats"], dtype=float),
        seed=summary["seed"],
    )


def load_evaluations(
    config: Config,
    recordings: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    plan: Sequence[tuple[str, str, str]],
) -> list[Evaluation]:
    """Every stored evaluation of ``plan``, with its task data rebuilt from the tables."""
    data = {}
    results = []
    for task, feature_set, model in plan:
        if (task, feature_set) not in data:
            data[task, feature_set] = task_data(config, recordings, tables, task, feature_set)
        results.append(load_evaluation(config, (task, feature_set, model), data[task, feature_set]))
    return results
