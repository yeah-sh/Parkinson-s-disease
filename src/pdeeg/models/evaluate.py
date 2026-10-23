"""Nested cross-validation, its metrics, bootstrap intervals over people and permutation nulls.

The outer loop gives every recording a prediction from a model that never saw its participant;
the inner loop, inside each outer training set, picks the hyperparameters. All metrics are taken
on the pooled out-of-fold predictions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

from pdeeg.config import Config
from pdeeg.models.cv import check_grouped, outer_splitter, permuted_labels, search, seed_for
from pdeeg.models.data import TaskData

METRICS = ("balanced_accuracy", "roc_auc", "sensitivity", "specificity")


@dataclass(frozen=True)
class Predictions:
    """Out-of-fold results of one nested cross-validation.

    ``score`` is the decision value of every row (higher: more likely positive), ``predicted``
    its predicted label and ``fold`` its outer fold. ``weights`` has one row per outer fold: the
    coefficients of a linear model on standardised features, or a boosting model's impurity
    importances (zero for features a selection left out). ``best_params`` lists what the inner
    loop chose in each fold.
    """

    score: np.ndarray
    predicted: np.ndarray
    fold: np.ndarray
    weights: np.ndarray
    best_params: list[dict[str, Any]]
    convergence_warnings: int


def _weights(pipeline, n_features: int) -> np.ndarray:
    model = pipeline.named_steps["model"]
    values = model.coef_[0] if hasattr(model, "coef_") else model.feature_importances_
    values = np.asarray(values, dtype=float).ravel()
    if "select" not in pipeline.named_steps:
        return values
    full = np.zeros(n_features)
    full[pipeline.named_steps["select"].get_support()] = values
    return full


def nested_cv(
    data: TaskData, model: str, config: Config, seed: int, y: np.ndarray | None = None
) -> Predictions:
    """Out-of-fold predictions of ``model`` for every row of ``data``.

    ``y`` replaces the labels (for a permutation). The outer split is drawn with ``seed``, the
    inner ones with seeds derived from it.
    """
    settings = config.model
    y = data.y if y is None else np.asarray(y)
    X, groups = data.X, data.groups
    score = np.full(len(y), np.nan)
    predicted = np.full(len(y), -1)
    fold = np.full(len(y), -1)
    weights, best, n_warnings = [], [], 0
    splits = outer_splitter(settings.cv, seed).split(X, y, groups)
    for k, (train, test) in enumerate(splits):
        check_grouped(train, test, groups)
        tuned = search(model, settings, X.shape[1], seed_for(seed, "inner", k))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            tuned.fit(X[train], y[train], groups=groups[train])
        for caught_warning in caught:
            if issubclass(caught_warning.category, ConvergenceWarning):
                n_warnings += 1
            else:  # counted are only the convergence warnings; the rest go on as usual
                warnings.showwarning(
                    caught_warning.message,
                    caught_warning.category,
                    caught_warning.filename,
                    caught_warning.lineno,
                )
        score[test] = tuned.decision_function(X[test])
        predicted[test] = tuned.predict(X[test])
        fold[test] = k
        weights.append(_weights(tuned.best_estimator_, X.shape[1]))
        best.append(tuned.best_params_)
    if (fold < 0).any():
        raise RuntimeError("the outer loop left rows without a prediction")
    return Predictions(score, predicted, fold, np.array(weights), best, n_warnings)


def metrics(y: np.ndarray, score: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Balanced accuracy, ROC AUC, sensitivity (positives found) and specificity."""
    y = np.asarray(y)
    both = len(set(y)) == 2
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if both else np.nan,
        "roc_auc": float(roc_auc_score(y, score)) if both else np.nan,
        "sensitivity": float(recall_score(y, predicted, pos_label=1, zero_division=np.nan)),
        "specificity": float(recall_score(y, predicted, pos_label=0, zero_division=np.nan)),
    }


def bootstrap_metrics(
    y: np.ndarray,
    score: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals of :func:`metrics`, resampling participants with all their rows.

    The predictions stay as they are: the interval is about which participants happened to be
    in the sample, not about how the models would change with other training data.
    """
    names = np.unique(groups)
    rows = {name: np.flatnonzero(groups == name) for name in names}
    draws = []
    for _ in range(n_resamples):
        chosen = np.concatenate([rows[name] for name in rng.choice(names, size=names.size)])
        draws.append(metrics(y[chosen], score[chosen], predicted[chosen]))
    table = pd.DataFrame(draws)
    tail = 100.0 * (1.0 - confidence) / 2.0
    return {
        name: tuple(float(v) for v in np.nanpercentile(table[name], [tail, 100.0 - tail]))
        for name in METRICS
    }


def _permutation(data: TaskData, model: str, config: Config, seed: int, i: int) -> dict:
    rng = np.random.default_rng(seed_for(seed, "permutation", i))
    y = permuted_labels(data.y, data.groups, data.paired, rng)
    predictions = nested_cv(data, model, config, seed, y=y)
    return metrics(y, predictions.score, predictions.predicted)


def permutation_null(
    data: TaskData, model: str, config: Config, seed: int, n_permutations: int, n_jobs: int
) -> pd.DataFrame:
    """The metrics of the whole nested cross-validation on ``n_permutations`` relabellings."""
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_permutation)(data, model, config, seed, i) for i in range(n_permutations)
    )
    return pd.DataFrame(rows)


def permutation_p(observed: float, null: pd.Series) -> float:
    """Share of relabellings scoring at least ``observed``, the observed labelling counted in."""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    return float((1 + np.sum(null >= observed - 1e-12)) / (1 + null.size))


def _repeat(data: TaskData, model: str, config: Config, seed: int, r: int) -> float:
    predictions = nested_cv(data, model, config, seed_for(seed, "split repeat", r))
    return metrics(data.y, predictions.score, predictions.predicted)["balanced_accuracy"]


def split_repeats(
    data: TaskData, model: str, config: Config, seed: int, n_repeats: int, n_jobs: int
) -> np.ndarray:
    """Balanced accuracy of the nested cross-validation on ``n_repeats`` other outer splits."""
    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(_repeat)(data, model, config, seed, r) for r in range(n_repeats)
        )
    )


@dataclass(frozen=True)
class Evaluation:
    """Everything known about one task, feature set and model."""

    task: str
    feature_set: str
    model: str
    data: TaskData
    predictions: Predictions
    observed: dict[str, float]
    intervals: dict[str, tuple[float, float]]
    null: pd.DataFrame
    p_values: dict[str, float]
    repeats: np.ndarray
    seed: int


def evaluate(
    data: TaskData,
    task: str,
    feature_set: str,
    model: str,
    config: Config,
    *,
    n_jobs: int,
) -> Evaluation:
    """Nested cross-validation with intervals, a permutation null and other outer splits."""
    cv = config.model.cv
    seed = seed_for(config.model.random_state, task, feature_set, model)
    predictions = nested_cv(data, model, config, seed)
    observed = metrics(data.y, predictions.score, predictions.predicted)
    intervals = bootstrap_metrics(
        data.y,
        predictions.score,
        predictions.predicted,
        data.groups,
        cv.n_bootstrap,
        cv.confidence,
        np.random.default_rng(seed_for(seed, "bootstrap")),
    )
    null = permutation_null(data, model, config, seed, cv.n_permutations, n_jobs)
    p_values = {name: permutation_p(observed[name], null[name]) for name in METRICS}
    repeats = split_repeats(data, model, config, seed, cv.split_repeats, n_jobs)
    return Evaluation(
        task=task,
        feature_set=feature_set,
        model=model,
        data=data,
        predictions=predictions,
        observed=observed,
        intervals=intervals,
        null=null,
        p_values=p_values,
        repeats=repeats,
        seed=seed,
    )
