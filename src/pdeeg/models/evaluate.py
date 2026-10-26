"""Nested cross-validation, its metrics, bootstrap intervals over people and permutation nulls.

The outer loop gives every recording a prediction from a model that never saw its participant;
the inner loop, inside each outer training set, picks the hyperparameters. All metrics are taken
on the pooled out-of-fold predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

from pdeeg.config import Config
from pdeeg.models.cv import (
    check_grouped,
    counting_warnings,
    outer_splitter,
    permuted_labels,
    platt,
    search,
    seed_for,
    weighted_fit,
)
from pdeeg.models.data import TaskData

METRICS = ("balanced_accuracy", "roc_auc", "sensitivity", "specificity")


@dataclass(frozen=True)
class Predictions:
    """Out-of-fold results of one nested cross-validation.

    ``score`` is the decision value of every row (higher: more likely positive), ``predicted``
    its predicted label and ``fold`` its outer fold. ``weights`` has one row per outer fold: the
    coefficients of a linear model on standardised features, or a boosting model's impurity
    importances (zero for features a selection left out). ``best_params`` lists what the inner
    loop chose in each fold. ``probability`` (the model's own, NaN for a model without one)
    and ``probability_platt`` (Platt-scaled inside each training set) are P(positive), filled
    only when the run calibrated.
    """

    score: np.ndarray
    predicted: np.ndarray
    fold: np.ndarray
    weights: np.ndarray
    best_params: list[dict[str, Any]]
    convergence_warnings: int
    probability: np.ndarray | None = None
    probability_platt: np.ndarray | None = None


def _weights(pipeline, n_features: int) -> np.ndarray:
    model = pipeline.named_steps["model"]
    values = model.coef_[0] if hasattr(model, "coef_") else model.feature_importances_
    values = np.asarray(values, dtype=float).ravel()
    if "select" not in pipeline.named_steps:
        return values
    full = np.zeros(n_features)
    full[pipeline.named_steps["select"].get_support()] = values
    return full


def fit_counting_warnings(
    estimator, X: np.ndarray, y: np.ndarray, groups: np.ndarray, weights: np.ndarray | None
) -> int:
    """Fit ``estimator`` (see :func:`weighted_fit`); returns how many fits did not converge."""
    return counting_warnings(lambda: weighted_fit(estimator, X, y, groups, weights))


def nested_cv(
    data: TaskData,
    model: str,
    config: Config,
    seed: int,
    y: np.ndarray | None = None,
    *,
    calibrate: bool = False,
) -> Predictions:
    """Out-of-fold predictions of ``model`` for every row of ``data``.

    ``y`` replaces the labels (for a permutation). The outer split is drawn with ``seed``, the
    inner ones with seeds derived from it. The weights of ``data``, if any, go into every fit
    and every inner-loop score. ``calibrate`` adds out-of-fold probabilities, raw and with
    Platt scaling learnt on each outer training set; the decisions do not change.
    """
    settings = config.model
    y = data.y if y is None else np.asarray(y)
    X, groups, sample_weight = data.X, data.groups, data.weights
    score = np.full(len(y), np.nan)
    predicted = np.full(len(y), -1)
    fold = np.full(len(y), -1)
    weights, best, n_warnings = [], [], 0
    probability = np.full(len(y), np.nan) if calibrate else None
    probability_platt = np.full(len(y), np.nan) if calibrate else None
    splits = outer_splitter(settings.cv, seed).split(X, y, groups)
    for k, (train, test) in enumerate(splits):
        check_grouped(train, test, groups)
        weighted = sample_weight is not None
        tuned = search(model, settings, X.shape[1], seed_for(seed, "inner", k), weighted=weighted)
        held = sample_weight[train] if weighted else None
        n_warnings += fit_counting_warnings(tuned, X[train], y[train], groups[train], held)
        score[test] = tuned.decision_function(X[test])
        predicted[test] = tuned.predict(X[test])
        fold[test] = k
        weights.append(_weights(tuned.best_estimator_, X.shape[1]))
        best.append(tuned.best_params_)
        if calibrate:
            if hasattr(tuned, "predict_proba"):
                probability[test] = tuned.predict_proba(X[test])[:, 1]
            scaled, count = platt(
                tuned, settings, X[train], y[train], groups[train], held, seed_for(seed, "platt", k)
            )
            n_warnings += count
            probability_platt[test] = scaled.predict_proba(X[test])[:, 1]
    if (fold < 0).any():
        raise RuntimeError("the outer loop left rows without a prediction")
    return Predictions(
        score,
        predicted,
        fold,
        np.array(weights),
        best,
        n_warnings,
        probability,
        probability_platt,
    )


def metrics(
    y: np.ndarray, score: np.ndarray, predicted: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, float]:
    """Balanced accuracy, ROC AUC, sensitivity (positives found) and specificity.

    With ``weights``, every row counts by its weight (each participant once, for a task
    weighted per participant).
    """
    y = np.asarray(y)
    both = len(set(y)) == 2
    w = {"sample_weight": weights}
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted, **w)) if both else np.nan,
        "roc_auc": float(roc_auc_score(y, score, **w)) if both else np.nan,
        "sensitivity": float(recall_score(y, predicted, pos_label=1, zero_division=np.nan, **w)),
        "specificity": float(recall_score(y, predicted, pos_label=0, zero_division=np.nan, **w)),
    }


def bootstrap_metrics(
    y: np.ndarray,
    score: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
    weights: np.ndarray | None = None,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals of :func:`metrics`, resampling participants with all their rows.

    The predictions stay as they are: the interval is about which participants happened to be
    in the sample, not about how the models would change with other training data. Rows keep
    their ``weights``.
    """
    names = np.unique(groups)
    rows = {name: np.flatnonzero(groups == name) for name in names}
    draws = []
    for _ in range(n_resamples):
        chosen = np.concatenate([rows[name] for name in rng.choice(names, size=names.size)])
        w = None if weights is None else weights[chosen]
        draws.append(metrics(y[chosen], score[chosen], predicted[chosen], w))
    table = pd.DataFrame(draws)
    tail = 100.0 * (1.0 - confidence) / 2.0
    return {
        name: tuple(float(v) for v in np.nanpercentile(table[name], [tail, 100.0 - tail]))
        for name in METRICS
    }


def _permutation(data: TaskData, model: str, config: Config, seed: int, i: int) -> dict:
    rng = np.random.default_rng(seed_for(seed, "permutation", i))
    y = permuted_labels(data.y, data.groups, data.within, rng)
    predictions = nested_cv(data, model, config, seed, y=y)
    return metrics(y, predictions.score, predictions.predicted, data.weights)


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
    return metrics(data.y, predictions.score, predictions.predicted, data.weights)[
        "balanced_accuracy"
    ]


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
    observed = metrics(data.y, predictions.score, predictions.predicted, data.weights)
    intervals = bootstrap_metrics(
        data.y,
        predictions.score,
        predictions.predicted,
        data.groups,
        cv.n_bootstrap,
        cv.confidence,
        np.random.default_rng(seed_for(seed, "bootstrap")),
        data.weights,
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
