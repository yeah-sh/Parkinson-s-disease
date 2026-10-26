"""Splits by participant, the inner-loop search, and label permutations that respect the design.

No split here ever puts one participant's recordings on both sides: the splitters are grouped,
and :func:`check_grouped` is asserted on every outer and inner split the evaluation makes.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pdeeg.config import CvConfig, ModelConfig

MODELS = ("logistic_regression", "linear_svm", "gradient_boosting")


class LeakageError(AssertionError):
    """A participant is on both sides of a split."""


def seed_for(seed: int, *key: object) -> int:
    """A 32-bit seed that depends on ``seed`` and the names in ``key``, and nothing else."""
    digest = hashlib.sha256("\x1f".join(map(str, key)).encode("utf-8")).digest()
    sequence = np.random.SeedSequence([seed, int.from_bytes(digest[:16], "big")])
    return int(sequence.generate_state(1)[0])


def outer_splitter(cv: CvConfig, seed: int) -> BaseCrossValidator:
    """The outer loop: stratified k-fold grouped by participant, or leave one participant out."""
    if cv.outer == "leave_one_subject_out":
        return LeaveOneGroupOut()
    return StratifiedGroupKFold(n_splits=cv.outer_folds, shuffle=True, random_state=seed)


class CheckedSplitter(BaseCrossValidator):
    """A grouped splitter that raises :class:`LeakageError` if a split ever shares a group."""

    # With metadata routing (the weighted tasks), a search passes groups only to a splitter that
    # asks for them; without it, groups are always passed.
    __metadata_request__split = {"groups": True}

    def __init__(self, splitter: BaseCrossValidator):
        self.splitter = splitter

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise LeakageError("a grouped split was asked for without groups")
        groups = np.asarray(groups)
        for train, test in self.splitter.split(X, y, groups):
            check_grouped(train, test, groups)
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.splitter.get_n_splits(X, y, groups)


def inner_splitter(cv: CvConfig, seed: int) -> CheckedSplitter:
    """The inner loop of the hyperparameter search: stratified k-fold grouped by participant."""
    return CheckedSplitter(
        StratifiedGroupKFold(n_splits=cv.inner_folds, shuffle=True, random_state=seed)
    )


def check_grouped(train: np.ndarray, test: np.ndarray, groups: np.ndarray) -> None:
    """Raise :class:`LeakageError` if a participant has rows in both ``train`` and ``test``."""
    shared = set(groups[train]) & set(groups[test])
    if shared:
        raise LeakageError(f"participant(s) in train and test: {', '.join(map(str, shared))}")


def classifier(name: str, config: ModelConfig, seed: int) -> tuple[object, dict[str, list]]:
    """The model ``name`` and its grid, with parameter names as the Pipeline sees them."""
    grids = config.classifiers
    if name == "logistic_regression":
        settings = grids.logistic_regression
        model = LogisticRegression(solver="saga", max_iter=settings.max_iter, random_state=seed)
        return model, {"model__C": list(settings.C), "model__l1_ratio": list(settings.l1_ratio)}
    if name == "linear_svm":
        return SVC(kernel="linear"), {"model__C": list(grids.linear_svm.C)}
    if name == "gradient_boosting":
        settings = grids.gradient_boosting
        return GradientBoostingClassifier(random_state=seed), {
            "model__n_estimators": list(settings.n_estimators),
            "model__max_depth": list(settings.max_depth),
            "model__learning_rate": list(settings.learning_rate),
        }
    raise ValueError(f"unknown model {name!r}; known are {', '.join(MODELS)}")


def search(
    name: str, config: ModelConfig, n_features: int, seed: int, *, weighted: bool = False
) -> GridSearchCV:
    """Scaling, (for channels) selection and the model in one Pipeline, tuned by grouped CV.

    Everything is fitted on the training rows it is given, inside every inner fold and again
    on all of them for the refit. ``weighted``: the scaler, the model and the inner score take
    ``sample_weight``; fit it inside :func:`weighted_fit`.
    """
    model, grid = classifier(name, config, seed)
    scale = StandardScaler()
    scoring = config.cv.inner_scoring
    if weighted:
        with sklearn.config_context(enable_metadata_routing=True):
            scale.set_fit_request(sample_weight=True)
            model.set_fit_request(sample_weight=True)
            scoring = get_scorer(scoring).set_score_request(sample_weight=True)
    steps = [("scale", scale)]
    if config.level == "channel":
        steps.append(("select", SelectKBest(f_classif)))
        grid["select__k"] = sorted({min(k, n_features) for k in config.channel_k})
    steps.append(("model", model))
    return GridSearchCV(
        Pipeline(steps),
        grid,
        cv=inner_splitter(config.cv, seed),
        scoring=scoring,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )


def weighted_fit(estimator, X: np.ndarray, y: np.ndarray, groups: np.ndarray, weights):
    """Fit a search of :func:`search` (``weighted`` when ``weights`` is not None) on rows.

    With weights, scikit-learn's metadata routing hands them to the scaler, the model and the
    inner score, and the groups to the splitter; it is switched on for this fit only.
    """
    if weights is None:
        return estimator.fit(X, y, groups=groups)
    with sklearn.config_context(enable_metadata_routing=True):
        return estimator.fit(X, y, groups=groups, sample_weight=weights)


def counting_warnings(fit: Callable[[], object]) -> int:
    """Call ``fit``; returns how many of the fits in it did not converge.

    Counted are only the convergence warnings; the rest go on as usual.
    """
    n_warnings = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fit()
    for caught_warning in caught:
        if issubclass(caught_warning.category, ConvergenceWarning):
            n_warnings += 1
        else:
            warnings.showwarning(
                caught_warning.message,
                caught_warning.category,
                caught_warning.filename,
                caught_warning.lineno,
            )
    return n_warnings


def platt(
    tuned, config: ModelConfig, X, y, groups, weights, seed: int, method: str = "sigmoid"
) -> tuple[CalibratedClassifierCV, int]:
    """The tuned model of :func:`search` with Platt scaling (or ``method``) learnt on these rows,
    and the number of fits that did not converge.

    The scaling is fitted to the model's predictions on grouped inner folds of the rows, and the
    model itself is refitted on all of them (``ensemble=False``), so the result is the same
    model with its probabilities rescaled. scikit-learn does not pass groups on to the splitter
    here, so the grouped splits are handed over ready-made.
    """
    splits = list(inner_splitter(config.cv, seed).split(X, y, groups))
    calibrated = CalibratedClassifierCV(
        clone(tuned.best_estimator_), method=method, cv=splits, ensemble=False
    )

    def fit() -> None:
        if weights is None:
            calibrated.fit(X, y)
            return
        with sklearn.config_context(enable_metadata_routing=True):
            calibrated.fit(X, y, sample_weight=weights)

    return calibrated, counting_warnings(fit)


def permuted_labels(
    y: np.ndarray, groups: np.ndarray, within: bool, rng: np.random.Generator
) -> np.ndarray:
    """Labels shuffled the way the null hypothesis allows.

    Between participants: whole participants swap labels (a participant's rows keep one
    label). Within participants: every participant's labels are all swapped or all kept, with
    even odds, so a participant with one recording of each side still has one of each.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    labels = {}
    for group in np.unique(groups):
        held = y[groups == group]
        if not within and len(set(held)) > 1:
            raise ValueError(f"participant {group} has both labels in a task between people")
        labels[group] = held
    if within:
        flip = dict(zip(labels, rng.random(len(labels)) < 0.5, strict=True))
        return np.where([flip[group] for group in groups], 1 - y, y)
    order = list(labels)
    shuffled = dict(zip(order, rng.permutation([labels[g][0] for g in order]), strict=True))
    return np.array([shuffled[group] for group in groups])
