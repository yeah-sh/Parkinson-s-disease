"""Splits by participant, the inner-loop search, and label permutations that respect the design.

No split here ever puts one participant's recordings on both sides: the splitters are grouped,
and :func:`check_grouped` is asserted on every outer and inner split the evaluation makes.
"""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
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


def search(name: str, config: ModelConfig, n_features: int, seed: int) -> GridSearchCV:
    """Scaling, (for channels) selection and the model in one Pipeline, tuned by grouped CV.

    Everything is fitted on the training rows it is given, inside every inner fold and again
    on all of them for the refit.
    """
    model, grid = classifier(name, config, seed)
    steps = [("scale", StandardScaler())]
    if config.level == "channel":
        steps.append(("select", SelectKBest(f_classif)))
        grid["select__k"] = sorted({min(k, n_features) for k in config.channel_k})
    steps.append(("model", model))
    return GridSearchCV(
        Pipeline(steps),
        grid,
        cv=inner_splitter(config.cv, seed),
        scoring=config.cv.inner_scoring,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )


def permuted_labels(
    y: np.ndarray, groups: np.ndarray, paired: bool, rng: np.random.Generator
) -> np.ndarray:
    """Labels shuffled the way the null hypothesis allows.

    Unpaired: whole participants swap labels (a participant's rows keep one label). Paired:
    within every participant the labels are swapped or not, with even odds, so each still has
    one recording of each side.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    labels = {}
    for group in np.unique(groups):
        held = y[groups == group]
        if not paired and len(set(held)) > 1:
            raise ValueError(f"participant {group} has both labels in an unpaired task")
        labels[group] = held
    if paired:
        flip = dict(zip(labels, rng.random(len(labels)) < 0.5, strict=True))
        return np.where([flip[group] for group in groups], 1 - y, y)
    order = list(labels)
    shuffled = dict(zip(order, rng.permutation([labels[g][0] for g in order]), strict=True))
    return np.array([shuffled[group] for group in groups])
