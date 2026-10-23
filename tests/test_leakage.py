"""No participant is ever on both sides of a split: every task, every splitter, every fold."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from pdeeg.config import load_config
from pdeeg.models.cv import (
    MODELS,
    CheckedSplitter,
    LeakageError,
    inner_splitter,
    outer_splitter,
    permuted_labels,
    search,
    seed_for,
)
from pdeeg.models.data import task_data
from pdeeg.models.evaluate import metrics, nested_cv
from test_models import synthetic_tables
from test_stats_analysis import synthetic_recordings

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG = load_config(CONFIG_DIR)
RECORDINGS = synthetic_recordings()
TABLES = synthetic_tables(RECORDINGS)
TASKS = list(CONFIG.model.tasks)
OUTER = ["stratified_group_kfold", "leave_one_subject_out"]


def with_outer(outer: str):
    model = CONFIG.model
    return dataclasses.replace(
        CONFIG, model=dataclasses.replace(model, cv=dataclasses.replace(model.cv, outer=outer))
    )


def assert_separate(train, test, groups, paired):
    assert not set(groups[train]) & set(groups[test])
    if paired:  # both recordings of a patient fall on the same side
        for group in set(groups[test]):
            assert (groups[test] == group).sum() == 2


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("outer", OUTER)
def test_no_participant_is_in_train_and_test_of_any_outer_or_inner_split(task, outer):
    config = with_outer(outer)
    data = task_data(config, RECORDINGS, TABLES, task, "psd")
    groups = data.groups
    # The scored split and the other splits the analysis repeats on.
    seeds = [seed_for(1, "split repeat", r) for r in range(config.model.cv.split_repeats)]
    for seed in [7, *seeds]:
        for train, test in outer_splitter(config.model.cv, seed).split(data.X, data.y, groups):
            assert_separate(train, test, groups, data.paired)
            inner = inner_splitter(config.model.cv, seed)
            for inner_train, inner_test in inner.split(data.X[train], data.y[train], groups[train]):
                assert_separate(inner_train, inner_test, groups[train], data.paired)


@pytest.mark.parametrize("task", TASKS)
def test_every_recording_is_tested_exactly_once(task):
    data = task_data(CONFIG, RECORDINGS, TABLES, task, "psd")
    splits = outer_splitter(CONFIG.model.cv, 3).split(data.X, data.y, data.groups)
    tested = np.concatenate([test for _, test in splits])

    assert sorted(tested) == list(range(len(data.y)))


def test_the_guard_catches_a_splitter_that_ignores_participants():
    data = task_data(CONFIG, RECORDINGS, TABLES, "off_vs_on", "psd")
    leaky = CheckedSplitter(KFold(n_splits=5, shuffle=True, random_state=0))

    with pytest.raises(LeakageError, match="in train and test"):
        list(leaky.split(data.X, data.y, data.groups))
    with pytest.raises(LeakageError, match="without groups"):
        list(inner_splitter(CONFIG.model.cv, 0).split(data.X, data.y))


@pytest.mark.parametrize("model", MODELS)
def test_scaling_and_tuning_sit_inside_the_grouped_search(model):
    tuned = search(model, CONFIG.model, 25, seed=0)

    assert isinstance(tuned.estimator.steps[0][1], StandardScaler)
    assert tuned.estimator.steps[-1][0] == "model"
    assert isinstance(tuned.cv, CheckedSplitter)
    channel = dataclasses.replace(CONFIG.model, level="channel")
    steps = dict(search(model, channel, 25, seed=0).estimator.steps)
    assert list(steps) == ["scale", "select", "model"]


@pytest.mark.parametrize("task", TASKS)
def test_permutations_keep_the_design(task):
    data = task_data(CONFIG, RECORDINGS, TABLES, task, "psd")
    rng = np.random.default_rng(0)

    for _ in range(20):
        y = permuted_labels(data.y, data.groups, data.paired, rng)
        assert sorted(y) == sorted(data.y)
        for group in np.unique(data.groups):
            held = y[data.groups == group]
            if data.paired:  # still one recording of each side
                assert sorted(held) == [0, 1]
            else:  # one label per person
                assert len(set(held)) == 1


def test_rows_of_one_person_cannot_carry_a_random_label_across_a_grouped_split():
    """Two near-identical recordings per person and a label that is pure chance per person.

    A split by recording lets a nearest neighbour find the person's other recording and its
    label; the grouped nested cross-validation cannot, and sits at chance.
    """
    rng = np.random.default_rng(4)
    people = 40
    person = rng.standard_normal((people, 6))
    X = np.repeat(person, 2, axis=0) + 0.01 * rng.standard_normal((2 * people, 6))
    y = np.repeat(rng.permutation(np.r_[np.zeros(20), np.ones(20)]).astype(int), 2)
    groups = np.repeat([f"p{i}" for i in range(people)], 2)

    by_recording = KFold(5, shuffle=True, random_state=0)
    leaky = cross_val_predict(KNeighborsClassifier(1), X, y, cv=by_recording)
    assert metrics(y, leaky, leaky)["balanced_accuracy"] > 0.8

    data = task_data(CONFIG, RECORDINGS, TABLES, "pd_vs_hc", "psd")
    data = dataclasses.replace(data, X=X, y=y, groups=groups, paired=False)
    predictions = nested_cv(data, "linear_svm", CONFIG, seed=5)
    assert metrics(y, predictions.score, predictions.predicted)["balanced_accuracy"] < 0.75
