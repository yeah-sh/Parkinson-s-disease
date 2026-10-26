"""No participant is ever on both sides of a split: every task, every splitter, every fold."""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from pdeeg.config import load_config
from pdeeg.models.cascade import (
    cascade_data,
    cascade_metrics,
    permuted_cascade_labels,
    run_cascade,
)
from pdeeg.models.cv import (
    MODELS,
    CheckedSplitter,
    LeakageError,
    inner_splitter,
    outer_splitter,
    permuted_labels,
    search,
    seed_for,
    weighted_fit,
)
from pdeeg.models.data import task_data, wide_features
from pdeeg.models.evaluate import metrics, nested_cv
from test_models import small_config, synthetic_tables
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


def assert_separate(train, test, groups):
    """No participant on both sides, so all of a participant's recordings fall on one side."""
    assert not set(groups[train]) & set(groups[test])
    for group in set(groups[test]):
        assert (groups[test] == group).sum() == (groups == group).sum()


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
            assert_separate(train, test, groups)
            inner = inner_splitter(config.model.cv, seed)
            for inner_train, inner_test in inner.split(data.X[train], data.y[train], groups[train]):
                assert_separate(inner_train, inner_test, groups[train])


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

    def per_person(labels):
        return sorted(pd.Series(labels).groupby(data.groups).max())

    for _ in range(20):
        y = permuted_labels(data.y, data.groups, data.within, rng)
        # As many patients (or OFF recordings) as before; with two recordings of a patient in
        # stage 1, the number of recordings per class moves with the participants.
        assert per_person(y) == per_person(data.y) or data.within
        assert sorted(y) == sorted(data.y) or not data.within
        for group in np.unique(data.groups):
            held, before = y[data.groups == group], data.y[data.groups == group]
            if data.within:  # a person's labels are all kept or all swapped
                assert list(held) in (list(before), list(1 - before))
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


# --- the single-recording tasks and the cascade ----------------------------------------------


def test_both_sessions_of_a_patient_count_once_and_stay_together_in_stage_1():
    data = task_data(CONFIG, RECORDINGS, TABLES, "pd_any_vs_hc", "psd")
    rows = data.rows.assign(group=data.groups, weight=data.weights)

    assert rows.groupby("group")["weight"].sum().round(12).eq(1.0).all()
    assert (rows.loc[rows["subject"] == "pd3", "weight"] == 0.5).all()
    on = rows[rows["session"] == "on"]
    assert not on["subject"].isin(["pd6", "pd16"]).any()  # exported already preprocessed
    assert {"pd6", "pd16"} <= set(rows.loc[rows["session"] == "off", "subject"])
    assert not data.within


def test_single_recording_off_on_leaves_out_pd6_and_pd16_and_is_not_centred():
    data = task_data(CONFIG, RECORDINGS, TABLES, "off_vs_on_single", "psd")
    wide = wide_features(TABLES["psd"], CONFIG)
    raw = wide.loc[list(zip(data.rows["subject"], data.rows["session"], strict=True))]

    assert not data.rows["subject"].isin(["pd6", "pd16"]).any()
    assert data.within and not data.paired and data.weights is None
    np.testing.assert_array_equal(data.X, raw.to_numpy())


def test_the_weighted_search_hands_groups_to_the_guard():
    data = task_data(CONFIG, RECORDINGS, TABLES, "pd_any_vs_hc", "psd")
    tuned = search("logistic_regression", CONFIG.model, data.X.shape[1], 0, weighted=True)
    tuned.cv = CheckedSplitter(KFold(n_splits=3, shuffle=True, random_state=0))

    with pytest.raises(LeakageError, match="in train and test"):
        weighted_fit(tuned, data.X, data.y, data.groups, data.weights)


def test_both_stages_train_on_training_participants_only():
    config = with_outer("stratified_group_kfold")
    data = cascade_data(config, RECORDINGS, TABLES)
    patients = set(data.groups[data.label != 0])
    excluded = {f"sub-{name}" for name in config.model.tasks["off_vs_on_single"].exclude_subjects}

    for seed in (1, 2):
        predictions = run_cascade(data, small_config(), seed)
        tested = []
        for fold in predictions.folds:
            test = set(fold["test"])
            assert not test & set(fold["stage1_train"])
            assert not test & set(fold["stage2_train"])
            assert set(fold["stage2_train"]) <= patients - excluded  # no control, no pd6/pd16
            assert test | set(fold["stage1_train"]) == set(data.groups)
            tested += fold["test"]
        assert sorted(tested) == sorted(set(data.groups))  # every participant tested once
        assert (predictions.fold >= 0).all()


def test_cascade_permutations_keep_the_design():
    data = cascade_data(CONFIG, RECORDINGS, TABLES)
    rng = np.random.default_rng(0)
    sizes = pd.Series(data.groups).value_counts()

    for _ in range(20):
        label = permuted_cascade_labels(data.label, data.groups, rng)
        patients = {g for g in np.unique(data.groups) if (label[data.groups == g] != 0).all()}
        assert len(patients) == len(set(data.groups[data.label != 0]))  # 15 patients
        for group in np.unique(data.groups):
            held = label[data.groups == group]
            assert len(set(held != 0)) == 1  # one group per participant
            if group in patients and sizes[group] == 2:
                assert sorted(held) == [1, 2]  # one OFF and one ON


def test_a_random_label_per_person_cannot_cross_the_cascades_splits():
    """Near-identical recordings per person, patient or control at random per person.

    A split by recording lets a nearest neighbour find the person's other recording and its
    label; the cascade, split by participant in both stages, sits at chance.
    """
    rng = np.random.default_rng(5)
    people = 40
    person = rng.standard_normal((people, 6))
    X = np.repeat(person, 2, axis=0) + 0.01 * rng.standard_normal((2 * people, 6))
    patient = rng.permutation(np.r_[np.zeros(20), np.ones(20)]).astype(bool)
    label = np.concatenate([[1, 2] if ill else [0, 0] for ill in patient])
    groups = np.repeat([f"p{i}" for i in range(people)], 2)

    by_recording = KFold(5, shuffle=True, random_state=0)
    leaky = cross_val_predict(KNeighborsClassifier(1), X, label != 0, cv=by_recording)
    assert metrics((label != 0).astype(int), leaky, leaky)["balanced_accuracy"] > 0.8

    data = cascade_data(CONFIG, RECORDINGS, TABLES)
    data = dataclasses.replace(
        data,
        X=X,
        label=label,
        groups=groups,
        weights=np.full(2 * people, 0.5),
        stage2_allowed=np.ones(2 * people, dtype=bool),
        rows=pd.DataFrame({"subject": groups, "session": np.tile(["a", "b"], people)}),
    )
    predictions = run_cascade(data, small_config(), 6)
    values = cascade_metrics(
        label,
        predictions.predicted,
        predictions.stage2,
        predictions.p_stage1,
        predictions.p_stage2,
        data.weights,
        data.classes,
    )
    assert values["stage1_balanced_accuracy"] < 0.75
    assert values["macro_balanced_accuracy"] < 0.55
