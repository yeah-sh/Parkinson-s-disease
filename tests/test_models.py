import dataclasses
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from pdeeg.config import load_config
from pdeeg.models.classify import best_result, summary_table, weight_table
from pdeeg.models.data import (
    TaskError,
    applicable,
    centre_within,
    site_of,
    task_data,
    task_rows,
    wide_features,
)
from pdeeg.models.evaluate import (
    bootstrap_metrics,
    evaluate,
    metrics,
    nested_cv,
    permutation_p,
)
from test_stats_analysis import synthetic_recordings

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CHANNELS = mne.channels.make_standard_montage("biosemi32").ch_names
REGIONS = ["frontal", "central", "temporal", "parietal", "occipital"]
# Feature types of the synthetic wide tables, and where a signal is planted: patients above
# controls in both sessions, and OFF above ON.
TYPES = {
    "mfdfa": ["envelope_beta_main_h2", "broadband_long_delta_alpha", "envelope_alpha_main_h2"],
    "psd": [f"log_relative_power_{band}" for band in ("delta", "theta", "alpha", "beta", "gamma")],
}
DISEASE = "envelope_beta_main_h2"
MEDICATION = "broadband_long_delta_alpha"


def synthetic_tables(recordings: pd.DataFrame, strength: float = 1.5) -> dict[str, pd.DataFrame]:
    """Wide tables in the layout of the real ones: a row per recording and segments."""
    rng = np.random.default_rng(6)
    sites = CHANNELS + REGIONS
    tables = {}
    for table, types in TYPES.items():
        person = {
            (s, t): rng.standard_normal(len(sites)) for s in recordings["subject"] for t in types
        }
        rows = []
        for recording in recordings.to_dict("records"):
            for segments in ("full", "clean"):
                row = {key: recording[key] for key in ("subject", "session", "group")}
                row["segments"] = segments
                for kind in types:
                    values = person[recording["subject"], kind] + 0.3 * rng.standard_normal(
                        len(sites)
                    )
                    if kind == DISEASE and recording["group"] == "PD":
                        values = values + strength
                    if kind == MEDICATION and recording["session"] == "off":
                        values = values + strength
                    row |= {f"{kind}_{site}": v for site, v in zip(sites, values, strict=True)}
                rows.append(row)
        tables[table] = pd.DataFrame(rows)
    return tables


def small_config(root: Path | None = None, **cv):
    """The repository configuration with small grids and few permutations, for speed."""
    config = load_config(CONFIG_DIR)
    model = config.model
    classifiers = model.classifiers
    model = dataclasses.replace(
        model,
        classifiers=dataclasses.replace(
            classifiers,
            logistic_regression=dataclasses.replace(
                classifiers.logistic_regression, C=(0.1, 1.0), l1_ratio=(0.0, 1.0)
            ),
            gradient_boosting=dataclasses.replace(
                classifiers.gradient_boosting, n_estimators=(20,), max_depth=(1,)
            ),
        ),
        cv=dataclasses.replace(
            model.cv,
            **({"n_permutations": 19, "n_bootstrap": 200, "split_repeats": 2} | cv),
        ),
        n_jobs=2,
    )
    if root is not None:
        model = dataclasses.replace(
            model,
            report=root / "reports" / "classification_results.md",
            figures_dir=root / "reports" / "figures" / "classification",
            tracking=dataclasses.replace(
                model.tracking, db=root / "mlflow.db", artifacts=root / "mlruns"
            ),
        )
    return dataclasses.replace(config, model=model)


RECORDINGS = synthetic_recordings()
TABLES = synthetic_tables(RECORDINGS)


# --- the data of a task --------------------------------------------------------------------------


def test_task_a_has_one_off_recording_per_patient_and_every_control():
    rows = task_rows(load_config(CONFIG_DIR), RECORDINGS, "pd_vs_hc")

    assert rows["subject"].is_unique
    assert set(rows.loc[rows["label"] == 1, "session"]) == {"off"}
    assert (rows["label"] == 1).sum() == 15 and (rows["label"] == 0).sum() == 16


def test_task_b_pairs_both_sessions_and_leaves_out_the_listed_patients():
    rows = task_rows(load_config(CONFIG_DIR), RECORDINGS, "off_vs_on")

    assert len(rows) == 26
    assert not rows["subject"].isin(["pd6", "pd16"]).any()
    assert rows.groupby("subject")["label"].apply(lambda labels: sorted(labels) == [0, 1]).all()


def test_an_unpaired_task_with_two_recordings_of_a_person_is_an_error():
    config = load_config(CONFIG_DIR)
    twice = pd.concat([RECORDINGS, RECORDINGS[RECORDINGS["subject"] == "hc1"]])

    with pytest.raises(TaskError, match="hc1"):
        task_rows(config, twice, "pd_vs_hc")


def test_region_level_takes_the_region_columns_and_channel_level_the_others():
    config = load_config(CONFIG_DIR)
    regions = wide_features(TABLES["psd"], config)
    channel = dataclasses.replace(config, model=dataclasses.replace(config.model, level="channel"))
    channels = wide_features(TABLES["psd"], channel)

    assert {site_of(c) for c in regions.columns} == set(REGIONS)
    assert {site_of(c) for c in channels.columns} == set(CHANNELS)
    assert regions.shape[1] == 5 * 5 and channels.shape[1] == 5 * 32
    assert len(regions) == len(RECORDINGS)  # one segments value


def test_centring_uses_each_persons_own_rows_only():
    X = np.array([[1.0, 2.0], [3.0, 6.0], [10.0, 0.0], [20.0, 4.0]])
    groups = np.array(["a", "a", "b", "b"])

    centred = centre_within(X, groups)

    np.testing.assert_allclose(centred, [[-1, -2], [1, 2], [-5, -2], [5, 2]])
    changed = X.copy()
    changed[2:] += 100.0  # another person's values change nothing for a
    np.testing.assert_allclose(centre_within(changed, groups)[:2], centred[:2])


def test_feature_sets_and_what_applies_where():
    config = load_config(CONFIG_DIR)

    both = task_data(config, RECORDINGS, TABLES, "pd_vs_hc", "psd_mfdfa")
    confounds = task_data(config, RECORDINGS, TABLES, "pd_vs_hc", "age_sex")
    paired = task_data(config, RECORDINGS, TABLES, "off_vs_on", "mfdfa")

    assert both.X.shape == (31, 5 * 5 + 3 * 5)
    assert confounds.features == ["age", "sex"]
    assert set(np.unique(confounds.X[:, 1])) <= {0.0, 1.0}
    assert not applicable(config, "off_vs_on", "age_sex")
    assert applicable(config, "pd_vs_hc", "age_sex")
    # Centred within patients: each patient's two rows sum to zero.
    sums = pd.DataFrame(paired.X).groupby(paired.groups).sum().to_numpy()
    np.testing.assert_allclose(sums, 0.0, atol=1e-12)


def test_a_person_with_a_missing_value_is_left_out_with_all_their_recordings():
    config = load_config(CONFIG_DIR)
    tables = {name: table.copy() for name, table in TABLES.items()}
    psd = tables["psd"]
    psd.loc[psd["subject"] == "pd9", "log_relative_power_alpha_frontal"] = np.nan

    data = task_data(config, RECORDINGS, tables, "off_vs_on", "psd")

    assert data.left_out == ["sub-pd9"]  # participants are named by the group key
    assert "sub-pd9" not in set(data.groups) and len(data.y) == 24


# --- metrics -------------------------------------------------------------------------------------


def test_metrics_by_hand():
    y = np.array([1, 1, 1, 0, 0])
    predicted = np.array([1, 1, 0, 0, 1])
    score = np.array([0.9, 0.8, 0.3, 0.2, 0.7])

    values = metrics(y, score, predicted)

    assert values["sensitivity"] == pytest.approx(2 / 3)
    assert values["specificity"] == pytest.approx(1 / 2)
    assert values["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert values["roc_auc"] == pytest.approx(5 / 6)


def test_bootstrap_resamples_people_with_all_their_rows():
    y = np.array([1, 0] * 10)
    groups = np.repeat(np.arange(10), 2)
    score = np.where(y == 1, 1.0, -1.0)

    intervals = bootstrap_metrics(y, score, y, groups, 200, 0.95, np.random.default_rng(0))

    # Every person has one row of each class, so every resample is still perfect.
    assert intervals["balanced_accuracy"] == (1.0, 1.0)


def test_permutation_p_counts_the_observed_labelling():
    null = pd.Series([0.4, 0.5, 0.6, 0.7])

    assert permutation_p(0.65, null) == pytest.approx(2 / 5)
    assert permutation_p(0.9, null) == pytest.approx(1 / 5)


# --- the procedure on synthetic data -------------------------------------------------------------


@pytest.mark.parametrize("task", ["pd_vs_hc", "off_vs_on"])
def test_a_planted_signal_is_classified_and_noise_is_not(task):
    config = small_config()
    kind = DISEASE if task == "pd_vs_hc" else MEDICATION
    data = task_data(config, RECORDINGS, TABLES, task, "mfdfa")
    signal = [i for i, f in enumerate(data.features) if f.startswith(kind)]
    noise = [i for i, f in enumerate(data.features) if f.startswith("envelope_alpha_main_h2")]

    found = nested_cv(dataclasses.replace(data, X=data.X[:, signal]), "linear_svm", config, 1)
    empty = nested_cv(dataclasses.replace(data, X=data.X[:, noise]), "linear_svm", config, 1)

    assert metrics(data.y, found.score, found.predicted)["balanced_accuracy"] > 0.8
    assert metrics(data.y, empty.score, empty.predicted)["balanced_accuracy"] < 0.75


def test_the_same_seed_gives_the_same_predictions():
    config = small_config()
    data = task_data(config, RECORDINGS, TABLES, "pd_vs_hc", "psd")

    first = nested_cv(data, "logistic_regression", config, 3)
    second = nested_cv(data, "logistic_regression", config, 3)

    np.testing.assert_array_equal(first.score, second.score)
    assert first.weights.shape == (config.model.cv.outer_folds, data.X.shape[1])


@pytest.fixture(scope="module")
def evaluations():
    config = small_config()
    results = []
    for task, feature_set in (
        ("pd_vs_hc", "mfdfa"),
        ("pd_vs_hc", "age_sex"),
        ("off_vs_on", "mfdfa"),
    ):
        data = task_data(config, RECORDINGS, TABLES, task, feature_set)
        for model in ("logistic_regression", "gradient_boosting"):
            results.append(evaluate(data, task, feature_set, model, config, n_jobs=2))
    return config, results


def test_evaluation_has_intervals_p_values_and_other_splits(evaluations):
    config, results = evaluations
    result = results[0]

    assert set(result.intervals) == set(result.observed) == set(result.p_values)
    low, high = result.intervals["balanced_accuracy"]
    assert low <= result.observed["balanced_accuracy"] <= high
    assert len(result.null) == config.model.cv.n_permutations
    # The planted signal is stronger than every relabelling.
    assert result.p_values["balanced_accuracy"] == pytest.approx(1 / 20)
    assert len(result.repeats) == config.model.cv.split_repeats


def test_summary_best_model_and_weights(evaluations):
    config, results = evaluations

    table = summary_table(results, config)
    assert len(table) == len(results)
    assert table.loc[table["feature_set"] == "age_sex", "baseline"].all()
    assert table["balanced_accuracy_q"].between(0, 1).all()
    best = best_result(results, config, "pd_vs_hc")
    assert best.feature_set == "mfdfa"
    weights = weight_table(best)
    assert len(weights) == best.data.X.shape[1]
    linear = next(r for r in results if r.model == "logistic_regression" and r.task == "pd_vs_hc")
    top = weight_table(linear).sort_values("mean", ascending=False).iloc[0]
    assert top["type"] == DISEASE  # towards patients, as planted


def test_every_evaluation_is_an_mlflow_run_with_what_produced_it(evaluations, tmp_path):
    import mlflow

    from pdeeg.models.tracking import Tracker

    _, results = evaluations
    config = small_config(tmp_path)
    inputs = {"psd": tmp_path / "psd.parquet"}
    TABLES["psd"].to_parquet(inputs["psd"])

    tracker = Tracker(config, inputs)
    run_id = tracker.log(results[0])

    run = mlflow.MlflowClient().get_run(run_id)
    assert run.data.tags["task"] == "pd_vs_hc" and run.data.tags["dataset_version"] == "1.0.5"
    assert len(run.data.tags["config_sha256"]) == 64
    assert "psd_table_sha256" in run.data.tags and "git_commit" in run.data.tags
    assert run.data.params["outer"] == "stratified_group_kfold"
    assert run.data.metrics["balanced_accuracy"] == pytest.approx(
        results[0].observed["balanced_accuracy"]
    )
    assert "balanced_accuracy_p" in run.data.metrics
    stored = {path.name for path in (tmp_path / "mlruns").rglob("*") if path.is_file()}
    assert {"predictions.parquet", "permutation_null.parquet", "weights.parquet"} <= stored


def test_report_has_the_ablation_audit_and_limitations(evaluations, tmp_path):
    from pdeeg.models.report import write_report

    _, results = evaluations
    config = small_config(tmp_path)

    files = write_report(config, results, summary_table(results, config))

    text = files[0].read_text(encoding="utf-8")
    for heading in ("Summary in plain language", "Design", "Results", "Audit",
                    "What the best models used", "Limitations"):  # fmt: skip
        assert f"## {heading}" in text
    assert "No external test set" in text and "curators" in text
    assert all(path.is_file() for path in files)
