"""The two-stage cascade: its data, metrics, calibration, and a planted three-class effect."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pdeeg.models.calibration import (
    brier,
    calibration_summary,
    miscalibration,
    multiclass_brier,
    recalibration,
    reference_brier,
    reliability_table,
)
from pdeeg.models.cascade import (
    CascadeStoreError,
    bootstrap_cascade,
    cascade_data,
    cascade_metrics,
    confusion,
    evaluate_cascade,
    load_cascade,
    run_cascade,
    save_cascade,
    three_class_probabilities,
)
from test_models import DISEASE, MEDICATION, small_config, synthetic_tables
from test_stats_analysis import synthetic_recordings

RECORDINGS = synthetic_recordings()
TABLES = synthetic_tables(RECORDINGS)
CLASSES = ("HC", "PD_OFF", "PD_ON")


def cascade_config(root=None, **cv):
    """Small grids and few permutations, the cascade on the MFDFA table."""
    config = small_config(root, **cv)
    model = config.model
    cascade = dataclasses.replace(model.cascade, feature_set="mfdfa")
    if root is not None:
        cascade = dataclasses.replace(
            cascade,
            report=root / "reports" / "cascade_results.md",
            figures_dir=root / "reports" / "figures" / "cascade",
        )
    return dataclasses.replace(config, model=dataclasses.replace(model, cascade=cascade))


def only(data, prefixes):
    """``data`` with the features whose names start with one of ``prefixes``."""
    keep = [i for i, name in enumerate(data.features) if name.startswith(tuple(prefixes))]
    return dataclasses.replace(data, X=data.X[:, keep], features=[data.features[i] for i in keep])


# --- the data --------------------------------------------------------------------------------


def test_the_cascade_has_every_stage_one_recording_with_three_labels():
    data = cascade_data(cascade_config(), RECORDINGS, TABLES)

    assert data.classes == CLASSES
    assert np.bincount(data.label).tolist() == [16, 15, 13]  # pd6 and pd16 ON left out
    assert len(np.unique(data.groups)) == 31
    stage2 = data.rows[(data.label != 0) & data.stage2_allowed]
    assert not stage2["subject"].isin(["pd6", "pd16"]).any() and len(stage2) == 26
    # Stage 1's weights: every participant counts once.
    assert np.allclose(pd.Series(data.weights).groupby(data.groups).sum(), 1.0)


# --- metrics ---------------------------------------------------------------------------------


def test_cascade_metrics_by_hand():
    label = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    predicted = np.array([0, 0, 0, 1, 1, 1, 0, 2, 1, 0])
    stage2 = np.array([1, 0, 1, 1, 1, 1, 1, 0, 1, 1])  # stage 2's own call on every row
    p1 = np.where(predicted != 0, 0.8, 0.2)
    p2 = np.where(stage2 == 1, 0.7, 0.3)

    values = cascade_metrics(label, predicted, stage2, p1, p2, None, CLASSES)

    assert confusion(label, predicted).tolist() == [[3, 1, 0], [1, 2, 0], [1, 1, 1]]
    assert values["sensitivity_HC"] == pytest.approx(3 / 4)
    assert values["sensitivity_PD_OFF"] == pytest.approx(2 / 3)
    assert values["sensitivity_PD_ON"] == pytest.approx(1 / 3)
    assert values["macro_balanced_accuracy"] == pytest.approx((3 / 4 + 2 / 3 + 1 / 3) / 3)
    # Stage 1: patients found 4 of 6, controls 3 of 4.
    assert values["stage1_balanced_accuracy"] == pytest.approx((4 / 6 + 3 / 4) / 2)
    # Stage 2 on the six patients' recordings, whatever stage 1 said: OFF 3 of 3, ON 1 of 3.
    assert values["stage2_balanced_accuracy"] == pytest.approx((1 + 1 / 3) / 2)
    assert values["pd_missed"] == pytest.approx(2 / 6)
    assert values["pd_wrong_state"] == pytest.approx(1 / 6)
    assert values["pd_correct"] == pytest.approx(3 / 6)
    assert values["hc_called_pd"] == pytest.approx(1 / 4)


def test_the_bootstrap_resamples_participants_with_all_their_recordings():
    # Every participant: one control, or a patient with one OFF and one ON, all called right.
    label = np.array([0, 0, 1, 2, 1, 2, 0])
    groups = np.array(["a", "b", "c", "c", "d", "d", "e"])
    p1 = np.where(label != 0, 0.9, 0.1)
    p2 = np.where(label == 1, 0.9, 0.1)
    stage2 = (label == 1).astype(int)

    intervals = bootstrap_cascade(
        label, label, stage2, p1, p2, None, groups, CLASSES, 200, 0.95, np.random.default_rng(0)
    )

    assert intervals["stage2_balanced_accuracy"] == (1.0, 1.0)  # a patient brings both sides
    assert intervals["macro_balanced_accuracy"] == (1.0, 1.0)


def test_three_class_probabilities_sum_to_one():
    probabilities = three_class_probabilities(np.array([0.2, 0.9]), np.array([0.5, 0.1]))

    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(probabilities[1], [0.1, 0.09, 0.81])
    assert multiclass_brier(np.eye(3), np.array([0, 1, 2])) == 0.0
    assert multiclass_brier(np.eye(3)[[1, 2, 0]], np.array([0, 1, 2])) == 2.0


# --- calibration -----------------------------------------------------------------------------


def test_brier_scores_by_hand():
    y = np.array([1, 0, 1, 1])
    p = np.array([0.9, 0.2, 0.6, 0.5])

    assert brier(y, p) == pytest.approx((0.01 + 0.04 + 0.16 + 0.25) / 4)
    assert reference_brier(y) == pytest.approx(0.75 * 0.25)
    weights = np.array([1.0, 1.0, 0.0, 0.0])
    assert brier(y, p, weights) == pytest.approx((0.01 + 0.04) / 2)


def test_recalibration_recovers_slope_and_intercept():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.5, 5000)
    y = (rng.random(x.size) < 1 / (1 + np.exp(-x))).astype(int)

    calibrated = recalibration(y, 1 / (1 + np.exp(-x)))
    extreme = recalibration(y, 1 / (1 + np.exp(-3 * x)))  # overconfident: three times too steep
    shifted = recalibration(y, 1 / (1 + np.exp(-(x + 1))))  # too high by 1 on the logit scale

    assert calibrated == pytest.approx((0.0, 1.0), abs=0.1)
    assert extreme[1] == pytest.approx(1 / 3, abs=0.05)
    assert shifted[0] == pytest.approx(-1.0, abs=0.1)


def test_the_rule_flags_overconfident_probabilities_and_passes_calibrated_ones():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, 400)
    y = (rng.random(x.size) < 1 / (1 + np.exp(-x))).astype(int)
    groups = np.arange(x.size)

    def reasons(p):
        summary = calibration_summary(y, p, groups, None, 300, 0.95, np.random.default_rng(3))
        return miscalibration(summary)

    assert reasons(1 / (1 + np.exp(-x))) == []
    assert "interval of the calibration slope leaves out 1" in reasons(1 / (1 + np.exp(-4 * x)))
    flipped = reasons(1 / (1 + np.exp(2 * x)))  # wrong way round
    assert "Brier score not below that of predicting the prevalence" in flipped


def test_reliability_bins_have_equal_counts_and_wilson_intervals():
    rng = np.random.default_rng(4)
    p = rng.random(50)
    y = (rng.random(50) < p).astype(int)

    table = reliability_table(y, p, 5)

    assert table["n_recordings"].tolist() == [10] * 5
    assert table["mean_probability"].is_monotonic_increasing
    assert (table["observed_low"] <= table["observed"]).all()
    assert (table["observed"] <= table["observed_high"]).all()


# --- the procedure on synthetic data ---------------------------------------------------------


def test_platt_scaling_rescales_the_same_model_within_each_fold():
    config = cascade_config()
    data = cascade_data(config, RECORDINGS, TABLES)

    predictions = run_cascade(data, config, 2, calibrate=True)

    assert np.isfinite(predictions.p_stage1_platt).all()
    for k in np.unique(predictions.fold):
        rows = predictions.fold == k
        for raw, scaled in (
            (predictions.p_stage1, predictions.p_stage1_platt),
            (predictions.p_stage2, predictions.p_stage2_platt),
        ):
            order = np.argsort(raw[rows], kind="stable")
            assert np.all(np.diff(scaled[rows][order]) >= -1e-12)  # same ranking


@pytest.fixture(scope="module")
def planted():
    """The cascade on the disease and medication features of the synthetic tables."""
    config = cascade_config(n_permutations=39, n_bootstrap=200, split_repeats=2)
    data = only(cascade_data(config, RECORDINGS, TABLES), [DISEASE, MEDICATION])
    return config, evaluate_cascade(data, config, n_jobs=2)


def test_a_planted_three_class_effect_is_recovered(planted):
    _, result = planted
    observed = result.observed

    assert observed["macro_balanced_accuracy"] > 0.8  # chance 1/3
    assert min(observed[f"sensitivity_{name}"] for name in CLASSES) > 0.7
    assert observed["stage1_balanced_accuracy"] > 0.8
    assert observed["stage2_balanced_accuracy"] > 0.8
    # Stronger than every one of the 39 relabellings.
    assert result.p_values["macro_balanced_accuracy"] == pytest.approx(1 / 40)
    low, high = result.intervals["macro_balanced_accuracy"]
    assert 1 / 3 < low <= observed["macro_balanced_accuracy"] <= high
    assert result.null["macro_balanced_accuracy"].median() < 0.5
    assert len(result.repeats) == 2
    assert result.confusion.sum() == len(result.data.label)


def test_noise_is_not_classified():
    config = cascade_config()
    data = only(cascade_data(config, RECORDINGS, TABLES), ["envelope_alpha_main_h2"])

    predictions = run_cascade(data, config, 3)
    values = cascade_metrics(
        data.label,
        predictions.predicted,
        predictions.stage2,
        predictions.p_stage1,
        predictions.p_stage2,
        data.weights,
        data.classes,
    )

    assert values["macro_balanced_accuracy"] < 0.55


def test_the_cascade_is_stored_and_loaded_again(planted, tmp_path):
    config, result = planted
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data, paths=dataclasses.replace(config.data.paths, processed=tmp_path)
        ),
    )

    save_cascade(config, result)
    loaded = load_cascade(config, result.data)

    np.testing.assert_array_equal(loaded.predictions.predicted, result.predictions.predicted)
    assert loaded.observed == pytest.approx(result.observed)
    assert loaded.p_values == result.p_values
    fewer = dataclasses.replace(result.data, label=np.zeros_like(result.data.label))
    with pytest.raises(CascadeStoreError, match="other data"):
        load_cascade(config, fewer)


def test_the_cascade_is_an_mlflow_run_and_a_report(planted, tmp_path):
    import mlflow

    from pdeeg.models.cascade_report import write_report
    from pdeeg.models.tracking import Tracker

    _, result = planted
    config = cascade_config(tmp_path)
    inputs = {"psd": tmp_path / "psd.parquet"}
    TABLES["psd"].to_parquet(inputs["psd"])

    run_id = Tracker(config, inputs).log_cascade(result, [inputs["psd"]])
    run = mlflow.MlflowClient().get_run(run_id)
    assert run.data.tags["task"] == "cascade" and "git_commit" in run.data.tags
    assert run.data.params["stage1"] == "pd_any_vs_hc"
    assert run.data.metrics["macro_balanced_accuracy"] == pytest.approx(
        result.observed["macro_balanced_accuracy"]
    )
    logged = ("macro_balanced_accuracy_p", "stage2_balanced_accuracy_ci_low", "stage1_raw_brier")
    assert {*logged, "stage2_platt_slope"} <= set(run.data.metrics)

    files = write_report(config, result, None)
    text = files[0].read_text(encoding="utf-8")
    headings = ("Summary in plain language", "Design", "Results", "Calibration")
    for heading in (*headings, "The stages on their own", "Limitations"):
        assert f"## {heading}" in text
    assert "after the pre-registration" in text and "curators" in text
    assert "above chance" in text  # the planted effect
    assert all(path.is_file() for path in files)
