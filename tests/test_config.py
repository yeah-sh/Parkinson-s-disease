from pathlib import Path

import pytest

from pdeeg.config import ConfigError, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_default_configs_load():
    config = load_config(CONFIG_DIR)

    assert config.root == CONFIG_DIR.parent
    assert config.data.dataset.id == "ds002778"
    assert config.data.paths.raw.is_absolute()
    assert config.data.paths.raw.is_relative_to(config.root)
    assert set(config.data.sessions) == {"hc", "off", "on"}
    assert set(config.model.tasks) == {"pd_vs_hc", "off_vs_on", "pd_any_vs_hc", "off_vs_on_single"}
    assert config.model.tasks["pd_any_vs_hc"].exclude_recordings == (("pd6", "on"), ("pd16", "on"))
    assert config.model.cascade.report.is_relative_to(config.root)
    assert all(low < high for low, high in config.psd.bands.values())
    mfdfa = config.mfdfa
    assert set(mfdfa.broadband.fit_ranges) == {"short", "long"}
    assert set(mfdfa.envelope.bands) == {"theta", "alpha", "beta"}
    assert mfdfa.inspection.report.is_relative_to(config.root)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("long: [0.3, 1.0]", "long: [1.0, 0.3]", "fit_ranges.long"),
        ("theta: [4.0, 8.0]", "theta: [8.0, 4.0]", "bands.theta"),
        ("theta: [4.0, 8.0]", "broadband: [4.0, 8.0]", "cannot be called"),
        ("qs: [-5.0, -2.0, 0.0, 2.0, 5.0]", "qs: [-5.0, 0.0, 5.0]", "include 2"),
        ("scales_per_range: 12", "n_scales: 12", "unknown key"),
    ],
)
def test_mfdfa_settings_are_validated(tmp_path, old, new, message):
    text = (CONFIG_DIR / "features" / "mfdfa.yaml").read_text(encoding="utf-8")
    assert old in text
    changed = tmp_path / "mfdfa.yaml"
    changed.write_text(text.replace(old, new), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(CONFIG_DIR, overrides={"mfdfa": changed})


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("weights: per_participant", "weights: per_person", "per_recording or per_participant"),
        ('[[pd6, "on"], [pd16, "on"]]', '[[pd6, "later"]]', "unknown session"),
        ("stage2: off_vs_on_single", "stage2: off_vs_on", "is paired"),
        ("stage1: pd_any_vs_hc", "stage1: pd_vs_hc", "positive side of stage1"),
        ("stage1: pd_any_vs_hc", "stage1: off_vs_on_single", "one group with another"),
        ("model: logistic_regression", "model: linear_svm", "needs probabilities"),
        ("feature_set: psd_mfdfa", "feature_set: eeg", "no feature set"),
        ("method: sigmoid", "method: beta", "sigmoid or isotonic"),
        ("n_bins: 5", "n_bins: 1", "at least 2 bins"),
    ],
)
def test_model_settings_are_validated(tmp_path, old, new, message):
    text = (CONFIG_DIR / "model.yaml").read_text(encoding="utf-8")
    assert text.count(old) == 1
    changed = tmp_path / "model.yaml"
    changed.write_text(text.replace(old, new), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(CONFIG_DIR, overrides={"model": changed})


def test_stats_config_holds_the_pre_registered_primary_features():
    stats = load_config(CONFIG_DIR).stats

    assert stats.features.primary == (
        "envelope_beta_main_h2",
        "envelope_beta_main_delta_alpha",
        "broadband_long_delta_alpha",
    )
    assert stats.cluster.n_permutations >= 5000
    assert stats.report.is_absolute()


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('pd_session: "off"', 'pd_session: "later"', "unknown session"),
        ('pd_session: "off"', "pd_session: hc", "different groups"),
        ('second: "on"', 'second: "off"', "same group"),
        ("segments: clean", "segments: tidy", "full or clean"),
        ("exploratory_tables: [mfdfa, psd]", "exploratory_tables: [eeg]", "unknown table"),
        ("cluster_forming_p: 0.05", "cluster_forming_p: 5", "between 0 and 1"),
        ("fdr_q: 0.05", "fdr: 0.05", "unknown key"),
    ],
)
def test_stats_settings_are_validated(tmp_path, old, new, message):
    text = (CONFIG_DIR / "stats.yaml").read_text(encoding="utf-8")
    assert old in text
    changed = tmp_path / "stats.yaml"
    changed.write_text(text.replace(old, new), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(CONFIG_DIR, overrides={"stats": changed})


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("confidence: 0.95", "confidence: 1.5", "confidence in"),
        ("amplitude_uv: [0.1, 1000.0]", "amplitude_uv: [1000.0, 0.1]", "0 < lower < upper"),
        (
            "uncertain: null  # set from `pdeeg train-final`",
            "uncertain: [0.6, 0.7]",
            "low <= threshold <= high",
        ),
        ("  stage2_paired:     #", "  stage3:     #", "predict.stages"),
    ],
)
def test_predict_settings_are_validated(tmp_path, old, new, message):
    text = (CONFIG_DIR / "predict.yaml").read_text(encoding="utf-8")
    assert text.count(old) == 1
    changed = tmp_path / "predict.yaml"
    changed.write_text(text.replace(old, new), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(CONFIG_DIR, overrides={"predict": changed})


FINAL = """final:
  stage1: {task: pd_any_vs_hc, feature_set: psd, model: linear_svm}
  stage2_single: {task: off_vs_on_single, feature_set: mfdfa, model: gradient_boosting}
  stage2_paired: {task: off_vs_on, feature_set: psd_mfdfa, model: logistic_regression}
"""


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (None, None, None),  # a valid final section loads
        ("stage1: {task: pd_any_vs_hc", "stage1: {task: pd_vs_hc", "task of model.cascade.stage1"),
        ("stage2_paired: {task: off_vs_on,", "stage2_paired: {task: off_vs_on_single,", "paired"),
        ("model: linear_svm", "model: knn", "unknown model"),
        ("  stage2_paired:", "  stage2_twice:", "in this order"),
    ],
)
def test_the_final_models_section_is_validated(tmp_path, old, new, message):
    final = FINAL if old is None else FINAL.replace(old, new)
    text = (CONFIG_DIR / "model.yaml").read_text(encoding="utf-8").replace("final: null", final)
    changed = tmp_path / "model.yaml"
    changed.write_text(text, encoding="utf-8")

    if message is None:
        final_models = load_config(CONFIG_DIR, overrides={"model": changed}).model.final
        assert final_models["stage2_single"].model == "gradient_boosting"
        return
    with pytest.raises(ConfigError, match=message):
        load_config(CONFIG_DIR, overrides={"model": changed})
