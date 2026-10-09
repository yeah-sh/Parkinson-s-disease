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
    assert set(config.model.tasks) == {"pd_vs_hc", "off_vs_on"}
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
