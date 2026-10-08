from pathlib import Path

from pdeeg.config import load_config

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
