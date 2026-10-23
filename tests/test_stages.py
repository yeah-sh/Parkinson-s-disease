import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from pdeeg.config import load_config
from pdeeg.stages import Stage, file_digest, settings_digest, without

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def project(root: Path):
    """The repository configuration with the project root and every data path under ``root``."""
    config = load_config(CONFIG_DIR)
    paths = dataclasses.replace(
        config.data.paths,
        raw=root / "data" / "raw",
        interim=root / "data" / "interim",
        processed=root / "data" / "processed",
        recordings=root / "data" / "interim" / "recordings.parquet",
    )
    return dataclasses.replace(
        config, root=root, data=dataclasses.replace(config.data, paths=paths)
    )


@pytest.fixture
def step(tmp_path):
    """A step with two inputs and one output, written once and marked done."""
    config = project(tmp_path)
    inputs = [tmp_path / "in" / "a.txt", tmp_path / "in" / "b.txt"]
    output = tmp_path / "out" / "c.txt"
    for path, text in zip([*inputs, output], ["a", "b", "c"], strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def make(settings=None):
        return Stage(config, "stats", settings or {"q": 0.05}, inputs, [output])

    make().done()
    return make, inputs, output, config


def test_a_step_is_current_until_its_settings_inputs_or_outputs_change(step):
    make, inputs, output, config = step

    assert make().is_current()
    assert (config.data.paths.interim / "stages" / "stats.json").is_file()
    assert not make({"q": 0.1}).is_current()

    inputs[0].write_text("A", encoding="utf-8")
    assert not make().is_current()
    inputs[0].write_text("a", encoding="utf-8")
    assert make().is_current()  # the same bytes again: content counts, not the file's time

    output.write_text("changed", encoding="utf-8")
    assert not make().is_current()
    output.write_text("c", encoding="utf-8")
    output.unlink()
    assert not make().is_current()


def test_a_step_without_a_stamp_or_with_a_broken_one_is_not_current(step):
    make, _, _, _ = step
    stage = make()

    stage.forget()
    assert not stage.is_current()
    stage.stamp.write_text("{not json", encoding="utf-8")
    assert not stage.is_current()


def test_the_stamp_records_digests_by_path_relative_to_the_root(step):
    make, inputs, _, _ = step
    stamp = json.loads(make().stamp.read_text(encoding="utf-8"))

    assert stamp["inputs"] == {
        "in/a.txt": file_digest(inputs[0]),
        "in/b.txt": file_digest(inputs[1]),
    }
    assert set(stamp["outputs"]) == {"out/c.txt"}


def test_a_copy_of_the_project_elsewhere_is_current(step, tmp_path_factory):
    make, _, _, config = step
    copy = tmp_path_factory.mktemp("elsewhere") / "project"
    shutil.copytree(config.root, copy)
    moved = project(copy)
    settings = {"q": 0.05, "path": copy / "x"}
    stage = Stage(moved, "stats", settings, [copy / "in" / "a.txt", copy / "in" / "b.txt"], [])
    original = Stage(config, "stats", {"q": 0.05, "path": config.root / "x"}, [], [])

    # A path in the settings is recorded relative to the root, so the keys agree.
    assert stage.key == original.key
    stage.done()
    assert stage.is_current()


def test_finishing_without_an_output_is_an_error(step):
    make, _, output, _ = step
    output.unlink()

    with pytest.raises(FileNotFoundError, match="c.txt"):
        make().done()


def test_only_known_steps_and_fields_can_be_named(tmp_path):
    config = project(tmp_path)

    with pytest.raises(ValueError, match="no step called"):
        Stage(config, "train", {}, [], [])
    stats = config.stats
    kept = without(stats, "n_jobs", "report", "figures_dir")
    assert "n_jobs" not in kept and kept["seed"] == stats.seed
    with pytest.raises(ValueError, match="no field"):
        without(stats, "jobs")


def test_settings_digest_ignores_order_but_not_values(tmp_path):
    assert settings_digest({"a": 1, "b": (2, 3)}, tmp_path) == settings_digest(
        {"b": [2, 3], "a": 1}, tmp_path
    )
    assert settings_digest({"a": 1}, tmp_path) != settings_digest({"a": 2}, tmp_path)
