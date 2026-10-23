"""The pipeline's steps: run once, skipped while current, and the same reports every time."""

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from pdeeg import cli, pipeline
from pdeeg.data.download import DownloadError, check_version
from pdeeg.features.tables import table_paths
from pdeeg.models.classify import planned
from pdeeg.models.store import StoreError, load_evaluations, stored_files
from test_models import small_config
from test_models import synthetic_tables as wide_tables
from test_stats_analysis import stats_config, synthetic_recordings
from test_stats_analysis import synthetic_tables as long_tables

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def pipeline_config(root: Path):
    """Quick statistics and classification in a project at ``root``, with its own configs."""
    if not (root / "configs").is_dir():
        shutil.copytree(CONFIG_DIR, root / "configs")
    config = dataclasses.replace(stats_config(root), root=root, config_dir=root / "configs")
    model = small_config(root, n_permutations=4, n_bootstrap=50, split_repeats=1).model
    model = dataclasses.replace(model, feature_sets={"psd": ("psd",), "age_sex": ("confounds",)})
    paths = dataclasses.replace(
        config.data.paths,
        raw=root / "raw",
        interim=root / "interim",
        processed=root / "processed",
        reports=root / "reports",
        recordings=root / "interim" / "recordings.parquet",
    )
    return dataclasses.replace(
        config, model=model, data=dataclasses.replace(config.data, paths=paths)
    )


def write_inputs(config) -> None:
    recordings = synthetic_recordings()
    config.data.paths.recordings.parent.mkdir(parents=True, exist_ok=True)
    recordings.to_parquet(config.data.paths.recordings, index=False)
    long, wide = long_tables(config, recordings), wide_tables(recordings)
    for kind in ("mfdfa", "psd"):
        paths = table_paths(config, kind)
        paths["long"].parent.mkdir(parents=True, exist_ok=True)
        long[kind].to_parquet(paths["long"], index=False)
        wide[kind].to_parquet(paths["wide"], index=False)


def report_files(config) -> dict[str, str]:
    """Every report and figure of the statistics and the classification, by SHA-256."""
    files = [config.stats.report, config.model.report]
    for folder in (config.stats.figures_dir, config.model.figures_dir):
        files += sorted(folder.glob("*.png"))
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """Statistics and classification run twice on the same inputs."""
    root = tmp_path_factory.mktemp("pipeline")
    config = pipeline_config(root)
    write_inputs(config)
    said = {}
    hashes = {}
    for run in ("first", "second"):
        said[run] = []
        pipeline.stats_step(config, say=said[run].append)
        pipeline.classify_step(config, say=said[run].append)
        hashes[run] = report_files(config)
    return config, said, hashes


def test_nothing_is_written_outside_the_temporary_project(ran):
    config, _, _ = ran
    root = config.root

    assert all(path.is_relative_to(root) for path in config.data.paths.__dict__.values())
    assert config.stats.report.is_relative_to(root) and config.model.report.is_relative_to(root)
    assert config.model.tracking.db.is_relative_to(root)
    assert (config.data.paths.interim / "stages" / "stats.json").is_file()
    assert (config.data.paths.interim / "stages" / "classify.json").is_file()


def test_the_second_run_finds_both_steps_current(ran):
    _, said, _ = ran

    assert any(line.startswith("stats: ") and "tests ->" in line for line in said["first"])
    assert any("MLflow run" in line for line in said["first"])
    assert "stats: current" in said["second"]
    assert "classify: current" in said["second"]
    assert not any("MLflow run" in line for line in said["second"])


def test_reports_and_figures_are_byte_for_byte_the_same(ran):
    _, _, hashes = ran

    assert len(hashes["first"]) >= 6
    assert hashes["first"] == hashes["second"]


def test_every_planned_evaluation_is_stored(ran):
    config, _, _ = ran
    plan = planned(config)

    assert len(plan) == 3 * 2 + 3  # both feature sets for PD vs HC, no age and sex within people
    assert all(path.is_file() for path in stored_files(config, plan))
    summary = json.loads((stored_files(config, plan)[3]).read_text(encoding="utf-8"))
    assert set(summary["observed"]) == {
        "balanced_accuracy",
        "roc_auc",
        "sensitivity",
        "specificity",
    }


def test_a_changed_setting_or_input_runs_the_step_again(ran, tmp_path):
    config, _, _ = ran
    copy = tmp_path / "copy"
    shutil.copytree(config.root, copy)
    moved = pipeline_config(copy)
    said = []

    pipeline.stats_step(moved, write=False, say=said.append)
    assert said == ["stats: current"]

    stricter = dataclasses.replace(moved, stats=dataclasses.replace(moved.stats, fdr_q=0.01))
    said.clear()
    pipeline.stats_step(stricter, write=False, say=said.append)
    assert said[0].startswith("stats: ") and said[0] != "stats: current"


def test_the_report_step_refuses_results_older_than_their_inputs(ran, tmp_path):
    config, _, _ = ran
    copy = tmp_path / "copy"
    shutil.copytree(config.root, copy)
    moved = pipeline_config(copy)
    path = table_paths(moved, "psd")["wide"]
    wide = pd.read_parquet(path)
    wide.iloc[:, -1] = wide.iloc[:, -1] + 1.0
    wide.to_parquet(path, index=False)

    with pytest.raises(pipeline.StepFailed, match="pdeeg classify"):
        pipeline.report_step(moved, say=lambda _: None)


def test_stored_evaluations_must_match_the_data_they_are_loaded_for(ran):
    config, _, _ = ran
    recordings = synthetic_recordings()
    tables = {kind: pd.read_parquet(table_paths(config, kind)["wide"]) for kind in ("psd", "mfdfa")}
    fewer = recordings[recordings["subject"] != "hc3"]

    assert len(load_evaluations(config, recordings, tables, planned(config))) == 9
    with pytest.raises(StoreError, match="other data"):
        load_evaluations(config, fewer, tables, planned(config))


# --- download -------------------------------------------------------------------------------------


def test_download_is_skipped_while_the_files_are_as_verified(tmp_path, monkeypatch):
    config = pipeline_config(tmp_path)
    calls = []

    def fake_download(config):
        calls.append(1)
        raw = config.data.paths.raw
        raw.mkdir(parents=True, exist_ok=True)
        (raw / "dataset_description.json").write_text("{}", encoding="utf-8")
        return raw

    monkeypatch.setattr("pdeeg.data.download.download", fake_download)
    said = []
    pipeline.download_step(config, say=said.append)
    pipeline.download_step(config, say=said.append)
    assert len(calls) == 1 and said[-1].startswith("download: current")

    (config.data.paths.raw / "stray.txt").write_text("x", encoding="utf-8")
    pipeline.download_step(config, say=said.append)
    assert len(calls) == 2


def test_a_directory_of_another_version_is_refused(tmp_path):
    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"DatasetDOI": "doi:10.18112/openneuro.ds002778.v1.0.4"}), encoding="utf-8"
    )

    check_version(tmp_path, "ds002778", "1.0.4")
    with pytest.raises(DownloadError, match="expected ds002778.v1.0.5"):
        check_version(tmp_path, "ds002778", "1.0.5")


# --- command line ---------------------------------------------------------------------------------


def test_every_step_is_a_command():
    commands = cli.build_parser()._subparsers._group_actions[0].choices

    for name in ("download", "preprocess", "features", "stats", "classify", "report", "all"):
        assert name in commands
    assert cli.build_parser().parse_args(["all", "--force"]).force


@pytest.fixture
def empty_project(tmp_path):
    """The repository's configs in a project without any data."""
    shutil.copytree(CONFIG_DIR, tmp_path / "configs")
    return tmp_path / "configs"


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("stats", "run `pdeeg recordings` first"),
        ("report", "run `pdeeg stats` and `pdeeg classify`"),
    ],
)
def test_a_step_without_its_inputs_says_what_to_run(empty_project, capsys, command, message):
    assert cli.main(["--config-dir", str(empty_project), command]) == 1
    assert message in capsys.readouterr().err
    assert not (empty_project.parent / "data" / "processed").exists()
