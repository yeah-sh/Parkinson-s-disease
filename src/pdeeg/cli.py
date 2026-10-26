"""Command-line entry point, installed as ``pdeeg``.

``pdeeg all`` runs the whole pipeline: download, recordings, preprocess, features, stats,
classify, cascade, report. Each of them is also a command, and every step is skipped while it is
current (:mod:`pdeeg.pipeline`), so running ``pdeeg all`` again only does what a change calls for.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from pdeeg import __version__
from pdeeg.config import CONFIG_DIR_ENV, Config, ConfigError, load_config

# Commands and what they do, in pipeline order. The ones under "steps" make up `pdeeg all`.
STEPS = {
    "download": "fetch the dataset from OpenNeuro, or check the copy on disk",
    "recordings": "index the raw dataset into the recordings table",
    "preprocess": "clean every recording and write the preprocessing QC report",
    "features": "band power, MFDFA with surrogates, and h(q) curves for every recording",
    "stats": "the pre-registered cluster tests of the features",
    "classify": "nested cross-validated classification, tracked with MLflow",
    "cascade": "two stages on one recording: patient or control, then OFF or ON medication",
    "report": "write every report again from what the steps stored",
}
PARTS = {
    "psd-features": "band-power features only (part of `features`)",
    "mfdfa-features": "MFDFA features and their QC report only (part of `features`)",
    "mfdfa-curves": "h(q) and f(alpha) curves only (part of `features`)",
}


def _say(message: str) -> None:
    print(message, flush=True)


def _known_errors() -> tuple[type[Exception], ...]:
    """Errors that say what is missing or wrong, printed without a traceback."""
    # Imported here because mne takes seconds to import and `pdeeg config` does not need it.
    from pdeeg.data.bids import DatasetError
    from pdeeg.data.download import DownloadError
    from pdeeg.features.tables import TablesError
    from pdeeg.models.cascade import CascadeStoreError
    from pdeeg.models.data import TaskError
    from pdeeg.models.store import StoreError
    from pdeeg.pipeline import StepFailed
    from pdeeg.stats.features import StatsError

    return (
        CascadeStoreError,
        DatasetError,
        DownloadError,
        StatsError,
        StepFailed,
        StoreError,
        TablesError,
        TaskError,
    )


def _run(command: str, config: Config, args: argparse.Namespace) -> None:
    from pdeeg import pipeline

    force = getattr(args, "force", False)
    n_jobs = getattr(args, "n_jobs", None)
    steps = {
        "download": lambda: pipeline.download_step(config, force=force, say=_say),
        "recordings": lambda: pipeline.recordings_step(config, force=force, say=_say),
        "preprocess": lambda: pipeline.preprocess_step(
            config, n_jobs=n_jobs, force=force, say=_say
        ),
        "features": lambda: pipeline.features_step(config, n_jobs=n_jobs, force=force, say=_say),
        "psd-features": lambda: pipeline.psd_step(
            config, n_jobs=1 if n_jobs is None else n_jobs, force=force, say=_say
        ),
        "mfdfa-features": lambda: pipeline.mfdfa_step(config, n_jobs=n_jobs, force=force, say=_say),
        "mfdfa-curves": lambda: pipeline.curves_step(config, n_jobs=n_jobs, force=force, say=_say),
        "stats": lambda: pipeline.stats_step(config, n_jobs=n_jobs, force=force, say=_say),
        "classify": lambda: pipeline.classify_step(config, n_jobs=n_jobs, force=force, say=_say),
        "cascade": lambda: pipeline.cascade_step(config, n_jobs=n_jobs, force=force, say=_say),
        "report": lambda: pipeline.report_step(config, say=_say),
        "all": lambda: pipeline.all_steps(config, force=force, say=_say),
    }
    steps[command]()


def _mfdfa_scaling(config: Config, n_jobs: int | None) -> None:
    import pandas as pd

    from pdeeg.features.scaling import run_inspection
    from pdeeg.features.scaling_report import write_report
    from pdeeg.pipeline import StepFailed

    table = config.data.paths.recordings
    if not table.is_file():
        raise StepFailed(f"{table} not found; run `pdeeg recordings` first")
    # Only what is needed to find the cleaned files: the group labels are never loaded.
    recordings = pd.read_parquet(table, columns=["subject", "session"])
    inspection = run_inspection(config, recordings, n_jobs=n_jobs)
    _say(f"Scaling report -> {write_report(config, inspection, len(recordings))}")


def _plain(value: Any) -> Any:
    """Reduce ``dataclasses.asdict`` output to types YAML can serialise."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _add_n_jobs(parser: argparse.ArgumentParser, what: str) -> None:
    parser.add_argument(
        "--n-jobs", type=int, default=None, help=f"{what} (default: from the configuration)"
    )


def _add_force(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force", action="store_true", help="run even if the stored outputs are current"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdeeg",
        description="MFDFA and band-power analysis of resting-state EEG in Parkinson's disease.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help=f"directory holding the YAML configs (default: ${CONFIG_DIR_ENV}, else ./configs)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    everything = commands.add_parser(
        "all", help="the whole pipeline, skipping every step that is current"
    )
    everything.add_argument(
        "--force",
        action="store_true",
        help="run every step again, preprocessing and MFDFA included (hours)",
    )
    for name, text in STEPS.items():
        step = commands.add_parser(name, help=text)
        if name != "report":
            _add_force(step)
        if name in ("preprocess", "features", "stats", "classify", "cascade"):
            _add_n_jobs(step, "processes to use")
    preprocess = commands.choices["preprocess"]
    preprocess.add_argument(
        "--config",
        type=Path,
        default=None,
        help="preprocessing YAML to use; the other settings are read from its directory "
        "unless --config-dir is given (default: preprocessing.yaml in the config directory)",
    )
    for name, text in PARTS.items():
        part = commands.add_parser(name, help=text)
        _add_force(part)
        _add_n_jobs(part, "processes to use")
    scaling = commands.add_parser(
        "mfdfa-scaling",
        help="plot the scaling of a few recordings, without their labels, to choose fit ranges",
    )
    _add_n_jobs(scaling, "processes to use")
    commands.add_parser("config", help="print the fully resolved configuration as YAML")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_dir, overrides = args.config_dir, {}
    if args.command == "preprocess" and args.config is not None:
        overrides["preprocessing"] = args.config
        config_dir = config_dir or args.config.resolve().parent
    try:
        config = load_config(config_dir, overrides=overrides)
    except ConfigError as exc:
        print(f"pdeeg: {exc}", file=sys.stderr)
        return 2
    if args.command == "config":
        yaml.safe_dump(_plain(dataclasses.asdict(config)), sys.stdout, sort_keys=False)
        return 0
    try:
        if args.command == "mfdfa-scaling":
            _mfdfa_scaling(config, args.n_jobs)
        else:
            _run(args.command, config, args)
    except _known_errors() as exc:
        print(f"pdeeg: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
