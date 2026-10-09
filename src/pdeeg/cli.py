"""Command-line entry point, installed as ``pdeeg``.

Pipeline stages are added as subcommands as they are written.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from pdeeg import __version__
from pdeeg.config import CONFIG_DIR_ENV, Config, ConfigError, load_config


def _write_recordings(config: Config) -> int:
    # Imported here because mne takes seconds to import and `pdeeg config` does not need it.
    from pdeeg.data.bids import DatasetError, list_recordings

    data = config.data
    try:
        recordings = list_recordings(
            data.paths.raw, data.sessions, rename_columns=data.participants_rename
        )
    except DatasetError as exc:
        print(f"pdeeg: {exc}", file=sys.stderr)
        return 1
    data.paths.recordings.parent.mkdir(parents=True, exist_ok=True)
    recordings.to_parquet(data.paths.recordings, index=False)

    summary = recordings.groupby(["group", "session"], sort=False).agg(
        recordings=("subject", "size"), participants=("subject", "nunique")
    )
    print(summary.to_string())
    print(f"{len(recordings)} recordings -> {data.paths.recordings}")
    return 0


def _preprocess(config: Config, n_jobs: int | None, force: bool) -> int:
    import pandas as pd

    from pdeeg.preprocessing.pipeline import run
    from pdeeg.preprocessing.report import write_report

    table = config.data.paths.recordings
    if not table.is_file():
        print(f"pdeeg: {table} not found; run `pdeeg recordings` first", file=sys.stderr)
        return 1
    recordings = pd.read_parquet(table)

    def show(result: dict[str, Any]) -> None:
        name = f"sub-{result['subject']} ses-{result['session']}"
        detail = result["detail"].strip().splitlines()
        print(
            f"{name}: {result['status']} ({result['seconds']} s)"
            + (f" - {detail[0]}" if detail else ""),
            flush=True,
        )

    results = run(config, recordings, n_jobs=n_jobs, force=force, progress=show)
    counts = Counter(result["status"] for result in results)
    print(", ".join(f"{count} {status}" for status, count in counts.items()))
    for result in results:
        if result["status"] == "failed":
            print(f"\nsub-{result['subject']} ses-{result['session']}:", file=sys.stderr)
            print(result["detail"], file=sys.stderr)
    print(f"QC report -> {write_report(config, recordings)}")
    return 0 if set(counts) <= {"processed", "cached"} else 1


def _mfdfa_scaling(config: Config, n_jobs: int | None) -> int:
    import pandas as pd

    from pdeeg.features.scaling import run_inspection
    from pdeeg.features.scaling_report import write_report

    table = config.data.paths.recordings
    if not table.is_file():
        print(f"pdeeg: {table} not found; run `pdeeg recordings` first", file=sys.stderr)
        return 1
    # Only what is needed to find the cleaned files: the group labels are never loaded.
    recordings = pd.read_parquet(table, columns=["subject", "session"])
    inspection = run_inspection(config, recordings, n_jobs=n_jobs)
    print(f"Scaling report -> {write_report(config, inspection, len(recordings))}")
    return 0


def _read_recordings(config: Config):
    import pandas as pd

    table = config.data.paths.recordings
    if not table.is_file():
        print(f"pdeeg: {table} not found; run `pdeeg recordings` first", file=sys.stderr)
        return None
    return pd.read_parquet(table)


def _psd_table(config: Config, recordings, n_jobs: int):
    """Build and write the band-power tables; returns the long one."""
    from pdeeg.features.psd import build_psd_table
    from pdeeg.features.tables import psd_wide, write_tables

    long = build_psd_table(config, recordings, n_jobs=n_jobs)
    paths = write_tables(config, "psd", long, psd_wide(long))
    print(f"Band-power features: {len(long)} values -> {paths['long']}")
    print(f"Wide table -> {paths['wide']}")
    return long


def _psd_features(config: Config, n_jobs: int) -> int:
    recordings = _read_recordings(config)
    if recordings is None:
        return 1
    _psd_table(config, recordings, n_jobs)
    return 0


def _mfdfa_features(config: Config, n_jobs: int | None, force: bool) -> int:
    import pandas as pd

    from pdeeg.features.extract import run
    from pdeeg.features.report import write_report
    from pdeeg.features.tables import (
        build_mfdfa_table,
        mfdfa_wide,
        table_paths,
        write_tables,
    )

    recordings = _read_recordings(config)
    if recordings is None:
        return 1

    def show(result: dict[str, Any]) -> None:
        detail = result["detail"].strip().splitlines()
        print(
            f"sub-{result['subject']} ses-{result['session']}: {result['status']} "
            f"({result['seconds']} s)" + (f" - {detail[0]}" if detail else ""),
            flush=True,
        )

    results = run(config, recordings, n_jobs=n_jobs, force=force, progress=show)
    counts = Counter(result["status"] for result in results)
    print(", ".join(f"{count} {status}" for status, count in counts.items()))
    if not set(counts) <= {"processed", "cached"}:
        for result in results:
            if result["status"] == "failed":
                print(f"\nsub-{result['subject']} ses-{result['session']}:", file=sys.stderr)
                print(result["detail"], file=sys.stderr)
        return 1

    long = build_mfdfa_table(config, recordings)
    paths = write_tables(config, "mfdfa", long, mfdfa_wide(long, config.mfdfa))
    print(f"MFDFA features: {len(long)} values -> {paths['long']}")
    print(f"Wide table -> {paths['wide']}")
    # The QC report sets the band powers next to the MFDFA features when they exist.
    psd_path = table_paths(config, "psd")["long"]
    psd = pd.read_parquet(psd_path) if psd_path.is_file() else None
    print(f"QC report -> {write_report(config, recordings, long, psd)}")
    return 0


def _plain(value: Any) -> Any:
    """Reduce ``dataclasses.asdict`` output to types YAML can serialise."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


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
    commands.add_parser("config", help="print the fully resolved configuration as YAML")
    commands.add_parser("recordings", help="index the raw dataset and save the recordings table")
    preprocess = commands.add_parser(
        "preprocess", help="clean every recording and write the preprocessing QC report"
    )
    preprocess.add_argument(
        "--config",
        type=Path,
        default=None,
        help="preprocessing YAML to use; the other settings are read from its directory "
        "unless --config-dir is given (default: preprocessing.yaml in the config directory)",
    )
    preprocess.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="recordings to process in parallel (default: n_jobs in the preprocessing config)",
    )
    preprocess.add_argument(
        "--force", action="store_true", help="redo recordings whose outputs are up to date"
    )
    features = commands.add_parser(
        "mfdfa-features",
        help="MFDFA features with surrogates for every recording, and their QC report",
    )
    features.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="channels to analyse in parallel (default: extraction.n_jobs in the MFDFA config)",
    )
    features.add_argument(
        "--force", action="store_true", help="redo recordings whose stored runs are up to date"
    )
    psd = commands.add_parser("psd-features", help="band-power features for every recording")
    psd.add_argument(
        "--n-jobs", type=int, default=1, help="recordings to analyse in parallel (default: 1)"
    )
    scaling = commands.add_parser(
        "mfdfa-scaling",
        help="plot the scaling of a few recordings, without their labels, to choose fit ranges",
    )
    scaling.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="processes to use (default: inspection.n_jobs in the MFDFA config)",
    )
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
    if args.command == "recordings":
        return _write_recordings(config)
    if args.command == "preprocess":
        return _preprocess(config, args.n_jobs, args.force)
    if args.command == "mfdfa-scaling":
        return _mfdfa_scaling(config, args.n_jobs)
    if args.command == "mfdfa-features":
        return _mfdfa_features(config, args.n_jobs, args.force)
    if args.command == "psd-features":
        return _psd_features(config, args.n_jobs)
    yaml.safe_dump(_plain(dataclasses.asdict(config)), sys.stdout, sort_keys=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
