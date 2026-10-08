"""Command-line entry point, installed as ``pdeeg``.

Pipeline stages are added as subcommands as they are written.
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config_dir)
    except ConfigError as exc:
        print(f"pdeeg: {exc}", file=sys.stderr)
        return 2
    if args.command == "recordings":
        return _write_recordings(config)
    yaml.safe_dump(_plain(dataclasses.asdict(config)), sys.stdout, sort_keys=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
