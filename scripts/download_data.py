"""Download the dataset named in configs/data.yaml into its raw-data directory.

Safe to re-run: openneuro-py compares every file already on disk with the server's size and
checksum, skips the ones that match and re-fetches only those that are missing or differ.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import niquests  # the HTTP client openneuro-py is built on
import openneuro

from pdeeg.config import CONFIG_DIR_ENV, ConfigError, load_config


@contextmanager
def _http3_disabled() -> Iterator[None]:
    """Keep openneuro-py's HTTP sessions on TCP while the block runs.

    openneuro.org advertises HTTP/3, so niquests moves later requests onto it. On networks where
    UDP port 443 is blocked those requests fail outright instead of falling back, and most of the
    dataset's small files cannot be fetched or verified. openneuro-py 2026.9.1 has no setting for
    this, so the session default is overridden here.
    """
    originals = {cls: cls.__init__ for cls in (niquests.Session, niquests.AsyncSession)}

    def tcp_only(original):
        @functools.wraps(original)
        def init(self, *args, **kwargs):
            kwargs["disable_http3"] = True
            original(self, *args, **kwargs)

        return init

    for cls, original in originals.items():
        cls.__init__ = tcp_only(original)
    try:
        yield
    finally:
        for cls, original in originals.items():
            cls.__init__ = original


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help=f"directory holding the YAML configs (default: ${CONFIG_DIR_ENV}, else ./configs)",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config_dir)
    except ConfigError as exc:
        print(f"download_data: {exc}", file=sys.stderr)
        return 2

    dataset = config.data.dataset
    target = config.data.paths.raw
    target.mkdir(parents=True, exist_ok=True)
    print(f"{dataset.id} {dataset.version} -> {target}")
    with _http3_disabled():
        openneuro.download(dataset=dataset.id, tag=dataset.version, target_dir=target)

    # openneuro-py never removes files, so a directory that once held another version could
    # still be a mixture. The DOI in the description names the version the files belong to.
    description = json.loads((target / "dataset_description.json").read_text(encoding="utf-8"))
    doi = description.get("DatasetDOI", "")
    expected = f"{dataset.id}.v{dataset.version}"
    if doi and not doi.endswith(expected):
        print(f"download_data: {target} holds {doi}, expected {expected}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
