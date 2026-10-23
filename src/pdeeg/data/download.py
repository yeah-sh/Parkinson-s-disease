"""Download the dataset named in configs/data.yaml into its raw-data directory.

Safe to re-run: openneuro-py compares every file already on disk with the server's size and
checksum, skips the ones that match and re-fetches only those that are missing or differ.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pdeeg.config import Config


class DownloadError(Exception):
    """The files on disk are not the configured dataset version."""


@contextmanager
def _http3_disabled() -> Iterator[None]:
    """Keep openneuro-py's HTTP sessions on TCP while the block runs.

    openneuro.org advertises HTTP/3, so niquests moves later requests onto it. On networks where
    UDP port 443 is blocked those requests fail outright instead of falling back, and most of the
    dataset's small files cannot be fetched or verified. openneuro-py 2026.9.1 has no setting for
    this, so the session default is overridden here.
    """
    import niquests  # the HTTP client openneuro-py is built on

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


def check_version(target: Path, dataset_id: str, version: str) -> None:
    """Raise :class:`DownloadError` unless ``target`` holds ``dataset_id`` at ``version``.

    openneuro-py never removes files, so a directory that once held another version could still
    be a mixture. The DOI in the description names the version the files belong to.
    """
    description = json.loads((target / "dataset_description.json").read_text(encoding="utf-8"))
    doi = description.get("DatasetDOI", "")
    expected = f"{dataset_id}.v{version}"
    if doi and not doi.endswith(expected):
        raise DownloadError(f"{target} holds {doi}, expected {expected}")


def download(config: Config) -> Path:
    """Fetch (or verify) the configured dataset version; returns where it is."""
    import openneuro

    dataset = config.data.dataset
    target = config.data.paths.raw
    target.mkdir(parents=True, exist_ok=True)
    with _http3_disabled():
        openneuro.download(dataset=dataset.id, tag=dataset.version, target_dir=target)
    check_version(target, dataset.id, dataset.version)
    return target
