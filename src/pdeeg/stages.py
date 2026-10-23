"""Skip a pipeline step whose settings and inputs are the ones it last ran with.

A :class:`Stage` names a step, its settings, its input files and its output files. ``done()``
records a stamp in ``<interim>/stages/<name>.json``: a digest of the settings, the SHA-256 of
every input and of every output. ``is_current()`` is true while the settings and every input
are as recorded and every output is still there, unchanged. Files are compared by content, so
rewriting a file with the same bytes keeps the steps after it current.

Paths in the settings are recorded relative to the project root, so a copy of the project in
another place (a container, a fresh clone) finds its stamps current.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pdeeg.config import Config

# Part of every step's key. Raise a step's number when its code changes in a way that its
# settings cannot show, so that what older code stored is computed again.
STEP_VERSIONS = {
    "download": 1,
    "recordings": 1,
    "psd": 1,
    "curves": 1,
    "stats": 1,
    "classify": 1,
}

_CHUNK = 1 << 20


def without(settings: Any, *names: str) -> dict[str, Any]:
    """The fields of the dataclass ``settings`` as a dict, leaving out ``names``.

    For leaving out what does not change a step's result: where it writes, how fast it runs.
    """
    fields = {field.name for field in dataclasses.fields(settings)}
    unknown = sorted(set(names) - fields)
    if unknown:
        raise ValueError(f"{type(settings).__name__} has no field(s) {', '.join(unknown)}")
    return {name: getattr(settings, name) for name in sorted(fields - set(names))}


def _plain(value: Any, root: Path) -> Any:
    """``value`` as JSON can hold it, with paths relative to ``root`` where they are inside it."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name), root) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _plain(item, root) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item, root) for item in value]
    if isinstance(value, Path):
        return _relative(value, root)
    return value


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def settings_digest(settings: Any, root: Path) -> str:
    text = json.dumps(_plain(settings, root), sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@functools.cache
def _file_digest(path: str, size: int, mtime_ns: int) -> str:
    # Cached on size and time as well as the path, and only for as long as the process runs:
    # one `pdeeg all` reads a file used by several steps once.
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    """SHA-256 of the contents of ``path``."""
    stat = path.stat()
    return _file_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


class Stage:
    """One step: what it depends on, what it writes, and whether that is still current."""

    def __init__(
        self,
        config: Config,
        name: str,
        settings: Any,
        inputs: Iterable[Path],
        outputs: Iterable[Path],
    ):
        if name not in STEP_VERSIONS:
            raise ValueError(f"no step called {name!r}; known are {', '.join(STEP_VERSIONS)}")
        self.name = name
        self.root = config.root
        self.inputs = sorted(dict.fromkeys(Path(path) for path in inputs))
        self.outputs = sorted(dict.fromkeys(Path(path) for path in outputs))
        self.key = settings_digest(
            {"step": name, "version": STEP_VERSIONS[name], "settings": settings}, self.root
        )
        self.stamp = config.data.paths.interim / "stages" / f"{name}.json"

    def _digests(self, paths: list[Path]) -> dict[str, str] | None:
        """Path (relative to the root) -> SHA-256, or None if a file is missing."""
        if not all(path.is_file() for path in paths):
            return None
        return {_relative(path, self.root): file_digest(path) for path in paths}

    def is_current(self) -> bool:
        """Whether the settings, the inputs and the outputs are the ones :meth:`done` recorded."""
        if not self.stamp.is_file():
            return False
        try:
            recorded = json.loads(self.stamp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if recorded.get("key") != self.key:
            return False
        if recorded.get("inputs") != self._digests(self.inputs):
            return False
        return recorded.get("outputs") == self._digests(self.outputs)

    def done(self) -> None:
        """Record that the step has just written its outputs from its inputs."""
        inputs, outputs = self._digests(self.inputs), self._digests(self.outputs)
        if inputs is None or outputs is None:
            missing = [str(p) for p in (*self.inputs, *self.outputs) if not p.is_file()]
            raise FileNotFoundError(f"{self.name}: missing {', '.join(missing)}")
        self.stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp = {"step": self.name, "key": self.key, "inputs": inputs, "outputs": outputs}
        self.stamp.write_text(json.dumps(stamp, indent=1, sort_keys=True), encoding="utf-8")

    def forget(self) -> None:
        """Remove the stamp, so that the step runs next time whatever it finds."""
        self.stamp.unlink(missing_ok=True)
