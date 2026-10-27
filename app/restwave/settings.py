"""The page's display settings, from configs/app.yaml (next to the pipeline's configuration)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import yaml

from pdeeg.config import Config
from pdeeg.features.extract import CLEAN

FILE = "app.yaml"


class SettingsError(ValueError):
    """configs/app.yaml is missing a value or has one that cannot be right."""


@dataclass(frozen=True)
class Sample:
    """The sample recording: made up (``subject`` None), or a ds002778 recording."""

    seed: int
    subject: str | None = None
    session: str | None = None


@dataclass(frozen=True)
class AppSettings:
    channels: tuple[str, ...]
    seconds: float
    sd: float
    series: str
    fit_range: str
    sample: Sample
    research_url: str


def load(config: Config) -> AppSettings:
    """configs/app.yaml of ``config``'s configuration directory, checked against it."""
    path = Path(config.config_dir) / FILE
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        signal, groups, complexity = raw["signal"], raw["groups"], raw["complexity"]
        sample = raw["sample"]
        if sample == "synthetic":
            chosen = Sample(int(raw["sample_seed"]))
        else:
            chosen = Sample(int(raw["sample_seed"]), str(sample["subject"]), str(sample["session"]))
        settings = AppSettings(
            channels=tuple(signal["channels"]),
            seconds=float(signal["seconds"]),
            sd=float(groups["sd"]),
            series=str(complexity["series"]),
            fit_range=str(complexity["fit_range"]),
            sample=chosen,
            research_url=str(raw["research_url"]),
        )
    except (KeyError, TypeError) as exc:
        raise SettingsError(f"{path}: missing or malformed {exc}") from exc
    montage = mne.channels.make_standard_montage(config.data.dataset.montage).ch_names
    unknown = [c for c in settings.channels if c not in montage]
    if unknown:
        raise SettingsError(f"{path}: signal.channels {unknown} are not montage channels")
    if config.stats.segments != CLEAN:
        # pdeeg.demo.spectrum, which draws this recording's curve, leaves the bad windows out.
        raise SettingsError(
            f"stats.segments is {config.stats.segments!r}; the page needs {CLEAN!r}"
        )
    if settings.series != "broadband" and settings.series not in config.mfdfa.envelope.bands:
        raise SettingsError(
            f"{path}: complexity.series {settings.series!r} is neither broadband nor one of "
            f"mfdfa.envelope.bands {list(config.mfdfa.envelope.bands)}"
        )
    ranges = (
        config.mfdfa.broadband.fit_ranges
        if settings.series == "broadband"
        else config.mfdfa.envelope.fit_ranges
    )
    if settings.fit_range not in ranges:
        raise SettingsError(
            f"{path}: complexity.fit_range {settings.fit_range!r} is not one of {list(ranges)}"
        )
    if settings.seconds <= 0 or settings.sd <= 0:
        raise SettingsError(f"{path}: signal.seconds and groups.sd must be positive")
    return settings
