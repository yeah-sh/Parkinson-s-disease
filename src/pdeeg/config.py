"""Load the YAML files under ``configs/`` into typed, immutable dataclasses.

No location is hard-coded in the package. The config directory is passed explicitly, or read from
``PDEEG_CONFIG_DIR``, or defaults to ``configs/`` under the current working directory. The project
root is the parent of the config directory, and every relative path in the YAML files is resolved
against it.

The YAML files are the single source of truth: every key is required and unknown keys are
rejected, so a typo fails at load time instead of silently falling back to a default.
"""

from __future__ import annotations

import os
import types
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

CONFIG_DIR_ENV = "PDEEG_CONFIG_DIR"
DEFAULT_CONFIG_DIRNAME = "configs"


class ConfigError(ValueError):
    """A config file is missing, malformed, or does not match its schema."""


# --- data.yaml ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConfig:
    id: str
    version: str
    task: str
    datatype: str
    extension: str
    line_freq: float
    montage: str


@dataclass(frozen=True)
class PathsConfig:
    raw: Path
    interim: Path
    processed: Path
    reports: Path
    recordings: Path


@dataclass(frozen=True)
class SessionConfig:
    group: str
    condition: str


@dataclass(frozen=True)
class DataConfig:
    dataset: DatasetConfig
    paths: PathsConfig
    sessions: dict[str, SessionConfig]
    participants_rename: dict[str, str]
    regions: dict[str, tuple[str, ...]]


# --- preprocessing.yaml ------------------------------------------------------------------------


@dataclass(frozen=True)
class CropConfig:
    duration: float
    start_event: str | None


@dataclass(frozen=True)
class LineNoiseConfig:
    candidates: tuple[float, ...]
    min_peak_ratio: float


@dataclass(frozen=True)
class FilterConfig:
    l_freq: float
    h_freq: float
    l_trans_bandwidth: float
    h_trans_bandwidth: float
    fir_window: str
    phase: str


@dataclass(frozen=True)
class IcaConfig:
    method: str
    extended: bool
    n_components: int | float | str
    max_iter: int
    random_state: int
    fit_l_freq: float
    fit_h_freq: float
    fit_notch: bool
    exclude_labels: tuple[str, ...]
    threshold: float


@dataclass(frozen=True)
class BadSegmentsConfig:
    window: float
    peak_to_peak_uv: float
    flat_uv: float


@dataclass(frozen=True)
class OutputConfig:
    dir: Path


@dataclass(frozen=True)
class QcConfig:
    report: Path
    figures_dir: Path
    max_bad_percent: float
    min_clean_stretch_s: float
    outlier_z: float


@dataclass(frozen=True)
class PreprocessingConfig:
    crop: CropConfig
    line_noise: LineNoiseConfig
    filter: FilterConfig
    reference: str | tuple[str, ...]
    ica: IcaConfig
    bad_segments: BadSegmentsConfig
    output: OutputConfig
    qc: QcConfig
    n_jobs: int


# --- features/mfdfa.yaml, features/psd.yaml ----------------------------------------------------


@dataclass(frozen=True)
class BroadbandConfig:
    fit_ranges: dict[str, tuple[float, float]]
    wide_features: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class EnvelopeConfig:
    bands: dict[str, tuple[float, float]]
    trans_bandwidth: float
    edge_trim: float
    fit_ranges: dict[str, tuple[float, float]]
    wide_features: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class BadWindowsConfig:
    min_windows: int
    min_scales: int


@dataclass(frozen=True)
class SurrogatesConfig:
    n_iaaft: int
    n_shuffle: int
    seed: int


@dataclass(frozen=True)
class MfdfaExtractionConfig:
    cache_dir: Path
    report: Path
    figures_dir: Path
    n_jobs: int


@dataclass(frozen=True)
class ScalingInspectionConfig:
    n_recordings: int
    seed: int
    channels: tuple[str, ...]
    qs: tuple[float, ...]
    n_scales: int
    broadband_scale_min: float
    envelope_scale_min: float
    slope_half_width: int
    null_exponents: tuple[float, ...]
    null_realisations: int
    report: Path
    figures_dir: Path
    n_jobs: int


@dataclass(frozen=True)
class MfdfaConfig:
    q_min: float
    q_max: float
    q_step: float
    detrend_order: int
    scales_per_range: int
    scale_max_frac: float
    min_variance_ratio: float
    poor_fit_r2: float
    iaaft_max_iter: int
    features: tuple[str, ...]
    broadband: BroadbandConfig
    envelope: EnvelopeConfig
    bad_windows: BadWindowsConfig
    surrogates: SurrogatesConfig
    extraction: MfdfaExtractionConfig
    inspection: ScalingInspectionConfig


@dataclass(frozen=True)
class PsdConfig:
    method: str
    fmin: float
    fmax: float
    window_sec: float
    overlap: float
    relative: bool
    log: bool
    bands: dict[str, tuple[float, float]]


# --- model.yaml --------------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskConfig:
    positive: tuple[str, ...]
    negative: tuple[str, ...]


@dataclass(frozen=True)
class CvConfig:
    scheme: str
    group_key: str
    n_permutations: int


@dataclass(frozen=True)
class TrackingConfig:
    db: Path
    artifacts: Path
    experiment: str


@dataclass(frozen=True)
class ModelConfig:
    random_state: int
    tasks: dict[str, TaskConfig]
    feature_sets: tuple[str, ...]
    classifiers: tuple[str, ...]
    cv: CvConfig
    tracking: TrackingConfig


# --- everything --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    root: Path
    config_dir: Path
    data: DataConfig
    preprocessing: PreprocessingConfig
    mfdfa: MfdfaConfig
    psd: PsdConfig
    model: ModelConfig


# Section name -> (file relative to the config directory, schema).
_SECTIONS: dict[str, tuple[str, type]] = {
    "data": ("data.yaml", DataConfig),
    "preprocessing": ("preprocessing.yaml", PreprocessingConfig),
    "mfdfa": ("features/mfdfa.yaml", MfdfaConfig),
    "psd": ("features/psd.yaml", PsdConfig),
    "model": ("model.yaml", ModelConfig),
}


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp)


def _convert(value: Any, tp: Any, where: str, root: Path) -> Any:
    """Check ``value`` (as parsed from YAML) against annotation ``tp`` and convert it."""
    origin = get_origin(tp)

    if origin in (Union, types.UnionType):
        for member in get_args(tp):
            try:
                return _convert(value, member, where, root)
            except ConfigError:
                continue
        raise ConfigError(f"{where}: expected {tp}, got {value!r}")

    if is_dataclass(tp):
        return _build(tp, value, where, root)

    if origin is tuple:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        members = get_args(tp)
        if len(members) == 2 and members[1] is Ellipsis:
            members = (members[0],) * len(value)
        elif len(members) != len(value):
            raise ConfigError(f"{where}: expected {len(members)} items, got {len(value)}")
        return tuple(
            _convert(item, member, f"{where}[{i}]", root)
            for i, (item, member) in enumerate(zip(value, members, strict=True))
        )

    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        _, member = get_args(tp)
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigError(
                    f"{where}: key {key!r} is not a string (quote YAML words such as on/off/yes/no)"
                )
            converted[key] = _convert(item, member, f"{where}.{key}", root)
        return converted

    if tp is Path:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a path string, got {value!r}")
        path = Path(value)
        return path if path.is_absolute() else root / path

    # bool is a subclass of int, so it must never satisfy a numeric field.
    is_bool = isinstance(value, bool)
    if tp is float and isinstance(value, int | float) and not is_bool:
        return float(value)
    if tp is int and isinstance(value, int) and not is_bool:
        return value
    if tp is bool and is_bool:
        return value
    if tp is str and isinstance(value, str):
        return value
    if tp is type(None) and value is None:
        return None
    if tp in (float, int, bool, str, type(None)):
        raise ConfigError(f"{where}: expected {_type_name(tp)}, got {value!r}")

    raise TypeError(f"{where}: unsupported annotation {tp!r} in config schema")


def _build(cls: type, raw: Any, where: str, root: Path) -> Any:
    """Build dataclass ``cls`` from mapping ``raw``; every field is required."""
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {raw!r}")
    names = [f.name for f in fields(cls)]
    unknown = sorted(str(key) for key in set(raw) - set(names))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s): {', '.join(unknown)}")
    missing = [name for name in names if name not in raw]
    if missing:
        raise ConfigError(f"{where}: missing key(s): {', '.join(missing)}")
    hints = get_type_hints(cls)
    return cls(**{n: _convert(raw[n], hints[n], f"{where}.{n}", root) for n in names})


def _load_yaml(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc


def _validate(config: Config) -> None:
    """Checks that span more than one field or file."""
    conditions = {session.condition for session in config.data.sessions.values()}
    for name, task in config.model.tasks.items():
        unknown = sorted({*task.positive, *task.negative} - conditions)
        if unknown:
            raise ConfigError(
                f"model.tasks.{name}: unknown condition(s) {', '.join(unknown)}; "
                f"data.sessions defines {', '.join(sorted(conditions))}"
            )
    for band, (low, high) in config.psd.bands.items():
        if not low < high:
            raise ConfigError(f"psd.bands.{band}: lower edge {low} must be below upper edge {high}")
    mfdfa = config.mfdfa
    if not mfdfa.q_min < mfdfa.q_max:
        raise ConfigError(f"mfdfa: q_min {mfdfa.q_min} must be below q_max {mfdfa.q_max}")
    for variant in ("broadband", "envelope"):
        section = getattr(mfdfa, variant)
        for name, (low, high) in section.fit_ranges.items():
            if not 0 < low < high:
                raise ConfigError(
                    f"mfdfa.{variant}.fit_ranges.{name}: need 0 < lower end < upper end, "
                    f"got {low} and {high}"
                )
        if set(section.wide_features) != set(section.fit_ranges):
            raise ConfigError(
                f"mfdfa.{variant}.wide_features: need one entry per fit range "
                f"({', '.join(section.fit_ranges)}), got {', '.join(section.wide_features)}"
            )
    if not 1 <= mfdfa.bad_windows.min_scales <= mfdfa.scales_per_range:
        raise ConfigError(
            f"mfdfa.bad_windows.min_scales: must lie between 1 and scales_per_range "
            f"({mfdfa.scales_per_range}), got {mfdfa.bad_windows.min_scales}"
        )
    placed: dict[str, str] = {}
    for region, channels in config.data.regions.items():
        for channel in channels:
            if channel in placed:
                raise ConfigError(
                    f"data.regions: {channel} is in both {placed[channel]} and {region}"
                )
            placed[channel] = region
    for band, (low, high) in mfdfa.envelope.bands.items():
        if not 0 < low < high:
            raise ConfigError(
                f"mfdfa.envelope.bands.{band}: need 0 < lower edge < upper edge, "
                f"got {low} and {high}"
            )
    inspection_qs = mfdfa.inspection.qs
    if 2.0 not in inspection_qs or list(inspection_qs) != sorted(set(inspection_qs)):
        raise ConfigError(
            f"mfdfa.inspection.qs: must be increasing and include 2, got {list(inspection_qs)}"
        )
    # The cleaned EEG itself is analysed under this name, next to the band envelopes.
    if "broadband" in mfdfa.envelope.bands:
        raise ConfigError("mfdfa.envelope.bands: a band cannot be called 'broadband'")
    ica = config.preprocessing.ica
    if isinstance(ica.n_components, str) and ica.n_components != "rank":
        raise ConfigError(
            f"preprocessing.ica.n_components: expected 'rank', an integer or a variance "
            f"fraction, got {ica.n_components!r}"
        )
    if ica.method not in ("infomax", "picard"):
        raise ConfigError(
            f"preprocessing.ica.method: expected infomax or picard, got {ica.method!r}"
        )


def resolve_config_dir(config_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the config directory: the argument, else ``$PDEEG_CONFIG_DIR``, else ``./configs``."""
    chosen = config_dir or os.environ.get(CONFIG_DIR_ENV) or Path.cwd() / DEFAULT_CONFIG_DIRNAME
    path = Path(chosen).resolve()
    if not path.is_dir():
        raise ConfigError(
            f"config directory not found: {path} (pass --config-dir or set {CONFIG_DIR_ENV})"
        )
    return path


def load_config(
    config_dir: str | os.PathLike[str] | None = None,
    *,
    overrides: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Config:
    """Load and validate every config file; relative paths come back absolute.

    ``overrides`` maps a section name (``"preprocessing"``, ...) to a YAML file to read for that
    section instead of the one in the config directory.
    """
    config_dir = resolve_config_dir(config_dir)
    root = config_dir.parent
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(_SECTIONS))
    if unknown:
        raise ConfigError(f"no config section named {', '.join(unknown)}")
    sections = {}
    for name, (relpath, schema) in _SECTIONS.items():
        path = Path(overrides[name]).resolve() if name in overrides else config_dir / relpath
        raw = _load_yaml(path)
        try:
            sections[name] = _build(schema, raw, name, root)
        except ConfigError as exc:
            raise ConfigError(f"{path}: {exc}") from None
    config = Config(root=root, config_dir=config_dir, **sections)
    _validate(config)
    return config
