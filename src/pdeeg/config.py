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
    paired: bool
    exclude_subjects: tuple[str, ...]
    exclude_recordings: tuple[tuple[str, str], ...]  # (subject, session)
    weights: str  # per_recording or per_participant


@dataclass(frozen=True)
class LogisticConfig:
    C: tuple[float, ...]
    l1_ratio: tuple[float, ...]
    max_iter: int


@dataclass(frozen=True)
class SvmConfig:
    C: tuple[float, ...]


@dataclass(frozen=True)
class BoostingConfig:
    n_estimators: tuple[int, ...]
    max_depth: tuple[int, ...]
    learning_rate: tuple[float, ...]


@dataclass(frozen=True)
class ClassifiersConfig:
    logistic_regression: LogisticConfig
    linear_svm: SvmConfig
    gradient_boosting: BoostingConfig


@dataclass(frozen=True)
class CvConfig:
    group_key: str
    outer: str
    outer_folds: int
    inner_folds: int
    inner_scoring: str
    split_repeats: int
    n_permutations: int
    n_bootstrap: int
    confidence: float


@dataclass(frozen=True)
class TrackingConfig:
    db: Path
    artifacts: Path
    experiment: str


@dataclass(frozen=True)
class CalibrationConfig:
    method: str
    n_bins: int


@dataclass(frozen=True)
class CascadeConfig:
    stage1: str
    stage2: str
    feature_set: str
    model: str
    calibration: CalibrationConfig
    report: Path
    figures_dir: Path


# The models `pdeeg train-final` fits and `pdeeg predict` uses, in the order they are applied.
FINAL_STAGES = ("stage1", "stage2_single", "stage2_paired")


@dataclass(frozen=True)
class FinalStageConfig:
    task: str
    feature_set: str
    model: str


@dataclass(frozen=True)
class ModelConfig:
    random_state: int
    segments: str
    level: str
    channel_k: tuple[int, ...]
    tasks: dict[str, TaskConfig]
    feature_sets: dict[str, tuple[str, ...]]
    classifiers: ClassifiersConfig
    cv: CvConfig
    tracking: TrackingConfig
    report: Path
    figures_dir: Path
    n_jobs: int
    cascade: CascadeConfig
    final: dict[str, FinalStageConfig] | None


# --- stats.yaml --------------------------------------------------------------------------------


@dataclass(frozen=True)
class StatsFeaturesConfig:
    primary: tuple[str, ...]
    exploratory_tables: tuple[str, ...]


@dataclass(frozen=True)
class BetweenConfig:
    pd_session: str
    hc_session: str


@dataclass(frozen=True)
class PairedConfig:
    first: str
    second: str
    exclude_subjects: tuple[str, ...]


@dataclass(frozen=True)
class ClusterConfig:
    adjacency: str
    n_permutations: int
    cluster_forming_p: float


@dataclass(frozen=True)
class BootstrapConfig:
    n_resamples: int
    confidence: float


@dataclass(frozen=True)
class StatsConfig:
    seed: int
    features: StatsFeaturesConfig
    segments: str
    between: BetweenConfig
    paired: PairedConfig
    cluster: ClusterConfig
    fdr_q: float
    covariates: tuple[str, ...]
    bootstrap: BootstrapConfig
    curves: tuple[str, ...]
    report: Path
    figures_dir: Path
    n_jobs: int


# --- predict.yaml ------------------------------------------------------------------------------


@dataclass(frozen=True)
class BandConfig:
    confidence: float
    n_bootstrap: int
    step: float


@dataclass(frozen=True)
class StageThresholdConfig:
    threshold: float
    uncertain: tuple[float, float] | None


@dataclass(frozen=True)
class PredictConfig:
    models_dir: Path
    output_dir: Path
    sfreq: float
    amplitude_uv: tuple[float, float]
    channel_aliases: dict[str, str]
    band: BandConfig
    stages: dict[str, StageThresholdConfig]
    n_top_features: int
    disclaimer: str


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
    stats: StatsConfig
    predict: PredictConfig


# Section name -> (file relative to the config directory, schema).
_SECTIONS: dict[str, tuple[str, type]] = {
    "data": ("data.yaml", DataConfig),
    "preprocessing": ("preprocessing.yaml", PreprocessingConfig),
    "mfdfa": ("features/mfdfa.yaml", MfdfaConfig),
    "psd": ("features/psd.yaml", PsdConfig),
    "model": ("model.yaml", ModelConfig),
    "stats": ("stats.yaml", StatsConfig),
    "predict": ("predict.yaml", PredictConfig),
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
    _validate_stats(config)
    _validate_model(config)
    _validate_predict(config)


def _validate_model(config: Config) -> None:
    """Checks of model.yaml that the schema cannot express."""
    model = config.model
    sessions = config.data.sessions
    groups = {session.condition: session.group for session in sessions.values()}
    for name, task in model.tasks.items():
        if set(task.positive) & set(task.negative):
            raise ConfigError(f"model.tasks.{name}: a condition is on both sides")
        if task.paired and len({groups[c] for c in (*task.positive, *task.negative)}) != 1:
            raise ConfigError(f"model.tasks.{name}: a paired task compares conditions of one group")
        if task.weights not in ("per_recording", "per_participant"):
            raise ConfigError(
                f"model.tasks.{name}.weights: expected per_recording or per_participant, "
                f"got {task.weights!r}"
            )
        if task.paired and task.weights != "per_recording":
            raise ConfigError(f"model.tasks.{name}: a paired task weighs every recording alike")
        unknown = sorted({session for _, session in task.exclude_recordings} - set(sessions))
        if unknown:
            raise ConfigError(
                f"model.tasks.{name}.exclude_recordings: unknown session(s) {', '.join(unknown)}; "
                f"data.sessions defines {', '.join(sessions)}"
            )
    for name, tables in model.feature_sets.items():
        unknown = sorted(set(tables) - {"psd", "mfdfa", "confounds"})
        if unknown or not tables:
            raise ConfigError(
                f"model.feature_sets.{name}: expected tables among psd, mfdfa, confounds, "
                f"got {list(tables)}"
            )
    if model.level not in ("region", "channel"):
        raise ConfigError(f"model.level: expected region or channel, got {model.level!r}")
    if model.segments not in ("full", "clean"):
        raise ConfigError(f"model.segments: expected full or clean, got {model.segments!r}")
    cv = model.cv
    if cv.outer not in ("stratified_group_kfold", "leave_one_subject_out"):
        raise ConfigError(
            f"model.cv.outer: expected stratified_group_kfold or leave_one_subject_out, "
            f"got {cv.outer!r}"
        )
    if cv.inner_scoring not in ("roc_auc", "balanced_accuracy"):
        raise ConfigError(
            f"model.cv.inner_scoring: expected roc_auc or balanced_accuracy, "
            f"got {cv.inner_scoring!r}"
        )
    if min(cv.outer_folds, cv.inner_folds) < 2:
        raise ConfigError("model.cv: outer_folds and inner_folds must be at least 2")
    if min(cv.n_permutations, cv.n_bootstrap) < 1 or cv.split_repeats < 0:
        raise ConfigError("model.cv: n_permutations and n_bootstrap must be at least 1")
    if not 0 < cv.confidence < 1:
        raise ConfigError(f"model.cv.confidence: must lie between 0 and 1, got {cv.confidence}")
    if not model.channel_k or min(model.channel_k) < 1:
        raise ConfigError("model.channel_k: needs at least one k of 1 or more")
    _validate_cascade(config)
    _validate_final(config)


def _validate_final(config: Config) -> None:
    """model.final: a configuration for every stage, of the tasks the stages answer."""
    model = config.model
    if model.final is None:
        return
    if tuple(model.final) != FINAL_STAGES:
        raise ConfigError(
            f"model.final: expected the stages {', '.join(FINAL_STAGES)} in this order, "
            f"got {', '.join(model.final)}"
        )
    for stage, chosen in model.final.items():
        where = f"model.final.{stage}"
        if chosen.task not in model.tasks:
            raise ConfigError(f"{where}.task: no task {chosen.task!r} in model.tasks")
        if chosen.feature_set not in model.feature_sets:
            raise ConfigError(f"{where}.feature_set: no feature set {chosen.feature_set!r}")
        if chosen.model not in ("logistic_regression", "linear_svm", "gradient_boosting"):
            raise ConfigError(f"{where}.model: unknown model {chosen.model!r}")
    final = model.final
    if final["stage1"].task != model.cascade.stage1:
        raise ConfigError("model.final.stage1: must be the task of model.cascade.stage1")
    if final["stage2_single"].task != model.cascade.stage2:
        raise ConfigError("model.final.stage2_single: must be the task of model.cascade.stage2")
    paired, single = model.tasks[final["stage2_paired"].task], model.tasks[model.cascade.stage2]
    if not paired.paired or (paired.positive, paired.negative) != (
        single.positive,
        single.negative,
    ):
        raise ConfigError(
            "model.final.stage2_paired: must be a paired task with the classes of stage 2"
        )


def _validate_predict(config: Config) -> None:
    """predict.yaml against itself (channel aliases are checked by pdeeg.predict.inputs)."""
    predict = config.predict
    if set(predict.stages) != set(FINAL_STAGES):
        raise ConfigError(f"predict.stages: expected {', '.join(FINAL_STAGES)}")
    for stage, settings in predict.stages.items():
        if not 0.0 < settings.threshold < 1.0:
            raise ConfigError(f"predict.stages.{stage}.threshold: must lie between 0 and 1")
        if settings.uncertain is not None:
            low, high = settings.uncertain
            if not 0.0 <= low <= settings.threshold <= high <= 1.0:
                raise ConfigError(
                    f"predict.stages.{stage}.uncertain: need 0 <= low <= threshold <= high <= 1, "
                    f"got [{low}, {high}]"
                )
    low, high = predict.amplitude_uv
    if not 0.0 < low < high:
        raise ConfigError("predict.amplitude_uv: need 0 < lower < upper")
    if not 0.0 < predict.band.confidence < 1.0 or predict.band.n_bootstrap < 1:
        raise ConfigError("predict.band: confidence in (0, 1) and n_bootstrap of 1 or more")
    if not 0.0 < predict.band.step < 0.5:
        raise ConfigError("predict.band.step: must lie between 0 and 0.5")


def _validate_cascade(config: Config) -> None:
    """model.cascade: stage 1 tells groups apart, stage 2 the conditions of its positive side."""
    model = config.model
    cascade = model.cascade
    groups = {session.condition: session.group for session in config.data.sessions.values()}
    for where in ("stage1", "stage2"):
        task = getattr(cascade, where)
        if task not in model.tasks:
            raise ConfigError(f"model.cascade.{where}: no task {task!r} in model.tasks")
        if model.tasks[task].paired:
            raise ConfigError(
                f"model.cascade.{where}: {task} is paired; a cascade classifies single recordings"
            )
    first, second = model.tasks[cascade.stage1], model.tasks[cascade.stage2]
    if len({groups[c] for c in (*first.positive, *first.negative)}) != 2:
        raise ConfigError("model.cascade.stage1: must compare one group with another")
    if set((*second.positive, *second.negative)) != set(first.positive):
        raise ConfigError(
            "model.cascade.stage2: its conditions must be those of the positive side of stage1"
        )
    if cascade.feature_set not in model.feature_sets:
        raise ConfigError(f"model.cascade.feature_set: no feature set {cascade.feature_set!r}")
    if cascade.model not in ("logistic_regression", "gradient_boosting"):
        raise ConfigError(
            "model.cascade.model: needs probabilities, so logistic_regression or "
            f"gradient_boosting, got {cascade.model!r}"
        )
    if cascade.calibration.method not in ("sigmoid", "isotonic"):
        raise ConfigError(
            "model.cascade.calibration.method: expected sigmoid or isotonic, "
            f"got {cascade.calibration.method!r}"
        )
    if cascade.calibration.n_bins < 2:
        raise ConfigError("model.cascade.calibration.n_bins: needs at least 2 bins")


def _validate_stats(config: Config) -> None:
    """Checks of stats.yaml against itself and the session labels of data.yaml.

    Whether the feature names exist in the feature tables is checked by
    :func:`pdeeg.stats.features.catalogue`, which knows how the tables name them.
    """
    stats = config.stats
    groups = {name: session.group for name, session in config.data.sessions.items()}
    for where, session in (
        ("between.pd_session", stats.between.pd_session),
        ("between.hc_session", stats.between.hc_session),
        ("paired.first", stats.paired.first),
        ("paired.second", stats.paired.second),
    ):
        if session not in groups:
            raise ConfigError(
                f"stats.{where}: unknown session {session!r}; data.sessions defines "
                f"{', '.join(groups)}"
            )
    if groups[stats.between.pd_session] == groups[stats.between.hc_session]:
        raise ConfigError("stats.between: the two sessions must belong to different groups")
    paired = stats.paired
    if paired.first == paired.second or groups[paired.first] != groups[paired.second]:
        raise ConfigError(
            "stats.paired: first and second must be different sessions of the same group"
        )
    if stats.segments not in ("full", "clean"):
        raise ConfigError(f"stats.segments: expected full or clean, got {stats.segments!r}")
    unknown = sorted(set(stats.features.exploratory_tables) - {"mfdfa", "psd"})
    if unknown:
        raise ConfigError(f"stats.features.exploratory_tables: unknown table(s) {unknown}")
    if len(set(stats.features.primary)) != len(stats.features.primary):
        raise ConfigError("stats.features.primary: a feature is listed twice")
    if stats.cluster.n_permutations < 1 or stats.bootstrap.n_resamples < 1:
        raise ConfigError("stats: n_permutations and n_resamples must be at least 1")
    for where, value in (
        ("cluster.cluster_forming_p", stats.cluster.cluster_forming_p),
        ("fdr_q", stats.fdr_q),
        ("bootstrap.confidence", stats.bootstrap.confidence),
    ):
        if not 0 < value < 1:
            raise ConfigError(f"stats.{where}: must lie between 0 and 1, got {value}")


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
