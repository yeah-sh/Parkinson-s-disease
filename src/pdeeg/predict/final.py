"""The final models (``pdeeg train-final``): every stage fitted on all eligible participants.

For each stage of ``model.final`` (task, feature set and model, the best of the ablation):

1. The stage's nested cross-validation is run again with the ablation's own seed, so its
   decisions are those the ablation stored (checked), now with probabilities: raw, and Platt
   scaled on each outer training set. From these out-of-fold probabilities the rule of
   ``predict.band`` gives the uncertain band (:func:`pdeeg.models.calibration.uncertain_band`).
   Models are saved only while ``configs/predict.yaml`` holds exactly those bands.
2. The model is tuned by the grouped inner loop on all of the task's participants, refitted on
   all of them, and Platt scaled on grouped inner folds of them.

Each stage goes to ``<models_dir>/<stage>/``: ``model.joblib`` (a scikit-learn Pipeline of the
calibrated scaler-and-model pipeline), ``feature_columns.json``, ``oof_predictions.parquet`` and
``metadata.json`` (configuration digests, dataset, git commit, library versions, the
cross-validated metrics with their intervals, the thresholds, and a model card). The
recordings trained on are listed, with hashes, in ``<models_dir>/training_recordings.parquet``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import mne
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

import pdeeg
from pdeeg.config import FINAL_STAGES, Config
from pdeeg.data.bids import load_raw
from pdeeg.models.calibration import uncertain_band
from pdeeg.models.cascade import cascade_dir
from pdeeg.models.classify import is_baseline
from pdeeg.models.cv import platt, search, seed_for
from pdeeg.models.data import TaskData, task_data
from pdeeg.models.evaluate import Predictions, fit_counting_warnings, metrics, nested_cv
from pdeeg.models.report import MODEL_LABELS, set_label, task_title
from pdeeg.models.store import evaluation_dir
from pdeeg.models.tracking import config_digest, git_state
from pdeeg.predict.features import feature_digest
from pdeeg.predict.inputs import REGISTRY_COLUMNS, signal_digest
from pdeeg.stages import file_digest

STAGE_TITLES = {
    "stage1": "Stage 1: patient or control",
    "stage2_single": "Stage 2: OFF or ON medication, one recording",
    "stage2_paired": "Stage 2, paired: which of a patient's two recordings is OFF",
}
REGISTRY = "training_recordings.parquet"
FILES = ("model.joblib", "feature_columns.json", "metadata.json", "oof_predictions.parquet")
_METRICS = ("balanced_accuracy", "roc_auc", "sensitivity", "specificity")

Say = Callable[[str], None]


class FinalError(Exception):
    """The final models cannot be trained as configured; the message says why."""


def stage_tasks(config: Config) -> dict[str, str]:
    """The task of every stage: the cascade's two, and the paired task with stage 2's classes."""
    model = config.model
    single = model.tasks[model.cascade.stage2]
    paired = [
        name
        for name, task in model.tasks.items()
        if task.paired and (task.positive, task.negative) == (single.positive, single.negative)
    ]
    if not paired:
        raise FinalError(f"no paired task with the classes of {model.cascade.stage2}")
    return {
        "stage1": model.cascade.stage1,
        "stage2_single": model.cascade.stage2,
        "stage2_paired": paired[0],
    }


def best_choices(summary: pd.DataFrame, config: Config) -> dict[str, tuple[str, str]]:
    """The (feature set, model) of every stage with the best balanced accuracy, then ROC AUC.

    Feature sets of age and sex alone are left out; among equals the first in the ablation's
    order wins.
    """
    choices = {}
    for stage, task in stage_tasks(config).items():
        rows = summary[
            (summary["task"] == task)
            & ~summary["feature_set"].map(lambda name: is_baseline(config, name))
        ]
        if rows.empty:
            raise FinalError(f"no evaluations of {task} in the classification summary")
        best = rows.sort_values(
            ["balanced_accuracy", "roc_auc"], ascending=False, kind="stable"
        ).iloc[0]
        choices[stage] = (str(best["feature_set"]), str(best["model"]))
    return choices


def _final_yaml(choices: Mapping[str, tuple[str, str]], config: Config) -> str:
    tasks = stage_tasks(config)
    lines = ["final:"]
    for stage, (feature_set, model) in choices.items():
        lines += [
            f"  {stage}:",
            f"    task: {tasks[stage]}",
            f"    feature_set: {feature_set}",
            f"    model: {model}",
        ]
    return "\n".join(lines)


def check_final(config: Config, summary: pd.DataFrame) -> None:
    """Raise :class:`FinalError` unless ``model.final`` holds the best of the ablation."""
    best = best_choices(summary, config)
    final = config.model.final
    if final is None:
        raise FinalError(
            "model.final is null. The best of the ablation, to put in configs/model.yaml:\n"
            + _final_yaml(best, config)
        )
    wrong = [
        stage
        for stage in FINAL_STAGES
        if (final[stage].feature_set, final[stage].model) != best[stage]
        or final[stage].task != stage_tasks(config)[stage]
    ]
    if wrong:
        raise FinalError(
            f"model.final is not the best of the ablation for {', '.join(wrong)}. "
            "It should read:\n" + _final_yaml(best, config)
        )


# --- out-of-fold probabilities and the uncertain band --------------------------------------------


def pair_normalised(probability: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """P(this one of a patient's two recordings is the positive one): p / (p + p of the other).

    For a linear model on centred features the two already sum to 1, and nothing changes.
    """
    probability = np.asarray(probability, dtype=float)
    total = pd.Series(probability).groupby(np.asarray(groups)).transform("sum").to_numpy()
    return probability / total


@dataclass(frozen=True)
class StagePlan:
    """What a stage's out-of-fold run gives: its data, the probabilities, band and how it does."""

    stage: str
    task: str
    feature_set: str
    model: str
    seed: int
    data: TaskData
    oof: pd.DataFrame
    threshold: float
    band: tuple[float, float]
    performance: dict[str, float]


def _call(probability: np.ndarray, threshold: float, band: tuple[float, float]) -> np.ndarray:
    """1 or 0 by the threshold, -1 inside the band."""
    low, high = band
    called = (probability >= threshold).astype(int)
    return np.where((probability >= low) & (probability <= high), -1, called)


def plan_stage(
    config: Config,
    recordings: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    stage: str,
) -> StagePlan:
    """The out-of-fold run of ``stage``, checked against the stored ablation, and its band."""
    chosen = config.model.final[stage]
    task, feature_set, model = chosen.task, chosen.feature_set, chosen.model
    data = task_data(config, recordings, tables, task, feature_set)
    seed = seed_for(config.model.random_state, task, feature_set, model)
    predictions: Predictions = nested_cv(data, model, config, seed, calibrate=True)
    stored_path = evaluation_dir(config, task, feature_set, model) / "predictions.parquet"
    if not stored_path.is_file():
        raise FinalError(f"{stored_path} not found; run `pdeeg classify` first")
    stored = pd.read_parquet(stored_path)
    if not (
        stored[["subject", "session"]].equals(data.rows[["subject", "session"]])
        and np.allclose(stored["score"].to_numpy(), predictions.score, rtol=1e-9, atol=1e-12)
        and np.array_equal(stored["predicted"].to_numpy(), predictions.predicted)
    ):
        raise FinalError(
            f"{task} / {feature_set} / {model}: the out-of-fold decisions differ from the "
            "stored ablation; run `pdeeg classify --force`"
        )
    probability = predictions.probability_platt
    if data.paired:
        probability = pair_normalised(probability, data.groups)
    oof = data.rows.assign(
        participant=data.groups,
        label=data.y,
        weight=1.0 if data.weights is None else data.weights,
        decision=predictions.score,
        predicted=predictions.predicted,
        probability_raw=predictions.probability,
        probability_platt=predictions.probability_platt,
        probability=probability,
        fold=predictions.fold,
    )
    band_settings = config.predict.band
    threshold = config.predict.stages[stage].threshold
    band = uncertain_band(
        data.y,
        probability,
        data.groups,
        data.weights,
        band_settings.n_bootstrap,
        band_settings.confidence,
        np.random.default_rng(seed_for(seed, "uncertain band")),
        step=band_settings.step,
        threshold=threshold,
    )
    band = (round(band[0], 6), round(band[1], 6))
    calls = _call(probability, threshold, band)
    weights = np.ones(len(calls)) if data.weights is None else data.weights
    certain = calls >= 0
    everywhere = metrics(data.y, probability, (probability >= threshold).astype(int), data.weights)
    labelled = None  # nothing labelled (a band of [0, 1]), or only one class among the labelled
    if certain.any():
        outside = metrics(data.y[certain], probability[certain], calls[certain], weights[certain])
        if np.isfinite(outside["balanced_accuracy"]):
            labelled = outside["balanced_accuracy"]
    performance = {
        "balanced_accuracy_at_threshold": everywhere["balanced_accuracy"],
        "share_labelled": float(weights[certain].sum() / weights.sum()),
        "n_labelled": int(certain.sum()),
        "n_recordings": len(calls),
        "balanced_accuracy_labelled": labelled,
    }
    return StagePlan(stage, task, feature_set, model, seed, data, oof, threshold, band, performance)


def check_bands(config: Config, plans: list[StagePlan]) -> None:
    """Raise :class:`FinalError` unless predict.yaml holds the bands the rule gives."""
    wrong = []
    for plan in plans:
        configured = config.predict.stages[plan.stage].uncertain
        if configured is None or not np.allclose(configured, plan.band, atol=1e-9):
            wrong.append(plan)
    if wrong:
        lines = [
            "configs/predict.yaml does not hold the uncertain bands the rule gives from the "
            "out-of-fold probabilities (predict.band). Set:"
        ]
        lines += [f"  {p.stage}: uncertain: [{p.band[0]:g}, {p.band[1]:g}]" for p in plans]
        raise FinalError("\n".join(lines))


# --- what the metadata carries ------------------------------------------------------------------


def _interval(row: pd.Series, metric: str) -> dict[str, float]:
    return {
        "value": float(row[metric]),
        "low": float(row[f"{metric}_low"]),
        "high": float(row[f"{metric}_high"]),
        "p": float(row[f"{metric}_p"]),
    }


def cross_validated(
    config: Config, summary: pd.DataFrame, plan: StagePlan, cascade: Mapping[str, Any]
) -> dict[str, Any]:
    """The stage's cross-validated metrics: of the chosen configuration (selected, so
    optimistic) and of the configuration fixed before any result."""
    rows = summary[summary["task"] == plan.task]
    candidates = rows[~rows["feature_set"].map(lambda name: is_baseline(config, name))]

    def entry(feature_set: str, model: str) -> dict[str, Any]:
        row = rows[(rows["feature_set"] == feature_set) & (rows["model"] == model)].iloc[0]
        return {
            "feature_set": feature_set,
            "model": model,
            "n_participants": int(row["n_participants"]),
            "n_recordings": int(row["n_recordings"]),
            **{metric: _interval(row, metric) for metric in _METRICS},
            "balanced_accuracy_q": float(row["balanced_accuracy_q"]),
            "other_splits": [float(row["repeat_min"]), float(row["repeat_max"])],
        }

    settings = config.model.cascade
    result = {
        "selected": entry(plan.feature_set, plan.model)
        | {
            "selected_among": len(candidates),
            "note": (
                f"The best balanced accuracy of {len(candidates)} feature-set and model "
                "combinations on this same cross-validation, so optimistic."
            ),
        },
        "prespecified": entry(settings.feature_set, settings.model)
        | {"note": "The cascade's configuration, fixed before any classifier was run."},
        "source": "reports/classification_results.md",
    }
    metric = {"stage1": "stage1", "stage2_single": "stage2"}.get(plan.stage)
    if metric is not None:
        observed, intervals, p = cascade["observed"], cascade["intervals"], cascade["p_values"]
        result["in_cascade"] = {
            "feature_set": cascade["feature_set"],
            "model": cascade["model"],
            "source": "reports/cascade_results.md",
            **{
                name: {
                    "value": observed[f"{metric}_{name}"],
                    "low": intervals[f"{metric}_{name}"][0],
                    "high": intervals[f"{metric}_{name}"][1],
                    "p": p[f"{metric}_{name}"],
                }
                for name in ("balanced_accuracy", "roc_auc")
            },
            "macro_balanced_accuracy": {
                "value": observed["macro_balanced_accuracy"],
                "low": intervals["macro_balanced_accuracy"][0],
                "high": intervals["macro_balanced_accuracy"][1],
                "p": p["macro_balanced_accuracy"],
            },
        }
    return result


def model_card(config: Config, plan: StagePlan, metrics_entry: Mapping[str, Any]) -> dict:
    """The model card section of a stage's metadata; every number comes from the data."""
    data = plan.data
    task = config.model.tasks[plan.task]
    counts = data.rows["condition"].value_counts().to_dict()
    selected = metrics_entry["selected"]["balanced_accuracy"]
    low, high = plan.band
    exclusions = (
        [f"participants {', '.join(task.exclude_subjects)}"] if task.exclude_subjects else []
    )
    exclusions += [f"recording {s} {t}" for s, t in task.exclude_recordings]
    dataset = config.data.dataset
    return {
        "name": f"pdeeg {plan.stage}: {STAGE_TITLES[plan.stage]}",
        "intended_use": (
            "Research demonstration of resting-state EEG features (band power, MFDFA) in "
            "Parkinson's disease. Not for diagnosis, screening, monitoring or any decision "
            "about a person's care."
        ),
        "task": f"{task_title(config, plan.task)} (`{plan.task}`)",
        "model": (
            f"{MODEL_LABELS[plan.model]} on {set_label(config, plan.feature_set)} "
            f"({data.X.shape[1]} features, {config.model.level} level), standardised, with "
            "Platt scaling"
        ),
        "training_data": {
            "dataset": f"{dataset.id} {dataset.version} (OpenNeuro)",
            "participants": len(np.unique(data.groups)),
            "recordings": len(data.y),
            "recordings_by_condition": {str(k): int(v) for k, v in counts.items()},
            "left_out": exclusions + [f"{name} (missing value)" for name in data.left_out],
            "weights": task.weights,
            "paired": task.paired,
        },
        "evaluation": (
            f"Nested cross-validation split by participant: balanced accuracy "
            f"{selected['value']:.2f} [{selected['low']:.2f}, {selected['high']:.2f}] "
            f"(permutation p = {selected['p']:.3g}). "
            + metrics_entry["selected"]["note"]
            + " The configuration fixed in advance is reported next to it."
        ),
        "decision": (
            f"Calls the positive class at a Platt-scaled probability of {plan.threshold:g} or "
            f"more; no label between {low:g} and {high:g}, where the out-of-fold evidence cannot "
            "tell the probability from a coin flip"
            + (" (that is, never)" if (low, high) == (0.0, 1.0) else "")
            + f". Out of fold, {plan.performance['share_labelled']:.0%} of recordings got a label."
        ),
        "limitations": [
            "The curators of ds002778 warn against presenting classification of these patients "
            "as diagnostic: 15 patients are too few for reliable test statistics.",
            "One dataset, one site, one EEG system (BioSemi, 32 channels, 512 Hz); no external "
            "test set. Performance on other people or equipment is unknown.",
            "The intervals are wide, and the selected configuration's score is optimistic.",
            "OFF and ON differ in more than the medication; a stage 2 model learns whatever "
            "differs between the sessions.",
            "Recordings at another sampling rate are resampled, which the models never saw.",
        ],
        "disclaimer": config.predict.disclaimer,
    }


# --- training ------------------------------------------------------------------------------------


def training_registry(config: Config, recordings: pd.DataFrame) -> pd.DataFrame:
    """Every recording of the dataset, with the SHA-256 of its file and of its signal."""
    raw_root = config.data.paths.raw
    rows = []
    for row in recordings.to_dict("records"):
        with mne.use_log_level("ERROR"):
            raw = load_raw(SimpleNamespace(**row), raw_root, config.data.dataset.montage)
        rows.append(
            {
                "subject": row["subject"],
                "session": row["session"],
                "bids_path": row["bids_path"],
                "file_sha256": file_digest(raw_root / row["bids_path"]),
                "signal_sha256": signal_digest(raw, config.data.dataset.montage),
            }
        )
    return pd.DataFrame(rows, columns=list(REGISTRY_COLUMNS))


def fit_final(config: Config, plan: StagePlan) -> tuple[Pipeline, dict[str, Any], int]:
    """The stage's model tuned and fitted on all its participants, with Platt scaling."""
    data = plan.data
    tuned = search(
        plan.model,
        config.model,
        data.X.shape[1],
        seed_for(plan.seed, "final"),
        weighted=data.weights is not None,
    )
    n_warnings = fit_counting_warnings(tuned, data.X, data.y, data.groups, data.weights)
    seed = seed_for(plan.seed, "final platt")
    calibrated, more = platt(tuned, config.model, data.X, data.y, data.groups, data.weights, seed)
    return Pipeline([("calibrated", calibrated)]), tuned.best_params_, n_warnings + more


def metadata(
    config: Config,
    plan: StagePlan,
    best_params: Mapping[str, Any],
    n_warnings: int,
    metrics_entry: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = config.data.dataset
    yaml_digests = {
        name: file_digest(config.config_dir / name) for name in ("model.yaml", "predict.yaml")
    }
    return {
        "stage": plan.stage,
        "title": STAGE_TITLES[plan.stage],
        "task": plan.task,
        "feature_set": plan.feature_set,
        "model": plan.model,
        "hyperparameters": dict(best_params),
        "convergence_warnings": n_warnings,
        "classes": {
            "positive": "/".join(config.model.tasks[plan.task].positive),
            "negative": "/".join(config.model.tasks[plan.task].negative),
        },
        "paired": plan.data.paired,
        "segments": config.model.segments,
        "level": config.model.level,
        "config": {
            "feature_sha256": feature_digest(config),
            "model_settings_sha256": config_digest(config),
            **{f"{name.replace('.', '_')}_sha256": d for name, d in yaml_digests.items()},
        },
        "dataset": {"id": dataset.id, "version": dataset.version},
        "git": git_state(config.root),
        "versions": {
            "pdeeg": pdeeg.__version__,
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "mne": mne.__version__,
        },
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cross_validated": metrics_entry,
        "thresholds": {
            "threshold": plan.threshold,
            "uncertain": list(plan.band),
            "rule": dataclasses.asdict(config.predict.band),
            "out_of_fold": plan.performance,
        },
        "model_card": model_card(config, plan, metrics_entry),
    }


def train_final(
    config: Config,
    recordings: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    summary: pd.DataFrame,
    *,
    registry: pd.DataFrame | None = None,
    say: Say = print,
) -> dict[str, list[Path]]:
    """Check the configuration, derive the bands, fit and save every stage; returns its files.

    ``registry`` replaces the list of training recordings that is otherwise read from the raw
    dataset (for tests without it).
    """
    check_final(config, summary)
    cascade_path = cascade_dir(config) / "evaluation.json"
    if not cascade_path.is_file():
        raise FinalError(f"{cascade_path} not found; run `pdeeg cascade` first")
    cascade = json.loads(cascade_path.read_text(encoding="utf-8"))
    plans = []
    for stage in FINAL_STAGES:
        plan = plan_stage(config, recordings, tables, stage)
        say(
            f"{stage}: {plan.task} / {plan.feature_set} / {plan.model}, out-of-fold band "
            f"[{plan.band[0]:g}, {plan.band[1]:g}], "
            f"{plan.performance['share_labelled']:.0%} labelled"
        )
        plans.append(plan)
    check_bands(config, plans)

    models_dir = config.predict.models_dir
    registry = training_registry(config, recordings) if registry is None else registry
    written: dict[str, list[Path]] = {}
    for plan in plans:
        pipeline, best_params, n_warnings = fit_final(config, plan)
        entry = cross_validated(config, summary, plan, cascade)
        folder = models_dir / plan.stage
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, folder / "model.joblib")
        (folder / "feature_columns.json").write_text(
            json.dumps(plan.data.features, indent=1), encoding="utf-8"
        )
        plan.oof.to_parquet(folder / "oof_predictions.parquet", index=False)
        info = metadata(config, plan, best_params, n_warnings, entry)
        (folder / "metadata.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        written[plan.stage] = [folder / name for name in FILES]
        say(f"{plan.stage}: saved to {folder}")
    models_dir.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(models_dir / REGISTRY, index=False)
    return written
