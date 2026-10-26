"""The ablation: every task, feature set and model, and the tables summarising it."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from pdeeg.config import Config
from pdeeg.models.cv import MODELS
from pdeeg.models.data import CONFOUNDS, applicable, feature_type, site_of, task_data
from pdeeg.models.evaluate import METRICS, Evaluation, evaluate
from pdeeg.stats.cluster import benjamini_hochberg

# Balanced accuracy above which a result is audited for leakage and confounds before it is
# reported.
AUDIT_ABOVE = 0.85


def planned(config: Config) -> list[tuple[str, str, str]]:
    """Every (task, feature set, model) that is evaluated, in configuration order.

    A feature set that says nothing in a task (age and sex within patients) is left out.
    """
    model = config.model
    return [
        (task, feature_set, name)
        for task in model.tasks
        for feature_set in model.feature_sets
        if applicable(config, task, feature_set)
        for name in MODELS
    ]


def run_classification(
    config: Config,
    recordings: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    *,
    n_jobs: int | None = None,
    progress: Callable[[Evaluation], None] | None = None,
    plan: list[tuple[str, str, str]] | None = None,
) -> list[Evaluation]:
    """Evaluate everything ``plan`` (by default :func:`planned`) lists, in that order."""
    jobs = config.model.n_jobs if n_jobs is None else n_jobs
    results, data = [], {}
    for task, feature_set, name in planned(config) if plan is None else plan:
        if (task, feature_set) not in data:
            data[task, feature_set] = task_data(config, recordings, tables, task, feature_set)
        result = evaluate(data[task, feature_set], task, feature_set, name, config, n_jobs=jobs)
        if progress is not None:
            progress(result)
        results.append(result)
    return results


def is_baseline(config: Config, feature_set: str) -> bool:
    """Whether a feature set holds nothing but the confounds."""
    return set(config.model.feature_sets[feature_set]) == {CONFOUNDS}


def summary_table(results: list[Evaluation], config: Config) -> pd.DataFrame:
    """One row per evaluation: sizes, every metric with its interval and p-value, and more.

    ``balanced_accuracy_q`` is the Benjamini-Hochberg q-value of the balanced-accuracy p-value
    among the evaluations of the same task.
    """
    rows = []
    for result in results:
        data = result.data
        row = {
            "task": result.task,
            "feature_set": result.feature_set,
            "model": result.model,
            "baseline": is_baseline(config, result.feature_set),
            "n_participants": len(np.unique(data.groups)),
            "n_recordings": len(data.y),
            "n_features": data.X.shape[1],
            "convergence_warnings": result.predictions.convergence_warnings,
        }
        for metric in METRICS:
            low, high = result.intervals[metric]
            row |= {
                metric: result.observed[metric],
                f"{metric}_low": low,
                f"{metric}_high": high,
                f"{metric}_p": result.p_values[metric],
            }
        repeats = result.repeats
        row |= {
            "repeat_min": float(repeats.min()) if repeats.size else np.nan,
            "repeat_median": float(np.median(repeats)) if repeats.size else np.nan,
            "repeat_max": float(repeats.max()) if repeats.size else np.nan,
            "null_ba_95": float(result.null["balanced_accuracy"].quantile(0.95)),
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    table["balanced_accuracy_q"] = np.nan
    for _, chosen in table.groupby("task", sort=False):
        table.loc[chosen.index, "balanced_accuracy_q"] = benjamini_hochberg(
            chosen["balanced_accuracy_p"].to_numpy()
        )
    table["audit"] = table["balanced_accuracy"] > AUDIT_ABOVE
    return table


def best_result(results: list[Evaluation], config: Config, task: str) -> Evaluation:
    """The evaluation of ``task`` with the highest balanced accuracy (then AUC), baselines aside.

    It is picked on the same predictions it is scored on, so its score flatters it.
    """
    candidates = [r for r in results if r.task == task and not is_baseline(config, r.feature_set)]
    return max(
        candidates,
        key=lambda r: (r.observed["balanced_accuracy"], r.observed["roc_auc"]),
    )


def weight_table(result: Evaluation) -> pd.DataFrame:
    """Each feature's weight over the outer folds: mean, standard deviation, folds it was used in.

    Weights are coefficients on standardised features for the linear models (sign: towards the
    positive class) and impurity importances for gradient boosting.
    """
    weights = result.predictions.weights
    return pd.DataFrame(
        {
            "feature": result.data.features,
            "type": [feature_type(f) for f in result.data.features],
            "site": [site_of(f) for f in result.data.features],
            "mean": weights.mean(axis=0),
            "sd": weights.std(axis=0, ddof=1) if len(weights) > 1 else np.nan,
            "folds_used": (np.abs(weights) > 0).sum(axis=0),
            "folds": len(weights),
        }
    )
