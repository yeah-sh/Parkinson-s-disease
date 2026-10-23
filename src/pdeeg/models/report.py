"""The classification report: the ablation, its uncertainty, what the models used, and limits."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pdeeg.config import Config
from pdeeg.models.classify import AUDIT_ABOVE, best_result, weight_table
from pdeeg.models.cv import MODELS
from pdeeg.models.data import CONFOUND_COLUMNS
from pdeeg.models.evaluate import Evaluation
from pdeeg.stats.report import session_label
from pdeeg.viz.classification import plot_ablation
from pdeeg.viz.stats import Topography, montage_info, plot_topographies
from pdeeg.viz.style import save

MODEL_LABELS = {
    "logistic_regression": "logistic regression",
    "linear_svm": "linear SVM",
    "gradient_boosting": "gradient boosting",
}
_TOP = 10  # features listed per best model


def set_label(config: Config, feature_set: str) -> str:
    tables = config.model.feature_sets[feature_set]
    names = {"psd": "band power", "mfdfa": "MFDFA", "confounds": "age + sex"}
    return " + ".join(names[table] for table in tables)


def task_title(config: Config, task: str) -> str:
    settings = config.model.tasks[task]
    conditions = {s.condition: name for name, s in config.data.sessions.items()}
    positive = " / ".join(session_label(config, conditions[c]) for c in settings.positive)
    negative = " / ".join(session_label(config, conditions[c]) for c in settings.negative)
    return f"{positive} vs {negative}" + (", within patients" if settings.paired else "")


def _image(path: Path, alt: str, base: Path) -> str:
    return f"![{alt}]({Path(os.path.relpath(path, base)).as_posix()})"


def _markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _p(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def _with_ci(row: pd.Series, metric: str) -> str:
    return f"{row[metric]:.2f} [{row[f'{metric}_low']:.2f}, {row[f'{metric}_high']:.2f}]"


def ablation_rows(table: pd.DataFrame, config: Config, task: str) -> list[list[str]]:
    rows = []
    for row in table[table["task"] == task].to_dict("records"):
        row = pd.Series(row)
        rows.append(
            [
                set_label(config, row["feature_set"]),
                MODEL_LABELS[row["model"]],
                str(row["n_features"]),
                _with_ci(row, "balanced_accuracy"),
                _p(row["balanced_accuracy_p"]),
                _p(row["balanced_accuracy_q"]),
                _with_ci(row, "roc_auc"),
                _p(row["roc_auc_p"]),
                _with_ci(row, "sensitivity"),
                _with_ci(row, "specificity"),
                f"{row['repeat_min']:.2f}-{row['repeat_max']:.2f}",
            ]
        )
    return rows


def weight_topographies(
    result: Evaluation, config: Config, per_row: int = 5
) -> list[tuple[str, list[Topography]]]:
    """One head per feature type: the mean weight over outer folds, drawn on its sites."""
    weights = weight_table(result)
    weights = weights[weights["type"] != weights["feature"]]  # age and sex have no site
    channels = list(montage_info_channels(config))
    regions = config.data.regions
    linear = result.model != "gradient_boosting"
    panels = []
    for kind, rows in weights.groupby("type", sort=False):
        by_site = dict(zip(rows["site"], rows["mean"], strict=True))
        if config.model.level == "region":
            site_of_channel = {ch: region for region, chs in regions.items() for ch in chs}
            values = np.array([by_site.get(site_of_channel[ch], np.nan) for ch in channels])
        else:
            values = np.array([by_site.get(ch, np.nan) for ch in channels])
        panels.append(
            Topography(
                values=values,
                title=kind.replace("log_relative_power_", "power ").replace("envelope_", ""),
                kind="t" if linear else "mean",
                limits=(0.0, 0.0),
                interpolation="nearest" if config.model.level == "region" else "cubic",
            )
        )
    every = np.concatenate([p.values for p in panels])
    every = every[np.isfinite(every)]
    top = float(np.abs(every).max()) if every.size else 1.0
    limits = (-top, top) if linear else (0.0, top)
    panels = [Topography(p.values, p.title, p.kind, limits, None, p.interpolation) for p in panels]
    return [("", panels[i : i + per_row]) for i in range(0, len(panels), per_row)]


def montage_info_channels(config: Config) -> list[str]:
    """Every channel of the regions, in the order of data.yaml."""
    return [channel for channels in config.data.regions.values() for channel in channels]


def plain_summary(table: pd.DataFrame, results: list[Evaluation], config: Config) -> list[str]:
    """The plain-language summary, built from the numbers."""
    lines = []
    for task in config.model.tasks:
        rows = table[table["task"] == task]
        if rows.empty:
            continue
        real = rows[~rows["baseline"]]
        best = best_result(results, config, task)
        best_row = rows[
            (rows["feature_set"] == best.feature_set) & (rows["model"] == best.model)
        ].iloc[0]
        n_people = int(best_row["n_participants"])
        survived = real[real["balanced_accuracy_q"] < 0.05]
        nominal = real[real["balanced_accuracy_p"] < 0.05]
        text = (
            f"**{task_title(config, task)}** ({n_people} participants). The best of the "
            f"{len(real)} feature-based combinations was {set_label(config, best.feature_set)} "
            f"with {MODEL_LABELS[best.model]}: balanced accuracy "
            f"{_with_ci(best_row, 'balanced_accuracy')}, permutation p = "
            f"{_p(best_row['balanced_accuracy_p'])}. It was picked as the best on these same "
            f"predictions, which flatters it. {len(nominal)} of {len(real)} combinations had "
            f"p < 0.05 before correction and {len(survived)} after Benjamini-Hochberg across "
            "the task."
        )
        baseline = rows[rows["baseline"]]
        if not baseline.empty:
            top = baseline.sort_values("balanced_accuracy").iloc[-1]
            text += (
                f" Age and sex alone reached {_with_ci(top, 'balanced_accuracy')} "
                f"({MODEL_LABELS[top['model']]})."
            )
        text += (
            f" Over {config.model.cv.split_repeats} other outer splits, the balanced accuracy of "
            f"the best combination ranged from {best_row['repeat_min']:.2f} to "
            f"{best_row['repeat_max']:.2f}: the split alone moves it that much."
        )
        lines.append(f"- {text}")
    if table["audit"].any():
        lines.append(
            f"- **Audit.** Some balanced accuracy exceeds {AUDIT_ABOVE:.2f}; see the audit "
            "section before reading anything into it."
        )
    lines.append(
        "- **What this is not.** None of this is a diagnostic test. The intervals are wide "
        "because the sample is small, there is no independent test set, and the dataset's "
        "curators warn specifically against presenting classification of these patients as "
        "diagnostic. The group statistics (reports/stats_results.md) are the primary result."
    )
    return lines


def write_report(config: Config, results: list[Evaluation], table: pd.DataFrame) -> list[Path]:
    """Write the Markdown report and its figures; returns the report followed by the figures."""
    model = config.model
    report, base = model.report, model.report.parent
    figures = model.figures_dir
    tasks = {task: task_title(config, task) for task in model.tasks if task in set(table["task"])}
    labels = {
        f"{feature_set} / {name}": f"{set_label(config, feature_set)} · {MODEL_LABELS[name]}"
        for feature_set in model.feature_sets
        for name in MODELS
    }
    written = []
    ablation = figures / "ablation.png"
    save(
        plot_ablation(
            table,
            tasks,
            labels,
            "Balanced accuracy by feature set and model",
            "Grey: age and sex only. Tick: 95th percentile of the permutation null. "
            "Line at 0.5: chance.",
        ),
        ablation,
    )
    written.append(ablation)
    info = montage_info(montage_info_channels(config), config.data.dataset.montage)
    weight_figures = {}
    for task in tasks:
        best = best_result(results, config, task)
        path = figures / f"weights_{task}.png"
        what = (
            "mean coefficient on standardised features (positive: towards "
            f"{task_title(config, task).split(' vs ')[0]})"
            if best.model != "gradient_boosting"
            else "mean impurity importance"
        )
        save(
            plot_topographies(
                info,
                weight_topographies(best, config),
                f"What the best model used: {tasks[task]}",
                f"{set_label(config, best.feature_set)}, {MODEL_LABELS[best.model]}: {what} over "
                f"{len(best.predictions.weights)} outer folds, one patch per region.",
            ),
            path,
        )
        weight_figures[task] = (best, path)
        written.append(path)

    cv = model.cv
    digest = hashlib.sha256((config.config_dir / "model.yaml").read_bytes()).hexdigest()
    lines = [
        "# Classification (exploratory)",
        "",
        f"Written by `pdeeg classify`. Settings: "
        f"`configs/model.yaml`, SHA-256 `{digest}`, written before any classifier was run. "
        f"Every evaluation is an MLflow run in experiment `{model.tracking.experiment}` "
        f"(`{model.tracking.db.name}`, artifacts in `{model.tracking.artifacts.name}/`).",
        "",
        "Classification is secondary here. With about 30 people, the group statistics "
        "(`reports/stats_results.md`) are the primary result.",
        "",
        "## Summary in plain language",
        "",
        *plain_summary(table, results, config),
        "",
        "## Design",
        "",
    ]
    for task in tasks:
        settings = model.tasks[task]
        sample = next(r for r in results if r.task == task).data
        lines.append(
            f"- **{tasks[task]}:** {len(np.unique(sample.groups))} participants, "
            f"{len(sample.y)} recordings"
            + (
                f"; left out: {', '.join(settings.exclude_subjects)} (ON sessions exported "
                "already preprocessed)"
                if settings.exclude_subjects
                else ""
            )
            + (
                ". Each patient's features are centred on the mean of their two recordings, so "
                "the question is which of a patient's two recordings is OFF; labels are "
                "permuted within patients."
                if settings.paired
                else ". One recording per participant; labels are permuted across participants."
            )
        )
    n_features = table.groupby("feature_set")["n_features"].first()
    lines += [
        f"- **Features:** `{model.segments}` segments, {model.level} level: "
        + "; ".join(
            f"{set_label(config, name)} {n_features[name]}"
            for name in model.feature_sets
            if name in n_features
        )
        + f". Age and sex are {', '.join(CONFOUND_COLUMNS)} (sex 1 for m).",
        f"- **Outer loop:** {cv.outer.replace('_', ' ')}, {cv.outer_folds} folds, grouped by "
        "participant and stratified by class. Leave-one-subject-out was timed first and would "
        "have taken weeks of CPU with 1000 permutations of the nested procedure "
        "(configs/model.yaml has the numbers).",
        f"- **Inner loop:** {cv.inner_folds}-fold grouped and stratified, scored by "
        f"{cv.inner_scoring.replace('_', ' ')}, inside every outer training set. Scaling"
        + (", feature selection" if model.level == "channel" else "")
        + " and the model form one scikit-learn Pipeline, so all of it is fitted on training "
        "participants only. Grids: "
        + "; ".join(
            f"{MODEL_LABELS[name]} "
            + ", ".join(
                f"{key} {list(values)}"
                for key, values in vars(getattr(model.classifiers, name)).items()
                if key != "max_iter"
            )
            for name in MODELS
        )
        + ".",
        "- **No leakage across participants:** every outer and inner split is checked at run "
        "time to share no participant, and `tests/test_leakage.py` checks every splitter of "
        "every task.",
        f"- **Metrics:** on the pooled out-of-fold predictions. Intervals: {cv.n_bootstrap} "
        f"bootstrap resamples of participants, {cv.confidence:.0%}. p-values: "
        f"{cv.n_permutations} relabellings, each through the whole nested procedure; p = (1 + "
        "number at least as good) / (1 + relabellings). q: Benjamini-Hochberg over the "
        "balanced-accuracy p-values of a task. Sensitivity is the share of the first class "
        "found.",
        f"- **Other splits:** the whole procedure repeated on {cv.split_repeats} other outer "
        "splits; the range of balanced accuracy shows what the split alone does.",
        "",
        "## Results",
        "",
    ]
    header = [
        "Features",
        "Model",
        "n features",
        "Balanced accuracy [95 % CI]",
        "p",
        "q",
        "ROC AUC [95 % CI]",
        "p (AUC)",
        "Sensitivity [95 % CI]",
        "Specificity [95 % CI]",
        f"BA over {cv.split_repeats} other splits",
    ]
    for task in tasks:
        lines += [
            f"### {tasks[task]}",
            "",
            *_markdown_table(header, ablation_rows(table, config, task)),
            "",
        ]
    lines += [_image(ablation, "Balanced accuracy by feature set and model", base), ""]
    warned = table[table["convergence_warnings"] > 0]
    if not warned.empty:
        lines += [
            f"The saga solver did not converge within its iteration limit in "
            f"{int(warned['convergence_warnings'].sum())} fits (of the observed runs), in "
            + ", ".join(
                f"{row['task']} / {row['feature_set']} / {row['model']}"
                for row in warned.to_dict("records")
            )
            + ".",
            "",
        ]

    lines += ["## Audit", ""]
    flagged = table[table["audit"]]
    if flagged.empty:
        lines += [
            f"No balanced accuracy exceeds {AUDIT_ABOVE:.2f}, so no result called for the "
            "leakage and confound audit.",
            "",
        ]
    else:
        lines += [
            f"Balanced accuracy above {AUDIT_ABOVE:.2f} in: "
            + ", ".join(
                f"{r['task']} / {r['feature_set']} / {r['model']} ({r['balanced_accuracy']:.2f})"
                for r in flagged.to_dict("records")
            )
            + ". These are not reported as results until the audit is written here.",
            "",
        ]

    lines += ["## What the best models used", ""]
    for task, (best, path) in weight_figures.items():
        weights = weight_table(best).assign(magnitude=lambda t: t["mean"].abs())
        top = weights.sort_values("magnitude", ascending=False).head(_TOP)
        lines += [
            f"### {tasks[task]}",
            "",
            f"{set_label(config, best.feature_set)} with {MODEL_LABELS[best.model]}, the "
            "highest balanced accuracy of the task. "
            + (
                "Weights are coefficients on standardised features, averaged over the outer "
                "folds; with correlated features a coefficient is shared out arbitrarily among "
                "them, so read the pattern, not single values."
                if best.model != "gradient_boosting"
                else "Weights are impurity importances averaged over the outer folds."
            ),
            "",
            *_markdown_table(
                ["Feature", "Mean weight", "SD over folds", "Folds using it"],
                [
                    [
                        row["feature"],
                        f"{row['mean']:+.3f}",
                        f"{row['sd']:.3f}",
                        f"{row['folds_used']} of {row['folds']}",
                    ]
                    for row in top.to_dict("records")
                ],
            ),
            "",
            _image(path, f"Weights, {task}", base),
            "",
        ]

    sizes = {
        task: len(np.unique(next(r for r in results if r.task == task).data.groups))
        for task in tasks
    }
    lines += [
        "## Limitations",
        "",
        "- **Sample size.** "
        + "; ".join(f"{tasks[task]}: {n} participants" for task, n in sizes.items())
        + ". Each misclassified participant moves balanced accuracy by several points, the "
        "intervals are correspondingly wide, and with this many people a classifier can look "
        "good or bad by chance; the spread over other splits shows how much.",
        "- **The curators' caution.** The dataset's README warns against presenting "
        "machine-learning classification of these patients against controls as a diagnostic "
        "result, because 15 patients are too few for reliable test statistics.",
        "- **No external test set.** Every number comes from cross-validation within one "
        "dataset from one site and one EEG system. Nothing here has been checked on people "
        "recorded elsewhere, and performance there is unknown.",
        f"- **Many comparisons.** {len(table)} combinations were run; the q-values correct the "
        "p-values within a task, but the best combination is still selected on the data it is "
        "scored on.",
        "- **Uncertainty left out.** The bootstrap resamples participants with their "
        "predictions fixed, so it does not include how the models would change with other "
        "training data; the spread over other splits shows part of that.",
        "- **Pooled scores.** ROC AUC pools decision values of models fitted in different "
        "folds, whose scales can differ.",
        "- **Confounds.** Age and sex are tested as a baseline, not removed. Medication "
        "history, disease duration and recording conditions are not modelled, and anything "
        "that differs between groups at recording time can be learned as if it were disease.",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return [report, *written]
