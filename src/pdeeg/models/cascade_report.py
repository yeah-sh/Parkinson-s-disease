"""The cascade report: three-class calls, each stage's share of the errors, and calibration."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from pdeeg.config import Config
from pdeeg.models.calibration import reliability_table
from pdeeg.models.cascade import CHANCE, CascadeEvaluation, stage_probabilities
from pdeeg.models.report import (
    MODEL_LABELS,
    _image,
    _markdown_table,
    _p,
    set_label,
    task_title,
)
from pdeeg.viz.cascade import plot_confusion, plot_nulls, plot_reliability
from pdeeg.viz.style import save

ALPHA = 0.05  # a stage "beats chance" when p < ALPHA and the interval lies above chance
_REPEAT_LABELS = {
    "macro_balanced_accuracy": "Three classes, macro balanced accuracy",
    "stage1_balanced_accuracy": "Stage 1, balanced accuracy",
    "stage2_balanced_accuracy": "Stage 2 on true patients, balanced accuracy",
}
# Short panel titles of the reliability figure.
_PANELS = {"stage1": "Stage 1: patient or control", "stage2": "Stage 2: OFF or ON (patients)"}
_CALIBRATION_LABELS = {
    "brier": "Brier",
    "reference_brier": "Brier of the prevalence",
    "brier_skill": "Brier skill",
    "intercept": "Calibration in the large",
    "slope": "Calibration slope",
}


def _ci(value: float, interval) -> str:
    low, high = interval
    return f"{value:.2f} [{low:.2f}, {high:.2f}]"


def beats_chance(result: CascadeEvaluation, metric: str) -> bool:
    """p below ALPHA and the whole interval above chance: the test this report applies."""
    low, _ = result.intervals[metric]
    return result.p_values[metric] < ALPHA and low > CHANCE[metric]


def _verdict(result: CascadeEvaluation, metric: str) -> str:
    low, high = result.intervals[metric]
    p = result.p_values[metric]
    chance = CHANCE[metric]
    if beats_chance(result, metric):
        return f"above chance ({chance:.2f}): p = {_p(p)} and the interval excludes chance"
    reasons = []
    if not p < ALPHA:
        reasons.append(f"p = {_p(p)}")
    if not low > chance:
        reasons.append(f"the interval [{low:.2f}, {high:.2f}] includes chance ({chance:.2f})")
    return "not shown to beat chance: " + " and ".join(reasons)


def _null_95(result: CascadeEvaluation, metric: str) -> float:
    return float(result.null[metric].quantile(0.95))


def _stage_rows(summary: pd.DataFrame, config: Config, tasks: list[str]) -> list[list[str]]:
    cascade = config.model.cascade
    rows = []
    for task in tasks:
        chosen = summary[
            (summary["task"] == task)
            & (summary["feature_set"] == cascade.feature_set)
            & (summary["model"] == cascade.model)
        ]
        if chosen.empty:
            continue
        row = chosen.iloc[0]
        rows.append(
            [
                task_title(config, task),
                f"`{task}`",
                str(int(row["n_participants"])),
                str(int(row["n_recordings"])),
                _ci(
                    row["balanced_accuracy"],
                    (row["balanced_accuracy_low"], row["balanced_accuracy_high"]),
                ),
                _p(row["balanced_accuracy_p"]),
                _p(row["balanced_accuracy_q"]),
                _ci(row["roc_auc"], (row["roc_auc_low"], row["roc_auc_high"])),
            ]
        )
    return rows


def write_report(
    config: Config, result: CascadeEvaluation, summary: pd.DataFrame | None
) -> list[Path]:
    """Write the Markdown report and its figures; returns the report followed by the figures.

    ``summary`` is the classification summary table, for the stages' own tasks; without it
    that section says so.
    """
    model = config.model
    cascade, cv = model.cascade, model.cv
    report, base, figures = cascade.report, cascade.report.parent, cascade.figures_dir
    data, predictions = result.data, result.predictions
    classes = list(data.classes)
    observed, intervals = result.observed, result.intervals
    calibration = result.calibration
    stage_names = {
        "stage1": f"Stage 1: {task_title(config, cascade.stage1)}",
        "stage2": f"Stage 2: {task_title(config, cascade.stage2)} (true patients)",
    }

    written = []
    confusion_path = figures / "confusion.png"
    save(
        plot_confusion(
            result.confusion,
            classes,
            "What the cascade called each recording",
            "Out-of-fold calls; shade and percentage are shares of the true class.",
        ),
        confusion_path,
    )
    written.append(confusion_path)
    null_path = figures / "permutation_nulls.png"
    save(
        plot_nulls(
            {
                "Three classes, macro BA": (
                    result.null["macro_balanced_accuracy"],
                    observed["macro_balanced_accuracy"],
                    CHANCE["macro_balanced_accuracy"],
                ),
                "Stage 1, balanced accuracy": (
                    result.null["stage1_balanced_accuracy"],
                    observed["stage1_balanced_accuracy"],
                    CHANCE["stage1_balanced_accuracy"],
                ),
                "Stage 2 on patients, BA": (
                    result.null["stage2_balanced_accuracy"],
                    observed["stage2_balanced_accuracy"],
                    CHANCE["stage2_balanced_accuracy"],
                ),
            },
            "Observed scores against their permutation nulls",
            f"{len(result.null)} relabellings at the level of participants, each through the "
            "whole cascade. Dashed: chance.",
        ),
        null_path,
    )
    written.append(null_path)
    reliability = {}
    tables = {}
    for version in ("raw", "platt"):
        for stage, (y, p, _, w) in stage_probabilities(data, predictions, version).items():
            table = reliability_table(y, p, cascade.calibration.n_bins, w)
            tables[stage, version] = table
            reliability.setdefault(_PANELS[stage], {})[version] = table
    reliability_path = figures / "reliability.png"
    save(
        plot_reliability(
            reliability,
            "Are the probabilities calibrated?",
            f"Out-of-fold probabilities in {cascade.calibration.n_bins} bins of equal count; "
            "bars: 95 % Wilson intervals. Diagonal: perfect calibration.",
        ),
        reliability_path,
    )
    written.append(reliability_path)

    digest = hashlib.sha256((config.config_dir / "model.yaml").read_bytes()).hexdigest()
    n_people = len(np.unique(data.groups))
    per_class = np.bincount(data.label, minlength=3)
    people_per_class = [len(np.unique(data.groups[data.label == k])) for k in range(3)]
    stage1, stage2 = model.tasks[cascade.stage1], model.tasks[cascade.stage2]
    flagged = [stage for stage in ("stage1", "stage2") if calibration[stage]["raw"]["reasons"]]
    grid = getattr(model.classifiers, cascade.model)

    lines = [
        "# Two-stage cascade (exploratory)",
        "",
        "Written by `pdeeg cascade`. Settings: `configs/model.yaml` (section `cascade`, tasks "
        f"`{cascade.stage1}` and `{cascade.stage2}`), SHA-256 `{digest}`. The cascade and both "
        "of its tasks were added on 2026-10-26, after the pre-registration of 2026-10-23 and "
        "before any classifier had been run on the data. The evaluation is an MLflow run "
        f"(`cascade/{result.feature_set}/{result.model}`) in experiment "
        f"`{model.tracking.experiment}`.",
        "",
        "Like all classification here, this is secondary to the group statistics "
        "(`reports/stats_results.md`).",
        "",
        "## Summary in plain language",
        "",
        f"- **The cascade** called every recording {', '.join(classes[:-1])} or {classes[-1]}. "
        f"Macro balanced accuracy (the mean of the three sensitivities; chance "
        f"{CHANCE['macro_balanced_accuracy']:.2f}) was "
        f"{_ci(observed['macro_balanced_accuracy'], intervals['macro_balanced_accuracy'])}: "
        f"{_verdict(result, 'macro_balanced_accuracy')}.",
        f"- **Stage 1**, patient or control: balanced accuracy "
        f"{_ci(observed['stage1_balanced_accuracy'], intervals['stage1_balanced_accuracy'])}, "
        f"{_verdict(result, 'stage1_balanced_accuracy')}.",
        f"- **Stage 2**, OFF or ON, on the true patients' recordings whatever stage 1 said: "
        f"balanced accuracy "
        f"{_ci(observed['stage2_balanced_accuracy'], intervals['stage2_balanced_accuracy'])}, "
        f"{_verdict(result, 'stage2_balanced_accuracy')}.",
        f"- **Where patients' recordings went:** {observed['pd_missed']:.0%} were called "
        f"controls by stage 1, {observed['pd_wrong_state']:.0%} were called patients in the "
        f"wrong state by stage 2, and {observed['pd_correct']:.0%} were called right. "
        f"{observed['hc_called_pd']:.0%} of the controls' recordings were called patients'.",
        "- **Calibration:** "
        + (
            "by the rule fixed in advance, "
            + " and ".join(stage_names[s].split(":")[0].lower() for s in flagged)
            + " gave poorly calibrated probabilities, so Platt-scaled probabilities are the ones "
            "to use there; after scaling, the rule finds "
            + "; ".join(
                f"{stage_names[s].split(':')[0].lower()} "
                + ("well calibrated" if not calibration[s]["platt"]["reasons"] else "still flagged")
                for s in flagged
            )
            + "."
            if flagged
            else "by the rule fixed in advance, neither stage's probabilities are poorly "
            "calibrated, so the raw probabilities are used."
        ),
        "- **What this is not.** Not a diagnostic test and not a medication monitor: the "
        "sample is small, the intervals are wide, there is no independent test set, and the "
        "dataset's curators warn against presenting classification of these patients as "
        "diagnostic.",
        "",
        "## Design",
        "",
        "- **Why it was added.** The tool is meant to answer two questions about one recording: "
        "is it a patient's or a control's, and if a patient's, OFF or ON medication? The "
        f"pre-registered tasks do not fit. `pd_vs_hc` uses only OFF recordings, while a "
        "patient seen in practice is usually ON. `off_vs_on` centres each patient's features "
        "on their own two recordings, so it needs both and cannot label one recording on its "
        f"own. Hence two tasks: `{cascade.stage1}` (stage 1) and `{cascade.stage2}` (stage 2, "
        "single-recording mode; `off_vs_on` stays as the paired mode).",
        f"- **Recordings:** {n_people} participants, {len(data.label)} recordings: "
        + ", ".join(
            f"{name} {count} ({people} participants)"
            for name, count, people in zip(classes, per_class, people_per_class, strict=True)
        )
        + ". "
        + (
            "Left out: "
            + ", ".join(f"{s} {t}" for s, t in stage1.exclude_recordings)
            + " (exported already preprocessed); those patients' other recordings stay in. "
            if stage1.exclude_recordings
            else ""
        )
        + (
            f"Stage 2 never trains on {', '.join(stage2.exclude_subjects)}, for the same "
            "reason; their OFF recordings are still called by the cascade when held out."
            if stage2.exclude_subjects
            else ""
        )
        + (f" Left out for a missing value: {', '.join(data.left_out)}." if data.left_out else ""),
        f"- **Each outer fold** ({cv.outer.replace('_', ' ')}, {cv.outer_folds} folds, grouped "
        "by participant, stratified by patient or control): stage 1 is tuned and fitted on all "
        "recordings of the training participants"
        + (
            ", weighted so that each participant counts once (0.5 per recording of a patient "
            "with two)"
            if stage1.weights == "per_participant"
            else ""
        )
        + "; stage 2 on the training patients' recordings only. Each has its own "
        f"{cv.inner_folds}-fold inner loop, grouped by participant and scored by "
        f"{cv.inner_scoring.replace('_', ' ')}. Every held-out recording is called "
        f"{classes[0]} when stage 1 says so, else {classes[1]} or {classes[2]} by stage 2. "
        "Stage 2 also calls the recordings stage 1 sends to the controls, so its errors on true "
        "patients are measured apart from stage 1's. Every split is checked at run time to "
        "share no participant between training and test, for both stages.",
        f"- **Features and model:** {set_label(config, cascade.feature_set)}, "
        f"{data.X.shape[1]} features ({model.segments} segments, {model.level} level), and "
        f"{MODEL_LABELS[cascade.model]} for both stages, with the grid "
        + ", ".join(
            f"{key} {list(values)}" for key, values in vars(grid).items() if key != "max_iter"
        )
        + ". This one configuration was fixed before any result; nothing was picked on these "
        "data.",
        "- **Metrics:** sensitivity of a class is the share of its recordings called that "
        "class; macro balanced accuracy is their mean (chance 1/3). Stage 1's balanced "
        "accuracy and ROC AUC weigh each participant once; stage 2's are on the true patients' "
        "recordings.",
        f"- **Intervals:** {cv.n_bootstrap} bootstrap resamples of participants (with all their "
        f"recordings), {cv.confidence:.0%}, on the out-of-fold calls.",
        f"- **p-values:** {cv.n_permutations} relabellings at the level of participants, each "
        "through the whole cascade, inner loops included. Patient or control is shuffled among "
        "the participants; every participant who is then a patient gets OFF and ON by a coin "
        "flip (one of each with two recordings, OFF or ON with one). p = (1 + number at least "
        "as good) / (1 + relabellings). Stage 1's and stage 2's p-values come from the same "
        "relabellings.",
        f"- **Other splits:** the cascade repeated on {cv.split_repeats} other outer splits, "
        "to show how much the split alone moves the scores.",
        "",
        "## Results",
        "",
        "### Three classes",
        "",
        *_markdown_table(
            ["True class \\ called", *classes, "Sensitivity [95 % CI]"],
            [
                [
                    name,
                    *(str(int(v)) for v in result.confusion[k]),
                    _ci(observed[f"sensitivity_{name}"], intervals[f"sensitivity_{name}"]),
                ]
                for k, name in enumerate(classes)
            ],
        ),
        "",
        f"Macro balanced accuracy "
        f"{_ci(observed['macro_balanced_accuracy'], intervals['macro_balanced_accuracy'])}, "
        f"p = {_p(result.p_values['macro_balanced_accuracy'])} (95th percentile of the null: "
        f"{_null_95(result, 'macro_balanced_accuracy'):.2f}; chance "
        f"{CHANCE['macro_balanced_accuracy']:.2f}).",
        "",
        _image(confusion_path, "Confusion matrix of the cascade", base),
        "",
        "### Each stage inside the cascade",
        "",
        *_markdown_table(
            [
                "Stage",
                "Balanced accuracy [95 % CI]",
                "p",
                "ROC AUC [95 % CI]",
                "p (AUC)",
                "Null 95th percentile (BA)",
            ],
            [
                [
                    stage_names[stage],
                    _ci(
                        observed[f"{stage}_balanced_accuracy"],
                        intervals[f"{stage}_balanced_accuracy"],
                    ),
                    _p(result.p_values[f"{stage}_balanced_accuracy"]),
                    _ci(observed[f"{stage}_roc_auc"], intervals[f"{stage}_roc_auc"]),
                    _p(result.p_values[f"{stage}_roc_auc"]),
                    f"{_null_95(result, f'{stage}_balanced_accuracy'):.2f}",
                ]
                for stage in ("stage1", "stage2")
            ],
        ),
        "",
        "### Where the errors come from",
        "",
        *_markdown_table(
            ["Recordings", "Share [95 % CI]"],
            [
                [text, _ci(observed[name], intervals[name])]
                for name, text in (
                    ("pd_missed", "Patients' called controls (stage 1 missed)"),
                    ("pd_wrong_state", "Patients' called patients in the wrong state"),
                    ("pd_correct", "Patients' called right"),
                    ("hc_called_pd", "Controls' called patients (stage 1 false alarm)"),
                )
            ],
        ),
        "",
        "### Other outer splits",
        "",
        *_markdown_table(
            ["Metric", "Minimum", "Median", "Maximum"],
            [
                [
                    _REPEAT_LABELS[name],
                    f"{result.repeats[name].min():.2f}",
                    f"{result.repeats[name].median():.2f}",
                    f"{result.repeats[name].max():.2f}",
                ]
                for name in result.repeats.columns
            ],
        ),
        "",
        _image(null_path, "Permutation nulls", base),
        "",
    ]
    if predictions.convergence_warnings:
        lines += [
            f"The saga solver did not converge within its iteration limit in "
            f"{predictions.convergence_warnings} fits of the observed run.",
            "",
        ]

    lines += [
        "## Calibration",
        "",
        "Out-of-fold probabilities: P(patient) from stage 1 for every recording (weighted as "
        "stage 1), P(OFF) from stage 2 for the true patients' recordings. Platt scaling "
        f"({cascade.calibration.method}) was fitted inside every outer training set, on the "
        "stage's grouped inner-fold predictions, and computed for every run whatever the "
        "result. The rule, fixed before any classifier was run: a stage is poorly calibrated "
        "if its Brier score is not below that of predicting the prevalence, or the interval of "
        "its calibration slope leaves out 1, or that of calibration in the large leaves out 0.",
        "",
    ]
    header = ["Stage", "Probabilities", *_CALIBRATION_LABELS.values(), "Rule"]
    rows = []
    for stage in ("stage1", "stage2"):
        for version in ("raw", "platt"):
            entry = calibration[stage][version]
            cells = [
                f"{entry['summary'][m]['value']:.3f}"
                if m == "reference_brier"
                else (
                    f"{entry['summary'][m]['value']:.3f} [{entry['summary'][m]['low']:.3f}, "
                    f"{entry['summary'][m]['high']:.3f}]"
                )
                for m in _CALIBRATION_LABELS
            ]
            verdict = "; ".join(entry["reasons"]) or "well calibrated"
            label = "Platt" if version == "platt" else "raw"
            rows.append([stage_names[stage].split(":")[0], label, *cells, verdict])
    brier3 = calibration["three_class_brier"]
    lines += [
        *_markdown_table(header, rows),
        "",
        f"Three-class Brier score of the cascade's probabilities (P(HC) = 1 - p1, P(OFF) = "
        f"p1 p2, P(ON) = p1 (1 - p2); 0 is perfect, 2 the worst): raw {brier3['raw']:.3f}, "
        f"Platt-scaled {brier3['platt']:.3f}; predicting the class shares for everyone gives "
        f"{brier3['reference']:.3f}.",
        "",
        _image(reliability_path, "Reliability diagrams", base),
        "",
    ]
    for stage in ("stage1", "stage2"):
        for version in ("raw", "platt"):
            table = tables[stage, version]
            lines += [
                f"<details><summary>{stage_names[stage]}, "
                f"{'Platt-scaled' if version == 'platt' else 'raw'} probabilities by bin"
                "</summary>",
                "",
                *_markdown_table(
                    ["Bin", "Recordings", "Mean probability", "Observed [95 % CI]"],
                    [
                        [
                            str(int(row["bin"]) + 1),
                            str(int(row["n_recordings"])),
                            f"{row['mean_probability']:.2f}",
                            f"{row['observed']:.2f} [{row['observed_low']:.2f}, "
                            f"{row['observed_high']:.2f}]",
                        ]
                        for row in table.to_dict("records")
                    ],
                ),
                "",
                "</details>",
                "",
            ]
    if flagged:
        platt = calibration["platt_cascade"]
        lines += [
            "**Secondary: the cascade called from the Platt-scaled probabilities** (the rule "
            "flags "
            + " and ".join(stage_names[s].split(":")[0].lower() for s in flagged)
            + "). The same models, rescaled, so this changes where the 0.5 threshold falls, "
            "not how the recordings are ranked. It was not permutation-tested.",
            "",
            *_markdown_table(
                ["True class \\ called", *classes, "Sensitivity [95 % CI]"],
                [
                    [
                        name,
                        *(str(int(v)) for v in platt["confusion"][k]),
                        _ci(
                            platt["observed"][f"sensitivity_{name}"],
                            platt["intervals"][f"sensitivity_{name}"],
                        ),
                    ]
                    for k, name in enumerate(classes)
                ],
            ),
            "",
            "Macro balanced accuracy "
            + "; ".join(
                f"{text}{_ci(platt['observed'][name], platt['intervals'][name])}"
                for name, text in (
                    ("macro_balanced_accuracy", ""),
                    ("stage1_balanced_accuracy", "stage 1 "),
                    ("stage2_balanced_accuracy", "stage 2 on patients "),
                )
            )
            + ".",
            "",
        ]

    lines += ["## The stages on their own", ""]
    tasks = [cascade.stage1, "pd_vs_hc", cascade.stage2, "off_vs_on"]
    tasks = [task for task in dict.fromkeys(tasks) if task in model.tasks]
    stage_rows = [] if summary is None else _stage_rows(summary, config, tasks)
    if stage_rows:
        lines += [
            f"The same configuration ({set_label(config, cascade.feature_set)}, "
            f"{MODEL_LABELS[cascade.model]}) in the classification ablation "
            "(`reports/classification_results.md`, which has every feature set and model), "
            "each task with its own outer split and permutation test. The pre-registered "
            "`pd_vs_hc` and the paired `off_vs_on` are shown for comparison.",
            "",
            *_markdown_table(
                [
                    "Task",
                    "Name",
                    "Participants",
                    "Recordings",
                    "Balanced accuracy [95 % CI]",
                    "p",
                    "q",
                    "ROC AUC [95 % CI]",
                ],
                stage_rows,
            ),
            "",
        ]
    else:
        lines += [
            "The classification summary was not found; run `pdeeg classify` for the stages' "
            "own tasks.",
            "",
        ]

    n_stage2 = [len(fold["stage2_train"]) for fold in predictions.folds]
    lines += [
        "## Limitations",
        "",
        f"- **Sample size.** {n_people} participants; stage 2 was trained on "
        f"{min(n_stage2)} to {max(n_stage2)} patients per outer fold. Each misclassified "
        "recording moves a sensitivity by several points, and the intervals are wide.",
        "- **The curators' caution.** The dataset's README warns against presenting "
        "machine-learning classification of these patients against controls as a diagnostic "
        "result, because 15 patients are too few for reliable test statistics.",
        "- **No external test set.** Every number comes from cross-validation within one "
        "dataset from one site and one EEG system.",
        "- **Medication and everything else.** Stage 2 learns whatever differs between a "
        "patient's two sessions. Anything other than the medication that differs "
        "systematically between OFF and ON sessions would be learnt as well.",
        "- **The null.** Shuffling patient and control among participants also moves the "
        "number of recordings: under the null a control can become a 'patient' with one "
        "recording, whose OFF or ON label is a coin flip. The p-values are about features "
        "carrying no information on the labels, not about this design alone.",
        "- **Calibration on few recordings.** The reliability bins hold about "
        f"{len(data.label) // cascade.calibration.n_bins} (stage 1) and "
        f"{int((data.label != 0).sum()) // cascade.calibration.n_bins} (stage 2) recordings "
        "each, and the intercept and slope have wide intervals.",
        "- **Fixed predictions in the bootstrap.** The intervals are about which participants "
        "were sampled, not about how the models would change with other training data; the "
        "other splits show part of that.",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    return [report, *written]
