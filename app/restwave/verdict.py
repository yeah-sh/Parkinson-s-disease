"""What the result of predict_files says, as the page says it. No Streamlit here.

A stage's entry in ``result["stages"]`` becomes a :class:`Call`: its kind (which colour and icon),
a headline and one sentence. How much to trust it comes from the stage's stored evaluation
(:func:`pdeeg.predict.predictor.stage_evidence`): every number in that line is read from it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pdeeg.predict.predictor import AUDIT_ABOVE, LABELS, UNCERTAIN

DETECTED, CLEAR, INCONCLUSIVE, NONE = "detected", "clear", "inconclusive", "none"
# What a gauge's two ends mean, per stage: (low end, high end).
ENDS = {
    "stage1": ("Healthy-like", "Parkinson's-like"),
    "stage2_single": ("ON-like", "OFF-like"),
    "stage2_paired": ("B looks OFF", "A looks OFF"),
}
SCORE_OF = {
    "stage1": "Parkinson's score",
    "stage2_single": "OFF-medication score",
    "stage2_paired": "Score for A being the OFF recording",
}
TRUTH = {
    "PD": "Parkinson's group",
    "HC": "healthy group",
    "OFF": "recorded off medication",
    "ON": "recorded on medication",
}


@dataclass(frozen=True)
class Trust:
    level: str  # "low" or "moderate"; never "high": nothing was tested outside one dataset
    line: str  # the sentence after "How much to trust this: <level>."


@dataclass(frozen=True)
class Call:
    kind: str  # DETECTED, CLEAR, INCONCLUSIVE or NONE
    eyebrow: str
    headline: str
    body: str
    entry: Mapping[str, Any]
    trust: Trust | None
    note: str | None = None


def entries(result: Mapping[str, Any], stage: str) -> list[Mapping[str, Any]]:
    return [entry for entry in result["stages"] if entry["stage"] == stage]


def never_labels(entry: Mapping[str, Any]) -> bool:
    """Whether the stage's uncertain band covers every score, so it never gives a label."""
    low, high = entry["uncertain_band"]
    return low <= 0.0 and high >= 1.0


def trust(stage: str, evidence: Mapping[str, Mapping[str, Any]], entry: Mapping[str, Any]) -> Trust:
    """How much to trust a stage, from its cross-validation (its stored evaluation).

    A balanced accuracy above the project's audit threshold (``audit_above``) is not reported
    until it has been audited (reports/classification_results.md), so it is not given here.
    """
    stored = evidence.get(stage) or {}
    cv = (stored or entry)["cross_validated"]
    audit = float(stored.get("audit_above", AUDIT_ABOVE))
    people = "patients" if stage.startswith("stage2") else "people"
    unaudited = cv["balanced_accuracy"] > audit
    if unaudited:
        line = (
            f"In testing on {cv['n_participants']} {people}, this step scored above {audit:.0%}, "
            "unusually high for so few people. The project checks such a score for mistakes "
            "before reporting it, and that check has not been done."
        )
    else:
        line = (
            f"In testing on {cv['n_participants']} {people}, this step was right "
            f"{cv['balanced_accuracy']:.0%} of the time (chance is {cv['chance']:.0%})."
        )
    if cv["better_than_chance"]:
        line += " That is better than chance, but it was tested on one small dataset only."
    else:
        line += (
            " It is not clearly better than chance."
            if unaudited
            else (" That is not clearly better than chance.")
        )
    if "uncertain_band" in entry and never_labels(entry):
        line += " Its scores overlap too much to label any single recording."
    return Trust("moderate" if cv["better_than_chance"] and not unaudited else "low", line)


def _skipped(entry: Mapping[str, Any], eyebrow: str) -> Call:
    return Call(
        NONE,
        eyebrow,
        "No result",
        f"This step did not run: {entry['skipped']}.",
        entry,
        None,
    )


def diagnosis(entry: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]]) -> Call:
    """Stage 1: Parkinson's pattern or not, or inconclusive."""
    eyebrow = "Result"
    if "skipped" in entry:
        return _skipped(entry, eyebrow)
    label = entry["label"]
    if label == UNCERTAIN:
        kind, headline = INCONCLUSIVE, "Inconclusive"
        body = "The signal sits between the two groups, so we don't call it either way."
    elif label == LABELS["stage1"][1]:
        kind, headline = DETECTED, "Parkinson's pattern detected"
        body = (
            "The brain-wave patterns in this recording look more like those of the people with "
            "Parkinson's disease the model learned from than those of the healthy people."
        )
    else:
        kind, headline = CLEAR, "No Parkinson's pattern detected"
        body = (
            "The brain-wave patterns in this recording look more like those of the healthy "
            "people the model learned from than those of the people with Parkinson's disease."
        )
    return Call(kind, eyebrow, headline, body, entry, trust("stage1", evidence, entry))


def medication(entry: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]]) -> Call:
    """Stage 2 on one recording: OFF or ON medication, or inconclusive."""
    eyebrow = "Medication"
    if "skipped" in entry:
        return _skipped(entry, eyebrow)
    label = entry["label"]
    if label == UNCERTAIN:
        kind, headline = INCONCLUSIVE, "Inconclusive"
        body = (
            "This step never gives a firm answer for a single recording: in testing, its scores "
            "for OFF and ON recordings overlapped too much."
            if never_labels(entry)
            else "The signal sits between the OFF- and ON-medication recordings the model "
            "learned from."
        )
    elif label == LABELS["stage2_single"][1]:
        kind, headline = DETECTED, "Pattern suggests OFF medication"
        body = "It looks more like the recordings made while patients were off their medication."
    else:
        kind, headline = CLEAR, "Pattern suggests ON medication"
        body = "It looks more like the recordings made while patients were on their medication."
    return Call(kind, eyebrow, headline, body, entry, trust("stage2_single", evidence, entry))


def paired(
    entry: Mapping[str, Any], names: Sequence[str], evidence: Mapping[str, Mapping[str, Any]]
) -> Call:
    """The paired stage 2: which of one patient's two recordings is the OFF-medication one."""
    eyebrow = "Medication · comparing the two recordings"
    if "skipped" in entry:
        return _skipped(entry, eyebrow)
    label = entry["label"]
    note = None
    if "note" in entry:
        note = (
            "Not both recordings showed a Parkinson's pattern. This comparison assumes they are "
            "one patient's OFF- and ON-medication recordings, as you said."
        )
    if label == UNCERTAIN:
        return Call(
            INCONCLUSIVE,
            eyebrow,
            "Inconclusive",
            "The two recordings look equally like the OFF-medication one.",
            entry,
            trust("stage2_paired", evidence, entry),
            note,
        )
    off, on = ("A", "B") if label == LABELS["stage2_paired"][1] else ("B", "A")
    first, second = names
    named = {"A": first, "B": second}
    return Call(
        DETECTED,
        eyebrow,
        f"Recording {off} looks like the OFF-medication recording",
        f"Side by side, recording {off} ({named[off]}) looks more like a recording made off "
        f"medication, and recording {on} ({named[on]}) like one made on medication.",
        entry,
        trust("stage2_paired", evidence, entry),
        note,
    )


def held_out_note(result: Mapping[str, Any]) -> str | None:
    """For a ds002778 recording: what kind of result this is, and the person's true group."""
    if not result.get("training_recording"):
        return None
    facts: dict[str, list[str]] = {}
    for entry in result["stages"]:
        if "true_label" in entry and entry["stage"] in ("stage1", "stage2_single"):
            truth = TRUTH.get(entry["true_label"], entry["true_label"])
            facts.setdefault(entry.get("recording", ""), []).append(truth)
    if len(facts) == 1:
        known = ", ".join(next(iter(facts.values())))
    else:
        known = "; ".join(f"{where}: {', '.join(parts)}" for where, parts in facts.items())
    return (
        "This recording is from the public dataset the models learned from. To keep the result "
        "fair, it is the test result from models that never saw this person."
        + (f" True group: {known}." if known else "")
    )
