"""Model Evaluation: what each stage said, what kind of prediction it is, and its evidence.

Every number is from pdeeg.predict.predictor: ``predict_files`` for this recording,
``stage_evidence`` (the saved models' metadata) for the cross-validation.
"""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from neurolab import charts, engine
from neurolab import components as ui
from neurolab.glossary import TERMS
from pdeeg.viz.predict import interval_bar

ui.page_header("evaluation")
config = engine.settings()
evidence = engine.evidence()
found = engine.analysis()

TITLES = {
    "stage1": "Stage 1 · Parkinson's vs healthy",
    "stage2_single": "Stage 2 · OFF vs ON medication",
    "stage2_paired": "Stage 2, paired · which recording is OFF medication",
}
# The class at 0 and the class at 1 of each stage's score.
ENDS = {
    "stage1": ("healthy", "Parkinson's"),
    "stage2_single": ("ON medication", "OFF medication"),
}
# The true classes of the out-of-fold recordings, negative then positive.
TRUE_CLASSES = {
    "stage1": ("Healthy (HC)", "Parkinson's (PD)"),
    "stage2_single": ("ON medication", "OFF medication"),
    "stage2_paired": ("ON recording", "OFF recording"),
}
SHOWN = {"PD": "Parkinson's", "HC": "healthy", "OFF": "OFF medication", "ON": "ON medication"}


def ends(stage: str, names: list[str]) -> tuple[str, str]:
    if stage == "stage2_paired":
        return (f"{names[1]} is OFF", f"{names[0]} is OFF")
    return ENDS[stage]


def band_text(threshold: float, low: float, high: float) -> str:
    if (low, high) == (0.0, 1.0):
        return (
            "This stage never gives a label: out of fold, none of its scores could be told "
            "from a coin flip."
        )
    if low == high:
        return (
            f"Threshold {threshold:g}. Out of fold, the scores on either side of it backed "
            "their label, so this stage has no uncertain band."
        )
    return (
        f"Threshold {threshold:g}; between {low:g} and {high:g} (shaded) the stage gives no "
        "label, because there its out-of-fold scores could not be told from a coin flip."
    )


def verdict(cv: dict[str, Any]) -> None:
    if cv["better_than_chance"]:
        ui.verdict(
            True,
            "Better than chance in cross-validation",
            f"By this project's rule: {cv['test']} (q = {cv['q']:.3f}). That says the stage "
            "learned something on this dataset, not that it would work on other people.",
        )
    else:
        ui.verdict(
            False,
            "Not better than chance in cross-validation",
            f"{'; '.join(cv['not_better_because']).capitalize()}. Its output is greyed out and "
            "should not be read as saying anything about this recording.",
        )


def reliability(cv: dict[str, Any], greyed: bool) -> None:
    """The stage's cross-validated balanced accuracy, interval, chance and p-value."""
    low, high = cv["ci95"]
    columns = st.columns(4, gap="small")
    columns[0].metric(
        "Balanced accuracy", f"{cv['balanced_accuracy']:.2f}", help=TERMS["Balanced accuracy"]
    )
    columns[1].metric("95% CI", f"{low:.2f} – {high:.2f}", help=TERMS["Confidence interval"])
    columns[2].metric("Chance", f"{cv['chance']:.2f}", help="Balanced accuracy of guessing.")
    columns[3].metric(
        "Permutation p",
        f"{cv['permutation_p']:.3f}",
        help=f"{TERMS['Permutation test']} q = {cv['q']:.3f}: {TERMS['q-value']}",
    )
    st.html(interval_bar(cv["balanced_accuracy"], low, high, cv["chance"], greyed=greyed))
    pre = cv["prespecified_configuration"]
    lines = [
        f"Cross-validation split by participant: {cv['n_participants']} people, "
        f"{cv['n_recordings']} recordings. {cv['note']}",
        f"The configuration fixed before any classifier was run ({pre['model']} on "
        f"{pre['feature_set']}): {pre['balanced_accuracy']:.2f} "
        f"[{pre['ci95'][0]:.2f}, {pre['ci95'][1]:.2f}].",
    ]
    if "in_cascade" in cv:
        cascade = cv["in_cascade"]
        lines.append(
            f"Inside the two-stage cascade: {cascade['balanced_accuracy']:.2f} "
            f"[{cascade['ci95'][0]:.2f}, {cascade['ci95'][1]:.2f}], permutation p = "
            f"{cascade['permutation_p']:.3f}."
        )
    st.caption(ui.md(" ".join(lines)))


def badge(stage: dict[str, Any], stage_ends: tuple[str, str]) -> None:
    if stage["uncertain"]:
        st.badge("Uncertain · no label", icon=":material/help:", color="orange")
        return
    label = stage["label"]
    called = stage_ends[1] if label in ("PD", "OFF", "first recording is OFF") else stage_ends[0]
    st.badge(SHOWN.get(label, called), icon=":material/label:", color="blue")


def stage_card(stage: dict[str, Any], key: str, names: list[str]) -> None:
    name = stage["stage"]
    where = stage.get("recording") or ", ".join(stage.get("recordings", [])) or "the pair"
    with st.container(key=f"nl-card-stage-{key}"):
        top = st.columns([4, 2], vertical_alignment="center")
        top[0].markdown(f"#### {TITLES[name]}")
        top[0].caption(ui.md(where))
        if "skipped" in stage:
            with top[1]:
                st.badge("Not run", icon=":material/block:", color="gray")
            st.markdown(f"**Not run:** {ui.md(stage['skipped'])}.")
            return
        cv = stage["cross_validated"]
        greyed = not cv["better_than_chance"]
        stage_ends = ends(name, names)
        low, high = stage["uncertain_band"]
        with top[1]:
            badge(stage, stage_ends)
        verdict(cv)
        st.space("small")
        with st.container(key=f"nl-{'greyed' if greyed else 'plain'}-{key}"):
            left, right = st.columns([3, 2], gap="medium")
            with left:
                context = evidence.get(name)
                this = "This pair" if name == "stage2_paired" else "This recording"
                if stage.get("out_of_fold"):
                    this += " (out of fold)"
                if context is not None:
                    st.altair_chart(
                        charts.probability(
                            context["oof"],
                            TRUE_CLASSES[name],
                            stage["threshold"],
                            stage["uncertain_band"],
                            stage["probability"],
                            greyed=greyed,
                            this_label=this,
                        ),
                        theme=None,
                        width="stretch",
                    )
                    st.caption(
                        "Dots: the out-of-fold scores of the training recordings, by their "
                        "true class, from models that never saw that participant. The vertical "
                        f"line: {this.lower()}."
                        + (
                            " The paired score is the share of the pair's two scores given to "
                            "the first recording."
                            if name == "stage2_paired"
                            else ""
                        )
                    )
            with right:
                what = "the first recording being OFF" if name == "stage2_paired" else stage_ends[1]
                st.markdown(f"Model score for {ui.md(what)}: **{stage['probability']:.2f}**")
                st.caption(TERMS["Model score"])
                st.markdown(band_text(stage["threshold"], low, high))
                if stage.get("out_of_fold"):
                    st.markdown(
                        f"Out of fold (fold {stage['fold']}); the true label is "
                        f"**{SHOWN.get(stage['true_label'], stage['true_label'])}**."
                    )
                    st.caption(
                        "Out-of-fold results come from models fitted during cross-validation; "
                        "which features moved them is not stored."
                    )
                if "note" in stage:
                    st.warning(ui.md(stage["note"]).capitalize() + ".", icon=":material/info:")
        with st.expander("How reliable is this?", expanded=True, icon=":material/verified:"):
            reliability(cv, greyed)
        if "top_features" in stage:
            top_features = stage["top_features"]
            with st.expander("Features that moved this score", icon=":material/stacked_bar_chart:"):
                st.caption(
                    ui.md(top_features["kind"]).capitalize()
                    + ": the "
                    + str(len(top_features["features"]))
                    + " largest, as `pdeeg predict` reports them. They describe what the "
                    "model used, not why a person would have Parkinson's disease."
                )
                st.altair_chart(
                    charts.contributions(top_features, stage_ends, greyed),
                    theme=None,
                    width="stretch",
                )


PROVENANCE = (
    (
        "external",
        "External, unseen recording",
        "Not recognised as a training recording (no match by file hash, signal hash or BIDS "
        "name), so scored by the final models trained on every eligible participant. The same "
        "person could still be in the training data under another recording.",
    ),
    (
        "held_out",
        "Held-out participant",
        "Scored by a model fitted without any recording of this person: cross-validation "
        "folds are split by participant.",
    ),
    (
        "oof",
        "Out-of-fold prediction",
        "The score this recording got inside cross-validation, from the model of the fold that "
        "left it out. The evaluation metrics are computed from these.",
    ),
    (
        "training",
        "Prediction on training data",
        "A final model on a recording it was fitted on: optimistic, so never shown here. A "
        "training recording gets its out-of-fold result instead.",
    ),
)


def provenance(training: bool) -> None:
    active = {"held_out", "oof"} if training else {"external"}
    cells = []
    for kind, title, text in PROVENANCE:
        if kind == "training":
            state = "Never shown"
        else:
            state = "This result" if kind in active else "Not this result"
        cells.append(
            f'<div class="nl-prov{" active" if kind in active else ""}">'
            f'<div class="t">{ui.icon("check" if kind in active else "dot")}{escape(title)}'
            f'</div><div class="d"><b>{state}.</b> {escape(text)}</div></div>'
        )
    st.html(f'<div class="nl-provenance">{"".join(cells)}</div>')


def this_recording() -> None:
    if found is None:
        ui.need_analysis("evaluation")
        return
    result = found.result
    if result is None:
        if found.failed_step == "evaluate":
            st.warning(
                f"**No prediction.** {ui.md(found.error or '')}. The recording was still "
                "cleaned and described; the other pages show it.",
                icon=":material/info:",
            )
        else:
            st.error(f"**No prediction.** {ui.md(found.error or '')}")
        return
    names = [name for name, _ in engine.files()]
    ui.section("What kind of prediction this is")
    provenance(result["training_recording"])
    if result["training_recording"]:
        matches = [entry["training_recording"] for entry in result["inputs"]]
        which = ", ".join(
            f"sub-{m['subject']} ses-{m['session']} (recognised by {m['matched_by']})"
            for m in matches
        )
        st.info(f"**A ds002778 training recording: {ui.md(which)}.** {result['explanation']}")
    st.space("small")
    ui.section("Stages")
    for position, stage in enumerate(result["stages"]):
        stage_card(stage, str(position), names)


def model_evidence() -> None:
    if not evidence:
        ui.empty(
            "No trained models",
            "Run `pdeeg train-final` to train the final models; their cross-validated "
            "evaluation is shown here.",
        )
        return
    st.caption(
        "How each saved model did in nested cross-validation split by participant, read from "
        "its metadata. None of this depends on the recording loaded."
    )
    for stage, info in evidence.items():
        cv = info["cross_validated"]
        greyed = not cv["better_than_chance"]
        card = info["model_card"]
        training = card["training_data"]
        with st.container(key=f"nl-card-evidence-{stage}"):
            top = st.columns([4, 2], vertical_alignment="center")
            top[0].markdown(f"#### {TITLES[stage]}")
            top[0].caption(ui.md(f"{card['model']} · task {card['task']}"))
            with top[1]:
                if greyed:
                    st.badge("Not better than chance", icon=":material/warning:", color="orange")
                else:
                    st.badge("Better than chance", icon=":material/verified:", color="blue")
            verdict(cv)
            if cv["balanced_accuracy"] > info["audit_above"]:
                st.warning(
                    f"Balanced accuracy above {info['audit_above']:g}: the project audits such "
                    "a result for leakage and confounds before reporting it "
                    "(reports/classification_results.md). Treat it as unaudited.",
                    icon=":material/policy:",
                )
            st.space("small")
            left, right = st.columns([1, 1], gap="medium")
            with left:
                st.markdown("**Selected configuration, cross-validated**")
                st.altair_chart(
                    charts.forest(info["metrics"], cv["chance"], greyed),
                    theme=None,
                    width="stretch",
                )
                counts = ", ".join(
                    f"{k} {v}" for k, v in training["recordings_by_condition"].items()
                )
                st.caption(
                    ui.md(
                        f"{training['participants']} participants, {training['recordings']} "
                        f"recordings ({counts}). Permutation p = {cv['permutation_p']:.3f}; "
                        f"q = {cv['q']:.3f}. {cv['note']}"
                    )
                )
            with right:
                st.markdown("**Out-of-fold scores, by true class**")
                st.altair_chart(
                    charts.probability(
                        info["oof"],
                        TRUE_CLASSES[stage],
                        info["threshold"],
                        info["uncertain_band"],
                        None,
                        greyed=greyed,
                    ),
                    theme=None,
                    width="stretch",
                )
                labelled = info["out_of_fold_labelling"]
                st.caption(
                    f"{labelled['n_labelled']} of {labelled['n_recordings']} recordings "
                    f"({labelled['share_labelled']:.0%}) fall outside the uncertain band and get "
                    "a label; "
                    + (
                        f"their balanced accuracy is {labelled['balanced_accuracy_labelled']:.2f}."
                        if labelled["n_labelled"]
                        else "so the stage never labels anything."
                    )
                )
            pre = cv["prespecified_configuration"]
            extra = (
                f"Configuration fixed in advance ({pre['model']} on {pre['feature_set']}): "
                f"{pre['balanced_accuracy']:.2f} [{pre['ci95'][0]:.2f}, {pre['ci95'][1]:.2f}]."
            )
            if "in_cascade" in cv:
                cascade = cv["in_cascade"]
                extra += (
                    f" Inside the two-stage cascade: {cascade['balanced_accuracy']:.2f} "
                    f"[{cascade['ci95'][0]:.2f}, {cascade['ci95'][1]:.2f}], permutation p = "
                    f"{cascade['permutation_p']:.3f}."
                )
            st.caption(ui.md(extra))
            with st.expander("Model card", icon=":material/badge:"):
                st.markdown(f"**Intended use.** {ui.md(card['intended_use'])}")
                st.markdown(f"**Evaluation.** {ui.md(card['evaluation'])}")
                st.markdown(f"**Decision.** {ui.md(card['decision'])}")
                st.markdown(
                    "**Limitations.**\n" + "\n".join(f"- {ui.md(x)}" for x in card["limitations"])
                )
                if training["left_out"]:
                    st.caption(
                        "Left out of training: " + ui.md(", ".join(training["left_out"])) + "."
                    )
                st.caption(f"Trained {info['trained_at']}.")


tab_this, tab_models = st.tabs(["This recording", "Model evidence"])
with tab_this:
    this_recording()
with tab_models:
    model_evidence()

ui.glossary(
    [
        "Model score",
        "Balanced accuracy",
        "Confidence interval",
        "Cross-validation by participant",
        "Out-of-fold prediction",
        "Permutation test",
        "q-value",
        "Better than chance",
        "Uncertain band",
    ],
    label="How to read these results",
)
