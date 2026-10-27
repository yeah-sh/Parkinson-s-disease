"""About & Methodology: the pipeline as configured, how it is validated, terms, and limits."""

from __future__ import annotations

import streamlit as st

from neurolab import components as ui
from neurolab import engine, methodology
from neurolab.glossary import TERMS

ui.page_header("about")
config = engine.settings()
evidence = engine.evidence()
who = engine.trained()
dataset = config.data.dataset

with st.container(key="nl-card-intro"):
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(
            "NeuroLab is the interface of **pdeeg**, which asks whether the temporal structure "
            "of resting-state EEG differs between people with Parkinson's disease and healthy "
            "controls, and within patients between sessions off and on medication. Its main "
            "features come from multifractal detrended fluctuation analysis (MFDFA); band "
            "power is the baseline."
        )
        st.markdown(
            "The classifiers here are **exploratory**: the project's primary analysis is a set "
            "of pre-registered group statistics (reports/stats_results.md). This app shows "
            "what the features and the two-stage tool do on one recording, next to how well "
            "each stage did in cross-validation."
        )
    with right:
        rows = [("Dataset", f"OpenNeuro {dataset.id}, version {dataset.version}")]
        if who is not None:
            rows += [
                ("Trained on", f"{who['participants']} people, {who['recordings']} recordings"),
            ]
        rows += [
            ("Montage", dataset.montage),
            ("Mains", f"{dataset.line_freq:g} Hz"),
            ("Configuration", str(config.config_dir)),
        ]
        ui.kv(rows)

st.space("small")
ui.section("The pipeline, with the settings it runs with")
st.caption(
    "Read from configs/ and the saved models when the app starts; the reasons for each "
    "setting are in the comments of those files and in reports/methods.md."
)
parts = methodology.sections(config, evidence)
columns = st.columns(2, gap="medium")
for position, part in enumerate(parts):
    with columns[position % 2], st.container(key=f"nl-card-method-{position}"):
        st.markdown(f"**{part.title}**")
        st.caption(part.summary)
        ui.kv(part.rows)

st.space("small")
ui.section("Glossary")
columns = st.columns(2, gap="medium")
for position, (term, definition) in enumerate(TERMS.items()):
    columns[position % 2].markdown(f"**{term}.** {definition}")

st.space("small")
ui.section("Limitations")
with st.container(key="nl-card-limits"):
    st.markdown(f"**{config.predict.disclaimer}.**")
    stage1 = evidence.get("stage1")
    limits = stage1["model_card"]["limitations"] if stage1 else []
    st.markdown(
        "\n".join(f"- {ui.md(item)}" for item in limits)
        + "\n- A model score is how a model's output maps to the classes it was trained on, "
        "not the probability that a person has Parkinson's disease or is off medication."
        + "\n- Each final model is the best of several configurations compared on the same "
        "cross-validation, so its score is optimistic; the configuration fixed in advance is "
        "shown next to it."
    )
    st.markdown(f"The dataset's curators, in its README: “{methodology.CURATORS}”")

st.space("small")
ui.section("Data and credit")
st.markdown(
    f"**Data.** [OpenNeuro {dataset.id}](https://openneuro.org/datasets/{dataset.id}/versions/"
    f"{dataset.version}), version {dataset.version}: *UC San Diego Resting State EEG Data from "
    "Patients with Parkinson's Disease*, by Alexander P. Rockhill, Nicko Jackson, Jobi George, "
    f"Adam Aron and Nicole C. Swann (doi:10.18112/openneuro.{dataset.id}.v{dataset.version}, "
    "licence CC0). The papers its curators ask users to cite are listed in the README."
)
st.markdown(
    "**Credit.** pdeeg re-engineers [Sankhadip-007/Detecttion-of-Parkinson-s-Disease-using-"
    "machine-learning-and-deep-learning](https://github.com/Sankhadip-007/Detecttion-of-"
    "Parkinson-s-Disease-using-machine-learning-and-deep-learning) by "
    "[Sankhadip Bera](https://github.com/Sankhadip-007). The question, the choice of dataset "
    "and the first pipeline come from that repository."
)
