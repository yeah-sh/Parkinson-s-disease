"""The pages of the app, as the navigation and the links between pages name them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageInfo:
    path: str  # relative to app/streamlit_app.py
    title: str
    icon: str
    lede: str


PAGES = {
    "overview": PageInfo(
        "views/overview.py",
        "Overview",
        ":material/space_dashboard:",
        "Load a resting-state EEG recording, check it, and run it through the same cleaning, "
        "MFDFA and models as `pdeeg predict`. A research demonstration on one public dataset.",
    ),
    "recording": PageInfo(
        "views/recording.py",
        "Recording Analysis",
        ":material/rule:",
        "Every input check, what failed and how to fix it, and what the cleaning did.",
    ),
    "quality": PageInfo(
        "views/quality.py",
        "EEG Signal Quality",
        ":material/monitor_heart:",
        "The analysed window before and after cleaning, the windows marked bad, and the "
        "spectrum and amplitude of every channel.",
    ),
    "multifractal": PageInfo(
        "views/multifractal.py",
        "Multifractal Analysis",
        ":material/insights:",
        "How the fluctuations of the cleaned EEG and of its band envelopes scale: fluctuation "
        "functions, the singularity spectrum and the features of every channel.",
    ),
    "topography": PageInfo(
        "views/topography.py",
        "Scalp Topography",
        ":material/blur_on:",
        "One feature over the scalp, channel by channel, with the channels ranked.",
    ),
    "evaluation": PageInfo(
        "views/evaluation.py",
        "Model Evaluation",
        ":material/query_stats:",
        "What each stage said about this recording, what kind of prediction that is, and how "
        "the stage did in cross-validation by participant.",
    ),
    "report": PageInfo(
        "views/report.py",
        "Research Report",
        ":material/description:",
        "Everything above in one printable document, with the raw result as JSON and text.",
    ),
    "about": PageInfo(
        "views/about.py",
        "About & Methodology",
        ":material/menu_book:",
        "The pipeline with the settings it runs with, how the models are validated, a "
        "glossary, and the limits of what this can show.",
    ),
}

SECTIONS = {
    "Workbench": ("overview", "recording"),
    "Analysis": ("quality", "multifractal", "topography"),
    "Evidence": ("evaluation", "report"),
    "Reference": ("about",),
}
