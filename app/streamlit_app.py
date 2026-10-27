"""Restwave: upload a resting-state EEG recording, see four graphs and a verdict. Not diagnostic.

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py        # or: make app

One page, top to bottom: the upload, the input check (pdeeg.predict.inputs), the run with its
real progress (pdeeg.predict.predictor.predict_files), the verdict of stage 1 with how much to
trust it (pdeeg.predict.predictor.stage_evidence), the medication call when stage 1 found a
Parkinson's pattern, four graphs of the cleaned signal, and the report. The research view with
every check, figure and number is app/research_app.py (`make research`).

This script lays the page out; what it shows is computed in restwave/ and neurolab/.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # restwave/ and neurolab/, next to this script
    sys.path.insert(0, str(HERE))

import streamlit as st  # noqa: E402

from neurolab import engine  # noqa: E402
from restwave import analysis, blocks, charts, plain, settings, theme  # noqa: E402
from restwave import verdict as words  # noqa: E402

SAMPLE = "rw_sample"  # session state: the digest of the sample recording, when it is loaded

st.set_page_config(
    page_title="Restwave · EEG pattern check",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.html(theme.CSS)
config = engine.settings()
app = settings.load(config)
config_dir = str(config.config_dir)


def trained_line() -> str:
    who = engine.trained()
    if who is None:
        return (
            "Research demo, not a medical diagnosis. No models have been trained on this "
            "computer yet (`pdeeg train-final`)."
        )
    return (
        f"Research demo, not a medical diagnosis. Trained on {who['participants']} people from "
        f"one public dataset (OpenNeuro {who['dataset']['id']})."
    )


def footer(outcome: analysis.Outcome | None = None) -> None:
    if outcome is not None and outcome.report_html:
        names = [name for name, _ in engine.files()]
        stem = Path(names[0]).stem + ("_pair" if len(names) == 2 else "")
        with st.container(key="rw-download"):
            st.download_button(
                "Download report (HTML)",
                outcome.report_html.encode("utf-8"),
                file_name=f"{stem}_restwave_report.html",
                mime="text/html",
                icon=":material/download:",
                on_click="ignore",
                help="Open it in a browser and print it to save it as a PDF.",
            )
    st.html(blocks.footer(app.research_url, trained_line()))


def forget() -> None:
    engine.clear()
    st.session_state.pop(SAMPLE, None)


# --- 1. the hero: upload, or the file being looked at ---------------------------------------

uploads = engine.files()
if uploads is None:
    st.html(blocks.hero())
    with st.container(key="rw-drop"):
        picked = st.file_uploader(
            "Drop a .bdf or .edf recording here",
            type=["bdf", "edf"],
            accept_multiple_files=True,
            key=f"rw-upload-{st.session_state.get(engine.UPLOADS, 0)}",
        )
    st.html(blocks.drop_hint())
    with st.container(key="rw-sample"):
        try_sample = st.button("Try a sample recording", icon=":material/play_arrow:")
    if picked and len(picked) > 2:
        st.html(
            blocks.stop(
                f"{len(picked)} files is too many.",
                "Upload one recording, or two recordings of the same patient (OFF and ON "
                "medication) together.",
                eyebrow="Too many files",
            )
        )
    elif picked:
        engine.load(tuple((f.name, f.getvalue()) for f in picked))
        st.rerun()
    if try_sample:
        engine.load((analysis.sample(config_dir),))
        st.session_state[SAMPLE] = engine.key()
        st.rerun()
    footer()
    st.stop()

names = [name for name, _ in uploads]
st.html(blocks.brand())
left, right = st.columns([5, 2], vertical_alignment="center")
left.html(blocks.files(names))
with right, st.container(key="rw-again"):
    st.button(
        "Analyse another recording", icon=":material/upload_file:", on_click=forget, width="stretch"
    )

# --- 2. the input check -------------------------------------------------------------------------

checks = engine.validation()
for index, report in enumerate(checks["reports"]):
    done = {check["name"]: check for check in report["checks"]}
    items = []
    for name in plain.CHIPS:
        check = done.get(name)
        status = check["status"] if check else "pending"
        items.append((status, plain.CHIP_TITLES[name], plain.chip(name, check, report, config)))
    if done.get("file", {}).get("status") == "failed":
        items = [("failed", "File", "Can't be opened")] + items
    st.html(blocks.chips(items, f"Recording {'AB'[index]}" if len(names) == 2 else None))
failed = next((r for r in checks["reports"] if r["error"] is not None), None)
if failed is not None:
    what, fix = plain.problem(failed, config)
    prefix = f"{failed['name']}: " if len(names) == 2 else ""
    st.html(blocks.stop(prefix + what, fix))
    footer()
    st.stop()
if checks["pair_error"]:
    st.html(
        blocks.stop(
            "These two files hold the same signal.",
            "To compare medication, upload the patient's OFF- and ON-medication recordings, "
            "which are two different files.",
        )
    )
    footer()
    st.stop()
if len(names) == 2:
    same = st.checkbox(
        "These are one patient's two recordings: one OFF and one ON medication",
        key=engine.SAME_PATIENT,
    )
    if not same:
        st.html(
            blocks.banner(
                "Tick the box if they are. Otherwise, analyse them one at a time.",
                strong="Two files: are they the same patient?",
            )
        )
        footer()
        st.stop()

# --- 3. processing, with its real progress --------------------------------------------------

outcome = analysis.analyse(engine.key(), config_dir, uploads, checks)
if outcome.error is not None:
    titles = {
        analysis.CHECK: ("These files can't be analysed together.", "Analyse them one at a time."),
        analysis.MODEL: (
            "The models are not set up on this computer.",
            "Run `pdeeg train-final` (it needs the downloaded dataset), then try again.",
        ),
    }
    title, fix = titles.get(
        outcome.failed_step,
        ("We could not measure this recording.", "Try another recording, or a longer one."),
    )
    st.html(blocks.stop(title, f"{fix} Details: {outcome.error}", eyebrow="Analysis stopped"))
    footer()
    st.stop()

result = outcome.result
evidence = engine.evidence()
paired = len(names) == 2

# --- 4. the verdict -------------------------------------------------------------------------

if st.session_state.get(SAMPLE) == engine.key() and not result["training_recording"]:
    st.html(
        blocks.banner(
            "It is made up, not anyone's EEG, so the result below says nothing about a person. "
            "It shows what the page does.",
            kind="flask",
            strong="This is a sample signal.",
        )
    )
held_out = words.held_out_note(result)
if held_out:
    st.html(blocks.banner(held_out, strong="Test result, not a new prediction."))


def gauge(entry) -> str:
    """The gauge of a stage's score, or nothing for a stage that did not run."""
    if "skipped" in entry:
        return ""
    stage = entry["stage"]
    tints = (theme.HC_WASH, theme.PD_WASH) if stage == "stage1" else (theme.PAPER_2, theme.PAPER_2)
    return charts.gauge(
        entry["probability"],
        entry["threshold"],
        entry["uncertain_band"],
        words.ENDS[stage],
        words.SCORE_OF[stage],
        tints,
    )


first_stage = words.entries(result, "stage1")
calls = [words.diagnosis(entry, evidence) for entry in first_stage]
if paired:
    gauges = [gauge(call.entry) for call in calls]
    st.html(blocks.verdict_pair(calls, gauges, names))
else:
    st.html(blocks.verdict(calls[0], gauge(calls[0].entry)))

# --- 5. medication --------------------------------------------------------------------------

if paired:
    for entry in words.entries(result, "stage2_paired"):
        if "skipped" not in entry:
            call = words.paired(entry, names, evidence)
            st.html(blocks.verdict(call, gauge(entry), small=True))
elif calls[0].kind == words.DETECTED:
    for entry in words.entries(result, "stage2_single"):
        if "skipped" not in entry:
            call = words.medication(entry, evidence)
            st.html(blocks.verdict(call, gauge(entry), small=True))

# --- 6. what we saw in the signal -----------------------------------------------------------

st.html(
    blocks.section(
        "What we saw in the signal",
        "The cleaned recording, measured the way the models measured the people they learned from.",
    )
)
groups = analysis.groups(config_dir)
shown = [(f"Recording {'AB'[i]}" if paired else "This recording", s) for i, s in
         enumerate(outcome.shown) if s is not None]  # fmt: skip
if groups is None:
    group_note = blocks.missing(
        "The group ranges need the training feature tables (`pdeeg features`), which are not "
        "on this computer."
    )
    group_items: list[tuple[str, str, str]] = []
else:
    group_note = ""
    group_items = [
        ("bar", theme.HC, f"Healthy group ({groups.people.get(charts.GROUPS[0], 0)} people)"),
        ("bar", theme.PD, f"Parkinson's group ({groups.people.get(charts.GROUPS[1], 0)} people)"),
    ]
marks = [("dot", theme.TEXT, label) if i == 0 else ("diamond", theme.TEXT, label)
         for i, (label, _) in enumerate(shown)]  # fmt: skip
lines = [("line", theme.TEXT, label) if i == 0 else ("dash", theme.TEXT, label)
         for i, (label, _) in enumerate(shown)]  # fmt: skip

top, bottom = st.columns(2, gap="medium"), st.columns(2, gap="medium")
with top[0], st.container(key="rw-card-signal"):
    first = shown[0][1] if shown else None
    about = (
        f" Rows are {charts.spacing(first.signal_uv):g} microvolts apart. "
        + plain.cleaning_words(first.removed, first.bad_percent)
        if first
        else ""
    )
    st.html(
        blocks.card_head(
            "Your brain signal",
            f"{app.seconds:g} seconds of the cleaned recording from {len(app.channels)} places "
            "on the head, front to back. These ripples are what the analysis measures." + about,
        )
    )
    for label, s in shown:
        if paired:
            st.html(f'<p class="rw-eyebrow" style="margin-top:.6rem">{label}</p>')
        st.altair_chart(
            charts.signal(s.signal_uv, s.channels, s.sfreq, s.start_s, config),
            theme=None,
            width="stretch",
        )
    if not shown:
        st.html(blocks.missing("The cleaned signal could not be drawn."))

with top[1], st.container(key="rw-card-balance"):
    st.html(
        blocks.card_head(
            "Brain-wave balance",
            "How the signal's power is shared between slow and fast brain waves, from the "
            "<b>slowest</b> (delta) to the <b>fastest</b> (gamma). Dots are this recording; "
            "the shaded bars show the typical range in each group.",
        )
    )
    st.html(blocks.legend(marks + group_items))
    st.altair_chart(
        charts.balance(
            [(label, s.band_power) for label, s in shown],
            None if groups is None else groups.band,
            config,
        ),
        theme=None,
        width="stretch",
    )
    if group_note:
        st.html(group_note)

with bottom[0], st.container(key="rw-card-complexity"):
    st.html(
        blocks.card_head(
            "Signal complexity",
            f"This curve sums up how irregular the ups and downs of "
            f"{plain.series_words(app.series, config)} are. A <b>wider</b> curve means "
            "<b>more irregular</b> fluctuations: calm and bursty stretches mixed together. The "
            "bold line is this recording; the shaded bands show how wide the curve typically is "
            "in each group.",
        )
    )
    st.html(blocks.legend(lines + group_items))
    st.altair_chart(
        charts.complexity(
            [(label, s.spectrum) for label, s in shown],
            None if groups is None else groups.spectrum,
        ),
        theme=None,
        width="stretch",
    )
    if group_note:
        st.html(group_note)

with bottom[1], st.container(key="rw-card-drivers"):
    st.html(
        blocks.card_head(
            "What drove the decision",
            "The measurements that pushed the result most, and which way. Longer bars pushed "
            "harder.",
        )
    )
    ends = ("healthy", "Parkinson's")
    for index, entry in enumerate(first_stage):
        if paired:
            st.html(f'<p class="rw-eyebrow" style="margin-top:.6rem">Recording {"AB"[index]}</p>')
        if "top_features" in entry:
            st.html(charts.drivers(entry["top_features"], ends, config))
        elif entry.get("out_of_fold"):
            st.html(
                blocks.missing(
                    "For a recording from the training dataset the result comes from a test "
                    "that does not keep which measurements drove it."
                )
            )
        else:
            st.html(blocks.missing("No measurements to show for this result."))

# --- 7. footer -----------------------------------------------------------------------------

footer(outcome)
