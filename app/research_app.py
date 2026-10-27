"""NeuroLab, the research demo of pdeeg: one EEG recording, or a patient's two. Not diagnostic.

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py        # or: make app

A BDF or EDF file with the channels of the dataset's montage is checked first; nothing else is
done with a file that fails a check. On request the app then runs the code of `pdeeg predict`
(pdeeg.predict.predictor.predict_files) with the final models of `pdeeg train-final`: stage 1,
Parkinson's or healthy; stage 2, OFF or ON medication, only after a stage 1 call of Parkinson's
outside its uncertain band; with two recordings of one patient, which of them is OFF. Every
stage shows how it did in cross-validation, and one that was not better than chance is greyed
out. A recording of the training dataset gets its out-of-fold results instead of a prediction.
The cleaned recording, the data the models saw, is described by pdeeg.demo: the signal before
and after cleaning, its spectrum, and its MFDFA.

This script sets up the shell (theme, navigation, sidebar); the pages are in views/ and what
they share in neurolab/.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # neurolab/, next to this script
    sys.path.insert(0, str(HERE))

import streamlit as st  # noqa: E402

from neurolab import components, theme  # noqa: E402
from neurolab.pages import PAGES, SECTIONS  # noqa: E402

st.set_page_config(
    page_title="NeuroLab · pdeeg research demo",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.html(theme.CSS)
st.logo(HERE / "assets" / "logo.svg", icon_image=HERE / "assets" / "mark.svg", size="large")

navigation = st.navigation(
    {
        section: [
            st.Page(
                PAGES[name].path,
                title=PAGES[name].title,
                icon=PAGES[name].icon,
                default=name == "overview",
            )
            for name in names
        ]
        for section, names in SECTIONS.items()
    }
)
components.sidebar()
navigation.run()
