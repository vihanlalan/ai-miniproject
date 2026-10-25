"""SynthDx Streamlit app.  Run from the project root:

    streamlit run synthdx/app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # allow `streamlit run` from any directory
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from synthdx.app import views  # noqa: E402
from synthdx.app.components import CSS, disclaimer  # noqa: E402

PAGES = {
    "Overview": views.page_overview,
    "Data Explorer": views.page_data,
    "GAN Lab": views.page_gan_lab,
    "Expert System Diagnosis": views.page_diagnosis,
    "Experiments Dashboard": views.page_experiments,
    "Fairness & Privacy": views.page_fairness_privacy,
    "Rule Browser": views.page_rules,
}

st.set_page_config(page_title="SynthDx", page_icon="🩺", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🩺 SynthDx")
    st.caption("Synthetic medical data + diagnostic expert system")
    page = st.radio("Navigate", list(PAGES), key="page", label_visibility="collapsed")
    st.divider()
    st.caption("Generative models · rule-based reasoning · ML evaluation.\n\n"
               "Academic demo, not for clinical use.")

disclaimer()
try:
    PAGES[page]()
except Exception as exc:  # last-resort friendly error instead of a raw traceback
    st.error("Something went wrong while rendering this page. "
             f"Details: {type(exc).__name__}: {exc}")
    with st.expander("Technical details"):
        st.exception(exc)
