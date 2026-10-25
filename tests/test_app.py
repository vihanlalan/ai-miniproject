"""UI tests with Streamlit's AppTest: every page renders; a diagnosis can be submitted."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "synthdx" / "app" / "main.py")
PAGES = ["Overview", "Data Explorer", "GAN Lab", "Expert System Diagnosis",
         "Experiments Dashboard", "Fairness & Privacy", "Rule Browser"]


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def _assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_loads(page):
    at = _app()
    at.sidebar.radio(key="page").set_value(page).run()
    _assert_clean(at)
    assert any(page.split()[0] in t.value for t in at.title) or page == "Overview"
    assert any("not for clinical use" in m.value for m in at.markdown)


def test_diagnosis_submission_and_what_if():
    at = _app()
    at.sidebar.radio(key="page").set_value("Expert System Diagnosis").run()
    at.number_input(key="f_chol").set_value(300)
    at.selectbox(key="f_thal").set_value(7)
    next(b for b in at.button if b.label == "Diagnose").click().run()
    _assert_clean(at)
    text = " ".join(m.value for m in at.markdown)
    assert "Heart disease likely" in text or "Heart disease unlikely" in text
    assert "R0" in text  # rule trace lines are shown
    # what-if: raise oldpeak and check the panel recomputes without errors
    at.slider(key="w_oldpeak").set_value(4.0).run()
    _assert_clean(at)
    assert any(m.label == "What-if CF" for m in at.metric)


def test_load_random_real_patient():
    at = _app()
    at.sidebar.radio(key="page").set_value("Expert System Diagnosis").run()
    at.button(key="load_real").click().run()
    _assert_clean(at)
    assert any("Random real record" in c.value for c in at.caption)


def test_gan_lab_quick_retrain_and_generate():
    at = _app()
    at.sidebar.radio(key="page").set_value("GAN Lab").run()
    at.radio(key="gan_model").set_value("cgan").run()
    at.radio(key="gan_mode").set_value("Quick retrain").run()
    at.slider(key="gan_epochs").set_value(5).run()
    at.button(key="gan_train").click().run()
    _assert_clean(at)
    at.number_input(key="gan_n").set_value(50)
    at.selectbox(key="gan_cond").set_value("Disease (1)")
    at.button(key="gan_generate").click().run()
    _assert_clean(at)
    assert at.session_state["synthetic"]["target"].eq(1).all()
    assert len(at.session_state["synthetic"]) == 50


def test_rule_browser_search():
    at = _app()
    at.sidebar.radio(key="page").set_value("Rule Browser").run()
    at.text_input(key="rule_q").set_value("thallium").run()
    _assert_clean(at)
    assert any("of 41 rules" in m.value for m in at.markdown)
