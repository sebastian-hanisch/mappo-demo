"""Rauchtests: die App läuft für Default und jedes Preset ohne Exception durch (fängt u.a.
StreamlitDuplicateElementId bei st.plotly_chart und Slider-Grenzfälle)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import cn_constants as C

APP_TIMEOUT = 180
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
PRESET_BUTTONS = {name: i for i, name in enumerate(C.PRESETS)}


def _run():
    at = AppTest.from_file(APP_PATH, default_timeout=APP_TIMEOUT)
    at.run()
    return at


def test_default_run_without_exception():
    at = _run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_each_preset_runs_without_exception(name):
    at = _run()
    at.button[PRESET_BUTTONS[name]].click().run()
    assert not at.exception, f"{name}: {[e.value for e in at.exception]}"
    preset = C.PRESETS[name]
    assert at.session_state["env_mode_radio"] == preset["env_mode"]
    assert at.session_state["episodes_slider"] == preset["episodes"]
    assert at.session_state["actor_radio"] == preset["actor"]
    assert at.session_state["method_radio"] == preset["method"]


def test_switching_actor_and_env_keeps_hidden_widget_values():
    at = _run()
    at.session_state["sigma_slider"] = 0.45
    at.radio(key="env_mode_radio").set_value(C.ENV_RANDOM).run()
    assert not at.exception
    assert at.session_state["sigma_slider"] == 0.45
    at.radio(key="scope_radio").set_value(C.SCOPE_FULL).run()
    at.radio(key="actor_radio").set_value(C.ACTOR_TABLE).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["scope_radio"] == C.SCOPE_FULL  # versteckt, aber erhalten
    at.radio(key="actor_radio").set_value(C.ACTOR_NET).run()
    assert not at.exception
    assert at.session_state["scope_radio"] == C.SCOPE_FULL


def test_method_and_clip_switches_run():
    at = _run()
    at.radio(key="method_radio").set_value(C.METHOD_IPPO).run()
    assert not at.exception, [e.value for e in at.exception]
    at.radio(key="clip_select").set_value("off").run()
    assert not at.exception, [e.value for e in at.exception]


def test_small_instance_edge_cases():
    at = _run()
    at.slider(key="n_jobs_slider").set_value(C.N_JOBS_MIN)
    at.slider(key="n_agents_slider").set_value(C.N_AGENTS_MIN)
    at.select_slider(key="episodes_slider").set_value(100).run()
    assert not at.exception, [e.value for e in at.exception]
