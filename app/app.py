"""Streamlit demo UI for the VAR Decision Predictor."""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SCENARIOS_PATH
from src.predictor import PredictionError, VerdictResult, predict_verdict
from src.verdict_match import rulings_match

st.set_page_config(page_title="VAR Decision Predictor", page_icon="⚽", layout="centered")

RULING_COLORS = {
    "Offside": "#d9534f",
    "Goal Disallowed": "#d9534f",
    "No Offside": "#5cb85c",
    "Goal Stands": "#5cb85c",
    "No Penalty": "#5cb85c",
    "No Card": "#5cb85c",
    "Penalty": "#d9534f",
    "Red Card": "#d9534f",
    "Yellow Card": "#f0ad4e",
    "VAR Review - No Clear Error": "#f0ad4e",
}


@st.cache_resource
def load_scenarios() -> list[dict]:
    return json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))


def ruling_color(ruling: str) -> str:
    for key, color in RULING_COLORS.items():
        if key.lower() in ruling.lower():
            return color
    return "#f0ad4e"


def render_verdict(result: VerdictResult, ground_truth: str | None = None):
    color = ruling_color(result.predicted_ruling)
    st.markdown(
        f"""
        <div style="border-left: 6px solid {color}; padding: 0.75rem 1rem;
                    background: rgba(128,128,128,0.08); border-radius: 6px;">
            <h3 style="margin:0; color:{color};">{result.predicted_ruling}</h3>
            <p style="margin:0.25rem 0 0 0; opacity:0.8;">Law cited: {result.law_citation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.progress(result.confidence_percent / 100, text=f"Confidence: {result.confidence_percent}%")
    st.write("**Rationale:** " + result.rationale)
    if result.key_factors:
        st.write("**Key factors:**")
        for factor in result.key_factors:
            st.markdown(f"- {factor}")

    if ground_truth:
        match = rulings_match(ground_truth, result.predicted_ruling)
        if match:
            st.success(f"Ground truth: {ground_truth} — prediction matches ✅")
        else:
            st.warning(f"Ground truth: {ground_truth} — compare against prediction above")

    with st.expander("Show retrieved IFAB Law text used for grounding"):
        st.text(result.retrieved_law_excerpt)


def run_prediction(incident_text: str, ground_truth: str | None = None):
    with st.spinner("Analyzing incident against IFAB Laws of the Game..."):
        try:
            result = predict_verdict(incident_text)
        except PredictionError as exc:
            st.error(str(exc))
            return
    render_verdict(result, ground_truth)


st.title("⚽ VAR Decision Predictor")
st.caption("Predicting the VAR ruling before it's revealed — grounded in the IFAB Laws of the Game.")

tab_preset, tab_custom = st.tabs(["Preset Scenarios", "Custom Input"])

with tab_preset:
    scenarios = load_scenarios()
    labels = [f"[{s['category']}] {s['description'][:70]}..." for s in scenarios]
    idx = st.selectbox("Select an incident", range(len(scenarios)), format_func=lambda i: labels[i])
    scenario = scenarios[idx]
    st.write(scenario["description"])
    st.caption(f"Difficulty: {scenario['difficulty']}")
    if st.button("Predict Verdict", key="preset_predict"):
        run_prediction(scenario["description"], scenario["ground_truth"])

with tab_custom:
    custom_text = st.text_area(
        "Describe the incident",
        placeholder="e.g. Defender's arm is raised above shoulder height and blocks a goal-bound shot inside the penalty box...",
        height=120,
    )
    if st.button("Predict Verdict", key="custom_predict"):
        if custom_text.strip():
            run_prediction(custom_text)
        else:
            st.warning("Enter an incident description first.")
