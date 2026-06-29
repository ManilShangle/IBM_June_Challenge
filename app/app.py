"""Streamlit demo UI for the VAR Decision Predictor."""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SCENARIOS_PATH
from src.predictor import MatchIntake, PredictionError, VerdictResult, build_incident_text, predict_verdict
from src.verdict_match import rulings_match
from src.vision_client import VisionClientError, describe_image
from src.video_utils import VideoExtractionError, extract_middle_frame_jpeg

st.set_page_config(page_title="VAR Decision Predictor", page_icon=":material/sports_soccer:", layout="centered")

ACCENT = "#2FD48A"
NEGATIVE = "#E5484D"
WARNING = "#E8B339"

RULING_COLORS = {
    "Offside": NEGATIVE,
    "Goal Disallowed": NEGATIVE,
    "Penalty": NEGATIVE,
    "Red Card": NEGATIVE,
    "No Offside": ACCENT,
    "Goal Stands": ACCENT,
    "No Penalty": ACCENT,
    "No Card": ACCENT,
    "Yellow Card": WARNING,
    "VAR Review - No Clear Error": WARNING,
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
}}

#MainMenu, footer, header [data-testid="stToolbar"] {{
    visibility: hidden;
}}

.block-container {{
    padding-top: 2.5rem;
    max-width: 720px;
}}

.brand-mark {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {ACCENT};
    margin-bottom: 0.4rem;
}}

h1 {{
    font-weight: 600 !important;
    letter-spacing: -0.01em;
    margin-bottom: 0.3rem !important;
}}

.subtitle {{
    color: #9A9EA5;
    font-size: 0.98rem;
    margin-bottom: 1.8rem;
    max-width: 56ch;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 1.6rem;
    border-bottom: 1px solid #232629;
}}

.stTabs [data-baseweb="tab"] {{
    height: 2.4rem;
    color: #9A9EA5;
    font-weight: 500;
}}

.stTabs [aria-selected="true"] {{
    color: {ACCENT} !important;
}}

div[data-testid="stFileUploader"] section {{
    border-radius: 12px;
    border: 1px solid #232629;
    background: #14161A;
}}

.stTextInput input, .stTextArea textarea {{
    border-radius: 12px !important;
    border: 1px solid #232629 !important;
    background: #14161A !important;
}}

.stButton button {{
    border-radius: 12px;
    font-weight: 600;
    border: none;
}}

.verdict-card {{
    border-radius: 12px;
    border: 1px solid #232629;
    background: #14161A;
    padding: 1.4rem 1.5rem;
    margin-top: 0.5rem;
}}

.verdict-ruling {{
    font-size: 1.4rem;
    font-weight: 600;
    margin: 0;
}}

.verdict-law {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #9A9EA5;
    margin-top: 0.25rem;
}}

.confidence-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #9A9EA5;
    margin-top: 1rem;
    margin-bottom: 0.3rem;
}}

.stProgress > div > div {{
    background-color: #232629;
}}

.section-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9A9EA5;
    margin-bottom: 0.4rem;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_scenarios() -> list[dict]:
    return json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))


def ruling_color(ruling: str) -> str:
    for key, color in RULING_COLORS.items():
        if key.lower() in ruling.lower():
            return color
    return WARNING


def render_verdict(result: VerdictResult, ground_truth: str | None = None, visual_description: str | None = None):
    color = ruling_color(result.predicted_ruling)
    st.markdown(
        f"""
        <div class="verdict-card">
            <p class="verdict-ruling" style="color:{color};">{result.predicted_ruling}</p>
            <p class="verdict-law">{result.law_citation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="confidence-label">Confidence: {result.confidence_percent}%</p>', unsafe_allow_html=True)
    st.progress(result.confidence_percent / 100)
    st.write(result.rationale)
    if result.key_factors:
        st.markdown('<p class="section-label">Key factors</p>', unsafe_allow_html=True)
        for factor in result.key_factors:
            st.markdown(f"- {factor}")

    if ground_truth:
        if rulings_match(ground_truth, result.predicted_ruling):
            st.success(f"Ground truth: {ground_truth}. Prediction matches.")
        else:
            st.warning(f"Ground truth: {ground_truth}. Compare against the prediction above.")

    if visual_description:
        with st.expander("What the footage analysis saw"):
            st.write(visual_description)

    with st.expander("Show retrieved IFAB Law text used for grounding"):
        st.text(result.retrieved_law_excerpt)


def run_prediction(intake: MatchIntake, ground_truth: str | None = None):
    with st.spinner("Analyzing the incident against the IFAB Laws of the Game..."):
        try:
            incident_text = build_incident_text(intake)
            result = predict_verdict(incident_text)
        except PredictionError as exc:
            st.error(str(exc))
            return
    render_verdict(result, ground_truth, intake.visual_description)


st.markdown('<p class="brand-mark">VAR Decision Predictor</p>', unsafe_allow_html=True)
st.title("Know the call before it's revealed")
st.markdown(
    '<p class="subtitle">Describe the incident, name the teams, and attach footage. '
    "The prediction is grounded in the IFAB Laws of the Game, with the cited rule shown "
    "alongside the call.</p>",
    unsafe_allow_html=True,
)

tab_intake, tab_samples = st.tabs(["New Incident", "Sample Scenarios"])

with tab_intake:
    st.markdown('<p class="section-label">Match</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        team_a = st.text_input("Team A", placeholder="e.g. Argentina", label_visibility="visible")
    with col_b:
        team_b = st.text_input("Team B", placeholder="e.g. France", label_visibility="visible")

    st.markdown('<p class="section-label">Situation</p>', unsafe_allow_html=True)
    situation = st.text_area(
        "Describe what happened",
        placeholder="e.g. Defender's arm is raised above shoulder height and blocks a "
        "goal-bound shot inside the penalty box during a corner-kick scramble.",
        height=120,
        label_visibility="collapsed",
    )

    st.markdown('<p class="section-label">Footage (optional)</p>', unsafe_allow_html=True)
    footage = st.file_uploader(
        "Upload a clip or still frame of the incident",
        type=["mp4", "mov", "jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if st.button("Predict the call", type="primary", key="intake_predict"):
        if not situation.strip():
            st.warning("Describe the situation first.")
        else:
            visual_description = ""
            if footage is not None:
                with st.spinner("Reading the footage..."):
                    try:
                        footage_bytes = footage.getvalue()
                        if footage.type and footage.type.startswith("video"):
                            frame_bytes = extract_middle_frame_jpeg(footage_bytes)
                        else:
                            frame_bytes = footage_bytes
                        visual_description = describe_image(frame_bytes)
                    except (VisionClientError, VideoExtractionError) as exc:
                        st.warning(f"Could not analyze footage: {exc}. Continuing with the text description only.")

            intake = MatchIntake(
                situation=situation,
                team_a=team_a,
                team_b=team_b,
                visual_description=visual_description,
            )
            run_prediction(intake)

with tab_samples:
    scenarios = load_scenarios()
    labels = [f"[{s['category']}] {s['description'][:70]}..." for s in scenarios]
    idx = st.selectbox("Select a sample incident", range(len(scenarios)), format_func=lambda i: labels[i])
    scenario = scenarios[idx]
    st.write(scenario["description"])
    st.caption(f"Difficulty: {scenario['difficulty']}")
    if st.button("Predict the call", key="sample_predict"):
        run_prediction(MatchIntake(situation=scenario["description"]), scenario["ground_truth"])
