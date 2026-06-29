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

ACCENT = "#1A8F5E"
NEGATIVE = "#C9252A"
WARNING = "#B45309"

# Tinted card backgrounds for each ruling family (light mode semantic tints)
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

RULING_BG = {
    "negative": "oklch(0.97 0.015 25)",
    "positive": "oklch(0.96 0.018 155)",
    "neutral":  "oklch(0.97 0.014 75)",
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    color: #0F1114;
}}

#MainMenu, footer, header [data-testid="stToolbar"] {{
    visibility: hidden;
}}

.block-container {{
    padding-top: 2.5rem;
    max-width: 720px;
}}

/* ── Header ── */
.brand-mark {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {ACCENT};
    margin-bottom: 0.4rem;
}}

h1 {{
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    color: #0F1114;
    margin-bottom: 0.3rem !important;
    text-wrap: balance;
}}

.subtitle {{
    color: #6B7280;
    font-size: 0.95rem;
    line-height: 1.55;
    margin-bottom: 1.8rem;
    max-width: 60ch;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 1.6rem;
    border-bottom: 1px solid #E2E4E8;
    background: transparent;
}}

.stTabs [data-baseweb="tab"] {{
    height: 2.4rem;
    color: #6B7280;
    font-weight: 500;
    background: transparent !important;
}}

.stTabs [aria-selected="true"] {{
    color: {ACCENT} !important;
}}

/* ── Form controls ── */
div[data-testid="stFileUploader"] section {{
    border-radius: 8px;
    border: 1px solid #D1D5DB;
    background: #FFFFFF;
}}

.stTextInput input, .stTextArea textarea {{
    border-radius: 8px !important;
    border: 1px solid #D1D5DB !important;
    background: #FFFFFF !important;
    color: #0F1114 !important;
}}

.stTextInput input:focus, .stTextArea textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px oklch(0.92 0.04 155) !important;
}}

/* ── Primary CTA button ── */
.stButton button[data-testid="stBaseButton-primary"],
.stButton button[kind="primary"],
.stButton button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: opacity 0.15s ease;
}}

.stButton button[data-testid="stBaseButton-primary"]:hover,
.stButton button:hover {{
    opacity: 0.88;
}}

/* ── Verdict card ── */
.verdict-card {{
    border-radius: 10px;
    border: 1px solid #E2E4E8;
    overflow: hidden;
    margin-top: 0.75rem;
    margin-bottom: 0.5rem;
}}

.verdict-card-header {{
    padding: 1.2rem 1.5rem 1rem;
}}

.verdict-ruling {{
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 0.2rem;
    letter-spacing: -0.02em;
    line-height: 1.15;
}}

.verdict-law {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #6B7280;
    margin: 0;
}}

/* ── Confidence bar ── */
.confidence-row {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.9rem 1.5rem;
    border-top: 1px solid #E2E4E8;
    background: #FFFFFF;
}}

.confidence-number {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #0F1114;
    min-width: 3.5ch;
}}

.confidence-track {{
    flex: 1;
    height: 6px;
    background: #E2E4E8;
    border-radius: 999px;
    overflow: hidden;
}}

.confidence-fill {{
    height: 100%;
    border-radius: 999px;
    transition: width 0.4s ease;
}}

.confidence-label-text {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #6B7280;
    white-space: nowrap;
}}

/* ── Section labels ── */
.section-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 0.4rem;
    margin-top: 0.2rem;
}}

/* ── Rationale + key factors ── */
.rationale-text {{
    font-size: 0.97rem;
    line-height: 1.65;
    color: #1a1d23;
    margin: 0.75rem 0 0.5rem;
}}

/* ── Hide Streamlit progress bar (replaced with custom) ── */
.stProgress {{
    display: none;
}}

/* ── Expander ── */
.streamlit-expanderHeader {{
    font-size: 0.85rem !important;
    color: #6B7280 !important;
}}

@media (prefers-reduced-motion: reduce) {{
    .confidence-fill {{
        transition: none;
    }}
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


def ruling_bg(ruling: str) -> str:
    color = ruling_color(ruling)
    if color == NEGATIVE:
        return RULING_BG["negative"]
    if color == ACCENT:
        return RULING_BG["positive"]
    return RULING_BG["neutral"]


def render_verdict(result: VerdictResult, ground_truth: str | None = None, visual_description: str | None = None):
    color = ruling_color(result.predicted_ruling)
    bg = ruling_bg(result.predicted_ruling)
    pct = result.confidence_percent
    st.markdown(
        f"""
        <div class="verdict-card">
            <div class="verdict-card-header" style="background:{bg};">
                <p class="verdict-ruling" style="color:{color};">{result.predicted_ruling}</p>
                <p class="verdict-law">{result.law_citation}</p>
            </div>
            <div class="confidence-row">
                <span class="confidence-number">{pct}%</span>
                <div class="confidence-track">
                    <div class="confidence-fill" style="width:{pct}%; background:{color};"></div>
                </div>
                <span class="confidence-label-text">confidence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="rationale-text">{result.rationale}</p>', unsafe_allow_html=True)
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
