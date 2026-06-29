"""Streamlit demo UI for the VAR Decision Predictor."""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import INCIDENTS_PATH
from src.predictor import MatchIntake, PredictionError, VerdictResult, build_incident_text, predict_verdict
from src.vision_client import VisionClientError, describe_image, describe_video_frames
from src.video_utils import VideoExtractionError, extract_frames_jpeg

st.set_page_config(
    page_title="VAR Decision Predictor",
    page_icon=":material/sports_soccer:",
    layout="centered",
)

ACCENT = "#1A8F5E"
NEGATIVE = "#C9252A"
WARNING = "#B45309"

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
.stButton button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: opacity 0.15s ease;
}}

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
    font-size: 2rem;
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

/* ── Law strip ── */
.verdict-law-strip {{
    padding: 0.6rem 1.5rem;
    border-top: 1px solid #E2E4E8;
    background: #FFFFFF;
}}

/* ── Plain-English box ── */
.law-plain-english {{
    background: #F5F6F8;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin: 0.75rem 0;
    font-size: 0.93rem;
    line-height: 1.6;
    color: #374151;
    border-left: 3px solid #E2E4E8;
}}

.law-plain-english-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 0.3rem;
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

/* ── Rationale ── */
.rationale-text {{
    font-size: 0.97rem;
    line-height: 1.65;
    color: #1a1d23;
    margin: 0.75rem 0 0.5rem;
}}

/* ── Expander ── */
.streamlit-expanderHeader {{
    font-size: 0.85rem !important;
    color: #6B7280 !important;
}}

/* ── Famous incident cards ── */
.incident-card {{
    border: 1px solid #E2E4E8;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    background: #FFFFFF;
}}

.incident-title {{
    font-size: 1rem;
    font-weight: 700;
    color: #0F1114;
    margin: 0 0 0.15rem;
}}

.incident-meta {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    color: #9CA3AF;
    margin-bottom: 0.5rem;
}}

.incident-summary {{
    font-size: 0.9rem;
    line-height: 1.6;
    color: #374151;
    margin-bottom: 0.65rem;
}}

.incident-infamous {{
    font-size: 0.82rem;
    font-style: italic;
    color: #6B7280;
    padding: 0.5rem 0.75rem;
    background: #F5F6F8;
    border-radius: 6px;
    margin-bottom: 0.65rem;
}}

/* ── Footage uploader hero ── */
div[data-testid="stFileUploader"] {{
    margin-bottom: 0.25rem;
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_incidents() -> list[dict]:
    return json.loads(INCIDENTS_PATH.read_text(encoding="utf-8"))


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


def render_verdict(result: VerdictResult, visual_description: str | None = None):
    color = ruling_color(result.predicted_ruling)
    bg = ruling_bg(result.predicted_ruling)
    st.markdown(
        f"""
        <div class="verdict-card">
            <div class="verdict-card-header" style="background:{bg};">
                <p class="verdict-ruling" style="color:{color};">{result.predicted_ruling}</p>
            </div>
            <div class="verdict-law-strip">
                <span class="verdict-law">{result.law_citation}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<p class="rationale-text">{result.rationale}</p>', unsafe_allow_html=True)

    if result.plain_english_law:
        st.markdown(
            f"""
            <div class="law-plain-english">
                <div class="law-plain-english-label">What this law means</div>
                {result.plain_english_law}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if visual_description:
        with st.expander("What the footage analysis saw"):
            st.write(visual_description)

    with st.expander("Show IFAB Law text used for grounding"):
        st.text(result.retrieved_law_excerpt)


def run_prediction(intake: MatchIntake, prefill_key: str = "intake"):
    with st.spinner("Calling it..."):
        try:
            incident_text = build_incident_text(intake)
            result = predict_verdict(incident_text)
        except PredictionError as exc:
            st.error(str(exc))
            return
    render_verdict(result, intake.visual_description if intake.visual_description else None)


# ── Page header ──────────────────────────────────────────────────────────────
st.markdown('<p class="brand-mark">VAR Decision Predictor</p>', unsafe_allow_html=True)
st.title("Beat the ref to the call")
st.markdown(
    '<p class="subtitle">Drop the footage. Get the ruling before the announcement, '
    "cited against the IFAB Laws of the Game.</p>",
    unsafe_allow_html=True,
)

tab_intake, tab_incidents = st.tabs(["New Incident", "Famous Incidents"])

# ── Tab 1: New Incident ───────────────────────────────────────────────────────
with tab_intake:
    st.markdown('<p class="section-label">Footage</p>', unsafe_allow_html=True)
    footage = st.file_uploader(
        "Upload a clip or still frame of the incident",
        type=["mp4", "mov", "jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    st.markdown('<p class="section-label">Situation</p>', unsafe_allow_html=True)

    # Pre-fill from famous incidents tab
    default_situation = st.session_state.get("prefill_situation", "")
    default_team_a = st.session_state.get("prefill_team_a", "")
    default_team_b = st.session_state.get("prefill_team_b", "")

    situation = st.text_area(
        "Describe what happened",
        value=default_situation,
        placeholder="e.g. Defender's arm raised above shoulder height blocks a goal-bound "
        "shot inside the penalty box during a corner-kick scramble.",
        height=100,
        label_visibility="collapsed",
    )

    st.markdown('<p class="section-label">Match</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        team_a = st.text_input("Team A", value=default_team_a, placeholder="e.g. Argentina", label_visibility="visible")
    with col_b:
        team_b = st.text_input("Team B", value=default_team_b, placeholder="e.g. France", label_visibility="visible")

    if st.button("Call it", type="primary", key="intake_predict"):
        if not situation.strip() and footage is None:
            st.warning("Add footage or describe the situation.")
        else:
            visual_description = ""
            if footage is not None:
                with st.spinner("Reading the footage..."):
                    try:
                        footage_bytes = footage.getvalue()
                        if footage.type and footage.type.startswith("video"):
                            frames = extract_frames_jpeg(footage_bytes)
                            visual_description = describe_video_frames(frames)
                        else:
                            visual_description = describe_image(footage_bytes)
                    except (VisionClientError, VideoExtractionError) as exc:
                        st.warning(f"Could not analyze footage: {exc}. Continuing with the text description only.")

            intake = MatchIntake(
                situation=situation,
                team_a=team_a,
                team_b=team_b,
                visual_description=visual_description,
            )
            run_prediction(intake)

    # Clear pre-fill after use
    if "prefill_situation" in st.session_state and situation == st.session_state.get("prefill_situation"):
        pass  # keep it until user edits or calls it

# ── Tab 2: Famous Incidents ───────────────────────────────────────────────────
with tab_incidents:
    st.markdown(
        '<p class="subtitle" style="margin-bottom:1.2rem;">Five VAR decisions that took forever '
        "and changed everything. Watch the clip, then load it to see what the Laws actually say.</p>",
        unsafe_allow_html=True,
    )

    incidents = load_incidents()
    for inc in incidents:
        teams_split = inc["teams"].split(" vs ")
        t_a = teams_split[0] if len(teams_split) > 0 else ""
        t_b = teams_split[1] if len(teams_split) > 1 else ""

        outcome_color = ruling_color(inc["outcome"])

        st.markdown(
            f"""
            <div class="incident-card">
                <p class="incident-title">{inc["title"]}</p>
                <p class="incident-meta">{inc["competition"]} &nbsp;·&nbsp; {inc["date"]}</p>
                <p class="incident-summary">{inc["summary"]}</p>
                <p class="incident-infamous">{inc["why_infamous"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_watch, col_load = st.columns([1, 1])
        with col_watch:
            st.link_button("Watch on YouTube", inc["youtube_url"], use_container_width=True)
        with col_load:
            if st.button("Load incident", key=f"load_{inc['id']}", use_container_width=True):
                st.session_state["prefill_situation"] = inc["situation_prefill"]
                st.session_state["prefill_team_a"] = t_a
                st.session_state["prefill_team_b"] = t_b
                st.toast(f"Loaded: {inc['title']}. Switch to New Incident to call it.")

        st.markdown("<div style='margin-bottom:0.25rem;'></div>", unsafe_allow_html=True)
