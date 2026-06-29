"""Streamlit demo UI for the VAR Decision Predictor."""
import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import INCIDENTS_PATH
from src.predictor import MatchIntake, PredictionError, VerdictResult, build_incident_text, predict_verdict
from src.vision_client import VisionClientError, describe_video_frames
from src.video_utils import VideoExtractionError, extract_frames_jpeg

st.set_page_config(
    page_title="VAR Decision Predictor",
    page_icon=":material/sports_soccer:",
    layout="centered",
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0C0E12"
SURFACE  = "#15181E"
RAISED   = "#1C1F27"
BORDER   = "#2C2F38"
INK      = "#EFF0F4"
MUTED    = "#8D9099"
POSITIVE = "#3DB87D"
NEGATIVE = "#D94840"
WARNING  = "#D6872A"

RULING_COLORS = {
    "Offside":                     NEGATIVE,
    "Goal Disallowed":             NEGATIVE,
    "Penalty":                     NEGATIVE,
    "Red Card":                    NEGATIVE,
    "No Offside":                  POSITIVE,
    "Goal Stands":                 POSITIVE,
    "No Penalty":                  POSITIVE,
    "No Card":                     POSITIVE,
    "Yellow Card":                 WARNING,
    "VAR Review - No Clear Error": WARNING,
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: {BG} !important;
    color: {INK};
}}

#MainMenu, footer, header [data-testid="stToolbar"] {{ visibility: hidden; }}

.block-container {{
    padding-top: 3rem;
    padding-bottom: 5rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    max-width: 680px;
    margin-left: auto !important;
    margin-right: auto !important;
}}

/* ── App header ── */
.app-header {{
    text-align: center;
    margin-bottom: 2rem;
}}

.app-name {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {MUTED};
    margin: 0 0 0.55rem;
}}

.app-title {{
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: {INK};
    margin: 0 0 0.5rem;
    text-wrap: balance;
    line-height: 1.15;
}}

.app-desc {{
    font-size: 0.88rem;
    line-height: 1.65;
    color: {MUTED};
    max-width: 46ch;
    margin: 0 auto;
    text-align: center;
    text-wrap: pretty;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 1px solid {BORDER};
    background: transparent;
    margin-bottom: 1.6rem;
    justify-content: center;
}}

.stTabs [data-baseweb="tab"] {{
    height: 2.6rem;
    color: {MUTED};
    font-weight: 500;
    font-size: 0.9rem;
    background: transparent !important;
    padding: 0 1.4rem !important;
}}

.stTabs [aria-selected="true"] {{
    color: {INK} !important;
    font-weight: 600 !important;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background-color: {INK} !important;
    height: 2px !important;
}}

/* ── Onboarding steps ── */
.steps {{
    display: flex;
    margin-bottom: 1.25rem;
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
    background: {SURFACE};
}}

.step {{
    flex: 1;
    padding: 0.9rem 1rem;
    display: flex;
    align-items: flex-start;
    gap: 0.65rem;
}}

.step + .step {{
    border-left: 1px solid {BORDER};
}}

.step-num {{
    font-size: 1rem;
    font-weight: 700;
    color: {MUTED};
    flex-shrink: 0;
    line-height: 1.3;
    min-width: 1rem;
}}

.step-text {{
    font-size: 0.8rem;
    line-height: 1.5;
    color: {MUTED};
}}

.step-text strong {{
    color: {INK};
    display: block;
    font-size: 0.85rem;
    margin-bottom: 0.1rem;
}}

/* ── File uploader ── */
div[data-testid="stFileUploader"] section {{
    border-radius: 10px;
    border: 1.5px dashed {BORDER};
    background: {SURFACE};
    transition: border-color 0.18s ease;
    cursor: pointer;
}}

div[data-testid="stFileUploader"] section:hover {{
    border-color: {MUTED};
}}

div[data-testid="stFileUploaderDropzoneInstructions"] div span {{
    color: {INK} !important;
    font-weight: 600;
    font-size: 0.95rem;
}}

/* Hide "Limit: 200MB per file • MP4, MOV..." text */
div[data-testid="stFileUploaderDropzoneInstructions"] small {{
    display: none !important;
}}

/* ── Buttons ── */
.stButton button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    transition: background 0.15s ease !important;
    width: 100%;
}}

button[kind="primaryFormSubmit"],
.stFormSubmitButton button,
.stButton button[kind="primary"] {{
    background: {INK} !important;
    color: {BG} !important;
    border: none !important;
    padding: 0.65rem 1.4rem !important;
}}

button[kind="primaryFormSubmit"]:hover,
.stFormSubmitButton button:hover,
.stButton button[kind="primary"]:hover {{
    background: #ffffff !important;
}}

.stButton button:not([kind="primary"]) {{
    background: {RAISED} !important;
    color: {INK} !important;
    border: 1px solid {BORDER} !important;
}}

.stButton button:not([kind="primary"]):hover {{
    background: {SURFACE} !important;
    border-color: {MUTED} !important;
}}

.stButton button:disabled,
.stFormSubmitButton button:disabled {{
    opacity: 0.35 !important;
    cursor: not-allowed !important;
}}

/* ── Verdict card ── */
.verdict-card {{
    border-radius: 10px;
    overflow: hidden;
    margin-top: 1.2rem;
    margin-bottom: 0.5rem;
    border: 1px solid {BORDER};
}}

.verdict-header {{
    padding: 1.4rem 1.6rem 1.1rem;
}}

.verdict-label {{
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.75;
    margin: 0 0 0.3rem;
}}

.verdict-ruling {{
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.025em;
    line-height: 1.1;
}}

.verdict-body {{
    padding: 1rem 1.6rem 1.4rem;
    background: {SURFACE};
    border-top: 1px solid {BORDER};
}}

.verdict-law {{
    font-size: 0.76rem;
    font-weight: 500;
    color: {MUTED};
    margin: 0 0 0.65rem;
    letter-spacing: 0.01em;
}}

.verdict-rationale {{
    font-size: 0.95rem;
    line-height: 1.65;
    color: {INK};
    opacity: 0.88;
    margin: 0 0 0.85rem;
}}

.verdict-plain-box {{
    background: {RAISED};
    border-radius: 7px;
    padding: 0.7rem 0.95rem;
}}

.verdict-plain-label {{
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 0.25rem;
}}

.verdict-plain-text {{
    font-size: 0.88rem;
    line-height: 1.6;
    color: {MUTED};
    margin: 0;
}}

/* ── Incident items ── */
.inc-title {{
    font-size: 0.97rem;
    font-weight: 700;
    color: {INK};
    margin: 0 0 0.08rem;
}}

.inc-meta {{
    font-size: 0.74rem;
    color: {MUTED};
    margin: 0 0 0.45rem;
}}

.inc-summary {{
    font-size: 0.82rem;
    line-height: 1.55;
    color: {MUTED};
    margin: 0.4rem 0 0.65rem;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    margin-top: 0.5rem;
}}

[data-testid="stExpander"] summary span {{
    color: {MUTED} !important;
    font-size: 0.82rem !important;
}}

/* ── Misc ── */
.stSpinner > div > div {{ border-top-color: {INK} !important; }}

.stAlert {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {INK} !important;
}}

[data-testid="stToast"] {{
    background: {RAISED} !important;
    border: 1px solid {BORDER} !important;
    color: {INK} !important;
}}

hr {{ border-color: {BORDER} !important; margin: 1.4rem 0 !important; }}

@media (prefers-reduced-motion: reduce) {{
    * {{ transition: none !important; }}
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


def render_verdict(result: VerdictResult, visual_desc: str | None = None):
    color = ruling_color(result.predicted_ruling)
    if color == NEGATIVE:
        hdr_tint = "rgba(217,72,64,0.13)"
    elif color == POSITIVE:
        hdr_tint = "rgba(61,184,125,0.11)"
    else:
        hdr_tint = "rgba(214,135,42,0.11)"

    # Build HTML as concatenated single-line strings to avoid
    # Markdown's 4-space-indent → code-block ambiguity
    plain_html = ""
    if result.plain_english_law:
        law_escaped = (
            result.plain_english_law
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        plain_html = (
            '<div class="verdict-plain-box">'
            '<div class="verdict-plain-label">What this law means</div>'
            f'<p class="verdict-plain-text">{law_escaped}</p>'
            '</div>'
        )

    card_html = (
        '<div class="verdict-card">'
        f'<div class="verdict-header" style="background:{hdr_tint};">'
        f'<p class="verdict-label" style="color:{color};">Predicted ruling</p>'
        f'<p class="verdict-ruling" style="color:{color};">{result.predicted_ruling}</p>'
        '</div>'
        f'<div class="verdict-body">'
        f'<p class="verdict-law">{result.law_citation}</p>'
        f'<p class="verdict-rationale">{result.rationale}</p>'
        f'{plain_html}'
        '</div>'
        '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    if visual_desc:
        with st.expander("What Granite Vision saw in the footage"):
            st.write(visual_desc)


def run_prediction_from_video(video_bytes: bytes):
    visual_desc = ""
    with st.spinner("Reading the footage with Granite Vision..."):
        try:
            frames = extract_frames_jpeg(video_bytes)
            visual_desc = describe_video_frames(frames)
        except (VisionClientError, VideoExtractionError) as exc:
            st.warning(f"Could not analyze footage: {exc}")
            return

    with st.spinner("Matching against the Laws of the Game... (first request may take 1-2 min to load the model)"):
        try:
            intake = MatchIntake(situation=visual_desc, visual_description=visual_desc)
            result = predict_verdict(build_incident_text(intake))
        except PredictionError as exc:
            st.error(str(exc))
            return

    render_verdict(result, visual_desc)


def run_prediction_from_text(situation: str):
    with st.spinner("Matching against the Laws of the Game... (first request may take 1-2 min to load the model)"):
        try:
            intake = MatchIntake(situation=situation)
            result = predict_verdict(build_incident_text(intake))
        except PredictionError as exc:
            st.error(str(exc))
            return
    render_verdict(result)


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-header">'
    '<p class="app-name">VAR Decision Predictor</p>'
    '<h1 class="app-title">What will the referee call?</h1>'
    '<p class="app-desc">Upload a match clip. IBM Granite Vision reads the footage and the IFAB Laws of the Game determine the ruling.</p>'
    '</div>',
    unsafe_allow_html=True,
)

tab_upload, tab_incidents = st.tabs(["Try it", "Famous incidents"])

# ── Tab 1: Upload ─────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown(
        '<div class="steps">'
        '<div class="step">'
        '<span class="step-num">1</span>'
        '<div class="step-text"><strong>Upload a clip or photo</strong>Video gives the best result. A still frame also works.</div>'
        '</div>'
        '<div class="step">'
        '<span class="step-num">2</span>'
        '<div class="step-text"><strong>Press "Predict ruling"</strong>AI reads the footage and applies the Laws of the Game.</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.form("upload_form", clear_on_submit=False):
        footage = st.file_uploader(
            "Match footage",
            type=["mp4", "mov", "jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "Predict ruling",
            type="primary",
            use_container_width=True,
        )
        if submitted and footage is not None:
            run_prediction_from_video(footage.getvalue())
        elif submitted:
            st.warning("Upload a clip or photo first.")

# ── Tab 2: Famous Incidents ───────────────────────────────────────────────────
with tab_incidents:
    st.markdown(
        f'<p style="font-size:0.85rem;color:{MUTED};margin-bottom:1.4rem;text-align:center;">Five decisions that stopped the game. Watch the clip, then press Analyze.</p>',
        unsafe_allow_html=True,
    )

    incidents = load_incidents()
    for inc in incidents:
        vid_id = inc["youtube_url"].split("v=")[-1].split("&")[0]

        st.markdown(
            f'<p class="inc-title">{inc["title"]}</p>'
            f'<p class="inc-meta">{inc["competition"]} &nbsp;·&nbsp; {inc["date"]}</p>',
            unsafe_allow_html=True,
        )

        # components.html() renders a real iframe; st.markdown strips iframes
        components.html(
            '<div style="background:#000;border-radius:8px;overflow:hidden;">'
            f'<iframe width="100%" height="370" src="https://www.youtube.com/embed/{vid_id}"'
            ' frameborder="0" allow="accelerometer; autoplay; clipboard-write;'
            ' encrypted-media; gyroscope; picture-in-picture" allowfullscreen'
            ' style="display:block;border:none;"></iframe>'
            '</div>',
            height=370,
            scrolling=False,
        )

        st.markdown(
            f'<p class="inc-summary">{inc["summary"]}</p>',
            unsafe_allow_html=True,
        )

        if st.button("Analyze this incident", key=f"analyze_{inc['id']}", use_container_width=True):
            run_prediction_from_text(inc["situation_prefill"])

        st.divider()
