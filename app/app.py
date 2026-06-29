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

ACCENT    = "#00D46A"
NEGATIVE  = "#FF5757"
WARNING   = "#FFAA33"
INK       = "#F0F6FC"
MUTED     = "#8B949E"
BG        = "#0D1117"
SURFACE   = "#161B22"
BORDER    = "#30363D"

RULING_COLORS = {
    "Offside":                  NEGATIVE,
    "Goal Disallowed":          NEGATIVE,
    "Penalty":                  NEGATIVE,
    "Red Card":                 NEGATIVE,
    "No Offside":               ACCENT,
    "Goal Stands":              ACCENT,
    "No Penalty":               ACCENT,
    "No Card":                  ACCENT,
    "Yellow Card":              WARNING,
    "VAR Review - No Clear Error": WARNING,
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: {BG} !important;
    color: {INK};
}}

#MainMenu, footer, header [data-testid="stToolbar"] {{ visibility: hidden; }}

.block-container {{
    padding-top: 2.5rem;
    max-width: 760px;
}}

/* ── Brand header ── */
.brand-lockup {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.6rem;
}}

.brand-dot {{
    width: 10px; height: 10px;
    border-radius: 50%;
    background: {ACCENT};
    box-shadow: 0 0 12px {ACCENT};
    flex-shrink: 0;
}}

.brand-mark {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {ACCENT};
}}

h1 {{
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    color: {INK} !important;
    margin-bottom: 0.25rem !important;
    text-wrap: balance;
    line-height: 1.1 !important;
}}

.subtitle {{
    color: {MUTED};
    font-size: 0.95rem;
    line-height: 1.55;
    margin-bottom: 0.5rem;
    max-width: 60ch;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 1.6rem;
    border-bottom: 1px solid {BORDER};
    background: transparent;
    margin-bottom: 1.4rem;
}}

.stTabs [data-baseweb="tab"] {{
    height: 2.4rem;
    color: {MUTED};
    font-weight: 500;
    background: transparent !important;
    padding: 0 !important;
}}

.stTabs [aria-selected="true"] {{
    color: {ACCENT} !important;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background-color: {ACCENT} !important;
}}

/* ── Upload zone ── */
div[data-testid="stFileUploader"] section {{
    border-radius: 12px;
    border: 2px dashed {BORDER};
    background: {SURFACE};
    padding: 2rem !important;
    transition: border-color 0.2s;
}}

div[data-testid="stFileUploader"] section:hover {{
    border-color: {ACCENT};
}}

div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] p {{
    color: {MUTED} !important;
}}

div[data-testid="stFileUploaderDropzoneInstructions"] div span {{
    color: {INK} !important;
    font-size: 1rem;
    font-weight: 600;
}}

/* ── Buttons ── */
.stButton button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.15s ease !important;
    border: none !important;
}}

.stButton button[kind="primary"],
.stButton button[data-testid="stBaseButton-primary"] {{
    background: {ACCENT} !important;
    color: #0D1117 !important;
    padding: 0.7rem 1.5rem !important;
}}

.stButton button:not([kind="primary"]) {{
    background: {SURFACE} !important;
    color: {INK} !important;
    border: 1px solid {BORDER} !important;
}}

.stButton button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0, 212, 106, 0.25) !important;
}}

.stLinkButton a {{
    border-radius: 8px !important;
    background: {SURFACE} !important;
    color: {INK} !important;
    border: 1px solid {BORDER} !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
    text-decoration: none !important;
}}

.stLinkButton a:hover {{
    border-color: {ACCENT} !important;
    color: {ACCENT} !important;
}}

/* ── Verdict card ── */
.verdict-card {{
    border-radius: 12px;
    overflow: hidden;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    border: 1px solid {BORDER};
}}

.verdict-card-header {{
    padding: 1.5rem 1.8rem 1.2rem;
}}

.verdict-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 0 0 0.4rem;
    opacity: 0.8;
}}

.verdict-ruling {{
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.03em;
    line-height: 1.1;
}}

.verdict-body {{
    padding: 1rem 1.8rem 1.4rem;
    background: {SURFACE};
}}

.verdict-law {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: {MUTED};
    margin: 0 0 0.8rem;
}}

.verdict-rationale {{
    font-size: 0.97rem;
    line-height: 1.65;
    color: #C9D1D9;
    margin: 0 0 0.8rem;
}}

.verdict-plain {{
    background: rgba(255,255,255,0.04);
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.9rem;
    line-height: 1.6;
    color: {MUTED};
}}

.verdict-plain-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {ACCENT};
    margin-bottom: 0.3rem;
    opacity: 0.8;
}}

/* ── Section label ── */
.section-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 0.5rem;
}}

/* ── Incident card ── */
.incident-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1.2rem;
}}

.incident-card-body {{
    padding: 0.9rem 1.1rem 0.75rem;
}}

.incident-title {{
    font-size: 0.97rem;
    font-weight: 700;
    color: {INK};
    margin: 0 0 0.2rem;
}}

.incident-desc {{
    font-size: 0.85rem;
    line-height: 1.55;
    color: {MUTED};
    margin: 0;
}}

/* ── Upload hint ── */
.upload-hint {{
    font-size: 0.8rem;
    color: {MUTED};
    margin-top: 0.4rem;
    margin-bottom: 1.2rem;
}}

/* ── Spinner ── */
.stSpinner > div > div {{
    border-top-color: {ACCENT} !important;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}

[data-testid="stExpander"] summary {{
    color: {MUTED} !important;
    font-size: 0.85rem !important;
}}

[data-testid="stExpander"] summary:hover {{
    color: {INK} !important;
}}

/* ── Alerts ── */
.stAlert {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {INK} !important;
}}

/* ── Demo clips note ── */
.demo-note {{
    background: rgba(0, 212, 106, 0.07);
    border: 1px solid rgba(0, 212, 106, 0.2);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: {MUTED};
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}}

.demo-note strong {{
    color: {ACCENT};
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


def yt_embed(video_url: str, height: int = 220) -> str:
    vid_id = video_url.split("v=")[-1].split("&")[0]
    return f"""
    <div style="border-radius:10px;overflow:hidden;margin:0;">
        <iframe width="100%" height="{height}" src="https://www.youtube.com/embed/{vid_id}"
            frameborder="0" allow="accelerometer; autoplay; clipboard-write;
            encrypted-media; gyroscope; picture-in-picture" allowfullscreen
            style="display:block;"></iframe>
    </div>"""


def render_verdict(result: VerdictResult, visual_desc: str | None = None):
    color = ruling_color(result.predicted_ruling)
    # Compute a faint tinted bg for the header
    if color == NEGATIVE:
        hdr_bg = "rgba(255,87,87,0.12)"
    elif color == ACCENT:
        hdr_bg = "rgba(0,212,106,0.12)"
    else:
        hdr_bg = "rgba(255,170,51,0.12)"

    st.markdown(
        f"""<div class="verdict-card">
            <div class="verdict-card-header" style="background:{hdr_bg};">
                <p class="verdict-label" style="color:{color};">VAR ruling</p>
                <p class="verdict-ruling" style="color:{color};">{result.predicted_ruling}</p>
            </div>
            <div class="verdict-body">
                <p class="verdict-law">{result.law_citation}</p>
                <p class="verdict-rationale">{result.rationale}</p>
                {"" if not result.plain_english_law else f'<div class="verdict-plain"><div class="verdict-plain-label">What this law means</div>{result.plain_english_law}</div>'}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    if visual_desc:
        with st.expander("What the footage analysis saw"):
            st.write(visual_desc)

    with st.expander("IFAB Law text used for grounding"):
        st.text(result.retrieved_law_excerpt)


def run_prediction_from_video(video_bytes: bytes, filename: str, fallback_situation: str = ""):
    visual_desc = ""
    if video_bytes:
        with st.spinner("Reading the footage..."):
            try:
                frames = extract_frames_jpeg(video_bytes)
                visual_desc = describe_video_frames(frames)
            except (VisionClientError, VideoExtractionError) as exc:
                st.warning(f"Could not analyze footage: {exc}.")
                if fallback_situation:
                    visual_desc = ""

    situation = visual_desc or fallback_situation
    if not situation:
        st.error("Could not extract a description from the footage. Try a different clip.")
        return

    with st.spinner("Calling it..."):
        try:
            intake = MatchIntake(situation=situation, visual_description=visual_desc)
            result = predict_verdict(build_incident_text(intake))
        except PredictionError as exc:
            st.error(str(exc))
            return

    render_verdict(result, visual_desc if visual_desc else None)


def run_prediction_from_text(situation: str):
    with st.spinner("Calling it..."):
        try:
            intake = MatchIntake(situation=situation)
            result = predict_verdict(build_incident_text(intake))
        except PredictionError as exc:
            st.error(str(exc))
            return
    render_verdict(result)


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    f"""<div class="brand-lockup">
        <div class="brand-dot"></div>
        <span class="brand-mark">VAR Decision Predictor</span>
    </div>""",
    unsafe_allow_html=True,
)
st.title("Beat the ref to the call")
st.markdown(
    '<p class="subtitle">Drop the footage. Get the ruling grounded in the IFAB Laws of the Game, before the announcement.</p>',
    unsafe_allow_html=True,
)

tab_upload, tab_incidents = st.tabs(["Upload Footage", "Famous Incidents"])

# ── Tab 1: Upload Footage ─────────────────────────────────────────────────────
with tab_upload:
    footage = st.file_uploader(
        "Drop a clip or frame",
        type=["mp4", "mov", "jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.markdown(
        '<p class="upload-hint">Upload a video clip or still image. IBM Granite Vision reads the footage and the Laws are searched automatically.</p>',
        unsafe_allow_html=True,
    )

    # Show demo clips hint
    demo_dir = Path(__file__).resolve().parent.parent / "data" / "demo_clips"
    demo_clips = sorted(demo_dir.glob("*.mp4")) if demo_dir.exists() else []
    if demo_clips:
        names = ", ".join(f.stem.replace("_", " ") for f in demo_clips)
        st.markdown(
            f'<div class="demo-note"><strong>Demo clips ready:</strong> {names} &mdash; find them in <code>data/demo_clips/</code></div>',
            unsafe_allow_html=True,
        )

    if st.button("Call it", type="primary", key="upload_predict", disabled=footage is None):
        run_prediction_from_video(footage.getvalue(), footage.name)

# ── Tab 2: Famous Incidents ───────────────────────────────────────────────────
with tab_incidents:
    incidents = load_incidents()
    for inc in incidents:
        st.markdown(
            f"""<div class="incident-card">
                <div style="border-radius:10px 10px 0 0; overflow:hidden;">
                    {yt_embed(inc["youtube_url"], height=210)}
                </div>
                <div class="incident-card-body">
                    <p class="incident-title">{inc["title"]} &nbsp;<span style="font-weight:400;color:{MUTED};font-size:0.8rem;">{inc["date"]}</span></p>
                    <p class="incident-desc">{inc["summary"]}</p>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        if st.button("Analyze this incident", key=f"analyze_{inc['id']}", use_container_width=True):
            run_prediction_from_text(inc["situation_prefill"])

        st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
