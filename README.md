# AdVARtage

Get the edge before the call. Built for the **IBM SkillsBuild AI Builders Challenge** (June 2026, theme: *"AI Inside the Match"*).

## The problem

VAR reviews leave fans, commentators, and broadcasters in suspense for minutes with no way to anticipate the outcome. Every submission I reviewed among the 238 other entries in the gallery only **explains a VAR call after it is announced** (VAR ENFORCER, E-VAR Companion, PitchSense AI), or predicts the **match result**, not the ruling itself. I searched for "predict," "offside," "referee," and "var" across the gallery: no other project forecasts the VAR decision before it is revealed.

## What it does

Upload a match clip. My AI reads the footage with **IBM Granite Vision**, retrieves the relevant section of the **IFAB Laws of the Game**, and returns a predicted ruling before the official announcement — with the exact Law cited and a plain-English explanation.

```
Upload clip → Granite Vision describes the incident → Laws retrieved → IBM Granite predicts ruling
```

Output per prediction:

- **Predicted ruling** (e.g. "Goal Disallowed", "Penalty", "No Offside")
- **Law citation** (e.g. "Law 11 - Offside")
- **Rationale** in plain language, grounded in the retrieved law text
- **Plain-English explanation** of what the cited law actually means

The Famous Incidents tab ships five pre-loaded clips of real VAR controversies (Man City vs Spurs 2019, France vs Tunisia WC 2022, Luis Diaz disallowed goal 2023, Japan vs Spain ghost ball WC 2022, Brazil vs Switzerland WC 2022) so the app can be demonstrated immediately without sourcing footage.

## Why this fits the challenge

The June Challenge focuses on trust, transparency, and fan understanding. VAR delay and opacity is one of the most discussed pain points in modern football broadcasting. This tool gives fans and commentators a transparent, rules-grounded read on a developing incident, not a silent wait or a post-hoc explanation.

| Criterion | How this project addresses it |
|---|---|
| **Technical execution** | IBM Granite, IBM Granite Vision, Docling, Langflow, OpenCV, and sentence-transformers working end-to-end. Modular, tested codebase with multiple Granite backends (Replicate, watsonx.ai, HuggingFace, Ollama, Langflow) switchable via `.env`. |
| **Innovation** | Verified against the 238-project gallery: no other entry predicts the VAR ruling before reveal. Video footage as primary input — not a typed sentence — is the angle no other submission takes. |
| **Challenge fit** | Uses all required IBM technologies (Granite for prediction, Granite Vision for footage analysis, Docling for PDF ingestion, Langflow for pipeline orchestration). Targets the fan-understanding focus area directly. |
| **Feasibility** | Runs on a free-tier Replicate account. Setup is under 10 minutes. Demo clips are included so the app can be shown immediately. |

## Tech stack

| Technology | Role |
|---|---|
| **IBM Granite** (`granite-4.0-h-small` via Replicate) | Generates the predicted ruling, law citation, rationale, and plain-English explanation |
| **IBM Granite Vision** (`granite-vision-3.3-2b` via Replicate) | Reads uploaded footage (video frames or images) and describes what is relevant to the VAR decision |
| **Docling** | Parses the official IFAB Laws of the Game 2025/26 PDF into structured, retrievable text chunks |
| **Langflow** | Orchestrates the retrieve-predict-explain pipeline visually; the exported flow calls the same IBM watsonx.ai Granite endpoint the Python app uses when `GRANITE_BACKEND=watsonx` |
| **Streamlit** | Demo UI |
| **OpenCV** | Extracts evenly-spaced frames from uploaded video clips for multi-frame analysis |
| **sentence-transformers** | In-memory semantic retrieval over ~130 IFAB Law chunks; no vector database required |

## Architecture

```
Upload footage (video or image)
        |
        v
video_utils.py          -- OpenCV extracts frames from video clips
        |
        v
vision_client.py        -- IBM Granite Vision describes player positions, contact,
        |                  arm height, pitch location across all frames
        v
predictor.build_incident_text()  -- combines visual description into incident text
        |
        v
retrieval.py            -- sentence-transformer embeds the incident, cosine-similarity
        |                  + keyword boost retrieves top-4 relevant IFAB Law chunks
        v
predictor.predict_verdict()  -- prompt sent to IBM Granite; response parsed as JSON
        |                       (ruling, law citation, rationale, plain-English law)
        v
app.py (Streamlit)      -- renders verdict card with ruling, citation, and explanation
```

The same pipeline is also expressed as a Langflow flow in [`/flows/var_predictor_flow.json`](flows/var_predictor_flow.json) for visual inspection.

## Setup

```bash
git clone https://github.com/ManilShangle/IBM_June_Challenge.git
cd IBM_June_Challenge

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

cp .env.example .env
# Set REPLICATE_API_TOKEN (free account + free IBM credit below)

python -m src.ingest_laws       # parse IFAB PDF with Docling  (one-time)
python -m src.build_index       # build retrieval index         (one-time)

streamlit run app/app.py
```

Tests:

```bash
python -m pytest tests/
```

## Getting a Replicate API token

1. Sign up at [replicate.com](https://replicate.com) with GitHub
2. Claim free IBM Granite credit at [replicate.fyi/ibm](https://replicate.fyi/ibm) (the official link for this challenge)
3. Copy your token from [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)
4. Set `REPLICATE_API_TOKEN=your_token` in `.env`

Footage analysis (IBM Granite Vision) runs on the same Replicate account.

## Granite backends

Switch `GRANITE_BACKEND` in `.env` without changing any code:

| Value | Notes |
|---|---|
| `replicate` (default) | Free tier. Vision analysis uses this backend. |
| `watsonx` | IBM watsonx.ai Lite plan (free). Fastest for text prediction once credentials are set. |
| `huggingface` | HuggingFace Inference API against a hosted `ibm-granite` model. |
| `ollama` | Local Granite checkpoint via Ollama. Fully offline, useful for unreliable demo-day Wi-Fi. |
| `langflow` | Langflow as the runtime. Start `langflow run`, import the flow from `/flows`, set `LANGFLOW_FLOW_ID`. |

## Demo video

https://youtu.be/GeqUicBTiOo

## Repository layout

```
app/        Streamlit UI
src/        Pipeline: retrieval, prediction, Granite and Vision clients, Docling ingestion
data/       IFAB Laws PDF, parsed law text, famous incident definitions
flows/      Exported Langflow flow
scripts/    Langflow end-to-end test
tests/      Automated tests
```

## Limitations

- Footage analysis uses up to 3 evenly-spaced frames from a clip, not full motion tracking.
- Predicts a single incident per upload, not a full match.
- AI predictions will not always match the official ruling. This is a demo tool, not a refereeing authority.
- Future: player-level geometric offside line analysis, live broadcast integration.

## Acknowledgments

- [IFAB Laws of the Game 2025/26](https://www.theifab.com/laws-of-the-game-documents/)
- IBM SkillsBuild AI Builders Challenge
