# VAR Decision Predictor: AI Inside the Match

Built for the **IBM SkillsBuild AI Builders Challenge** (June 2026 Innovation Challenge, theme: *"AI Inside the Match"*).

## Problem

VAR (Video Assistant Referee) reviews leave fans, commentators, and broadcasters in suspense for minutes with no way to anticipate the outcome. Every AI tool we found among the 238 other challenge submissions only **explains a VAR call after it's announced** (e.g. VAR ENFORCER, E-VAR Companion, PitchSense AI), or predicts the **match outcome**, not the ruling itself. We searched the submission gallery for "predict," "offside," "referee," and "var": nobody is forecasting the VAR decision itself before it's revealed.

## Our AI / technical approach

The app takes a full intake of the incident, not just a sentence: the two teams involved, a description of the situation, and footage (an image or a video clip) if available. It predicts the most likely VAR ruling, before it's officially revealed, with:

- A **predicted ruling** (e.g. "Offside, goal disallowed")
- A **confidence score** reflecting how clear-cut vs. borderline the incident is
- A **rationale** grounded in the actual **IFAB Laws of the Game 2025/26**, with the exact Law cited
- The retrieved Law text shown for full transparency, so the prediction is never a black box

When footage is attached, **IBM Granite Vision** describes what's visible in the frame (player positions, contact, arm position relative to the body, where on the pitch it happened), and that visual read is folded into the same incident text the rest of the pipeline already reasons over. Video clips have a representative frame extracted automatically.

13 preset incident scenarios (offside, handball, penalty/DOGSO, red card, mixing clear-cut and genuinely borderline cases) are also included with known textbook ground truths, so the text-only prediction path can be validated rather than taken on faith. Against this set, the pipeline currently matches the textbook ruling on 10/13 (about 77%) scenarios. The misses are themselves informative: they tend to land on genuinely contested edge cases (e.g. the "level = onside" boundary), which is consistent with how real VAR controversies arise.

## Why this matters for the challenge

This project speaks directly to the June Challenge's "trust and transparency" and "fan understanding" focus areas. VAR delay and opacity is one of the most widely discussed pain points in modern football broadcasting; this tool gives fans, commentators, and broadcasters a transparent, rules-grounded read on a developing incident instead of a silent wait.

Prediction with a confidence score is the headline feature: anticipation is the value, not retrospective explanation. The rules-grounded rationale exists to justify why the model is confident or not, distinguishing it from a black-box guess, but it's the supporting evidence, not the pitch. Taking teams and footage as real intake, not just typed text, is what makes the prediction usable on an actual incident rather than only on a pre-written description, and it's the angle no other submission in the gallery takes.

## How this fits the judging criteria

| Criterion | How this project addresses it |
|---|---|
| **Technical execution** | Five IBM and open-source technologies working together end-to-end (IBM Granite, IBM Granite Vision, Docling, Langflow, plus OpenCV and sentence-transformers for retrieval), a modular and tested codebase (`src/`, `tests/`), multiple Granite backends with automatic fallback so a single API hiccup cannot sink a live demo, and a working exported Langflow flow alongside the Streamlit app. |
| **Innovation** | Verified directly against the 238-project submission gallery that no other entry predicts the VAR ruling itself before reveal. Combining real match intake (teams, situation, footage) with vision-grounded analysis and a confidence-calibrated, law-cited prediction is a genuinely novel combination, not a repeat of the post-hoc explainers or match-outcome predictors already submitted. |
| **Challenge fit** | Built around the official focus areas (trust and transparency, fan understanding), uses required technologies (Granite, Docling, Langflow), and targets a real, widely felt pain point: not knowing what's coming during a VAR review. |
| **Feasibility / real-world impact** | Runs today on a laptop with a free-tier API key, with documented setup and a sanity-check script that validates every preset scenario before a demo. Honest about its current scope (frame-level vision, not full video tracking) so reviewers see a credible extension path rather than an overclaimed one. |

## Tech stack

- **IBM Granite** (via watsonx.ai or Replicate) generates the predicted ruling, confidence, and rationale
- **IBM Granite Vision** reads uploaded footage and describes what's relevant to the decision
- **Docling** parses the official IFAB Laws of the Game PDF into structured, retrievable text
- **Langflow** orchestrates the retrieve, predict, explain pipeline visually (see [`/flows`](flows/)). The exported flow calls the same IBM watsonx.ai Granite endpoint that the Python app uses when `GRANITE_BACKEND=watsonx`, so the visual flow and the Streamlit app are two interfaces to the same backend.
- **Streamlit** is the demo UI
- **OpenCV** extracts a representative frame from uploaded video clips
- Lightweight in-memory retrieval (sentence-transformer embeddings, cosine similarity, and keyword boosting) needs no vector database for the ~50-130 Law chunks involved

## Architecture

```
Match intake (teams + situation text + optional footage)
        |
        v
vision_client.py / video_utils.py  -- if footage attached, IBM Granite Vision describes
        |                             the frame; video clips get a middle frame extracted
        v
predictor.build_incident_text()   -- combines teams + situation + visual read into one
        |                             incident description
        v
retrieval.py   -- embeds the incident text, cosine-similarity + keyword boost against
        |          IFAB Law chunks (top-4 relevant excerpts, chunked per subsection)
        v
predictor.py   -- builds a structured prompt, calls IBM Granite via granite_client.py
        |          (strict JSON: ruling, confidence, law citation, rationale, key factors)
        v
app.py (Streamlit) -- renders the verdict card, confidence bar, citation, retrieved Law text
```

The same retrieve-then-predict pipeline is also expressed as a Langflow flow (see [`/flows/var_predictor_flow.json`](flows/var_predictor_flow.json)) for visual inspection, independent of the Streamlit app's runtime path.

## How to run

```bash
git clone https://github.com/ManilShangle/IBM_June_Challenge.git
cd IBM_June_Challenge
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

cp .env.example .env
# fill in REPLICATE_API_TOKEN (see "Granite access" below), or switch GRANITE_BACKEND
# to watsonx / huggingface / ollama and fill in those credentials instead

python -m src.ingest_laws       # one-time: parse IFAB PDF with Docling
python -m src.build_index       # one-time: build the retrieval index

streamlit run app/app.py
```

Sanity-check the text-only pipeline against every preset scenario before demoing:

```bash
python -m scripts.run_demo_check
```

Run the automated tests:

```bash
python -m pytest tests/
```

## Granite access

This project supports four interchangeable Granite backends for text prediction, switchable via `.env` (`GRANITE_BACKEND`). Footage analysis specifically requires the Replicate backend, since that's the only one wired up to IBM Granite Vision.

| Backend | Use case |
|---|---|
| `replicate` (default) | Lowest-friction path: sign in with GitHub, claim free credit via [replicate.fyi/ibm](https://replicate.fyi/ibm) (the official link IBM provides for this challenge's Granite workshops), generate a token. No IBM Cloud account needed. |
| `watsonx` | IBM watsonx.ai Lite plan (free), the full IBM Cloud path, more setup overhead |
| `huggingface` | Hugging Face Inference API against a hosted `ibm-granite` model, alternative fallback |
| `ollama` | Local Granite checkpoint via Ollama, fully offline fallback for unreliable demo-day WiFi |
| `langflow` | Langflow as the live runtime: starts the exported flow in [`/flows`](flows/), calls its REST API (`/api/v1/run/{flow_id}`), and reads the Granite response from the Chat Output node. Run `langflow run` locally, import the flow, and set `LANGFLOW_FLOW_ID` in `.env`. |

If a single backend has an outage or rate-limit issue on demo day, switching `GRANITE_BACKEND` in `.env` is a one-line change, not a rewrite.

## Demo video

A recorded walkthrough (3 minutes or under) will be linked here before final submission on the challenge platform.

## Repository contents

```
app/        Streamlit demo UI
src/        Core pipeline: retrieval, prediction, Granite and Vision clients, Docling ingestion
data/       IFAB Laws source PDF, Docling-parsed law text, preset scenarios
flows/      Exported Langflow flow (visual pipeline orchestration)
scripts/    Batch sanity-check script for validating predictions against ground truth
tests/      Automated unit and smoke tests
```

## Limitations & future work

- Footage analysis sends up to 5 evenly-spaced frames from a video clip to IBM Granite Vision in a single call, giving the model temporal context across the incident. Full motion-tracking or player-level geometry (automated offside lines) remains future work.
- Predicts a single incident's ruling, not a full match.
- This is a research prototype, not a legal or refereeing authority. Predictions are AI-generated and will not always match the official outcome (see the accuracy note above).
- Future work: player-level geometric analysis for tight offside calls, live broadcast stream integration, larger ground-truth validation corpus.

## Acknowledgments

- [IFAB Laws of the Game 2025/26](https://www.theifab.com/laws-of-the-game-documents/), source rules text
- IBM SkillsBuild AI Builders Challenge
