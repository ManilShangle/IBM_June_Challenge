# VAR Decision Predictor — AI Inside the Match

Built for the **IBM SkillsBuild AI Builders Challenge** (June 2026, theme: *"AI Inside the Match"*).

## Problem

VAR (Video Assistant Referee) reviews leave fans, commentators, and broadcasters in suspense for minutes with no way to anticipate the outcome. Every AI tool we found among the 200+ other challenge submissions only **explains a VAR call after it's announced** (e.g. VAR ENFORCER, E-VAR Companion, PitchSense AI), or predicts the **match outcome**, not the ruling itself. We searched the submission gallery for "predict," "offside," "referee," and "var" — nobody is forecasting the VAR decision itself before it's revealed.

## What it does

Given a text description of an in-match incident, the system predicts the most likely VAR ruling — **before** it's officially revealed — with:
- A **predicted ruling** (e.g. "Offside — goal disallowed")
- A **confidence score** reflecting how clear-cut vs. borderline the incident is
- A **rationale** grounded in the actual **IFAB Laws of the Game 2025/26**, with the exact Law cited
- The retrieved Law text shown for full transparency, so the prediction is never a black box

13 preset incident scenarios (offside, handball, penalty/DOGSO, red card — mixing clear-cut and genuinely borderline cases) are included with known textbook ground truths, so the prediction can be validated rather than taken on faith. Against this set, the pipeline currently matches the textbook ruling on 10/13 (~77%) scenarios; the misses are themselves informative — they tend to land on genuinely contested edge cases (e.g. the "level = onside" boundary), which is consistent with how real VAR controversies arise.

## Why this is different

Prediction with a confidence score is the headline feature — anticipation is the value, not retrospective explanation. The rules-grounded rationale exists to justify *why* the model is confident or not, distinguishing it from a black-box guess, but it's the supporting evidence, not the pitch.

## Tech stack

- **IBM Granite** (via watsonx.ai) — generates the predicted ruling, confidence, and rationale
- **Docling** — parses the official IFAB Laws of the Game PDF into structured, retrievable text
- **Langflow** — visual orchestration of the retrieve → predict → explain pipeline (see [`/flows`](flows/))
- **Streamlit** — demo UI
- Lightweight in-memory retrieval (sentence-transformer embeddings + cosine similarity + keyword boosting) — no vector database needed for the ~15-25 Law chunks involved

## Architecture

```
User input (preset or free text)
        │
        ▼
retrieval.py   — embeds query, cosine-similarity + keyword boost against IFAB Law chunks
        │  (top-4 relevant Law excerpts, chunked per-subsection for precision)
        ▼
predictor.py   — builds a structured prompt, calls IBM Granite via granite_client.py
        │  (strict JSON: ruling, confidence, law citation, rationale, key factors)
        ▼
app.py (Streamlit) — renders verdict card, confidence bar, citation, retrieved Law text
```

The same retrieve → predict pipeline is also expressed as a Langflow flow (see [`/flows/var_predictor_flow.json`](flows/var_predictor_flow.json)) for visual inspection, independent of the Streamlit app's runtime path.

## How to run

```bash
git clone <this-repo>
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

Sanity-check the pipeline against every preset scenario before demoing:

```bash
python -m scripts.run_demo_check
```

## Granite access

This project supports four interchangeable Granite backends, switchable via `.env` (`GRANITE_BACKEND`):

| Backend | Use case |
|---|---|
| `replicate` (default) | Lowest-friction path — sign in with GitHub, claim free credit via [replicate.fyi/ibm](https://replicate.fyi/ibm) (the official link IBM provides for this challenge's Granite workshops), generate a token. No IBM Cloud account needed. |
| `watsonx` | IBM watsonx.ai Lite plan (free) — the full IBM Cloud path, more setup overhead |
| `huggingface` | Hugging Face Inference API against a hosted `ibm-granite` model — alternative fallback |
| `ollama` | Local Granite checkpoint via Ollama — fully offline fallback for unreliable demo-day WiFi |

## Demo video

[link to ≤3 minute demo video]

## Limitations & future work

- Input is a text description of an incident, not live video/computer-vision ingestion of a broadcast feed — there was neither time nor broadcast rights for that in a 2-day build.
- Predicts a single incident's ruling, not a full match.
- This is a research prototype, not a legal/refereeing authority — predictions are AI-generated and will not always match the official outcome (see accuracy note above).
- Future work: real video/CV-based incident detection, multi-camera angle reasoning, live match integration.

## Acknowledgments

- [IFAB Laws of the Game 2025/26](https://www.theifab.com/laws-of-the-game-documents/) — source rules text
- IBM SkillsBuild AI Builders Challenge
