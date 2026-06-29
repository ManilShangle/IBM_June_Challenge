"""Core prediction pipeline: incident text -> retrieved law context ->
Granite call -> structured VerdictResult.
"""
import json
import re
from dataclasses import dataclass

from src.granite_client import GraniteClientError, generate
from src.prompts import JSON_RETRY_SUFFIX, build_verdict_prompt
from src.retrieval import LawChunk, format_law_chunks_for_prompt, retrieve_relevant_laws


@dataclass
class MatchIntake:
    situation: str
    team_a: str = ""
    team_b: str = ""
    visual_description: str = ""


def build_incident_text(intake: MatchIntake) -> str:
    """Combines the situation description, team names, and a footage-derived
    visual description (if any) into the single incident text the rest of
    the pipeline already expects.
    """
    parts = []
    if intake.team_a.strip() and intake.team_b.strip():
        parts.append(f"Match: {intake.team_a.strip()} vs {intake.team_b.strip()}.")
    if intake.situation.strip():
        parts.append(f"Situation: {intake.situation.strip()}")
    if intake.visual_description.strip():
        parts.append(f"Visual analysis from footage: {intake.visual_description.strip()}")
    return " ".join(parts)


@dataclass
class VerdictResult:
    predicted_ruling: str
    law_citation: str
    rationale: str
    plain_english_law: str
    retrieved_law_excerpt: str
    retrieved_chunks: list[LawChunk]
    # kept for backwards-compat but no longer populated
    confidence_percent: int = 0
    key_factors: list[str] = None

    def __post_init__(self):
        if self.key_factors is None:
            self.key_factors = []


class PredictionError(RuntimeError):
    pass


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw_text: str) -> dict:
    match = _JSON_BLOCK_RE.search(raw_text)
    if not match:
        raise ValueError(f"No JSON object found in model output: {raw_text!r}")
    return json.loads(match.group(0))


def predict_verdict(incident_text: str) -> VerdictResult:
    chunks = retrieve_relevant_laws(incident_text)
    law_text = format_law_chunks_for_prompt(chunks)
    prompt = build_verdict_prompt(incident_text, law_text)

    try:
        raw = generate(prompt, max_new_tokens=380)
    except GraniteClientError as exc:
        raise PredictionError(f"Granite request failed: {exc}") from exc

    try:
        parsed = _extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        try:
            raw_retry = generate(prompt + JSON_RETRY_SUFFIX)
        except GraniteClientError as exc:
            raise PredictionError(f"Granite request failed on retry: {exc}") from exc
        try:
            parsed = _extract_json(raw_retry)
        except (ValueError, json.JSONDecodeError) as exc:
            raise PredictionError(
                "Granite did not return valid JSON after one retry. "
                "Try rephrasing the incident description."
            ) from exc

    try:
        return VerdictResult(
            predicted_ruling=str(parsed["predicted_ruling"]),
            law_citation=str(parsed["law_citation"]),
            rationale=str(parsed["rationale"]),
            plain_english_law=str(parsed.get("plain_english_law", "")),
            retrieved_law_excerpt=law_text,
            retrieved_chunks=chunks,
        )
    except KeyError as exc:
        raise PredictionError(f"Granite JSON missing expected field: {exc}") from exc
