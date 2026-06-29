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
class VerdictResult:
    predicted_ruling: str
    confidence_percent: int
    law_citation: str
    rationale: str
    key_factors: list[str]
    retrieved_law_excerpt: str
    retrieved_chunks: list[LawChunk]


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
        raw = generate(prompt, max_new_tokens=900)
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
            confidence_percent=int(parsed["confidence_percent"]),
            law_citation=str(parsed["law_citation"]),
            rationale=str(parsed["rationale"]),
            key_factors=[str(f) for f in parsed.get("key_factors", [])],
            retrieved_law_excerpt=law_text,
            retrieved_chunks=chunks,
        )
    except KeyError as exc:
        raise PredictionError(f"Granite JSON missing expected field: {exc}") from exc
