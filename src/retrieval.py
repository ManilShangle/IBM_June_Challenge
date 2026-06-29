"""Retrieve the law sections most relevant to an incident description.

Combines embedding cosine-similarity (semantic match) with a deterministic
keyword-boost layer (insurance against embedding misses on a small corpus).
"""
import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL_NAME,
    LAW_EMBEDDINGS_PATH,
    LAW_SECTIONS_JSON_PATH,
    RETRIEVAL_TOP_K,
)

# keyword -> law numbers it should boost. Checked against the lowercased
# incident text; any match adds KEYWORD_BOOST to that law's similarity score.
KEYWORD_LAW_MAP: dict[str, list[int]] = {
    "offside": [11],
    "last defender": [11],
    "onside": [11],
    "hand": [12],
    "arm": [12],
    "handball": [12],
    "tackle": [12, 14],
    "foul": [12],
    "red card": [12],
    "yellow card": [12],
    "dogso": [12],
    "denying": [12],
    "serious foul play": [12],
    "violent conduct": [12],
    "penalty": [14],
    "penalty kick": [14],
    "box": [14],
    "goalkeeper": [12, 14],
}
KEYWORD_BOOST = 0.15


@dataclass
class LawChunk:
    law_no: int
    title: str
    text: str
    section_id: str
    score: float


@lru_cache(maxsize=1)
def _load_sections() -> list[dict]:
    return json.loads(LAW_SECTIONS_JSON_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_embeddings() -> np.ndarray:
    return np.load(LAW_EMBEDDINGS_PATH)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _keyword_boost_vector(query: str, sections: list[dict]) -> np.ndarray:
    query_lower = query.lower()
    boosts = np.zeros(len(sections))
    matched_laws: set[int] = set()
    for keyword, law_nos in KEYWORD_LAW_MAP.items():
        if keyword in query_lower:
            matched_laws.update(law_nos)
    if not matched_laws:
        return boosts
    for i, section in enumerate(sections):
        if section["law_no"] in matched_laws:
            boosts[i] += KEYWORD_BOOST
    return boosts


def retrieve_relevant_laws(query: str, k: int = RETRIEVAL_TOP_K) -> list[LawChunk]:
    sections = _load_sections()
    embeddings = _load_embeddings()
    model = _load_model()

    query_vec = model.encode([query], normalize_embeddings=True)[0]
    similarities = embeddings @ query_vec
    similarities = similarities + _keyword_boost_vector(query, sections)

    top_idx = np.argsort(-similarities)[:k]
    return [
        LawChunk(
            law_no=sections[i]["law_no"],
            title=sections[i]["title"],
            text=sections[i]["text"],
            section_id=sections[i]["section_id"],
            score=float(similarities[i]),
        )
        for i in top_idx
    ]


def format_law_chunks_for_prompt(chunks: list[LawChunk]) -> str:
    parts = []
    for chunk in chunks:
        parts.append(f"[Law {chunk.law_no} - {chunk.title}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)
