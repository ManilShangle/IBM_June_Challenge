"""Smoke tests: verify retrieval and prediction return well-shaped results.

Requires data/processed/law_sections.json and law_embeddings.npy to exist
(run `python -m src.ingest_laws` and `python -m src.build_index` first),
and a working Granite backend configured in .env for test_predict_verdict_shape.
"""
import pytest

from src.retrieval import retrieve_relevant_laws
from src.verdict_match import rulings_match


def test_rulings_match_handles_no_vs_not_phrasing():
    assert rulings_match("Not Offside (level = onside)", "No Offside")


def test_rulings_match_ignores_descriptive_suffix_negation():
    assert rulings_match("Yellow Card - unsporting behavior/no excessive force", "Yellow Card")


def test_rulings_match_rejects_opposite_polarity():
    assert not rulings_match("Penalty - handball", "No Penalty")


def test_retrieval_returns_nonempty_chunks():
    chunks = retrieve_relevant_laws("striker offside as the ball is played forward")
    assert len(chunks) > 0
    assert all(chunk.text for chunk in chunks)
    assert all(chunk.law_no for chunk in chunks)


def test_retrieval_offside_query_favors_law_11():
    chunks = retrieve_relevant_laws("clear offside position, last defender, through ball")
    law_nos = [c.law_no for c in chunks]
    assert 11 in law_nos


@pytest.mark.skip(reason="Requires a configured Granite backend (watsonx/HF/ollama) - run manually")
def test_predict_verdict_shape():
    from src.predictor import predict_verdict

    result = predict_verdict(
        "Defender's arm is raised above shoulder height and blocks a goal-bound shot inside the box."
    )
    assert result.predicted_ruling
    assert 0 <= result.confidence_percent <= 100
    assert result.law_citation
    assert result.rationale
