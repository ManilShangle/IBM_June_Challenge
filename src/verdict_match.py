"""Shared helper for comparing a predicted ruling against a scenario's
ground truth. Used by both the Streamlit app and run_demo_check.py so the
two never drift out of sync.
"""
import re

_NEGATION_WORDS = {"no", "not"}


def _normalize(ruling: str) -> tuple[bool, set[str]]:
    """Returns (is_negated, core_keywords) for a ruling string.

    Only the primary ruling label (before any "(" parenthetical or "-"
    explanatory suffix) is considered, so descriptive text like "- unsporting
    behavior/no excessive force" doesn't get misread as a negation of the
    ruling itself.
    """
    text = ruling.lower().split("(")[0].split("-")[0].split("/")[0]
    words = re.findall(r"[a-z]+", text)
    is_negated = any(w in _NEGATION_WORDS for w in words)
    core = {w for w in words if w not in _NEGATION_WORDS and w not in {"a", "the"}}
    return is_negated, core


def rulings_match(ground_truth: str, predicted_ruling: str) -> bool:
    gt_negated, gt_keywords = _normalize(ground_truth)
    pred_negated, pred_keywords = _normalize(predicted_ruling)

    if not gt_keywords or not pred_keywords:
        return False
    if gt_negated != pred_negated:
        return False
    return bool(gt_keywords & pred_keywords)
