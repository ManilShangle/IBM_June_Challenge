"""Sanity-check script: run every preset scenario through the full pipeline
and print predicted ruling vs. ground truth, so you can eyeball accuracy
and tune the prompt before the live demo.

Run: python -m scripts.run_demo_check
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SCENARIOS_PATH
from src.predictor import PredictionError, predict_verdict
from src.verdict_match import rulings_match


def main():
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    passed, failed, errored = 0, 0, 0

    for scenario in scenarios:
        print(f"\n=== {scenario['id']} [{scenario['category']}/{scenario['difficulty']}] ===")
        print(f"Incident: {scenario['description']}")
        print(f"Ground truth: {scenario['ground_truth']}")
        try:
            result = predict_verdict(scenario["description"])
        except PredictionError as exc:
            print(f"ERROR: {exc}")
            errored += 1
            continue

        print(f"Predicted: {result.predicted_ruling} ({result.confidence_percent}%)")
        print(f"Law cited: {result.law_citation}")
        print(f"Rationale: {result.rationale}")

        match = rulings_match(scenario["ground_truth"], result.predicted_ruling)
        print("MATCH" if match else "MISMATCH (review manually - borderline cases may legitimately differ)")
        if match:
            passed += 1
        else:
            failed += 1

    print(f"\n=== Summary: {passed} matched, {failed} mismatched, {errored} errored / {len(scenarios)} total ===")


if __name__ == "__main__":
    main()
