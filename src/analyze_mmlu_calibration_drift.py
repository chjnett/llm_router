"""Diagnose MMLU confidence calibration drift without changing the frozen policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .analyze_answer_only_confidence import cross_validated_probability
from .analyze_mmlu_logit_confidence import FEATURES
from .certify_mmlu_logit_confidence import evaluate
from .common import read_jsonl, write_json


STEM = {"abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry", "college_computer_science", "college_mathematics", "college_physics", "computer_security", "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_biology", "high_school_chemistry", "high_school_computer_science", "high_school_mathematics", "high_school_physics", "high_school_statistics", "machine_learning"}
HUMANITIES = {"formal_logic", "high_school_european_history", "high_school_us_history", "high_school_world_history", "international_law", "jurisprudence", "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy", "prehistory", "professional_law", "world_religions"}
SOCIAL = {"econometrics", "high_school_geography", "high_school_government_and_politics", "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology", "human_sexuality", "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy"}


def subject_group(subject: str) -> str:
    if subject in STEM:
        return "STEM"
    if subject in HUMANITIES:
        return "humanities"
    if subject in SOCIAL:
        return "social_sciences"
    return "other"


def expected_calibration_error(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    value = 0.0
    for index in range(bins):
        mask = (probability >= edges[index]) & (probability < edges[index + 1] if index < bins - 1 else probability <= edges[index + 1])
        if mask.any():
            value += float(mask.mean()) * abs(float(probability[mask].mean()) - float(labels[mask].mean()))
    return value if total else 0.0


def calibration_metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "items": len(labels), "accuracy": float(labels.mean()), "mean_predicted_probability": float(probability.mean()),
        "ece_10": expected_calibration_error(labels, probability),
        "brier": float(brier_score_loss(labels, probability)),
        "auc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
    }


def contiguous_threshold_ranges(rows: list[dict], step: float = 0.005) -> list[list[float]]:
    values = [float(row["threshold"]) for row in rows if row["both_performance_pass"]]
    if not values:
        return []
    ranges = []
    start = previous = values[0]
    for value in values[1:]:
        if value - previous > step + 1e-9:
            ranges.append([start, previous])
            start = value
        previous = value
    ranges.append([start, previous])
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-features", default="artifacts/confidence/mmlu_logit/qwen_1_5b.jsonl")
    parser.add_argument("--selection-input", default="artifacts/data/mmlu_validation_200.jsonl")
    parser.add_argument("--independent-features", default="artifacts/confidence/mmlu_logit_independent/qwen_1_5b.jsonl")
    parser.add_argument("--independent-input", default="artifacts/data/mmlu_test_independent_500.jsonl")
    parser.add_argument("--selection-upper", default="artifacts/model_screening/mmlu_logit_4bit_b8_200/qwen_7b/predictions.jsonl")
    parser.add_argument("--independent-upper", default="artifacts/model_screening/mmlu_logit_4bit_b8_500/qwen_7b/predictions.jsonl")
    parser.add_argument("--latency-ratio", type=float, default=0.24704482170737938)
    parser.add_argument("--output", default="paper/data/mmlu_calibration_drift_analysis.json")
    args = parser.parse_args()

    selection = read_jsonl(args.selection_features)
    independent = read_jsonl(args.independent_features)
    x_selection = np.asarray([[float(row[name]) for name in FEATURES] for row in selection], dtype=np.float32)
    y_selection = np.asarray([bool(row["correct"]) for row in selection])
    x_independent = np.asarray([[float(row[name]) for name in FEATURES] for row in independent], dtype=np.float32)
    y_independent = np.asarray([bool(row["correct"]) for row in independent])
    selection_probability = cross_validated_probability(x_selection, y_selection)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))
    model.fit(x_selection, y_selection)
    independent_probability = model.predict_proba(x_independent)[:, 1]

    selection_upper = {row["id"]: bool(row["correct"]) for row in read_jsonl(args.selection_upper)}
    independent_upper = {row["id"]: bool(row["correct"]) for row in read_jsonl(args.independent_upper)}
    upper_selection = np.asarray([selection_upper[row["id"]] for row in selection])
    upper_independent = np.asarray([independent_upper[row["id"]] for row in independent])
    cut = 250

    threshold_rows = []
    for threshold in np.linspace(0.40, 0.75, 71):
        cert = evaluate(independent_probability[:cut], y_independent[:cut], upper_independent[:cut], float(threshold), args.latency_ratio)
        final = evaluate(independent_probability[cut:], y_independent[cut:], upper_independent[cut:], float(threshold), args.latency_ratio)
        threshold_rows.append({"threshold": float(threshold), "certification": cert, "final_test": final, "both_performance_pass": cert["performance_gate_pass"] and final["performance_gate_pass"]})

    metadata = {row["id"]: row["task_metadata"]["subject"] for row in read_jsonl(args.independent_input)}
    groups = {}
    for group in ("STEM", "humanities", "social_sciences", "other"):
        indices = np.asarray([index for index, row in enumerate(independent) if subject_group(metadata[row["id"]]) == group])
        if len(indices):
            groups[group] = evaluate(independent_probability[indices], y_independent[indices], upper_independent[indices], 0.545, args.latency_ratio)

    robust = [row for row in threshold_rows if row["both_performance_pass"]]
    payload = {
        "protocol": "post-hoc drift diagnosis only; thresholds in this report are not confirmatory",
        "fixed_threshold": 0.545,
        "calibration": {
            "selection_oof": calibration_metrics(y_selection, selection_probability),
            "certification": calibration_metrics(y_independent[:cut], independent_probability[:cut]),
            "final_test": calibration_metrics(y_independent[cut:], independent_probability[cut:]),
        },
        "fixed_policy": {
            "selection_oof": evaluate(selection_probability, y_selection, upper_selection, 0.545, args.latency_ratio),
            "certification": evaluate(independent_probability[:cut], y_independent[:cut], upper_independent[:cut], 0.545, args.latency_ratio),
            "final_test": evaluate(independent_probability[cut:], y_independent[cut:], upper_independent[cut:], 0.545, args.latency_ratio),
        },
        "independent_subject_groups_at_fixed_threshold": groups,
        "threshold_sensitivity": threshold_rows,
        "posthoc_robust_threshold_ranges": contiguous_threshold_ranges(robust),
        "decision": "Use any robust interval only as a preregistered candidate on new data; never relabel the failed certification result.",
    }
    write_json(args.output, payload)
    print(f"robust_ranges={payload['posthoc_robust_threshold_ranges']} -> {args.output}")


if __name__ == "__main__":
    main()
