"""Check whether certification-only recalibration justifies a fresh reserve run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .analyze_mbpp_hidden_pretriage import probe_policy_metrics
from .certify_mbpp_hidden_pretriage import matrix
from .common import read_jsonl, write_json


def threshold_diagnostics(probability, lower, upper, generation_ratio, probe_ratio, quality_floor=0.95, latency_target=0.90):
    thresholds = sorted(set(np.linspace(0.0, 1.0, 201).tolist() + probability.tolist()))
    curve = [
        probe_policy_metrics(probability, threshold, lower, upper, generation_ratio, probe_ratio)
        for threshold in thresholds
    ]
    feasible = [
        point for point in curve
        if point["quality_retention"] >= quality_floor
        and point["normalized_latency"] <= latency_target
    ]
    quality_safe = [point for point in curve if point["quality_retention"] >= quality_floor]
    latency_safe = [point for point in curve if point["normalized_latency"] <= latency_target]
    return {
        "selected": min(feasible, key=lambda point: (point["normalized_latency"], -point["accuracy"])) if feasible else None,
        "best_quality_constrained": min(quality_safe, key=lambda point: (point["normalized_latency"], -point["accuracy"])),
        "best_latency_constrained": max(latency_safe, key=lambda point: (point["accuracy"], -point["latency_reduction"])) if latency_safe else None,
    }


def select_threshold(probability, lower, upper, generation_ratio, probe_ratio, quality_floor=0.95, latency_target=0.90):
    return threshold_diagnostics(
        probability, lower, upper, generation_ratio, probe_ratio, quality_floor, latency_target
    )["selected"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-features", default="artifacts/mbpp_hidden_probe/qwen_1_5b/features.jsonl")
    parser.add_argument("--selection-result", default="paper/data/mbpp_screening_analysis.json")
    parser.add_argument("--independent-input", default="artifacts/data/mbpp_independent_200.jsonl")
    parser.add_argument("--independent-features", default="artifacts/mbpp_hidden_probe_independent/qwen_1_5b/features.jsonl")
    parser.add_argument("--independent-result", default="paper/data/mbpp_independent_model_analysis.json")
    parser.add_argument("--certification", default="paper/data/mbpp_hidden_pretriage_certification.json")
    parser.add_argument("--output", default="paper/data/mbpp_probe_recalibration_screen.json")
    args = parser.parse_args()

    selection_features = read_jsonl(args.selection_features)
    with Path(args.selection_result).open("r", encoding="utf-8") as handle:
        selection_result = json.load(handle)
    rows = read_jsonl(args.independent_input)
    by_id = {row["id"]: row for row in read_jsonl(args.independent_features)}
    certification_indices = np.asarray([index for index, row in enumerate(rows) if row["split"] == "certification"])
    with Path(args.independent_result).open("r", encoding="utf-8") as handle:
        independent = json.load(handle)
    with Path(args.certification).open("r", encoding="utf-8") as handle:
        prior = json.load(handle)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced", C=0.01, random_state=2034),
    ).fit(
        matrix(selection_features),
        np.asarray(selection_result["lower"]["evaluation"]["correct"], dtype=int),
    )
    certification_features = [by_id[rows[index]["id"]] for index in certification_indices]
    probability = model.predict_proba(matrix(certification_features))[:, 1]
    lower = np.asarray(independent["lower"]["evaluation"]["correct"], dtype=bool)[certification_indices]
    upper = np.asarray(independent["upper"]["evaluation"]["correct"], dtype=bool)[certification_indices]
    diagnostics = threshold_diagnostics(
        probability, lower, upper,
        float(prior["generation_latency_ratio"]), float(prior["probe_latency_ratio"]),
    )
    payload = {
        "protocol": "post-failure diagnostic; threshold selected on certification 100 only; final 100 not used for selection",
        "fixed_model": "Qwen1.5B last hidden/logit, LR C=0.01, seed=2034 trained on selection 50",
        "certification_items": len(certification_indices),
        "selected_policy": diagnostics["selected"],
        "best_quality_constrained_policy": diagnostics["best_quality_constrained"],
        "best_latency_constrained_policy": diagnostics["best_latency_constrained"],
        "continue_to_fresh_reserve_50": diagnostics["selected"] is not None,
        "warning": "If continued, only the untouched remaining MBPP test reserve may be used for evaluation.",
    }
    write_json(args.output, payload)
    print(f"continue_to_fresh_reserve_50={payload['continue_to_fresh_reserve_50']} -> {args.output}")


if __name__ == "__main__":
    main()
