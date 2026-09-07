"""Certify the frozen MBPP hidden-state pre-router on independent splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .analyze_mbpp_hidden_pretriage import probe_policy_metrics
from .common import read_jsonl, write_json


def matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([
        row["hidden"] + [row["next_entropy"], row["next_max_probability"], row["next_margin"], row["prompt_tokens"]]
        for row in rows
    ], dtype=np.float32)


def evaluate_split(probability, lower, upper, indices, threshold, generation_ratio, probe_ratio) -> dict:
    selected_probability = probability[indices]
    selected_lower = lower[indices]
    selected_upper = upper[indices]
    metrics = probe_policy_metrics(
        selected_probability, threshold, selected_lower, selected_upper, generation_ratio, probe_ratio
    )
    metrics["items"] = int(len(indices))
    metrics["oof_or_test_roc_auc"] = float(roc_auc_score(selected_lower, selected_probability))
    metrics["oof_or_test_average_precision"] = float(average_precision_score(selected_lower, selected_probability))
    metrics["quality_floor_pass"] = metrics["quality_retention"] >= 0.95
    metrics["latency_target_pass"] = metrics["normalized_latency"] <= 0.90
    metrics["joint_gate_pass"] = metrics["quality_floor_pass"] and metrics["latency_target_pass"]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-features", default="artifacts/mbpp_hidden_probe/qwen_1_5b/features.jsonl")
    parser.add_argument("--selection-result", default="paper/data/mbpp_screening_analysis.json")
    parser.add_argument("--independent-input", default="artifacts/data/mbpp_independent_200.jsonl")
    parser.add_argument("--independent-features", default="artifacts/mbpp_hidden_probe_independent/qwen_1_5b/features.jsonl")
    parser.add_argument("--independent-probe-report", default="artifacts/mbpp_hidden_probe_independent/qwen_1_5b/report.json")
    parser.add_argument("--independent-result", default="paper/data/mbpp_independent_model_analysis.json")
    parser.add_argument("--output", default="paper/data/mbpp_hidden_pretriage_certification.json")
    parser.add_argument("--threshold", type=float, default=0.505)
    parser.add_argument("--seed", type=int, default=2034)
    args = parser.parse_args()

    selection_features = read_jsonl(args.selection_features)
    independent_rows = read_jsonl(args.independent_input)
    independent_by_id = {row["id"]: row for row in read_jsonl(args.independent_features)}
    independent_features = [independent_by_id[row["id"]] for row in independent_rows]
    with Path(args.selection_result).open("r", encoding="utf-8") as handle:
        selection_result = json.load(handle)
    with Path(args.independent_result).open("r", encoding="utf-8") as handle:
        independent_result = json.load(handle)
    with Path(args.independent_probe_report).open("r", encoding="utf-8") as handle:
        probe_report = json.load(handle)

    train_x = matrix(selection_features)
    train_y = np.asarray(selection_result["lower"]["evaluation"]["correct"], dtype=bool)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced", C=0.01, random_state=args.seed),
    ).fit(train_x, train_y.astype(int))
    probability = model.predict_proba(matrix(independent_features))[:, 1]
    lower = np.asarray(independent_result["lower"]["evaluation"]["correct"], dtype=bool)
    upper = np.asarray(independent_result["upper"]["evaluation"]["correct"], dtype=bool)
    generation_ratio = float(independent_result["latency_ratio"])
    upper_p50 = float(independent_result["upper"]["metrics"]["latency_ms_p50"])
    probe_ratio = float(probe_report["latency_ms_p50"]) / upper_p50
    splits = np.asarray([row["split"] for row in independent_rows])
    certification = evaluate_split(
        probability, lower, upper, np.flatnonzero(splits == "certification"), args.threshold,
        generation_ratio, probe_ratio,
    )
    final = evaluate_split(
        probability, lower, upper, np.flatnonzero(splits == "final"), args.threshold,
        generation_ratio, probe_ratio,
    )
    payload = {
        "protocol": "selection-50 fit; frozen Qwen1.5B hidden/logit schema, LR C=0.01 seed=2034, threshold=0.505; independent MBPP test 100+100",
        "cost_assumption": "accepted Lower requests reuse probe KV cache; rejected requests pay probe plus Upper",
        "selection_items": len(selection_features),
        "independent_items": len(independent_rows),
        "generation_latency_ratio": generation_ratio,
        "probe_latency_ratio": probe_ratio,
        "threshold": args.threshold,
        "certification": certification,
        "final": final,
        "confirmed": certification["joint_gate_pass"] and final["joint_gate_pass"],
        "warning": "Projected component latency under KV-cache reuse; end-to-end implementation remains required if confirmed.",
    }
    write_json(args.output, payload)
    print(f"confirmed={payload['confirmed']} -> {args.output}")


if __name__ == "__main__":
    main()
