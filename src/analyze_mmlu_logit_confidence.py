"""OOF confidence-policy gate for one-forward MMLU Lower models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .analyze_answer_only_confidence import cross_validated_probability, select_policy
from .common import read_jsonl, write_json


FEATURES = ("prob_a", "prob_b", "prob_c", "prob_d", "max_probability", "probability_margin", "normalized_entropy", "logit_spread")


def analyze_pair(features_path: str, upper_path: str, latency_ratio: float) -> dict:
    feature_rows = read_jsonl(features_path)
    upper_by_id = {row["id"]: row for row in read_jsonl(upper_path)}
    feature_rows = [row for row in feature_rows if row["id"] in upper_by_id]
    x = np.asarray([[float(row[name]) for name in FEATURES] for row in feature_rows], dtype=np.float32)
    lower = np.asarray([bool(row["correct"]) for row in feature_rows])
    upper = np.asarray([bool(upper_by_id[row["id"]]["correct"]) for row in feature_rows])
    probability = cross_validated_probability(x, lower)
    selected, curve = select_policy(probability, lower, upper, latency_ratio)
    practical = [point for point in curve if point["quality_retention"] >= 0.95 and point["normalized_latency"] <= 0.90]
    return {
        "rows": len(feature_rows), "features": list(FEATURES),
        "lower_accuracy": float(lower.mean()), "upper_accuracy": float(upper.mean()),
        "lower_latency_ratio": latency_ratio,
        "routing_auc": float(roc_auc_score(lower, probability)),
        "average_precision": float(average_precision_score(lower, probability)),
        "selected_quality_floor_95": selected,
        "practical_10_percent_reduction_achieved": bool(practical),
        "best_practical_point": min(practical, key=lambda row: row["normalized_latency"]) if practical else None,
        "curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening", default="artifacts/results/mmlu_logit_screening.json")
    parser.add_argument("--features-dir", default="artifacts/confidence/mmlu_logit")
    parser.add_argument("--output", default="paper/data/mmlu_logit_confidence_analysis.json")
    args = parser.parse_args()
    with Path(args.screening).open("r", encoding="utf-8") as handle:
        screening = json.load(handle)
    indexed = {(row["model"], row["batch"], row["limit"]): row for row in screening["runs"]}
    upper = indexed[("qwen_7b", 8, 200)]
    upper_predictions = str(Path(upper["report"]).with_name("predictions.jsonl"))
    results = {}
    for pair, lower_name in (("M0", "qwen_1_5b"), ("M2", "smollm2_1_7b")):
        lower = indexed[(lower_name, 8, 200)]
        ratio = lower["metrics"]["latency_ms_p50"] / upper["metrics"]["latency_ms_p50"]
        results[pair] = analyze_pair(
            str(Path(args.features_dir) / f"{lower_name}.jsonl"), upper_predictions, ratio
        )
    payload = {
        "protocol": "exploratory 5-fold OOF logistic confidence; threshold selected on the same OOF predictions",
        "confirmation_required": True, "pairs": results,
    }
    write_json(args.output, payload)
    print(" ".join(f"{pair}:auc={row['routing_auc']:.3f},practical={row['practical_10_percent_reduction_achieved']}" for pair, row in results.items()))


if __name__ == "__main__":
    main()
