"""Apply the fixed cost and oracle gates to the Korean model-pair pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/results/kmmlu_korean_pair_screening.json")
    parser.add_argument("--output", default="paper/data/kmmlu_korean_pair_screening_analysis.json")
    args = parser.parse_args()
    with Path(args.input).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    indexed = {row["model"]: row for row in payload["runs"]}
    lower_row = indexed["hyperclovax_0_5b"]
    upper_row = indexed["qwen_7b"]
    lower = lower_row["metrics"]
    upper = upper_row["metrics"]
    lower_predictions = read_jsonl(Path(lower_row["report"]).with_name("predictions.jsonl"))
    upper_predictions = {
        row["id"]: row for row in read_jsonl(Path(upper_row["report"]).with_name("predictions.jsonl"))
    }
    oracle_accuracy = sum(
        bool(row["correct"]) or bool(upper_predictions[row["id"]]["correct"])
        for row in lower_predictions
    ) / len(lower_predictions)
    latency_ratio = lower["latency_ms_p50"] / upper["latency_ms_p50"]
    oracle_normalized_latency = latency_ratio + 1.0 - lower["accuracy"]
    checks = {
        "upper_accuracy_advantage_5pp": upper["accuracy"] - lower["accuracy"] >= 0.05,
        "latency_ratio_below_0_5": latency_ratio < 0.5,
        "oracle_quality_95pct": oracle_accuracy >= 0.95 * upper["accuracy"],
        "oracle_latency_10pct_reduction": oracle_normalized_latency <= 0.9,
    }
    passed = all(checks.values())
    result = {
        "protocol": f"exploratory {len(lower_predictions)}-item KMMLU compatibility screen",
        "confirmation_required": True,
        "lower": "hyperclovax_0_5b",
        "upper": "qwen_7b",
        "sample_count": len(lower_predictions),
        "lower_accuracy": lower["accuracy"],
        "upper_accuracy": upper["accuracy"],
        "latency_ratio": latency_ratio,
        "oracle_accuracy": oracle_accuracy,
        "oracle_normalized_latency": oracle_normalized_latency,
        "checks": checks,
        "continue_to_confidence": passed,
    }
    write_json(args.output, result)
    print(f"continue_to_confidence={passed} -> {args.output}")


if __name__ == "__main__":
    main()
