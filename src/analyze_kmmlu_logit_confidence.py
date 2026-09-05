"""Run the exploratory OOF confidence gate for the viable KMMLU pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_mmlu_logit_confidence import analyze_pair
from .common import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening", default="artifacts/results/kmmlu_logit_screening.json")
    parser.add_argument(
        "--features", default="artifacts/confidence/kmmlu_logit/qwen_1_5b.jsonl"
    )
    parser.add_argument("--output", default="paper/data/kmmlu_logit_confidence_analysis.json")
    args = parser.parse_args()

    with Path(args.screening).open("r", encoding="utf-8") as handle:
        screening = json.load(handle)
    indexed = {(row["model"], row["batch"], row["limit"]): row for row in screening["runs"]}
    lower = indexed[("qwen_1_5b", 8, 200)]
    upper = indexed[("qwen_7b", 8, 200)]
    latency_ratio = lower["metrics"]["latency_ms_p50"] / upper["metrics"]["latency_ms_p50"]
    upper_predictions = str(Path(upper["report"]).with_name("predictions.jsonl"))
    result = analyze_pair(args.features, upper_predictions, latency_ratio)
    payload = {
        "dataset": "HAERAE-HUB/KMMLU",
        "protocol": "exploratory 5-fold OOF logistic confidence; threshold selected on the same OOF predictions",
        "confirmation_required": True,
        "pair": "M0",
        "result": result,
    }
    write_json(args.output, payload)
    print(
        f"auc={result['routing_auc']:.3f}, "
        f"practical={result['practical_10_percent_reduction_achieved']} -> {args.output}"
    )


if __name__ == "__main__":
    main()
