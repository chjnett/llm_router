"""Evaluate exploratory OOF confidence routing for HyperCLOVA/Qwen on KMMLU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_mmlu_logit_confidence import analyze_pair
from .common import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening", default="artifacts/results/kmmlu_korean_pair_screening_200.json")
    parser.add_argument(
        "--features", default="artifacts/confidence/kmmlu_hyperclovax/hyperclovax_0_5b.jsonl"
    )
    parser.add_argument("--output", default="paper/data/kmmlu_korean_pair_confidence_analysis.json")
    args = parser.parse_args()
    with Path(args.screening).open("r", encoding="utf-8") as handle:
        screening = json.load(handle)
    indexed = {row["model"]: row for row in screening["runs"]}
    lower = indexed["hyperclovax_0_5b"]
    upper = indexed["qwen_7b"]
    ratio = lower["metrics"]["latency_ms_p50"] / upper["metrics"]["latency_ms_p50"]
    upper_predictions = str(Path(upper["report"]).with_name("predictions.jsonl"))
    result = analyze_pair(args.features, upper_predictions, ratio)
    payload = {
        "dataset": "HAERAE-HUB/KMMLU",
        "pair": "HyperCLOVAX-SEED-0.5B / Qwen2.5-7B",
        "protocol": "exploratory 5-fold OOF logistic confidence; threshold selected on the same OOF predictions",
        "confirmation_required": True,
        "result": result,
    }
    write_json(args.output, payload)
    print(
        f"auc={result['routing_auc']:.3f}, "
        f"practical={result['practical_10_percent_reduction_achieved']} -> {args.output}"
    )


if __name__ == "__main__":
    main()
