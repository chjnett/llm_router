"""Analyze short-answer MMLU model-pair screening gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import write_json


PAIRS = (
    ("M0", "qwen_1_5b", "qwen_7b"),
    ("M1", "smollm2_360m", "smollm2_1_7b"),
    ("M2", "smollm2_1_7b", "qwen_7b"),
)


def analyze(payload: dict) -> dict:
    indexed = {(row["model"], row["batch"], row["limit"]): row for row in payload["runs"]}
    rows = []
    for pair_id, lower_name, upper_name in PAIRS:
        for batch, limit in ((8, 200), (1, 50)):
            lower = indexed[(lower_name, batch, limit)]["metrics"]
            upper = indexed[(upper_name, batch, limit)]["metrics"]
            latency_ratio = lower["latency_ms_p50"] / upper["latency_ms_p50"]
            energy_ratio = lower["gross_energy_joules_per_item"] / upper["gross_energy_joules_per_item"]
            accuracy_gap = upper["accuracy"] - lower["accuracy"]
            checks = {
                "accuracy_gap": accuracy_gap >= 0.05,
                "latency_ratio": latency_ratio < 0.50,
                "parse_success": lower["parse_success_rate"] >= 0.98 and upper["parse_success_rate"] >= 0.98,
                "token_limit": lower["token_limit_rate"] <= 0.05,
                "vram": max(lower["peak_vram_reserved_gb"], upper["peak_vram_reserved_gb"]) <= 22.0,
            }
            rows.append({
                "pair": pair_id,
                "lower": lower_name,
                "upper": upper_name,
                "batch_size": batch,
                "sample_count": limit,
                "lower_accuracy": lower["accuracy"],
                "upper_accuracy": upper["accuracy"],
                "accuracy_gap_pp": 100.0 * accuracy_gap,
                "latency_ratio": latency_ratio,
                "energy_ratio": energy_ratio,
                "lower_token_limit_rate": lower["token_limit_rate"],
                "oracle_latency_margin": lower["accuracy"] - latency_ratio,
                "oracle_energy_margin": lower["accuracy"] - energy_ratio,
                "checks": checks,
                "screening_pass": all(checks.values()),
            })
    return {
        "completed_runs": payload.get("completed"),
        "failed_runs": len(payload.get("failures", [])),
        "conditions": rows,
        "screening_passes": [row for row in rows if row["screening_pass"]],
        "decision": (
            "Short answers remove most of the Qwen latency inversion, but no pair reaches the predeclared 0.50 ratio "
            "and no pair is oracle-feasible. Test Qwen1.5B FP16 versus its 4-bit result before abandoning local wall-clock routing."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/results/mmlu_short_answer_screening.json")
    parser.add_argument("--output", default="paper/data/mmlu_short_answer_screening_analysis.json")
    args = parser.parse_args()
    with Path(args.input).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result = analyze(payload)
    write_json(args.output, result)
    print(f"passes={len(result['screening_passes'])} -> {args.output}")


if __name__ == "__main__":
    main()
