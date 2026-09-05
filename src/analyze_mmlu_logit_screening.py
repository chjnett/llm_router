"""Analyze one-forward MMLU option-logit screening and oracle cascades."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, write_json


PAIRS = (
    ("M0", "qwen_1_5b", "qwen_7b"),
    ("M1", "smollm2_360m", "smollm2_1_7b"),
    ("M2", "smollm2_1_7b", "qwen_7b"),
)


def _oracle_accuracy(lower_path: str, upper_path: str) -> float:
    lower = read_jsonl(lower_path)
    upper = {row["id"]: row for row in read_jsonl(upper_path)}
    return sum(bool(row["correct"]) or bool(upper[row["id"]]["correct"]) for row in lower) / len(lower)


def analyze(payload: dict) -> dict:
    indexed = {(row["model"], row["batch"], row["limit"]): row for row in payload["runs"]}
    rows = []
    for pair_id, lower_name, upper_name in PAIRS:
        for batch, limit in ((8, 200), (1, 50)):
            lower_row = indexed[(lower_name, batch, limit)]
            upper_row = indexed[(upper_name, batch, limit)]
            lower = lower_row["metrics"]
            upper = upper_row["metrics"]
            latency_ratio = lower["latency_ms_p50"] / upper["latency_ms_p50"]
            energy_ratio = lower["gross_energy_joules_per_item"] / upper["gross_energy_joules_per_item"]
            oracle_latency = latency_ratio + (1.0 - lower["accuracy"])
            oracle_energy = energy_ratio + (1.0 - lower["accuracy"])
            oracle_accuracy = _oracle_accuracy(
                str(Path(lower_row["report"]).with_name("predictions.jsonl")),
                str(Path(upper_row["report"]).with_name("predictions.jsonl")),
            )
            checks = {
                "accuracy_gap": upper["accuracy"] - lower["accuracy"] >= 0.05,
                "latency_ratio": latency_ratio < 0.50,
                "parse_success": lower["parse_success_rate"] >= 0.98,
                "token_limit": lower["token_limit_rate"] <= 0.05,
                "vram": max(lower["peak_vram_reserved_gb"], upper["peak_vram_reserved_gb"]) <= 22.0,
            }
            rows.append({
                "pair": pair_id, "lower": lower_name, "upper": upper_name,
                "batch_size": batch, "sample_count": limit,
                "lower_accuracy": lower["accuracy"], "upper_accuracy": upper["accuracy"],
                "oracle_system_accuracy": oracle_accuracy,
                "latency_ratio": latency_ratio, "energy_ratio": energy_ratio,
                "oracle_normalized_latency": oracle_latency, "oracle_normalized_energy": oracle_energy,
                "checks": checks, "screening_pass": all(checks.values()),
                "practical_oracle_latency_pass": oracle_latency <= 0.90 and oracle_accuracy >= 0.95 * upper["accuracy"],
                "practical_oracle_energy_pass": oracle_energy <= 0.90 and oracle_accuracy >= 0.95 * upper["accuracy"],
            })
    candidates = [row for row in rows if row["screening_pass"] and row["practical_oracle_latency_pass"]]
    return {
        "completed_runs": payload.get("completed"), "failed_runs": len(payload.get("failures", [])),
        "conditions": rows, "latency_candidates": candidates,
        "decision": (
            "M0 batch 8 and M2 batch 8/1 are practical oracle candidates. "
            "Collect lower option-probability features and run an OOF confidence-policy gate before any full-domain expansion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/results/mmlu_logit_screening.json")
    parser.add_argument("--output", default="paper/data/mmlu_logit_screening_analysis.json")
    args = parser.parse_args()
    with Path(args.input).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result = analyze(payload)
    write_json(args.output, result)
    print(f"candidates={len(result['latency_candidates'])} -> {args.output}")


if __name__ == "__main__":
    main()
