"""Summarize whether short Lower outputs can theoretically break even."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import write_json


def analyze(payload: dict) -> dict:
    runs = payload["runs"]
    upper = {
        (row["batch"], row["limit"]): row
        for row in runs
        if row["model"] == "qwen_7b" and row["mode"] == "task"
    }
    rows = []
    for row in runs:
        if row["model"] == "qwen_7b" or row["mode"] == "task":
            continue
        baseline = upper[(row["batch"], row["limit"])]
        metrics = row["metrics"]
        upper_metrics = baseline["metrics"]
        latency_ratio = metrics["latency_ms_p50"] / upper_metrics["latency_ms_p50"]
        lower_accuracy = metrics["accuracy"]
        energy = metrics.get("gross_energy_joules_per_item")
        upper_energy = upper_metrics.get("gross_energy_joules_per_item")
        energy_ratio = energy / upper_energy if energy is not None and upper_energy else None
        latency_margin = lower_accuracy - latency_ratio
        energy_margin = lower_accuracy - energy_ratio if energy_ratio is not None else None
        rows.append({
            "model": row["model"],
            "mode": row["mode"],
            "batch_size": row["batch"],
            "sample_count": row["limit"],
            "accuracy": lower_accuracy,
            "upper_accuracy": upper_metrics["accuracy"],
            "tokens_mean": metrics["generated_tokens_mean"],
            "latency_ms_p50": metrics["latency_ms_p50"],
            "latency_ratio_vs_upper": latency_ratio,
            "energy_joules_per_item": energy,
            "energy_ratio_vs_upper": energy_ratio,
            "token_limit_rate": metrics["token_limit_rate"],
            "oracle_latency_saving_margin": latency_margin,
            "oracle_energy_saving_margin": energy_margin,
            "latency_feasible_with_perfect_selector": latency_margin > 0 and metrics["token_limit_rate"] <= 0.05,
            "energy_feasible_with_perfect_selector": energy_margin is not None and energy_margin > 0 and metrics["token_limit_rate"] <= 0.05,
        })
    candidates = [row for row in rows if row["latency_feasible_with_perfect_selector"]]
    candidates.sort(key=lambda row: row["oracle_latency_saving_margin"], reverse=True)
    return {
        "protocol": "Lower is always called; a perfect selector accepts exactly correct Lower outputs and calls Upper otherwise",
        "warning": "Oracle feasibility is an upper bound, not an achieved routing result",
        "completed_runs": payload.get("completed"),
        "failed_runs": len(payload.get("failures", [])),
        "conditions": rows,
        "latency_candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/results/output_length_ablation.json")
    parser.add_argument("--output", default="paper/data/output_length_ablation_analysis.json")
    args = parser.parse_args()
    with Path(args.input).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result = analyze(payload)
    write_json(args.output, result)
    print(f"latency_candidates={len(result['latency_candidates'])} -> {args.output}")


if __name__ == "__main__":
    main()

