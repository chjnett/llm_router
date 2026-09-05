"""Analyze the Qwen concise token-budget sweep against fixed baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import write_json


def _index_baselines(payload: dict, model: str) -> dict[tuple[int, int], dict]:
    return {
        (row["batch"], row["limit"]): row
        for row in payload["runs"]
        if row["model"] == model and row["mode"] == "task"
    }


def analyze(sweep: dict, baselines: dict) -> dict:
    lower = _index_baselines(baselines, "qwen_1_5b")
    upper = _index_baselines(baselines, "qwen_7b")
    rows = []
    for run in sweep["runs"]:
        key = (run["batch"], run["limit"])
        metrics = run["metrics"]
        lower_metrics = lower[key]["metrics"]
        upper_metrics = upper[key]["metrics"]
        accuracy = metrics["accuracy"]
        latency_vs_lower = metrics["latency_ms_p50"] / lower_metrics["latency_ms_p50"]
        latency_vs_upper = metrics["latency_ms_p50"] / upper_metrics["latency_ms_p50"]
        energy_vs_lower = metrics["gross_energy_joules_per_item"] / lower_metrics["gross_energy_joules_per_item"]
        energy_vs_upper = metrics["gross_energy_joules_per_item"] / upper_metrics["gross_energy_joules_per_item"]
        quality_retention = accuracy / lower_metrics["accuracy"]
        compression_gate = (
            quality_retention >= 0.95
            and latency_vs_lower <= 0.90
            and metrics["token_limit_rate"] <= 0.05
        )
        rows.append({
            "budget": run["budget"],
            "batch_size": run["batch"],
            "sample_count": run["limit"],
            "accuracy": accuracy,
            "accuracy_change_vs_full_lower_pp": 100.0 * (accuracy - lower_metrics["accuracy"]),
            "quality_retention_vs_full_lower": quality_retention,
            "token_limit_rate": metrics["token_limit_rate"],
            "latency_ms_p50": metrics["latency_ms_p50"],
            "latency_ratio_vs_full_lower": latency_vs_lower,
            "latency_ratio_vs_upper": latency_vs_upper,
            "energy_joules_per_item": metrics["gross_energy_joules_per_item"],
            "energy_ratio_vs_full_lower": energy_vs_lower,
            "energy_ratio_vs_upper": energy_vs_upper,
            "compression_gate_pass": compression_gate,
            "oracle_latency_saving_margin_vs_upper": accuracy - latency_vs_upper,
            "oracle_energy_saving_margin_vs_upper": accuracy - energy_vs_upper,
            "oracle_latency_feasible": accuracy > latency_vs_upper,
            "oracle_energy_feasible": accuracy > energy_vs_upper,
        })
    rows.sort(key=lambda row: (row["batch_size"], row["budget"]))
    return {
        "protocol": {
            "compression_gate": "retain >=95% of full Lower accuracy, reduce Lower p50 latency >=10%, token-limit rate <=5%",
            "oracle_gate": "Lower is always called; a perfect selector accepts every correct Lower output and calls Upper otherwise",
        },
        "completed_runs": sweep.get("completed"),
        "failed_runs": len(sweep.get("failures", [])),
        "conditions": rows,
        "compression_candidates": [row for row in rows if row["compression_gate_pass"]],
        "oracle_latency_candidates": [row for row in rows if row["oracle_latency_feasible"]],
        "oracle_energy_candidates": [row for row in rows if row["oracle_energy_feasible"]],
        "decision": (
            "Use 256 tokens as the minimum stable Qwen Lower budget for online inference; "
            "do not continue a same-domain wall-clock cascade because no budget is oracle-feasible versus Qwen7B. "
            "Test a short-answer domain next, where generation length is controlled across models."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/results/token_budget_sweep.json")
    parser.add_argument("--baselines", default="artifacts/results/output_length_ablation.json")
    parser.add_argument("--output", default="paper/data/token_budget_sweep_analysis.json")
    args = parser.parse_args()
    with Path(args.input).open("r", encoding="utf-8") as handle:
        sweep = json.load(handle)
    with Path(args.baselines).open("r", encoding="utf-8") as handle:
        baselines = json.load(handle)
    result = analyze(sweep, baselines)
    write_json(args.output, result)
    print(
        f"compression={len(result['compression_candidates'])} "
        f"oracle_latency={len(result['oracle_latency_candidates'])} "
        f"oracle_energy={len(result['oracle_energy_candidates'])} -> {args.output}"
    )


if __name__ == "__main__":
    main()
