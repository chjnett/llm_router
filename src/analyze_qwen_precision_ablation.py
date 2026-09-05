"""Compare Qwen1.5B FP16 runs with 4-bit Lower and Qwen7B baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import write_json


def analyze(precision: dict, budgets: dict, mmlu: dict, output_ablation: dict) -> dict:
    budget_index = {(row["batch"], row["limit"]): row["metrics"] for row in budgets["runs"] if row["budget"] == 256}
    mmlu_index = {(row["model"], row["batch"], row["limit"]): row["metrics"] for row in mmlu["runs"]}
    upper_math = {
        (row["batch"], row["limit"]): row["metrics"]
        for row in output_ablation["runs"] if row["model"] == "qwen_7b" and row["mode"] == "task"
    }
    rows = []
    for run in precision["runs"]:
        key = (run["batch"], run["limit"])
        fp16 = run["metrics"]
        if run["domain"] == "mmlu":
            lower_4bit = mmlu_index[("qwen_1_5b", *key)]
            upper = mmlu_index[("qwen_7b", *key)]
        else:
            lower_4bit = budget_index[key]
            upper = upper_math[key]
        latency_vs_4bit = fp16["latency_ms_p50"] / lower_4bit["latency_ms_p50"]
        latency_vs_upper = fp16["latency_ms_p50"] / upper["latency_ms_p50"]
        energy_vs_upper = fp16["gross_energy_joules_per_item"] / upper["gross_energy_joules_per_item"]
        checks = {
            "latency_ratio": latency_vs_upper < 0.50,
            "parse_success": fp16["parse_success_rate"] >= 0.98,
            "token_limit": fp16["token_limit_rate"] <= 0.05,
            "vram": fp16["peak_vram_reserved_gb"] <= 22.0,
        }
        rows.append({
            "domain": run["domain"], "batch_size": run["batch"], "sample_count": run["limit"],
            "accuracy": fp16["accuracy"], "parse_success_rate": fp16["parse_success_rate"],
            "latency_ms_p50": fp16["latency_ms_p50"], "latency_ratio_vs_4bit": latency_vs_4bit,
            "latency_ratio_vs_upper": latency_vs_upper, "energy_ratio_vs_upper": energy_vs_upper,
            "oracle_latency_margin": fp16["accuracy"] - latency_vs_upper,
            "oracle_energy_margin": fp16["accuracy"] - energy_vs_upper,
            "checks": checks, "screening_pass": all(checks.values()),
        })
    return {
        "completed_runs": precision.get("completed"), "failed_runs": len(precision.get("failures", [])),
        "conditions": rows, "screening_passes": [row for row in rows if row["screening_pass"]],
        "decision": (
            "FP16 substantially improves Qwen1.5B but does not make a generated-answer cascade latency-feasible. "
            "Stop generated-answer wall-clock tuning and test one-forward option-logit scoring on MMLU."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", default="artifacts/results/qwen_precision_ablation.json")
    parser.add_argument("--budgets", default="artifacts/results/token_budget_sweep.json")
    parser.add_argument("--mmlu", default="artifacts/results/mmlu_short_answer_screening.json")
    parser.add_argument("--output-ablation", default="artifacts/results/output_length_ablation.json")
    parser.add_argument("--output", default="paper/data/qwen_precision_ablation_analysis.json")
    args = parser.parse_args()
    payloads = []
    for path in (args.precision, args.budgets, args.mmlu, args.output_ablation):
        with Path(path).open("r", encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    result = analyze(*payloads)
    write_json(args.output, result)
    print(f"passes={len(result['screening_passes'])} -> {args.output}")


if __name__ == "__main__":
    main()
