"""Compare low-token verifier variants on a selection-only policy frontier."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .analyze_verifier_performance import route_items
from .common import load_config, read_jsonl, write_json
from .run_low_cost_verifier import load_target
from .run_risk_bound_calibration import fit_gsm8k_confidence_model


def output_diagnostics(root: Path, name: str, split: str, lower_by_id: dict[str, dict]) -> dict:
    rows = read_jsonl(root / "inference" / name / f"{split}.jsonl")
    agreements = [row.get("predicted_number") == lower_by_id[row["id"]].get("predicted_number") for row in rows]
    agreed = [lower_by_id[row["id"]]["correct"] for row, same in zip(rows, agreements) if same]
    tokens = np.asarray([int(row.get("generated_tokens", 0)) for row in rows])
    return {
        "accuracy": float(np.mean([row["correct"] for row in rows])),
        "agreement_rate": float(np.mean(agreements)),
        "agreement_precision": float(np.mean(agreed)) if agreed else None,
        "generated_tokens_mean": float(tokens.mean()),
        "generated_tokens_p95": float(np.quantile(tokens, 0.95)),
        "final_answer_format_rate": float(np.mean([
            bool(re.search(r"Final answer\s*:", row["prediction"], re.IGNORECASE)) for row in rows
        ])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--pilot-root", default="artifacts")
    parser.add_argument("--target-root", default="artifacts/fresh_recertification")
    parser.add_argument("--split", default="recertification")
    parser.add_argument(
        "--benchmark",
        default="artifacts/fresh_recertification/results/micro_reasoning_latency_benchmark.json",
    )
    parser.add_argument("--selection-unsafe-cap", type=float, default=0.03)
    parser.add_argument(
        "--output",
        default="artifacts/fresh_recertification/results/micro_verifier_frontier.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(args.target_root)
    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    full_tokens = float(benchmark["summary"]["full_second_pass"]["generated_tokens_mean"]["median"])
    reasoning_rows = read_jsonl(root / "inference" / "lower_reasoning_128" / f"{args.split}.jsonl")
    costs = {
        "lower_answer_only": {
            "value": float(benchmark["answer_only_latency_ratio_vs_full_second_pass"]),
            "source": "measured_latency",
        },
        "lower_micro_equation": {
            "value": float(benchmark["micro_reasoning_latency_ratio_vs_full_second_pass"]),
            "source": "measured_latency",
        },
        "lower_reasoning_128": {
            "value": float(np.mean([row["generated_tokens"] for row in reasoning_rows]) / full_tokens),
            "source": "token_ratio_estimate",
        },
    }

    model = fit_gsm8k_confidence_model(Path(args.pilot_root))
    lower_rows = {
        row["id"]: row
        for row in read_jsonl(root / "inference" / "lower_concise" / f"{args.split}.jsonl")
    }
    variants = {}
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    for name, cost in costs.items():
        inputs = load_target(model, root, args.split, "upper_concise", name)
        upper_accuracy = float(inputs[3].mean())
        quality_target = float(cfg["router"]["quality_floor"]) * upper_accuracy
        feasible = []
        for low in grid:
            for high in grid:
                if low > high:
                    continue
                items = route_items(
                    inputs,
                    float(low),
                    float(high),
                    float(cfg["cost"]["lower"]),
                    cost["value"],
                    float(cfg["cost"]["upper"]),
                )
                metrics = {
                    "low_threshold": float(low),
                    "high_threshold": float(high),
                    "accuracy": float(items["correct"].mean()),
                    "unsafe_rate": float(items["unsafe"].mean()),
                    "normalized_cost": float(items["cost"].mean()),
                    "lower_coverage": float(items["accept"].mean()),
                }
                if metrics["accuracy"] >= quality_target and metrics["unsafe_rate"] <= args.selection_unsafe_cap:
                    feasible.append(metrics)
        selected = min(feasible, key=lambda row: (row["normalized_cost"], -row["accuracy"])) if feasible else None
        variants[name] = {
            "verifier_cost_lower_equivalents": cost,
            "upper_accuracy": upper_accuracy,
            "quality_target": quality_target,
            "diagnostics": output_diagnostics(root, name, args.split, lower_rows),
            "feasible_policy_count": len(feasible),
            "best_selection_policy": selected,
            "cost_gate_pass": bool(selected and selected["normalized_cost"] <= 0.90),
        }

    payload = {
        "protocol": "selection-only verifier prompt frontier; no use of the remaining 188-row certification reserve",
        "count": len(lower_rows),
        "selection_unsafe_cap": args.selection_unsafe_cap,
        "variants": variants,
        "advance_to_reserved_certification": any(row["cost_gate_pass"] for row in variants.values()),
    }
    write_json(args.output, payload)
    print(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
