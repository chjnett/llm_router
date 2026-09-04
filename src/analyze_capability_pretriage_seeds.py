"""Aggregate fixed-split capability pre-router results across training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-results",
        nargs="+",
        default=[
            "artifacts/capability_pretriage/training_result.json",
            "artifacts/capability_pretriage_seed43/training_result.json",
            "artifacts/capability_pretriage_seed44/training_result.json",
        ],
    )
    parser.add_argument(
        "--external-results",
        nargs="+",
        default=[
            "artifacts/asdiv_external_test/results/capability_pretriage.json",
            "artifacts/asdiv_external_test/results/capability_pretriage_seed43.json",
            "artifacts/asdiv_external_test/results/capability_pretriage_seed44.json",
        ],
    )
    parser.add_argument(
        "--output",
        default="artifacts/asdiv_external_test/results/capability_pretriage_3seed.json",
    )
    args = parser.parse_args()
    if len(args.training_results) != len(args.external_results):
        raise ValueError("training and external result counts must match")

    rows = []
    for training_path, external_path in zip(args.training_results, args.external_results):
        training = json.loads(Path(training_path).read_text(encoding="utf-8"))
        external = json.loads(Path(external_path).read_text(encoding="utf-8"))
        rows.append({
            "seed": int(training.get("training_seed", 42)),
            "validation_auc": float(training["best_validation_auc"]),
            "validation_cost_reduction": 1.0 - float(training["selected_policy"]["normalized_cost"]),
            "validation_gate_pass": bool(training["advance_to_asdiv"]),
            "asdiv_accuracy": float(external["accuracy"]),
            "asdiv_quality_retention": float(external["quality_retention_vs_upper"]),
            "asdiv_cost_reduction": float(external["cost_reduction_vs_upper"]),
            "asdiv_unsafe_rate": float(external["unsafe_rate"]),
            "asdiv_risk_upper_95": float(external["unsafe_upper_bound_95"]),
            "asdiv_all_gates_pass": bool(external["deployment_gate"]["overall_pass"]),
        })

    metric_names = [
        "validation_auc",
        "validation_cost_reduction",
        "asdiv_accuracy",
        "asdiv_quality_retention",
        "asdiv_cost_reduction",
        "asdiv_unsafe_rate",
        "asdiv_risk_upper_95",
    ]
    summary = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in rows], dtype=float)
        summary[name] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    payload = {
        "protocol": "fixed GSM8K split; seeds vary training initialization/sampling only; ASDiv thresholds never retuned",
        "seed_count": len(rows),
        "rows": rows,
        "summary": summary,
        "external_all_seeds_pass": all(row["asdiv_all_gates_pass"] for row in rows),
        "internal_all_seeds_pass": all(row["validation_gate_pass"] for row in rows),
    }
    write_json(args.output, payload)
    print(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
