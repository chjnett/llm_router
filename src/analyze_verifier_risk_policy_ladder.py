"""Build a selection-only safety-margin ladder for the answer-only verifier.

Certification and official-test results are retrospective diagnostics only. A policy
chosen after reading those diagnostics must be certified again on a fresh holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import load_config, write_json
from .run_low_cost_verifier import load_target
from .run_risk_bound_calibration import binomial_upper, fit_gsm8k_confidence_model
from .analyze_verifier_performance import route_items


def evaluate(inputs, low, high, cfg, verifier_cost):
    routed = route_items(
        inputs,
        low,
        high,
        float(cfg["cost"]["lower"]),
        verifier_cost,
        float(cfg["cost"]["upper"]),
    )
    count = len(routed["correct"])
    unsafe_count = int(routed["unsafe"].sum())
    return {
        "count": count,
        "accuracy": float(routed["correct"].mean()),
        "unsafe_count": unsafe_count,
        "unsafe_rate": float(routed["unsafe"].mean()),
        "unsafe_upper_95": binomial_upper(unsafe_count, count),
        "normalized_cost": float(routed["cost"].mean()),
        "lower_coverage": float(routed["accept"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/svamp_cross_task.yaml")
    parser.add_argument("--pilot-root", default="artifacts")
    parser.add_argument("--target-root", default="artifacts/svamp")
    parser.add_argument("--benchmark", default="artifacts/results/verifier_latency_benchmark.json")
    parser.add_argument("--quality-margin", type=float, default=0.02)
    parser.add_argument("--max-selection-unsafe-count", type=int, default=6)
    parser.add_argument("--output", default="artifacts/results/verifier_risk_policy_ladder.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    verifier_cost = float(benchmark["answer_only_latency_ratio_vs_full_second_pass"])
    model = fit_gsm8k_confidence_model(Path(args.pilot_root))
    root = Path(args.target_root)
    splits = {
        name: load_target(model, root, name, "upper_concise", "lower_answer_only")
        for name in ("risk_selection", "risk_certification", "test")
    }

    selection_upper_accuracy = float(splits["risk_selection"][3].mean())
    selection_quality_target = (
        float(cfg["router"]["quality_floor"]) * selection_upper_accuracy + args.quality_margin
    )
    candidates = []
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    for low in grid:
        for high in grid:
            if low > high:
                continue
            metrics = evaluate(splits["risk_selection"], float(low), float(high), cfg, verifier_cost)
            if metrics["accuracy"] >= selection_quality_target:
                candidates.append({"low_threshold": float(low), "high_threshold": float(high), **metrics})

    ladder = []
    for unsafe_cap in range(args.max_selection_unsafe_count + 1):
        feasible = [row for row in candidates if row["unsafe_count"] <= unsafe_cap]
        if not feasible:
            ladder.append({"selection_unsafe_cap": unsafe_cap, "selected": None})
            continue
        selected = min(feasible, key=lambda row: (row["normalized_cost"], -row["accuracy"]))
        low = selected["low_threshold"]
        high = selected["high_threshold"]
        certification = evaluate(splits["risk_certification"], low, high, cfg, verifier_cost)
        official_test = evaluate(splits["test"], low, high, cfg, verifier_cost)
        certification_upper_accuracy = float(splits["risk_certification"][3].mean())
        test_upper_accuracy = float(splits["test"][3].mean())
        retrospective_gate = {
            "risk_upper_5pct_pass": certification["unsafe_upper_95"] <= 0.05,
            "quality_floor_pass": certification["accuracy"] >= float(cfg["router"]["quality_floor"]) * certification_upper_accuracy,
            "cost_reduction_10pct_pass": certification["normalized_cost"] <= 0.90,
        }
        retrospective_gate["all_pass"] = all(retrospective_gate.values())
        ladder.append({
            "selection_unsafe_cap": unsafe_cap,
            "selected": selected,
            "risk_certification_diagnostic": {
                "always_upper_accuracy": certification_upper_accuracy,
                **certification,
            },
            "official_test_diagnostic": {
                "always_upper_accuracy": test_upper_accuracy,
                **official_test,
            },
            "retrospective_gate": retrospective_gate,
        })

    passing = [row for row in ladder if row.get("retrospective_gate", {}).get("all_pass")]
    recommendation = None
    if passing:
        # Prefer the strongest selection-only safety margin. Certification is shown
        # only as a retrospective diagnostic and is never used for this ordering.
        recommendation = min(
            passing,
            key=lambda row: (
                row["selection_unsafe_cap"],
                row["selected"]["normalized_cost"],
                -row["selected"]["accuracy"],
            ),
        )

    payload = {
        "protocol": "selection-only unsafe-count ladder; certification/test used only as retrospective diagnostics",
        "measured_verifier_cost_lower_equivalents": verifier_cost,
        "selection_quality_target": selection_quality_target,
        "selection_upper_accuracy": selection_upper_accuracy,
        "ladder": ladder,
        "retrospective_candidate_for_fresh_holdout": recommendation,
        "fresh_holdout_required": recommendation is not None,
    }
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
