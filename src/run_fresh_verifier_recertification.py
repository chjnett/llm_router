"""Evaluate the frozen answer-only verifier policy once on a fresh holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze_verifier_performance import route_items
from .common import load_config, write_json
from .run_low_cost_verifier import load_target
from .run_risk_bound_calibration import binomial_upper, fit_gsm8k_confidence_model


def bootstrap(items: dict[str, np.ndarray], upper: np.ndarray, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    count = len(upper)
    indices = rng.integers(0, count, size=(draws, count))
    result = {}
    for name, values in (
        ("accuracy", items["correct"]),
        ("unsafe_rate", items["unsafe"]),
        ("normalized_cost", items["cost"]),
        ("always_upper_accuracy", upper.astype(float)),
    ):
        samples = values[indices].mean(axis=1)
        result[name] = [float(x) for x in np.quantile(samples, [0.025, 0.975])]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--pilot-root", default="artifacts")
    parser.add_argument("--target-root", default="artifacts/fresh_recertification")
    parser.add_argument("--benchmark", default="artifacts/results/verifier_latency_benchmark.json")
    parser.add_argument("--split", default="recertification")
    parser.add_argument("--low-threshold", type=float, default=0.12)
    parser.add_argument("--high-threshold", type=float, default=0.86)
    parser.add_argument("--risk-limit", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument(
        "--output",
        default="artifacts/fresh_recertification/results/frozen_policy_recertification.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    verifier_cost = float(benchmark["answer_only_latency_ratio_vs_full_second_pass"])
    model = fit_gsm8k_confidence_model(Path(args.pilot_root))
    inputs = load_target(
        model,
        Path(args.target_root),
        args.split,
        "upper_concise",
        "lower_answer_only",
    )
    probability, agreement, lower, upper, verifier_tokens = inputs
    items = route_items(
        inputs,
        args.low_threshold,
        args.high_threshold,
        float(cfg["cost"]["lower"]),
        verifier_cost,
        float(cfg["cost"]["upper"]),
    )
    count = len(lower)
    unsafe_count = int(items["unsafe"].sum())
    accuracy = float(items["correct"].mean())
    upper_accuracy = float(upper.mean())
    normalized_cost = float(items["cost"].mean())
    risk_upper = binomial_upper(unsafe_count, count, args.alpha)
    quality_target = float(cfg["router"]["quality_floor"]) * upper_accuracy

    gates = {
        "risk_upper_5pct_pass": risk_upper <= args.risk_limit,
        "quality_floor_pass": accuracy >= quality_target,
        "cost_reduction_10pct_pass": normalized_cost <= 0.90,
    }
    gates["overall_pass"] = all(gates.values())
    payload = {
        "protocol": "one-shot fresh holdout; policy frozen before inference; no holdout tuning",
        "policy": {"low_threshold": args.low_threshold, "high_threshold": args.high_threshold},
        "count": count,
        "always_upper_accuracy": upper_accuracy,
        "quality_target_95pct_of_upper": quality_target,
        "accuracy": accuracy,
        "quality_retention_vs_upper": accuracy / upper_accuracy if upper_accuracy else None,
        "unsafe_count": unsafe_count,
        "unsafe_rate": float(items["unsafe"].mean()),
        "unsafe_upper_bound_95": risk_upper,
        "normalized_cost": normalized_cost,
        "cost_reduction_vs_upper": 1.0 - normalized_cost,
        "lower_coverage": float(items["accept"].mean()),
        "number_agreement_rate": float(agreement.mean()),
        "answer_only_generated_tokens_mean": float(np.mean(verifier_tokens)),
        "answer_only_generated_tokens_p95": float(np.quantile(verifier_tokens, 0.95)),
        "bootstrap_10000": bootstrap(items, upper, args.draws, int(cfg["seed"])),
        "deployment_gate": gates,
    }
    write_json(args.output, payload)
    print(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
