"""Paired performance analysis for C3 and the frozen answer-only verifier policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .common import load_config, write_json
from .run_low_cost_verifier import load_target as load_verifier_target
from .run_risk_bound_calibration import fit_gsm8k_confidence_model, load_target as load_c3_target


def route_items(inputs, low, high, lower_cost, second_cost, upper_cost):
    probability, agreement, lower, upper = inputs[:4]
    direct = probability >= high
    second = (probability >= low) & (probability < high)
    accept = direct | (second & agreement)
    final_correct = np.where(accept, lower, upper).astype(float)
    unsafe = (accept & ~lower).astype(float)
    cost = lower_cost + second.astype(float) * second_cost + (~accept).astype(float) * upper_cost
    return {
        "correct": final_correct,
        "unsafe": unsafe,
        "cost": cost / upper_cost,
        "accept": accept,
    }


def interval(values):
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def paired_bootstrap(c3, verifier, draws=10000, seed=42):
    rng = np.random.default_rng(seed)
    count = len(c3["correct"])
    indices = rng.integers(0, count, size=(draws, count))
    payload = {"draws": draws, "methods": {}, "verifier_minus_c3": {}}
    for name, items in (("c3", c3), ("answer_only", verifier)):
        payload["methods"][name] = {}
        for metric in ("correct", "unsafe", "cost"):
            means = items[metric][indices].mean(axis=1)
            payload["methods"][name][metric] = {
                "point": float(items[metric].mean()),
                "ci95": interval(means),
            }
    for metric in ("correct", "unsafe", "cost"):
        deltas = (verifier[metric][indices] - c3[metric][indices]).mean(axis=1)
        payload["verifier_minus_c3"][metric] = {
            "point": float((verifier[metric] - c3[metric]).mean()),
            "ci95": interval(deltas),
        }
    c3_bool = c3["correct"].astype(bool)
    verifier_bool = verifier["correct"].astype(bool)
    c3_only = int((c3_bool & ~verifier_bool).sum())
    verifier_only = int((~c3_bool & verifier_bool).sum())
    discordant = c3_only + verifier_only
    payload["mcnemar_exact"] = {
        "c3_only_correct": c3_only,
        "verifier_only_correct": verifier_only,
        "discordant": discordant,
        "pvalue": float(binomtest(min(c3_only, verifier_only), discordant, 0.5).pvalue) if discordant else 1.0,
    }
    return payload


def selection_pareto(inputs, cfg, verifier_cost):
    points = []
    for low in np.round(np.arange(0.01, 1.0, 0.01), 2):
        for high in np.round(np.arange(low, 1.0, 0.01), 2):
            items = route_items(
                inputs, low, high,
                cfg["cost"]["lower"], verifier_cost, cfg["cost"]["upper"],
            )
            points.append({
                "low_threshold": float(low),
                "high_threshold": float(high),
                "accuracy": float(items["correct"].mean()),
                "unsafe": float(items["unsafe"].mean()),
                "cost": float(items["cost"].mean()),
            })
    frontier = []
    best_accuracy = -1.0
    for point in sorted(points, key=lambda row: (row["cost"], -row["accuracy"])):
        if point["accuracy"] > best_accuracy:
            frontier.append(point)
            best_accuracy = point["accuracy"]
    return frontier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/svamp_cross_task.yaml")
    parser.add_argument("--pilot-root", default="artifacts")
    parser.add_argument("--target-root", default="artifacts/svamp")
    parser.add_argument("--benchmark", default="artifacts/results/verifier_latency_benchmark.json")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--output", default="artifacts/results/verifier_performance_analysis.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    measured_verifier_cost = float(benchmark["answer_only_latency_ratio_vs_full_second_pass"])
    model = fit_gsm8k_confidence_model(Path(args.pilot_root))
    root = Path(args.target_root)
    policies = {"c3": (0.70, 0.82), "answer_only": (0.12, 0.80)}
    payload = {
        "protocol": "frozen policies; paired bootstrap on identical rows; verifier cost calibrated by measured latency ratio",
        "measured_verifier_cost_lower_equivalents": measured_verifier_cost,
        "policies": {name: {"low_threshold": values[0], "high_threshold": values[1]} for name, values in policies.items()},
        "splits": {},
    }
    for split in ("risk_certification", "test"):
        c3_inputs = load_c3_target(model, root, split, "upper_concise")
        verifier_inputs = load_verifier_target(model, root, split, "upper_concise", "lower_answer_only")
        c3_items = route_items(c3_inputs, *policies["c3"], cfg["cost"]["lower"], cfg["cost"]["lower"], cfg["cost"]["upper"])
        verifier_items = route_items(verifier_inputs, *policies["answer_only"], cfg["cost"]["lower"], measured_verifier_cost, cfg["cost"]["upper"])
        payload["splits"][split] = paired_bootstrap(c3_items, verifier_items, args.draws, cfg["seed"])
        payload["splits"][split]["always_upper_accuracy"] = float(c3_inputs[3].mean())
    selection = load_verifier_target(model, root, "risk_selection", "upper_concise", "lower_answer_only")
    payload["selection_pareto"] = selection_pareto(selection, cfg, measured_verifier_cost)
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
