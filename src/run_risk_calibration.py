"""Select confidence-router thresholds with validation-only safety constraints."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import load_config, write_json
from .metrics import routing_labels, system_metrics
from .run_confidence_router import load_split


def threshold_curve(probability, lower, upper, cfg):
    thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    return [
        {
            "threshold": float(threshold),
            **system_metrics(
                probability >= threshold,
                lower,
                upper,
                cfg["cost"]["lower"],
                cfg["cost"]["upper"],
            ),
        }
        for threshold in thresholds
    ]


def select_guarded(curve, upper_accuracy, quality_floor, margin, unsafe_cap):
    target = min(1.0, quality_floor * upper_accuracy + margin)
    feasible = [
        point
        for point in curve
        if point["task_accuracy"] >= target
        and point["unsafe_routing_rate_all"] <= unsafe_cap
    ]
    if not feasible:
        return None, target
    return min(feasible, key=lambda point: (point["normalized_cost"], -point["task_accuracy"])), target


def bootstrap_metrics(route_lower, lower, upper, cfg, draws=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(lower)
    accuracy, cost, unsafe = [], [], []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        metrics = system_metrics(
            route_lower[idx], lower[idx], upper[idx],
            cfg["cost"]["lower"], cfg["cost"]["upper"],
        )
        accuracy.append(metrics["task_accuracy"])
        cost.append(metrics["normalized_cost"])
        unsafe.append(metrics["unsafe_routing_rate_all"])
    return {
        "draws": draws,
        "task_accuracy_95ci": np.quantile(accuracy, [0.025, 0.975]).tolist(),
        "normalized_cost_95ci": np.quantile(cost, [0.025, 0.975]).tolist(),
        "unsafe_routing_rate_all_95ci": np.quantile(unsafe, [0.025, 0.975]).tolist(),
    }


def evaluate_model(name, model, transform, train, val, test, cfg, args):
    train_x, val_x, test_x = transform(train), transform(val), transform(test)
    labels, eligible = routing_labels(train[2], train[3])
    model.fit(train_x[eligible], labels[eligible])
    val_probability = model.predict_proba(val_x)[:, 1]
    test_probability = model.predict_proba(test_x)[:, 1]
    curve = threshold_curve(val_probability, val[2], val[3], cfg)
    selected, target = select_guarded(
        curve,
        float(val[3].mean()),
        float(cfg["router"]["quality_floor"]),
        args.quality_margin,
        args.unsafe_cap,
    )
    result = {
        "validation_always_upper_accuracy": float(val[3].mean()),
        "validation_quality_target": target,
        "selection_constraints": {
            "quality_margin": args.quality_margin,
            "unsafe_cap": args.unsafe_cap,
        },
        "selected_on_validation": selected,
    }
    if selected is None:
        result["status"] = "NO_FEASIBLE_THRESHOLD"
        return name, result
    route_lower = test_probability >= selected["threshold"]
    metrics = system_metrics(
        route_lower, test[2], test[3], cfg["cost"]["lower"], cfg["cost"]["upper"]
    )
    test_upper = float(test[3].mean())
    query_cost = args.query_baseline_cost
    result.update({
        "status": "EVALUATED",
        "test_always_upper_accuracy": test_upper,
        "test": metrics,
        "test_bootstrap": bootstrap_metrics(route_lower, test[2], test[3], cfg),
        "gate": {
            "quality_floor_pass": metrics["task_accuracy"] >= cfg["router"]["quality_floor"] * test_upper,
            "cost_reduction_vs_query": 1.0 - metrics["normalized_cost"] / query_cost,
            "cost_reduction_10pct_pass": metrics["normalized_cost"] <= query_cost * 0.90,
            "unsafe_5pct_pass": metrics["unsafe_routing_rate_all"] <= args.unsafe_cap,
        },
    })
    result["gate"]["overall_pass"] = all(
        result["gate"][key]
        for key in ("quality_floor_pass", "cost_reduction_10pct_pass", "unsafe_5pct_pass")
    )
    return name, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--confidence-dir", default="artifacts/confidence/lower_concise")
    parser.add_argument("--embedding-dir", default="artifacts/embeddings")
    parser.add_argument("--inference-dir", default="artifacts/inference")
    parser.add_argument("--lower-name", default="lower_concise")
    parser.add_argument("--upper-name", default="upper_7b_concise")
    parser.add_argument("--quality-margin", type=float, default=0.02)
    parser.add_argument("--unsafe-cap", type=float, default=0.05)
    parser.add_argument("--query-baseline-cost", type=float, default=0.852678570)
    parser.add_argument("--output", default="artifacts/results/risk_calibration.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train, val, test = [load_split(split, args) for split in ("router_train", "validation", "test")]
    variants = [
        ("confidence_lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)), lambda data: data[1]),
        ("confidence_hgb", HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0, random_state=42), lambda data: data[1]),
    ]
    results = {"protocol": "validation-only guarded threshold; test used once for reporting", "routers": {}}
    for name, model, transform in variants:
        key, value = evaluate_model(name, model, transform, train, val, test, cfg, args)
        results["routers"][key] = value
    write_json(args.output, results)
    print(args.output)


if __name__ == "__main__":
    main()
