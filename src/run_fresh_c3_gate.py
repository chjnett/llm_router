"""Evaluate the frozen C3 cascade once on the non-overlapping fresh holdout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import load_config, read_jsonl, write_json
from .metrics import routing_labels
from .run_confidence_router import FEATURES
from .run_selective_consistency import cascade


def keyed(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in read_jsonl(path)}


def feature_matrix(path: Path, ids: list[str]) -> np.ndarray:
    rows = keyed(path)
    return np.asarray([[float(rows[row_id][name]) for name in FEATURES] for row_id in ids], dtype=np.float32)


def prediction_arrays(root: Path, split: str, ids: list[str], lower_name: str, upper_name: str):
    lower = keyed(root / lower_name / f"{split}.jsonl")
    alternate = keyed(root / "lower_second_prompt" / f"{split}.jsonl")
    upper = keyed(root / upper_name / f"{split}.jsonl")
    low_correct = np.asarray([lower[row_id]["correct"] for row_id in ids], dtype=bool)
    high_correct = np.asarray([upper[row_id]["correct"] for row_id in ids], dtype=bool)
    agreement = np.asarray([
        lower[row_id].get("predicted_number") is not None
        and lower[row_id].get("predicted_number") == alternate[row_id].get("predicted_number")
        for row_id in ids
    ], dtype=bool)
    return low_correct, high_correct, agreement


def bootstrap(route_inputs, draws=10000, seed=42):
    probability, agreement, lower, upper, low_threshold, high_threshold, cfg = route_inputs
    rng = np.random.default_rng(seed)
    values = {"task_accuracy": [], "normalized_cascade_cost": [], "unsafe_routing_rate_all": []}
    for _ in range(draws):
        idx = rng.integers(0, len(lower), len(lower))
        metrics = cascade(probability[idx], agreement[idx], lower[idx], upper[idx], low_threshold, high_threshold, cfg)
        for key in values:
            values[key].append(metrics[key])
    return {f"{key}_95ci": np.quantile(value, [0.025, 0.975]).tolist() for key, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--pilot-root", default="artifacts")
    parser.add_argument("--fresh-root", default="artifacts/fresh_holdout")
    parser.add_argument("--lower-name", default="lower_concise")
    parser.add_argument("--upper-name", default="upper_concise")
    parser.add_argument("--low-threshold", type=float, default=0.20)
    parser.add_argument("--high-threshold", type=float, default=0.75)
    parser.add_argument("--output", default="artifacts/results/fresh_c3_gate.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    pilot = Path(args.pilot_root)
    fresh = Path(args.fresh_root)

    train_conf = read_jsonl(pilot / "confidence" / "lower_concise" / "router_train.jsonl")
    train_ids = [row["id"] for row in train_conf]
    train_x = np.asarray([[float(row[name]) for name in FEATURES] for row in train_conf], dtype=np.float32)
    train_lower = keyed(pilot / "inference" / "lower_concise" / "router_train.jsonl")
    train_upper = keyed(pilot / "inference" / "upper_7b_concise" / "router_train.jsonl")
    low = np.asarray([train_lower[row_id]["correct"] for row_id in train_ids], dtype=bool)
    high = np.asarray([train_upper[row_id]["correct"] for row_id in train_ids], dtype=bool)
    labels, eligible = routing_labels(low, high)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))
    model.fit(train_x[eligible], labels[eligible])

    payload = {
        "protocol": "C3 frozen before fresh evaluation; thresholds 0.20/0.75; no fresh-set tuning",
        "policy": {"low_threshold": args.low_threshold, "high_threshold": args.high_threshold},
        "splits": {},
    }
    for split in ("validation", "test"):
        data = read_jsonl(fresh / "data" / f"{split}.jsonl")
        ids = [row["id"] for row in data]
        x = feature_matrix(fresh / "confidence" / args.lower_name / f"{split}.jsonl", ids)
        probability = model.predict_proba(x)[:, 1]
        lower, upper, agreement = prediction_arrays(fresh / "inference", split, ids, args.lower_name, args.upper_name)
        metrics = cascade(probability, agreement, lower, upper, args.low_threshold, args.high_threshold, cfg)
        payload["splits"][split] = {
            "count": len(ids),
            "always_upper_accuracy": float(upper.mean()),
            "number_agreement_rate": float(agreement.mean()),
            **metrics,
        }
        if split == "test":
            payload["splits"][split]["bootstrap_10000"] = bootstrap((probability, agreement, lower, upper, args.low_threshold, args.high_threshold, cfg))

    test = payload["splits"]["test"]
    payload["gate"] = {
        "quality_floor_pass": test["task_accuracy"] >= cfg["router"]["quality_floor"] * test["always_upper_accuracy"],
        "cost_reduction_vs_always_upper": 1.0 - test["normalized_cascade_cost"],
        "cost_reduction_10pct_pass": test["normalized_cascade_cost"] <= 0.90,
        "unsafe_5pct_pass": test["unsafe_routing_rate_all"] <= 0.05,
    }
    payload["gate"]["overall_pass"] = all(payload["gate"][key] for key in ("quality_floor_pass", "cost_reduction_10pct_pass", "unsafe_5pct_pass"))
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
