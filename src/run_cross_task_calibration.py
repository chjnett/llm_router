"""Calibrate only C3 thresholds on a target-task validation split."""

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
from .run_fresh_c3_gate import feature_matrix, keyed, prediction_arrays
from .run_selective_consistency import cascade


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--pilot-root", default="artifacts")
    parser.add_argument("--target-root", default="artifacts/svamp")
    parser.add_argument("--upper-name", default="upper_concise")
    parser.add_argument("--quality-margin", type=float, default=0.02)
    parser.add_argument("--unsafe-cap", type=float, default=0.04)
    parser.add_argument("--output", default="artifacts/results/svamp_calibrated_c3.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    pilot, target = Path(args.pilot_root), Path(args.target_root)

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

    packed = {}
    for split in ("validation", "test"):
        ids = [row["id"] for row in read_jsonl(target / "data" / f"{split}.jsonl")]
        x = feature_matrix(target / "confidence" / "lower_concise" / f"{split}.jsonl", ids)
        probability = model.predict_proba(x)[:, 1]
        lower, upper, agreement = prediction_arrays(target / "inference", split, ids, "lower_concise", args.upper_name)
        packed[split] = (probability, agreement, lower, upper)

    val_probability, val_agreement, val_lower, val_upper = packed["validation"]
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    target_accuracy = cfg["router"]["quality_floor"] * float(val_upper.mean()) + args.quality_margin
    candidates = []
    for low_threshold in grid:
        for high_threshold in grid:
            if low_threshold > high_threshold:
                continue
            metrics = cascade(val_probability, val_agreement, val_lower, val_upper, low_threshold, high_threshold, cfg)
            candidates.append({"low_threshold": float(low_threshold), "high_threshold": float(high_threshold), **metrics})
    feasible = [row for row in candidates if row["task_accuracy"] >= target_accuracy and row["unsafe_routing_rate_all"] <= args.unsafe_cap]
    selected = min(feasible, key=lambda row: (row["normalized_cascade_cost"], -row["task_accuracy"])) if feasible else None
    payload = {
        "protocol": "GSM8K confidence model + target validation threshold calibration; official target test untouched",
        "validation_constraints": {"quality_target": target_accuracy, "unsafe_cap": args.unsafe_cap},
        "selected_on_target_validation": selected,
    }
    if selected:
        probability, agreement, lower, upper = packed["test"]
        metrics = cascade(probability, agreement, lower, upper, selected["low_threshold"], selected["high_threshold"], cfg)
        payload["test"] = {"always_upper_accuracy": float(upper.mean()), **metrics}
        payload["gate"] = {
            "quality_floor_pass": metrics["task_accuracy"] >= cfg["router"]["quality_floor"] * float(upper.mean()),
            "cost_reduction_vs_always_upper": 1.0 - metrics["normalized_cascade_cost"],
            "cost_reduction_10pct_pass": metrics["normalized_cascade_cost"] <= 0.90,
            "unsafe_5pct_pass": metrics["unsafe_routing_rate_all"] <= 0.05,
        }
        payload["gate"]["overall_pass"] = all(payload["gate"][key] for key in ("quality_floor_pass", "cost_reduction_10pct_pass", "unsafe_5pct_pass"))
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
