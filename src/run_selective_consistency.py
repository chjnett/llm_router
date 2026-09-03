"""Evaluate a validation-selected two-pass Lower consistency cascade."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import load_config, read_jsonl, write_json
from .metrics import routing_labels
from .run_confidence_router import load_split


def number_agreement(split, args, ids):
    alternate = {
        row["id"]: row
        for row in read_jsonl(Path(args.inference_dir) / args.alternate_lower_name / f"{split}.jsonl")
    }
    concise = {
        row["id"]: row
        for row in read_jsonl(Path(args.inference_dir) / args.lower_name / f"{split}.jsonl")
    }
    return np.asarray([
        concise[row_id].get("predicted_number") is not None
        and concise[row_id].get("predicted_number") == alternate[row_id].get("predicted_number")
        for row_id in ids
    ], dtype=bool)


def cascade(probability, agreement, lower, upper, low, high, cfg):
    direct_accept = probability >= high
    second_pass = (probability >= low) & (probability < high)
    agreement_accept = second_pass & agreement
    accept_lower = direct_accept | agreement_accept
    escalate = ~accept_lower
    final_correct = np.where(accept_lower, lower, upper)
    lower_cost, upper_cost = cfg["cost"]["lower"], cfg["cost"]["upper"]
    total_cost = lower_cost + second_pass.astype(float) * lower_cost + escalate.astype(float) * upper_cost
    return {
        "task_accuracy": float(final_correct.mean()),
        "lower_coverage": float(accept_lower.mean()),
        "direct_accept_rate": float(direct_accept.mean()),
        "second_pass_rate": float(second_pass.mean()),
        "agreement_accept_rate": float(agreement_accept.mean()),
        "upper_call_rate": float(escalate.mean()),
        "unsafe_routing_rate_all": float((accept_lower & ~lower).mean()),
        "normalized_cascade_cost": float(total_cost.mean() / upper_cost),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--confidence-dir", default="artifacts/confidence/lower_concise")
    parser.add_argument("--embedding-dir", default="artifacts/embeddings")
    parser.add_argument("--inference-dir", default="artifacts/inference")
    parser.add_argument("--lower-name", default="lower_concise")
    parser.add_argument("--alternate-lower-name", default="lower")
    parser.add_argument("--upper-name", default="upper_7b_concise")
    parser.add_argument("--quality-margin", type=float, default=0.02)
    parser.add_argument("--unsafe-cap", type=float, default=0.05)
    parser.add_argument("--evaluation-unsafe-target", type=float, default=0.05)
    parser.add_argument("--output", default="artifacts/results/selective_consistency.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train, val, test = [load_split(split, args) for split in ("router_train", "validation", "test")]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))
    labels, eligible = routing_labels(train[2], train[3])
    model.fit(train[1][eligible], labels[eligible])
    val_probability = model.predict_proba(val[1])[:, 1]
    test_probability = model.predict_proba(test[1])[:, 1]
    val_ids = np.load(Path(args.embedding_dir) / "validation.npz")["ids"].tolist()
    test_ids = np.load(Path(args.embedding_dir) / "test.npz")["ids"].tolist()
    val_agreement = number_agreement("validation", args, val_ids)
    test_agreement = number_agreement("test", args, test_ids)
    target = min(1.0, cfg["router"]["quality_floor"] * float(val[3].mean()) + args.quality_margin)
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    candidates = []
    for low in grid:
        for high in grid:
            if low > high:
                continue
            metrics = cascade(val_probability, val_agreement, val[2], val[3], low, high, cfg)
            candidates.append({"low_threshold": float(low), "high_threshold": float(high), **metrics})
    feasible = [row for row in candidates if row["task_accuracy"] >= target and row["unsafe_routing_rate_all"] <= args.unsafe_cap]
    selected = min(feasible, key=lambda row: (row["normalized_cascade_cost"], -row["task_accuracy"])) if feasible else None
    payload = {
        "protocol": "validation-only two-threshold selection; all Lower and Upper calls included in cost",
        "validation_quality_target": target,
        "validation_number_agreement_rate": float(val_agreement.mean()),
        "test_number_agreement_rate": float(test_agreement.mean()),
        "selected_on_validation": selected,
    }
    if selected:
        test_metrics = cascade(
            test_probability, test_agreement, test[2], test[3],
            selected["low_threshold"], selected["high_threshold"], cfg,
        )
        payload["test"] = test_metrics
        payload["gate"] = {
            "quality_floor_pass": test_metrics["task_accuracy"] >= cfg["router"]["quality_floor"] * float(test[3].mean()),
            "cost_reduction_vs_always_upper": 1.0 - test_metrics["normalized_cascade_cost"],
            "cost_reduction_10pct_pass": test_metrics["normalized_cascade_cost"] <= 0.90,
            "unsafe_5pct_pass": test_metrics["unsafe_routing_rate_all"] <= args.evaluation_unsafe_target,
        }
        payload["gate"]["overall_pass"] = all(payload["gate"][key] for key in ("quality_floor_pass", "cost_reduction_10pct_pass", "unsafe_5pct_pass"))
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
