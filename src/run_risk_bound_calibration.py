"""Select C3 thresholds using a one-sided Clopper-Pearson risk bound."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import load_config, read_jsonl, write_json
from .metrics import routing_labels
from .run_confidence_router import FEATURES
from .run_fresh_c3_gate import feature_matrix, keyed, prediction_arrays
from .run_selective_consistency import cascade


def binomial_upper(successes: int, trials: int, alpha: float = 0.05) -> float:
    """One-sided exact upper confidence bound for a binomial rate."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes >= trials:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 1, trials - successes))


def fit_gsm8k_confidence_model(root: Path):
    rows = read_jsonl(root / "confidence" / "lower_concise" / "router_train.jsonl")
    ids = [row["id"] for row in rows]
    x = np.asarray([[float(row[name]) for name in FEATURES] for row in rows], dtype=np.float32)
    lower = keyed(root / "inference" / "lower_concise" / "router_train.jsonl")
    upper = keyed(root / "inference" / "upper_7b_concise" / "router_train.jsonl")
    low = np.asarray([lower[row_id]["correct"] for row_id in ids], dtype=bool)
    high = np.asarray([upper[row_id]["correct"] for row_id in ids], dtype=bool)
    labels, eligible = routing_labels(low, high)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))
    model.fit(x[eligible], labels[eligible])
    return model


def load_target(model, root: Path, split: str, upper_name: str):
    ids = [row["id"] for row in read_jsonl(root / "data" / f"{split}.jsonl")]
    x = feature_matrix(root / "confidence" / "lower_concise" / f"{split}.jsonl", ids)
    probability = model.predict_proba(x)[:, 1]
    lower, upper, agreement = prediction_arrays(root / "inference", split, ids, "lower_concise", upper_name)
    return probability, agreement, lower, upper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--pilot-root", default="artifacts")
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--upper-name", default="upper_concise")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--risk-limit", type=float, default=0.05)
    parser.add_argument("--quality-margin", type=float, default=0.02)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = fit_gsm8k_confidence_model(Path(args.pilot_root))
    target_root = Path(args.target_root)
    validation = load_target(model, target_root, "validation", args.upper_name)
    test = load_target(model, target_root, "test", args.upper_name)
    val_probability, val_agreement, val_lower, val_upper = validation
    quality_target = cfg["router"]["quality_floor"] * float(val_upper.mean()) + args.quality_margin
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    candidates = []
    for low_threshold in grid:
        for high_threshold in grid:
            if low_threshold > high_threshold:
                continue
            metrics = cascade(val_probability, val_agreement, val_lower, val_upper, low_threshold, high_threshold, cfg)
            unsafe_count = int(round(metrics["unsafe_routing_rate_all"] * len(val_lower)))
            risk_upper = binomial_upper(unsafe_count, len(val_lower), args.alpha)
            candidates.append({
                "low_threshold": float(low_threshold),
                "high_threshold": float(high_threshold),
                "unsafe_count": unsafe_count,
                "unsafe_upper_bound": risk_upper,
                **metrics,
            })
    feasible = [
        row for row in candidates
        if row["task_accuracy"] >= quality_target and row["unsafe_upper_bound"] <= args.risk_limit
    ]
    selected = min(feasible, key=lambda row: (row["normalized_cascade_cost"], -row["task_accuracy"])) if feasible else None
    payload = {
        "protocol": "validation-only one-sided Clopper-Pearson unsafe-risk upper bound",
        "confidence_level": 1.0 - args.alpha,
        "risk_limit": args.risk_limit,
        "validation_quality_target": quality_target,
        "feasible_policy_count": len(feasible),
        "selected_on_validation": selected,
    }
    if selected is not None:
        probability, agreement, lower, upper = test
        metrics = cascade(probability, agreement, lower, upper, selected["low_threshold"], selected["high_threshold"], cfg)
        unsafe_count = int(round(metrics["unsafe_routing_rate_all"] * len(lower)))
        payload["test"] = {
            "count": len(lower),
            "always_upper_accuracy": float(upper.mean()),
            "unsafe_count": unsafe_count,
            "unsafe_upper_bound_95": binomial_upper(unsafe_count, len(lower), args.alpha),
            **metrics,
        }
        payload["gate"] = {
            "quality_floor_pass": metrics["task_accuracy"] >= cfg["router"]["quality_floor"] * float(upper.mean()),
            "cost_reduction_vs_always_upper": 1.0 - metrics["normalized_cascade_cost"],
            "cost_reduction_10pct_pass": metrics["normalized_cascade_cost"] <= 0.90,
            "unsafe_point_5pct_pass": metrics["unsafe_routing_rate_all"] <= args.risk_limit,
        }
        payload["gate"]["overall_point_gate_pass"] = all(
            payload["gate"][key]
            for key in ("quality_floor_pass", "cost_reduction_10pct_pass", "unsafe_point_5pct_pass")
        )
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
