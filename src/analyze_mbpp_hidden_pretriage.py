"""Evaluate MBPP pre-triage from a Lower model's one-forward representation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import read_jsonl, write_json


def probe_policy_metrics(probability, threshold, lower, upper, generation_ratio, probe_ratio) -> dict:
    """Cost assumes accepted requests reuse the probe KV cache for Lower decoding."""
    accept = probability >= threshold
    correct = np.where(accept, lower, upper)
    accuracy = float(correct.mean())
    upper_accuracy = float(upper.mean())
    normalized_latency = float(
        accept.mean() * generation_ratio + (~accept).mean() * (1.0 + probe_ratio)
    )
    return {
        "threshold": float(threshold),
        "accepted": int(accept.sum()),
        "accept_rate": float(accept.mean()),
        "unsafe_accepts": int(np.logical_and(accept, ~lower).sum()),
        "accuracy": accuracy,
        "upper_accuracy": upper_accuracy,
        "quality_retention": accuracy / upper_accuracy if upper_accuracy else 1.0,
        "normalized_latency": normalized_latency,
        "latency_reduction": 1.0 - normalized_latency,
    }


def analyze(feature_dir: Path, result_path: Path, seed: int, quality_floor: float, latency_target: float) -> dict:
    features = read_jsonl(feature_dir / "features.jsonl")
    with (feature_dir / "report.json").open("r", encoding="utf-8") as handle:
        probe_report = json.load(handle)
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not all(row["finite"] for row in features):
        raise ValueError(f"Non-finite hidden probe features in {feature_dir}")
    lower = np.asarray(result["lower"]["evaluation"]["correct"], dtype=bool)
    upper = np.asarray(result["upper"]["evaluation"]["correct"], dtype=bool)
    x = np.asarray([
        row["hidden"] + [row["next_entropy"], row["next_max_probability"], row["next_margin"], row["prompt_tokens"]]
        for row in features
    ], dtype=np.float32)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced", C=0.01, random_state=seed),
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probability = cross_val_predict(model, x, lower.astype(int), cv=folds, method="predict_proba")[:, 1]
    generation_ratio = float(result["latency_ratio"])
    upper_p50 = float(result["upper"]["metrics"]["latency_ms_p50"])
    probe_ratio = float(probe_report["latency_ms_p50"]) / upper_p50
    thresholds = sorted(set(np.linspace(0.0, 1.0, 201).tolist() + probability.tolist()))
    curve = [
        probe_policy_metrics(probability, value, lower, upper, generation_ratio, probe_ratio)
        for value in thresholds
    ]
    joint = [point for point in curve if point["quality_retention"] >= quality_floor and point["normalized_latency"] <= latency_target]
    quality_safe = [point for point in curve if point["quality_retention"] >= quality_floor]
    selected = min(joint, key=lambda point: (point["normalized_latency"], -point["accuracy"])) if joint else None
    oracle = probe_policy_metrics(lower.astype(float), 0.5, lower, upper, generation_ratio, probe_ratio)
    return {
        "model": result["lower"]["model"],
        "items": len(features),
        "feature_dimension": int(x.shape[1]),
        "probe_latency_ms_p50": float(probe_report["latency_ms_p50"]),
        "probe_latency_ratio": probe_ratio,
        "generation_latency_ratio": generation_ratio,
        "oof_roc_auc": float(roc_auc_score(lower, probability)),
        "oof_average_precision": float(average_precision_score(lower, probability)),
        "selected_policy": selected,
        "best_quality_constrained_policy": min(quality_safe, key=lambda point: (point["normalized_latency"], -point["accuracy"])),
        "oracle_reusing_kv_cache": oracle,
        "gate_pass": selected is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", default="artifacts/mbpp_hidden_probe")
    parser.add_argument("--qwen-result", default="paper/data/mbpp_screening_analysis.json")
    parser.add_argument("--smol-result", default="paper/data/mbpp_smol360_screening_analysis.json")
    parser.add_argument("--output", default="paper/data/mbpp_hidden_pretriage_analysis.json")
    parser.add_argument("--seed", type=int, default=2034)
    parser.add_argument("--quality-floor", type=float, default=0.95)
    parser.add_argument("--latency-target", type=float, default=0.90)
    args = parser.parse_args()
    root = Path(args.feature_root)
    pairs = [
        analyze(root / "qwen_1_5b", Path(args.qwen_result), args.seed, args.quality_floor, args.latency_target),
        analyze(root / "smollm2_360m", Path(args.smol_result), args.seed, args.quality_floor, args.latency_target),
    ]
    payload = {
        "protocol": "exploratory 5-fold OOF frozen hidden-state/logit probe; threshold selected on the same OOF predictions",
        "cost_assumption": "accepted Lower requests reuse probe KV cache; rejected requests pay probe plus Upper",
        "quality_floor": args.quality_floor,
        "latency_target": args.latency_target,
        "pairs": pairs,
        "continue_to_independent_200": any(pair["gate_pass"] for pair in pairs),
        "warning": "The 50-item OOF screen is model/threshold selection, not independent confirmation.",
    }
    write_json(args.output, payload)
    print(f"continue_to_independent_200={payload['continue_to_independent_200']} -> {args.output}")


if __name__ == "__main__":
    main()
