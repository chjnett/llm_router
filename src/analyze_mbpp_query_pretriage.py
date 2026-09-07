"""Exploratory query-only direct-routing gate for MBPP.

The pre-router either sends a request directly to Lower or directly to Upper. It
does not pay the Lower-first cascade overhead on requests assigned to Upper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, make_pipeline

from .common import read_jsonl, write_json


def direct_policy_metrics(
    probability: np.ndarray,
    threshold: float,
    lower: np.ndarray,
    upper: np.ndarray,
    latency_ratio: float,
) -> dict:
    accept = probability >= threshold
    correct = np.where(accept, lower, upper)
    upper_accuracy = float(upper.mean())
    accuracy = float(correct.mean())
    accepted = int(accept.sum())
    unsafe = int(np.logical_and(accept, ~lower).sum())
    normalized_latency = float(accept.mean() * latency_ratio + (~accept).mean())
    return {
        "threshold": float(threshold),
        "accepted": accepted,
        "accept_rate": float(accept.mean()),
        "unsafe_accepts": unsafe,
        "accuracy": accuracy,
        "upper_accuracy": upper_accuracy,
        "quality_retention": accuracy / upper_accuracy if upper_accuracy else 1.0,
        "normalized_latency": normalized_latency,
        "latency_reduction": 1.0 - normalized_latency,
    }


def oof_probability(texts: list[str], labels: np.ndarray, seed: int) -> np.ndarray:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=1500, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=2500, sublinear_tf=True)),
    ])
    model = make_pipeline(
        features,
        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5, random_state=seed),
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_predict(model, texts, labels.astype(int), cv=folds, method="predict_proba")[:, 1]


def analyze_pair(
    rows: list[dict],
    result: dict,
    seed: int,
    quality_floor: float,
    latency_target: float,
) -> dict:
    lower = np.asarray(result["lower"]["evaluation"]["correct"], dtype=bool)
    upper = np.asarray(result["upper"]["evaluation"]["correct"], dtype=bool)
    latency_ratio = float(result["latency_ratio"])
    texts = [row["prompt"] for row in rows]
    probability = oof_probability(texts, lower, seed)
    thresholds = sorted(set(np.linspace(0.0, 1.0, 201).tolist() + probability.tolist()))
    curve = [direct_policy_metrics(probability, value, lower, upper, latency_ratio) for value in thresholds]
    feasible = [
        point for point in curve
        if point["quality_retention"] >= quality_floor
        and point["normalized_latency"] <= latency_target
    ]
    quality_feasible = [point for point in curve if point["quality_retention"] >= quality_floor]
    latency_feasible = [point for point in curve if point["normalized_latency"] <= latency_target]
    selected = min(feasible, key=lambda point: (point["normalized_latency"], -point["accuracy"])) if feasible else None
    oracle_probability = lower.astype(float)
    oracle = direct_policy_metrics(oracle_probability, 0.5, lower, upper, latency_ratio)
    return {
        "model": result["lower"]["model"],
        "items": len(rows),
        "latency_ratio": latency_ratio,
        "lower_pass_at_1": float(lower.mean()),
        "upper_pass_at_1": float(upper.mean()),
        "oof_roc_auc": float(roc_auc_score(lower, probability)),
        "oof_average_precision": float(average_precision_score(lower, probability)),
        "selected_policy": selected,
        "best_quality_constrained_policy": min(
            quality_feasible, key=lambda point: (point["normalized_latency"], -point["accuracy"])
        ),
        "best_latency_constrained_policy": max(
            latency_feasible, key=lambda point: (point["accuracy"], -point["latency_reduction"])
        ) if latency_feasible else None,
        "oracle_direct_router": oracle,
        "gate_pass": selected is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/data/mbpp_screening_50.jsonl")
    parser.add_argument("--qwen-result", default="paper/data/mbpp_screening_analysis.json")
    parser.add_argument("--smol-result", default="paper/data/mbpp_smol360_screening_analysis.json")
    parser.add_argument("--output", default="paper/data/mbpp_query_pretriage_analysis.json")
    parser.add_argument("--seed", type=int, default=2032)
    parser.add_argument("--quality-floor", type=float, default=0.95)
    parser.add_argument("--latency-target", type=float, default=0.90)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    results = []
    for path in (args.qwen_result, args.smol_result):
        with Path(path).open("r", encoding="utf-8") as handle:
            results.append(analyze_pair(rows, json.load(handle), args.seed, args.quality_floor, args.latency_target))
    payload = {
        "protocol": "exploratory 5-fold OOF query-only TF-IDF LR; threshold selected on the same OOF predictions",
        "features": "public prompt and public tests only; no Lower output or correctness at inference",
        "quality_floor": args.quality_floor,
        "latency_target": args.latency_target,
        "pairs": results,
        "continue_to_independent_200": any(result["gate_pass"] for result in results),
        "warning": "The 50-item OOF screen is model/threshold selection, not independent confirmation.",
    }
    write_json(args.output, payload)
    print(f"continue_to_independent_200={payload['continue_to_independent_200']} -> {args.output}")


if __name__ == "__main__":
    main()
