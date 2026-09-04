"""Cross-validated feasibility test for selecting rare correct answer-only outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import read_jsonl, write_json
from .run_risk_bound_calibration import binomial_upper
from .score_screening_confidence import FEATURES


def cross_validated_probability(features: np.ndarray, labels: np.ndarray, seed: int = 42) -> np.ndarray:
    folds = min(5, int(labels.sum()), int((~labels).sum()))
    if folds < 2:
        raise ValueError("At least two positive and negative rows are required")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probability = np.zeros(len(labels), dtype=float)
    for train, test in splitter.split(features, labels):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        )
        model.fit(features[train], labels[train])
        probability[test] = model.predict_proba(features[test])[:, 1]
    return probability


def select_policy(probability, lower, upper, lower_cost_ratio, quality_floor=0.95) -> tuple[dict | None, list[dict]]:
    curve = []
    upper_accuracy = float(upper.mean())
    for threshold in np.linspace(0.0, 1.0, 201):
        accept = probability >= threshold
        correct = np.where(accept, lower, upper)
        unsafe = accept & ~lower & upper
        accepted = int(accept.sum())
        point = {
            "threshold": float(threshold),
            "acceptance_rate": float(accept.mean()),
            "system_accuracy": float(correct.mean()),
            "quality_retention": float(correct.mean() / upper_accuracy) if upper_accuracy else 0.0,
            "unsafe_count": int(unsafe.sum()),
            "unsafe_rate_given_accept": float(unsafe.sum() / accepted) if accepted else 0.0,
            "unsafe_exact_95_upper": float(binomial_upper(int(unsafe.sum()), accepted)) if accepted else 1.0,
            "normalized_latency": float(lower_cost_ratio + 1.0 - accept.mean()),
        }
        curve.append(point)
    eligible = [point for point in curve if point["quality_retention"] >= quality_floor]
    selected = min(eligible, key=lambda point: point["normalized_latency"]) if eligible else None
    return selected, curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence", required=True)
    parser.add_argument("--lower-predictions", required=True)
    parser.add_argument("--upper-predictions", required=True)
    parser.add_argument("--lower-latency-ms", type=float, required=True)
    parser.add_argument("--upper-latency-ms", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    confidence = read_jsonl(args.confidence)
    lower_by_id = {row["id"]: row for row in read_jsonl(args.lower_predictions)}
    upper_by_id = {row["id"]: row for row in read_jsonl(args.upper_predictions)}
    rows = [row for row in confidence if row["finite"] and row["id"] in lower_by_id and row["id"] in upper_by_id]
    features = np.asarray([[float(row[name]) for name in FEATURES] for row in rows], dtype=np.float32)
    lower = np.asarray([bool(lower_by_id[row["id"]]["correct"]) for row in rows])
    upper = np.asarray([bool(upper_by_id[row["id"]]["correct"]) for row in rows])
    probability = cross_validated_probability(features, lower, args.seed)
    ratio = args.lower_latency_ms / args.upper_latency_ms
    selected, curve = select_policy(probability, lower, upper, ratio)
    payload = {
        "protocol": "exploratory stratified out-of-fold confidence; threshold selected on the same OOF predictions",
        "confirmation_required": True,
        "features": FEATURES,
        "rows": len(rows),
        "positive_rows": int(lower.sum()),
        "upper_accuracy": float(upper.mean()),
        "lower_accuracy": float(lower.mean()),
        "lower_latency_ratio": ratio,
        "routing_auc": float(roc_auc_score(lower, probability)),
        "average_precision": float(average_precision_score(lower, probability)),
        "selected_quality_floor_95": selected,
        "latency_break_even_achieved": bool(selected and selected["normalized_latency"] < 1.0),
        "curve": curve,
    }
    write_json(args.output, payload)
    print(f"auc={payload['routing_auc']:.3f} ap={payload['average_precision']:.3f} break_even={payload['latency_break_even_achieved']} -> {args.output}")


if __name__ == "__main__":
    main()

