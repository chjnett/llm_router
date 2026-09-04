"""One-shot certification of the frozen query pre-triage cascade."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import load_config, write_json
from .run_risk_bound_calibration import binomial_upper
from .select_hybrid_direct_router import load_features
from .select_pretriage_cascade import metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--pilot-root", default="artifacts")
    parser.add_argument("--target-root", default="artifacts/reserved_certification")
    parser.add_argument("--split", default="certification")
    parser.add_argument("--pre-threshold", type=float, default=0.66)
    parser.add_argument("--post-threshold", type=float, default=0.56)
    parser.add_argument("--risk-limit", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="artifacts/reserved_certification/results/pretriage_certification.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train = load_features(Path(args.pilot_root), "router_train", "upper_7b_concise")
    target = load_features(Path(args.target_root), args.split, "upper_concise")
    pre_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", C=0.25, random_state=42),
    ).fit(train[1], train[2].astype(int))
    post_model = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=8,
        class_weight="balanced",
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ).fit(train[0], train[2].astype(int))
    pre_probability = pre_model.predict_proba(target[1])[:, 1]
    post_probability = post_model.predict_proba(target[0])[:, 1]
    result = metrics(
        pre_probability, post_probability, target[2], target[3],
        args.pre_threshold, args.post_threshold, cfg,
    )
    count = len(target[2])
    unsafe_count = int(round(result["unsafe_rate"] * count))
    risk_upper = binomial_upper(unsafe_count, count, args.alpha)
    upper_accuracy = float(target[3].mean())
    quality_target = float(cfg["router"]["quality_floor"]) * upper_accuracy
    gates = {
        "risk_upper_5pct_pass": risk_upper <= args.risk_limit,
        "quality_floor_pass": result["accuracy"] >= quality_target,
        "cost_reduction_10pct_pass": result["normalized_cost"] <= 0.90,
    }
    gates["overall_pass"] = all(gates.values())
    payload = {
        "protocol": "one-shot final reserve; seed, models and thresholds frozen before inference",
        "policy": {
            "pre_threshold": args.pre_threshold,
            "post_threshold": args.post_threshold,
            "post_model": "confidence_extra_trees",
        },
        "count": count,
        "always_upper_accuracy": upper_accuracy,
        "quality_target": quality_target,
        "quality_retention_vs_upper": result["accuracy"] / upper_accuracy if upper_accuracy else None,
        "unsafe_count": unsafe_count,
        "unsafe_upper_bound_95": risk_upper,
        "cost_reduction_vs_upper": 1.0 - result["normalized_cost"],
        **result,
        "deployment_gate": gates,
    }
    write_json(args.output, payload)
    print(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
