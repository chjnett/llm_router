"""Final one-shot evaluation of the CV-selected adaptive pre-triage policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import load_config, write_json
from .cross_validate_adaptive_pretriage import fit_models
from .run_risk_bound_calibration import binomial_upper
from .select_hybrid_direct_router import load_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--pilot-root", default="artifacts")
    parser.add_argument("--selection-root", default="artifacts/fresh_recertification")
    parser.add_argument("--reserve-root", default="artifacts/reserved_certification")
    parser.add_argument("--target-root", default="artifacts/gsm8k_official_test")
    parser.add_argument("--pre-threshold", type=float, default=0.84)
    parser.add_argument("--post-threshold", type=float, default=0.49)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument(
        "--output",
        default="artifacts/gsm8k_official_test/results/adaptive_pretriage_official_test.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pilot = load_features(Path(args.pilot_root), "router_train", "upper_7b_concise")
    selection = load_features(Path(args.selection_root), "recertification", "upper_concise")
    reserve = load_features(Path(args.reserve_root), "certification", "upper_concise")
    target = load_features(Path(args.target_root), "test", "upper_concise")
    train_confidence = np.concatenate([pilot[0], selection[0], reserve[0]], axis=0)
    train_semantic = np.concatenate([pilot[1], selection[1], reserve[1]], axis=0)
    train_lower = np.concatenate([pilot[2], selection[2], reserve[2]], axis=0)
    pre, post = fit_models(train_semantic, train_confidence, train_lower)
    pre_probability = pre.predict_proba(target[1])[:, 1]
    post_probability = post.predict_proba(target[0])[:, 1]
    direct_upper = pre_probability < args.pre_threshold
    run_lower = ~direct_upper
    accept = run_lower & (post_probability >= args.post_threshold)
    upper_call = ~accept
    final_correct = np.where(accept, target[2], target[3]).astype(float)
    unsafe = (accept & ~target[2]).astype(float)
    item_cost = (
        run_lower.astype(float) * float(cfg["cost"]["lower"])
        + upper_call.astype(float) * float(cfg["cost"]["upper"])
    ) / float(cfg["cost"]["upper"])
    count = len(target[2])
    unsafe_count = int(unsafe.sum())
    accuracy = float(final_correct.mean())
    upper_accuracy = float(target[3].mean())
    normalized_cost = float(item_cost.mean())
    rng = np.random.default_rng(int(cfg["seed"]))
    indices = rng.integers(0, count, size=(args.draws, count))
    bootstrap = {
        name: [float(x) for x in np.quantile(values[indices].mean(axis=1), [0.025, 0.975])]
        for name, values in (
            ("accuracy", final_correct),
            ("unsafe_rate", unsafe),
            ("normalized_cost", item_cost),
            ("always_upper_accuracy", target[3].astype(float)),
        )
    }
    risk_upper = binomial_upper(unsafe_count, count, 0.05)
    quality_target = float(cfg["router"]["quality_floor"]) * upper_accuracy
    gates = {
        "quality_floor_pass": accuracy >= quality_target,
        "cost_reduction_10pct_pass": normalized_cost <= 0.90,
        "risk_upper_5pct_pass": risk_upper <= 0.05,
    }
    gates["overall_pass"] = all(gates.values())
    payload = {
        "protocol": "one-shot untouched official GSM8K test; CV-selected thresholds frozen before inference",
        "training_rows": len(train_lower),
        "test_count": count,
        "policy": {"pre_threshold": args.pre_threshold, "post_threshold": args.post_threshold},
        "always_upper_accuracy": upper_accuracy,
        "quality_target": quality_target,
        "accuracy": accuracy,
        "quality_retention_vs_upper": accuracy / upper_accuracy,
        "unsafe_count": unsafe_count,
        "unsafe_rate": float(unsafe.mean()),
        "unsafe_upper_bound_95": risk_upper,
        "normalized_cost": normalized_cost,
        "cost_reduction_vs_upper": 1.0 - normalized_cost,
        "lower_call_rate": float(run_lower.mean()),
        "lower_coverage": float(accept.mean()),
        "pretriage_upper_rate": float(direct_upper.mean()),
        "upper_call_rate": float(upper_call.mean()),
        "bootstrap_10000": bootstrap,
        "deployment_gate": gates,
    }
    write_json(args.output, payload)
    print(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
