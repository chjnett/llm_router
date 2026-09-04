"""Report all CV-derived median policies on the official test without post-selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import load_config, write_json
from .cross_validate_adaptive_pretriage import fit_models
from .run_risk_bound_calibration import binomial_upper
from .select_hybrid_direct_router import load_features
from .select_pretriage_cascade import metrics


POLICIES = {
    "inner_cap_2pct_median": (0.89, 0.28),
    "inner_cap_3pct_median_primary": (0.84, 0.49),
    "inner_cap_4pct_median": (0.80, 0.51),
    "inner_cap_5pct_median": (0.85, 0.39),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--output", default="artifacts/gsm8k_official_test/results/pretriage_policy_ladder.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    pilot = load_features(Path("artifacts"), "router_train", "upper_7b_concise")
    selection = load_features(Path("artifacts/fresh_recertification"), "recertification", "upper_concise")
    reserve = load_features(Path("artifacts/reserved_certification"), "certification", "upper_concise")
    target = load_features(Path("artifacts/gsm8k_official_test"), "test", "upper_concise")
    train_confidence = np.concatenate([pilot[0], selection[0], reserve[0]], axis=0)
    train_semantic = np.concatenate([pilot[1], selection[1], reserve[1]], axis=0)
    train_lower = np.concatenate([pilot[2], selection[2], reserve[2]], axis=0)
    pre, post = fit_models(train_semantic, train_confidence, train_lower)
    pre_probability = pre.predict_proba(target[1])[:, 1]
    post_probability = post.predict_proba(target[0])[:, 1]
    upper_accuracy = float(target[3].mean())
    quality_target = float(cfg["router"]["quality_floor"]) * upper_accuracy
    rows = {}
    for name, (pre_threshold, post_threshold) in POLICIES.items():
        row = metrics(
            pre_probability, post_probability, target[2], target[3],
            pre_threshold, post_threshold, cfg,
        )
        unsafe_count = int(round(row["unsafe_rate"] * len(target[2])))
        row.update({
            "unsafe_count": unsafe_count,
            "unsafe_upper_bound_95": binomial_upper(unsafe_count, len(target[2]), 0.05),
            "quality_retention_vs_upper": row["accuracy"] / upper_accuracy,
            "cost_reduction_vs_upper": 1.0 - row["normalized_cost"],
        })
        row["gates"] = {
            "quality_floor_pass": row["accuracy"] >= quality_target,
            "cost_reduction_10pct_pass": row["normalized_cost"] <= 0.90,
            "risk_upper_5pct_pass": row["unsafe_upper_bound_95"] <= 0.05,
        }
        row["gates"]["overall_pass"] = all(row["gates"].values())
        rows[name] = row
    payload = {
        "protocol": "all four nested-CV median policies reported; primary remains 3pct policy; no official-test selection",
        "test_count": len(target[2]),
        "always_upper_accuracy": upper_accuracy,
        "policies": rows,
    }
    write_json(args.output, payload)
    print(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
