"""Nested cross-validation diagnostic for a distribution-adapted pre-triage router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import load_config, write_json
from .run_risk_bound_calibration import binomial_upper
from .select_hybrid_direct_router import load_features
from .select_pretriage_cascade import metrics


def fit_models(semantic, confidence, labels):
    pre = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", C=0.25, random_state=42),
    ).fit(semantic, labels.astype(int))
    post = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=8,
        class_weight="balanced",
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ).fit(confidence, labels.astype(int))
    return pre, post


def choose(pre_probability, post_probability, lower, upper, cfg, unsafe_cap):
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    target = float(cfg["router"]["quality_floor"]) * float(upper.mean())
    feasible = []
    for pre_threshold in grid:
        for post_threshold in grid:
            row = metrics(pre_probability, post_probability, lower, upper, pre_threshold, post_threshold, cfg)
            if row["accuracy"] >= target and row["unsafe_rate"] <= unsafe_cap:
                feasible.append(row)
    return min(feasible, key=lambda row: (row["normalized_cost"], -row["accuracy"])) if feasible else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--pilot-root", default="artifacts")
    parser.add_argument("--selection-root", default="artifacts/fresh_recertification")
    parser.add_argument("--reserve-root", default="artifacts/reserved_certification")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--unsafe-cap", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="artifacts/reserved_certification/results/adaptive_pretriage_nested_cv.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pilot = load_features(Path(args.pilot_root), "router_train", "upper_7b_concise")
    selection = load_features(Path(args.selection_root), "recertification", "upper_concise")
    reserve = load_features(Path(args.reserve_root), "certification", "upper_concise")
    confidence = np.concatenate([selection[0], reserve[0]], axis=0)
    semantic = np.concatenate([selection[1], reserve[1]], axis=0)
    lower = np.concatenate([selection[2], reserve[2]], axis=0)
    upper = np.concatenate([selection[3], reserve[3]], axis=0)

    final_correct = np.zeros(len(lower), dtype=bool)
    unsafe = np.zeros(len(lower), dtype=bool)
    item_cost = np.zeros(len(lower), dtype=float)
    fold_rows = []
    outer = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    for fold, (outer_train, outer_test) in enumerate(outer.split(confidence, lower.astype(int)), start=1):
        model_idx, policy_idx = train_test_split(
            outer_train,
            test_size=0.25,
            random_state=100 + fold,
            stratify=lower[outer_train].astype(int),
        )
        train_confidence = np.concatenate([pilot[0], confidence[model_idx]], axis=0)
        train_semantic = np.concatenate([pilot[1], semantic[model_idx]], axis=0)
        train_labels = np.concatenate([pilot[2], lower[model_idx]], axis=0)
        pre, post = fit_models(train_semantic, train_confidence, train_labels)
        selected = choose(
            pre.predict_proba(semantic[policy_idx])[:, 1],
            post.predict_proba(confidence[policy_idx])[:, 1],
            lower[policy_idx], upper[policy_idx], cfg, args.unsafe_cap,
        )
        if selected is None:
            selected = {"pre_threshold": 1.0, "post_threshold": 1.0}
        pre_test = pre.predict_proba(semantic[outer_test])[:, 1]
        post_test = post.predict_proba(confidence[outer_test])[:, 1]
        direct_upper = pre_test < selected["pre_threshold"]
        run_lower = ~direct_upper
        accept = run_lower & (post_test >= selected["post_threshold"])
        final_correct[outer_test] = np.where(accept, lower[outer_test], upper[outer_test])
        unsafe[outer_test] = accept & ~lower[outer_test]
        item_cost[outer_test] = (
            run_lower.astype(float) * float(cfg["cost"]["lower"])
            + (~accept).astype(float) * float(cfg["cost"]["upper"])
        ) / float(cfg["cost"]["upper"])
        fold_rows.append({
            "fold": fold,
            "model_train_count": len(model_idx) + len(pilot[2]),
            "policy_selection_count": len(policy_idx),
            "test_count": len(outer_test),
            "selected": selected,
            "test_accuracy": float(final_correct[outer_test].mean()),
            "test_unsafe_rate": float(unsafe[outer_test].mean()),
            "test_normalized_cost": float(item_cost[outer_test].mean()),
        })

    unsafe_count = int(unsafe.sum())
    accuracy = float(final_correct.mean())
    upper_accuracy = float(upper.mean())
    normalized_cost = float(item_cost.mean())
    payload = {
        "protocol": "exploratory nested 5-fold CV; domain adaptation on fresh rows; not an independent certificate",
        "count": len(lower),
        "folds": fold_rows,
        "always_upper_accuracy": upper_accuracy,
        "accuracy": accuracy,
        "quality_retention_vs_upper": accuracy / upper_accuracy,
        "unsafe_count": unsafe_count,
        "unsafe_rate": float(unsafe.mean()),
        "unsafe_upper_bound_95": binomial_upper(unsafe_count, len(lower), 0.05),
        "normalized_cost": normalized_cost,
        "cost_reduction_vs_upper": 1.0 - normalized_cost,
        "diagnostic_gate": {
            "quality_floor_pass": accuracy >= float(cfg["router"]["quality_floor"]) * upper_accuracy,
            "cost_reduction_10pct_pass": normalized_cost <= 0.90,
        },
    }
    write_json(args.output, payload)
    print(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
