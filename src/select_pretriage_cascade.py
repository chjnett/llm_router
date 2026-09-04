"""Select a query pre-triage plus output-aware post-routing cascade."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import load_config, write_json
from .select_hybrid_direct_router import load_features


def metrics(pre_probability, post_probability, lower, upper, pre_threshold, post_threshold, cfg):
    direct_upper = pre_probability < pre_threshold
    run_lower = ~direct_upper
    accept_lower = run_lower & (post_probability >= post_threshold)
    upper_call = ~accept_lower
    final = np.where(accept_lower, lower, upper)
    total_cost = run_lower.astype(float) * float(cfg["cost"]["lower"]) + upper_call.astype(float) * float(cfg["cost"]["upper"])
    return {
        "pre_threshold": float(pre_threshold),
        "post_threshold": float(post_threshold),
        "accuracy": float(final.mean()),
        "unsafe_rate": float((accept_lower & ~lower).mean()),
        "lower_coverage": float(accept_lower.mean()),
        "lower_call_rate": float(run_lower.mean()),
        "pretriage_upper_rate": float(direct_upper.mean()),
        "upper_call_rate": float(upper_call.mean()),
        "normalized_cost": float(total_cost.mean() / float(cfg["cost"]["upper"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--pilot-root", default="artifacts")
    parser.add_argument("--selection-root", default="artifacts/fresh_recertification")
    parser.add_argument("--split", default="recertification")
    parser.add_argument("--selection-unsafe-cap", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="artifacts/fresh_recertification/results/pretriage_cascade_selection.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train = load_features(Path(args.pilot_root), "router_train", "upper_7b_concise")
    target = load_features(Path(args.selection_root), args.split, "upper_concise")
    pre_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", C=0.25, random_state=42),
    ).fit(train[1], train[2].astype(int))
    pre_train = pre_model.predict_proba(train[1])[:, 1]
    pre_target = pre_model.predict_proba(target[1])[:, 1]
    post_models = {
        "confidence_lr": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", C=0.25, random_state=42),
        ).fit(train[0], train[2].astype(int)),
        "confidence_extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=8,
            class_weight="balanced",
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ).fit(train[0], train[2].astype(int)),
    }
    upper_accuracy = float(target[3].mean())
    quality_target = float(cfg["router"]["quality_floor"]) * upper_accuracy
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    variants = {}
    selectable = []
    for name, post_model in post_models.items():
        post_target = post_model.predict_proba(target[0])[:, 1]
        candidates = []
        for pre_threshold in grid:
            for post_threshold in grid:
                row = metrics(
                    pre_target, post_target, target[2], target[3],
                    float(pre_threshold), float(post_threshold), cfg,
                )
                if row["accuracy"] >= quality_target and row["unsafe_rate"] <= args.selection_unsafe_cap:
                    candidates.append(row)
        selected = min(candidates, key=lambda row: (row["normalized_cost"], -row["accuracy"])) if candidates else None
        variants[name] = {
            "post_lower_success_auc": float(roc_auc_score(target[2].astype(int), post_target)),
            "feasible_policy_count": len(candidates),
            "selected": selected,
            "cost_gate_pass": bool(selected and selected["normalized_cost"] <= 0.90),
        }
        if selected is not None:
            selectable.append({"post_model": name, **selected})
    chosen = min(selectable, key=lambda row: (row["normalized_cost"], -row["accuracy"])) if selectable else None
    payload = {
        "protocol": "query-only semantic pre-triage before Lower; output-confidence post-router after Lower; selection-only",
        "selection_count": len(target[2]),
        "selection_unsafe_cap": args.selection_unsafe_cap,
        "quality_target": quality_target,
        "upper_accuracy": upper_accuracy,
        "pretriage_lower_success_auc": float(roc_auc_score(target[2].astype(int), pre_target)),
        "pretriage_train_auc": float(roc_auc_score(train[2].astype(int), pre_train)),
        "variants": variants,
        "frozen_candidate_for_reserved_certification": chosen,
        "advance_to_reserved_certification": bool(chosen and chosen["normalized_cost"] <= 0.90),
    }
    write_json(args.output, payload)
    print(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
