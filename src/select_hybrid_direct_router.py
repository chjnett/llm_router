"""Evaluate confidence, semantic, and hybrid verifier-free routing features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import load_config, read_jsonl, write_json
from .run_confidence_router import FEATURES
from .select_direct_auxiliary_router import direct_metrics


def keyed(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in read_jsonl(path)}


def load_features(root: Path, split: str, upper_name: str):
    embeddings = np.load(root / "embeddings" / f"{split}.npz")
    ids = embeddings["ids"].tolist()
    confidence = keyed(root / "confidence" / "lower_concise" / f"{split}.jsonl")
    lower = keyed(root / "inference" / "lower_concise" / f"{split}.jsonl")
    upper = keyed(root / "inference" / upper_name / f"{split}.jsonl")
    conf = np.asarray([[float(confidence[row_id][name]) for name in FEATURES] for row_id in ids], dtype=np.float32)
    semantic = embeddings["embeddings"].astype(np.float32)
    low = np.asarray([lower[row_id]["correct"] for row_id in ids], dtype=bool)
    high = np.asarray([upper[row_id]["correct"] for row_id in ids], dtype=bool)
    return conf, semantic, low, high


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--pilot-root", default="artifacts")
    parser.add_argument("--selection-root", default="artifacts/fresh_recertification")
    parser.add_argument("--split", default="recertification")
    parser.add_argument("--selection-unsafe-cap", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="artifacts/fresh_recertification/results/hybrid_direct_selection.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train = load_features(Path(args.pilot_root), "router_train", "upper_7b_concise")
    target = load_features(Path(args.selection_root), args.split, "upper_concise")
    train_sets = {
        "confidence_lr": train[0],
        "semantic_lr": train[1],
        "hybrid_lr": np.concatenate([train[0], train[1]], axis=1),
        "confidence_extra_trees": train[0],
        "confidence_random_forest": train[0],
        "confidence_hist_gradient_boosting": train[0],
    }
    target_sets = {
        "confidence_lr": target[0],
        "semantic_lr": target[1],
        "hybrid_lr": np.concatenate([target[0], target[1]], axis=1),
        "confidence_extra_trees": target[0],
        "confidence_random_forest": target[0],
        "confidence_hist_gradient_boosting": target[0],
    }
    upper_accuracy = float(target[3].mean())
    quality_target = float(cfg["router"]["quality_floor"]) * upper_accuracy
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    variants = {}
    selectable = []
    estimators = {
        "confidence_lr": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=0.25, random_state=42)
        ),
        "semantic_lr": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=0.25, random_state=42)
        ),
        "hybrid_lr": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=0.25, random_state=42)
        ),
        "confidence_extra_trees": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=8, class_weight="balanced", max_features="sqrt", random_state=42, n_jobs=-1
        ),
        "confidence_random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=8, class_weight="balanced", max_features="sqrt", random_state=42, n_jobs=-1
        ),
        "confidence_hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, random_state=42
        ),
    }
    for name, x_train in train_sets.items():
        model = estimators[name]
        model.fit(x_train, train[2].astype(int))
        probability = model.predict_proba(target_sets[name])[:, 1]
        candidates = [direct_metrics(probability, target[2], target[3], threshold, cfg) for threshold in grid]
        feasible = [
            row for row in candidates
            if row["accuracy"] >= quality_target and row["unsafe_rate"] <= args.selection_unsafe_cap
        ]
        selected = min(feasible, key=lambda row: (row["normalized_cost"], -row["accuracy"])) if feasible else None
        variants[name] = {
            "lower_success_auc": float(roc_auc_score(target[2].astype(int), probability)),
            "feasible_policy_count": len(feasible),
            "selected": selected,
            "cost_gate_pass": bool(selected and selected["normalized_cost"] <= 0.90),
        }
        if selected is not None:
            selectable.append({"variant": name, **selected})
    chosen = min(selectable, key=lambda row: (row["normalized_cost"], -row["accuracy"])) if selectable else None
    payload = {
        "protocol": "pilot router_train fit; fresh 500 selection; direct Lower/Upper routing without verifier",
        "selection_count": len(target[2]),
        "selection_unsafe_cap": args.selection_unsafe_cap,
        "upper_accuracy": upper_accuracy,
        "quality_target": quality_target,
        "variants": variants,
        "best_candidate": chosen,
        "advance_to_reserved_certification": bool(chosen and chosen["normalized_cost"] <= 0.90),
    }
    write_json(args.output, payload)
    print(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
