"""Select a verifier-free output-aware confidence router on a held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .common import load_config, write_json
from .train_auxiliary_confidence_router import load_split, predict, train_head


def direct_metrics(probability, lower, upper, threshold, cfg):
    accept = probability >= threshold
    final = np.where(accept, lower, upper)
    cost = float(cfg["cost"]["lower"]) + (~accept).astype(float) * float(cfg["cost"]["upper"])
    return {
        "threshold": float(threshold),
        "accuracy": float(final.mean()),
        "unsafe_rate": float((accept & ~lower).mean()),
        "lower_coverage": float(accept.mean()),
        "upper_call_rate": float((~accept).mean()),
        "normalized_cost": float(cost.mean() / float(cfg["cost"]["upper"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--pilot-root", default="artifacts")
    parser.add_argument("--selection-root", default="artifacts/fresh_recertification")
    parser.add_argument("--split", default="recertification")
    parser.add_argument("--selection-unsafe-cap", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument(
        "--output",
        default="artifacts/fresh_recertification/results/direct_auxiliary_selection.json",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pilot = Path(args.pilot_root)
    selection = Path(args.selection_root)
    train = load_split(
        pilot / "data", pilot / "inference", pilot / "confidence", "router_train",
        "lower_concise", "upper_7b_concise", "lower",
    )
    target = load_split(
        selection / "data", selection / "inference", selection / "confidence", args.split,
        "lower_concise", "upper_concise", "lower_answer_only",
    )
    scaler = StandardScaler().fit(train[0])
    train_x = scaler.transform(train[0]).astype(np.float32)
    target_x = scaler.transform(target[0]).astype(np.float32)
    upper_accuracy = float(target[2].mean())
    quality_target = float(cfg["router"]["quality_floor"]) * upper_accuracy
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)

    seeds = {}
    selectable = []
    for seed in cfg.get("seeds", [42, 43, 44]):
        model, losses = train_head(train_x, train[1], seed, args.epochs, args.learning_rate)
        probability = predict(model, target_x)
        candidates = [direct_metrics(probability, target[1], target[2], threshold, cfg) for threshold in grid]
        feasible = [
            row for row in candidates
            if row["accuracy"] >= quality_target and row["unsafe_rate"] <= args.selection_unsafe_cap
        ]
        selected = min(feasible, key=lambda row: (row["normalized_cost"], -row["accuracy"])) if feasible else None
        item = {
            "final_train_loss": losses,
            "lower_success_auc": float(roc_auc_score(target[1].astype(int), probability)),
            "feasible_policy_count": len(feasible),
            "selected": selected,
        }
        seeds[str(seed)] = item
        if selected is not None:
            selectable.append({"seed": int(seed), **selected})

    chosen = min(selectable, key=lambda row: (row["normalized_cost"], -row["accuracy"])) if selectable else None
    payload = {
        "protocol": "train on pilot router_train; select one seed and threshold on fresh 500; no verifier call",
        "selection_count": len(target[1]),
        "upper_accuracy": upper_accuracy,
        "quality_target": quality_target,
        "selection_unsafe_cap": args.selection_unsafe_cap,
        "seeds": seeds,
        "frozen_candidate_for_reserved_certification": chosen,
        "advance_to_reserved_certification": bool(chosen and chosen["normalized_cost"] <= 0.90),
    }
    write_json(args.output, payload)
    print(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
