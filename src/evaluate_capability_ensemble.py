"""Evaluate a frozen capability seed ensemble on one external dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from transformers import AutoTokenizer

from .common import load_config, read_jsonl, write_json
from .run_confidence_router import FEATURES
from .run_risk_bound_calibration import binomial_upper
from .select_capability_ensemble_risk_bound import ensemble_probability
from .select_pretriage_cascade import metrics
from .train_adaptive_capability_pretriage import load_training_rows
from .train_bge_routing_encoder import tokenize


def keyed(path: Path):
    return {row["id"]: row for row in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument(
        "--selection",
        default="artifacts/capability_ensemble/risk_bound_selection.json",
    )
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    policy = selection["selected_policy"]
    if policy is None:
        raise RuntimeError("risk-bound selection produced no feasible policy")
    checkpoints = [
        torch.load(Path(root) / "capability_router.pt", map_location="cpu", weights_only=False)
        for root in selection["model_roots"]
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.target_root)
    rows = read_jsonl(root / "data" / "test.jsonl")
    ids = [row["id"] for row in rows]
    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0]["model_id"])
    tokens = tokenize(
        tokenizer, [row["question"] for row in rows],
        int(cfg["contrastive"]["max_length"]),
    )
    pre_probability, _ = ensemble_probability(
        checkpoints, tokens, np.arange(len(rows)), device
    )

    _, train_confidence, train_lower, _ = load_training_rows()
    train_indices = np.asarray(checkpoints[0]["train_indices"])
    post_model = ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=8, class_weight="balanced",
        max_features="sqrt", random_state=42, n_jobs=-1,
    ).fit(train_confidence[train_indices], train_lower[train_indices].astype(int))
    confidence_rows = keyed(root / "confidence" / "lower_concise" / "test.jsonl")
    confidence = np.asarray([
        [float(confidence_rows[row_id][name]) for name in FEATURES] for row_id in ids
    ], dtype=np.float32)
    post_probability = post_model.predict_proba(confidence)[:, 1]
    lower_rows = keyed(root / "inference" / "lower_concise" / "test.jsonl")
    upper_rows = keyed(root / "inference" / "upper_concise" / "test.jsonl")
    lower = np.asarray([lower_rows[row_id]["correct"] for row_id in ids], dtype=bool)
    upper = np.asarray([upper_rows[row_id]["correct"] for row_id in ids], dtype=bool)
    result = metrics(
        pre_probability, post_probability, lower, upper,
        policy["pre_threshold"], policy["post_threshold"], cfg,
    )
    accept_lower = (pre_probability >= policy["pre_threshold"]) & (
        post_probability >= policy["post_threshold"]
    )
    unsafe_count = int((accept_lower & ~lower).sum())
    risk_upper = binomial_upper(unsafe_count, len(lower), 0.05)
    upper_accuracy = float(upper.mean())
    quality_target = float(cfg["router"]["quality_floor"]) * upper_accuracy
    gates = {
        "quality_floor_pass": result["accuracy"] >= quality_target,
        "cost_reduction_10pct_pass": result["normalized_cost"] <= 0.90,
        "risk_upper_5pct_pass": risk_upper <= 0.05,
    }
    gates["overall_pass"] = all(gates.values())
    payload = {
        "protocol": f"frozen 3-seed ensemble selected on GSM8K risk upper bound; no {args.dataset_name} retuning",
        "dataset": args.dataset_name,
        "test_count": len(lower),
        "policy": {
            "pre_threshold": policy["pre_threshold"],
            "post_threshold": policy["post_threshold"],
        },
        "always_lower_accuracy": float(lower.mean()),
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
