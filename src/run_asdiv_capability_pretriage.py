"""Evaluate the frozen capability-aware pre-triage policy on untouched ASDiv."""

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
from .select_pretriage_cascade import metrics
from .train_adaptive_capability_pretriage import load_training_rows, predict_bge
from .train_bge_routing_encoder import RoutingBGE, tokenize


def keyed(path: Path):
    return {row["id"]: row for row in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--model-root", default="artifacts/capability_pretriage")
    parser.add_argument("--target-root", default="artifacts/asdiv_external_test")
    parser.add_argument(
        "--output",
        default="artifacts/asdiv_external_test/results/capability_pretriage.json",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_root = Path(args.model_root)
    checkpoint = torch.load(model_root / "capability_router.pt", map_location="cpu", weights_only=False)
    training = json.loads((model_root / "training_result.json").read_text(encoding="utf-8"))
    selected = training["selected_policy"]
    if selected is None:
        raise RuntimeError("no validation-selected policy is available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoutingBGE(
        checkpoint["model_id"], checkpoint["projection_dim"],
        checkpoint["hidden_dim"], checkpoint["unfrozen_layers"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["model_id"])
    root = Path(args.target_root)
    rows = read_jsonl(root / "data" / "test.jsonl")
    ids = [row["id"] for row in rows]
    tokens = tokenize(tokenizer, [row["question"] for row in rows], int(cfg["contrastive"]["max_length"]))
    pre_probability = predict_bge(model, tokens, np.arange(len(rows)), device)
    _, train_confidence, train_lower, _ = load_training_rows()
    train_indices = np.asarray(checkpoint["train_indices"])
    post = ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=8, class_weight="balanced",
        max_features="sqrt", random_state=42, n_jobs=-1,
    ).fit(train_confidence[train_indices], train_lower[train_indices].astype(int))
    confidence_rows = keyed(root / "confidence" / "lower_concise" / "test.jsonl")
    target_confidence = np.asarray([
        [float(confidence_rows[row_id][name]) for name in FEATURES] for row_id in ids
    ], dtype=np.float32)
    post_probability = post.predict_proba(target_confidence)[:, 1]
    lower_rows = keyed(root / "inference" / "lower_concise" / "test.jsonl")
    upper_rows = keyed(root / "inference" / "upper_concise" / "test.jsonl")
    lower = np.asarray([lower_rows[row_id]["correct"] for row_id in ids], dtype=bool)
    upper = np.asarray([upper_rows[row_id]["correct"] for row_id in ids], dtype=bool)
    result = metrics(
        pre_probability, post_probability, lower, upper,
        selected["pre_threshold"], selected["post_threshold"], cfg,
    )
    count = len(lower)
    accept_lower = (pre_probability >= selected["pre_threshold"]) & (
        post_probability >= selected["post_threshold"]
    )
    unsafe_count = int((accept_lower & ~lower).sum())
    upper_accuracy = float(upper.mean())
    quality_target = float(cfg["router"]["quality_floor"]) * upper_accuracy
    risk_upper = binomial_upper(unsafe_count, count, 0.05)
    gates = {
        "quality_floor_pass": result["accuracy"] >= quality_target,
        "cost_reduction_10pct_pass": result["normalized_cost"] <= 0.90,
        "risk_upper_5pct_pass": risk_upper <= 0.05,
    }
    gates["overall_pass"] = all(gates.values())
    payload = {
        "protocol": "one-shot untouched ASDiv; capability model and policy selected on GSM8K only",
        "test_count": count,
        "training_rows": len(train_lower),
        "policy": {
            "pre_threshold": selected["pre_threshold"],
            "post_threshold": selected["post_threshold"],
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
