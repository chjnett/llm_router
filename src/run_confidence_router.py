"""Evaluate confidence-aware selective routing against query-only BGE."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import load_config, read_jsonl, write_json
from .metrics import routing_labels, select_operating_point, system_metrics


FEATURES = ["mean_logprob", "min_logprob", "std_logprob", "mean_entropy", "max_entropy", "mean_margin", "min_margin", "completion_tokens", "completion_chars", "number_count", "has_final_answer"]


def load_split(split: str, args):
    confidence = {row["id"]: row for row in read_jsonl(Path(args.confidence_dir) / f"{split}.jsonl")}
    lower = {row["id"]: row for row in read_jsonl(Path(args.inference_dir) / args.lower_name / f"{split}.jsonl")}
    upper = {row["id"]: row for row in read_jsonl(Path(args.inference_dir) / args.upper_name / f"{split}.jsonl")}
    packed = np.load(Path(args.embedding_dir) / f"{split}.npz")
    ids = packed["ids"].tolist()
    conf = np.asarray([[float(confidence[row_id][name]) for name in FEATURES] for row_id in ids], dtype=np.float32)
    low = np.asarray([lower[row_id]["correct"] for row_id in ids], dtype=bool)
    high = np.asarray([upper[row_id]["correct"] for row_id in ids], dtype=bool)
    return packed["embeddings"], conf, low, high


def choose(model, x, lower, upper, cfg):
    probability = model.predict_proba(x)[:, 1]
    curve = [{"threshold": t, **system_metrics(probability >= t, lower, upper, cfg["cost"]["lower"], cfg["cost"]["upper"])} for t in cfg["router"]["thresholds"]]
    return select_operating_point(curve, float(upper.mean()), cfg["router"]["quality_floor"])


def evaluate(name, model, train_x, val_x, test_x, train, val, test, cfg):
    labels, eligible = routing_labels(train[2], train[3])
    model.fit(train_x[eligible], labels[eligible])
    point = choose(model, val_x, val[2], val[3], cfg)
    probability = model.predict_proba(test_x)[:, 1]
    test_labels, test_eligible = routing_labels(test[2], test[3])
    metrics = system_metrics(probability >= point["threshold"], test[2], test[3], cfg["cost"]["lower"], cfg["cost"]["upper"])
    return name, {"selected_on_validation": point, "test": metrics, "routing_auc_eligible": float(roc_auc_score(test_labels[test_eligible], probability[test_eligible]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--confidence-dir", default="artifacts/confidence/lower_concise")
    parser.add_argument("--embedding-dir", default="artifacts/embeddings")
    parser.add_argument("--inference-dir", default="artifacts/inference")
    parser.add_argument("--lower-name", default="lower_concise")
    parser.add_argument("--upper-name", default="upper_7b_concise")
    parser.add_argument("--output", default="artifacts/results/confidence_router.json")
    args = parser.parse_args(); cfg = load_config(args.config)
    train, val, test = [load_split(split, args) for split in ("router_train", "validation", "test")]
    variants = [
        ("confidence_lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)), lambda d: d[1]),
        ("bge_plus_confidence_lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)), lambda d: np.concatenate([d[0], d[1]], axis=1)),
        ("confidence_hgb", HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0, random_state=42), lambda d: d[1]),
    ]
    results = {"features": FEATURES, "routers": {}}
    for name, model, transform in variants:
        key, value = evaluate(name, model, transform(train), transform(val), transform(test), train, val, test, cfg)
        results["routers"][key] = value
    write_json(args.output, results); print(args.output)


if __name__ == "__main__":
    main()
