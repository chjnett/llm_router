from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from .common import load_config, read_jsonl, write_json
from .metrics import routing_labels, select_operating_point, system_metrics


def load_condition(name: str, upper_name: str, split: str):
    lower_rows = {row["id"]: row for row in read_jsonl(Path("artifacts/inference") / name / f"{split}.jsonl")}
    upper_rows = {row["id"]: row for row in read_jsonl(Path("artifacts/inference") / upper_name / f"{split}.jsonl")}
    embeddings = np.load(Path("artifacts/embeddings") / f"{split}.npz")
    ids = embeddings["ids"].tolist()
    return (
        embeddings["embeddings"],
        np.array([lower_rows[row_id]["correct"] for row_id in ids]),
        np.array([upper_rows[row_id]["correct"] for row_id in ids]),
    )


def fit_lr(data, seed):
    x, lower, upper = data
    labels, eligible = routing_labels(lower, upper)
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(x[eligible], labels[eligible])


def fit_knn(data, k):
    x, lower, upper = data
    labels, eligible = routing_labels(lower, upper)
    return KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(x[eligible], labels[eligible])


def curve(probabilities, data, thresholds, costs):
    _, lower, upper = data
    return [
        {"threshold": threshold, **system_metrics(probabilities >= threshold, lower, upper, costs["lower"], costs["upper"])}
        for threshold in thresholds
    ]


def choose(model, validation, cfg):
    probabilities = model.predict_proba(validation[0])[:, 1]
    return select_operating_point(
        curve(probabilities, validation, cfg["router"]["thresholds"], cfg["cost"]),
        float(validation[2].mean()),
        cfg["router"]["quality_floor"],
    )


def evaluate(model, threshold, test, costs):
    route = model.predict_proba(test[0])[:, 1] >= threshold
    return system_metrics(route, test[1], test[2], costs["lower"], costs["upper"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--lower-name", default="lower")
    parser.add_argument("--upper-name", default="upper")
    parser.add_argument("--improved", default="cluster_seed_42")
    parser.add_argument("--output", default="artifacts/results/adaptation_cluster_seed_42.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    old = {split: load_condition(args.lower_name, args.upper_name, split) for split in ("router_train", "validation", "test")}
    new = {split: load_condition(args.improved, args.upper_name, split) for split in ("router_train", "validation", "test")}
    results = {"condition": args.improved, "label_shift": {}, "lr": {}, "knn": {}}
    for split in ("router_train", "validation", "test"):
        old_labels, old_eligible = routing_labels(old[split][1], old[split][2])
        new_labels, new_eligible = routing_labels(new[split][1], new[split][2])
        common = old_eligible & new_eligible
        results["label_shift"][split] = {
            "all_examples": float((old[split][1] != new[split][1]).mean()),
            "common_eligible": float((old_labels[common] != new_labels[common]).mean()),
            "lower_accuracy_before": float(old[split][1].mean()),
            "lower_accuracy_after": float(new[split][1].mean()),
        }
    old_lr = fit_lr(old["router_train"], cfg["seed"])
    old_lr_point = choose(old_lr, old["validation"], cfg)
    new_lr = fit_lr(new["router_train"], cfg["seed"])
    new_lr_point = choose(new_lr, new["validation"], cfg)
    results["lr"] = {
        "old_router_old_lower": evaluate(old_lr, old_lr_point["threshold"], old["test"], cfg["cost"]),
        "no_adaptation_new_lower": evaluate(old_lr, old_lr_point["threshold"], new["test"], cfg["cost"]),
        "retrain_new_lower": evaluate(new_lr, new_lr_point["threshold"], new["test"], cfg["cost"]),
        "old_threshold": old_lr_point["threshold"],
        "new_threshold": new_lr_point["threshold"],
    }
    candidates = []
    for k in cfg["router"]["k_values"]:
        model = fit_knn(old["router_train"], k)
        point = choose(model, old["validation"], cfg)
        candidates.append((point["normalized_cost"], -point["task_accuracy"], k, model, point))
    _, _, old_k, old_knn, old_point = min(candidates)
    candidates = []
    for k in cfg["router"]["k_values"]:
        model = fit_knn(new["router_train"], k)
        point = choose(model, new["validation"], cfg)
        candidates.append((point["normalized_cost"], -point["task_accuracy"], k, model, point))
    _, _, new_k, new_knn, new_point = min(candidates)
    results["knn"] = {
        "old_router_old_lower": evaluate(old_knn, old_point["threshold"], old["test"], cfg["cost"]),
        "no_adaptation_new_lower": evaluate(old_knn, old_point["threshold"], new["test"], cfg["cost"]),
        "pool_rebuild_new_lower": evaluate(new_knn, new_point["threshold"], new["test"], cfg["cost"]),
        "old_k": old_k,
        "new_k": new_k,
        "old_threshold": old_point["threshold"],
        "new_threshold": new_point["threshold"],
    }
    write_json(args.output, results)
    print(args.output)


if __name__ == "__main__":
    main()
