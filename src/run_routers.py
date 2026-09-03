from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from .common import load_config, read_jsonl, write_json
from .metrics import routing_labels, select_operating_point, system_metrics


SPLITS = ("router_train", "validation", "test")


def load_split(split: str, lower_name: str = "lower", upper_name: str = "upper", embedding_dir: str = "artifacts/embeddings"):
    lower_rows = read_jsonl(Path("artifacts/inference") / lower_name / f"{split}.jsonl")
    upper_rows = read_jsonl(Path("artifacts/inference") / upper_name / f"{split}.jsonl")
    embedding_file = np.load(Path(embedding_dir) / f"{split}.npz")
    lower = {row["id"]: row for row in lower_rows}
    upper = {row["id"]: row for row in upper_rows}
    ids = embedding_file["ids"].tolist()
    if set(ids) != set(lower) or set(ids) != set(upper):
        raise RuntimeError(f"ID mismatch in {split}")
    lower_correct = np.array([lower[row_id]["correct"] for row_id in ids])
    upper_correct = np.array([upper[row_id]["correct"] for row_id in ids])
    return ids, embedding_file["embeddings"], lower_correct, upper_correct


def curve(probability_lower, lower, upper, thresholds, costs):
    return [
        {
            "threshold": threshold,
            **system_metrics(probability_lower >= threshold, lower, upper, costs["lower"], costs["upper"]),
        }
        for threshold in thresholds
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--output", default="artifacts/results/baseline_routers.json")
    parser.add_argument("--lower-name", default="lower")
    parser.add_argument("--upper-name", default="upper")
    parser.add_argument("--embedding-dir", default="artifacts/embeddings")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = {split: load_split(split, args.lower_name, args.upper_name, args.embedding_dir) for split in SPLITS}
    _, x_train, lower_train, upper_train = data["router_train"]
    y_train, eligible = routing_labels(lower_train, upper_train)
    x_train, y_train = x_train[eligible], y_train[eligible]
    results = {"baselines": {}, "routers": {}}
    _, _, lower_test, upper_test = data["test"]
    results["baselines"]["always_lower"] = system_metrics(np.ones_like(lower_test), lower_test, upper_test, **{"lower_cost": cfg["cost"]["lower"], "upper_cost": cfg["cost"]["upper"]})
    results["baselines"]["always_upper"] = system_metrics(np.zeros_like(lower_test), lower_test, upper_test, **{"lower_cost": cfg["cost"]["lower"], "upper_cost": cfg["cost"]["upper"]})
    results["baselines"]["oracle"] = system_metrics(lower_test, lower_test, upper_test, **{"lower_cost": cfg["cost"]["lower"], "upper_cost": cfg["cost"]["upper"]})
    for seed in cfg["seeds"]:
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(x_train, y_train)
        _, x_val, lower_val, upper_val = data["validation"]
        val_prob = lr.predict_proba(x_val)[:, 1]
        val_curve = curve(val_prob, lower_val, upper_val, cfg["router"]["thresholds"], cfg["cost"])
        val_point = select_operating_point(val_curve, float(upper_val.mean()), cfg["router"]["quality_floor"])
        test_prob = lr.predict_proba(data["test"][1])[:, 1]
        route = test_prob >= val_point["threshold"]
        y_test, eligible_test = routing_labels(lower_test, upper_test)
        results["routers"][f"lr_seed_{seed}"] = {
            "selected_on_validation": val_point,
            "test": system_metrics(route, lower_test, upper_test, cfg["cost"]["lower"], cfg["cost"]["upper"]),
            "routing_accuracy_eligible": float(accuracy_score(y_test[eligible_test], route[eligible_test])),
            "routing_auc_eligible": float(roc_auc_score(y_test[eligible_test], test_prob[eligible_test])),
        }
    for k in cfg["router"]["k_values"]:
        knn = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="cosine").fit(x_train, y_train)
        val_prob = knn.predict_proba(data["validation"][1])[:, 1]
        val_curve = curve(val_prob, data["validation"][2], data["validation"][3], cfg["router"]["thresholds"], cfg["cost"])
        val_point = select_operating_point(val_curve, float(data["validation"][3].mean()), cfg["router"]["quality_floor"])
        test_prob = knn.predict_proba(data["test"][1])[:, 1]
        route = test_prob >= val_point["threshold"]
        results["routers"][f"knn_k_{k}"] = {
            "selected_on_validation": val_point,
            "test": system_metrics(route, lower_test, upper_test, cfg["cost"]["lower"], cfg["cost"]["upper"]),
        }
    write_json(args.output, results)
    print(args.output)


if __name__ == "__main__":
    main()
