from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from .common import load_config, read_jsonl, write_json, write_jsonl


def balanced_indices(labels: np.ndarray, budget: int, rng: np.random.Generator) -> list[int]:
    groups = {int(label): rng.permutation(np.flatnonzero(labels == label)).tolist() for label in np.unique(labels)}
    chosen: list[int] = []
    while len(chosen) < budget and any(groups.values()):
        for label in sorted(groups):
            if groups[label] and len(chosen) < budget:
                chosen.append(groups[label].pop())
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--lower-name", default="lower")
    parser.add_argument("--upper-name", default="upper")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = read_jsonl("artifacts/data/distill_train.jsonl")
    lower_path = Path("artifacts/inference") / args.lower_name / "distill_train.jsonl"
    lower = {row["id"]: row for row in read_jsonl(lower_path)}
    upper = {row["id"]: row for row in read_jsonl(Path("artifacts/inference") / args.upper_name / "distill_train.jsonl")}
    embedding_file = np.load("artifacts/embeddings/distill_train.npz")
    embeddings = {row_id: vector for row_id, vector in zip(embedding_file["ids"].tolist(), embedding_file["embeddings"])}
    failures = [row for row in data if not lower[row["id"]]["correct"] and upper[row["id"]]["correct"]]
    budget = min(cfg["distillation"]["budget"], len(failures))
    matrix = np.stack([embeddings[row["id"]] for row in failures])
    cluster_count = min(cfg["distillation"]["clusters"], len(failures))
    summary = {"resolvable_failures": len(failures), "budget": budget, "clusters": cluster_count, "seeds": {}}
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        random_selected = rng.choice(len(failures), size=budget, replace=False).tolist()
        labels = KMeans(n_clusters=cluster_count, random_state=seed, n_init=20).fit_predict(matrix)
        cluster_selected = balanced_indices(labels, budget, rng)
        summary["seeds"][str(seed)] = {
            "random_cluster_coverage": int(len(np.unique(labels[random_selected]))),
            "balanced_cluster_coverage": int(len(np.unique(labels[cluster_selected]))),
        }
        for method, selected in (("random", random_selected), ("cluster", cluster_selected)):
            rows = []
            for index in selected:
                source = failures[index]
                rows.append(
                    {
                        "id": source["id"],
                        "question": source["question"],
                        "teacher_response": upper[source["id"]]["prediction"],
                        "cluster": int(labels[index]),
                        "method": method,
                        "seed": seed,
                    }
                )
            write_jsonl(Path("artifacts/distillation") / f"{method}_seed_{seed}.jsonl", rows)
    write_json("artifacts/distillation/selection_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
