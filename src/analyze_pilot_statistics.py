"""Paired significance and label-stability analysis for the KSC pilot."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .common import read_jsonl, write_json


def correctness(name: str, split: str = "test") -> tuple[list[str], np.ndarray]:
    rows = read_jsonl(Path("artifacts/inference") / name / f"{split}.jsonl")
    return [row["id"] for row in rows], np.asarray([row["correct"] for row in rows], dtype=bool)


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(a_only, b_only) + 1)) / (2**n)
        p = min(1.0, 2 * tail)
    return {"baseline_only_correct": a_only, "candidate_only_correct": b_only, "discordant": n, "p_value_two_sided": p}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, draws: int) -> dict:
    delta = b.astype(float) - a.astype(float)
    indices = rng.integers(0, len(delta), size=(draws, len(delta)))
    samples = delta[indices].mean(axis=1)
    return {
        "delta_accuracy": float(delta.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "probability_improvement": float((samples > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/results/pilot_statistics.json")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    ids, baseline = correctness("lower_concise")
    _, upper = correctness("upper_7b_concise")
    names = [f"adapter_{method}_seed_{seed}" for method in ("random", "cluster") for seed in (42, 43, 44)]
    comparisons = {}
    matrix = []
    for name in names:
        candidate_ids, candidate = correctness(name)
        if candidate_ids != ids:
            raise RuntimeError(f"ID order mismatch: {name}")
        matrix.append(candidate)
        comparisons[name] = {
            "baseline_accuracy": float(baseline.mean()),
            "candidate_accuracy": float(candidate.mean()),
            "bootstrap": paired_bootstrap(baseline, candidate, rng, args.draws),
            "mcnemar": exact_mcnemar(baseline, candidate),
        }
    labels = np.stack(matrix).astype(float)
    probability = labels.mean(axis=0)
    eps = 1e-12
    entropy = -(probability * np.log2(probability + eps) + (1 - probability) * np.log2(1 - probability + eps))
    eligible = baseline | upper
    routing_base = baseline[eligible]
    routing_adapters = labels[:, eligible]
    results = {
        "n_test": len(ids),
        "bootstrap_draws": args.draws,
        "comparisons": comparisons,
        "label_stability": {
            "all_six_unanimous_rate": float(((labels.sum(axis=0) == 0) | (labels.sum(axis=0) == len(names))).mean()),
            "mean_binary_entropy": float(entropy.mean()),
            "items_with_three_vs_three_split": int((labels.sum(axis=0) == 3).sum()),
            "eligible_examples": int(eligible.sum()),
            "mean_adapter_disagreement_with_base_on_eligible": float((routing_adapters != routing_base).mean()),
        },
    }
    write_json(args.output, results)
    print(args.output)


if __name__ == "__main__":
    main()
