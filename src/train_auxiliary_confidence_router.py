"""Train C4 confidence heads with BCE, pairwise ranking, and Brier losses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .common import load_config, read_jsonl, set_seed, write_json
from .run_confidence_router import FEATURES
from .run_selective_consistency import cascade


class ConfidenceHead(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(len(FEATURES), width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def keyed(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in read_jsonl(path)}


def load_split(data_root: Path, inference_root: Path, confidence_root: Path, split: str, lower_name: str, upper_name: str, alternate_name: str):
    data = read_jsonl(data_root / f"{split}.jsonl")
    ids = [row["id"] for row in data]
    confidence = keyed(confidence_root / lower_name / f"{split}.jsonl")
    lower = keyed(inference_root / lower_name / f"{split}.jsonl")
    upper = keyed(inference_root / upper_name / f"{split}.jsonl")
    alternate = keyed(inference_root / alternate_name / f"{split}.jsonl")
    features = np.asarray([[float(confidence[row_id][name]) for name in FEATURES] for row_id in ids], dtype=np.float32)
    lower_correct = np.asarray([lower[row_id]["correct"] for row_id in ids], dtype=bool)
    upper_correct = np.asarray([upper[row_id]["correct"] for row_id in ids], dtype=bool)
    agreement = np.asarray([
        lower[row_id].get("predicted_number") is not None
        and lower[row_id].get("predicted_number") == alternate[row_id].get("predicted_number")
        for row_id in ids
    ], dtype=bool)
    return features, lower_correct, upper_correct, agreement


def train_head(train_x, train_y, seed, epochs, learning_rate):
    set_seed(seed)
    model = ConfidenceHead(32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    x = torch.tensor(train_x, dtype=torch.float32)
    y = torch.tensor(train_y.astype(np.float32))
    bce_fn = torch.nn.BCEWithLogitsLoss()
    positive = torch.nonzero(y > 0.5, as_tuple=False).flatten()
    negative = torch.nonzero(y < 0.5, as_tuple=False).flatten()
    generator = torch.Generator().manual_seed(seed)
    last = {}
    for epoch in range(epochs):
        logits = model(x)
        probability = torch.sigmoid(logits)
        bce = bce_fn(logits, y)
        pair_count = min(len(positive), len(negative))
        pos = positive[torch.randperm(len(positive), generator=generator)[:pair_count]]
        neg = negative[torch.randperm(len(negative), generator=generator)[:pair_count]]
        ranking = torch.nn.functional.softplus(-(logits[pos] - logits[neg])).mean()
        brier = torch.mean((probability - y) ** 2)
        loss = bce + 0.2 * ranking + 0.1 * brier
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last = {"epoch": epoch + 1, "total": float(loss.item()), "bce": float(bce.item()), "ranking": float(ranking.item()), "brier": float(brier.item())}
    return model, last


def predict(model, values):
    with torch.inference_mode():
        return torch.sigmoid(model(torch.tensor(values, dtype=torch.float32))).numpy()


def ece(probability, labels, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        selected = (probability >= left) & (probability < right if right < 1.0 else probability <= right)
        if selected.any():
            total += selected.mean() * abs(probability[selected].mean() - labels[selected].mean())
    return float(total)


def select_policy(probability, agreement, lower, upper, cfg):
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    target = cfg["router"]["quality_floor"] * float(upper.mean()) + 0.02
    candidates = []
    for low in grid:
        for high in grid:
            if low > high:
                continue
            row = cascade(probability, agreement, lower, upper, low, high, cfg)
            candidates.append({"low_threshold": float(low), "high_threshold": float(high), **row})
    feasible = [row for row in candidates if row["task_accuracy"] >= target and row["unsafe_routing_rate_all"] <= 0.04]
    return min(feasible, key=lambda row: (row["normalized_cascade_cost"], -row["task_accuracy"])) if feasible else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--pilot-root", default="artifacts")
    parser.add_argument("--fresh-root", default="artifacts/fresh_holdout")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--output", default="artifacts/results/auxiliary_confidence_router.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    pilot, fresh = Path(args.pilot_root), Path(args.fresh_root)
    train = load_split(pilot / "data", pilot / "inference", pilot / "confidence", "router_train", "lower_concise", "upper_7b_concise", "lower")
    calibration = load_split(pilot / "data", pilot / "inference", pilot / "confidence", "validation", "lower_concise", "upper_7b_concise", "lower")
    fresh_validation = load_split(fresh / "data", fresh / "inference", fresh / "confidence", "validation", "lower_concise", "upper_concise", "lower_second_prompt")
    fresh_test = load_split(fresh / "data", fresh / "inference", fresh / "confidence", "test", "lower_concise", "upper_concise", "lower_second_prompt")
    scaler = StandardScaler().fit(train[0])
    train_x = scaler.transform(train[0]).astype(np.float32)
    results = {"objective": "BCE + 0.2 pairwise ranking + 0.1 Brier", "selection_data": "pilot validation only", "fresh_test_used_for_tuning": False, "seeds": {}}
    for seed in cfg.get("seeds", [42, 43, 44]):
        model, losses = train_head(train_x, train[1], seed, args.epochs, args.learning_rate)
        calibration_probability = predict(model, scaler.transform(calibration[0]).astype(np.float32))
        selected = select_policy(calibration_probability, calibration[3], calibration[1], calibration[2], cfg)
        item = {"final_train_loss": losses, "selected_on_pilot_validation": selected}
        if selected is not None:
            for name, split in (("fresh_validation", fresh_validation), ("fresh_test", fresh_test)):
                probability = predict(model, scaler.transform(split[0]).astype(np.float32))
                metrics = cascade(probability, split[3], split[1], split[2], selected["low_threshold"], selected["high_threshold"], cfg)
                item[name] = {
                    **metrics,
                    "lower_success_auc": float(roc_auc_score(split[1].astype(int), probability)),
                    "ece_10bin": ece(probability, split[1].astype(float)),
                    "always_upper_accuracy": float(split[2].mean()),
                }
                item[name]["gate_pass"] = (
                    metrics["task_accuracy"] >= cfg["router"]["quality_floor"] * float(split[2].mean())
                    and metrics["normalized_cascade_cost"] <= 0.90
                    and metrics["unsafe_routing_rate_all"] <= 0.05
                )
        results["seeds"][str(seed)] = item
    valid = [item["fresh_test"] for item in results["seeds"].values() if "fresh_test" in item]
    results["fresh_test_mean"] = {
        key: float(np.mean([item[key] for item in valid]))
        for key in ("task_accuracy", "normalized_cascade_cost", "unsafe_routing_rate_all", "lower_success_auc", "ece_10bin")
    } if valid else None
    results["fresh_test_all_seeds_gate_pass"] = bool(valid) and all(item["gate_pass"] for item in valid)
    write_json(args.output, results)
    print(args.output)


if __name__ == "__main__":
    main()
