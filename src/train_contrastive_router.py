"""Train a routing-specific contrastive projection on frozen BGE embeddings.

This conservative phase-1 experiment learns a small projection head before
unfreezing BGE itself. It isolates whether routing supervision improves the
geometry without adding another large model download or overfitting 1k items.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .common import load_config, read_jsonl, set_seed, write_json
from .metrics import routing_labels


class RoutingProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, output_dim)
        )
        self.classifier = nn.Linear(output_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = F.normalize(self.projector(x), dim=-1)
        return z, self.classifier(z)


def supervised_contrastive(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    semantic: torch.Tensor | None = None,
    hard_threshold: float | None = None,
    hard_weight: float = 1.0,
) -> torch.Tensor:
    similarity = z @ z.T / temperature
    identity = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positives = labels[:, None].eq(labels[None, :]) & ~identity
    logits = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    denominator_weight = torch.ones_like(logits)
    if semantic is not None and hard_threshold is not None and hard_weight > 1:
        semantic_similarity = F.normalize(semantic, dim=-1) @ F.normalize(semantic, dim=-1).T
        hard = labels[:, None].ne(labels[None, :]) & (semantic_similarity >= hard_threshold)
        denominator_weight = torch.where(hard, torch.full_like(logits, hard_weight), denominator_weight)
    exp_logits = torch.exp(logits) * ~identity * denominator_weight
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    valid = positives.sum(dim=1) > 0
    mean_log_prob = (log_prob * positives).sum(dim=1) / positives.sum(dim=1).clamp_min(1)
    return -mean_log_prob[valid].mean()


def aligned(split: str, lower_name: str, upper_name: str, embedding_dir: Path):
    packed = np.load(embedding_dir / f"{split}.npz")
    ids = packed["ids"].tolist()
    lower = {r["id"]: r for r in read_jsonl(Path("artifacts/inference") / lower_name / f"{split}.jsonl")}
    upper = {r["id"]: r for r in read_jsonl(Path("artifacts/inference") / upper_name / f"{split}.jsonl")}
    low = np.array([lower[i]["correct"] for i in ids])
    high = np.array([upper[i]["correct"] for i in ids])
    labels, eligible = routing_labels(low, high)
    return ids, packed["embeddings"].astype("float32"), labels.astype("int64"), eligible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--lower-name", default="lower_concise")
    parser.add_argument("--upper-name", default="upper_7b_concise")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--hard-negatives", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ccfg = cfg["contrastive"]
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path("artifacts/embeddings")
    suffix = "contrastive_hard" if args.hard_negatives else "contrastive"
    output = Path(args.output_dir or f"artifacts/embeddings_{suffix}_seed_{args.seed}")
    output.mkdir(parents=True, exist_ok=True)

    train = aligned("router_train", args.lower_name, args.upper_name, base_dir)
    val = aligned("validation", args.lower_name, args.upper_name, base_dir)
    x_train = torch.from_numpy(train[1][train[3]])
    y_train = torch.from_numpy(train[2][train[3]])
    counts = torch.bincount(y_train, minlength=2).float()
    weights = (1.0 / counts.clamp_min(1))[y_train]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=ccfg["batch_size"], sampler=sampler)
    model = RoutingProjection(x_train.shape[1], ccfg["hidden_dim"], ccfg["projection_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ccfg["learning_rate"], weight_decay=1e-4)
    x_val = torch.from_numpy(val[1][val[3]]).to(device)
    y_val = torch.from_numpy(val[2][val[3]]).to(device)
    hard_threshold = None
    if args.hard_negatives:
        normalized = F.normalize(x_train, dim=-1)
        semantic_similarity = normalized @ normalized.T
        opposite = y_train[:, None].ne(y_train[None, :])
        hard_threshold = torch.quantile(semantic_similarity[opposite], ccfg["hard_negative_quantile"]).item()
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(ccfg["epochs"]):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z, logits = model(x)
            loss = supervised_contrastive(z, y, ccfg["temperature"], x if args.hard_negatives else None, hard_threshold, ccfg["hard_negative_weight"]) + ccfg["classification_weight"] * F.cross_entropy(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.inference_mode():
            z, logits = model(x_val)
            val_loss = supervised_contrastive(z, y_val, ccfg["temperature"], x_val if args.hard_negatives else None, hard_threshold, ccfg["hard_negative_weight"]) + ccfg["classification_weight"] * F.cross_entropy(logits, y_val)
            val_acc = (logits.argmax(1) == y_val).float().mean().item()
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss.item(), "val_accuracy": val_acc})
        print(json.dumps(history[-1]), flush=True)
        if val_loss.item() < best - 1e-4:
            best, best_state, stale = val_loss.item(), copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= ccfg["patience"]:
                break
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "input_dim": x_train.shape[1], "config": ccfg}, output / "projection.pt")
    for split in ("router_train", "distill_train", "validation", "test"):
        packed = np.load(base_dir / f"{split}.npz")
        x = torch.from_numpy(packed["embeddings"].astype("float32")).to(device)
        model.eval()
        with torch.inference_mode():
            z, _ = model(x)
        np.savez_compressed(output / f"{split}.npz", ids=packed["ids"], embeddings=z.cpu().numpy())
    write_json(output / "training_metrics.json", {"seed": args.seed, "eligible_train": int(train[3].sum()), "best_val_loss": best, "hard_negatives": args.hard_negatives, "hard_threshold": hard_threshold, "history": history})
    print(output)


if __name__ == "__main__":
    main()
