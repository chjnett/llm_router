"""Fine-tune the last BGE layers for capability-aware routing embeddings."""

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
from transformers import AutoModel, AutoTokenizer

from .common import load_config, read_jsonl, set_seed, write_json
from .train_contrastive_router import aligned, supervised_contrastive


class RoutingBGE(nn.Module):
    def __init__(self, model_id: str, projection_dim: int, hidden_dim: int, unfrozen_layers: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for layer in self.encoder.encoder.layer[-unfrozen_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        dim = self.encoder.config.hidden_size
        self.projector = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, projection_dim))
        self.classifier = nn.Linear(projection_dim, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        z = F.normalize(self.projector(hidden), dim=-1)
        return z, self.classifier(z)


def questions_for(ids: list[str], split: str) -> list[str]:
    rows = {row["id"]: row["question"] for row in read_jsonl(Path("artifacts/data") / f"{split}.jsonl")}
    return [rows[row_id] for row_id in ids]


def tokenize(tokenizer, questions: list[str], max_length: int):
    return tokenizer(questions, padding=True, truncation=True, max_length=max_length, return_tensors="pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--lower-name", default="lower_concise")
    parser.add_argument("--upper-name", default="upper_7b_concise")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config); ccfg = cfg["contrastive"]
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path("artifacts/embeddings")
    output = Path(f"artifacts/embeddings_bge_routing_seed_{args.seed}"); output.mkdir(parents=True, exist_ok=True)
    model_id = cfg["models"]["embedding"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = RoutingBGE(model_id, ccfg["projection_dim"], ccfg["hidden_dim"], ccfg["encoder_unfrozen_layers"]).to(device)

    train = aligned("router_train", args.lower_name, args.upper_name, base_dir)
    val = aligned("validation", args.lower_name, args.upper_name, base_dir)
    train_ids = [row_id for row_id, keep in zip(train[0], train[3]) if keep]
    val_ids = [row_id for row_id, keep in zip(val[0], val[3]) if keep]
    train_tokens = tokenize(tokenizer, questions_for(train_ids, "router_train"), ccfg["max_length"])
    val_tokens = tokenize(tokenizer, questions_for(val_ids, "validation"), ccfg["max_length"])
    y_train = torch.from_numpy(train[2][train[3]])
    y_val = torch.from_numpy(val[2][val[3]])
    semantic_train = torch.from_numpy(train[1][train[3]])
    semantic_val = torch.from_numpy(val[1][val[3]])
    normalized = F.normalize(semantic_train, dim=-1)
    sim = normalized @ normalized.T
    opposite = y_train[:, None].ne(y_train[None, :])
    hard_threshold = torch.quantile(sim[opposite], ccfg["hard_negative_quantile"]).item()
    dataset = TensorDataset(train_tokens["input_ids"], train_tokens["attention_mask"], y_train, semantic_train)
    counts = torch.bincount(y_train, minlength=2).float()
    sampler = WeightedRandomSampler((1 / counts.clamp_min(1))[y_train], len(y_train), replacement=True)
    loader = DataLoader(dataset, batch_size=ccfg["encoder_batch_size"], sampler=sampler)
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.projector.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([{"params": encoder_params, "lr": ccfg["encoder_learning_rate"]}, {"params": head_params, "lr": ccfg["learning_rate"]}], weight_decay=1e-4)
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(ccfg["encoder_epochs"]):
        model.train(); losses = []
        for ids, mask, labels, semantic in loader:
            ids, mask, labels, semantic = ids.to(device), mask.to(device), labels.to(device), semantic.to(device)
            z, logits = model(ids, mask)
            loss = supervised_contrastive(z, labels, ccfg["temperature"], semantic, hard_threshold, ccfg["hard_negative_weight"]) + ccfg["classification_weight"] * F.cross_entropy(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
        model.eval()
        with torch.inference_mode():
            z, logits = model(val_tokens["input_ids"].to(device), val_tokens["attention_mask"].to(device))
            val_loss = supervised_contrastive(z, y_val.to(device), ccfg["temperature"], semantic_val.to(device), hard_threshold, ccfg["hard_negative_weight"]) + ccfg["classification_weight"] * F.cross_entropy(logits, y_val.to(device))
            val_acc = (logits.argmax(1) == y_val.to(device)).float().mean().item()
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss.item(), "val_accuracy": val_acc})
        print(json.dumps(history[-1]), flush=True)
        if val_loss.item() < best - 1e-4:
            best, best_state, stale = val_loss.item(), copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 3: break
    model.load_state_dict(best_state); model.eval()
    trainable = {name: value.cpu() for name, value in best_state.items() if name.startswith("projector") or name.startswith("classifier") or any(f"encoder.layer.{i}." in name for i in range(len(model.encoder.encoder.layer)-ccfg["encoder_unfrozen_layers"], len(model.encoder.encoder.layer)))}
    torch.save({"state_dict": trainable, "config": ccfg, "model_id": model_id}, output / "routing_adapter.pt")
    for split in ("router_train", "distill_train", "validation", "test"):
        packed = np.load(base_dir / f"{split}.npz"); ids = packed["ids"].tolist()
        tokens = tokenize(tokenizer, questions_for(ids, split), ccfg["max_length"])
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(ids), 64):
                z, _ = model(tokens["input_ids"][start:start+64].to(device), tokens["attention_mask"][start:start+64].to(device)); vectors.append(z.cpu())
        np.savez_compressed(output / f"{split}.npz", ids=packed["ids"], embeddings=torch.cat(vectors).numpy())
    write_json(output / "training_metrics.json", {"seed": args.seed, "best_val_loss": best, "hard_threshold": hard_threshold, "history": history})
    print(output)


if __name__ == "__main__":
    main()
