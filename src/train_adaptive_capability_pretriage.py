"""Train a capability-aware BGE pre-router and freeze a two-stage policy."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import AutoTokenizer

from .common import load_config, read_jsonl, set_seed, write_json
from .run_confidence_router import FEATURES
from .select_pretriage_cascade import metrics
from .train_bge_routing_encoder import RoutingBGE, tokenize
from .train_contrastive_router import supervised_contrastive


SOURCES = (
    (Path("artifacts"), "router_train", "upper_7b_concise"),
    (Path("artifacts/fresh_recertification"), "recertification", "upper_concise"),
    (Path("artifacts/reserved_certification"), "certification", "upper_concise"),
)


def keyed(path: Path):
    return {row["id"]: row for row in read_jsonl(path)}


def load_training_rows():
    questions, confidence, lower, upper = [], [], [], []
    for root, split, upper_name in SOURCES:
        data = read_jsonl(root / "data" / f"{split}.jsonl")
        low = keyed(root / "inference" / "lower_concise" / f"{split}.jsonl")
        high = keyed(root / "inference" / upper_name / f"{split}.jsonl")
        conf = keyed(root / "confidence" / "lower_concise" / f"{split}.jsonl")
        for row in data:
            row_id = row["id"]
            questions.append(row["question"])
            confidence.append([float(conf[row_id][name]) for name in FEATURES])
            lower.append(bool(low[row_id]["correct"]))
            upper.append(bool(high[row_id]["correct"]))
    return questions, np.asarray(confidence, dtype=np.float32), np.asarray(lower), np.asarray(upper)


def predict_bge(model, tokens, indices, device, batch_size=64):
    probabilities = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            _, logits = model(tokens["input_ids"][batch].to(device), tokens["attention_mask"][batch].to(device))
            probabilities.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
    return torch.cat(probabilities).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--unsafe-cap", type=float, default=0.03)
    parser.add_argument("--output-root", default="artifacts/capability_pretriage")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    questions, confidence, lower, upper = load_training_rows()
    indices = np.arange(len(lower))
    train_idx, validation_idx = train_test_split(
        indices, test_size=0.20, random_state=int(cfg["seed"]), stratify=lower.astype(int)
    )
    ccfg = cfg["contrastive"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["embedding"])
    tokens = tokenize(tokenizer, questions, int(ccfg["max_length"]))
    model = RoutingBGE(
        cfg["models"]["embedding"], int(ccfg["projection_dim"]),
        int(ccfg["hidden_dim"]), int(ccfg["encoder_unfrozen_layers"]),
    ).to(device)
    labels = torch.from_numpy(lower.astype(np.int64))
    dataset = TensorDataset(tokens["input_ids"][train_idx], tokens["attention_mask"][train_idx], labels[train_idx])
    counts = torch.bincount(labels[train_idx], minlength=2).float()
    sampler = WeightedRandomSampler((1.0 / counts.clamp_min(1))[labels[train_idx]], len(train_idx), replacement=True)
    loader = DataLoader(dataset, batch_size=int(ccfg["encoder_batch_size"]), sampler=sampler)
    encoder_params = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    head_params = list(model.projector.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": float(ccfg["encoder_learning_rate"])},
        {"params": head_params, "lr": float(ccfg["learning_rate"])},
    ], weight_decay=1e-4)
    best_loss, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for input_ids, attention_mask, batch_labels in loader:
            z, logits = model(input_ids.to(device), attention_mask.to(device))
            batch_labels = batch_labels.to(device)
            ce = F.cross_entropy(logits, batch_labels)
            contrastive = supervised_contrastive(z, batch_labels, float(ccfg["temperature"]))
            loss = ce + 0.2 * contrastive
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.inference_mode():
            _, logits = model(
                tokens["input_ids"][validation_idx].to(device),
                tokens["attention_mask"][validation_idx].to(device),
            )
            val_loss = float(F.cross_entropy(logits, labels[validation_idx].to(device)).item())
            val_probability = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_ce": val_loss,
            "validation_auc": float(roc_auc_score(lower[validation_idx].astype(int), val_probability)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    pre_probability = predict_bge(model, tokens, validation_idx, device)
    post = ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=8, class_weight="balanced",
        max_features="sqrt", random_state=42, n_jobs=-1,
    ).fit(confidence[train_idx], lower[train_idx].astype(int))
    post_probability = post.predict_proba(confidence[validation_idx])[:, 1]
    quality_target = float(cfg["router"]["quality_floor"]) * float(upper[validation_idx].mean())
    candidates = []
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    for pre_threshold in grid:
        for post_threshold in grid:
            row = metrics(
                pre_probability, post_probability, lower[validation_idx], upper[validation_idx],
                float(pre_threshold), float(post_threshold), cfg,
            )
            if row["accuracy"] >= quality_target and row["unsafe_rate"] <= args.unsafe_cap:
                candidates.append(row)
    selected = min(candidates, key=lambda row: (row["normalized_cost"], -row["accuracy"])) if candidates else None
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "model_id": cfg["models"]["embedding"],
        "projection_dim": int(ccfg["projection_dim"]),
        "hidden_dim": int(ccfg["hidden_dim"]),
        "unfrozen_layers": int(ccfg["encoder_unfrozen_layers"]),
        "train_indices": train_idx,
        "validation_indices": validation_idx,
    }, output / "capability_router.pt")
    payload = {
        "protocol": "GSM8K-only training and validation; ASDiv untouched; CE + 0.2 supervised contrastive",
        "total_rows": len(lower),
        "train_rows": len(train_idx),
        "validation_rows": len(validation_idx),
        "unsafe_cap": args.unsafe_cap,
        "history": history,
        "best_validation_ce": best_loss,
        "best_validation_auc": float(roc_auc_score(lower[validation_idx].astype(int), pre_probability)),
        "quality_target": quality_target,
        "selected_policy": selected,
        "advance_to_asdiv": bool(selected and selected["normalized_cost"] <= 0.90),
    }
    write_json(output / "training_result.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
