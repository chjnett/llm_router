"""Select a seed-ensemble capability policy under an exact risk bound."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from .common import load_config, write_json
from .run_risk_bound_calibration import binomial_upper
from .select_pretriage_cascade import metrics
from .train_adaptive_capability_pretriage import load_training_rows, predict_bge
from .train_bge_routing_encoder import RoutingBGE, tokenize


DEFAULT_MODEL_ROOTS = [
    "artifacts/capability_pretriage",
    "artifacts/capability_pretriage_seed43",
    "artifacts/capability_pretriage_seed44",
]


def load_model(checkpoint: dict, device: torch.device) -> RoutingBGE:
    model = RoutingBGE(
        checkpoint["model_id"], checkpoint["projection_dim"],
        checkpoint["hidden_dim"], checkpoint["unfrozen_layers"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def ensemble_probability(
    checkpoints: list[dict], tokens: dict[str, torch.Tensor], indices: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, list[np.ndarray]]:
    per_seed = []
    for checkpoint in checkpoints:
        model = load_model(checkpoint, device)
        per_seed.append(predict_bge(model, tokens, indices, device))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.mean(per_seed, axis=0), per_seed


def exact_unsafe_count(
    pre_probability: np.ndarray, post_probability: np.ndarray, lower: np.ndarray,
    pre_threshold: float, post_threshold: float,
) -> int:
    accept_lower = (pre_probability >= pre_threshold) & (post_probability >= post_threshold)
    return int((accept_lower & ~lower).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--model-roots", nargs="+", default=DEFAULT_MODEL_ROOTS)
    parser.add_argument("--risk-limit", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="artifacts/capability_ensemble/risk_bound_selection.json",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    checkpoints = [
        torch.load(Path(root) / "capability_router.pt", map_location="cpu", weights_only=False)
        for root in args.model_roots
    ]
    train_indices = np.asarray(checkpoints[0]["train_indices"])
    validation_indices = np.asarray(checkpoints[0]["validation_indices"])
    if any(
        not np.array_equal(validation_indices, np.asarray(checkpoint["validation_indices"]))
        for checkpoint in checkpoints[1:]
    ):
        raise ValueError("all ensemble members must use the same fixed validation split")

    questions, confidence, lower, upper = load_training_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoints[0]["model_id"])
    tokens = tokenize(
        tokenizer, questions, int(cfg["contrastive"]["max_length"])
    )
    pre_probability, per_seed = ensemble_probability(
        checkpoints, tokens, validation_indices, device
    )
    post_model = ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=8, class_weight="balanced",
        max_features="sqrt", random_state=42, n_jobs=-1,
    ).fit(confidence[train_indices], lower[train_indices].astype(int))
    post_probability = post_model.predict_proba(confidence[validation_indices])[:, 1]
    validation_lower = lower[validation_indices]
    validation_upper = upper[validation_indices]
    quality_target = float(cfg["router"]["quality_floor"]) * float(validation_upper.mean())

    candidates = []
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    for pre_threshold in grid:
        for post_threshold in grid:
            row = metrics(
                pre_probability, post_probability, validation_lower, validation_upper,
                float(pre_threshold), float(post_threshold), cfg,
            )
            unsafe_count = exact_unsafe_count(
                pre_probability, post_probability, validation_lower,
                float(pre_threshold), float(post_threshold),
            )
            risk_upper = binomial_upper(unsafe_count, len(validation_lower), 0.05)
            row.update({"unsafe_count": unsafe_count, "unsafe_upper_bound_95": risk_upper})
            if row["accuracy"] >= quality_target and risk_upper <= args.risk_limit:
                candidates.append(row)
    selected = min(
        candidates,
        key=lambda row: (row["normalized_cost"], -row["accuracy"]),
    ) if candidates else None
    payload = {
        "protocol": "fixed GSM8K validation; mean probability over seeds 42/43/44; exact one-sided 95% unsafe-risk bound constrained during selection",
        "ensemble_size": len(checkpoints),
        "model_roots": args.model_roots,
        "validation_count": len(validation_indices),
        "quality_target": quality_target,
        "risk_limit": args.risk_limit,
        "per_seed_validation_auc": [
            float(roc_auc_score(validation_lower.astype(int), probability))
            for probability in per_seed
        ],
        "ensemble_validation_auc": float(
            roc_auc_score(validation_lower.astype(int), pre_probability)
        ),
        "feasible_policy_count": len(candidates),
        "selected_policy": selected,
        "promotion_gate": bool(
            selected and selected["normalized_cost"] <= 0.90
        ),
    }
    write_json(args.output, payload)
    print(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
