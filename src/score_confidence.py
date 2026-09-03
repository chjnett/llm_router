"""Teacher-force cached Lower outputs to extract routing confidence features."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import torch

from .common import load_config, read_jsonl, write_jsonl
from .inference import CONCISE_SYSTEM_PROMPT, format_prompt, load_model


def score_batch(tokenizer, model, rows: list[dict], predictions: list[dict]) -> list[dict]:
    prompts = [format_prompt(tokenizer, row["question"], CONCISE_SYSTEM_PROMPT) for row in rows]
    completions = [prediction["prediction"] for prediction in predictions]
    prompt_ids = [tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
    full = [prompt + completion for prompt, completion in zip(prompts, completions)]
    tokenizer.padding_side = "right"
    encoded = tokenizer(full, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        logits = model(**encoded).logits.float()
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    probs = torch.softmax(logits[:, :-1], dim=-1)
    targets = encoded["input_ids"][:, 1:]
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(probs * log_probs).sum(-1)
    top2 = probs.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    output = []
    for index, (row, prediction, pids, completion) in enumerate(zip(rows, predictions, prompt_ids, completions)):
        start = max(len(pids) - 1, 0)
        length = int(encoded["attention_mask"][index].sum().item())
        end = max(length - 1, start + 1)
        lp = token_logp[index, start:end]
        ent = entropy[index, start:end]
        mar = margin[index, start:end]
        output.append({
            "id": row["id"],
            "split": row["split"],
            "correct": prediction["correct"],
            "mean_logprob": float(lp.mean().item()),
            "min_logprob": float(lp.min().item()),
            "std_logprob": float(lp.std(unbiased=False).item()),
            "mean_entropy": float(ent.mean().item()),
            "max_entropy": float(ent.max().item()),
            "mean_margin": float(mar.mean().item()),
            "min_margin": float(mar.min().item()),
            "completion_tokens": int(len(lp)),
            "completion_chars": len(completion),
            "number_count": len(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", completion)),
            "has_final_answer": bool(re.search(r"Final answer\s*:", completion, re.IGNORECASE)),
            "finite": all(math.isfinite(value) for value in (lp.mean().item(), ent.mean().item(), mar.mean().item())),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--prediction-name", default="lower_concise")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--inference-dir", default="artifacts/inference")
    parser.add_argument("--output-dir", default="artifacts/confidence/lower_concise")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split", action="append", dest="splits")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tokenizer, model = load_model(cfg["models"]["lower"], True)
    output_dir = Path(args.output_dir)
    for split in args.splits or ("router_train", "distill_train", "validation", "test"):
        rows = read_jsonl(Path(args.data_dir) / f"{split}.jsonl")
        predictions = read_jsonl(Path(args.inference_dir) / args.prediction_name / f"{split}.jsonl")
        by_id = {row["id"]: row for row in predictions}
        existing_path = output_dir / f"{split}.jsonl"
        existing = read_jsonl(existing_path) if existing_path.exists() else []
        completed = {row["id"] for row in existing}
        pending = [row for row in rows if row["id"] not in completed]
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            existing.extend(score_batch(tokenizer, model, batch, [by_id[row["id"]] for row in batch]))
            write_jsonl(existing_path, existing)
            print(f"confidence/{split}: {len(existing)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
