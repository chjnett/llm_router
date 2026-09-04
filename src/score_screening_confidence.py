"""Extract output-confidence features for task-neutral screening predictions."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from .common import read_jsonl, write_jsonl

import torch

from .inference import format_prompt, load_model
from .model_registry import MODEL_REGISTRY, get_model_spec
from .task_harness import adapt_row, system_prompt


FEATURES = [
    "mean_logprob", "min_logprob", "std_logprob", "mean_entropy", "max_entropy",
    "mean_margin", "min_margin", "completion_tokens", "completion_chars",
    "number_count", "has_final_answer",
]


def score_batch(tokenizer, model, examples, predictions, prompt_mode: str) -> list[dict]:
    prompts = [format_prompt(tokenizer, item.prompt, system_prompt(item.task_type, prompt_mode)) for item in examples]
    completions = [prediction["prediction"] for prediction in predictions]
    prompt_ids = [tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
    full = [prompt + completion for prompt, completion in zip(prompts, completions)]
    tokenizer.padding_side = "right"
    encoded = tokenizer(full, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        logits = model(**encoded).logits.float()
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    probabilities = torch.softmax(logits[:, :-1], dim=-1)
    targets = encoded["input_ids"][:, 1:]
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(probabilities * log_probs).sum(-1)
    top2 = probabilities.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    output = []
    for index, (item, prediction, ids, completion) in enumerate(zip(examples, predictions, prompt_ids, completions)):
        start = max(len(ids) - 1, 0)
        length = int(encoded["attention_mask"][index].sum().item())
        end = max(length - 1, start + 1)
        lp = token_logp[index, start:end]
        ent = entropy[index, start:end]
        mar = margin[index, start:end]
        values = (lp.mean().item(), ent.mean().item(), mar.mean().item())
        output.append({
            "id": item.id,
            "model_key": prediction["model_key"],
            "prompt_mode": prompt_mode,
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
            "has_final_answer": bool(re.search(r"Final answer\s*:", completion, re.I)),
            "finite": all(math.isfinite(value) for value in values),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), required=True)
    parser.add_argument("--input", default="artifacts/data/validation.jsonl")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-mode", default="answer_only", choices=["task", "micro_reasoning", "answer_only"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for confidence extraction")
    spec = get_model_spec(args.model_key)
    raw_rows = read_jsonl(args.input)
    examples = [adapt_row(row, "numeric") for row in raw_rows]
    by_id = {row["id"]: row for row in read_jsonl(args.predictions)}
    missing = [item.id for item in examples if item.id not in by_id]
    if missing:
        raise ValueError(f"Missing {len(missing)} predictions; first={missing[0]}")
    tokenizer, model = load_model(spec.model_id, spec.default_4bit and not args.no_4bit)
    output = []
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start : start + args.batch_size]
        output.extend(score_batch(tokenizer, model, batch, [by_id[item.id] for item in batch], args.prompt_mode))
        write_jsonl(args.output, output)
        print(f"confidence/{spec.key}: {len(output)}/{len(examples)}", flush=True)


if __name__ == "__main__":
    main()

