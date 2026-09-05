"""Collect option-probability confidence features for viable MMLU Lower models."""

from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path

import torch

from .common import read_jsonl, write_jsonl
from .inference import format_prompt, load_model
from .model_registry import get_model_spec
from .run_mmlu_logit_screening import SYSTEM, option_token_ids
from .task_harness import adapt_row


MODELS = ("qwen_1_5b", "smollm2_1_7b")


def collect(model_key: str, input_path: str, output_path: str, batch_size: int = 8, limit: int = 200) -> None:
    spec = get_model_spec(model_key)
    examples = [adapt_row(row, "multiple_choice") for row in read_jsonl(input_path)[:limit]]
    tokenizer, model = load_model(spec.model_id, quantize_4bit=False)
    candidate_ids = option_token_ids(tokenizer)
    rows = []
    started = time.perf_counter()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = [format_prompt(tokenizer, item.prompt, SYSTEM) + "Final answer:" for item in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            logits = model(**encoded).logits[:, -1, candidate_ids].float()
            probabilities = torch.softmax(logits, dim=-1)
        for item, item_logits, item_probabilities in zip(batch, logits.cpu(), probabilities.cpu()):
            probs = item_probabilities.tolist()
            order = sorted(probs, reverse=True)
            prediction = int(item_probabilities.argmax().item())
            reference = int(item.reference)
            entropy = -sum(value * math.log(max(value, 1e-12)) for value in probs) / math.log(4.0)
            rows.append({
                "id": item.id, "model": model_key, "correct": prediction == reference,
                "prediction": "ABCD"[prediction], "reference": "ABCD"[reference],
                "prob_a": probs[0], "prob_b": probs[1], "prob_c": probs[2], "prob_d": probs[3],
                "max_probability": order[0], "probability_margin": order[0] - order[1],
                "normalized_entropy": entropy, "logit_spread": float(item_logits.max() - item_logits.min()),
            })
        print(f"{model_key}: {min(start + batch_size, len(examples))}/{len(examples)}", flush=True)
    write_jsonl(output_path, rows)
    print(f"{model_key}: {len(rows)} rows in {time.perf_counter() - started:.2f}s -> {output_path}")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/data/mmlu_validation_200.jsonl")
    parser.add_argument("--output-dir", default="artifacts/confidence/mmlu_logit")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    for model in MODELS:
        target = Path(args.output_dir) / f"{model}.jsonl"
        if target.exists():
            print(f"SKIP {model}: {target}")
            continue
        collect(model, args.input, str(target), args.batch_size, args.limit)


if __name__ == "__main__":
    main()
