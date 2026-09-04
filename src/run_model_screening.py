"""Run one model on task-neutral JSONL while measuring real GPU cost."""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path

from .common import read_jsonl, set_seed, write_json, write_jsonl

import torch

from .inference import format_prompt, load_model
from .model_registry import MODEL_REGISTRY, get_model_spec
from .task_harness import adapt_row, score_prediction, system_prompt


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def summarize(rows: list[dict], elapsed_seconds: float, peak_allocated: int, peak_reserved: int) -> dict:
    latencies = [float(row["latency_ms"]) for row in rows]
    judged = [row for row in rows if row["correct"] is not None]
    parsed = [row for row in judged if row["parsed_answer"] is not None]
    return {
        "items": len(rows),
        "judged_items": len(judged),
        "accuracy": sum(bool(row["correct"]) for row in judged) / len(judged) if judged else None,
        "parse_success_rate": len(parsed) / len(judged) if judged else None,
        "generated_tokens_mean": statistics.mean(row["generated_tokens"] for row in rows) if rows else 0.0,
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "items_per_second": len(rows) / elapsed_seconds if elapsed_seconds else 0.0,
        "peak_vram_allocated_gb": peak_allocated / 1024**3,
        "peak_vram_reserved_gb": peak_reserved / 1024**3,
    }


def validate_rows(raw_rows: list[dict], task_type: str) -> list:
    examples = [adapt_row(row, task_type) for row in raw_rows]
    ids = [example.id for example in examples]
    if any(not value for value in ids):
        raise ValueError("Every screening row must have a non-empty id")
    if len(ids) != len(set(ids)):
        raise ValueError("Screening row ids must be unique")
    for example in examples:
        system_prompt(example.task_type)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--task-type", default="numeric", choices=["numeric", "multiple_choice", "exact_match", "code", "instruction_following"])
    parser.add_argument("--output-dir", default="artifacts/model_screening")
    parser.add_argument("--run-name", help="Output subdirectory; defaults to <input-stem>_<task-type>")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    spec = get_model_spec(args.model_key)
    raw_rows = read_jsonl(args.input)[: args.limit]
    examples = validate_rows(raw_rows, args.task_type)
    if args.validate_only:
        print(f"validated {len(examples)} rows for {args.model_key}/{args.task_type}")
        return

    set_seed(args.seed)
    quantize = spec.default_4bit and not args.no_4bit
    tokenizer, model = load_model(spec.model_id, quantize)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    predictions: list[dict] = []
    started = time.perf_counter()
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start : start + args.batch_size]
        prompts = [format_prompt(tokenizer, item.prompt, system_prompt(item.task_type)) for item in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = 1000.0 * (time.perf_counter() - batch_started) / len(batch)
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for example, text, tokens in zip(batch, texts, new_tokens):
            parsed, correct = score_prediction(text, example)
            predictions.append({
                "id": example.id,
                "split": example.split,
                "task_type": example.task_type,
                "model_key": spec.key,
                "model_id": spec.model_id,
                "prediction": text,
                "parsed_answer": parsed,
                "correct": correct,
                "generated_tokens": int((tokens != tokenizer.pad_token_id).sum().item()),
                "latency_ms": latency_ms,
            })
        print(f"{spec.key}: {min(start + args.batch_size, len(examples))}/{len(examples)}", flush=True)

    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    run_name = args.run_name or f"{Path(args.input).stem}_{args.task_type}"
    target = Path(args.output_dir) / run_name / spec.key
    write_jsonl(target / "predictions.jsonl", predictions)
    report = {
        "model": spec.to_dict(),
        "task_type": args.task_type,
        "input": str(Path(args.input)),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "quantized_4bit": quantize,
        "metrics": summarize(predictions, elapsed, peak_allocated, peak_reserved),
    }
    write_json(target / "report.json", report)
    print(target / "report.json")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
