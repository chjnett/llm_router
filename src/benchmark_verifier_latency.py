"""Benchmark full-solution and answer-only second passes on identical prompts."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .common import load_config, read_jsonl, set_seed, write_json
from .inference import (
    ANSWER_ONLY_SYSTEM_PROMPT,
    MICRO_REASONING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    format_prompt,
    load_model,
)


def completion_lengths(token_rows, eos_token_id: int | None) -> list[int]:
    lengths = []
    for row in token_rows:
        values = row.tolist()
        if eos_token_id is not None and eos_token_id in values:
            values = values[: values.index(eos_token_id)]
        lengths.append(len(values))
    return lengths


def run_condition(tokenizer, model, questions, system_prompt, batch_size, max_new_tokens):
    total_tokens = 0
    item_latencies = []
    started = time.perf_counter()
    for start in range(0, len(questions), batch_size):
        batch = questions[start : start + batch_size]
        prompts = [format_prompt(tokenizer, question, system_prompt) for question in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        torch.cuda.synchronize()
        batch_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        batch_seconds = time.perf_counter() - batch_started
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        total_tokens += sum(completion_lengths(new_tokens, tokenizer.eos_token_id))
        item_latencies.extend([batch_seconds / len(batch)] * len(batch))
    elapsed = time.perf_counter() - started
    return {
        "items": len(questions),
        "elapsed_seconds": elapsed,
        "milliseconds_per_item": 1000.0 * elapsed / len(questions),
        "gpu_generate_milliseconds_per_item": 1000.0 * float(np.mean(item_latencies)),
        "generated_tokens_total": total_tokens,
        "generated_tokens_mean": total_tokens / len(questions),
        "items_per_second": len(questions) / elapsed,
    }


def summarize(runs):
    keys = (
        "elapsed_seconds",
        "milliseconds_per_item",
        "gpu_generate_milliseconds_per_item",
        "generated_tokens_mean",
        "items_per_second",
    )
    return {
        key: {
            "mean": float(statistics.mean(run[key] for run in runs)),
            "median": float(statistics.median(run[key] for run in runs)),
        }
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/svamp_cross_task.yaml")
    parser.add_argument("--data", default="artifacts/svamp/data/risk_certification.jsonl")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", default="artifacts/results/verifier_latency_benchmark.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    rows = read_jsonl(Path(args.data))[: args.limit]
    questions = [row["question"] for row in rows]
    tokenizer, model = load_model(cfg["models"]["lower"], quantize_4bit=True)

    # Warm up kernels without including them in either condition.
    run_condition(tokenizer, model, questions[: args.batch_size], ANSWER_ONLY_SYSTEM_PROMPT, args.batch_size, 16)
    conditions = {
        "full_second_pass": (SYSTEM_PROMPT, cfg["generation"]["max_new_tokens"]),
        "answer_only_verifier": (ANSWER_ONLY_SYSTEM_PROMPT, cfg["generation"].get("answer_only_max_new_tokens", 64)),
        "micro_reasoning_verifier": (MICRO_REASONING_SYSTEM_PROMPT, 64),
    }
    runs = {name: [] for name in conditions}
    order = []
    for repeat in range(args.repeats):
        condition_names = list(conditions)
        order.extend(condition_names if repeat % 2 == 0 else reversed(condition_names))
    for index, name in enumerate(order, start=1):
        prompt, max_tokens = conditions[name]
        result = run_condition(tokenizer, model, questions, prompt, args.batch_size, max_tokens)
        runs[name].append(result)
        print(f"{index}/{len(order)} {name}: {result['milliseconds_per_item']:.2f} ms/item", flush=True)

    summary = {name: summarize(values) for name, values in runs.items()}
    full_ms = summary["full_second_pass"]["milliseconds_per_item"]["median"]
    verifier_ms = summary["answer_only_verifier"]["milliseconds_per_item"]["median"]
    micro_ms = summary["micro_reasoning_verifier"]["milliseconds_per_item"]["median"]
    payload = {
        "protocol": "same model, questions, batch size, quantization and GPU; forward/reverse order after warmup",
        "model": cfg["models"]["lower"],
        "device": torch.cuda.get_device_name(0),
        "sample_count": len(rows),
        "batch_size": args.batch_size,
        "repeats_per_condition": args.repeats,
        "runs": runs,
        "summary": summary,
        "answer_only_latency_ratio_vs_full_second_pass": verifier_ms / full_ms,
        "answer_only_latency_reduction": 1.0 - verifier_ms / full_ms,
        "micro_reasoning_latency_ratio_vs_full_second_pass": micro_ms / full_ms,
        "micro_reasoning_latency_reduction": 1.0 - micro_ms / full_ms,
    }
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
