"""Benchmark one-forward MMLU option-logit scoring across model pairs."""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from .common import read_jsonl, set_seed, write_json, write_jsonl
from .inference import format_prompt, load_model
from .model_registry import get_model_spec
from .power_metrics import PowerSampler
from .run_model_screening import summarize
from .task_harness import adapt_row


MODELS = ("qwen_1_5b", "qwen_7b", "smollm2_360m", "smollm2_1_7b")
SYSTEM = "Select the correct option. Do not explain."


def conditions(input_path: str) -> list[dict]:
    return [
        {"model": model, "batch": batch, "limit": limit, "input": input_path,
         "quantize_4bit": model == "qwen_7b"}
        for batch, limit in ((8, 200), (1, 50)) for model in MODELS
    ]


def run_name(condition: dict) -> str:
    precision = "4bit" if condition["quantize_4bit"] else "fp16"
    return f"mmlu_logit_{precision}_b{condition['batch']}_{condition['limit']}"


def report_path(output_dir: Path, condition: dict) -> Path:
    return output_dir / run_name(condition) / condition["model"] / "report.json"


def option_token_ids(tokenizer) -> list[int]:
    ids = []
    for label in "ABCD":
        encoded = tokenizer.encode(f" {label}", add_special_tokens=False)
        if not encoded:
            raise ValueError(f"Tokenizer cannot encode option {label}")
        ids.append(encoded[-1])
    if len(set(ids)) != 4:
        raise ValueError(f"Option token ids are not unique: {ids}")
    return ids


def run_condition(output_dir: Path, condition: dict) -> dict:
    spec = get_model_spec(condition["model"])
    examples = [adapt_row(row, "multiple_choice") for row in read_jsonl(condition["input"])[: condition["limit"]]]
    tokenizer, model = load_model(
        spec.model_id,
        condition["quantize_4bit"],
        trust_remote_code=spec.trust_remote_code,
    )
    candidate_ids = option_token_ids(tokenizer)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rows = []
    sampler = PowerSampler()
    sampler.start()
    started = time.perf_counter()
    for start in range(0, len(examples), condition["batch"]):
        batch = examples[start : start + condition["batch"]]
        prompts = [format_prompt(tokenizer, item.prompt, SYSTEM) + "Final answer:" for item in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        torch.cuda.synchronize()
        batch_started = time.perf_counter()
        with torch.inference_mode():
            logits = model(**encoded).logits[:, -1, candidate_ids]
        torch.cuda.synchronize()
        latency_ms = 1000.0 * (time.perf_counter() - batch_started) / len(batch)
        predictions = logits.argmax(dim=-1).tolist()
        for item, prediction in zip(batch, predictions):
            label = "ABCD"[prediction]
            reference = "ABCD"[item.reference] if isinstance(item.reference, int) else str(item.reference).upper()
            rows.append({
                "id": item.id, "prediction": label, "parsed_answer": label, "correct": label == reference,
                "generated_tokens": 1, "hit_token_limit": False, "latency_ms": latency_ms,
            })
        print(f"{spec.key}: {min(start + condition['batch'], len(examples))}/{len(examples)}", flush=True)
    elapsed = time.perf_counter() - started
    power = sampler.stop(elapsed)
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    target = output_dir / run_name(condition) / spec.key
    write_jsonl(target / "predictions.jsonl", rows)
    report = {
        "model": spec.to_dict(), "method": "option_logit", "input": condition["input"],
        "batch_size": condition["batch"], "quantized_4bit": condition["quantize_4bit"],
        "metrics": summarize(rows, elapsed, peak_allocated, peak_reserved, power),
    }
    write_json(target / "report.json", report)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def aggregate(output_dir: Path, matrix: list[dict]) -> dict:
    runs = []
    for condition in matrix:
        path = report_path(output_dir, condition)
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
            runs.append({**condition, "metrics": report["metrics"], "report": str(path)})
    return {"completed": len(runs), "planned": len(matrix), "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/data/mmlu_validation_200.jsonl")
    parser.add_argument("--output-dir", default="artifacts/model_screening")
    parser.add_argument("--log", default="artifacts/logs/mmlu_logit_screening.log")
    parser.add_argument("--summary", default="artifacts/results/mmlu_logit_screening.json")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    matrix = conditions(args.input)
    output_dir = Path(args.output_dir)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== run started {datetime.now().isoformat()} ===\n")
        for index, condition in enumerate(matrix, start=1):
            label = f"[{index}/{len(matrix)}] {condition['model']} b{condition['batch']} n{condition['limit']}"
            if report_path(output_dir, condition).exists() and not args.rerun:
                print(f"SKIP {label}", flush=True)
                continue
            print(f"START {label}", flush=True)
            log.write(f"\nSTART {label}\n")
            try:
                run_condition(output_dir, condition)
            except Exception as error:
                failures.append({"condition": condition, "error": repr(error)})
                print(f"FAILED {label}: {error}", flush=True)
            write_json(args.summary, {**aggregate(output_dir, matrix), "failures": failures})
    payload = {**aggregate(output_dir, matrix), "failures": failures}
    write_json(args.summary, payload)
    print(f"DONE {payload['completed']}/{payload['planned']} failures={len(failures)} -> {args.summary}")


if __name__ == "__main__":
    set_seed(42)
    main()
