"""Extract one-forward prompt representations for MBPP pre-triage."""

from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path

import torch

from .common import read_jsonl, set_seed, write_json, write_jsonl
from .inference import format_prompt, load_model
from .model_registry import MODEL_REGISTRY, get_model_spec
from .run_model_screening import percentile
from .task_harness import adapt_row, system_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), required=True)
    parser.add_argument("--input", default="artifacts/data/mbpp_screening_50.jsonl")
    parser.add_argument("--output-dir", default="artifacts/mbpp_hidden_probe")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2033)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for comparable one-forward latency")

    spec = get_model_spec(args.model_key)
    rows = read_jsonl(args.input)[: args.limit]
    examples = [adapt_row(row, "code") for row in rows]
    set_seed(args.seed)
    quantize = spec.default_4bit and not args.no_4bit
    tokenizer, model = load_model(spec.model_id, quantize)
    tokenizer.padding_side = "right"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    output = []
    latencies = []
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start : start + args.batch_size]
        prompts = [format_prompt(tokenizer, item.prompt, system_prompt("code", "task")) for item in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result = model(**encoded, output_hidden_states=True, use_cache=True)
        torch.cuda.synchronize()
        latency_ms = 1000.0 * (time.perf_counter() - started) / len(batch)
        latencies.extend([latency_ms] * len(batch))
        lengths = encoded["attention_mask"].sum(dim=1).long()
        for index, item in enumerate(batch):
            position = int(lengths[index].item()) - 1
            hidden = result.hidden_states[-1][index, position].float()
            logits = result.logits[index, position].float()
            probabilities = torch.softmax(logits, dim=-1)
            log_probabilities = torch.log_softmax(logits, dim=-1)
            top2 = probabilities.topk(2).values
            entropy = -(probabilities * log_probabilities).sum().item()
            values = [entropy, top2[0].item(), (top2[0] - top2[1]).item()]
            output.append({
                "id": item.id,
                "model_key": spec.key,
                "prompt_tokens": int(lengths[index].item()),
                "probe_latency_ms": latency_ms,
                "next_entropy": float(entropy),
                "next_max_probability": float(top2[0].item()),
                "next_margin": float((top2[0] - top2[1]).item()),
                "hidden": hidden.cpu().tolist(),
                "finite": all(math.isfinite(value) for value in values) and bool(torch.isfinite(hidden).all()),
            })
        print(f"hidden-probe/{spec.key}: {min(start + args.batch_size, len(examples))}/{len(examples)}", flush=True)

    target = Path(args.output_dir) / spec.key
    write_jsonl(target / "features.jsonl", output)
    write_json(target / "report.json", {
        "model": spec.to_dict(),
        "items": len(output),
        "batch_size": args.batch_size,
        "quantized_4bit": quantize,
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "peak_vram_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
    })
    print(target / "report.json")
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
