from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .common import load_config, read_jsonl, set_seed, write_jsonl
from .scoring import extract_final_number, gsm8k_correct


SYSTEM_PROMPT = (
    "Solve the math problem carefully. End your response with exactly "
    "'Final answer: <number>'."
)
CONCISE_SYSTEM_PROMPT = (
    "Solve the math problem. Keep the reasoning to at most three short lines. "
    "Your final line must be exactly 'Final answer: <number>'."
)
ANSWER_ONLY_SYSTEM_PROMPT = (
    "Solve the problem independently, but return only one line in the exact form "
    "'Final answer: <number>'. Do not include reasoning or explanation."
)


def format_prompt(tokenizer, question: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(model_id: str, quantize_4bit: bool, adapter: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    if quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def infer_rows(
    rows: list[dict],
    model_id: str,
    batch_size: int,
    max_new_tokens: int,
    quantize_4bit: bool,
    output_path: Path | None = None,
    existing: list[dict] | None = None,
    adapter: str | None = None,
    concise: bool = False,
    answer_only: bool = False,
) -> list[dict]:
    tokenizer, model = load_model(model_id, quantize_4bit, adapter)
    output: list[dict] = list(existing or [])
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompt = ANSWER_ONLY_SYSTEM_PROMPT if answer_only else (CONCISE_SYSTEM_PROMPT if concise else SYSTEM_PROMPT)
        prompts = [format_prompt(tokenizer, row["question"], prompt) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for row, text, tokens in zip(batch, texts, new_tokens):
            output.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "prediction": text,
                    "predicted_number": extract_final_number(text),
                    "correct": gsm8k_correct(text, row["answer"]),
                    "model": model_id,
                    "generated_tokens": int((tokens != tokenizer.pad_token_id).sum().item()),
                }
            )
        if output_path is not None:
            write_jsonl(output_path, output)
        print(f"{model_id}: {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--model", choices=["lower", "upper"], required=True)
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--adapter")
    parser.add_argument("--output-name")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--inference-dir", default="artifacts/inference")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--concise", action="store_true")
    parser.add_argument("--answer-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    splits = args.splits or ["router_train", "distill_train", "validation", "test"]
    for split in splits:
        rows = read_jsonl(Path(args.data_dir) / f"{split}.jsonl")
        if args.limit:
            rows = rows[: args.limit]
        output_name = args.output_name or args.model
        output_path = Path(args.inference_dir) / output_name / f"{split}.jsonl"
        existing = read_jsonl(output_path) if output_path.exists() else []
        completed = {row["id"] for row in existing}
        rows = [row for row in rows if row["id"] not in completed]
        if not rows:
            print(f"{args.model}/{split}: cache complete ({len(existing)})")
            continue
        predictions = infer_rows(
            rows,
            cfg["models"][args.model],
            args.batch_size or cfg["generation"]["batch_size"],
            args.max_new_tokens or cfg["generation"]["max_new_tokens"],
            not args.no_4bit,
            output_path,
            existing,
            args.adapter,
            args.concise,
            args.answer_only,
        )
        write_jsonl(output_path, predictions)


if __name__ == "__main__":
    main()
