from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .common import load_config, read_jsonl, set_seed, write_json
from .inference import CONCISE_SYSTEM_PROMPT


class SFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=768):
        self.items = []
        for row in rows:
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": CONCISE_SYSTEM_PROMPT}, {"role": "user", "content": row["question"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(row["teacher_response"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + answer_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
            self.items.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(items, pad_token_id):
    length = max(len(item["input_ids"]) for item in items)
    input_ids, labels, masks = [], [], []
    for item in items:
        padding = length - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        labels.append(item["labels"] + [-100] * padding)
        masks.append([1] * len(item["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(masks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--method", choices=["random", "cluster"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    rows = read_jsonl(Path("artifacts/distillation") / f"{args.method}_seed_{args.seed}.jsonl")
    model_id = cfg["models"]["lower"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization, dtype=torch.float16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=cfg["distillation"]["lora_rank"],
        lora_alpha=cfg["distillation"]["lora_alpha"],
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    dataset = SFTDataset(rows, tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda items: collate(items, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg["distillation"]["learning_rate"])
    accumulation = 8
    losses = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_steps = cfg["distillation"]["epochs"] * len(loader)
    for epoch in range(cfg["distillation"]["epochs"]):
        for step, batch in enumerate(loader):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            loss = model(**batch).loss / accumulation
            loss.backward()
            losses.append(float(loss.item() * accumulation))
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            completed = epoch * len(loader) + step + 1
            if completed % 8 == 0:
                print(f"{args.method}/{args.seed}: {completed}/{total_steps} loss={losses[-1]:.4f}", flush=True)
    output = Path("artifacts/adapters") / f"{args.method}_seed_{args.seed}"
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    write_json(output / "training_metrics.json", {"examples": len(rows), "epochs": cfg["distillation"]["epochs"], "mean_loss": sum(losses) / len(losses), "final_loss": losses[-1]})
    print(output)


if __name__ == "__main__":
    main()
