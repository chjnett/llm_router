from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .common import load_config, read_jsonl


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def embed_texts(texts: list[str], model_id: str, batch_size: int = 32) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).cuda().eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start : start + batch_size], padding=True, truncation=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            vectors = mean_pool(model(**batch).last_hidden_state, batch["attention_mask"])
            vectors = F.normalize(vectors, p=2, dim=1)
        chunks.append(vectors.cpu().numpy())
    return np.concatenate(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--output-dir", default="artifacts/embeddings")
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split in args.splits or ["router_train", "distill_train", "validation", "test"]:
        rows = read_jsonl(Path(args.data_dir) / f"{split}.jsonl")
        vectors = embed_texts(
            [row["question"] for row in rows],
            cfg["models"]["embedding"],
            args.batch_size,
        )
        np.savez_compressed(output / f"{split}.npz", ids=np.array([row["id"] for row in rows]), embeddings=vectors)
        print(split, vectors.shape)


if __name__ == "__main__":
    main()
