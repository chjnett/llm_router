"""Prepare the official SVAMP test and a fixed train-derived validation set."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import load_dataset

from .common import load_config, write_json, write_jsonl


def convert(row: dict, split: str) -> dict:
    body = str(row["Body"]).strip()
    question = str(row["Question"]).strip()
    answer = str(row["Answer"]).strip()
    return {
        "id": f"svamp-{row['ID']}",
        "question": f"{body}\n{question}",
        "answer": f"#### {answer}",
        "split": split,
        "source_split": "test" if split == "test" else "train",
        "type": row.get("Type"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/svamp_cross_task.yaml")
    parser.add_argument("--output-dir", default="artifacts/svamp/data")
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = load_dataset(cfg["dataset"]["name"])
    train = list(dataset["train"])
    test = list(dataset["test"])
    rng = random.Random(cfg["seed"])
    rng.shuffle(train)
    validation = train[: cfg["dataset"]["validation_size"]]
    if len(test) != cfg["dataset"]["test_size"]:
        raise RuntimeError(f"Expected {cfg['dataset']['test_size']} official test rows, got {len(test)}")
    output = Path(args.output_dir)
    validation_rows = [convert(row, "validation") for row in validation]
    test_rows = [convert(row, "test") for row in test]
    if {row["id"] for row in validation_rows} & {row["id"] for row in test_rows}:
        raise RuntimeError("SVAMP validation/test overlap detected")
    write_jsonl(output / "validation.jsonl", validation_rows)
    write_jsonl(output / "test.jsonl", test_rows)
    write_json(output / "manifest.json", {
        "dataset": cfg["dataset"]["name"],
        "seed": cfg["seed"],
        "validation": {"count": len(validation_rows), "source": "official train"},
        "test": {"count": len(test_rows), "source": "official test", "selection": "all rows"},
        "cross_split_overlap": 0,
    })
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
