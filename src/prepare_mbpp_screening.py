"""Prepare a fixed 50-item MBPP screening set without consuming the full test split."""

from __future__ import annotations

import argparse
import random

from .common import write_jsonl


def convert_row(row: dict, source_split: str) -> dict:
    return {
        "id": f"mbpp-{source_split}-{int(row['task_id'])}",
        "prompt": str(row["prompt"]),
        "reference": {"test_count": len(row["test_list"])},
        "task_type": "code",
        "split": "screening",
        "task_metadata": {
            "dataset": "google-research-datasets/mbpp",
            "source_split": source_split,
            "task_id": int(row["task_id"]),
            "test_imports": list(row["test_imports"]),
            "test_list": list(row["test_list"]),
        },
    }


def main() -> None:
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/data/mbpp_screening_50.jsonl")
    parser.add_argument("--seed", type=int, default=2031)
    args = parser.parse_args()
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized")
    rows = [convert_row(dict(row), "validation") for row in dataset["validation"]]
    test_indices = list(range(len(dataset["test"])))
    random.Random(args.seed).shuffle(test_indices)
    rows.extend(convert_row(dict(dataset["test"][index]), "test") for index in test_indices[:7])
    random.Random(args.seed).shuffle(rows)
    write_jsonl(args.output, rows)
    print(f"wrote validation=43, test=7 -> {args.output}")


if __name__ == "__main__":
    main()
