"""Prepare a fixed 50-item MBPP screening set without consuming the full test split."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

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


def load_splits() -> dict:
    """Prefer completed Arrow cache files to avoid a slow Hub/cache-lock probe."""
    from datasets import Dataset, config, load_dataset

    cache_root = Path(config.HF_DATASETS_CACHE)
    candidates = sorted(
        cache_root.glob("google-research-datasets___mbpp/sanitized/*/*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        validation = candidate / "mbpp-validation.arrow"
        test = candidate / "mbpp-test.arrow"
        if validation.is_file() and test.is_file():
            print(f"using completed local MBPP cache: {candidate}")
            return {
                "validation": Dataset.from_file(str(validation)),
                "test": Dataset.from_file(str(test)),
            }
    return load_dataset("google-research-datasets/mbpp", "sanitized")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/data/mbpp_screening_50.jsonl")
    parser.add_argument("--seed", type=int, default=2031)
    args = parser.parse_args()
    dataset = load_splits()
    rows = [convert_row(dict(row), "validation") for row in dataset["validation"]]
    test_indices = list(range(len(dataset["test"])))
    random.Random(args.seed).shuffle(test_indices)
    rows.extend(convert_row(dict(dataset["test"][index]), "test") for index in test_indices[:7])
    random.Random(args.seed).shuffle(rows)
    write_jsonl(args.output, rows)
    print(f"wrote validation=43, test=7 -> {args.output}")


if __name__ == "__main__":
    main()
