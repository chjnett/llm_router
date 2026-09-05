"""Prepare fixed, subject-balanced MMLU test certification and final splits."""

from __future__ import annotations

import argparse

from datasets import load_dataset

from .common import write_jsonl
from .prepare_mmlu_screening import balanced_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/data/mmlu_test_independent_500.jsonl")
    parser.add_argument("--certification-size", type=int, default=250)
    parser.add_argument("--test-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    dataset = load_dataset("cais/mmlu", "all", split="test")
    rows = []
    for index, row in enumerate(dataset):
        item = dict(row)
        item["_source_index"] = index
        rows.append(item)
    total = args.certification_size + args.test_size
    selected = balanced_sample(rows, total, args.seed)
    for index, row in enumerate(selected):
        row["id"] = row["id"].replace("mmlu-validation", "mmlu-test")
        row["split"] = "certification" if index < args.certification_size else "final_test"
    write_jsonl(args.output, selected)
    print(f"wrote certification={args.certification_size}, final_test={args.test_size} -> {args.output}")


if __name__ == "__main__":
    main()
