"""Prepare non-overlapping MBPP certification and final splits."""

from __future__ import annotations

import argparse
import random

from .common import read_jsonl, write_jsonl
from .prepare_mbpp_screening import convert_row, load_splits


def split_independent_rows(test_rows, excluded_task_ids: set[int], seed: int) -> list[dict]:
    candidates = [dict(row) for row in test_rows if int(row["task_id"]) not in excluded_task_ids]
    random.Random(seed).shuffle(candidates)
    if len(candidates) < 200:
        raise ValueError(f"Need 200 non-overlapping MBPP test rows, found {len(candidates)}")
    selected = candidates[:200]
    output = []
    for index, row in enumerate(selected):
        item = convert_row(row, "test")
        item["split"] = "certification" if index < 100 else "final"
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="artifacts/data/mbpp_screening_50.jsonl")
    parser.add_argument("--output", default="artifacts/data/mbpp_independent_200.jsonl")
    parser.add_argument("--seed", type=int, default=2035)
    args = parser.parse_args()
    selection = read_jsonl(args.selection)
    excluded = {
        int(row["task_metadata"]["task_id"])
        for row in selection if row["task_metadata"]["source_split"] == "test"
    }
    dataset = load_splits()
    rows = split_independent_rows(dataset["test"], excluded, args.seed)
    write_jsonl(args.output, rows)
    print(f"wrote certification=100, final=100, excluded_test={len(excluded)} -> {args.output}")


if __name__ == "__main__":
    main()
