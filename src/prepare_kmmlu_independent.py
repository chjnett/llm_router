"""Prepare non-overlapping KMMLU certification and final-test rows."""

from __future__ import annotations

import argparse
import random

from .common import read_jsonl, write_jsonl
from .prepare_kmmlu_screening import DEFAULT_SUBJECTS, convert_row


def main() -> None:
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", default="artifacts/data/kmmlu_screening_200.jsonl")
    parser.add_argument("--output", default="artifacts/data/kmmlu_independent_500.jsonl")
    parser.add_argument("--per-subject", type=int, default=25)
    parser.add_argument("--certification-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2029)
    args = parser.parse_args()
    excluded = {row["id"] for row in read_jsonl(args.exclude)}
    selected = []
    for offset, subject in enumerate(DEFAULT_SUBJECTS):
        dataset = load_dataset("HAERAE-HUB/KMMLU", subject, split="test")
        candidates = [
            convert_row(dict(dataset[index]), subject, index)
            for index in range(len(dataset))
            if f"kmmlu-test-{subject}-{index}" not in excluded
        ]
        random.Random(args.seed + offset).shuffle(candidates)
        if len(candidates) < args.per_subject:
            raise ValueError(f"{subject} has only {len(candidates)} non-overlapping rows")
        selected.extend(candidates[: args.per_subject])
        print(f"prepared {subject}: {args.per_subject}", flush=True)
    random.Random(args.seed).shuffle(selected)
    if not 0 < args.certification_size < len(selected):
        raise ValueError("--certification-size must split the prepared rows")
    for index, row in enumerate(selected):
        row["split"] = "certification" if index < args.certification_size else "final_test"
    write_jsonl(args.output, selected)
    print(
        f"wrote certification={args.certification_size}, "
        f"final_test={len(selected) - args.certification_size}, overlap=0 -> {args.output}"
    )


if __name__ == "__main__":
    main()
