"""Prepare a deterministic subject-balanced MMLU validation screening file."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict, deque

from .common import write_jsonl


def balanced_sample(rows: list[dict], limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("subject", "unknown"))].append(row)
    queues = {}
    for subject, values in groups.items():
        rng.shuffle(values)
        queues[subject] = deque(values)
    selected = []
    subjects = sorted(queues)
    while len(selected) < limit and any(queues.values()):
        for subject in subjects:
            if queues[subject]:
                row = queues[subject].popleft()
                selected.append({
                    "id": f"mmlu-validation-{subject}-{row.get('_source_index', len(selected))}",
                    "question": row["question"],
                    "choices": list(row["choices"]),
                    "answer": int(row["answer"]),
                    "split": "screening",
                    "task_metadata": {"dataset": "cais/mmlu", "subject": subject},
                })
                if len(selected) == limit:
                    break
    if len(selected) < limit:
        raise ValueError(f"MMLU validation has only {len(selected)} usable rows; requested {limit}")
    return selected


def main() -> None:
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/data/mmlu_validation_200.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = load_dataset("cais/mmlu", "all", split="validation")
    rows = []
    for index, row in enumerate(dataset):
        item = dict(row)
        item["_source_index"] = index
        rows.append(item)
    selected = balanced_sample(rows, args.limit, args.seed)
    write_jsonl(args.output, selected)
    print(f"wrote {len(selected)} rows across {len({row['task_metadata']['subject'] for row in selected})} subjects -> {args.output}")


if __name__ == "__main__":
    main()
