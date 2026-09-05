"""Prepare a deterministic instruction-family-balanced IFEval screening set."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict, deque

from .common import write_jsonl


def primary_family(row: dict) -> str:
    instruction_ids = list(row["instruction_id_list"])
    return instruction_ids[0].split(":", 1)[0] if instruction_ids else "unknown"


def balanced_sample(rows: list[dict], limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[primary_family(row)].append(row)
    queues = {}
    for family, values in groups.items():
        rng.shuffle(values)
        queues[family] = deque(values)
    selected = []
    while len(selected) < limit and any(queues.values()):
        for family in sorted(queues):
            if not queues[family]:
                continue
            row = queues[family].popleft()
            selected.append({
                "id": f"ifeval-{int(row['key'])}",
                "prompt": str(row["prompt"]),
                "reference": {"instruction_count": len(row["instruction_id_list"])},
                "task_type": "instruction_following",
                "split": "screening",
                "task_metadata": {
                    "dataset": "google/IFEval",
                    "key": int(row["key"]),
                    "instruction_id_list": list(row["instruction_id_list"]),
                    "kwargs": list(row["kwargs"]),
                    "primary_family": family,
                },
            })
            if len(selected) == limit:
                break
    if len(selected) < limit:
        raise ValueError(f"IFEval has only {len(selected)} usable rows; requested {limit}")
    return selected


def main() -> None:
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/data/ifeval_screening_50.jsonl")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2030)
    args = parser.parse_args()
    dataset = load_dataset("google/IFEval", split="train")
    selected = balanced_sample([dict(row) for row in dataset], args.limit, args.seed)
    write_jsonl(args.output, selected)
    families = sorted({row["task_metadata"]["primary_family"] for row in selected})
    print(f"wrote {len(selected)} rows across {len(families)} primary families -> {args.output}")


if __name__ == "__main__":
    main()
