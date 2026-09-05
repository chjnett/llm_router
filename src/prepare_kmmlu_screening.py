"""Prepare a deterministic, broad-domain KMMLU screening set."""

from __future__ import annotations

import argparse
import random

from .common import write_jsonl


# Twenty subjects spanning STEM, applied engineering, business, law/social science,
# health, and Korean history. Ten examples per subject gives a transparent 200-item pilot.
DEFAULT_SUBJECTS = (
    "Accounting",
    "Agricultural-Sciences",
    "Biology",
    "Chemistry",
    "Civil-Engineering",
    "Computer-Science",
    "Criminal-Law",
    "Economics",
    "Education",
    "Electrical-Engineering",
    "Environmental-Science",
    "Health",
    "Information-Technology",
    "Law",
    "Management",
    "Math",
    "Political-Science-and-Sociology",
    "Psychology",
    "Social-Welfare",
    "Korean-History",
)


def convert_row(row: dict, subject: str, source_index: int) -> dict:
    """Convert KMMLU's 1-based answer into the router's 0-based MC schema."""
    answer = int(row["answer"])
    if answer not in {1, 2, 3, 4}:
        raise ValueError(f"Unexpected KMMLU answer {answer!r} in {subject}")
    return {
        "id": f"kmmlu-test-{subject}-{source_index}",
        "question": str(row["question"]),
        "choices": [str(row[label]) for label in "ABCD"],
        "answer": answer - 1,
        "split": "screening",
        "task_metadata": {
            "dataset": "HAERAE-HUB/KMMLU",
            "subject": subject,
            "source_split": "test",
            "human_accuracy": row.get("Human Accuracy"),
        },
    }


def main() -> None:
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/data/kmmlu_screening_200.jsonl")
    parser.add_argument("--per-subject", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    args = parser.parse_args()
    if args.per_subject <= 0:
        raise ValueError("--per-subject must be positive")

    selected = []
    for offset, subject in enumerate(args.subjects):
        dataset = load_dataset("HAERAE-HUB/KMMLU", subject, split="test")
        if len(dataset) < args.per_subject:
            raise ValueError(f"{subject} has only {len(dataset)} test rows")
        indices = list(range(len(dataset)))
        random.Random(args.seed + offset).shuffle(indices)
        for source_index in indices[: args.per_subject]:
            selected.append(convert_row(dict(dataset[source_index]), subject, source_index))
        print(f"prepared {subject}: {args.per_subject}", flush=True)

    random.Random(args.seed).shuffle(selected)
    write_jsonl(args.output, selected)
    print(f"wrote {len(selected)} rows across {len(args.subjects)} subjects -> {args.output}")


if __name__ == "__main__":
    main()
