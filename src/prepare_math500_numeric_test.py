"""Prepare the strictly numeric-answer subset of untouched MATH-500."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .common import write_json, write_jsonl


SIMPLE_NUMBER = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")


def parse_simple_number(value: object) -> float | None:
    text = str(value).strip().replace("$", "")
    if not SIMPLE_NUMBER.fullmatch(text):
        return None
    return float(text.replace(",", ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", default="artifacts/math500_numeric_test")
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split)
    rows = []
    skipped_non_simple_numeric = 0
    for index, source in enumerate(dataset):
        gold = parse_simple_number(source["answer"])
        if gold is None:
            skipped_non_simple_numeric += 1
            continue
        rows.append({
            "id": f"math500-{index:04d}",
            "split": "test",
            "question": str(source["problem"]).strip(),
            "answer": f"#### {gold:g}",
            "source_answer": source["answer"],
            "subject": source.get("subject"),
            "level": source.get("level"),
            "unique_id": source.get("unique_id"),
        })
    output = Path(args.output_root)
    write_jsonl(output / "data" / "test.jsonl", rows)
    write_json(output / "manifest.json", {
        "protocol": "untouched MATH-500 numeric-answer subset; 0.60/0.60 policy frozen before download; no model or policy tuning",
        "dataset": args.dataset,
        "source_split": args.split,
        "source_count": len(dataset),
        "normalized_count": len(rows),
        "skipped_non_simple_numeric": skipped_non_simple_numeric,
        "numeric_filter": SIMPLE_NUMBER.pattern,
    })
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
