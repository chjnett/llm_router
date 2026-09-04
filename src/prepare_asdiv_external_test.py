"""Download and normalize ASDiv as an untouched external evaluation set."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import write_json, write_jsonl
from .scoring import extract_final_number


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="EleutherAI/asdiv")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-root", default="artifacts/asdiv_external_test")
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split)
    rows = []
    skipped = 0
    for index, source in enumerate(dataset):
        gold = extract_final_number(str(source["answer"]))
        if gold is None:
            skipped += 1
            continue
        body = str(source.get("body", "")).strip()
        question = str(source.get("question", "")).strip()
        text = f"{body} {question}".strip()
        rows.append({
            "id": f"asdiv-{index:05d}",
            "split": "test",
            "question": text,
            "answer": f"#### {gold:g}",
            "source_answer": source["answer"],
            "solution_type": source.get("solution_type"),
            "formula": source.get("formula"),
        })
    output = Path(args.output_root)
    write_jsonl(output / "data" / "test.jsonl", rows)
    write_json(output / "manifest.json", {
        "protocol": "untouched ASDiv external test; no ASDiv policy or model tuning",
        "dataset": args.dataset,
        "source_split": args.split,
        "source_count": len(dataset),
        "normalized_count": len(rows),
        "skipped_non_numeric": skipped,
        "license": "cc-by-nc-4.0",
    })
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
