"""Download and normalize the MAWPS held-out split for external evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import write_json, write_jsonl
from .scoring import extract_final_number


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="garrethlee/MAWPS")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", default="artifacts/mawps_external_test")
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
        rows.append({
            "id": f"mawps-{index:05d}",
            "split": "test",
            "question": str(source["question"]).strip(),
            "answer": f"#### {gold:g}",
            "source_answer": source["answer"],
        })
    output = Path(args.output_root)
    write_jsonl(output / "data" / "test.jsonl", rows)
    write_json(output / "manifest.json", {
        "protocol": "MAWPS held-out split; no MAWPS model or policy tuning",
        "dataset": args.dataset,
        "source_split": args.split,
        "source_count": len(dataset),
        "normalized_count": len(rows),
        "skipped_non_numeric": skipped,
        "license": "not specified on the selected Hugging Face dataset card",
    })
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
