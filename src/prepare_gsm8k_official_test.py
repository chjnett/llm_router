"""Materialize the untouched official GSM8K test split for final evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow-file", required=True)
    parser.add_argument("--output-root", default="artifacts/gsm8k_official_test")
    args = parser.parse_args()

    from datasets import Dataset

    dataset = Dataset.from_file(args.arrow_file)
    rows = [
        {**dict(row), "id": f"gsm8k-official-test-{index:05d}", "split": "test"}
        for index, row in enumerate(dataset)
    ]
    output = Path(args.output_root)
    write_jsonl(output / "data" / "test.jsonl", rows)
    write_json(output / "manifest.json", {
        "protocol": "untouched official GSM8K test; model and thresholds frozen before inference",
        "count": len(rows),
        "source": str(Path(args.arrow_file).name),
        "frozen_policy": {
            "domain_adaptation_rows": 688,
            "pre_model": "semantic logistic regression",
            "pre_threshold": 0.84,
            "post_model": "confidence Extra Trees",
            "post_threshold": 0.49,
        },
    })
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
