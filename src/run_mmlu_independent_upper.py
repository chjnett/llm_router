"""Run the fixed Qwen7B option-logit Upper on independent MMLU test rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from .run_mmlu_logit_screening import run_condition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/data/mmlu_test_independent_500.jsonl")
    parser.add_argument("--output-dir", default="artifacts/model_screening")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    condition = {
        "model": "qwen_7b", "batch": 8, "limit": args.limit,
        "input": args.input, "quantize_4bit": True,
    }
    report = run_condition(Path(args.output_dir), condition)
    print(report["metrics"])


if __name__ == "__main__":
    main()
