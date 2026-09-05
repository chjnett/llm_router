"""Run the fixed Qwen7B option-logit Upper on independent MMLU test rows."""

from __future__ import annotations

from pathlib import Path

from .run_mmlu_logit_screening import run_condition


def main() -> None:
    condition = {
        "model": "qwen_7b", "batch": 8, "limit": 500,
        "input": "artifacts/data/mmlu_test_independent_500.jsonl", "quantize_4bit": True,
    }
    report = run_condition(Path("artifacts/model_screening"), condition)
    print(report["metrics"])


if __name__ == "__main__":
    main()
