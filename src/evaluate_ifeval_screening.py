"""Score IFEval generations with the pinned official strict/loose evaluator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .common import read_jsonl, write_json


ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "third_party" / "google_research"
NLTK_ROOT = ROOT / "third_party" / "nltk_data"
sys.path.insert(0, str(VENDOR_ROOT))
os.environ.setdefault("NLTK_DATA", str(NLTK_ROOT))

from instruction_following_eval import evaluation_lib  # noqa: E402


def score(input_rows: list[dict], prediction_rows: list[dict], loose: bool) -> dict:
    predictions = {row["id"]: row["prediction"] for row in prediction_rows}
    outputs = []
    details = []
    evaluator = (
        evaluation_lib.test_instruction_following_loose
        if loose
        else evaluation_lib.test_instruction_following_strict
    )
    for row in input_rows:
        metadata = row["task_metadata"]
        example = evaluation_lib.InputExample(
            key=int(metadata["key"]),
            instruction_id_list=list(metadata["instruction_id_list"]),
            prompt=row["prompt"],
            kwargs=list(metadata["kwargs"]),
        )
        response = predictions[row["id"]]
        output = evaluator(example, {row["prompt"]: response})
        outputs.append(output)
        details.append({
            "id": row["id"],
            "follow_all": bool(output.follow_all_instructions),
            "follow_instruction_list": list(output.follow_instruction_list),
        })
    instruction_flags = [flag for output in outputs for flag in output.follow_instruction_list]
    return {
        "prompt_accuracy": sum(output.follow_all_instructions for output in outputs) / len(outputs),
        "instruction_accuracy": sum(instruction_flags) / len(instruction_flags),
        "prompt_correct": [bool(output.follow_all_instructions) for output in outputs],
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/data/ifeval_screening_50.jsonl")
    parser.add_argument("--lower-dir", required=True)
    parser.add_argument("--upper-dir", required=True)
    parser.add_argument("--output", default="paper/data/ifeval_screening_analysis.json")
    args = parser.parse_args()
    inputs = read_jsonl(args.input)
    lower_predictions = read_jsonl(Path(args.lower_dir) / "predictions.jsonl")
    upper_predictions = read_jsonl(Path(args.upper_dir) / "predictions.jsonl")
    with (Path(args.lower_dir) / "report.json").open("r", encoding="utf-8") as handle:
        lower_report = json.load(handle)
    with (Path(args.upper_dir) / "report.json").open("r", encoding="utf-8") as handle:
        upper_report = json.load(handle)
    lower_strict = score(inputs, lower_predictions, loose=False)
    upper_strict = score(inputs, upper_predictions, loose=False)
    lower_loose = score(inputs, lower_predictions, loose=True)
    upper_loose = score(inputs, upper_predictions, loose=True)
    latency_ratio = (
        lower_report["metrics"]["latency_ms_p50"] / upper_report["metrics"]["latency_ms_p50"]
    )
    lower_correct = lower_strict.pop("prompt_correct")
    upper_correct = upper_strict.pop("prompt_correct")
    oracle_accuracy = sum(a or b for a, b in zip(lower_correct, upper_correct)) / len(inputs)
    oracle_normalized_latency = latency_ratio + 1.0 - lower_strict["prompt_accuracy"]
    checks = {
        "upper_accuracy_advantage_5pp": upper_strict["prompt_accuracy"] - lower_strict["prompt_accuracy"] >= 0.05,
        "latency_ratio_below_0_5": latency_ratio < 0.5,
        "oracle_quality_95pct": oracle_accuracy >= 0.95 * upper_strict["prompt_accuracy"],
        "oracle_latency_10pct_reduction": oracle_normalized_latency <= 0.9,
        "lower_token_limit_below_10pct": lower_report["metrics"]["token_limit_rate"] <= 0.1,
    }
    payload = {
        "protocol": f"exploratory official IFEval strict/loose screen on {len(inputs)} rows",
        "official_evaluator_commit": "932d4685e23f671b9e8c2abc72dd228ba5ff9252",
        "lower": {"model": lower_report["model"], "strict": lower_strict, "loose": lower_loose, "metrics": lower_report["metrics"]},
        "upper": {"model": upper_report["model"], "strict": upper_strict, "loose": upper_loose, "metrics": upper_report["metrics"]},
        "latency_ratio": latency_ratio,
        "oracle_strict_prompt_accuracy": oracle_accuracy,
        "oracle_normalized_latency": oracle_normalized_latency,
        "checks": checks,
        "continue_to_200": all(checks.values()),
    }
    write_json(args.output, payload)
    print(f"continue_to_200={payload['continue_to_200']} -> {args.output}")


if __name__ == "__main__":
    main()
