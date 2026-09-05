"""Run restartable short-answer MMLU screening across public model families."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .common import write_json


MODELS = ("qwen_1_5b", "qwen_7b", "smollm2_360m", "smollm2_1_7b")


def conditions(input_path: str) -> list[dict]:
    return [
        {"model": model, "batch": batch, "limit": limit, "input": input_path}
        for batch, limit in ((8, 200), (1, 50))
        for model in MODELS
    ]


def run_name(condition: dict) -> str:
    return f"mmlu_short_b{condition['batch']}_{condition['limit']}"


def report_path(output_dir: Path, condition: dict) -> Path:
    return output_dir / run_name(condition) / condition["model"] / "report.json"


def run_condition(output_dir: Path, condition: dict, log) -> int:
    command = [
        sys.executable, "-m", "src.run_model_screening",
        "--model-key", condition["model"],
        "--input", condition["input"],
        "--task-type", "multiple_choice",
        "--prompt-mode", "answer_only",
        "--run-name", run_name(condition),
        "--output-dir", str(output_dir),
        "--limit", str(condition["limit"]),
        "--batch-size", str(condition["batch"]),
        "--max-new-tokens", "16",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    return process.wait()


def aggregate(output_dir: Path, matrix: list[dict]) -> dict:
    runs = []
    for condition in matrix:
        path = report_path(output_dir, condition)
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
            runs.append({**condition, "metrics": report["metrics"], "report": str(path)})
    return {"completed": len(runs), "planned": len(matrix), "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/data/mmlu_validation_200.jsonl")
    parser.add_argument("--output-dir", default="artifacts/model_screening")
    parser.add_argument("--log", default="artifacts/logs/mmlu_short_answer_screening.log")
    parser.add_argument("--summary", default="artifacts/results/mmlu_short_answer_screening.json")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    matrix = conditions(args.input)
    output_dir = Path(args.output_dir)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== run started {datetime.now().isoformat()} ===\n")
        for index, condition in enumerate(matrix, start=1):
            label = f"[{index}/{len(matrix)}] {condition['model']} b{condition['batch']} n{condition['limit']}"
            if report_path(output_dir, condition).exists() and not args.rerun:
                print(f"SKIP {label}", flush=True)
                continue
            print(f"START {label}", flush=True)
            log.write(f"\nSTART {label}\n")
            code = run_condition(output_dir, condition, log)
            if code:
                failures.append({"condition": condition, "exit_code": code})
                print(f"FAILED {label} exit={code}", flush=True)
            write_json(args.summary, {**aggregate(output_dir, matrix), "failures": failures})
    payload = {**aggregate(output_dir, matrix), "failures": failures}
    write_json(args.summary, payload)
    print(f"DONE {payload['completed']}/{payload['planned']} failures={len(failures)} -> {args.summary}")


if __name__ == "__main__":
    main()
