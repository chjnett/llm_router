"""Run restartable Qwen1.5B FP16 checks against existing 4-bit baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .common import write_json


def conditions() -> list[dict]:
    return [
        {"domain": "mmlu", "input": "artifacts/data/mmlu_validation_200.jsonl", "task": "multiple_choice", "mode": "answer_only", "tokens": 16, "batch": 8, "limit": 200},
        {"domain": "mmlu", "input": "artifacts/data/mmlu_validation_200.jsonl", "task": "multiple_choice", "mode": "answer_only", "tokens": 16, "batch": 1, "limit": 50},
        {"domain": "math", "input": "artifacts/data/validation.jsonl", "task": "numeric", "mode": "task", "tokens": 256, "batch": 8, "limit": 200},
        {"domain": "math", "input": "artifacts/data/validation.jsonl", "task": "numeric", "mode": "task", "tokens": 256, "batch": 1, "limit": 50},
    ]


def run_name(condition: dict) -> str:
    return f"qwen_fp16_{condition['domain']}_{condition['tokens']}_b{condition['batch']}_{condition['limit']}"


def report_path(output_dir: Path, condition: dict) -> Path:
    return output_dir / run_name(condition) / "qwen_1_5b" / "report.json"


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
    parser.add_argument("--output-dir", default="artifacts/model_screening")
    parser.add_argument("--log", default="artifacts/logs/qwen_precision_ablation.log")
    parser.add_argument("--summary", default="artifacts/results/qwen_precision_ablation.json")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    matrix = conditions()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== run started {datetime.now().isoformat()} ===\n")
        for index, condition in enumerate(matrix, start=1):
            label = f"[{index}/{len(matrix)}] {condition['domain']} fp16 b{condition['batch']} n{condition['limit']}"
            if report_path(output_dir, condition).exists() and not args.rerun:
                print(f"SKIP {label}", flush=True)
                continue
            command = [
                sys.executable, "-m", "src.run_model_screening", "--model-key", "qwen_1_5b",
                "--input", condition["input"], "--task-type", condition["task"], "--prompt-mode", condition["mode"],
                "--run-name", run_name(condition), "--output-dir", str(output_dir), "--limit", str(condition["limit"]),
                "--batch-size", str(condition["batch"]), "--max-new-tokens", str(condition["tokens"]), "--no-4bit",
            ]
            print(f"START {label}", flush=True)
            log.write(f"\nSTART {label}\n")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
            if code:
                failures.append({"condition": condition, "exit_code": code})
            write_json(args.summary, {**aggregate(output_dir, matrix), "failures": failures})
    payload = {**aggregate(output_dir, matrix), "failures": failures}
    write_json(args.summary, payload)
    print(f"DONE {payload['completed']}/{payload['planned']} failures={len(failures)} -> {args.summary}")


if __name__ == "__main__":
    main()
