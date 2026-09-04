"""Run the long, restartable output-length and serving-mode experiment matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .common import write_json


LOWER_MODELS = ("qwen_1_5b", "smollm2_360m", "smollm2_1_7b")
ALL_MODELS = LOWER_MODELS + ("qwen_7b",)


def conditions(input_path: str) -> list[dict]:
    result = []
    for batch_size, limit in ((8, 200), (1, 50)):
        for model in ALL_MODELS:
            result.append({"model": model, "mode": "task", "tokens": 512, "batch": batch_size, "limit": limit, "input": input_path})
        for model in LOWER_MODELS:
            for mode in ("micro_reasoning", "answer_only"):
                result.append({"model": model, "mode": mode, "tokens": 64, "batch": batch_size, "limit": limit, "input": input_path})
    return result


def report_path(output_dir: Path, condition: dict) -> Path:
    run_name = f"output_ablation_{condition['mode']}_b{condition['batch']}_{condition['limit']}"
    return output_dir / run_name / condition["model"] / "report.json"


def run_condition(output_dir: Path, condition: dict, log) -> int:
    run_name = f"output_ablation_{condition['mode']}_b{condition['batch']}_{condition['limit']}"
    command = [
        sys.executable, "-m", "src.run_model_screening",
        "--model-key", condition["model"],
        "--input", condition["input"],
        "--task-type", "numeric",
        "--prompt-mode", condition["mode"],
        "--run-name", run_name,
        "--output-dir", str(output_dir),
        "--limit", str(condition["limit"]),
        "--batch-size", str(condition["batch"]),
        "--max-new-tokens", str(condition["tokens"]),
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    return process.wait()


def aggregate(output_dir: Path, matrix: list[dict]) -> dict:
    rows = []
    for condition in matrix:
        path = report_path(output_dir, condition)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        rows.append({**condition, "metrics": report["metrics"], "report": str(path)})
    return {"completed": len(rows), "planned": len(matrix), "runs": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/data/validation.jsonl")
    parser.add_argument("--output-dir", default="artifacts/model_screening")
    parser.add_argument("--log", default="artifacts/logs/output_length_ablation.log")
    parser.add_argument("--summary", default="artifacts/results/output_length_ablation.json")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    matrix = conditions(args.input)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== run started {datetime.now().isoformat()} ===\n")
        for index, condition in enumerate(matrix, start=1):
            path = report_path(output_dir, condition)
            label = f"[{index}/{len(matrix)}] {condition['model']} {condition['mode']} b{condition['batch']} n{condition['limit']}"
            if path.exists() and not args.rerun:
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

