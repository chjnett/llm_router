"""Score MBPP generations in guarded, isolated child processes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from .common import read_jsonl, write_json


FENCE = re.compile(r"^\s*```(?:python)?\s*|\s*```\s*$", re.IGNORECASE)


def clean_code(text: str) -> str:
    return FENCE.sub("", text.strip()).strip()


def judge(rows: list[dict], predictions: list[dict], timeout: float) -> dict:
    by_id = {row["id"]: row for row in predictions}
    worker = Path(__file__).with_name("mbpp_worker.py")
    details = []
    for row in rows:
        payload = {
            "code": clean_code(by_id[row["id"]]["prediction"]),
            "test_imports": row["task_metadata"]["test_imports"],
            "tests": row["task_metadata"]["test_list"],
        }
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(worker)],
                input=json.dumps(payload), text=True, capture_output=True,
                timeout=timeout, check=False,
            )
            try:
                message = json.loads(result.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                message = {"passed": False, "error": "WorkerProtocolError"}
        except subprocess.TimeoutExpired:
            message = {"passed": False, "error": "TimeoutExpired"}
        details.append({"id": row["id"], **message})
    passed = [bool(row["passed"]) for row in details]
    return {
        "pass_at_1": sum(passed) / len(passed),
        "correct": passed,
        "timeouts": sum(row.get("error") == "TimeoutExpired" for row in details),
        "guard_rejections": sum(row.get("error") == "PermissionError" for row in details),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/data/mbpp_screening_50.jsonl")
    parser.add_argument("--lower-dir", required=True)
    parser.add_argument("--upper-dir", required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", default="paper/data/mbpp_screening_analysis.json")
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    lower_predictions = read_jsonl(Path(args.lower_dir) / "predictions.jsonl")
    upper_predictions = read_jsonl(Path(args.upper_dir) / "predictions.jsonl")
    with (Path(args.lower_dir) / "report.json").open("r", encoding="utf-8") as handle:
        lower_report = json.load(handle)
    with (Path(args.upper_dir) / "report.json").open("r", encoding="utf-8") as handle:
        upper_report = json.load(handle)
    lower = judge(rows, lower_predictions, args.timeout)
    upper = judge(rows, upper_predictions, args.timeout)
    latency_ratio = lower_report["metrics"]["latency_ms_p50"] / upper_report["metrics"]["latency_ms_p50"]
    oracle = sum(a or b for a, b in zip(lower["correct"], upper["correct"])) / len(rows)
    oracle_latency = latency_ratio + 1.0 - lower["pass_at_1"]
    checks = {
        "upper_accuracy_advantage_5pp": upper["pass_at_1"] - lower["pass_at_1"] >= 0.05,
        "latency_ratio_below_0_5": latency_ratio < 0.5,
        "oracle_quality_95pct": oracle >= 0.95 * upper["pass_at_1"],
        "oracle_latency_10pct_reduction": oracle_latency <= 0.9,
        "token_limit_below_10pct": lower_report["metrics"]["token_limit_rate"] <= 0.1,
    }
    payload = {
        "protocol": f"exploratory guarded-subprocess MBPP sanitized screen on {len(rows)} rows",
        "lower": {"model": lower_report["model"], "evaluation": lower, "metrics": lower_report["metrics"]},
        "upper": {"model": upper_report["model"], "evaluation": upper, "metrics": upper_report["metrics"]},
        "latency_ratio": latency_ratio,
        "oracle_pass_at_1": oracle,
        "oracle_normalized_latency": oracle_latency,
        "checks": checks,
        "continue_to_200": all(checks.values()),
        "security_note": "AST allowlist plus Python -I -S child process and per-item timeout; not an OS security boundary",
    }
    write_json(args.output, payload)
    print(f"continue_to_200={payload['continue_to_200']} -> {args.output}")


if __name__ == "__main__":
    main()
