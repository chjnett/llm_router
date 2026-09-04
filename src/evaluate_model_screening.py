"""Apply prespecified model-pair screening gates to measured reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import load_config, write_json


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_pair(lower: dict, upper: dict, gates: dict) -> dict:
    low = lower["metrics"]
    high = upper["metrics"]
    accuracy_gap = float(high["accuracy"] - low["accuracy"])
    cost_ratio = float(low["latency_ms_p50"] / high["latency_ms_p50"])
    parse_success = min(float(low["parse_success_rate"]), float(high["parse_success_rate"]))
    peak_vram = max(float(low["peak_vram_reserved_gb"]), float(high["peak_vram_reserved_gb"]))
    token_limit_rate = max(float(low.get("token_limit_rate", 0.0)), float(high.get("token_limit_rate", 0.0)))
    checks = {
        "accuracy_gap": {
            "value": accuracy_gap,
            "threshold": float(gates["minimum_accuracy_gap"]),
            "pass": accuracy_gap >= float(gates["minimum_accuracy_gap"]),
        },
        "measured_cost_ratio": {
            "value": cost_ratio,
            "threshold": float(gates["maximum_cost_ratio"]),
            "pass": cost_ratio < float(gates["maximum_cost_ratio"]),
        },
        "parse_success": {
            "value": parse_success,
            "threshold": float(gates["minimum_parse_success"]),
            "pass": parse_success >= float(gates["minimum_parse_success"]),
        },
        "peak_vram_gb": {
            "value": peak_vram,
            "threshold": float(gates["maximum_peak_vram_gb"]),
            "pass": peak_vram <= float(gates["maximum_peak_vram_gb"]),
        },
        "token_limit_rate": {
            "value": token_limit_rate,
            "threshold": float(gates["maximum_token_limit_rate"]),
            "pass": token_limit_rate <= float(gates["maximum_token_limit_rate"]),
        },
    }
    return {
        "lower_model": lower["model"],
        "upper_model": upper["model"],
        "sample_count": min(int(low["items"]), int(high["items"])),
        "lower_accuracy": low["accuracy"],
        "upper_accuracy": high["accuracy"],
        "lower_generated_tokens_mean": low["generated_tokens_mean"],
        "upper_generated_tokens_mean": high["generated_tokens_mean"],
        "checks": checks,
        "screening_pass": all(check["pass"] for check in checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lower-report", required=True)
    parser.add_argument("--upper-report", required=True)
    parser.add_argument("--config", default="configs/model_domain_screening.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = evaluate_pair(read_json(args.lower_report), read_json(args.upper_report), cfg["screening"]["gates"])
    write_json(args.output, result)
    print(f"screening_pass={result['screening_pass']} -> {args.output}")


if __name__ == "__main__":
    main()
