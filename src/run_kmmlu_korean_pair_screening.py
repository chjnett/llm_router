"""Screen a public Korean-specialized Lower/Upper pair on 50 KMMLU rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import write_json
from .run_mmlu_logit_screening import report_path, run_condition


MODELS = ("hyperclovax_0_5b", "exaone_2_4b")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/data/kmmlu_screening_200.jsonl")
    parser.add_argument("--output-dir", default="artifacts/kmmlu_korean_pair_screening")
    parser.add_argument("--output", default="artifacts/results/kmmlu_korean_pair_screening.json")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    runs = []
    failures = []
    for model in MODELS:
        condition = {
            "model": model,
            "batch": args.batch_size,
            "limit": args.limit,
            "input": args.input,
            "quantize_4bit": False,
        }
        try:
            report = run_condition(output_dir, condition)
            runs.append({**condition, "metrics": report["metrics"], "report": str(report_path(output_dir, condition))})
        except Exception as error:
            failures.append({"condition": condition, "error": repr(error)})
            print(f"FAILED {model}: {error}", flush=True)
            break
        write_json(args.output, {"completed": len(runs), "planned": len(MODELS), "runs": runs, "failures": failures})
    payload = {"completed": len(runs), "planned": len(MODELS), "runs": runs, "failures": failures}
    write_json(args.output, payload)
    print(f"DONE {len(runs)}/{len(MODELS)} failures={len(failures)} -> {args.output}")


if __name__ == "__main__":
    main()
