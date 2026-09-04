"""Prepare the final untouched GSM8K reserve after all selection splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import read_jsonl, write_json, write_jsonl


def ids_in(directory: Path) -> set[str]:
    ids: set[str] = set()
    for path in directory.glob("*.jsonl") if directory.exists() else []:
        ids.update(row["id"] for row in read_jsonl(path))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", default="artifacts/ksc_full/data/test.jsonl")
    parser.add_argument("--source-lower", default="artifacts/ksc_full/inference/lower_concise/test.jsonl")
    parser.add_argument("--output-root", default="artifacts/reserved_certification")
    parser.add_argument("--count", type=int, default=188)
    args = parser.parse_args()

    exclude_dirs = [
        Path("artifacts/data"),
        Path("artifacts/fresh_holdout/data"),
        Path("artifacts/fresh_recertification/data"),
    ]
    excluded = set().union(*(ids_in(path) for path in exclude_dirs))
    source = read_jsonl(args.source_data)
    available = [row for row in source if row["id"] not in excluded]
    selected = [{**row, "split": "certification"} for row in available[: args.count]]
    if len(selected) != args.count:
        raise RuntimeError(f"requested {args.count}, found {len(selected)} untouched rows")
    selected_ids = {row["id"] for row in selected}
    lower_by_id = {row["id"]: row for row in read_jsonl(args.source_lower)}
    lower = [{**lower_by_id[row["id"]], "split": "certification"} for row in selected]
    output = Path(args.output_root)
    write_jsonl(output / "data" / "certification.jsonl", selected)
    write_jsonl(output / "inference" / "lower_concise" / "certification.jsonl", lower)
    write_json(output / "manifest.json", {
        "protocol": "final untouched reserve; frozen pre-triage policy; no threshold tuning",
        "count": len(selected),
        "source_available_after_all_exclusions": len(available),
        "excluded_id_count": len(excluded),
        "overlap_with_any_prior_split": len(selected_ids & excluded),
        "frozen_policy": {
            "pre_model": "semantic logistic regression",
            "pre_threshold": 0.66,
            "post_model": "confidence Extra Trees",
            "post_threshold": 0.56,
        },
    })
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
