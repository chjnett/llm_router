"""Prepare a one-shot GSM8K recertification set disjoint from all prior experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import read_jsonl, write_json, write_jsonl


def collect_ids(directory: Path) -> set[str]:
    ids: set[str] = set()
    if directory.exists():
        for path in directory.glob("*.jsonl"):
            ids.update(row["id"] for row in read_jsonl(path))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-data-dir", default="artifacts/data")
    parser.add_argument("--prior-holdout-data-dir", default="artifacts/fresh_holdout/data")
    parser.add_argument("--source-data", default="artifacts/ksc_full/data/test.jsonl")
    parser.add_argument(
        "--source-lower",
        default="artifacts/ksc_full/inference/lower_concise/test.jsonl",
    )
    parser.add_argument("--output-root", default="artifacts/fresh_recertification")
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()

    pilot_ids = collect_ids(Path(args.pilot_data_dir))
    prior_holdout_ids = collect_ids(Path(args.prior_holdout_data_dir))
    excluded_ids = pilot_ids | prior_holdout_ids
    source_rows = read_jsonl(args.source_data)
    available = [row for row in source_rows if row["id"] not in excluded_ids]
    selected = available[: args.count]
    if len(selected) != args.count:
        raise RuntimeError(f"requested {args.count} rows, but only {len(selected)} unused rows remain")

    selected = [{**row, "split": "recertification"} for row in selected]
    selected_ids = {row["id"] for row in selected}
    lower_by_id = {row["id"]: row for row in read_jsonl(args.source_lower)}
    missing = selected_ids - lower_by_id.keys()
    if missing:
        raise RuntimeError(f"missing {len(missing)} cached Lower predictions")
    lower = [{**lower_by_id[row["id"]], "split": "recertification"} for row in selected]

    output_root = Path(args.output_root)
    write_jsonl(output_root / "data" / "recertification.jsonl", selected)
    write_jsonl(output_root / "inference" / "lower_concise" / "recertification.jsonl", lower)
    write_json(
        output_root / "manifest.json",
        {
            "protocol": "one-shot fresh recertification; no threshold selection on these rows",
            "count": len(selected),
            "source_count": len(source_rows),
            "source_available_after_exclusion": len(available),
            "pilot_ids_excluded": len(pilot_ids),
            "prior_holdout_ids_excluded": len(prior_holdout_ids),
            "pilot_overlap": len(selected_ids & pilot_ids),
            "prior_holdout_overlap": len(selected_ids & prior_holdout_ids),
            "policy_frozen_before_inference": {
                "low_threshold": 0.12,
                "high_threshold": 0.86,
            },
        },
    )
    print(output_root / "manifest.json")


if __name__ == "__main__":
    main()
