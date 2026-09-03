"""Build a GSM8K holdout that has no ID overlap with the 2k pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-data-dir", default="artifacts/data")
    parser.add_argument("--full-data-dir", default="artifacts/ksc_full/data")
    parser.add_argument("--full-inference-dir", default="artifacts/ksc_full/inference")
    parser.add_argument("--output-root", default="artifacts/fresh_holdout")
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=400)
    args = parser.parse_args()

    pilot_ids: set[str] = set()
    for path in Path(args.pilot_data_dir).glob("*.jsonl"):
        pilot_ids.update(row["id"] for row in read_jsonl(path))

    output_root = Path(args.output_root)
    counts = {"validation": args.validation_size, "test": args.test_size}
    manifest = {"pilot_ids_excluded": len(pilot_ids), "splits": {}}
    seen: set[str] = set()
    for split, count in counts.items():
        source = read_jsonl(Path(args.full_data_dir) / f"{split}.jsonl")
        selected = [row for row in source if row["id"] not in pilot_ids and row["id"] not in seen][:count]
        if len(selected) != count:
            raise RuntimeError(f"{split}: requested {count}, found {len(selected)} fresh examples")
        selected_ids = {row["id"] for row in selected}
        seen.update(selected_ids)
        write_jsonl(output_root / "data" / f"{split}.jsonl", selected)

        cached = read_jsonl(Path(args.full_inference_dir) / "lower_concise" / f"{split}.jsonl")
        by_id = {row["id"]: row for row in cached}
        missing = selected_ids - by_id.keys()
        if missing:
            raise RuntimeError(f"{split}: missing {len(missing)} cached Lower concise predictions")
        write_jsonl(output_root / "inference" / "lower_concise" / f"{split}.jsonl", [by_id[row["id"]] for row in selected])
        manifest["splits"][split] = {
            "count": len(selected),
            "pilot_overlap": len(selected_ids & pilot_ids),
            "source_available_after_exclusion": sum(row["id"] not in pilot_ids for row in source),
        }

    manifest["cross_split_overlap"] = len(
        {row["id"] for row in read_jsonl(output_root / "data" / "validation.jsonl")}
        & {row["id"] for row in read_jsonl(output_root / "data" / "test.jsonl")}
    )
    write_json(output_root / "manifest.json", manifest)
    print(output_root / "manifest.json")


if __name__ == "__main__":
    main()
