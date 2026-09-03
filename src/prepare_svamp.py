"""Prepare official SVAMP test and disjoint train-derived calibration splits."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import Dataset, load_dataset

from .common import load_config, write_json, write_jsonl


def convert(row: dict, split: str) -> dict:
    body = str(row["Body"]).strip()
    question = str(row["Question"]).strip()
    answer = str(row["Answer"]).strip()
    return {
        "id": f"svamp-{row['ID']}",
        "question": f"{body}\n{question}",
        "answer": f"#### {answer}",
        "split": split,
        "source_split": "test" if split == "test" else "train",
        "type": row.get("Type"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/svamp_cross_task.yaml")
    parser.add_argument("--output-dir", default="artifacts/svamp/data")
    parser.add_argument("--local-arrow-dir")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.local_arrow_dir:
        arrow_dir = Path(args.local_arrow_dir)
        train = list(Dataset.from_file(str(arrow_dir / "svamp-train.arrow")))
        test = list(Dataset.from_file(str(arrow_dir / "svamp-test.arrow")))
    else:
        dataset = load_dataset(cfg["dataset"]["name"])
        train = list(dataset["train"])
        test = list(dataset["test"])
    rng = random.Random(cfg["seed"])
    rng.shuffle(train)
    validation = train[: cfg["dataset"]["validation_size"]]
    offset = cfg["dataset"]["validation_size"]
    selection_end = offset + cfg["dataset"]["risk_selection_size"]
    certification_end = selection_end + cfg["dataset"]["risk_certification_size"]
    risk_selection = train[offset:selection_end]
    risk_certification = train[selection_end:certification_end]
    if len(test) != cfg["dataset"]["test_size"]:
        raise RuntimeError(f"Expected {cfg['dataset']['test_size']} official test rows, got {len(test)}")
    output = Path(args.output_dir)
    validation_rows = [convert(row, "validation") for row in validation]
    selection_rows = [convert(row, "risk_selection") for row in risk_selection]
    certification_rows = [convert(row, "risk_certification") for row in risk_certification]
    test_rows = [convert(row, "test") for row in test]
    groups = [validation_rows, selection_rows, certification_rows, test_rows]
    all_ids = [row["id"] for group in groups for row in group]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("SVAMP split overlap detected")
    write_jsonl(output / "validation.jsonl", validation_rows)
    write_jsonl(output / "risk_selection.jsonl", selection_rows)
    write_jsonl(output / "risk_certification.jsonl", certification_rows)
    write_jsonl(output / "test.jsonl", test_rows)
    write_json(output / "manifest.json", {
        "dataset": cfg["dataset"]["name"],
        "seed": cfg["seed"],
        "validation": {"count": len(validation_rows), "source": "official train"},
        "risk_selection": {"count": len(selection_rows), "source": "official train; policy selection only"},
        "risk_certification": {"count": len(certification_rows), "source": "official train; fixed-policy certification only"},
        "test": {"count": len(test_rows), "source": "official test", "selection": "all rows"},
        "cross_split_overlap": 0,
    })
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
