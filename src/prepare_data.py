from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import load_config, set_seed, write_json, write_jsonl


def split_rows(rows: list[dict], seed: int, fractions: dict[str, float]) -> dict[str, list[dict]]:
    if not rows:
        raise ValueError("dataset is empty")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    indices = np.random.default_rng(seed).permutation(len(rows)).tolist()
    names = ["router_train", "distill_train", "validation"]
    counts = {name: int(len(rows) * fractions[name]) for name in names}
    counts["test"] = len(rows) - sum(counts.values())
    cursor = 0
    selected: dict[str, list[int]] = {}
    for name in [*names, "test"]:
        selected[name] = indices[cursor : cursor + counts[name]]
        cursor += counts[name]
    groups = {
        "router_train": selected["router_train"],
        "distill_train": selected["distill_train"],
        "validation": selected["validation"],
        "test": selected["test"],
    }
    result: dict[str, list[dict]] = {}
    for split, selected in groups.items():
        result[split] = [dict(rows[index], id=f"gsm8k-{index:05d}", split=split) for index in selected]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--output", default="artifacts/data")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    from datasets import load_dataset

    dataset = load_dataset(cfg["dataset"]["name"], cfg["dataset"]["config"], split="train")
    rows = [dict(row) for row in dataset]
    max_examples = cfg["dataset"].get("max_examples")
    if max_examples:
        rows = rows[: int(max_examples)]
    splits = split_rows(rows, cfg["seed"], cfg["dataset"]["splits"])
    output = Path(args.output)
    ids: set[str] = set()
    manifest = {"seed": cfg["seed"], "counts": {}}
    for name, split_rows_ in splits.items():
        split_ids = {row["id"] for row in split_rows_}
        if ids & split_ids:
            raise RuntimeError("data leakage: duplicate IDs across splits")
        ids |= split_ids
        write_jsonl(output / f"{name}.jsonl", split_rows_)
        manifest["counts"][name] = len(split_rows_)
    write_json(output / "manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
