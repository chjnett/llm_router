"""Write browser-friendly live status for a cached inference run."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


SPLITS = {"router_train": 1000, "distill_train": 400, "validation": 200, "test": 400}


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--data-dir", default="artifacts/data")
    parser.add_argument("--inference-dir", default="artifacts/inference")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    root = Path("artifacts")
    run_dir = Path(args.inference_dir) / args.output_name
    status_path = root / "live_status.json"
    log_path = root / "live_log.txt"
    selected_splits = args.splits or list(SPLITS)
    expected = {}
    for name in selected_splits:
        data_path = Path(args.data_dir) / f"{name}.jsonl"
        expected[name] = count_rows(data_path) if data_path.exists() else SPLITS[name]
    total = sum(expected.values())
    started = time.monotonic()
    previous_count = -1
    samples: list[tuple[float, int]] = []

    while True:
        counts = {split: count_rows(run_dir / f"{split}.jsonl") for split in expected}
        completed = sum(counts.values())
        now = time.monotonic()
        samples.append((now, completed))
        samples = samples[-13:]
        rate = 0.0
        if len(samples) > 1:
            duration = samples[-1][0] - samples[0][0]
            gained = samples[-1][1] - samples[0][1]
            if duration > 0 and gained > 0:
                rate = gained / duration
        remaining = (total - completed) / rate if rate else None
        stamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
        active = next((name for name in expected if counts[name] < expected[name]), "complete")
        status = {
            "phase": args.phase,
            "state": "COMPLETE" if completed >= total else "RUNNING",
            "updated_at": stamp,
            "completed": completed,
            "total": total,
            "counts": counts,
            "active_split": active,
            "elapsed_seconds": round(now - started),
            "rate_per_minute": round(rate * 60, 1),
            "eta_seconds": round(remaining) if remaining is not None else None,
        }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        if completed != previous_count:
            eta_text = f" · ETA {round(remaining / 60)} min" if remaining is not None else " · ETA calculating"
            line = f"[{stamp}] {args.phase} · {active} · {completed}/{total}{eta_text}\n"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
            previous_count = completed
        if completed >= total:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
