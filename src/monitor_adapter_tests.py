"""Publish aggregate adapter-test progress for the local report."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


RUNS = [f"adapter_{method}_seed_{seed}" for method in ("random", "cluster") for seed in (42, 43, 44)]


def rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def main() -> None:
    root = Path("artifacts")
    previous = -1
    while True:
        counts = {name: rows(root / "inference" / name / "test.jsonl") for name in RUNS}
        complete = sum(value >= 400 for value in counts.values())
        done_rows = sum(min(value, 400) for value in counts.values())
        active = next((name for name in RUNS if counts[name] < 400), "complete")
        stamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
        (root / "live_status.json").write_text(json.dumps({"phase": "adapter_test_evaluation", "state": "COMPLETE" if complete == 6 else "RUNNING", "updated_at": stamp, "completed": done_rows, "total": 2400, "counts": counts, "active_split": active, "elapsed_seconds": 0, "rate_per_minute": 0, "eta_seconds": None}, ensure_ascii=False, indent=2), encoding="utf-8")
        if complete != previous:
            with (root / "live_log.txt").open("a", encoding="utf-8") as handle:
                handle.write(f"[{stamp}] adapter_test_evaluation · {complete}/6 adapters complete · active {active}\n")
            previous = complete
        if complete == 6:
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
