"""Expose sequential LoRA training progress to the local HTML dashboard."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


RUNS = [f"{method}_seed_{seed}" for method in ("random", "cluster") for seed in (42, 43, 44)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--started-unix", type=float, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    root = Path("artifacts")
    adapters = root / "adapters"
    status_path, log_path = root / "live_status.json", root / "live_log.txt"
    previous = -1
    while True:
        finished = [name for name in RUNS if (adapters / name / "training_metrics.json").exists() and (adapters / name / "training_metrics.json").stat().st_mtime >= args.started_unix]
        completed = len(finished)
        active = next((name for name in RUNS if name not in finished), "complete")
        stamp = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
        status_path.write_text(json.dumps({"phase": "targeted_kd_lora_training", "state": "COMPLETE" if completed == len(RUNS) else "RUNNING", "updated_at": stamp, "completed": completed, "total": len(RUNS), "counts": {name: int(name in finished) for name in RUNS}, "active_split": active, "elapsed_seconds": 0, "rate_per_minute": 0, "eta_seconds": None}, ensure_ascii=False, indent=2), encoding="utf-8")
        if completed != previous:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{stamp}] targeted_kd_lora_training · {completed}/6 complete · next {active}\n")
            previous = completed
        if completed == len(RUNS):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
