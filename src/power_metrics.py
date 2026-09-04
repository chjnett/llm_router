"""Lightweight gross GPU energy sampling through nvidia-smi."""

from __future__ import annotations

import subprocess
import threading


def summarize_power(samples: list[float], elapsed_seconds: float) -> dict:
    if not samples:
        return {"power_samples": 0, "power_watts_mean": None, "power_watts_max": None, "gross_energy_joules": None}
    mean = sum(samples) / len(samples)
    return {
        "power_samples": len(samples),
        "power_watts_mean": mean,
        "power_watts_max": max(samples),
        "gross_energy_joules": mean * elapsed_seconds,
    }


class PowerSampler:
    def __init__(self, interval_seconds: float = 0.5):
        self.interval_seconds = interval_seconds
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def read_watts() -> float | None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return float(result.stdout.strip().splitlines()[0])
        except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def _sample(self) -> None:
        while not self._stop.is_set():
            value = self.read_watts()
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self, elapsed_seconds: float) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_seconds * 2, 1.0))
        return summarize_power(self.samples, elapsed_seconds)

