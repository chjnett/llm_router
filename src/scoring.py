from __future__ import annotations

import math
import re


_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_FINAL_PATTERNS = (
    re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"(?:final answer|answer)\s*(?:is|:)?\s*\\?boxed\{?\s*([-+]?\d[\d,]*(?:\.\d+)?)", re.I),
)


def normalize_number(value: str) -> float | None:
    value = value.replace(",", "").strip().rstrip(".}")
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def extract_final_number(text: str) -> float | None:
    for pattern in _FINAL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return normalize_number(matches[-1])
    matches = _NUMBER.findall(text)
    return normalize_number(matches[-1]) if matches else None


def gsm8k_correct(prediction: str, reference: str) -> bool:
    predicted = extract_final_number(prediction)
    gold = extract_final_number(reference)
    if predicted is None or gold is None:
        return False
    return math.isclose(predicted, gold, rel_tol=0.0, abs_tol=1e-6)

