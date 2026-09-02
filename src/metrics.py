from __future__ import annotations

import numpy as np


def routing_labels(lower_correct: np.ndarray, upper_correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower_correct = np.asarray(lower_correct, dtype=bool)
    upper_correct = np.asarray(upper_correct, dtype=bool)
    eligible = lower_correct | upper_correct
    labels = lower_correct.astype(int)
    return labels, eligible


def system_metrics(
    route_lower: np.ndarray,
    lower_correct: np.ndarray,
    upper_correct: np.ndarray,
    lower_cost: float,
    upper_cost: float,
) -> dict[str, float]:
    route_lower = np.asarray(route_lower, dtype=bool)
    lower_correct = np.asarray(lower_correct, dtype=bool)
    upper_correct = np.asarray(upper_correct, dtype=bool)
    final_correct = np.where(route_lower, lower_correct, upper_correct)
    costs = np.where(route_lower, lower_cost, upper_cost)
    return {
        "task_accuracy": float(final_correct.mean()),
        "lower_coverage": float(route_lower.mean()),
        "upper_call_rate": float((~route_lower).mean()),
        "unsafe_routing_rate_all": float((route_lower & ~lower_correct).mean()),
        "unsafe_given_lower": float((route_lower & ~lower_correct).sum() / max(route_lower.sum(), 1)),
        "both_wrong_rate": float((~lower_correct & ~upper_correct).mean()),
        "normalized_cost": float(costs.mean() / upper_cost),
    }


def select_operating_point(points: list[dict], always_upper_accuracy: float, quality_floor: float) -> dict:
    minimum = quality_floor * always_upper_accuracy
    feasible = [point for point in points if point["task_accuracy"] >= minimum]
    if not feasible:
        return min(points, key=lambda point: (-point["task_accuracy"], point["normalized_cost"]))
    return min(feasible, key=lambda point: (point["normalized_cost"], -point["task_accuracy"]))

