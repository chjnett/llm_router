"""Diagnose an unlabeled domain-level cascade break-even guard."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .common import load_config, write_json


DEFAULT_RESULTS = [
    "artifacts/asdiv_external_test/results/capability_ensemble_risk_bound_04.json",
    "artifacts/mawps_external_test/results/capability_ensemble_risk_bound_04.json",
    "artifacts/math500_numeric_test/results/capability_ensemble_frozen_04.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--results", nargs="+", default=DEFAULT_RESULTS)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="artifacts/capability_ensemble/feasibility_guard.json",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    upper_cost = float(cfg["cost"]["upper"])
    lower_cost = float(cfg["cost"]["lower"])
    cost_ratio = lower_cost / upper_cost
    rows = []
    for result_path in args.results:
        result = json.loads(Path(result_path).read_text(encoding="utf-8"))
        count = int(result["test_count"])
        observed_saving = float(result["lower_coverage"]) - (
            float(result["lower_call_rate"]) * cost_ratio
        )
        reported_saving = float(result["cost_reduction_vs_upper"])
        if not math.isclose(observed_saving, reported_saving, abs_tol=1e-9):
            raise ValueError(f"cost identity mismatch in {result_path}")
        # Per-request normalized saving lies in [-C_L/C_U, 1-C_L/C_U],
        # whose range is exactly one. Hoeffding therefore gives this bound.
        radius = math.sqrt(math.log(1.0 / args.alpha) / (2.0 * count))
        lower_bound = observed_saving - radius
        use_cascade = lower_bound > 0.0
        rows.append({
            "dataset": result["dataset"],
            "count": count,
            "lower_call_rate": result["lower_call_rate"],
            "lower_acceptance_rate": result["lower_coverage"],
            "observed_cost_reduction": observed_saving,
            "hoeffding_radius": radius,
            "cost_reduction_lower_bound_95": lower_bound,
            "guard_decision": "cascade" if use_cascade else "always_upper",
            "guarded_accuracy": result["accuracy"] if use_cascade else result["always_upper_accuracy"],
            "guarded_normalized_cost": result["normalized_cost"] if use_cascade else 1.0,
            "avoided_cost_overhead": max(0.0, float(result["normalized_cost"]) - 1.0)
            if not use_cascade else 0.0,
        })
    payload = {
        "protocol": "post-hoc architecture diagnostic; the guard uses only observable route decisions and model cost, not answer correctness",
        "alpha": args.alpha,
        "lower_to_upper_cost_ratio": cost_ratio,
        "decision_rule": "enable cascade only when the Hoeffding 95% lower bound on normalized cost reduction is greater than zero",
        "rows": rows,
        "all_guarded_costs_non_increasing": all(
            row["guarded_normalized_cost"] <= 1.0 for row in rows
        ),
        "confirmation_requirement": "freeze the guard and evaluate on new traffic or another untouched task",
    }
    write_json(args.output, payload)
    print(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
