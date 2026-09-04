"""Summarize the post-hoc 3/4/5% capability-ensemble risk ladder."""

from __future__ import annotations

import json
from pathlib import Path

from .common import write_json


ROOTS = {
    0.03: "_03",
    0.04: "_04",
    0.05: "",
}


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for risk_limit, suffix in ROOTS.items():
        selection = load(f"artifacts/capability_ensemble/risk_bound_selection{suffix}.json")
        asdiv = load(
            f"artifacts/asdiv_external_test/results/capability_ensemble_risk_bound{suffix}.json"
        )
        mawps = load(
            f"artifacts/mawps_external_test/results/capability_ensemble_risk_bound{suffix}.json"
        )
        policy = selection["selected_policy"]
        rows.append({
            "validation_risk_limit": risk_limit,
            "policy": {
                "pre_threshold": policy["pre_threshold"],
                "post_threshold": policy["post_threshold"],
            },
            "validation": {
                "accuracy": policy["accuracy"],
                "cost_reduction": 1.0 - policy["normalized_cost"],
                "unsafe_upper_bound_95": policy["unsafe_upper_bound_95"],
                "all_gates_pass": selection["promotion_gate"],
            },
            "asdiv": {
                "accuracy": asdiv["accuracy"],
                "cost_reduction": asdiv["cost_reduction_vs_upper"],
                "unsafe_upper_bound_95": asdiv["unsafe_upper_bound_95"],
                "all_gates_pass": asdiv["deployment_gate"]["overall_pass"],
            },
            "mawps": {
                "accuracy": mawps["accuracy"],
                "cost_reduction": mawps["cost_reduction_vs_upper"],
                "unsafe_upper_bound_95": mawps["unsafe_upper_bound_95"],
                "all_gates_pass": mawps["deployment_gate"]["overall_pass"],
            },
            "both_external_gates_pass": bool(
                asdiv["deployment_gate"]["overall_pass"]
                and mawps["deployment_gate"]["overall_pass"]
            ),
        })
    payload = {
        "protocol": "post-hoc sensitivity analysis initiated after observing the MAWPS 5% policy failure; not a confirmatory selection",
        "rows": rows,
        "diagnostic_cross_dataset_candidate": next(
            (row for row in rows if row["both_external_gates_pass"]), None
        ),
        "confirmation_requirement": "freeze the diagnostic candidate and evaluate once on a third untouched dataset",
    }
    output = "artifacts/capability_ensemble/risk_bound_ladder.json"
    write_json(output, payload)
    print(output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
