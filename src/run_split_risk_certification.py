"""Select a policy and certify its unsafe risk on a disjoint split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import load_config, write_json
from .run_risk_bound_calibration import binomial_upper, fit_gsm8k_confidence_model, load_target
from .run_selective_consistency import cascade


def count_unsafe(metrics: dict, size: int) -> int:
    return int(round(metrics["unsafe_routing_rate_all"] * size))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot_gsm8k.yaml")
    parser.add_argument("--pilot-root", default="artifacts")
    parser.add_argument("--target-root", default="artifacts/svamp")
    parser.add_argument("--upper-name", default="upper_concise")
    parser.add_argument("--selection-unsafe-cap", type=float, default=0.02)
    parser.add_argument("--risk-limit", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--quality-margin", type=float, default=0.02)
    parser.add_argument("--output", default="artifacts/results/svamp_split_risk_certification.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = fit_gsm8k_confidence_model(Path(args.pilot_root))
    root = Path(args.target_root)
    selection = load_target(model, root, "risk_selection", args.upper_name)
    certification = load_target(model, root, "risk_certification", args.upper_name)
    test = load_target(model, root, "test", args.upper_name)

    probability, agreement, lower, upper = selection
    quality_target = cfg["router"]["quality_floor"] * float(upper.mean()) + args.quality_margin
    candidates = []
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    for low_threshold in grid:
        for high_threshold in grid:
            if low_threshold > high_threshold:
                continue
            metrics = cascade(probability, agreement, lower, upper, low_threshold, high_threshold, cfg)
            candidates.append({"low_threshold": float(low_threshold), "high_threshold": float(high_threshold), **metrics})
    feasible = [
        row for row in candidates
        if row["task_accuracy"] >= quality_target
        and row["unsafe_routing_rate_all"] <= args.selection_unsafe_cap
    ]
    selected = min(feasible, key=lambda row: (row["normalized_cascade_cost"], -row["task_accuracy"])) if feasible else None
    payload = {
        "protocol": "policy selection and exact risk certification use disjoint target-task splits",
        "selection_constraints": {"quality_target": quality_target, "unsafe_point_cap": args.selection_unsafe_cap},
        "selected_on_risk_selection": selected,
    }
    if selected is not None:
        cert_probability, cert_agreement, cert_lower, cert_upper = certification
        cert_metrics = cascade(cert_probability, cert_agreement, cert_lower, cert_upper, selected["low_threshold"], selected["high_threshold"], cfg)
        cert_unsafe = count_unsafe(cert_metrics, len(cert_lower))
        cert_upper_bound = binomial_upper(cert_unsafe, len(cert_lower), args.alpha)
        certified = cert_upper_bound <= args.risk_limit
        payload["risk_certification"] = {
            "count": len(cert_lower),
            "unsafe_count": cert_unsafe,
            "unsafe_upper_bound_95": cert_upper_bound,
            "certified_at_5pct": certified,
            **cert_metrics,
        }
        test_probability, test_agreement, test_lower, test_upper = test
        test_metrics = cascade(test_probability, test_agreement, test_lower, test_upper, selected["low_threshold"], selected["high_threshold"], cfg)
        payload["official_test_diagnostic"] = {
            "count": len(test_lower),
            "always_upper_accuracy": float(test_upper.mean()),
            "unsafe_count": count_unsafe(test_metrics, len(test_lower)),
            **test_metrics,
        }
        payload["deployment_gate"] = {
            "risk_certificate_pass": certified,
            "quality_floor_pass": cert_metrics["task_accuracy"] >= cfg["router"]["quality_floor"] * float(cert_upper.mean()),
            "cost_reduction_10pct_pass": cert_metrics["normalized_cascade_cost"] <= 0.90,
        }
        payload["deployment_gate"]["overall_pass"] = all(payload["deployment_gate"].values())
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
