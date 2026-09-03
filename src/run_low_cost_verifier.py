"""Select and independently certify an answer-only verifier cascade."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import load_config, read_jsonl, write_json
from .run_fresh_c3_gate import feature_matrix, keyed
from .run_risk_bound_calibration import binomial_upper, fit_gsm8k_confidence_model


def verifier_cascade(probability, agreement, lower, upper, low, high, cfg, verifier_cost=None):
    direct_accept = probability >= high
    verify = (probability >= low) & (probability < high)
    agreement_accept = verify & agreement
    accept_lower = direct_accept | agreement_accept
    escalate = ~accept_lower
    final_correct = np.where(accept_lower, lower, upper)
    lower_cost = float(cfg["cost"]["lower"])
    upper_cost = float(cfg["cost"]["upper"])
    check_cost = float(cfg["cost"]["verifier"] if verifier_cost is None else verifier_cost)
    total_cost = lower_cost + verify.astype(float) * check_cost + escalate.astype(float) * upper_cost
    return {
        "task_accuracy": float(final_correct.mean()),
        "lower_coverage": float(accept_lower.mean()),
        "direct_accept_rate": float(direct_accept.mean()),
        "verifier_call_rate": float(verify.mean()),
        "agreement_accept_rate": float(agreement_accept.mean()),
        "upper_call_rate": float(escalate.mean()),
        "unsafe_routing_rate_all": float((accept_lower & ~lower).mean()),
        "normalized_cascade_cost": float(total_cost.mean() / upper_cost),
    }


def load_target(model, root: Path, split: str, upper_name: str, verifier_name: str):
    ids = [row["id"] for row in read_jsonl(root / "data" / f"{split}.jsonl")]
    probability = model.predict_proba(feature_matrix(root / "confidence" / "lower_concise" / f"{split}.jsonl", ids))[:, 1]
    lower_rows = keyed(root / "inference" / "lower_concise" / f"{split}.jsonl")
    verifier_rows = keyed(root / "inference" / verifier_name / f"{split}.jsonl")
    upper_rows = keyed(root / "inference" / upper_name / f"{split}.jsonl")
    lower = np.asarray([lower_rows[row_id]["correct"] for row_id in ids], dtype=bool)
    upper = np.asarray([upper_rows[row_id]["correct"] for row_id in ids], dtype=bool)
    agreement = np.asarray([
        lower_rows[row_id].get("predicted_number") is not None
        and lower_rows[row_id].get("predicted_number") == verifier_rows[row_id].get("predicted_number")
        for row_id in ids
    ], dtype=bool)
    verifier_tokens = [int(verifier_rows[row_id].get("generated_tokens", 0)) for row_id in ids]
    return probability, agreement, lower, upper, verifier_tokens


def evaluate(inputs, low, high, cfg, verifier_cost=None):
    probability, agreement, lower, upper, _ = inputs
    return verifier_cascade(probability, agreement, lower, upper, low, high, cfg, verifier_cost)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/svamp_cross_task.yaml")
    parser.add_argument("--pilot-root", default="artifacts")
    parser.add_argument("--target-root", default="artifacts/svamp")
    parser.add_argument("--upper-name", default="upper_concise")
    parser.add_argument("--verifier-name", default="lower_answer_only")
    parser.add_argument("--selection-unsafe-cap", type=float, default=0.02)
    parser.add_argument("--risk-limit", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--quality-margin", type=float, default=0.02)
    parser.add_argument("--output", default="artifacts/results/svamp_low_cost_verifier.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = fit_gsm8k_confidence_model(Path(args.pilot_root))
    root = Path(args.target_root)
    selection = load_target(model, root, "risk_selection", args.upper_name, args.verifier_name)
    certification = load_target(model, root, "risk_certification", args.upper_name, args.verifier_name)
    test = load_target(model, root, "test", args.upper_name, args.verifier_name)

    quality_target = cfg["router"]["quality_floor"] * float(selection[3].mean()) + args.quality_margin
    candidates = []
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    for low in grid:
        for high in grid:
            if low > high:
                continue
            metrics = evaluate(selection, low, high, cfg)
            candidates.append({"low_threshold": float(low), "high_threshold": float(high), **metrics})
    feasible = [
        row for row in candidates
        if row["task_accuracy"] >= quality_target
        and row["unsafe_routing_rate_all"] <= args.selection_unsafe_cap
    ]
    selected = min(feasible, key=lambda row: (row["normalized_cascade_cost"], -row["task_accuracy"])) if feasible else None
    payload = {
        "protocol": "answer-only verifier policy selection and disjoint exact risk certification",
        "assumed_verifier_cost_lower_equivalents": cfg["cost"]["verifier"],
        "selection_constraints": {"quality_target": quality_target, "unsafe_point_cap": args.selection_unsafe_cap},
        "selected_on_risk_selection": selected,
    }
    if selected is not None:
        low, high = selected["low_threshold"], selected["high_threshold"]
        cert_metrics = evaluate(certification, low, high, cfg)
        cert_unsafe = int(round(cert_metrics["unsafe_routing_rate_all"] * len(certification[2])))
        cert_bound = binomial_upper(cert_unsafe, len(certification[2]), args.alpha)
        verifier_rate = cert_metrics["verifier_call_rate"]
        upper_rate = cert_metrics["upper_call_rate"]
        upper_cost = float(cfg["cost"]["upper"])
        lower_cost = float(cfg["cost"]["lower"])
        break_even = max(0.0, (0.9 * upper_cost - lower_cost - upper_rate * upper_cost) / verifier_rate) if verifier_rate else 0.0
        payload["risk_certification"] = {
            "count": len(certification[2]),
            "unsafe_count": cert_unsafe,
            "unsafe_upper_bound_95": cert_bound,
            "certified_at_5pct": cert_bound <= args.risk_limit,
            "answer_only_generated_tokens_mean": float(np.mean(certification[4])),
            "answer_only_generated_tokens_p95": float(np.quantile(certification[4], 0.95)),
            "verifier_cost_break_even_for_10pct_saving": break_even,
            "cost_sensitivity": {
                str(cost): evaluate(certification, low, high, cfg, cost)["normalized_cascade_cost"]
                for cost in (0.2, 0.35, 0.5, 0.75, 1.0)
            },
            **cert_metrics,
        }
        test_metrics = evaluate(test, low, high, cfg)
        payload["official_test_diagnostic"] = {
            "count": len(test[2]),
            "always_upper_accuracy": float(test[3].mean()),
            "unsafe_count": int(round(test_metrics["unsafe_routing_rate_all"] * len(test[2]))),
            "answer_only_generated_tokens_mean": float(np.mean(test[4])),
            **test_metrics,
        }
        payload["deployment_gate"] = {
            "risk_certificate_pass": cert_bound <= args.risk_limit,
            "quality_floor_pass": cert_metrics["task_accuracy"] >= cfg["router"]["quality_floor"] * float(certification[3].mean()),
            "cost_reduction_10pct_pass": cert_metrics["normalized_cascade_cost"] <= 0.90,
        }
        payload["deployment_gate"]["overall_pass"] = all(payload["deployment_gate"].values())
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
