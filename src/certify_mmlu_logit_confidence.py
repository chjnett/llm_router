"""Evaluate the frozen M0 option-logit confidence policy on independent splits."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .analyze_mmlu_logit_confidence import FEATURES
from .common import read_jsonl, write_json
from .run_risk_bound_calibration import binomial_upper


def evaluate(probability, lower, upper, threshold: float, latency_ratio: float) -> dict:
    accept = probability >= threshold
    correct = np.where(accept, lower, upper)
    unsafe = accept & ~lower & upper
    accepted = int(accept.sum())
    upper_accuracy = float(upper.mean())
    normalized_latency = float(latency_ratio + 1.0 - accept.mean())
    return {
        "items": len(lower), "threshold": threshold,
        "lower_accuracy": float(lower.mean()), "upper_accuracy": upper_accuracy,
        "acceptance_rate": float(accept.mean()), "system_accuracy": float(correct.mean()),
        "quality_retention": float(correct.mean() / upper_accuracy) if upper_accuracy else 0.0,
        "normalized_latency": normalized_latency, "latency_reduction": 1.0 - normalized_latency,
        "unsafe_count": int(unsafe.sum()),
        "unsafe_rate_given_accept": float(unsafe.sum() / accepted) if accepted else 0.0,
        "unsafe_exact_95_upper": float(binomial_upper(int(unsafe.sum()), accepted)) if accepted else 1.0,
        "performance_gate_pass": bool(correct.mean() >= 0.95 * upper_accuracy and normalized_latency <= 0.90),
        "strict_risk_gate_pass": bool(accepted and binomial_upper(int(unsafe.sum()), accepted) <= 0.05),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-features", default="artifacts/confidence/mmlu_logit/qwen_1_5b.jsonl")
    parser.add_argument("--independent-features", default="artifacts/confidence/mmlu_logit_independent/qwen_1_5b.jsonl")
    parser.add_argument("--upper-predictions", default="artifacts/model_screening/mmlu_logit_4bit_b8_500/qwen_7b/predictions.jsonl")
    parser.add_argument("--threshold", type=float, default=0.545)
    parser.add_argument("--latency-ratio", type=float, default=0.24704482170737938)
    parser.add_argument("--certification-size", type=int, default=250)
    parser.add_argument("--output", default="paper/data/mmlu_logit_independent_certification.json")
    args = parser.parse_args()

    selection = read_jsonl(args.selection_features)
    independent = read_jsonl(args.independent_features)
    upper_by_id = {row["id"]: row for row in read_jsonl(args.upper_predictions)}
    x_train = np.asarray([[float(row[name]) for name in FEATURES] for row in selection], dtype=np.float32)
    y_train = np.asarray([bool(row["correct"]) for row in selection])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
    )
    model.fit(x_train, y_train)
    x_test = np.asarray([[float(row[name]) for name in FEATURES] for row in independent], dtype=np.float32)
    probability = model.predict_proba(x_test)[:, 1]
    lower = np.asarray([bool(row["correct"]) for row in independent])
    upper = np.asarray([bool(upper_by_id[row["id"]]["correct"]) for row in independent])
    cut = args.certification_size
    payload = {
        "protocol": "frozen model/features/threshold from MMLU validation; disjoint subject-balanced MMLU test rows",
        "selection_rows": len(selection), "threshold": args.threshold, "latency_ratio": args.latency_ratio,
        "certification": evaluate(probability[:cut], lower[:cut], upper[:cut], args.threshold, args.latency_ratio),
        "final_test": evaluate(probability[cut:], lower[cut:], upper[cut:], args.threshold, args.latency_ratio),
    }
    payload["overall_decision"] = (
        "confirmed" if payload["certification"]["performance_gate_pass"] and payload["final_test"]["performance_gate_pass"]
        else "not_confirmed"
    )
    write_json(args.output, payload)
    print(f"decision={payload['overall_decision']} cert={payload['certification']} test={payload['final_test']}")


if __name__ == "__main__":
    main()
