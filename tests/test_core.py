import numpy as np

from src.metrics import routing_labels, select_operating_point, system_metrics
from src.prepare_data import split_rows
from src.scoring import extract_final_number, gsm8k_correct
from src.run_selective_consistency import cascade
from src.run_risk_bound_calibration import binomial_upper
from src.run_low_cost_verifier import verifier_cascade
from src.benchmark_verifier_latency import completion_lengths
from src.analyze_verifier_performance import route_items
from src.model_registry import get_model_spec
from src.task_harness import adapt_row, extract_choice, score_prediction, system_prompt
from src.run_model_screening import summarize, validate_rows
from src.evaluate_model_screening import evaluate_pair
from src.power_metrics import summarize_power
from src.run_output_length_ablation import conditions
from src.analyze_output_length_ablation import analyze
from src.analyze_answer_only_confidence import select_policy
from src.run_token_budget_sweep import conditions as token_budget_conditions


def test_gsm8k_scoring_uses_final_answer():
    assert extract_final_number("work 3 then 7. Answer: 1,024") == 1024
    assert gsm8k_correct("The answer is \\boxed{42}", "reasoning\n#### 42")
    assert not gsm8k_correct("Answer: 41", "#### 42")
    assert extract_final_number("Final answer: -$76") == -76


def test_splits_are_disjoint_and_complete():
    rows = [{"question": str(index), "answer": str(index)} for index in range(100)]
    fractions = {"router_train": 0.5, "distill_train": 0.2, "validation": 0.1, "test": 0.2}
    result = split_rows(rows, 42, fractions)
    ids = [row["id"] for split in result.values() for row in split]
    assert len(ids) == len(set(ids)) == 100
    assert {name: len(values) for name, values in result.items()} == {
        "router_train": 50,
        "distill_train": 20,
        "validation": 10,
        "test": 20,
    }


def test_metrics_include_both_wrong_and_unsafe_rates():
    lower = np.array([1, 0, 0, 1], dtype=bool)
    upper = np.array([1, 1, 0, 0], dtype=bool)
    labels, eligible = routing_labels(lower, upper)
    assert labels.tolist() == [1, 0, 0, 1]
    assert eligible.tolist() == [1, 1, 0, 1]
    metrics = system_metrics(np.ones(4, dtype=bool), lower, upper, 1.0, 4.0)
    assert metrics["task_accuracy"] == 0.5
    assert metrics["both_wrong_rate"] == 0.25
    assert metrics["unsafe_given_lower"] == 0.5


def test_operating_point_respects_quality_floor():
    points = [
        {"task_accuracy": 0.94, "normalized_cost": 0.3},
        {"task_accuracy": 0.96, "normalized_cost": 0.5},
    ]
    assert select_operating_point(points, 1.0, 0.95)["normalized_cost"] == 0.5


def test_selective_consistency_counts_every_model_call():
    probability = np.array([0.9, 0.5, 0.5, 0.1])
    agreement = np.array([0, 1, 0, 1], dtype=bool)
    lower = np.array([1, 1, 0, 0], dtype=bool)
    upper = np.ones(4, dtype=bool)
    cfg = {"cost": {"lower": 1.0, "upper": 4.0}}
    metrics = cascade(probability, agreement, lower, upper, 0.2, 0.75, cfg)
    assert metrics["direct_accept_rate"] == 0.25
    assert metrics["second_pass_rate"] == 0.5
    assert metrics["upper_call_rate"] == 0.5
    # Initial Lower 4 + second Lower 2 + Upper 2*4 = 14; Always Upper = 16.
    assert metrics["normalized_cascade_cost"] == 14 / 16
    assert metrics["task_accuracy"] == 1.0


def test_binomial_upper_bound_is_conservative():
    assert 0.0 < binomial_upper(0, 200) < 0.02
    assert binomial_upper(7, 200) > 0.05
    assert binomial_upper(4, 200) < binomial_upper(5, 200)


def test_low_cost_verifier_uses_fractional_second_call_cost():
    cfg = {"cost": {"lower": 1.0, "verifier": 0.25, "upper": 4.0}}
    metrics = verifier_cascade(
        np.asarray([0.9, 0.5, 0.1]),
        np.asarray([False, True, False]),
        np.asarray([True, True, False]),
        np.asarray([True, True, True]),
        0.4,
        0.8,
        cfg,
    )
    # Costs are 1, 1.25, and 5: mean 2.4167, normalized by Upper cost 4.
    assert np.isclose(metrics["normalized_cascade_cost"], (1.0 + 1.25 + 5.0) / 3 / 4)
    assert metrics["task_accuracy"] == 1.0


def test_completion_lengths_stop_before_eos():
    rows = np.asarray([[10, 11, 2, 2], [20, 21, 22, 23]])
    assert completion_lengths(rows, eos_token_id=2) == [2, 4]


def test_performance_analysis_returns_item_level_metrics():
    inputs = (
        np.asarray([0.9, 0.5, 0.1]),
        np.asarray([False, True, False]),
        np.asarray([True, True, False]),
        np.asarray([True, True, True]),
    )
    items = route_items(inputs, 0.4, 0.8, 1.0, 0.25, 4.0)
    assert items["correct"].tolist() == [1.0, 1.0, 1.0]
    assert np.isclose(items["cost"].mean(), (1.0 + 1.25 + 5.0) / 3 / 4)


def test_model_registry_resolves_public_cross_family_models():
    assert get_model_spec("smollm2_360m").family == "smollm2"
    assert get_model_spec("qwen_7b").model_id == "Qwen/Qwen2.5-7B-Instruct"


def test_task_adapter_scores_numeric_and_multiple_choice():
    numeric = adapt_row({"id": "n1", "question": "1+1?", "answer": "#### 2"})
    assert score_prediction("Final answer: 2", numeric) == (2.0, True)
    choice = adapt_row({"id": "m1", "question": "Pick", "choices": ["x", "y"], "answer": 1})
    assert "A. x" in choice.prompt and "B. y" in choice.prompt
    assert extract_choice("Reasoning A. Final answer: B", 2) == "B"
    assert score_prediction("Final answer: B", choice) == ("B", True)
    assert "at most three short lines" in system_prompt("numeric")


def test_screening_validation_and_summary_are_deterministic():
    examples = validate_rows([{"id": "x", "prompt": "2+2", "reference": "4"}], "numeric")
    assert examples[0].prompt == "2+2"
    rows = [{"latency_ms": 10.0, "correct": True, "parsed_answer": 4, "generated_tokens": 3, "hit_token_limit": False}]
    report = summarize(rows, elapsed_seconds=0.02, peak_allocated=1024**3, peak_reserved=2 * 1024**3)
    assert report["accuracy"] == 1.0
    assert report["parse_success_rate"] == 1.0
    assert report["peak_vram_reserved_gb"] == 2.0
    assert report["token_limit_rate"] == 0.0


def test_pair_screening_rejects_expensive_lower_model():
    def report(key, accuracy, latency, tokens, vram):
        return {
            "model": {"key": key},
            "metrics": {
                "items": 200,
                "accuracy": accuracy,
                "latency_ms_p50": latency,
                "parse_success_rate": 1.0,
                "peak_vram_reserved_gb": vram,
                "generated_tokens_mean": tokens,
                "token_limit_rate": 0.0,
            },
        }

    gates = {
        "minimum_accuracy_gap": 0.05,
        "maximum_cost_ratio": 0.5,
        "minimum_parse_success": 0.98,
        "maximum_peak_vram_gb": 22.0,
        "maximum_token_limit_rate": 0.05,
    }
    result = evaluate_pair(report("small", 0.1, 350, 116, 1), report("large", 0.3, 270, 84, 4), gates)
    assert result["checks"]["accuracy_gap"]["pass"]
    assert not result["checks"]["measured_cost_ratio"]["pass"]
    assert not result["screening_pass"]


def test_power_summary_integrates_gross_energy():
    result = summarize_power([100.0, 120.0], 2.0)
    assert result["power_watts_mean"] == 110.0
    assert result["gross_energy_joules"] == 220.0


def test_output_ablation_matrix_is_complete():
    matrix = conditions("input.jsonl")
    assert len(matrix) == 20
    assert sum(row["mode"] == "task" for row in matrix) == 8
    assert sum(row["tokens"] == 64 for row in matrix) == 12


def test_output_ablation_oracle_feasibility_is_only_an_upper_bound():
    def run(model, mode, accuracy, latency, energy):
        return {
            "model": model, "mode": mode, "batch": 8, "limit": 200,
            "metrics": {
                "accuracy": accuracy, "generated_tokens_mean": 8,
                "latency_ms_p50": latency, "gross_energy_joules_per_item": energy,
                "token_limit_rate": 0.0,
            },
        }
    payload = {"completed": 2, "failures": [], "runs": [
        run("qwen_7b", "task", 0.8, 100, 100),
        run("qwen_1_5b", "answer_only", 0.2, 10, 10),
    ]}
    result = analyze(payload)
    assert result["latency_candidates"][0]["oracle_latency_saving_margin"] == 0.1
    assert "upper bound" in result["warning"]


def test_confidence_policy_counts_lower_overhead_and_upper_fallback():
    probability = np.asarray([0.9, 0.8, 0.1, 0.0])
    lower = np.asarray([True, False, False, False])
    upper = np.asarray([True, True, True, True])
    selected, _ = select_policy(probability, lower, upper, lower_cost_ratio=0.1, quality_floor=0.75)
    assert selected is not None
    assert selected["normalized_latency"] < 1.0
    assert selected["quality_retention"] >= 0.75


def test_token_budget_sweep_matrix_has_online_and_batched_conditions():
    matrix = token_budget_conditions("input.jsonl")
    assert len(matrix) == 8
    assert {row["budget"] for row in matrix} == {96, 128, 192, 256}
    assert {(row["batch"], row["limit"]) for row in matrix} == {(8, 200), (1, 50)}


def test_token_budget_analysis_separates_compression_from_cascade_feasibility():
    from src.analyze_token_budget_sweep import analyze as analyze_budgets

    def baseline(model, batch, limit, accuracy, latency, energy):
        return {"model": model, "mode": "task", "batch": batch, "limit": limit, "metrics": {
            "accuracy": accuracy, "latency_ms_p50": latency, "gross_energy_joules_per_item": energy,
        }}

    baselines = {"runs": [
        baseline("qwen_1_5b", 1, 50, 0.64, 100.0, 100.0),
        baseline("qwen_7b", 1, 50, 0.86, 50.0, 120.0),
    ]}
    sweep = {"completed": 1, "failures": [], "runs": [{
        "budget": 256, "batch": 1, "limit": 50,
        "metrics": {"accuracy": 0.62, "token_limit_rate": 0.04, "latency_ms_p50": 89.0,
                    "gross_energy_joules_per_item": 90.0},
    }]}
    result = analyze_budgets(sweep, baselines)
    assert len(result["compression_candidates"]) == 1
    assert result["oracle_latency_candidates"] == []


def test_mmlu_balanced_sampler_and_short_answer_matrix():
    from src.prepare_mmlu_screening import balanced_sample
    from src.run_mmlu_short_answer_screening import conditions as mmlu_conditions

    rows = [
        {"question": f"q{i}", "choices": ["a", "b"], "answer": i % 2, "subject": f"s{i % 2}", "_source_index": i}
        for i in range(8)
    ]
    selected = balanced_sample(rows, 6, 42)
    assert len(selected) == 6
    assert {row["task_metadata"]["subject"] for row in selected} == {"s0", "s1"}
    matrix = mmlu_conditions("mmlu.jsonl")
    assert len(matrix) == 8
    assert {row["model"] for row in matrix} == {"qwen_1_5b", "qwen_7b", "smollm2_360m", "smollm2_1_7b"}
    assert {(row["batch"], row["limit"]) for row in matrix} == {(8, 200), (1, 50)}


def test_multiple_choice_answer_only_prompt_is_supported():
    assert "only one line" in system_prompt("multiple_choice", "answer_only")


def test_mmlu_analysis_rejects_slow_lower_even_with_accuracy_gap():
    from src.analyze_mmlu_short_answer import analyze as analyze_mmlu

    def run(model, accuracy, latency, energy, batch=8, limit=200):
        return {"model": model, "batch": batch, "limit": limit, "metrics": {
            "accuracy": accuracy, "latency_ms_p50": latency, "gross_energy_joules_per_item": energy,
            "parse_success_rate": 1.0, "token_limit_rate": 0.0, "peak_vram_reserved_gb": 6.0,
        }}
    payload = {"completed": 4, "failures": [], "runs": [
        run("qwen_1_5b", 0.5, 90, 60), run("qwen_7b", 0.7, 100, 100),
        run("smollm2_360m", 0.2, 80, 50), run("smollm2_1_7b", 0.4, 70, 70),
        run("qwen_1_5b", 0.5, 90, 60, 1, 50), run("qwen_7b", 0.7, 100, 100, 1, 50),
        run("smollm2_360m", 0.2, 80, 50, 1, 50), run("smollm2_1_7b", 0.4, 70, 70, 1, 50),
    ]}
    result = analyze_mmlu(payload)
    assert result["screening_passes"] == []


def test_qwen_precision_ablation_covers_domains_and_serving_modes():
    from src.run_qwen_precision_ablation import conditions as precision_conditions

    matrix = precision_conditions()
    assert len(matrix) == 4
    assert {row["domain"] for row in matrix} == {"mmlu", "math"}
    assert {(row["batch"], row["limit"]) for row in matrix} == {(8, 200), (1, 50)}


def test_mmlu_logit_matrix_uses_best_known_precision():
    from src.run_mmlu_logit_screening import conditions as logit_conditions

    matrix = logit_conditions("mmlu.jsonl")
    assert len(matrix) == 8
    assert all(row["quantize_4bit"] == (row["model"] == "qwen_7b") for row in matrix)
