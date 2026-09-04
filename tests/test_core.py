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
from src.task_harness import adapt_row, extract_choice, score_prediction
from src.run_model_screening import summarize, validate_rows


def test_gsm8k_scoring_uses_final_answer():
    assert extract_final_number("work 3 then 7. Answer: 1,024") == 1024
    assert gsm8k_correct("The answer is \\boxed{42}", "reasoning\n#### 42")
    assert not gsm8k_correct("Answer: 41", "#### 42")


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


def test_screening_validation_and_summary_are_deterministic():
    examples = validate_rows([{"id": "x", "prompt": "2+2", "reference": "4"}], "numeric")
    assert examples[0].prompt == "2+2"
    rows = [{"latency_ms": 10.0, "correct": True, "parsed_answer": 4, "generated_tokens": 3}]
    report = summarize(rows, elapsed_seconds=0.02, peak_allocated=1024**3, peak_reserved=2 * 1024**3)
    assert report["accuracy"] == 1.0
    assert report["parse_success_rate"] == 1.0
    assert report["peak_vram_reserved_gb"] == 2.0
