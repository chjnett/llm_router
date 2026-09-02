# Adaptive SLM–LLM Routing: KSC Pilot

This repository turns the original HTML research plan into a reproducible GSM8K pilot.
The primary claim under test is whether retraining a logistic-regression router or rebuilding
a kNN expert pool recovers the quality–cost frontier after targeted LoRA changes the lower
model's capability.

## Guardrails

- Router training, distillation, validation, and test examples are disjoint by stable ID.
- Hyperparameters and routing thresholds are selected on validation only.
- Both-wrong examples are excluded from binary router fitting but retained in system metrics.
- Results include task accuracy, lower coverage, upper-call rate, unsafe routing, both-wrong
  rate, and cost normalized to always-upper.
- Random and cluster-balanced distillation use the same budget and training schedule.
- Final comparisons are repeated over three router/distillation seeds.

## First commands

```powershell
conda run -n llm-practice python -m pytest -q
conda run -n llm-practice python -m src.prepare_data --config configs/pilot_gsm8k.yaml
```

Generated data, model outputs, embeddings, adapters, and results are stored below `artifacts/`
and intentionally excluded from Git.

The checked-in pilot config uses a 1.5B/3B model pair because this workstation currently has
insufficient free disk for the 7B source checkpoint. The final KSC confirmation run should
switch `upper` to Qwen2.5-7B-Instruct and restore the normalized parameter-cost ratio to 4.667.
