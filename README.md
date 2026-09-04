# Adaptive SLM–LLM Routing: KSC Pilot

This repository implements a performance-focused SLM–LLM routing study on GSM8K and SVAMP.
The paper method uses output-aware confidence and a short independent answer-only verifier.
The post-paper study additionally evaluates a query pre-triage stage that skips the Lower
model for clearly difficult requests before applying output-aware post-routing.

## Guardrails

- Router training, distillation, validation, and test examples are disjoint by stable ID.
- Hyperparameters and routing thresholds are selected on validation only.
- Both-wrong examples are excluded from binary router fitting but retained in system metrics.
- Results include task accuracy, lower coverage, upper-call rate, unsafe routing, both-wrong
  rate, and cost normalized to always-upper.
- Random and cluster-balanced distillation use the same budget and training schedule.
- Final comparisons are repeated over three router/distillation seeds.
- Policy selection and risk certification use disjoint SVAMP splits.
- The answer-only policy is frozen before certification and official-test diagnostics.
- Safety remains a reported secondary metric; the 95% unsafe-risk upper bound of 5.195%
  narrowly misses the prespecified 5% target and is not presented as a safety guarantee.

## First commands

```powershell
conda run -n llm-practice python -m pytest -q
conda run -n llm-practice python -m src.prepare_data --config configs/pilot_gsm8k.yaml
```

## Performance-focused reproduction

After the cached GSM8K/SVAMP model outputs and confidence features are available:

```powershell
conda run -n llm-practice python -m src.run_low_cost_verifier
conda run -n llm-practice python -m src.benchmark_verifier_latency --limit 128 --batch-size 16 --repeats 2
conda run -n llm-practice python -m src.analyze_verifier_performance --draws 10000
```

The RTX 3090 benchmark compares full, answer-only, and micro-reasoning second passes on the
same 128 questions in forward/reverse order after warmup. The paired analysis keeps the selected thresholds fixed
and reports bootstrap confidence intervals and an exact McNemar test.

## Post-paper pre-triage diagnostics

The following commands reproduce the verifier frontier, distribution-adapted nested CV, and
the frozen official GSM8K test evaluation after their cached inference/features are available:

```powershell
conda run -n llm-practice python -m src.analyze_micro_verifier_frontier
conda run -n llm-practice python -m src.select_pretriage_cascade
conda run -n llm-practice python -m src.cross_validate_adaptive_pretriage --unsafe-cap 0.03
conda run -n llm-practice python -m src.run_official_adaptive_pretriage
conda run -n llm-practice python -m src.analyze_official_pretriage_ladder
```

On the untouched official GSM8K test (1,319 rows), the primary CV-derived policy retains
97.46% of Always-Upper accuracy and has a 3.50% one-sided 95% unsafe-risk upper bound, while
reducing normalized cost by 7.19%. It therefore passes quality and risk but misses the
prespecified 10% cost-reduction target.

## Capability-aware external evaluation

The next-stage experiment fine-tunes the last two BGE encoder layers to predict whether the
Lower model can answer a request, using 1,688 GSM8K routing labels only. The two-stage policy
is selected and frozen on a stratified GSM8K validation split before ASDiv is evaluated.

```powershell
conda run -n llm-practice python -m src.prepare_asdiv_external_test
conda run -n llm-practice python -m src.train_adaptive_capability_pretriage
conda run -n llm-practice python -m src.run_asdiv_capability_pretriage
```

Across training seeds 42/43/44, the frozen policies average 88.91% accuracy on 2,249
numeric-answer ASDiv examples and retain 97.97% of Always-Upper accuracy. Mean normalized cost
reduction is 13.47% (range 11.68--14.68%), and the one-sided exact 95% unsafe-risk upper bounds
range from 3.78% to 4.32%. All three seeds pass all three external gates. On the fixed GSM8K
validation split, two of three seeds pass the 10% cost rule; seed 42 is a 0.089 percentage-point
near-miss. The result is therefore treated as promising external replication rather than a
universal guarantee.

The second external benchmark uses the 355-example MAWPS held-out split without threshold
retuning. Across the same three seeds, accuracy averages 91.92%, quality retention averages
98.59%, and normalized cost reduction averages 18.22%. The strict risk certificate is less
stable: only seed 42 stays below the 5% one-sided upper bound (4.73%); seeds 43 and 44 reach
6.77% and 5.08%. Thus performance and cost generalize to a second dataset, while universal
safety certification does not. The next experiment targets seed variance with a prespecified
ensemble or validation risk-upper constraint, not MAWPS-specific threshold tuning.

The three-seed mean-probability ensemble is selected on the fixed GSM8K validation split by
directly requiring a one-sided exact 95% unsafe-risk upper bound at or below 5%. The selected
0.59/0.60 policy improves capability AUC to 0.607 and passes the internal quality, cost, and
risk gates. It also passes all gates on ASDiv (97.80% quality retention, 14.57% cost reduction,
4.36% risk upper bound). On MAWPS it retains 98.19% quality and reduces cost by 19.56%, but its
6.43% risk upper bound still fails. No threshold is retuned using MAWPS.

A post-hoc sensitivity ladder compares validation risk-upper limits of 3%, 4%, and 5%. The
4% policy (pre/post thresholds 0.60/0.60) passes all external gates on both ASDiv (12.21% cost
reduction, 3.59% risk upper) and MAWPS (16.38%, 4.73%), while narrowly missing the internal 10%
cost target at 9.47%. Because this ladder was run after observing the MAWPS failure, 0.60/0.60
is only a diagnostic candidate. Confirmatory use requires freezing it before a third untouched
dataset is evaluated.

For confirmatory evaluation, the diagnostic 0.60/0.60 policy is frozen before downloading
MATH-500. A strict regex retains 317/500 examples whose reference answer is a simple integer
or decimal, avoiding incorrect conversion of fractions and symbolic expressions. On this
harder domain, the cascade reaches 58.99% accuracy versus 58.36% for Always Upper and makes no
unsafe Lower acceptance (one-sided 95% upper bound 0.94%). However, only 2.21% of requests are
accepted by Lower and normalized cost rises to 1.029. This confirms that the policy is not
universally cost-effective and motivates a domain-level cascade feasibility guard.

The diagnostic feasibility guard uses only observable route decisions and model costs:
`saving = lower_acceptance_rate - lower_call_rate * (C_L / C_U)`. Cascade execution is enabled
only when a Hoeffding 95% lower bound on this saving is positive. It keeps cascade routing for
ASDiv and MAWPS, whose conservative saving bounds are 9.63% and 9.88%, and switches the
MATH-500 numeric subset to Always Upper, removing the observed 2.93% overhead. This guard is a
post-hoc architectural diagnostic and still requires prospective validation on new traffic.

## Model-family and domain expansion roadmap

The next study phase tests whether the method is specific to Qwen and arithmetic. It first
adds a common model/task/scorer/latency harness, then screens each model pair on only 200
examples before authorizing expensive full evaluation. The planned model conditions are the
Qwen2.5 1.5B/7B control, public SmolLM2 360M/1.7B, SmolLM2 1.7B/Qwen2.5 7B cross-family
routing, Gemma 3 1B/4B after accepting its terms, and Llama 3.2 1B/Llama 3.1 8B after model
access approval.

Domain expansion proceeds from executable Python (MBPP, then HumanEval) to English knowledge
(MMLU), Korean knowledge (KMMLU), and deterministic instruction following (IFEval). Each task
uses a reproducible task-specific scorer rather than a free-form LLM judge. Full test and
three-seed runs are permitted only when a 200-example pilot has an Upper--Lower accuracy gap
of at least 5 percentage points, measured `C_L/C_U < 0.50`, at least 98% output parsing, at
least 95% quality retention, and capability AUC of at least 0.60. See
`MODEL_DOMAIN_EXPANSION_PLAN.md` for the fixed order, safety rules, stop gates, and estimated
4--7 workdays / 32--65 GPU hours.

The common cross-family harness is now available. Validate an input without loading a model,
then run a measured screening condition as follows:

```powershell
python -m src.run_model_screening --model-key qwen_1_5b --input artifacts/data/test.jsonl --limit 10 --validate-only
python -m src.run_model_screening --model-key smollm2_360m --input artifacts/data/test.jsonl --limit 200 --batch-size 16
```

Outputs are isolated by input/task/model below `artifacts/model_screening/`. Each report records
accuracy, parse success, generated tokens, p50/p95 latency, throughput, and peak GPU memory.
Code and instruction-following rows deliberately remain unjudged until their official isolated
scorers are connected.

The first 200-example GSM8K validation screen is complete for M0/M1/M2. Accuracy gaps are
large enough, but measured batch-8 Lower/Upper latency ratios are 1.632, 1.538, and 0.673;
all fail the fixed `<0.50` latency gate. A batch-1 Qwen check also fails at 1.683. Smaller
models produced longer answers and were not faster on this GPU. Compute-proxy ratios based on
model size times generated tokens remain below 0.40, so compute cost and wall-clock latency
must now be reported separately. The tracked audit summary is
`paper/data/model_screening_math_200.json`. Runs capped at 128/256 tokens were invalidated by
truncation; only concise-prompt 512-token results are retained.

The restartable long-run ablation compares task/512, micro-reasoning/64, and answer-only/64
under batch 8 (200 items) and batch 1 (50 items). It samples gross GPU power every 0.5 seconds,
writes progress after every condition, skips completed reports on restart, and continues after
individual failures:

```powershell
conda run --no-capture-output -n llm-practice python -m src.run_output_length_ablation
```

Live logs are written to `artifacts/logs/output_length_ablation.log` and the restartable
aggregate is `artifacts/results/output_length_ablation.json`.

All 20 ablation conditions completed without a failed run. Short outputs sharply reduce
latency and gross GPU energy but also collapse standalone Lower accuracy. Under an oracle that
accepts exactly correct Lower outputs, only three conditions retain positive latency margin:
Qwen answer-only batch 8 (+3.5 points), SmolLM2-1.7B answer-only batch 8 (+2.3), and the same
SmolLM2 condition at batch 1 (+0.6). These are upper bounds, not achieved routing results. The
tracked analysis is `paper/data/output_length_ablation_analysis.json`; the next gate tests
whether confidence features can identify the rare correct answer-only outputs precisely enough.

That confidence gate also completed. Qwen1.5B reaches OOF AUC 0.722 and AP 0.380, but its
95%-quality operating point saves only 0.018% latency and accepts eight unsafe outputs among
16 Lower acceptances (50%; exact 95% upper bound 72.14%). SmolLM2-1.7B reaches AUC 0.618,
AP 0.142, and does not break even. Neither condition achieves the 10% practical latency target
or a 5% risk certificate. These are exploratory OOF results with threshold selection on the
same OOF predictions. The 64-token Answer-only Lower path is stopped; the next bounded test is
a concise 96/128/192/256-token Qwen budget sweep.

Generated data, model outputs, embeddings, and adapters are stored below `artifacts/` and
excluded from Git. Small manifests and final JSON result summaries are force-tracked when
needed for auditability.

The active configs use the 1.5B/7B pair and a normalized Upper cost of 4.667. See
`adaptive_slm_llm_routing_architecture_v1.html` for the unified architecture, experiment
timeline, negative results, and current performance tables.

For paper writing, use `OVERLEAF_KSC_PAPER_GUIDE.md` as the single source of instructions.
Publication figures are tracked in `paper/figures/` as PDF, SVG, and PNG, and can be regenerated
from `paper/data/paper_results.json` with `python -m src.render_paper_figures`.
For one-shot Overleaf entry, paste `paper/overleaf/main_single.tex` into the template's
`main.tex`, then upload the four `paper/figures/*.pdf` files to `figures/`.
