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
